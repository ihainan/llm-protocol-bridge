package gemini

import (
	"encoding/json"

	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// StreamEncoder converts ir.Events into Gemini streamGenerateContent chunks.
type StreamEncoder struct {
	opts    EncodeOptions
	model   string
	started bool
	// pending accumulates parts for the current candidate before a stop event
	// flushes them as a chunk.
	pending []wirePart
	// pendingIdx is the candidate index of pending parts.
	pendingIdx int
	// usage holds the last usage snapshot seen.
	usage *wireUsage
	// stop holds the finish reason for the current candidate.
	stop string
}

// NewStreamEncoder returns a Gemini stream encoder.
func NewStreamEncoder(opts EncodeOptions) *StreamEncoder {
	return &StreamEncoder{opts: opts}
}

// Encode serialises one ir.Event to Gemini SSE / JSON-stream bytes appended to dst.
func (e *StreamEncoder) Encode(dst []byte, ev ir.Event) ([]byte, error) {
	switch ev.Type {
	case ir.EvStart:
		e.model = ev.Model
		if e.model == "" {
			e.model = e.opts.Model
		}
		e.started = true
		if ev.Usage != nil {
			e.usage = irUsageToWire(ev.Usage)
		}

	case ir.EvText:
		e.pending = append(e.pending, wirePart{Text: ev.Text})

	case ir.EvReasoningStart:
		// no-op; first EvReasoning event will add a thought part

	case ir.EvReasoning:
		e.pending = append(e.pending, wirePart{Text: ev.Text, Thought: true})

	case ir.EvSignature:
		if ev.Signature != nil && len(e.pending) > 0 {
			last := &e.pending[len(e.pending)-1]
			last.ThoughtSignature = ev.Signature.Value
		}

	case ir.EvToolStart:
		// Accumulate; EvToolEnd will add the functionCall part with final args.
		e.pending = append(e.pending, wirePart{
			FunctionCall: &wireFunctionCall{
				ID:   ev.ID,
				Name: ev.Name,
			},
		})

	case ir.EvToolArgs:
		// Update the last pending function call's args incrementally; we overwrite
		// since Gemini sends complete args in one shot.
		for i := len(e.pending) - 1; i >= 0; i-- {
			if e.pending[i].FunctionCall != nil {
				e.pending[i].FunctionCall.Args = json.RawMessage(ev.Text)
				break
			}
		}

	case ir.EvToolEnd:
		if ev.Arguments != nil {
			for i := len(e.pending) - 1; i >= 0; i-- {
				if e.pending[i].FunctionCall != nil {
					e.pending[i].FunctionCall.Args = ev.Arguments
					break
				}
			}
		}

	case ir.EvStop:
		e.pendingIdx = ev.Index
		if ev.Stop != nil {
			e.stop = irStopToGemini(ev.Stop)
		} else {
			e.stop = "STOP"
		}
		dst = e.flush(dst)

	case ir.EvUsage:
		if ev.Usage != nil {
			e.usage = irUsageToWire(ev.Usage)
		}
		// Emit a usage-only chunk.
		chunk := wireResponse{
			UsageMetadata: e.usage,
			ModelVersion:  e.model,
		}
		dst = e.emit(dst, chunk)

	case ir.EvError:
		if ev.Err != nil {
			dst = BuildStreamError(dst, ev.Err)
		}
	}
	return dst, nil
}

// Close flushes any remaining pending parts.
func (e *StreamEncoder) Close(dst []byte) ([]byte, error) {
	if len(e.pending) > 0 {
		dst = e.flush(dst)
	}
	return dst, nil
}

// Keepalive sends an empty SSE comment.
func (e *StreamEncoder) Keepalive(dst []byte) []byte {
	return sse.AppendComment(dst, "", false)
}

func (e *StreamEncoder) flush(dst []byte) []byte {
	if len(e.pending) == 0 && e.stop == "" {
		return dst
	}
	parts := e.pending
	e.pending = nil

	var cands []wireCandidate
	cand := wireCandidate{
		Index:        e.pendingIdx,
		Content:      wireContent{Role: "model", Parts: parts},
		FinishReason: e.stop,
	}
	e.stop = ""
	cands = append(cands, cand)

	chunk := wireResponse{
		Candidates:   cands,
		ModelVersion: e.model,
	}
	return e.emit(dst, chunk)
}

func (e *StreamEncoder) emit(dst []byte, chunk wireResponse) []byte {
	b, err := json.Marshal(chunk)
	if err != nil {
		return dst
	}
	if e.opts.AltSSE {
		return sse.AppendData(dst, b)
	}
	dst = append(dst, b...)
	dst = append(dst, '\n')
	return dst
}

// BuildResponse serialises a whole ir.Response to a non-streaming Gemini body.
func BuildResponse(resp *ir.Response, opts EncodeOptions) ([]byte, error) {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	model := opts.Model
	if model == "" {
		model = resp.Model
	}

	var cands []wireCandidate
	for _, out := range resp.Outputs {
		var parts []wirePart
		for i, p := range out.Parts {
			fp := ir.Root.Key("candidates").At(out.Index).Key("content").Key("parts").At(i)
			wps, err := irPartToWirePart(p, fp, env, Dialect{}, nil, true)
			if err != nil {
				return nil, err
			}
			parts = append(parts, wps...)
		}
		cand := wireCandidate{
			Index:        out.Index,
			Content:      wireContent{Role: "model", Parts: parts},
			FinishReason: irStopToGemini(&out.Stop),
		}
		cands = append(cands, cand)
	}

	wr := wireResponse{
		Candidates:    cands,
		UsageMetadata: irUsageToWire(&resp.Usage),
		ModelVersion:  model,
	}
	return json.Marshal(wr)
}

func irStopToGemini(s *ir.Stop) string {
	if s == nil {
		return "STOP"
	}
	switch s.Reason {
	case ir.StopEndTurn:
		return "STOP"
	case ir.StopMaxTokens, ir.StopContextWindow:
		return "MAX_TOKENS"
	case ir.StopSequence:
		return "STOP"
	case ir.StopToolUse:
		return "TOOL_CALLS"
	case ir.StopContentFilter, ir.StopRefusal:
		return "SAFETY"
	case ir.StopOther:
		return "OTHER"
	}
	return "STOP"
}

func irUsageToWire(u *ir.Usage) *wireUsage {
	if u == nil || !u.Any() {
		return nil
	}
	w := &wireUsage{
		PromptTokenCount:     u.PromptTokens(),
		CandidatesTokenCount: u.OutputTokens,
		TotalTokenCount:      u.TotalTokens(),
	}
	if u.Has(ir.UCacheRead) {
		w.CachedContentTokenCount = u.CacheReadTokens
	}
	if u.Has(ir.UReasoning) {
		w.ThoughtsTokenCount = u.ReasoningTokens
	}
	if u.Has(ir.UToolUsePrompt) {
		w.ToolUsePromptTokenCount = u.ToolUsePromptTokens
	}
	return w
}
