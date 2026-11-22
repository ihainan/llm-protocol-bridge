package gemini

import (
	"encoding/json"
	"sort"
	"strconv"
	"strings"

	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

type pendingGeminiCall struct {
	id        string
	name      string
	args      strings.Builder
	signature string
	order     int
}

type geminiCand struct {
	calls       map[string]*pendingGeminiCall
	lastCallKey string
	callSeq     int
}

// StreamEncoder converts ir.Events into Gemini streamGenerateContent chunks.
type StreamEncoder struct {
	opts        EncodeOptions
	model       string
	started     bool
	errored     bool
	arrayOpened bool
	arrayClosed bool
	chunkCount  int
	cands       map[int]*geminiCand
	usage       *ir.Usage
}

// NewStreamEncoder returns a Gemini stream encoder.
func NewStreamEncoder(opts EncodeOptions) *StreamEncoder {
	return &StreamEncoder{
		opts:  opts,
		cands: map[int]*geminiCand{},
	}
}

func (e *StreamEncoder) cand(idx int) *geminiCand {
	c := e.cands[idx]
	if c == nil {
		c = &geminiCand{calls: map[string]*pendingGeminiCall{}}
		e.cands[idx] = c
	}
	return c
}

// Encode serialises one ir.Event to Gemini SSE / JSON-stream bytes appended to dst.
func (e *StreamEncoder) Encode(dst []byte, ev ir.Event) ([]byte, error) {
	if e.errored {
		return dst, nil
	}
	switch ev.Type {
	case ir.EvStart:
		e.model = ev.Model
		if e.model == "" {
			e.model = e.opts.Model
		}
		e.started = true
		if ev.Usage != nil {
			if e.usage == nil {
				u := *ev.Usage
				e.usage = &u
			} else {
				e.usage.Merge(*ev.Usage)
			}
		}

	case ir.EvText, ir.EvRefusal:
		if ev.Text == "" {
			break
		}
		dst = e.emitCandidateParts(dst, ev.Index, []wirePart{{Text: ev.Text}}, "")

	case ir.EvImage:
		if ev.Media != nil && ev.Media.Data != "" {
			mt := ev.Media.MediaType
			if mt == "" {
				mt = "image/png"
			}
			dst = e.emitCandidateParts(dst, ev.Index, []wirePart{{InlineData: &wireBlob{
				MimeType: mt,
				Data:     ev.Media.Data,
			}}}, "")
		} else if ev.Media != nil && ev.Media.URL != "" {
			dst = e.emitCandidateParts(dst, ev.Index, []wirePart{{FileData: &wireFileData{
				MimeType: ev.Media.MediaType,
				FileURI:  ev.Media.URL,
			}}}, "")
		}

	case ir.EvCitation:
		if ev.Citation != nil {
			if err := e.opts.Env.Lose(ir.LossCitations, ir.Root.Key("candidates"), "citations not supported in Gemini stream output"); err != nil {
				return dst, err
			}
		}

	case ir.EvReasoningStart:
		// No wire chunk needed until thought text arrives.

	case ir.EvReasoning:
		if ev.Text == "" {
			break
		}
		dst = e.emitCandidateParts(dst, ev.Index, []wirePart{{Text: ev.Text, Thought: true}}, "")

	case ir.EvSignature:
		if ev.Signature == nil || ev.Signature.Value == "" {
			break
		}
		sigVal := encodeGeminiSignature(e.opts.Env, ev.Signature)
		c := e.cand(ev.Index)
		if c.lastCallKey != "" {
			if pc := c.calls[c.lastCallKey]; pc != nil {
				pc.signature = sigVal
				break
			}
		}
		dst = e.emitCandidateParts(dst, ev.Index, []wirePart{{ThoughtSignature: sigVal}}, "")

	case ir.EvToolStart:
		c := e.cand(ev.Index)
		c.callSeq++
		key := ev.ID
		if key == "" {
			key = "__call_" + strconv.Itoa(c.callSeq)
		}
		pc := &pendingGeminiCall{
			id:    ev.ID,
			name:  ev.Name,
			order: c.callSeq,
		}
		if ev.Signature != nil && ev.Signature.Value != "" {
			pc.signature = encodeGeminiSignature(e.opts.Env, ev.Signature)
		}
		c.calls[key] = pc
		c.lastCallKey = key

	case ir.EvToolArgs:
		c := e.cand(ev.Index)
		pc := c.lookup(ev.ID)
		if pc != nil && ev.Text != "" {
			pc.args.WriteString(ev.Text)
		}

	case ir.EvToolEnd:
		c := e.cand(ev.Index)
		key, pc := c.lookupWithKey(ev.ID)
		if pc != nil {
			if ev.Err == nil {
				dst = e.emitCall(dst, ev.Index, pc, ev.Arguments)
			}
			delete(c.calls, key)
			if c.lastCallKey == key {
				c.lastCallKey = ""
			}
		}

	case ir.EvBuiltin:
		if ev.Builtin == nil {
			break
		}
		if ev.Builtin.Kind == ir.BuiltinCodeExecution {
			if ev.Builtin.IsResult {
				outcome := "OUTCOME_OK"
				if ev.Builtin.Error != "" {
					outcome = ev.Builtin.Error
				}
				dst = e.emitCandidateParts(dst, ev.Index, []wirePart{{
					CodeExecutionResult: &wireCodeExecutionResult{
						Outcome: outcome,
						Output:  ev.Builtin.Content,
					},
				}}, "")
				break
			}
			var payload struct {
				Language string `json:"language"`
				Code     string `json:"code"`
			}
			_ = json.Unmarshal(ev.Builtin.Input, &payload)
			dst = e.emitCandidateParts(dst, ev.Index, []wirePart{{
				ExecutableCode: &wireExecutableCode{
					Language: payload.Language,
					Code:     payload.Code,
				},
			}}, "")
			break
		}
		if err := e.opts.Env.Lose(ir.LossBuiltinActivity, ir.Root.Key("candidates"), "builtin activity not supported by Gemini"); err != nil {
			return dst, err
		}

	case ir.EvStop:
		c := e.cand(ev.Index)
		dst = e.flushPendingCalls(dst, ev.Index, c)
		reason := "STOP"
		if ev.Stop != nil {
			reason = irStopToGemini(ev.Stop)
		}
		dst = e.emitCandidateParts(dst, ev.Index, nil, reason)

	case ir.EvUsage:
		if ev.Usage != nil {
			if e.usage == nil {
				u := *ev.Usage
				e.usage = &u
			} else {
				e.usage.Merge(*ev.Usage)
			}
		}
		if e.usage != nil {
			chunk := wireResponse{
				UsageMetadata: irUsageToWire(e.usage),
				ModelVersion:  e.model,
			}
			dst = e.emit(dst, chunk)
		}

	case ir.EvError:
		if ev.Err != nil {
			e.errored = true
			e.cands = map[int]*geminiCand{}
			if e.opts.AltSSE {
				dst = BuildStreamErrorSSE(dst, ev.Err)
			} else if e.arrayOpened && !e.arrayClosed {
				dst = append(dst, ',')
				dst = BuildStreamError(dst, ev.Err)
				dst = append(dst, ']')
				e.arrayClosed = true
			} else {
				dst = BuildStreamError(dst, ev.Err)
			}
		}
	}
	return dst, nil
}

// Close flushes any remaining pending tool calls and closes the JSON array when AltSSE is false.
func (e *StreamEncoder) Close(dst []byte) ([]byte, error) {
	if e.errored {
		return dst, nil
	}
	if len(e.cands) > 0 {
		idxs := make([]int, 0, len(e.cands))
		for idx := range e.cands {
			idxs = append(idxs, idx)
		}
		sort.Ints(idxs)
		for _, idx := range idxs {
			dst = e.flushPendingCalls(dst, idx, e.cands[idx])
		}
	}
	if !e.opts.AltSSE && e.arrayOpened && !e.arrayClosed {
		dst = append(dst, ']')
		e.arrayClosed = true
	}
	return dst, nil
}

// Keepalive sends an empty SSE comment (when AltSSE) or whitespace.
func (e *StreamEncoder) Keepalive(dst []byte) []byte {
	if e.opts.AltSSE {
		return sse.AppendComment(dst, "", false)
	}
	return append(dst, ' ')
}

func (c *geminiCand) lookup(id string) *pendingGeminiCall {
	_, pc := c.lookupWithKey(id)
	return pc
}

func (c *geminiCand) lookupWithKey(id string) (string, *pendingGeminiCall) {
	if id != "" {
		if pc := c.calls[id]; pc != nil {
			return id, pc
		}
	}
	if c.lastCallKey != "" {
		if pc := c.calls[c.lastCallKey]; pc != nil {
			return c.lastCallKey, pc
		}
	}
	return "", nil
}

func (e *StreamEncoder) flushPendingCalls(dst []byte, idx int, c *geminiCand) []byte {
	if len(c.calls) == 0 {
		return dst
	}
	keys := make([]string, 0, len(c.calls))
	for k := range c.calls {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		return c.calls[keys[i]].order < c.calls[keys[j]].order
	})
	for _, k := range keys {
		dst = e.emitCall(dst, idx, c.calls[k], nil)
		delete(c.calls, k)
	}
	c.lastCallKey = ""
	return dst
}

func (e *StreamEncoder) emitCall(dst []byte, idx int, pc *pendingGeminiCall, finalArgs json.RawMessage) []byte {
	argsStr := strings.TrimSpace(string(finalArgs))
	if argsStr == "" {
		argsStr = strings.TrimSpace(pc.args.String())
	}
	if argsStr == "" || !json.Valid([]byte(argsStr)) {
		argsStr = "{}"
	}
	wp := wirePart{
		FunctionCall: &wireFunctionCall{
			ID:   pc.id,
			Name: pc.name,
			Args: json.RawMessage(argsStr),
		},
		ThoughtSignature: pc.signature,
	}
	return e.emitCandidateParts(dst, idx, []wirePart{wp}, "")
}

func (e *StreamEncoder) emitCandidateParts(dst []byte, idx int, parts []wirePart, finishReason string) []byte {
	cand := wireCandidate{
		Index:        idx,
		FinishReason: finishReason,
	}
	if len(parts) > 0 {
		cand.Content = wireContent{Role: "model", Parts: parts}
	}
	chunk := wireResponse{
		Candidates:   []wireCandidate{cand},
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
	if !e.arrayOpened {
		dst = append(dst, '[')
		e.arrayOpened = true
	} else {
		dst = append(dst, ',')
	}
	dst = append(dst, b...)
	e.chunkCount++
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
			if len(p.Citations) > 0 {
				if err := env.Lose(ir.LossCitations, fp, "citations not supported in Gemini response output"); err != nil {
					return nil, err
				}
			}
			wps, err := irPartToWirePart(p, fp, env, Dialect{}, nil, nil, true)
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
		return "STOP"
	case ir.StopContentFilter, ir.StopRefusal:
		return "SAFETY"
	case ir.StopOther:
		return "OTHER"
	}
	return "STOP"
}

func encodeGeminiSignature(env *ir.Env, sig *ir.Signature) string {
	if sig == nil || sig.Value == "" {
		return ""
	}
	prov := ""
	if env != nil {
		prov = env.ProviderID
	}
	if sig.Origin.Format == ir.FormatGemini && sig.Origin.Provider == "" && prov == "" {
		return sig.Value
	}
	if sig.Origin.Format == ir.FormatGemini && !needsGeminiSeal(sig.Origin, env) {
		return sig.Value
	}
	return env.Sealer().Seal(ir.SealRecord{
		Origin: sig.Origin,
		Kind:   ir.SealSignature,
		Value:  sig.Value,
	})
}

func needsGeminiSeal(origin ir.Origin, env *ir.Env) bool {
	if origin.Format != ir.FormatGemini {
		return true
	}
	if origin.Provider != "" {
		return true
	}
	return env != nil && env.ProviderID != ""
}

func irUsageToWire(u *ir.Usage) *wireUsage {
	if u == nil || !u.Any() {
		return nil
	}
	prompt := u.PromptTokens() - u.ToolUsePromptTokens
	if prompt < 0 {
		prompt = 0
	}
	cands := u.OutputTokens - u.ReasoningTokens
	if cands < 0 {
		cands = 0
	}
	total := u.TotalTokens()
	w := &wireUsage{
		PromptTokenCount:     &prompt,
		CandidatesTokenCount: &cands,
		TotalTokenCount:      &total,
	}
	if u.Has(ir.UCacheRead) {
		v := u.CacheReadTokens
		w.CachedContentTokenCount = &v
	}
	if u.Has(ir.UReasoning) {
		v := u.ReasoningTokens
		w.ThoughtsTokenCount = &v
	}
	if u.Has(ir.UToolUsePrompt) {
		v := u.ToolUsePromptTokens
		w.ToolUsePromptTokenCount = &v
	}
	return w
}
