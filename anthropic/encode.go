package anthropic

import (
	"encoding/json"
	"strings"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

type pendingTool struct {
	id        string
	name      string
	args      strings.Builder
	finalArgs json.RawMessage
	done      bool
}

// StreamEncoder converts ir.Events into Anthropic Messages SSE chunks.
type StreamEncoder struct {
	opts    EncodeOptions
	id      string
	model   string
	started bool
	errored bool
	// blockIdx is the next content block index to assign.
	blockIdx int
	// openBlock is the index of the currently open content block (-1 = none).
	openBlock int
	// openBlockType is the type string of the open block.
	openBlockType     string
	thinkingSigVal    strings.Builder
	thinkingSigOrigin ir.Origin
	activeCallID      string
	pendingTools      []*pendingTool
	pendingByID       map[string]*pendingTool
	stop              *ir.Stop
	usage             *ir.Usage
}

// NewStreamEncoder returns an Anthropic stream encoder.
func NewStreamEncoder(opts EncodeOptions) *StreamEncoder {
	return &StreamEncoder{
		opts:        opts,
		openBlock:   -1,
		pendingByID: map[string]*pendingTool{},
	}
}

func (e *StreamEncoder) ensureStarted(dst []byte) []byte {
	if e.started {
		return dst
	}
	e.started = true
	if e.id == "" {
		e.id = "msg_" + e.opts.Env.NewID("", 24)
	}
	if e.model == "" {
		e.model = e.opts.Model
	}
	dst = e.emitMessageStart(dst, ir.Event{Usage: e.usage})
	dst = e.emitPing(dst)
	return dst
}

// Encode serialises one ir.Event to Anthropic SSE bytes appended to dst.
func (e *StreamEncoder) Encode(dst []byte, ev ir.Event) ([]byte, error) {
	if e.errored {
		return dst, nil
	}
	switch ev.Type {
	case ir.EvStart:
		e.id = ev.ID
		if e.id == "" {
			e.id = "msg_" + e.opts.Env.NewID("", 24)
		}
		e.model = e.opts.Model
		if e.model == "" && ev.Model != "" {
			e.model = ev.Model
		}
		if ev.Usage != nil {
			if e.usage == nil {
				u := *ev.Usage
				e.usage = &u
			} else {
				e.usage.Merge(*ev.Usage)
			}
		}
		if !e.started {
			e.started = true
			dst = e.emitMessageStart(dst, ev)
			dst = e.emitPing(dst)
		}

	case ir.EvText, ir.EvRefusal:
		dst = e.ensureStarted(dst)
		dst = e.ensureTextBlock(dst)
		dst = e.appendSSE(dst, "content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": e.openBlock,
			"delta": map[string]any{"type": "text_delta", "text": ev.Text},
		})

	case ir.EvCitation:
		if ev.Citation != nil {
			if e.openBlock >= 0 && e.openBlockType == "text" {
				var citObj any
				if len(ev.Citation.Raw) > 0 {
					citObj = json.RawMessage(ev.Citation.Raw)
				} else {
					citObj = map[string]any{
						"type":       "web_search_result_location",
						"url":        ev.Citation.URL,
						"title":      ev.Citation.Title,
						"cited_text": ev.Citation.CitedText,
					}
				}
				dst = e.appendSSE(dst, "content_block_delta", map[string]any{
					"type":  "content_block_delta",
					"index": e.openBlock,
					"delta": map[string]any{"type": "citations_delta", "citation": citObj},
				})
			} else if err := e.opts.Env.Lose(ir.LossCitations, ir.Root.Key("content"), "citation without open text block in Anthropic stream"); err != nil {
				return dst, err
			}
		}

	case ir.EvImage:
		if ev.Media != nil {
			if err := e.opts.Env.Lose(ir.LossOutputMedia, ir.Root.Key("content"), "output media not supported in Anthropic assistant stream"); err != nil {
				return dst, err
			}
		}

	case ir.EvReasoningStart:
		dst = e.ensureStarted(dst)
		dst = e.closeBlock(dst)
		idx := e.blockIdx
		e.blockIdx++
		e.openBlock = idx
		e.openBlockType = "thinking"
		dst = e.appendSSE(dst, "content_block_start", map[string]any{
			"type":          "content_block_start",
			"index":         idx,
			"content_block": map[string]any{"type": "thinking", "thinking": ""},
		})

	case ir.EvReasoning:
		dst = e.ensureStarted(dst)
		if e.openBlock < 0 || e.openBlockType != "thinking" {
			dst = e.closeBlock(dst)
			idx := e.blockIdx
			e.blockIdx++
			e.openBlock = idx
			e.openBlockType = "thinking"
			dst = e.appendSSE(dst, "content_block_start", map[string]any{
				"type":          "content_block_start",
				"index":         idx,
				"content_block": map[string]any{"type": "thinking", "thinking": ""},
			})
		}
		dst = e.appendSSE(dst, "content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": e.openBlock,
			"delta": map[string]any{"type": "thinking_delta", "thinking": ev.Text},
		})

	case ir.EvSignature:
		if ev.Signature != nil && e.openBlock >= 0 && e.openBlockType == "thinking" {
			e.thinkingSigVal.WriteString(ev.Signature.Value)
			if e.thinkingSigOrigin.Format == "" {
				e.thinkingSigOrigin = ev.Signature.Origin
			}
		}

	case ir.EvSealed:
		if ev.Sealed != nil {
			dst = e.ensureStarted(dst)
			dst = e.closeBlock(dst)
			idx := e.blockIdx
			e.blockIdx++
			var dataStr string
			if ev.Sealed.Kind == ir.SealedAnthropicRedacted && !needsAnthropicSeal(ev.Sealed.Origin, e.opts.Env) {
				_ = json.Unmarshal(ev.Sealed.Data, &dataStr)
			} else {
				sk := ir.SealEncrypted
				if ev.Sealed.Kind == ir.SealedAnthropicRedacted {
					sk = ir.SealRedacted
				}
				dataStr = e.opts.Env.Sealer().Seal(ir.SealRecord{
					Origin: ev.Sealed.Origin,
					Kind:   sk,
					Value:  string(ev.Sealed.Data),
				})
			}
			dst = e.appendSSE(dst, "content_block_start", map[string]any{
				"type":  "content_block_start",
				"index": idx,
				"content_block": map[string]any{
					"type": "redacted_thinking",
					"data": dataStr,
				},
			})
			dst = e.appendSSE(dst, "content_block_stop", map[string]any{
				"type":  "content_block_stop",
				"index": idx,
			})
		}

	case ir.EvToolStart:
		dst = e.ensureStarted(dst)
		if e.openBlock >= 0 && e.openBlockType == "tool_use" {
			// Another tool call is already streaming; queue this one until the active block closes.
			pt := &pendingTool{id: ev.ID, name: ev.Name}
			e.pendingTools = append(e.pendingTools, pt)
			if ev.ID != "" {
				e.pendingByID[ev.ID] = pt
			}
			break
		}
		dst = e.openToolBlock(dst, ev.ID, ev.Name)

	case ir.EvToolArgs:
		if ev.ID != "" && ev.ID != e.activeCallID {
			if pt := e.pendingByID[ev.ID]; pt != nil {
				pt.args.WriteString(ev.Text)
				break
			}
		}
		if e.openBlock >= 0 && e.openBlockType == "tool_use" && ev.Text != "" {
			dst = e.appendSSE(dst, "content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": e.openBlock,
				"delta": map[string]any{"type": "input_json_delta", "partial_json": ev.Text},
			})
		}

	case ir.EvToolEnd:
		if ev.ID != "" && ev.ID != e.activeCallID {
			if pt := e.pendingByID[ev.ID]; pt != nil {
				pt.finalArgs = ev.Arguments
				pt.done = true
				break
			}
		}
		dst = e.closeBlock(dst)
		e.activeCallID = ""
		dst = e.drainPendingTools(dst)

	case ir.EvBuiltin:
		if ev.Builtin == nil {
			break
		}
		if ev.Builtin.Kind == ir.BuiltinWebSearch || ev.Builtin.Kind == ir.BuiltinWebFetch {
			dst = e.ensureStarted(dst)
			dst = e.closeBlock(dst)
			idx := e.blockIdx
			e.blockIdx++
			id := ev.Builtin.ID
			if id == "" {
				id = "srvtoolu_" + e.opts.Env.NewID("", 16)
			}
			var cb map[string]any
			if !ev.Builtin.IsResult {
				var inObj any = map[string]any{}
				if len(ev.Builtin.Input) > 0 {
					inObj = json.RawMessage(ev.Builtin.Input)
				} else if ev.Builtin.Query != "" {
					inObj = map[string]any{"query": ev.Builtin.Query}
				} else if ev.Builtin.URL != "" {
					inObj = map[string]any{"url": ev.Builtin.URL}
				}
				cb = map[string]any{
					"type":  "server_tool_use",
					"id":    id,
					"name":  string(ev.Builtin.Kind),
					"input": inObj,
				}
			} else {
				resType := "web_search_tool_result"
				if ev.Builtin.Kind == ir.BuiltinWebFetch {
					resType = "web_fetch_tool_result"
				}
				var content any = []any{}
				if len(ev.Builtin.Hits) > 0 {
					hits := make([]map[string]any, 0, len(ev.Builtin.Hits))
					for _, h := range ev.Builtin.Hits {
						hits = append(hits, map[string]any{
							"type":  "web_search_result",
							"url":   h.URL,
							"title": h.Title,
						})
					}
					content = hits
				} else if ev.Builtin.Content != "" {
					content = ev.Builtin.Content
				}
				cb = map[string]any{
					"type":        resType,
					"tool_use_id": id,
					"content":     content,
				}
			}
			dst = e.appendSSE(dst, "content_block_start", map[string]any{
				"type":          "content_block_start",
				"index":         idx,
				"content_block": cb,
			})
			dst = e.appendSSE(dst, "content_block_stop", map[string]any{
				"type":  "content_block_stop",
				"index": idx,
			})
			break
		}
		if err := e.opts.Env.Lose(ir.LossBuiltinActivity, ir.Root.Key("content"), "builtin activity not supported by Anthropic"); err != nil {
			return dst, err
		}

	case ir.EvStop:
		dst = e.closeBlock(dst)
		e.activeCallID = ""
		dst = e.flushAllPendingTools(dst)
		if ev.Stop != nil {
			st := *ev.Stop
			e.stop = &st
		} else {
			st := ir.Stop{Reason: ir.StopEndTurn}
			e.stop = &st
		}

	case ir.EvUsage:
		if ev.Usage != nil {
			if e.usage == nil {
				u := *ev.Usage
				e.usage = &u
			} else {
				e.usage.Merge(*ev.Usage)
			}
		}

	case ir.EvError:
		if ev.Err != nil {
			e.errored = true
			dst = BuildStreamError(dst, ev.Err)
		}
	}
	return dst, nil
}

// Close writes the merged message_delta and message_stop events unless the stream ended with an error.
func (e *StreamEncoder) Close(dst []byte) ([]byte, error) {
	if e.errored {
		return dst, nil
	}
	if !e.started && e.stop == nil && e.usage == nil {
		return dst, nil
	}
	dst = e.ensureStarted(dst)
	dst = e.closeBlock(dst)
	e.activeCallID = ""
	dst = e.flushAllPendingTools(dst)
	reason := stopToAnthropic(e.stop)
	seq := ""
	if e.stop != nil {
		seq = e.stop.Sequence
	}
	var usageMap map[string]any
	if e.usage != nil {
		usageMap = buildAnthropicUsage(*e.usage)
	} else {
		usageMap = map[string]any{"output_tokens": 0}
	}
	dst = e.appendSSE(dst, "message_delta", map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   reason,
			"stop_sequence": seqOrNull(seq),
		},
		"usage": usageMap,
	})
	dst = e.appendSSE(dst, "message_stop", map[string]any{"type": "message_stop"})
	return dst, nil
}

// Keepalive sends a ping event.
func (e *StreamEncoder) Keepalive(dst []byte) []byte {
	return e.emitPing(dst)
}

func (e *StreamEncoder) emitMessageStart(dst []byte, ev ir.Event) []byte {
	usageMap := map[string]any{
		"input_tokens":  0,
		"output_tokens": 0,
	}
	if ev.Usage != nil {
		usageMap["input_tokens"] = ev.Usage.InputTokens
		if ev.Usage.Has(ir.UCacheRead) {
			usageMap["cache_read_input_tokens"] = ev.Usage.CacheReadTokens
		}
		if ev.Usage.Has(ir.UCacheWrite) {
			usageMap["cache_creation_input_tokens"] = ev.Usage.CacheWriteTokens
		}
	} else if e.opts.PromptEstimate > 0 {
		usageMap["input_tokens"] = e.opts.PromptEstimate
	}
	model := e.model
	if model == "" {
		model = e.opts.Model
	}
	msg := map[string]any{
		"id":            e.id,
		"type":          "message",
		"role":          "assistant",
		"model":         model,
		"content":       []any{},
		"stop_reason":   nil,
		"stop_sequence": nil,
		"usage":         usageMap,
	}
	return e.appendSSE(dst, "message_start", map[string]any{
		"type":    "message_start",
		"message": msg,
	})
}

func (e *StreamEncoder) emitPing(dst []byte) []byte {
	return e.appendSSE(dst, "ping", map[string]any{"type": "ping"})
}

func (e *StreamEncoder) ensureTextBlock(dst []byte) []byte {
	if e.openBlock >= 0 && e.openBlockType == "text" {
		return dst
	}
	dst = e.closeBlock(dst)
	idx := e.blockIdx
	e.blockIdx++
	e.openBlock = idx
	e.openBlockType = "text"
	return e.appendSSE(dst, "content_block_start", map[string]any{
		"type":          "content_block_start",
		"index":         idx,
		"content_block": map[string]any{"type": "text", "text": ""},
	})
}

func (e *StreamEncoder) openToolBlock(dst []byte, id, name string) []byte {
	dst = e.closeBlock(dst)
	idx := e.blockIdx
	e.blockIdx++
	e.openBlock = idx
	e.openBlockType = "tool_use"
	e.activeCallID = id
	return e.appendSSE(dst, "content_block_start", map[string]any{
		"type":  "content_block_start",
		"index": idx,
		"content_block": map[string]any{
			"type":  "tool_use",
			"id":    ir.AnthropicSafeID(id),
			"name":  name,
			"input": json.RawMessage(`{}`),
		},
	})
}

func (e *StreamEncoder) drainPendingTools(dst []byte) []byte {
	for len(e.pendingTools) > 0 {
		pt := e.pendingTools[0]
		e.pendingTools = e.pendingTools[1:]
		if pt.id != "" {
			delete(e.pendingByID, pt.id)
		}
		dst = e.openToolBlock(dst, pt.id, pt.name)
		if s := pt.args.String(); s != "" {
			dst = e.appendSSE(dst, "content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": e.openBlock,
				"delta": map[string]any{"type": "input_json_delta", "partial_json": s},
			})
		} else if len(pt.finalArgs) > 0 {
			dst = e.appendSSE(dst, "content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": e.openBlock,
				"delta": map[string]any{"type": "input_json_delta", "partial_json": string(pt.finalArgs)},
			})
		}
		if pt.done {
			dst = e.closeBlock(dst)
			e.activeCallID = ""
			continue
		}
		// Keep this tool open to receive further live deltas.
		break
	}
	return dst
}

func (e *StreamEncoder) flushAllPendingTools(dst []byte) []byte {
	for len(e.pendingTools) > 0 {
		pt := e.pendingTools[0]
		pt.done = true
		dst = e.drainPendingTools(dst)
	}
	return dst
}

func (e *StreamEncoder) closeBlock(dst []byte) []byte {
	if e.openBlock < 0 {
		return dst
	}
	idx := e.openBlock
	if e.openBlockType == "thinking" && e.thinkingSigVal.Len() > 0 {
		sigVal := e.thinkingSigVal.String()
		if needsAnthropicSeal(e.thinkingSigOrigin, e.opts.Env) {
			sigVal = e.opts.Env.Sealer().Seal(ir.SealRecord{
				Origin: e.thinkingSigOrigin,
				Kind:   ir.SealSignature,
				Value:  sigVal,
			})
		}
		dst = e.appendSSE(dst, "content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": idx,
			"delta": map[string]any{"type": "signature_delta", "signature": sigVal},
		})
	}
	e.thinkingSigVal.Reset()
	e.thinkingSigOrigin = ir.Origin{}
	e.openBlock = -1
	e.openBlockType = ""
	return e.appendSSE(dst, "content_block_stop", map[string]any{
		"type":  "content_block_stop",
		"index": idx,
	})
}

func (e *StreamEncoder) appendSSE(dst []byte, eventName string, v any) []byte {
	b, err := jsonx.Marshal(v)
	if err != nil {
		return dst
	}
	return sse.AppendEvent(dst, eventName, b)
}

// BuildResponse serialises a whole ir.Response to a non-streaming Anthropic body.
func BuildResponse(resp *ir.Response, opts EncodeOptions) ([]byte, error) {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	id := resp.ID
	if id == "" {
		id = "msg_" + env.NewID("", 24)
	}
	model := opts.Model
	if model == "" {
		model = resp.Model
	}

	var out *ir.Output
	if len(resp.Outputs) > 0 {
		out = &resp.Outputs[0]
	}

	var blocks []wireContentBlock
	if out != nil {
		for i, p := range out.Parts {
			fp := ir.Root.Key("content").At(i)
			bs, err := responsePartToBlocks(p, fp, env)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, bs...)
		}
	}
	if blocks == nil {
		blocks = []wireContentBlock{}
	}

	var stopReason, stopSeq string
	if out != nil {
		stopReason = stopToAnthropic(&out.Stop)
		stopSeq = out.Stop.Sequence
	}

	content, err := jsonx.Marshal(blocks)
	if err != nil {
		return nil, err
	}

	wr := map[string]any{
		"id":            id,
		"type":          "message",
		"role":          "assistant",
		"model":         model,
		"content":       json.RawMessage(content),
		"stop_reason":   stopReason,
		"stop_sequence": seqOrNull(stopSeq),
		"usage":         buildAnthropicUsage(resp.Usage),
	}
	return jsonx.Marshal(wr)
}

func needsAnthropicSeal(origin ir.Origin, env *ir.Env) bool {
	if origin.Format != ir.FormatAnthropic {
		return true
	}
	if origin.Provider != "" {
		return true
	}
	return env != nil && env.ProviderID != ""
}

func responsePartToBlocks(part ir.Part, p ir.Path, env *ir.Env) ([]wireContentBlock, error) {
	if part.Type == ir.PartReasoning && part.Reasoning != nil {
		r := part.Reasoning
		var out []wireContentBlock
		if r.Text != "" {
			sig := ""
			if r.Signature != nil && r.Signature.Value != "" {
				if !needsAnthropicSeal(r.Signature.Origin, env) {
					sig = r.Signature.Value
				} else {
					sig = env.Sealer().Seal(ir.SealRecord{
						Origin:  r.Signature.Origin,
						Kind:    ir.SealSignature,
						Value:   r.Signature.Value,
						TextSum: ir.TextSum(r.Text),
					})
				}
			}
			out = append(out, wireContentBlock{Type: "thinking", Thinking: r.Text, Signature: sig})
		}
		if r.Sealed != nil {
			var dataStr string
			if r.Sealed.Kind == ir.SealedAnthropicRedacted && !needsAnthropicSeal(r.Sealed.Origin, env) {
				_ = json.Unmarshal(r.Sealed.Data, &dataStr)
			} else {
				sk := ir.SealEncrypted
				if r.Sealed.Kind == ir.SealedAnthropicRedacted {
					sk = ir.SealRedacted
				}
				dataStr = env.Sealer().Seal(ir.SealRecord{
					Origin: r.Sealed.Origin,
					Kind:   sk,
					Value:  string(r.Sealed.Data),
				})
			}
			out = append(out, wireContentBlock{Type: "redacted_thinking", Data: dataStr})
		}
		return out, nil
	}
	if part.Type == ir.PartBuiltinCall || part.Type == ir.PartBuiltinResult {
		if part.Builtin != nil && (part.Builtin.Kind == ir.BuiltinWebSearch || part.Builtin.Kind == ir.BuiltinWebFetch) {
			id := part.Builtin.ID
			if id == "" {
				id = "srvtoolu_" + env.NewID("", 16)
			}
			if !part.Builtin.IsResult {
				inRaw := part.Builtin.Input
				if len(inRaw) == 0 {
					inRaw = json.RawMessage(`{}`)
				}
				return []wireContentBlock{{Type: "server_tool_use", ID: id, Name: string(part.Builtin.Kind), Input: inRaw}}, nil
			}
			resType := "web_search_tool_result"
			if part.Builtin.Kind == ir.BuiltinWebFetch {
				resType = "web_fetch_tool_result"
			}
			var content json.RawMessage
			if len(part.Builtin.Hits) > 0 {
				hits := make([]map[string]any, 0, len(part.Builtin.Hits))
				for _, h := range part.Builtin.Hits {
					hits = append(hits, map[string]any{"type": "web_search_result", "url": h.URL, "title": h.Title})
				}
				content, _ = jsonx.Marshal(hits)
			} else {
				content, _ = jsonx.Marshal(part.Builtin.Content)
			}
			return []wireContentBlock{{Type: resType, ToolUseID: id, Content: content}}, nil
		}
		if err := env.Lose(ir.LossBuiltinActivity, p, "builtin activity not supported by Anthropic"); err != nil {
			return nil, err
		}
		return nil, nil
	}
	return partToBlocks(part, p, env)
}

func stopToAnthropic(s *ir.Stop) string {
	if s == nil {
		return "end_turn"
	}
	switch s.Reason {
	case ir.StopEndTurn:
		return "end_turn"
	case ir.StopMaxTokens, ir.StopContextWindow:
		return "max_tokens"
	case ir.StopSequence:
		return "stop_sequence"
	case ir.StopToolUse:
		return "tool_use"
	case ir.StopContentFilter, ir.StopRefusal:
		return "refusal"
	case ir.StopPauseTurn:
		return "pause_turn"
	}
	return "end_turn"
}

func buildAnthropicUsage(u ir.Usage) map[string]any {
	// Anthropic input_tokens excludes cache_read_input_tokens and cache_creation_input_tokens.
	m := map[string]any{
		"input_tokens":  u.InputTokens,
		"output_tokens": u.OutputTokens,
	}
	if u.Has(ir.UCacheRead) {
		m["cache_read_input_tokens"] = u.CacheReadTokens
	}
	if u.Has(ir.UCacheWrite) {
		m["cache_creation_input_tokens"] = u.CacheWriteTokens
	}
	if u.Has(ir.UCacheWriteSplit) && (u.CacheWrite5mTokens > 0 || u.CacheWrite1hTokens > 0) {
		cc := map[string]any{}
		if u.CacheWrite5mTokens > 0 {
			cc["ephemeral_5m_input_tokens"] = u.CacheWrite5mTokens
		}
		if u.CacheWrite1hTokens > 0 {
			cc["ephemeral_1h_input_tokens"] = u.CacheWrite1hTokens
		}
		m["cache_creation"] = cc
	}
	if u.WebSearchRequests > 0 || u.WebFetchRequests > 0 {
		stu := map[string]any{}
		if u.WebSearchRequests > 0 {
			stu["web_search_requests"] = u.WebSearchRequests
		}
		if u.WebFetchRequests > 0 {
			stu["web_fetch_requests"] = u.WebFetchRequests
		}
		m["server_tool_use"] = stu
	}
	return m
}

func seqOrNull(seq string) any {
	if seq == "" {
		return nil
	}
	return seq
}
