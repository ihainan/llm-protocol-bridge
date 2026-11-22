package chat

import (
	"encoding/json"
	"strings"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// StreamEncoder converts ir.Events into Chat Completions SSE chunks.
type StreamEncoder struct {
	opts    EncodeOptions
	id      string
	model   string
	created int64
	started bool
	errored bool
	// callIdxBy tracks the next tool call slot to assign for each choice index.
	callIdxBy    map[int]int
	callSlots    map[int]map[string]int
	callHasArgs  map[int]map[string]bool
	lastCallID   map[int]string
	lastCallSlot map[int]int
	usage        ir.Usage
}

// NewStreamEncoder returns a Chat Completions stream encoder.
func NewStreamEncoder(opts EncodeOptions) *StreamEncoder {
	return &StreamEncoder{
		opts:         opts,
		callIdxBy:    map[int]int{},
		callSlots:    map[int]map[string]int{},
		callHasArgs:  map[int]map[string]bool{},
		lastCallID:   map[int]string{},
		lastCallSlot: map[int]int{},
	}
}

// Encode serialises one ir.Event to Chat SSE bytes and appends to dst.
// Not all events produce output; the return is nil-extended when there is nothing to send.
func (e *StreamEncoder) Encode(dst []byte, ev ir.Event) ([]byte, error) {
	switch ev.Type {
	case ir.EvStart:
		if ev.Usage != nil {
			e.usage.Merge(*ev.Usage)
		}
		// Keep response identity stable across multi-round continuations (NextRound).
		if !e.started {
			e.started = true
			e.id = ev.ID
			if e.id == "" {
				e.id = "chatcmpl-" + e.opts.Env.NewID("", 24)
			}
			e.model = e.opts.Model
			if e.model == "" && ev.Model != "" {
				e.model = ev.Model
			}
			e.created = e.opts.Env.Now().Unix()

			chunk := e.baseChunk(0)
			chunk["choices"] = []map[string]any{{
				"index":         0,
				"delta":         map[string]any{"role": "assistant", "content": ""},
				"finish_reason": nil,
			}}
			dst = e.appendChunk(dst, chunk)
		}

	case ir.EvText:
		chunk := e.baseChunk(ev.Index)
		chunk["choices"] = []map[string]any{{
			"index":         ev.Index,
			"delta":         map[string]any{"content": ev.Text},
			"finish_reason": nil,
		}}
		dst = e.appendChunk(dst, chunk)

	case ir.EvReasoning:
		chunk := e.baseChunk(ev.Index)
		chunk["choices"] = []map[string]any{{
			"index":         ev.Index,
			"delta":         map[string]any{"reasoning_content": ev.Text},
			"finish_reason": nil,
		}}
		dst = e.appendChunk(dst, chunk)

	case ir.EvToolStart:
		idx := e.callIdxBy[ev.Index]
		e.callIdxBy[ev.Index] = idx + 1
		if e.callSlots[ev.Index] == nil {
			e.callSlots[ev.Index] = map[string]int{}
			e.callHasArgs[ev.Index] = map[string]bool{}
		}
		if ev.ID != "" {
			e.callSlots[ev.Index][ev.ID] = idx
		}
		e.lastCallID[ev.Index] = ev.ID
		e.lastCallSlot[ev.Index] = idx
		wireName := flattenToolName(ev.Namespace, ev.Name)
		wireID := encodeChatCallID(e.opts.Env, ev.ID, ev.Signature)
		chunk := e.baseChunk(ev.Index)
		tc := map[string]any{
			"index": idx,
			"id":    wireID,
			"type":  "function",
			"function": map[string]any{
				"name":      wireName,
				"arguments": "",
			},
		}
		chunk["choices"] = []map[string]any{{
			"index":         ev.Index,
			"delta":         map[string]any{"tool_calls": []any{tc}},
			"finish_reason": nil,
		}}
		dst = e.appendChunk(dst, chunk)

	case ir.EvToolArgs:
		idx, key := e.resolveCallSlot(ev.Index, ev.ID)
		if e.callHasArgs[ev.Index] == nil {
			e.callHasArgs[ev.Index] = map[string]bool{}
		}
		if ev.Text != "" {
			e.callHasArgs[ev.Index][key] = true
		}
		chunk := e.baseChunk(ev.Index)
		tc := map[string]any{
			"index": idx,
			"function": map[string]any{
				"arguments": ev.Text,
			},
		}
		chunk["choices"] = []map[string]any{{
			"index":         ev.Index,
			"delta":         map[string]any{"tool_calls": []any{tc}},
			"finish_reason": nil,
		}}
		dst = e.appendChunk(dst, chunk)

	case ir.EvToolEnd:
		idx, key := e.resolveCallSlot(ev.Index, ev.ID)
		if e.callHasArgs[ev.Index] == nil || !e.callHasArgs[ev.Index][key] {
			argsStr := strings.TrimSpace(string(ev.Arguments))
			if argsStr == "" {
				argsStr = "{}"
			}
			if e.callHasArgs[ev.Index] == nil {
				e.callHasArgs[ev.Index] = map[string]bool{}
			}
			e.callHasArgs[ev.Index][key] = true
			chunk := e.baseChunk(ev.Index)
			tc := map[string]any{
				"index": idx,
				"function": map[string]any{
					"arguments": argsStr,
				},
			}
			chunk["choices"] = []map[string]any{{
				"index":         ev.Index,
				"delta":         map[string]any{"tool_calls": []any{tc}},
				"finish_reason": nil,
			}}
			dst = e.appendChunk(dst, chunk)
		}

	case ir.EvStop:
		finish := finishFromStop(ev.Stop)
		chunk := e.baseChunk(ev.Index)
		chunk["choices"] = []map[string]any{{
			"index":         ev.Index,
			"delta":         map[string]any{},
			"finish_reason": finish,
		}}
		dst = e.appendChunk(dst, chunk)
		// Reset per-choice tool index for the next candidate.
		delete(e.callIdxBy, ev.Index)
		delete(e.callSlots, ev.Index)
		delete(e.callHasArgs, ev.Index)
		delete(e.lastCallID, ev.Index)
		delete(e.lastCallSlot, ev.Index)

	case ir.EvUsage:
		if ev.Usage != nil {
			e.usage.Merge(*ev.Usage)
			u := buildUsage(e.usage)
			chunk := e.baseChunk(0)
			chunk["choices"] = []any{}
			chunk["usage"] = u
			dst = e.appendChunk(dst, chunk)
		}

	case ir.EvError:
		if ev.Err != nil {
			e.errored = true
			dst = BuildStreamError(dst, ev.Err)
		}
	}
	return dst, nil
}

// Close writes the [DONE] SSE marker unless the stream ended with an error.
func (e *StreamEncoder) Close(dst []byte) ([]byte, error) {
	if e.errored {
		return dst, nil
	}
	dst = append(dst, "data: [DONE]\n\n"...)
	return dst, nil
}

// Keepalive writes a Chat SSE comment (: keepalive).
func (e *StreamEncoder) Keepalive(dst []byte) []byte {
	return sse.AppendComment(dst, "keepalive", true)
}

func (e *StreamEncoder) resolveCallSlot(candIdx int, callID string) (int, string) {
	if callID != "" {
		if m := e.callSlots[candIdx]; m != nil {
			if slot, ok := m[callID]; ok {
				return slot, callID
			}
		}
	}
	return e.lastCallSlot[candIdx], e.lastCallID[candIdx]
}

func (e *StreamEncoder) baseChunk(idx int) map[string]any {
	_ = idx
	return map[string]any{
		"id":      e.id,
		"object":  "chat.completion.chunk",
		"created": e.created,
		"model":   e.model,
	}
}

func (e *StreamEncoder) appendChunk(dst []byte, chunk map[string]any) []byte {
	b, err := jsonx.Marshal(chunk)
	if err != nil {
		return dst
	}
	return sse.AppendData(dst, b)
}

// BuildResponse serialises a whole ir.Response to a non-streaming Chat Completions body.
func BuildResponse(resp *ir.Response, opts EncodeOptions) ([]byte, error) {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	id := resp.ID
	if id == "" {
		id = "chatcmpl-" + env.NewID("", 24)
	}
	model := opts.Model
	if model == "" {
		model = resp.Model
	}
	created := resp.Created
	if created == 0 {
		created = env.Now().Unix()
	}

	choices := make([]wireChoice, 0, len(resp.Outputs))
	for _, out := range resp.Outputs {
		wm := buildAssistantMessage(out.Parts, env)
		choices = append(choices, wireChoice{
			Index:        out.Index,
			Message:      wm,
			FinishReason: finishFromStop(&out.Stop),
		})
	}

	wr := wireResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: created,
		Model:   model,
		Choices: choices,
	}
	if resp.Usage.Any() {
		raw, _ := jsonx.Marshal(usageToWire(resp.Usage))
		wr.Usage = raw
	}
	return jsonx.Marshal(wr)
}

func encodeChatCallID(env *ir.Env, id string, sig *ir.Signature) string {
	if sig != nil && sig.Value != "" && sig.Origin.Format == ir.FormatGemini {
		sealed := env.Sealer().Seal(ir.SealRecord{
			Origin: sig.Origin,
			Kind:   ir.SealSignature,
			Value:  sig.Value,
		})
		return id + "::" + sealed
	}
	return id
}

// buildAssistantMessage converts output parts to a wireMessage.
func buildAssistantMessage(parts []ir.Part, env *ir.Env) wireMessage {
	wm := wireMessage{Role: "assistant"}
	var textParts []wireContentPart
	var toolCalls []wireToolCall
	var reasoning string

	for _, p := range parts {
		switch p.Type {
		case ir.PartText:
			textParts = append(textParts, wireContentPart{Type: "text", Text: p.Text})
		case ir.PartRefusal:
			wm.Refusal = p.Text
		case ir.PartReasoning:
			if p.Reasoning != nil {
				reasoning += p.Reasoning.Text
			}
		case ir.PartToolCall:
			if p.Call != nil {
				args := string(p.Call.Arguments)
				if p.Call.Kind == ir.ToolCustom && args == "" {
					b, _ := jsonx.Marshal(map[string]string{"input": p.Call.Input})
					args = string(b)
				}
				if strings.TrimSpace(args) == "" {
					args = "{}"
				}
				tc := wireToolCall{
					ID:   encodeChatCallID(env, p.Call.ID, p.Call.Signature),
					Type: "function",
					Function: wireFunctionCall{
						Name:      flattenToolName(p.Call.Namespace, p.Call.Name),
						Arguments: args,
					},
				}
				toolCalls = append(toolCalls, tc)
			}
		}
	}

	if reasoning != "" {
		wm.ReasoningContent = reasoning
	}
	if len(toolCalls) > 0 {
		wm.ToolCalls = toolCalls
	}
	switch len(textParts) {
	case 0:
		if len(toolCalls) > 0 {
			wm.Content = jsonRaw("null")
		}
	case 1:
		wm.Content = jsonRawString(textParts[0].Text)
	default:
		// Chat Completions assistant response message.content is always a string (or null), never an array.
		var sb strings.Builder
		for _, tp := range textParts {
			sb.WriteString(tp.Text)
		}
		wm.Content = jsonRawString(sb.String())
	}
	return wm
}

// finishFromStop maps ir.Stop to a Chat finish_reason string.
func finishFromStop(s *ir.Stop) string {
	if s == nil {
		return "stop"
	}
	switch s.Reason {
	case ir.StopEndTurn, ir.StopPauseTurn, ir.StopSequence:
		return "stop"
	case ir.StopMaxTokens, ir.StopContextWindow:
		return "length"
	case ir.StopToolUse:
		return "tool_calls"
	case ir.StopContentFilter, ir.StopRefusal:
		return "content_filter"
	}
	return "stop"
}

// buildUsage assembles the wire usage object from ir.Usage.
func buildUsage(u ir.Usage) map[string]any {
	prompt := u.PromptTokens()
	completion := u.OutputTokens
	total := prompt + completion

	out := map[string]any{
		"prompt_tokens":     prompt,
		"completion_tokens": completion,
		"total_tokens":      total,
	}
	if u.Has(ir.UCacheRead) {
		out["prompt_tokens_details"] = map[string]any{
			"cached_tokens": u.CacheReadTokens,
		}
	}
	if u.Has(ir.UReasoning) {
		out["completion_tokens_details"] = map[string]any{
			"reasoning_tokens": u.ReasoningTokens,
		}
	}
	return out
}

// usageToWire converts ir.Usage to wireUsage for non-streaming response.
func usageToWire(u ir.Usage) *wireUsage {
	wu := &wireUsage{
		PromptTokens:     u.PromptTokens(),
		CompletionTokens: u.OutputTokens,
		TotalTokens:      u.PromptTokens() + u.OutputTokens,
	}
	if u.Has(ir.UCacheRead) {
		wu.PromptTokensDetails = &wirePromptTokensDetails{
			CachedTokens: u.CacheReadTokens,
		}
	}
	if u.Has(ir.UReasoning) {
		wu.CompletionTokensDetails = &wireCompletionTokensDetails{
			ReasoningTokens: u.ReasoningTokens,
		}
	}
	return wu
}

// suppress unused import warning
var _ = json.Marshal
