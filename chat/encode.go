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
	callSentArgs map[int]map[string]*strings.Builder
	callEnded    map[int]map[string]bool
	callKind     map[int]map[string]ir.ToolKind
	customOpened map[int]map[string]bool
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
		callSentArgs: map[int]map[string]*strings.Builder{},
		callEnded:    map[int]map[string]bool{},
		callKind:     map[int]map[string]ir.ToolKind{},
		customOpened: map[int]map[string]bool{},
		lastCallID:   map[int]string{},
		lastCallSlot: map[int]int{},
	}
}

// Encode serialises one ir.Event to Chat SSE bytes and appends to dst.
// Not all events produce output; the return is nil-extended when there is nothing to send.
func (e *StreamEncoder) Encode(dst []byte, ev ir.Event) ([]byte, error) {
	if e.errored {
		return dst, nil
	}
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

	case ir.EvRefusal:
		chunk := e.baseChunk(ev.Index)
		chunk["choices"] = []map[string]any{{
			"index":         ev.Index,
			"delta":         map[string]any{"refusal": ev.Text},
			"finish_reason": nil,
		}}
		dst = e.appendChunk(dst, chunk)

	case ir.EvCitation:
		if ev.Citation != nil {
			if err := e.opts.Env.Lose(ir.LossCitations, ir.Root.Key("choices"), "citations not supported by Chat Completions"); err != nil {
				return dst, err
			}
		}

	case ir.EvImage:
		if ev.Media != nil {
			if err := e.opts.Env.Lose(ir.LossOutputMedia, ir.Root.Key("choices"), "output media not supported by Chat Completions"); err != nil {
				return dst, err
			}
		}

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
			e.callSentArgs[ev.Index] = map[string]*strings.Builder{}
			e.callEnded[ev.Index] = map[string]bool{}
			e.callKind[ev.Index] = map[string]ir.ToolKind{}
			e.customOpened[ev.Index] = map[string]bool{}
		}
		key := ev.ID
		if ev.ID != "" {
			e.callSlots[ev.Index][ev.ID] = idx
		}
		e.callKind[ev.Index][key] = ev.Kind
		e.callSentArgs[ev.Index][key] = &strings.Builder{}
		e.callEnded[ev.Index][key] = false
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
		if e.callSentArgs[ev.Index] == nil {
			e.callSentArgs[ev.Index] = map[string]*strings.Builder{}
		}
		if e.callSentArgs[ev.Index][key] == nil {
			e.callSentArgs[ev.Index][key] = &strings.Builder{}
		}
		argDelta := ev.Text
		if e.callKind[ev.Index] != nil && e.callKind[ev.Index][key] == ir.ToolCustom {
			if e.customOpened[ev.Index] == nil {
				e.customOpened[ev.Index] = map[string]bool{}
			}
			var sb strings.Builder
			if !e.customOpened[ev.Index][key] {
				e.customOpened[ev.Index][key] = true
				sb.WriteString(`{"input":"`)
			}
			if b, err := jsonx.Marshal(ev.Text); err == nil && len(b) >= 2 {
				sb.Write(b[1 : len(b)-1])
			}
			e.callSentArgs[ev.Index][key].WriteString(ev.Text)
			argDelta = sb.String()
			e.callHasArgs[ev.Index][key] = true
		} else if ev.Text != "" {
			e.callSentArgs[ev.Index][key].WriteString(ev.Text)
			e.callHasArgs[ev.Index][key] = true
		}
		chunk := e.baseChunk(ev.Index)
		tc := map[string]any{
			"index": idx,
			"function": map[string]any{
				"arguments": argDelta,
			},
		}
		chunk["choices"] = []map[string]any{{
			"index":         ev.Index,
			"delta":         map[string]any{"tool_calls": []any{tc}},
			"finish_reason": nil,
		}}
		dst = e.appendChunk(dst, chunk)

	case ir.EvBuiltin:
		if ev.Builtin != nil {
			if err := e.opts.Env.Lose(ir.LossBuiltinActivity, ir.Root.Key("choices"), "builtin activity not supported by Chat Completions"); err != nil {
				return dst, err
			}
		}

	case ir.EvToolEnd:
		if ev.Err != nil {
			// An already-emitted live Chat tool_calls delta cannot be retracted;
			// abort with a stream error instead of finalizing args or emitting [DONE].
			e.errored = true
			dst = BuildStreamError(dst, ev.Err)
			break
		}
		idx, key := e.resolveCallSlot(ev.Index, ev.ID)
		if e.callEnded[ev.Index] != nil {
			e.callEnded[ev.Index][key] = true
		}
		var sent string
		if e.callSentArgs[ev.Index] != nil && e.callSentArgs[ev.Index][key] != nil {
			sent = e.callSentArgs[ev.Index][key].String()
		}
		if e.callKind[ev.Index] != nil && e.callKind[ev.Index][key] == ir.ToolCustom {
			opened := e.customOpened[ev.Index] != nil && e.customOpened[ev.Index][key]
			full, suffix, ok := jsonx.ReconcileCustomInput(sent, opened, ev.Arguments)
			if !ok {
				e.errored = true
				dst = BuildStreamError(dst, &ir.APIError{
					Status:  502,
					Type:    ir.ErrServer,
					Message: "streamed custom tool input does not match final input",
				})
				break
			}
			var tail string
			if opened {
				var sb strings.Builder
				if suffix != "" {
					if b, err := jsonx.Marshal(suffix); err == nil && len(b) >= 2 {
						sb.Write(b[1 : len(b)-1])
					}
				}
				sb.WriteString(`"}`)
				tail = sb.String()
			} else {
				b, _ := jsonx.Marshal(map[string]string{"input": full})
				tail = string(b)
			}
			if e.callHasArgs[ev.Index] == nil {
				e.callHasArgs[ev.Index] = map[string]bool{}
			}
			e.callHasArgs[ev.Index][key] = true
			chunk := e.baseChunk(ev.Index)
			tc := map[string]any{
				"index": idx,
				"function": map[string]any{
					"arguments": tail,
				},
			}
			chunk["choices"] = []map[string]any{{
				"index":         ev.Index,
				"delta":         map[string]any{"tool_calls": []any{tc}},
				"finish_reason": nil,
			}}
			dst = e.appendChunk(dst, chunk)
			break
		}
		full, suffix, ok := jsonx.ReconcileFunctionArgs(sent, string(ev.Arguments))
		if !ok {
			e.errored = true
			dst = BuildStreamError(dst, &ir.APIError{
				Status:  502,
				Type:    ir.ErrServer,
				Message: "streamed tool arguments do not match final arguments",
			})
			break
		}
		emitDelta := suffix
		if sent == "" {
			emitDelta = full
		}
		if emitDelta != "" {
			if e.callHasArgs[ev.Index] == nil {
				e.callHasArgs[ev.Index] = map[string]bool{}
			}
			e.callHasArgs[ev.Index][key] = true
			chunk := e.baseChunk(ev.Index)
			tc := map[string]any{
				"index": idx,
				"function": map[string]any{
					"arguments": emitDelta,
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
		truncatedOrFiltered := ev.Stop != nil && (ev.Stop.Truncated() || ev.Stop.Filtered())
		if !truncatedOrFiltered {
			for _, ended := range e.callEnded[ev.Index] {
				if !ended {
					e.errored = true
					dst = BuildStreamError(dst, &ir.APIError{
						Status:  502,
						Type:    ir.ErrServer,
						Message: "tool call not completed before stop",
					})
					return dst, nil
				}
			}
		}
		finish := finishFromStop(ev.Stop)
		chunk := e.baseChunk(ev.Index)
		chunk["choices"] = []map[string]any{{
			"index":         ev.Index,
			"delta":         map[string]any{},
			"finish_reason": finish,
		}}
		dst = e.appendChunk(dst, chunk)
		// Reset per-choice tool state for the finished candidate.
		delete(e.callIdxBy, ev.Index)
		delete(e.callSlots, ev.Index)
		delete(e.callHasArgs, ev.Index)
		delete(e.callSentArgs, ev.Index)
		delete(e.callEnded, ev.Index)
		delete(e.callKind, ev.Index)
		delete(e.customOpened, ev.Index)
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
	for _, m := range e.callEnded {
		for _, ended := range m {
			if !ended {
				e.errored = true
				dst = BuildStreamError(dst, &ir.APIError{
					Status:  502,
					Type:    ir.ErrServer,
					Message: "tool call not completed before stream close",
				})
				return dst, nil
			}
		}
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
		wm, err := buildAssistantMessage(out.Parts, env)
		if err != nil {
			return nil, err
		}
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
	return ir.EncodeCallSignatureID(env, id, sig)
}

// buildAssistantMessage converts output parts to a wireMessage.
func buildAssistantMessage(parts []ir.Part, env *ir.Env) (wireMessage, error) {
	wm := wireMessage{Role: "assistant"}
	var textParts []wireContentPart
	var toolCalls []wireToolCall
	var reasoning string

	for i, p := range parts {
		if len(p.Citations) > 0 {
			if err := env.Lose(ir.LossCitations, ir.Root.Key("choices").At(0).Key("message").At(i), "citations not supported by Chat Completions"); err != nil {
				return wireMessage{}, err
			}
		}
		switch p.Type {
		case ir.PartText:
			textParts = append(textParts, wireContentPart{Type: "text", Text: p.Text})
		case ir.PartImage:
			if err := env.Lose(ir.LossOutputMedia, ir.Root.Key("choices").At(0).Key("message").At(i), "output media not supported by Chat Completions"); err != nil {
				return wireMessage{}, err
			}
		case ir.PartRefusal:
			wm.Refusal = p.Text
		case ir.PartReasoning:
			if p.Reasoning != nil {
				reasoning += p.Reasoning.Text
			}
		case ir.PartBuiltinCall, ir.PartBuiltinResult:
			if err := env.Lose(ir.LossBuiltinActivity, ir.Root.Key("choices").At(0).Key("message").At(i), "builtin activity not supported by Chat Completions"); err != nil {
				return wireMessage{}, err
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
	return wm, nil
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
	prompt := u.PromptTokens()
	comp := u.OutputTokens
	tot := prompt + comp
	wu := &wireUsage{
		PromptTokens:     &prompt,
		CompletionTokens: &comp,
		TotalTokens:      &tot,
	}
	if u.Has(ir.UCacheRead) {
		v := u.CacheReadTokens
		wu.PromptTokensDetails = &wirePromptTokensDetails{
			CachedTokens: &v,
		}
	}
	if u.Has(ir.UReasoning) {
		v := u.ReasoningTokens
		wu.CompletionTokensDetails = &wireCompletionTokensDetails{
			ReasoningTokens: &v,
		}
	}
	return wu
}

// suppress unused import warning
var _ = json.Marshal
