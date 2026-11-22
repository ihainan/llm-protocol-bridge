package chat

import (
	"encoding/json"
	"strings"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// StreamDecoder converts Chat Completions streaming chunks into ir.Event.
type StreamDecoder struct {
	env   *ir.Env
	dia   Dialect
	names map[string]ir.ToolRef

	// State per stream.
	started    bool
	id         string
	model      string
	created    int64
	candidates map[int]*chatCandidate // index -> candidate state
	// currentToolIndex maps chat tool-call index -> the open tool call's IR state
	// (chat calls share one index per call, and multiple calls may interleave).
	toolCalls map[int]*openCall

	// Names we've seen, for dedup of call IDs within one response.
	seenCallIDs map[string]bool

	done     bool
	terminal bool
}

// openCall tracks one tool call in flight within a candidate.
type openCall struct {
	// partIndex is the index of the tool-call part in the candidate's parts slice.
	partIndex int
	// nameBuilt accumulates the streamed name fragments.
	nameBuilt strings.Builder
	// argsBuilt accumulates the streamed arguments fragments.
	argsBuilt strings.Builder
	// nameSeen is whether the first fragment with a name has arrived (so we can
	// emit a tool_start event the first time we know a call's index).
	started bool
}

// chatCandidate is per-choice state in the stream.
type chatCandidate struct {
	index int
	// openCallIdx is the index of the tool call whose arguments are currently
	// streaming (used only for ordering).
	openCallIdx int
	hasCall     bool
}

// NewStreamDecoder returns a Chat Completions stream decoder.
func NewStreamDecoder(opts DecodeOptions) *StreamDecoder {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	env.Style = ir.StyleBracket
	return &StreamDecoder{
		env:         env,
		dia:         opts.Dialect,
		names:       opts.Names,
		candidates:  map[int]*chatCandidate{},
		toolCalls:   map[int]*openCall{},
		seenCallIDs: map[string]bool{},
	}
}

// Decode converts one SSE event into zero or more IR events.
// The caller owns out; Decode appends to it and returns the slice.
func (d *StreamDecoder) Decode(out []ir.Event, ev sse.Event) ([]ir.Event, error) {
	if sse.IsDone(ev.Data) {
		d.done = true
		return out, nil
	}
	if len(ev.Data) == 0 {
		return out, nil
	}
	// A bare JSON line may carry an error.
	if jsonx.Kind(ev.Data) != '{' {
		// Try to parse as an error envelope.
		if e := ParseError(0, ev.Data); e != nil && e.Message != "" {
			out = append(out, ir.Event{Type: ir.EvError, Err: e})
			d.terminal = true
			return out, nil
		}
		return out, nil
	}

	var chunk wireChunk
	if json.Unmarshal(ev.Data, &chunk) != nil {
		return out, nil // tolerate malformed chunks; treat as keepalive
	}

	// Error in the chunk body.
	if chunk.Error != nil && chunk.Error.Message != "" {
		e := &ir.APIError{
			Status:  0,
			Type:    ir.TypeForStatus(0),
			Message: chunk.Error.Message,
			Code:    stringOrEmpty(chunk.Error.Code),
			Param:   chunk.Error.Param,
			Raw:     append([]byte(nil), ev.Data...),
		}
		if chunk.Error.Type != "" {
			e.Type = mapErrorType(chunk.Error.Type)
		}
		out = append(out, ir.Event{Type: ir.EvError, Err: e})
		d.terminal = true
		return out, nil
	}

	// First chunk: emit start with id, model, created. Prompt-side usage may
	// arrive in the first chunk (some providers) but more commonly in the last.
	if !d.started {
		d.started = true
		d.id = chunk.ID
		d.model = chunk.Model
		d.created = chunk.Created
		// If the first chunk already has usage (some dialects send prompt usage
		// early), attach it to start.
		var usage *ir.Usage
		if chunk.Usage != nil {
			u := decodeUsage(chunk.Usage, d.dia)
			usage = &u
		}
		out = append(out, ir.Event{
			Type:  ir.EvStart,
			ID:    chunk.ID,
			Model: chunk.Model,
			Usage: usage,
		})
	}

	// Decode each choice.
	for _, ch := range chunk.Choices {
		ci := ch.Index
		if d.candidates[ci] == nil {
			d.candidates[ci] = &chatCandidate{index: ci}
		}
		out = d.decodeChoice(out, ci, ch)
	}

	// Usage on a later chunk (the final usage chunk has choices: []).
	if chunk.Usage != nil {
		u := decodeUsage(chunk.Usage, d.dia)
		out = append(out, ir.Event{Type: ir.EvUsage, Usage: &u})
	}

	return out, nil
}

func (d *StreamDecoder) decodeChoice(out []ir.Event, ci int, ch wireStreamChoice) []ir.Event {
	delta := ch.Delta

	// Role chunk (only on the first delta of a choice).
	if delta.Role != "" {
		// No explicit event needed; the start event covers the id/model.
	}

	// Content text delta.
	if contentStr := rawToString(delta.Content); contentStr != "" {
		out = append(out, ir.Event{Type: ir.EvText, Index: ci, Text: contentStr})
	}

	// Reasoning delta.
	reasoning := delta.ReasoningContent
	if reasoning == "" {
		reasoning = delta.Reasoning
	}
	if reasoning != "" {
		out = append(out, ir.Event{Type: ir.EvReasoning, Index: ci, Text: reasoning})
	}

	// Refusal.
	if delta.Refusal != "" {
		out = append(out, ir.Event{Type: ir.EvText, Index: ci, Text: delta.Refusal})
	}

	// Tool calls.
	if len(delta.ToolCalls) > 0 {
		for _, tc := range delta.ToolCalls {
			out = d.decodeToolCallDelta(out, ci, tc)
		}
	}

	// Finish reason.
	if ch.FinishReason != "" {
		out = d.finalizeChoice(out, ci, ch.FinishReason)
	}

	return out
}

func (d *StreamDecoder) decodeToolCallDelta(out []ir.Event, ci int, tc wireToolCall) []ir.Event {
	// Determine the call index. Chat uses an integer index in streaming.
	idx := 0
	if tc.Index != nil {
		idx = *tc.Index
	}
	call, exists := d.toolCalls[idx]
	if !exists {
		call = &openCall{partIndex: -1}
		d.toolCalls[idx] = call
	}

	if !call.started {
		// Emit tool_start once we have either a name or any arguments.
		call.started = true
		out = append(out, ir.Event{
			Type:      ir.EvToolStart,
			Index:     ci,
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Namespace: "",
			Kind:      ir.ToolFunction,
		})
	}

	// Accumulate name fragments (chat may send the name across multiple chunks,
	// or re-send it whole after a prefix).
	if tc.Function.Name != "" {
		call.nameBuilt.WriteString(tc.Function.Name)
	}

	// Accumulate argument fragments.
	if tc.Function.Arguments != "" {
		call.argsBuilt.WriteString(tc.Function.Arguments)
		out = append(out, ir.Event{
			Type:  ir.EvToolArgs,
			Index: ci,
			Text:  tc.Function.Arguments,
		})
	}

	return out
}

func (d *StreamDecoder) finalizeChoice(out []ir.Event, ci int, finish string) []ir.Event {
	// Close all open tool calls for this choice.
	for idx, call := range d.toolCalls {
		if !call.started {
			continue
		}
		args := call.argsBuilt.String()
		out = append(out, ir.Event{
			Type:      ir.EvToolEnd,
			Index:     ci,
			Arguments: json.RawMessage(args),
		})
		// Replace the partial name with the fully accumulated one if present.
		_ = idx
	}

	stop := stopFromFinish(finish)
	out = append(out, ir.Event{Type: ir.EvStop, Index: ci, Stop: &stop})

	// Clear call state for this choice so the next choice can start fresh.
	d.toolCalls = map[int]*openCall{}
	return out
}

// stopFromFinish maps a Chat finish_reason to an ir.Stop.
func stopFromFinish(finish string) ir.Stop {
	switch finish {
	case "stop":
		return ir.Stop{Reason: ir.StopEndTurn, Raw: finish}
	case "length":
		return ir.Stop{Reason: ir.StopMaxTokens, Raw: finish}
	case "tool_calls":
		return ir.Stop{Reason: ir.StopToolUse, Raw: finish}
	case "content_filter":
		return ir.Stop{Reason: ir.StopContentFilter, Raw: finish}
	}
	return ir.Stop{Reason: ir.StopEndTurn, Raw: finish}
}

// End finalises the stream. If the upstream ended without a terminal event, an
// error event is appended (unless we already saw one).
func (d *StreamDecoder) End(how ir.EndKind) *ir.Event {
	if d.terminal {
		return nil
	}
	return ir.EndErrorEvent(how)
}

// stringOrEmpty extracts a string from a JSON value that may be string or null.
func stringOrEmpty(raw json.RawMessage) string {
	if len(raw) == 0 || jsonx.IsNull(raw) {
		return ""
	}
	s, _ := jsonx.String(raw)
	return s
}

// rawToString decodes a JSON string or null json.RawMessage to a Go string.
// A non-string value yields "".
func rawToString(raw json.RawMessage) string {
	if len(raw) == 0 || jsonx.IsNull(raw) {
		return ""
	}
	s, _ := jsonx.String(raw)
	return s
}

// decodeUsage decodes a wire usage object into ir.Usage using the dialect.
func decodeUsage(u *wireUsage, dia Dialect) ir.Usage {
	out := ir.Usage{}
	out.InputTokens = u.PromptTokens
	out.OutputTokens = u.CompletionTokens
	out.Reported |= ir.UInput | ir.UOutput

	// Cache read.
	if u.PromptTokensDetails != nil {
		out.CacheReadTokens = u.PromptTokensDetails.CachedTokens
		out.Reported |= ir.UCacheRead
	}
	// Reasoning tokens.
	if u.CompletionTokensDetails != nil {
		out.ReasoningTokens = u.CompletionTokensDetails.ReasoningTokens
		out.Reported |= ir.UReasoning
	}

	// Prompt-token adjustment: by default InputTokens = prompt_tokens, but if the
	// dialect says the prompt count already excludes cache, we add cache back in
	// (so InputTokens reflects the total prompt input). The IR.PromptTokens()
	// helper does InputTokens + CacheRead + CacheWrite; if the dialect includes
	// cache in prompt_tokens (the common case), we must subtract to avoid
	// double-counting.
	if !dia.PromptExcludesCache {
		// prompt_tokens includes cached tokens; subtract them so InputTokens
		// reflects non-cached input only.
		if out.InputTokens > out.CacheReadTokens+out.CacheWriteTokens {
			out.InputTokens -= out.CacheReadTokens + out.CacheWriteTokens
		} else if out.CacheReadTokens+out.CacheWriteTokens > 0 {
			// Edge: cache > prompt (shouldn't happen, but be defensive).
			out.InputTokens = 0
		}
	}
	// Reasoning exclusion from completion tokens.
	if dia.ReasoningExcludedFromCompletion && out.ReasoningTokens > 0 {
		out.OutputTokens += out.ReasoningTokens
	}

	return out
}

// ParseResponse decodes a non-streaming Chat Completions response body.
func ParseResponse(body []byte, opts DecodeOptions) (*ir.Response, error) {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	env.Style = ir.StyleBracket

	var wr wireResponse
	if json.Unmarshal(body, &wr) != nil {
		// Maybe an error body.
		if e := ParseError(0, body); e != nil && e.Message != "" {
			return nil, e
		}
		return nil, ir.Protocol("chat: response body is not a valid object")
	}

	resp := &ir.Response{
		ID:      wr.ID,
		Model:   wr.Model,
		Created: wr.Created,
	}
	for _, ch := range wr.Choices {
		out := ir.Output{Index: ch.Index}
		// Message -> parts.
		parts, err := parseAssistantMessage(ch.Message, ir.Root.Key("choices").At(ch.Index).Key("message"), env)
		if err != nil {
			return nil, err
		}
		out.Parts = parts
		out.Stop = stopFromFinish(ch.FinishReason)
		out.Stop = ir.FinalStop(out.Stop, out.Parts)
		resp.Outputs = append(resp.Outputs, out)
	}
	if wr.Usage != nil {
		resp.Usage = decodeUsage(wr.Usage, opts.Dialect)
	}
	if len(resp.Outputs) == 0 {
		resp.Outputs = []ir.Output{{Stop: ir.FinalStop(ir.Stop{}, nil)}}
	}
	return resp, nil
}

// parseAssistantMessage converts a non-streaming assistant wireMessage to parts.
func parseAssistantMessage(wm wireMessage, p ir.Path, env *ir.Env) ([]ir.Part, error) {
	var parts []ir.Part
	// Reasoning.
	if wm.ReasoningContent != "" {
		parts = append(parts, ir.Part{
			Type:      ir.PartReasoning,
			Reasoning: &ir.Reasoning{Text: wm.ReasoningContent},
		})
	} else if wm.Reasoning != "" {
		parts = append(parts, ir.Part{
			Type:      ir.PartReasoning,
			Reasoning: &ir.Reasoning{Text: wm.Reasoning},
		})
	}
	// Content.
	switch jsonx.Kind(wm.Content) {
	case '"':
		s, _ := jsonx.String(wm.Content)
		if s != "" {
			parts = append(parts, ir.TextPart(s))
		}
	case '[':
		elems, ok := jsonx.Array(wm.Content)
		if !ok {
			return nil, env.Err(ir.Invalid(p.Key("content"), "must be a string or array"))
		}
		for i, elem := range elems {
			part, err := parseContentPart(elem, p.Key("content").At(i), env)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		}
	}
	// Refusal.
	if wm.Refusal != "" {
		parts = append(parts, ir.Part{Type: ir.PartRefusal, Text: wm.Refusal})
	}
	// Tool calls.
	for i, tc := range wm.ToolCalls {
		fp := p.Key("tool_calls").At(i)
		part, err := wireToolCallToPart(tc, fp, env)
		if err != nil {
			return nil, err
		}
		parts = append(parts, part)
	}
	return parts, nil
}
