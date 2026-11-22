package chat

import (
	"encoding/json"
	"sort"
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

	// Names we've seen, for dedup of call IDs within one response.
	seenCallIDs map[string]bool

	done      bool
	sawFinish bool
	terminal  bool
}

// openCall tracks one tool call in flight within a candidate.
type openCall struct {
	id        string
	name      string
	namespace string
	kind      ir.ToolKind
	// nameBuilt accumulates the streamed name fragments.
	nameBuilt strings.Builder
	// argsBuilt accumulates the streamed arguments fragments.
	argsBuilt strings.Builder
	// started is whether EvToolStart has been emitted for this call.
	started bool
}

// chatCandidate is per-choice state in the stream.
type chatCandidate struct {
	index     int
	toolCalls map[int]*openCall
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
		if !jsonx.IsNull(chunk.Usage) {
			if u, ok := decodeUsageRaw(chunk.Usage, d.dia); ok {
				usage = &u
			}
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
			d.candidates[ci] = &chatCandidate{index: ci, toolCalls: map[int]*openCall{}}
		}
		out = d.decodeChoice(out, ci, ch)
	}

	// Usage on a later chunk (the final usage chunk has choices: []).
	if !jsonx.IsNull(chunk.Usage) {
		if u, ok := decodeUsageRaw(chunk.Usage, d.dia); ok {
			out = append(out, ir.Event{Type: ir.EvUsage, Usage: &u})
		}
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
	cand := d.candidates[ci]
	if cand == nil {
		cand = &chatCandidate{index: ci, toolCalls: map[int]*openCall{}}
		d.candidates[ci] = cand
	}
	// Determine the call index. Chat uses an integer index in streaming.
	idx := 0
	if tc.Index != nil {
		idx = *tc.Index
	}
	call, exists := cand.toolCalls[idx]
	if !exists {
		call = &openCall{kind: ir.ToolFunction}
		cand.toolCalls[idx] = call
	}

	if tc.ID != "" && call.id == "" {
		call.id = tc.ID
	}
	if tc.Function.Name != "" {
		call.nameBuilt.WriteString(tc.Function.Name)
	}

	if !call.started {
		call.started = true
		if call.id == "" {
			call.id = d.env.NewID("call_", 16)
		}
		name := call.nameBuilt.String()
		ns := ""
		kind := ir.ToolFunction
		if ref, ok := d.names[name]; ok {
			name = ref.Name
			ns = ref.Namespace
			if ref.Kind != "" {
				kind = ref.Kind
			}
		}
		call.name = name
		call.namespace = ns
		call.kind = kind
		out = append(out, ir.Event{
			Type:      ir.EvToolStart,
			Index:     ci,
			Part:      idx,
			ID:        call.id,
			Name:      name,
			Namespace: ns,
			Kind:      kind,
		})
	}

	// Accumulate argument fragments.
	if tc.Function.Arguments != "" {
		call.argsBuilt.WriteString(tc.Function.Arguments)
		out = append(out, ir.Event{
			Type:  ir.EvToolArgs,
			Index: ci,
			Part:  idx,
			ID:    call.id,
			Text:  tc.Function.Arguments,
		})
	}

	return out
}

func (d *StreamDecoder) finalizeChoice(out []ir.Event, ci int, finish string) []ir.Event {
	cand := d.candidates[ci]
	if cand != nil && len(cand.toolCalls) > 0 {
		idxs := make([]int, 0, len(cand.toolCalls))
		for idx := range cand.toolCalls {
			idxs = append(idxs, idx)
		}
		sort.Ints(idxs)
		for _, idx := range idxs {
			call := cand.toolCalls[idx]
			if !call.started {
				continue
			}
			argsStr := strings.TrimSpace(call.argsBuilt.String())
			var argsRaw json.RawMessage
			if call.kind == ir.ToolCustom {
				var m map[string]string
				if json.Unmarshal([]byte(argsStr), &m) == nil {
					argsRaw, _ = jsonx.Marshal(m["input"])
				} else if argsStr != "" {
					argsRaw, _ = jsonx.Marshal(argsStr)
				}
			} else {
				if argsStr == "" {
					argsStr = "{}"
				}
				argsRaw = json.RawMessage(argsStr)
			}
			out = append(out, ir.Event{
				Type:      ir.EvToolEnd,
				Index:     ci,
				Part:      idx,
				ID:        call.id,
				Arguments: argsRaw,
			})
		}
		cand.toolCalls = map[int]*openCall{}
	}

	stop := stopFromFinish(finish)
	out = append(out, ir.Event{Type: ir.EvStop, Index: ci, Stop: &stop})
	d.sawFinish = true
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

// End finalises the stream. A stream is complete only if a valid finish_reason
// or terminal error was received; [DONE] without a finish_reason still reports
// truncation on EOF/timeout/error.
func (d *StreamDecoder) End(how ir.EndKind) *ir.Event {
	if d.terminal || d.sawFinish {
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

func decodeUsageRaw(raw json.RawMessage, dia Dialect) (ir.Usage, bool) {
	if jsonx.IsNull(raw) {
		return ir.Usage{}, false
	}
	var u wireUsage
	if json.Unmarshal(raw, &u) != nil {
		return ir.Usage{}, false
	}
	return decodeUsage(&u, raw, dia), true
}

// decodeUsage decodes a wire usage object into ir.Usage using the dialect.
func decodeUsage(u *wireUsage, raw json.RawMessage, dia Dialect) ir.Usage {
	out := ir.Usage{
		OutputTokens: u.CompletionTokens,
		Reported:     ir.UOutput,
	}

	// Cache read: standard prompt_tokens_details.cached_tokens, else dialect fallbacks.
	if u.PromptTokensDetails != nil {
		out.CacheReadTokens = u.PromptTokensDetails.CachedTokens
		out.Reported |= ir.UCacheRead
	} else if len(dia.CacheReadFields) > 0 && len(raw) > 0 {
		for _, field := range dia.CacheReadFields {
			if n, ok := lookupIntPath(raw, field); ok {
				out.CacheReadTokens = n
				out.Reported |= ir.UCacheRead
				break
			}
		}
	}

	// Cache write: dialect-configured field paths.
	if len(dia.CacheWriteFields) > 0 && len(raw) > 0 {
		for _, field := range dia.CacheWriteFields {
			if n, ok := lookupIntPath(raw, field); ok {
				out.CacheWriteTokens = n
				out.Reported |= ir.UCacheWrite
				break
			}
		}
	}

	// Reasoning tokens.
	if u.CompletionTokensDetails != nil {
		out.ReasoningTokens = u.CompletionTokensDetails.ReasoningTokens
		out.Reported |= ir.UReasoning
	}

	if dia.PromptExcludesCache {
		out.InputTokens = u.PromptTokens
		out.Reported |= ir.UInput
	} else {
		out.FromInclusivePrompt(u.PromptTokens)
	}
	if dia.ReasoningExcludedFromCompletion && out.ReasoningTokens > 0 {
		out.OutputTokens += out.ReasoningTokens
	}
	return out
}

func lookupIntPath(raw json.RawMessage, dotPath string) (int, bool) {
	cur := raw
	for _, part := range strings.Split(dotPath, ".") {
		kvs, err := jsonx.Object(cur)
		if err != nil {
			return 0, false
		}
		found := false
		for _, kv := range kvs {
			if kv.Key == part {
				cur = kv.Value
				found = true
				break
			}
		}
		if !found {
			return 0, false
		}
	}
	n, ok := jsonx.Int(cur)
	return int(n), ok
}

// ParseResponse decodes a non-streaming Chat Completions response body.
func ParseResponse(body []byte, opts DecodeOptions) (*ir.Response, error) {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	env.Style = ir.StyleBracket

	// Check for an error envelope before decoding as a completion response.
	if jsonx.Kind(body) == '{' {
		var errEnv struct {
			Error *wireError `json:"error"`
		}
		if json.Unmarshal(body, &errEnv) == nil && errEnv.Error != nil && errEnv.Error.Message != "" {
			return nil, ParseError(0, body)
		}
	}

	var wr wireResponse
	if json.Unmarshal(body, &wr) != nil {
		if e := ParseError(0, body); e != nil && e.Message != "" {
			return nil, e
		}
		return nil, ir.Protocol("chat: response body is not a valid object")
	}
	if len(wr.Choices) == 0 {
		return nil, ir.Protocol("chat: response has no choices")
	}

	resp := &ir.Response{
		ID:      wr.ID,
		Model:   wr.Model,
		Created: wr.Created,
	}
	for _, ch := range wr.Choices {
		out := ir.Output{Index: ch.Index}
		// Message -> parts.
		parts, err := parseAssistantMessage(ch.Message, ir.Root.Key("choices").At(ch.Index).Key("message"), opts.Names, env)
		if err != nil {
			return nil, err
		}
		out.Parts = parts
		out.Stop = stopFromFinish(ch.FinishReason)
		out.Stop = ir.FinalStop(out.Stop, out.Parts)
		resp.Outputs = append(resp.Outputs, out)
	}
	if !jsonx.IsNull(wr.Usage) {
		if u, ok := decodeUsageRaw(wr.Usage, opts.Dialect); ok {
			resp.Usage = u
		}
	}
	if len(resp.Outputs) == 0 {
		resp.Outputs = []ir.Output{{Stop: ir.FinalStop(ir.Stop{}, nil)}}
	}
	return resp, nil
}

// parseAssistantMessage converts a non-streaming assistant wireMessage to parts.
func parseAssistantMessage(wm wireMessage, p ir.Path, names map[string]ir.ToolRef, env *ir.Env) ([]ir.Part, error) {
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
		if part.Call != nil && names != nil {
			if ref, ok := names[part.Call.Name]; ok {
				part.Call.Name = ref.Name
				part.Call.Namespace = ref.Namespace
				if ref.Kind != "" {
					part.Call.Kind = ref.Kind
				}
				if part.Call.Kind == ir.ToolCustom && part.Call.Input == "" && len(part.Call.Arguments) > 0 {
					var m map[string]string
					if json.Unmarshal(part.Call.Arguments, &m) == nil {
						part.Call.Input = m["input"]
					}
				}
			}
		}
		parts = append(parts, part)
	}
	return parts, nil
}
