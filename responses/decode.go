package responses

import (
	"encoding/json"
	"strings"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// StreamDecoder converts Responses API SSE chunks into ir.Events.
type StreamDecoder struct {
	env          *ir.Env
	dia          Dialect
	names        map[string]ir.ToolRef
	started      bool
	terminal     bool
	hasToolCalls bool

	// per-item state keyed by output_index
	itemType    map[int]string // "message" | "function_call" | "custom_tool_call" | "reasoning"
	callID      map[int]string
	callName    map[int]string
	callKind    map[int]ir.ToolKind
	callArgs    map[int]*strings.Builder
	toolStarted map[int]bool
}

// NewStreamDecoder returns a Responses stream decoder.
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
		itemType:    map[int]string{},
		callID:      map[int]string{},
		callName:    map[int]string{},
		callKind:    map[int]ir.ToolKind{},
		callArgs:    map[int]*strings.Builder{},
		toolStarted: map[int]bool{},
	}
}

// Decode converts one SSE event into zero or more IR events.
func (d *StreamDecoder) Decode(out []ir.Event, ev sse.Event) ([]ir.Event, error) {
	if d.terminal {
		return out, nil
	}
	if sse.IsDone(ev.Data) {
		return out, nil
	}
	if len(ev.Data) == 0 {
		return out, nil
	}
	if jsonx.Kind(ev.Data) != '{' {
		return out, nil
	}

	var se wireStreamEvent
	if err := json.Unmarshal(ev.Data, &se); err != nil {
		return out, nil
	}

	switch se.Type {
	case "response.created", "response.in_progress":
		if !d.started {
			d.started = true
			var id, model string
			if se.Response != nil {
				id = se.Response.ID
				model = se.Response.Model
			}
			out = append(out, ir.Event{Type: ir.EvStart, ID: id, Model: model})
		}

	case "response.completed", "response.done":
		stop := ir.Stop{Reason: ir.StopEndTurn, Raw: "completed"}
		if d.hasToolCalls {
			stop.Reason = ir.StopToolUse
		}
		out = append(out, ir.Event{Type: ir.EvStop, Stop: &stop})
		if se.Response != nil && !jsonx.IsNull(se.Response.Usage) {
			u := decodeResponsesUsage(se.Response.Usage, d.dia)
			out = append(out, ir.Event{Type: ir.EvUsage, Usage: &u})
		}
		d.terminal = true

	case "response.incomplete":
		stop := ir.Stop{Reason: ir.StopMaxTokens, Raw: "incomplete"}
		if se.Response != nil {
			if se.Response.IncompleteDetails != nil && se.Response.IncompleteDetails.Reason == "content_filter" {
				stop.Reason = ir.StopContentFilter
				stop.Raw = "content_filter"
			} else if se.Response.Error != nil {
				stop.Reason = ir.StopContentFilter
			}
		}
		out = append(out, ir.Event{Type: ir.EvStop, Stop: &stop})
		if se.Response != nil && !jsonx.IsNull(se.Response.Usage) {
			u := decodeResponsesUsage(se.Response.Usage, d.dia)
			out = append(out, ir.Event{Type: ir.EvUsage, Usage: &u})
		}
		d.terminal = true

	case "response.failed":
		var apiErr *ir.APIError
		if se.Response != nil && se.Response.Error != nil {
			apiErr = &ir.APIError{
				Code:    se.Response.Error.Code,
				Message: se.Response.Error.Message,
				Type:    refineType(mapErrorType(se.Response.Error.Type), se.Response.Error.Code, se.Response.Error.Message),
				Raw:     append([]byte(nil), ev.Data...),
			}
		} else if se.Error != nil {
			apiErr = &ir.APIError{
				Code:    se.Error.Code,
				Message: se.Error.Message,
				Type:    refineType(mapErrorType(se.Error.Type), se.Error.Code, se.Error.Message),
				Raw:     append([]byte(nil), ev.Data...),
			}
		} else {
			apiErr = &ir.APIError{Type: ir.ErrServer, Message: "response failed", Raw: append([]byte(nil), ev.Data...)}
		}
		out = append(out, ir.Event{Type: ir.EvError, Err: apiErr})
		d.terminal = true

	case "error":
		var apiErr *ir.APIError
		if se.Error != nil {
			apiErr = &ir.APIError{
				Code:    se.Error.Code,
				Message: se.Error.Message,
				Param:   se.Error.Param,
				Type:    refineType(mapErrorType(se.Error.Type), se.Error.Code, se.Error.Message),
				Raw:     append([]byte(nil), ev.Data...),
			}
		} else if se.Message != "" || se.Code != "" {
			apiErr = &ir.APIError{
				Code:    se.Code,
				Message: se.Message,
				Param:   se.Param,
				Type:    refineType(ir.ErrServer, se.Code, se.Message),
				Raw:     append([]byte(nil), ev.Data...),
			}
		}
		if apiErr != nil {
			out = append(out, ir.Event{Type: ir.EvError, Err: apiErr})
			d.terminal = true
		}

	case "response.output_item.added":
		if se.Item != nil {
			idx := se.OutputIndex
			d.itemType[idx] = se.Item.Type
			switch se.Item.Type {
			case "function_call":
				id := se.Item.CallID
				if id == "" {
					id = se.Item.ID
				}
				d.callID[idx] = id
				d.callName[idx] = se.Item.Name
				d.callKind[idx] = ir.ToolFunction
				d.callArgs[idx] = &strings.Builder{}
				d.toolStarted[idx] = true
				d.hasToolCalls = true
				out = d.emitToolStart(out, idx, id, se.Item.Name, "", ir.ToolFunction)
			case "custom_tool_call":
				id := se.Item.CallID
				if id == "" {
					id = se.Item.ID
				}
				d.callID[idx] = id
				d.callName[idx] = se.Item.Name
				d.callKind[idx] = ir.ToolCustom
				d.callArgs[idx] = &strings.Builder{}
				d.toolStarted[idx] = true
				d.hasToolCalls = true
				out = d.emitToolStart(out, idx, id, se.Item.Name, se.Item.Namespace, ir.ToolCustom)
			case "reasoning":
				out = append(out, ir.Event{Type: ir.EvReasoningStart, ID: se.Item.ID})
			}
		}

	case "response.output_item.done":
		if se.Item != nil {
			idx := se.OutputIndex
			switch se.Item.Type {
			case "message":
				// Individual message item done does not end the whole response;
				// response.completed / response.incomplete emits EvStop.
			case "function_call":
				id := d.callID[idx]
				if id == "" {
					id = se.Item.CallID
					if id == "" {
						id = se.Item.ID
					}
					d.callID[idx] = id
				}
				if !d.toolStarted[idx] {
					d.toolStarted[idx] = true
					d.hasToolCalls = true
					out = d.emitToolStart(out, idx, id, se.Item.Name, "", ir.ToolFunction)
				}
				argsStr := strings.TrimSpace(se.Item.Arguments)
				if argsStr == "" && d.callArgs[idx] != nil {
					argsStr = strings.TrimSpace(d.callArgs[idx].String())
				}
				if argsStr == "" {
					argsStr = "{}"
				}
				out = append(out, ir.Event{Type: ir.EvToolEnd, Part: idx, ID: id, Arguments: json.RawMessage(argsStr)})
				delete(d.callArgs, idx)
			case "custom_tool_call":
				id := d.callID[idx]
				if id == "" {
					id = se.Item.CallID
					if id == "" {
						id = se.Item.ID
					}
					d.callID[idx] = id
				}
				if !d.toolStarted[idx] {
					d.toolStarted[idx] = true
					d.hasToolCalls = true
					out = d.emitToolStart(out, idx, id, se.Item.Name, se.Item.Namespace, ir.ToolCustom)
				}
				var raw json.RawMessage
				inp := se.Item.Input
				if inp == "" && d.callArgs[idx] != nil {
					inp = d.callArgs[idx].String()
				}
				if inp != "" {
					raw, _ = jsonx.Marshal(inp)
				}
				out = append(out, ir.Event{Type: ir.EvToolEnd, Part: idx, ID: id, Arguments: raw})
				delete(d.callArgs, idx)
			case "reasoning":
				if se.Item.EncryptedContent != "" {
					data, _ := jsonx.Marshal(se.Item.EncryptedContent)
					out = append(out, ir.Event{
						Type: ir.EvSealed,
						Sealed: &ir.Sealed{
							Kind:   ir.SealedResponsesEncrypted,
							Data:   data,
							Origin: ir.Origin{Format: ir.FormatResponses, Provider: d.env.ProviderID},
						},
					})
				}
			}
		}

	case "response.content_part.added", "response.content_part.done", "response.output_text.done":
		// No action; text arrives via response.output_text.delta.

	case "response.output_text.delta":
		out = append(out, ir.Event{Type: ir.EvText, Text: se.Delta})

	case "response.output_text.annotation.added":
		if se.Annotation != nil {
			out = append(out, ir.Event{
				Type: ir.EvCitation,
				Citation: &ir.Citation{
					Type:       se.Annotation.Type,
					URL:        se.Annotation.URL,
					Title:      se.Annotation.Title,
					StartIndex: se.Annotation.StartIndex,
					EndIndex:   se.Annotation.EndIndex,
				},
			})
		}

	case "response.refusal.delta":
		out = append(out, ir.Event{Type: ir.EvRefusal, Text: se.Delta})

	case "response.function_call_arguments.delta":
		idx := se.OutputIndex
		if !d.started {
			d.started = true
			out = append(out, ir.Event{Type: ir.EvStart})
		}
		if !d.toolStarted[idx] {
			d.toolStarted[idx] = true
			d.hasToolCalls = true
			out = d.emitToolStart(out, idx, d.callID[idx], d.callName[idx], "", ir.ToolFunction)
		}
		if se.Delta != "" {
			if d.callArgs[idx] == nil {
				d.callArgs[idx] = &strings.Builder{}
			}
			d.callArgs[idx].WriteString(se.Delta)
			out = append(out, ir.Event{Type: ir.EvToolArgs, Part: idx, ID: d.callID[idx], Text: se.Delta})
		}

	case "response.function_call_arguments.done":
		// Final arguments; EvToolEnd emitted by response.output_item.done.

	case "response.custom_tool_call_input.delta":
		idx := se.OutputIndex
		if !d.toolStarted[idx] {
			d.toolStarted[idx] = true
			d.hasToolCalls = true
			out = d.emitToolStart(out, idx, d.callID[idx], d.callName[idx], "", ir.ToolCustom)
		}
		if se.Delta != "" {
			if d.callArgs[idx] == nil {
				d.callArgs[idx] = &strings.Builder{}
			}
			d.callArgs[idx].WriteString(se.Delta)
			out = append(out, ir.Event{Type: ir.EvToolArgs, Part: idx, ID: d.callID[idx], Text: se.Delta})
		}

	case "response.custom_tool_call_input.done":
		// Final input; EvToolEnd emitted by response.output_item.done.

	case "response.reasoning_summary_text.delta":
		out = append(out, ir.Event{Type: ir.EvReasoning, Text: se.Delta, Summary: true})

	case "response.reasoning_text.delta":
		out = append(out, ir.Event{Type: ir.EvReasoning, Text: se.Delta})

	case "response.reasoning_summary_text.done", "response.reasoning_text.done":
		// No extra event.
	}

	return out, nil
}

func (d *StreamDecoder) emitToolStart(out []ir.Event, partIdx int, id, name, ns string, kind ir.ToolKind) []ir.Event {
	if ref, ok := d.names[name]; ok {
		name = ref.Name
		if ns == "" {
			ns = ref.Namespace
		}
		if ref.Kind != "" {
			kind = ref.Kind
		}
	}
	return append(out, ir.Event{
		Type:      ir.EvToolStart,
		Part:      partIdx,
		ID:        id,
		Name:      name,
		Namespace: ns,
		Kind:      kind,
	})
}

// End signals how the upstream stream ended.
func (d *StreamDecoder) End(how ir.EndKind) *ir.Event {
	if d.terminal {
		return nil
	}
	return ir.EndErrorEvent(how)
}

// ParseResponse decodes a non-streaming Responses API response body.
func ParseResponse(body []byte, opts DecodeOptions) (*ir.Response, error) {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	env.Style = ir.StyleBracket

	var check struct {
		Error *wireError `json:"error"`
	}
	if json.Unmarshal(body, &check) == nil && check.Error != nil && check.Error.Message != "" {
		return nil, ParseError(0, body)
	}

	var wr wireResponse
	if err := json.Unmarshal(body, &wr); err != nil {
		return nil, ir.Protocol("failed to decode Responses response: " + err.Error())
	}

	if wr.Status == "failed" {
		if wr.Error != nil {
			return nil, &ir.APIError{
				Code:    wr.Error.Code,
				Message: wr.Error.Message,
				Type:    refineType(mapErrorType(wr.Error.Type), wr.Error.Code, wr.Error.Message),
				Raw:     append([]byte(nil), body...),
			}
		}
		return nil, &ir.APIError{Type: ir.ErrServer, Message: "response failed", Raw: append([]byte(nil), body...)}
	}
	if (wr.Object != "" && wr.Object != "response") || (wr.Object == "" && wr.Status == "" && len(wr.Output) == 0) {
		return nil, ir.Protocol("responses: response missing required fields")
	}

	resp := &ir.Response{
		ID:    wr.ID,
		Model: wr.Model,
	}
	if !jsonx.IsNull(wr.Usage) {
		resp.Usage = decodeResponsesUsage(wr.Usage, opts.Dialect)
	}

	out := ir.Output{}
	for i, item := range wr.Output {
		fp := ir.Root.Key("output").At(i)
		parts, _, err := outputItemToIR(item, fp, opts.Names, env)
		if err != nil {
			return nil, err
		}
		out.Parts = append(out.Parts, parts...)
	}

	switch wr.Status {
	case "completed", "":
		out.Stop = ir.Stop{Reason: ir.StopEndTurn, Raw: wr.Status}
	case "incomplete":
		if wr.IncompleteDetails != nil && wr.IncompleteDetails.Reason == "content_filter" {
			out.Stop = ir.Stop{Reason: ir.StopContentFilter, Raw: "content_filter"}
		} else {
			out.Stop = ir.Stop{Reason: ir.StopMaxTokens, Raw: wr.Status}
		}
	}
	out.Stop = ir.FinalStop(out.Stop, out.Parts)

	resp.Outputs = []ir.Output{out}
	return resp, nil
}

func outputItemToIR(item wireOutputItem, p ir.Path, names map[string]ir.ToolRef, env *ir.Env) ([]ir.Part, ir.Stop, error) {
	switch item.Type {
	case "message":
		var parts []ir.Part
		for i, cp := range item.Content {
			fp := p.Key("content").At(i)
			part, err := outputPartToIR(cp, fp, env)
			if err != nil {
				return nil, ir.Stop{}, err
			}
			parts = append(parts, part)
		}
		return parts, statusToStop(item.Status), nil

	case "function_call":
		args := json.RawMessage(item.Arguments)
		if len(args) == 0 {
			args = json.RawMessage(`{}`)
		}
		name := item.Name
		ns := ""
		kind := ir.ToolFunction
		if ref, ok := names[name]; ok {
			name = ref.Name
			ns = ref.Namespace
			if ref.Kind != "" {
				kind = ref.Kind
			}
		}
		id := item.CallID
		if id == "" {
			id = item.ID
		}
		part := ir.Part{
			Type: ir.PartToolCall,
			Call: &ir.ToolCall{
				ID:        id,
				Name:      name,
				Namespace: ns,
				Kind:      kind,
				Arguments: args,
			},
		}
		return []ir.Part{part}, ir.Stop{}, nil

	case "custom_tool_call":
		id := item.CallID
		if id == "" {
			id = item.ID
		}
		part := ir.Part{
			Type: ir.PartToolCall,
			Call: &ir.ToolCall{
				ID:        id,
				Name:      item.Name,
				Namespace: item.Namespace,
				Kind:      ir.ToolCustom,
				Input:     item.Input,
			},
		}
		return []ir.Part{part}, ir.Stop{}, nil

	case "reasoning":
		r := &ir.Reasoning{ID: item.ID}
		if item.EncryptedContent != "" {
			data, _ := jsonx.Marshal(item.EncryptedContent)
			r.Sealed = &ir.Sealed{
				Kind:   ir.SealedResponsesEncrypted,
				Data:   data,
				Origin: ir.Origin{Format: ir.FormatResponses, Provider: env.ProviderID},
			}
		}
		for _, sp := range item.Summary {
			r.Summary = append(r.Summary, sp.Text)
			r.Text += sp.Text
		}
		return []ir.Part{{Type: ir.PartReasoning, Reasoning: r}}, ir.Stop{}, nil

	case "web_search_call":
		return []ir.Part{{
			Type: ir.PartBuiltinCall,
			Builtin: &ir.BuiltinActivity{
				Kind:   ir.BuiltinWebSearch,
				ID:     item.ID,
				Origin: ir.FormatResponses,
			},
		}}, ir.Stop{}, nil
	}

	return nil, ir.Stop{}, nil
}

func outputPartToIR(cp wireOutputPart, p ir.Path, env *ir.Env) (ir.Part, error) {
	switch cp.Type {
	case "output_text", "text":
		pt := ir.TextPart(cp.Text)
		for _, a := range cp.Annotations {
			pt.Citations = append(pt.Citations, ir.Citation{
				Type:       a.Type,
				URL:        a.URL,
				Title:      a.Title,
				StartIndex: a.StartIndex,
				EndIndex:   a.EndIndex,
			})
		}
		return pt, nil
	case "refusal":
		return ir.Part{Type: ir.PartRefusal, Text: cp.Refusal}, nil
	}
	return ir.TextPart(""), nil
}

func statusToStop(status string) ir.Stop {
	switch status {
	case "completed":
		return ir.Stop{Reason: ir.StopEndTurn, Raw: status}
	case "incomplete":
		return ir.Stop{Reason: ir.StopMaxTokens, Raw: status}
	}
	return ir.Stop{}
}

func decodeResponsesUsage(raw json.RawMessage, dia Dialect) ir.Usage {
	var u wireUsage
	if err := json.Unmarshal(raw, &u); err != nil {
		return ir.Usage{}
	}
	out := ir.Usage{
		OutputTokens: u.OutputTokens,
		Reported:     ir.UOutput,
	}
	if u.InputTokensDetails != nil {
		if u.InputTokensDetails.CachedTokens > 0 {
			out.CacheReadTokens = u.InputTokensDetails.CachedTokens
			out.Reported |= ir.UCacheRead
		}
		if u.InputTokensDetails.CacheWriteTokens > 0 {
			out.CacheWriteTokens = u.InputTokensDetails.CacheWriteTokens
			out.Reported |= ir.UCacheWrite
		}
	}
	if !out.Has(ir.UCacheWrite) && dia.CacheWriteField != "" {
		var m map[string]any
		if json.Unmarshal(raw, &m) == nil {
			if v, ok := lookupIntField(m, "input_tokens_details."+dia.CacheWriteField); ok {
				out.CacheWriteTokens = v
				out.Reported |= ir.UCacheWrite
			} else if v, ok := lookupIntField(m, dia.CacheWriteField); ok {
				out.CacheWriteTokens = v
				out.Reported |= ir.UCacheWrite
			}
		}
	}
	out.FromInclusivePrompt(u.InputTokens)
	if u.OutputTokensDetails != nil && u.OutputTokensDetails.ReasoningTokens > 0 {
		out.ReasoningTokens = u.OutputTokensDetails.ReasoningTokens
		out.Reported |= ir.UReasoning
	}
	return out
}

func lookupIntField(m map[string]any, dotted string) (int, bool) {
	cur := any(m)
	start := 0
	for i := 0; i <= len(dotted); i++ {
		if i < len(dotted) && dotted[i] != '.' {
			continue
		}
		key := dotted[start:i]
		start = i + 1
		obj, ok := cur.(map[string]any)
		if !ok {
			return 0, false
		}
		cur, ok = obj[key]
		if !ok {
			return 0, false
		}
	}
	switch v := cur.(type) {
	case float64:
		return int(v), true
	case int:
		return v, true
	}
	return 0, false
}
