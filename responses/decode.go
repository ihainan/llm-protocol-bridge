package responses

import (
	"encoding/json"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// StreamDecoder converts Responses API SSE chunks into ir.Events.
type StreamDecoder struct {
	env      *ir.Env
	dia      Dialect
	names    map[string]ir.ToolRef
	started  bool
	terminal bool
	seqNext  int

	// per-item state keyed by output_index
	itemType map[int]string // "message" | "function_call" | "reasoning"
	callID   map[int]string
	callName map[int]string
}

// NewStreamDecoder returns a Responses stream decoder.
func NewStreamDecoder(opts DecodeOptions) *StreamDecoder {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	env.Style = ir.StyleBracket
	return &StreamDecoder{
		env:      env,
		dia:      opts.Dialect,
		names:    opts.Names,
		itemType: map[int]string{},
		callID:   map[int]string{},
		callName: map[int]string{},
	}
}

// Decode converts one SSE event into zero or more IR events.
func (d *StreamDecoder) Decode(out []ir.Event, ev sse.Event) ([]ir.Event, error) {
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
			var model string
			if se.Response != nil {
				model = se.Response.Model
			}
			startEv := ir.Event{Type: ir.EvStart, Model: model}
			out = append(out, startEv)
		}

	case "response.completed", "response.done":
		if se.Response != nil && se.Response.Usage != nil {
			u := decodeResponsesUsage(se.Response.Usage, d.dia)
			out = append(out, ir.Event{Type: ir.EvUsage, Usage: &u})
		}
		// Stop is handled by output_item.done events.

	case "response.incomplete":
		stop := ir.Stop{Reason: ir.StopMaxTokens, Raw: "incomplete"}
		if se.Response != nil {
			if se.Response.Error != nil {
				stop.Reason = ir.StopContentFilter
			}
		}
		out = append(out, ir.Event{Type: ir.EvStop, Stop: &stop})
		d.terminal = true

	case "response.failed":
		var apiErr *ir.APIError
		if se.Response != nil && se.Response.Error != nil {
			apiErr = &ir.APIError{
				Code:    se.Response.Error.Code,
				Message: se.Response.Error.Message,
				Type:    mapErrorType(se.Response.Error.Type),
			}
		} else if se.Error != nil {
			apiErr = &ir.APIError{
				Code:    se.Error.Code,
				Message: se.Error.Message,
				Type:    mapErrorType(se.Error.Type),
			}
		} else {
			apiErr = &ir.APIError{Type: ir.ErrServer, Message: "response failed"}
		}
		out = append(out, ir.Event{Type: ir.EvError, Err: apiErr})
		d.terminal = true

	case "error":
		if se.Error != nil {
			out = append(out, ir.Event{Type: ir.EvError, Err: &ir.APIError{
				Code:    se.Error.Code,
				Message: se.Error.Message,
				Type:    mapErrorType(se.Error.Type),
			}})
			d.terminal = true
		}

	case "output_item.added":
		if se.Item != nil {
			d.itemType[se.OutputIndex] = se.Item.Type
			if se.Item.Type == "function_call" {
				d.callID[se.OutputIndex] = se.Item.CallID
				d.callName[se.OutputIndex] = se.Item.Name
			}
		}

	case "output_item.done":
		if se.Item != nil {
			idx := se.OutputIndex
			switch se.Item.Type {
			case "message":
				// Message stop: end_turn.
				stop := ir.Stop{Reason: ir.StopEndTurn, Raw: "completed"}
				out = append(out, ir.Event{Type: ir.EvStop, Stop: &stop})
			case "function_call":
				// Final arguments reconcile; tool end already emitted via delta.
				if se.Item.Arguments != "" {
					out = append(out, ir.Event{
						Type:      ir.EvToolEnd,
						Arguments: json.RawMessage(se.Item.Arguments),
					})
				}
			case "reasoning":
				// Reasoning item done: nothing extra needed.
				_ = idx
			}
		}

	case "content_part.added":
		// No action; text arrives via output_text.delta.

	case "content_part.done":
		// No action.

	case "output_text.delta":
		out = append(out, ir.Event{Type: ir.EvText, Text: se.Delta})

	case "output_text.done":
		// Final text: no extra event needed.

	case "refusal.delta":
		// Refusals forwarded as text.
		out = append(out, ir.Event{Type: ir.EvText, Text: se.Delta})

	case "function_call_arguments.delta":
		idx := se.OutputIndex
		if !d.started {
			d.started = true
			out = append(out, ir.Event{Type: ir.EvStart})
		}
		// Emit ToolStart on first args delta.
		if _, seen := d.callID[idx]; !seen {
			d.callID[idx] = ""
		}
		if name, ok := d.callName[idx]; ok && se.Delta != "" {
			id := d.callID[idx]
			ns := ""
			if ref, ok2 := d.names[name]; ok2 {
				ns = ref.Namespace
			}
			// Emit ToolStart only once (first delta).
			delete(d.callName, idx)
			out = append(out,
				ir.Event{Type: ir.EvToolStart, ID: id, Name: name, Namespace: ns, Kind: ir.ToolFunction},
			)
		}
		out = append(out, ir.Event{Type: ir.EvToolArgs, Text: se.Delta})

	case "function_call_arguments.done":
		// Final arguments; EvToolEnd emitted by output_item.done.

	case "reasoning_summary_text.delta":
		// Reasoning summary delta.
		out = append(out, ir.Event{Type: ir.EvReasoning, Text: se.Delta})

	case "reasoning_summary_text.done":
		// No extra event.
	}

	return out, nil
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

	// Check for error envelope.
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

	resp := &ir.Response{
		ID:    wr.ID,
		Model: wr.Model,
	}
	if wr.Usage != nil {
		resp.Usage = decodeResponsesUsage(wr.Usage, opts.Dialect)
	}

	out := ir.Output{}
	for i, item := range wr.Output {
		fp := ir.Root.Key("output").At(i)
		parts, stop, err := outputItemToIR(item, fp, opts.Names, env)
		if err != nil {
			return nil, err
		}
		out.Parts = append(out.Parts, parts...)
		if stop.Reason != ir.StopUnset {
			out.Stop = stop
		}
	}

	// Map status to stop reason when not already set.
	if out.Stop.Reason == ir.StopUnset {
		switch wr.Status {
		case "completed":
			out.Stop = ir.Stop{Reason: ir.StopEndTurn, Raw: wr.Status}
		case "incomplete":
			out.Stop = ir.Stop{Reason: ir.StopMaxTokens, Raw: wr.Status}
		case "failed":
			if wr.Error != nil {
				return nil, &ir.APIError{
					Code:    wr.Error.Code,
					Message: wr.Error.Message,
					Type:    mapErrorType(wr.Error.Type),
				}
			}
		}
	}

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
		stop := statusToStop(item.Status)
		return parts, stop, nil

	case "function_call":
		args := json.RawMessage(item.Arguments)
		if len(args) == 0 {
			args = json.RawMessage(`{}`)
		}
		ns := ""
		if ref, ok := names[item.Name]; ok {
			ns = ref.Namespace
		}
		part := ir.Part{
			Type: ir.PartToolCall,
			Call: &ir.ToolCall{
				ID:        item.CallID,
				Name:      item.Name,
				Namespace: ns,
				Kind:      ir.ToolFunction,
				Arguments: args,
			},
		}
		return []ir.Part{part}, ir.Stop{}, nil

	case "reasoning":
		r := &ir.Reasoning{ID: item.ID}
		if item.EncryptedContent != "" {
			data, _ := json.Marshal(item.EncryptedContent)
			r.Sealed = &ir.Sealed{Kind: ir.SealedResponsesEncrypted, Data: data}
		} else {
			var sb string
			for _, sp := range item.Summary {
				sb += sp.Text
			}
			r.Text = sb
		}
		return []ir.Part{{Type: ir.PartReasoning, Reasoning: r}}, ir.Stop{}, nil

	case "web_search_call":
		return nil, ir.Stop{}, nil
	}

	return nil, ir.Stop{}, nil
}

func outputPartToIR(cp wireOutputPart, p ir.Path, env *ir.Env) (ir.Part, error) {
	switch cp.Type {
	case "output_text", "text":
		return ir.TextPart(cp.Text), nil
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

func decodeResponsesUsage(u *wireUsage, dia Dialect) ir.Usage {
	out := ir.Usage{
		OutputTokens: u.OutputTokens,
		Reported:     ir.UInput | ir.UOutput,
	}
	var cached, cacheWrite int
	if u.InputTokensDetails != nil {
		cached = u.InputTokensDetails.CachedTokens
		cacheWrite = u.InputTokensDetails.CacheWriteTokens
	}
	out.CacheReadTokens = cached
	out.CacheWriteTokens = cacheWrite
	out.InputTokens = u.InputTokens - cached - cacheWrite
	if cached > 0 {
		out.Reported |= ir.UCacheRead
	}
	if cacheWrite > 0 {
		out.Reported |= ir.UCacheWrite
	}
	if u.OutputTokensDetails != nil && u.OutputTokensDetails.ReasoningTokens > 0 {
		out.ReasoningTokens = u.OutputTokensDetails.ReasoningTokens
		out.Reported |= ir.UReasoning
	}
	return out
}
