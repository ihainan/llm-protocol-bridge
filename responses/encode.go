package responses

import (
	"encoding/json"
	"strings"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

type pendingRespTool struct {
	id        string
	name      string
	ns        string
	kind      ir.ToolKind
	args      strings.Builder
	finalArgs json.RawMessage
	done      bool
}

// StreamEncoder converts ir.Events into Responses API SSE chunks.
type StreamEncoder struct {
	opts    EncodeOptions
	id      string
	model   string
	started bool
	stopped bool
	errored bool
	seq     int

	// outputIndex is the current item's output_index (-1 before any item).
	outputIndex int

	// openItemType records the type of the currently open item ("message",
	// "function_call", "custom_tool_call", "reasoning", "") so we know when to close it.
	openItemType string

	// itemID is the wire item id assigned to the open item.
	itemID string
	// callID is the tool call_id (may differ from itemID).
	callID       string
	callName     string
	callNS       string
	pendingTools []*pendingRespTool
	pendingByID  map[string]*pendingRespTool

	textAccum   strings.Builder
	argAccum    strings.Builder
	reasonAccum strings.Builder

	outputs []map[string]any
	stop    ir.Stop
	usage   *ir.Usage
}

// NewStreamEncoder returns a Responses stream encoder.
func NewStreamEncoder(opts EncodeOptions) *StreamEncoder {
	return &StreamEncoder{opts: opts, outputIndex: -1, pendingByID: map[string]*pendingRespTool{}}
}

func (e *StreamEncoder) ensureStarted(dst []byte) []byte {
	if e.started {
		return dst
	}
	e.started = true
	if e.id == "" {
		e.id = "resp_" + e.opts.Env.NewID("", 24)
	}
	if e.model == "" {
		e.model = e.opts.Model
	}
	dst = e.emitResponseEvent(dst, "response.created", "in_progress")
	dst = e.emitResponseEvent(dst, "response.in_progress", "in_progress")
	return dst
}

// Encode serialises one ir.Event to Responses SSE bytes appended to dst.
func (e *StreamEncoder) Encode(dst []byte, ev ir.Event) ([]byte, error) {
	if e.errored {
		return dst, nil
	}
	switch ev.Type {
	case ir.EvStart:
		if !e.started {
			e.id = ev.ID
			if e.id == "" {
				e.id = "resp_" + e.opts.Env.NewID("", 24)
			}
			e.model = e.opts.Model
			if e.model == "" {
				e.model = ev.Model
			}
			dst = e.ensureStarted(dst)
		}
		if ev.Usage != nil {
			if e.usage == nil {
				u := *ev.Usage
				e.usage = &u
			} else {
				e.usage.Merge(*ev.Usage)
			}
		}

	case ir.EvText:
		dst = e.ensureStarted(dst)
		dst = e.ensureMessageOpen(dst)
		dst = e.appendEvent(dst, "response.output_text.delta", map[string]any{
			"sequence_number": e.nextSeq(),
			"item_id":         e.itemID,
			"output_index":    e.outputIndex,
			"content_index":   0,
			"delta":           ev.Text,
		})
		e.textAccum.WriteString(ev.Text)

	case ir.EvReasoningStart:
		dst = e.ensureStarted(dst)
		dst = e.closeItem(dst)
		e.openItemType = "reasoning"
		e.outputIndex++
		e.itemID = ev.ID
		if e.itemID == "" {
			e.itemID = "rs_" + e.opts.Env.NewID("", 16)
		}
		e.reasonAccum.Reset()
		dst = e.appendEvent(dst, "response.output_item.added", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"item": map[string]any{
				"type":    "reasoning",
				"id":      e.itemID,
				"summary": []any{},
			},
		})

	case ir.EvReasoning:
		dst = e.ensureStarted(dst)
		if e.openItemType != "reasoning" {
			dst = e.closeItem(dst)
			e.openItemType = "reasoning"
			e.outputIndex++
			e.itemID = "rs_" + e.opts.Env.NewID("", 16)
			e.reasonAccum.Reset()
			dst = e.appendEvent(dst, "response.output_item.added", map[string]any{
				"sequence_number": e.nextSeq(),
				"output_index":    e.outputIndex,
				"item": map[string]any{
					"type":    "reasoning",
					"id":      e.itemID,
					"summary": []any{},
				},
			})
		}
		e.reasonAccum.WriteString(ev.Text)
		dst = e.appendEvent(dst, "response.reasoning_summary_text.delta", map[string]any{
			"sequence_number": e.nextSeq(),
			"item_id":         e.itemID,
			"output_index":    e.outputIndex,
			"summary_index":   0,
			"delta":           ev.Text,
		})

	case ir.EvSignature:
		// Responses reasoning signatures are carried via encrypted_content.

	case ir.EvSealed:
		if ev.Sealed == nil {
			break
		}
		dst = e.ensureStarted(dst)
		dst = e.closeItem(dst)
		e.outputIndex++
		e.itemID = "rs_" + e.opts.Env.NewID("", 16)
		var encStr string
		if ev.Sealed.Kind == ir.SealedResponsesEncrypted {
			_ = json.Unmarshal(ev.Sealed.Data, &encStr)
		} else if ev.Sealed.Kind == ir.SealedAnthropicRedacted {
			encStr = e.opts.Env.Sealer().Seal(ir.SealRecord{
				Origin: ev.Sealed.Origin,
				Kind:   ir.SealRedacted,
				Value:  string(ev.Sealed.Data),
			})
		} else {
			encStr = e.opts.Env.Sealer().Seal(ir.SealRecord{
				Origin: ev.Sealed.Origin,
				Kind:   ir.SealEncrypted,
				Value:  string(ev.Sealed.Data),
			})
		}
		item := map[string]any{
			"type":              "reasoning",
			"id":                e.itemID,
			"summary":           []any{},
			"encrypted_content": encStr,
		}
		dst = e.appendEvent(dst, "response.output_item.added", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"item":            item,
		})
		doneItem := map[string]any{
			"type":              "reasoning",
			"id":                e.itemID,
			"summary":           []any{},
			"encrypted_content": encStr,
			"status":            "completed",
		}
		e.outputs = append(e.outputs, doneItem)
		dst = e.appendEvent(dst, "response.output_item.done", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"item":            doneItem,
		})
		e.openItemType = ""

	case ir.EvToolStart:
		dst = e.ensureStarted(dst)
		if e.openItemType == "function_call" || e.openItemType == "custom_tool_call" {
			pt := &pendingRespTool{id: ev.ID, name: ev.Name, ns: ev.Namespace, kind: ev.Kind}
			e.pendingTools = append(e.pendingTools, pt)
			if ev.ID != "" {
				e.pendingByID[ev.ID] = pt
			}
			break
		}
		dst = e.openToolItem(dst, ev.ID, ev.Name, ev.Namespace, ev.Kind)

	case ir.EvToolArgs:
		if ev.ID != "" && ev.ID != e.callID {
			if pt := e.pendingByID[ev.ID]; pt != nil {
				pt.args.WriteString(ev.Text)
				break
			}
		}
		switch e.openItemType {
		case "function_call":
			e.argAccum.WriteString(ev.Text)
			dst = e.appendEvent(dst, "response.function_call_arguments.delta", map[string]any{
				"sequence_number": e.nextSeq(),
				"output_index":    e.outputIndex,
				"item_id":         e.itemID,
				"delta":           ev.Text,
			})
		case "custom_tool_call":
			e.argAccum.WriteString(ev.Text)
			dst = e.appendEvent(dst, "response.custom_tool_call_input.delta", map[string]any{
				"sequence_number": e.nextSeq(),
				"output_index":    e.outputIndex,
				"item_id":         e.itemID,
				"delta":           ev.Text,
			})
		}

	case ir.EvToolEnd:
		if ev.ID != "" && ev.ID != e.callID {
			if pt := e.pendingByID[ev.ID]; pt != nil {
				pt.finalArgs = ev.Arguments
				pt.done = true
				break
			}
		}
		dst = e.finishToolItem(dst, ev.Arguments)
		dst = e.drainPendingTools(dst)

	case ir.EvStop:
		e.stop = ir.Stop{}
		if ev.Stop != nil {
			e.stop = *ev.Stop
		}
		e.stopped = true
		dst = e.closeItem(dst)
		dst = e.flushAllPendingTools(dst)
		// Defer terminal response.completed / response.incomplete to Close()
		// so trailing EvUsage events (Chat, Anthropic, EventsOf) are included.

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
			dst = e.ensureStarted(dst)
			e.errored = true
			respObj := e.buildResponseObject("failed", false)
			respObj["error"] = map[string]any{
				"code":    ev.Err.Code,
				"message": ev.Err.Message,
			}
			dst = e.appendEvent(dst, "response.failed", map[string]any{
				"sequence_number": e.nextSeq(),
				"response":        respObj,
			})
		}
	}
	return dst, nil
}

// Close emits the terminal response.completed or response.incomplete event
// (unless the stream already terminated with response.failed).
func (e *StreamEncoder) Close(dst []byte) ([]byte, error) {
	if e.errored {
		return dst, nil
	}
	if !e.started && !e.stopped && len(e.outputs) == 0 {
		return dst, nil
	}
	dst = e.ensureStarted(dst)
	dst = e.closeItem(dst)
	dst = e.flushAllPendingTools(dst)
	status := stopToResponsesStatus(e.stop)
	evName := "response.completed"
	if status == "incomplete" {
		evName = "response.incomplete"
	}
	respObj := e.buildResponseObject(status, true)
	if status == "incomplete" {
		reason := "max_output_tokens"
		if e.stop.Filtered() {
			reason = "content_filter"
		}
		respObj["incomplete_details"] = map[string]any{"reason": reason}
	}
	dst = e.appendEvent(dst, evName, map[string]any{
		"sequence_number": e.nextSeq(),
		"response":        respObj,
	})
	return dst, nil
}

// Keepalive sends a response.in_progress event.
func (e *StreamEncoder) Keepalive(dst []byte) []byte {
	return e.emitResponseEvent(dst, "response.in_progress", "in_progress")
}

func (e *StreamEncoder) ensureMessageOpen(dst []byte) []byte {
	if e.openItemType == "message" {
		return dst
	}
	dst = e.closeItem(dst)
	e.openItemType = "message"
	e.outputIndex++
	e.itemID = "msg_" + e.opts.Env.NewID("", 16)
	e.textAccum.Reset()
	dst = e.appendEvent(dst, "response.output_item.added", map[string]any{
		"sequence_number": e.nextSeq(),
		"output_index":    e.outputIndex,
		"item": map[string]any{
			"type":    "message",
			"id":      e.itemID,
			"role":    "assistant",
			"status":  "in_progress",
			"content": []any{},
		},
	})
	dst = e.appendEvent(dst, "response.content_part.added", map[string]any{
		"sequence_number": e.nextSeq(),
		"item_id":         e.itemID,
		"output_index":    e.outputIndex,
		"content_index":   0,
		"part": map[string]any{
			"type": "output_text",
			"text": "",
		},
	})
	return dst
}

func (e *StreamEncoder) openToolItem(dst []byte, id, name, ns string, kind ir.ToolKind) []byte {
	dst = e.closeItem(dst)
	e.outputIndex++
	e.callName = name
	e.callNS = ns
	e.callID = id
	if e.callID == "" {
		e.callID = "call_" + e.opts.Env.NewID("", 16)
	}
	e.argAccum.Reset()
	if kind == ir.ToolCustom {
		e.openItemType = "custom_tool_call"
		e.itemID = "ctc_" + e.opts.Env.NewID("", 16)
		item := map[string]any{
			"type":    "custom_tool_call",
			"id":      e.itemID,
			"call_id": e.callID,
			"name":    name,
			"input":   "",
			"status":  "in_progress",
		}
		if ns != "" {
			item["namespace"] = ns
		}
		return e.appendEvent(dst, "response.output_item.added", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"item":            item,
		})
	}
	e.openItemType = "function_call"
	e.itemID = "fc_" + e.opts.Env.NewID("", 16)
	return e.appendEvent(dst, "response.output_item.added", map[string]any{
		"sequence_number": e.nextSeq(),
		"output_index":    e.outputIndex,
		"item": map[string]any{
			"type":      "function_call",
			"id":        e.itemID,
			"call_id":   e.callID,
			"name":      name,
			"arguments": "",
			"status":    "in_progress",
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
		dst = e.openToolItem(dst, pt.id, pt.name, pt.ns, pt.kind)
		if s := pt.args.String(); s != "" {
			e.argAccum.WriteString(s)
			evName := "response.function_call_arguments.delta"
			if pt.kind == ir.ToolCustom {
				evName = "response.custom_tool_call_input.delta"
			}
			dst = e.appendEvent(dst, evName, map[string]any{
				"sequence_number": e.nextSeq(),
				"output_index":    e.outputIndex,
				"item_id":         e.itemID,
				"delta":           s,
			})
		}
		if pt.done {
			dst = e.finishToolItem(dst, pt.finalArgs)
			continue
		}
		break
	}
	return dst
}

func (e *StreamEncoder) flushAllPendingTools(dst []byte) []byte {
	for len(e.pendingTools) > 0 {
		e.pendingTools[0].done = true
		dst = e.drainPendingTools(dst)
	}
	return dst
}

func (e *StreamEncoder) finishToolItem(dst []byte, finalArgs json.RawMessage) []byte {
	switch e.openItemType {
	case "function_call":
		args := strings.TrimSpace(string(finalArgs))
		if args == "" {
			args = strings.TrimSpace(e.argAccum.String())
		}
		if args == "" {
			args = "{}"
		}
		dst = e.appendEvent(dst, "response.function_call_arguments.done", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"item_id":         e.itemID,
			"arguments":       args,
		})
		doneItem := map[string]any{
			"type":      "function_call",
			"id":        e.itemID,
			"call_id":   e.callID,
			"name":      e.callName,
			"arguments": args,
			"status":    "completed",
		}
		e.outputs = append(e.outputs, doneItem)
		dst = e.appendEvent(dst, "response.output_item.done", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"item":            doneItem,
		})
		e.openItemType = ""
		e.argAccum.Reset()
	case "custom_tool_call":
		input := e.argAccum.String()
		if len(finalArgs) > 0 {
			var s string
			if json.Unmarshal(finalArgs, &s) == nil {
				input = s
			}
		}
		dst = e.appendEvent(dst, "response.custom_tool_call_input.done", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"item_id":         e.itemID,
			"input":           input,
		})
		doneItem := map[string]any{
			"type":    "custom_tool_call",
			"id":      e.itemID,
			"call_id": e.callID,
			"name":    e.callName,
			"input":   input,
			"status":  "completed",
		}
		if e.callNS != "" {
			doneItem["namespace"] = e.callNS
		}
		e.outputs = append(e.outputs, doneItem)
		dst = e.appendEvent(dst, "response.output_item.done", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"item":            doneItem,
		})
		e.openItemType = ""
		e.argAccum.Reset()
	}
	return dst
}

func (e *StreamEncoder) closeItem(dst []byte) []byte {
	switch e.openItemType {
	case "message":
		text := e.textAccum.String()
		dst = e.appendEvent(dst, "response.output_text.done", map[string]any{
			"sequence_number": e.nextSeq(),
			"item_id":         e.itemID,
			"output_index":    e.outputIndex,
			"content_index":   0,
			"text":            text,
		})
		dst = e.appendEvent(dst, "response.content_part.done", map[string]any{
			"sequence_number": e.nextSeq(),
			"item_id":         e.itemID,
			"output_index":    e.outputIndex,
			"content_index":   0,
			"part": map[string]any{
				"type": "output_text",
				"text": text,
			},
		})
		doneItem := map[string]any{
			"type":   "message",
			"id":     e.itemID,
			"role":   "assistant",
			"status": "completed",
			"content": []map[string]any{
				{"type": "output_text", "text": text},
			},
		}
		e.outputs = append(e.outputs, doneItem)
		dst = e.appendEvent(dst, "response.output_item.done", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"item":            doneItem,
		})
	case "reasoning":
		rtext := e.reasonAccum.String()
		summary := []map[string]any{}
		if rtext != "" {
			dst = e.appendEvent(dst, "response.reasoning_summary_text.done", map[string]any{
				"sequence_number": e.nextSeq(),
				"item_id":         e.itemID,
				"output_index":    e.outputIndex,
				"summary_index":   0,
				"text":            rtext,
			})
			summary = append(summary, map[string]any{"type": "summary_text", "text": rtext})
		}
		doneItem := map[string]any{
			"type":    "reasoning",
			"id":      e.itemID,
			"summary": summary,
			"status":  "completed",
		}
		e.outputs = append(e.outputs, doneItem)
		dst = e.appendEvent(dst, "response.output_item.done", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"item":            doneItem,
		})
	case "function_call", "custom_tool_call":
		return e.finishToolItem(dst, nil)
	}
	e.openItemType = ""
	e.textAccum.Reset()
	e.reasonAccum.Reset()
	return dst
}

func (e *StreamEncoder) buildResponseObject(status string, includeUsage bool) map[string]any {
	outs := e.outputs
	if outs == nil {
		outs = []map[string]any{}
	}
	m := map[string]any{
		"id":     e.id,
		"object": "response",
		"model":  e.model,
		"status": status,
		"output": outs,
	}
	if includeUsage && e.usage != nil {
		m["usage"] = irUsageToWire(e.usage)
	}
	return m
}

func (e *StreamEncoder) emitResponseEvent(dst []byte, evType, status string) []byte {
	return e.appendEvent(dst, evType, map[string]any{
		"sequence_number": e.nextSeq(),
		"response":        e.buildResponseObject(status, false),
	})
}

func (e *StreamEncoder) appendEvent(dst []byte, evType string, v any) []byte {
	if m, ok := v.(map[string]any); ok {
		m["type"] = evType
	}
	b, err := jsonx.Marshal(v)
	if err != nil {
		return dst
	}
	return sse.AppendEvent(dst, evType, b)
}

func (e *StreamEncoder) nextSeq() int {
	s := e.seq
	e.seq++
	return s
}

// BuildResponse serialises a whole ir.Response to a non-streaming Responses body.
func BuildResponse(resp *ir.Response, opts EncodeOptions) ([]byte, error) {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	id := resp.ID
	if id == "" {
		id = "resp_" + env.NewID("", 24)
	}
	model := opts.Model
	if model == "" {
		model = resp.Model
	}

	var out []wireOutputItem
	for _, o := range resp.Outputs {
		var parts []wireOutputItem
		for i, p := range o.Parts {
			fp := ir.Root.Key("output").At(i)
			its, err := irPartToOutputItem(p, fp, env)
			if err != nil {
				return nil, err
			}
			parts = append(parts, its...)
		}
		out = append(out, parts...)
	}
	if out == nil {
		out = []wireOutputItem{}
	}

	stop := respStop(resp)
	status := stopToResponsesStatus(stop)
	var usageRaw json.RawMessage
	if wu := irUsageToWire(&resp.Usage); wu != nil {
		usageRaw, _ = json.Marshal(wu)
	}
	wr := wireResponse{
		ID:     id,
		Object: "response",
		Model:  model,
		Status: status,
		Output: out,
		Usage:  usageRaw,
	}
	if status == "incomplete" {
		reason := "max_output_tokens"
		if stop.Filtered() {
			reason = "content_filter"
		}
		wr.IncompleteDetails = &wireIncompleteDetails{Reason: reason}
	}
	return json.Marshal(wr)
}

func respStop(resp *ir.Response) ir.Stop {
	if len(resp.Outputs) > 0 {
		return resp.Outputs[0].Stop
	}
	return ir.Stop{}
}

func irPartToOutputItem(p ir.Part, fp ir.Path, env *ir.Env) ([]wireOutputItem, error) {
	switch p.Type {
	case ir.PartText:
		id := "msg_" + env.NewID("", 16)
		return []wireOutputItem{{
			Type:    "message",
			ID:      id,
			Role:    "assistant",
			Status:  "completed",
			Content: []wireOutputPart{{Type: "output_text", Text: p.Text}},
		}}, nil

	case ir.PartRefusal:
		id := "msg_" + env.NewID("", 16)
		return []wireOutputItem{{
			Type:    "message",
			ID:      id,
			Role:    "assistant",
			Status:  "completed",
			Content: []wireOutputPart{{Type: "refusal", Refusal: p.Text}},
		}}, nil

	case ir.PartReasoning:
		if p.Reasoning == nil {
			return nil, nil
		}
		item := wireOutputItem{
			Type: "reasoning",
			ID:   p.Reasoning.ID,
		}
		if item.ID == "" {
			item.ID = "rs_" + env.NewID("", 16)
		}
		if p.Reasoning.Sealed != nil {
			if p.Reasoning.Sealed.Kind == ir.SealedResponsesEncrypted {
				var encStr string
				_ = json.Unmarshal(p.Reasoning.Sealed.Data, &encStr)
				item.EncryptedContent = encStr
			} else if p.Reasoning.Sealed.Kind == ir.SealedAnthropicRedacted {
				item.EncryptedContent = env.Sealer().Seal(ir.SealRecord{
					Origin: p.Reasoning.Sealed.Origin,
					Kind:   ir.SealRedacted,
					Value:  string(p.Reasoning.Sealed.Data),
				})
			}
		} else if p.Reasoning.Signature != nil && p.Reasoning.Signature.Value != "" && !p.Reasoning.Signature.Origin.Matches(ir.FormatResponses, env.ProviderID) {
			item.EncryptedContent = env.Sealer().Seal(ir.SealRecord{
				Origin: p.Reasoning.Signature.Origin,
				Kind:   ir.SealSignature,
				Value:  p.Reasoning.Signature.Value,
				Text:   p.Reasoning.Text,
			})
		}
		for _, s := range p.Reasoning.Summary {
			item.Summary = append(item.Summary, wireReasoningPart{Type: "summary_text", Text: s})
		}
		if len(item.Summary) == 0 && p.Reasoning.Text != "" {
			item.Summary = []wireReasoningPart{{Type: "summary_text", Text: p.Reasoning.Text}}
		}
		return []wireOutputItem{item}, nil

	case ir.PartToolCall:
		if p.Call == nil {
			return nil, nil
		}
		c := p.Call
		args := string(c.Arguments)
		if args == "" {
			args = "{}"
		}
		switch c.Kind {
		case ir.ToolCustom:
			input := c.Input
			if input == "" {
				var m map[string]string
				if json.Unmarshal(c.Arguments, &m) == nil {
					input = m["input"]
				}
			}
			return []wireOutputItem{{
				Type:      "custom_tool_call",
				ID:        "ctc_" + env.NewID("", 16),
				CallID:    c.ID,
				Name:      c.Name,
				Namespace: c.Namespace,
				Input:     input,
			}}, nil
		default:
			return []wireOutputItem{{
				Type:      "function_call",
				ID:        "fc_" + env.NewID("", 16),
				CallID:    c.ID,
				Name:      c.Name,
				Arguments: args,
				Status:    "completed",
			}}, nil
		}

	case ir.PartBuiltinCall, ir.PartBuiltinResult:
		return nil, nil

	default:
		return nil, env.Err(ir.Unsupported(fp, string(p.Type), "part type not supported in Responses output"))
	}
}

func stopToResponsesStatus(s ir.Stop) string {
	switch s.Reason {
	case ir.StopEndTurn, ir.StopToolUse, ir.StopSequence, ir.StopPauseTurn:
		return "completed"
	case ir.StopMaxTokens, ir.StopContextWindow:
		return "incomplete"
	case ir.StopContentFilter, ir.StopRefusal:
		return "incomplete"
	}
	return "completed"
}

func irUsageToWire(u *ir.Usage) *wireUsage {
	if u == nil || !u.Any() {
		return nil
	}
	w := &wireUsage{
		InputTokens:  u.PromptTokens(),
		OutputTokens: u.OutputTokens,
		TotalTokens:  u.TotalTokens(),
	}
	if u.Has(ir.UCacheRead) || u.Has(ir.UCacheWrite) {
		w.InputTokensDetails = &wireInputDetails{
			CachedTokens:     u.CacheReadTokens,
			CacheWriteTokens: u.CacheWriteTokens,
		}
	}
	if u.Has(ir.UReasoning) {
		w.OutputTokensDetails = &wireOutputDetails{ReasoningTokens: u.ReasoningTokens}
	}
	return w
}
