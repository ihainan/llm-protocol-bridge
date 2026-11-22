package responses

import (
	"encoding/json"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// StreamEncoder converts ir.Events into Responses API SSE chunks.
type StreamEncoder struct {
	opts    EncodeOptions
	id      string
	model   string
	started bool
	seq     int

	// Per-output-item index tracking.
	outputIndex int

	// openItemType records the type of the currently open item ("message",
	// "function_call", "reasoning", "") so we know when to close it.
	openItemType string

	// itemID is the id assigned to the open item.
	itemID string

	// callName records the name of the open function call.
	callName string

	// textAccum accumulates streamed text for the open message item.
	textAccum string

	// stop is the stop reason received (mapped to responses status).
	stop ir.Stop

	// usage tracks the last usage snapshot.
	usage *ir.Usage
}

// NewStreamEncoder returns a Responses stream encoder.
func NewStreamEncoder(opts EncodeOptions) *StreamEncoder {
	return &StreamEncoder{opts: opts}
}

// Encode serialises one ir.Event to Responses SSE bytes appended to dst.
func (e *StreamEncoder) Encode(dst []byte, ev ir.Event) ([]byte, error) {
	switch ev.Type {
	case ir.EvStart:
		e.id = ev.ID
		if e.id == "" {
			e.id = "resp_" + e.opts.Env.NewID("", 24)
		}
		e.model = ev.Model
		if e.model == "" {
			e.model = e.opts.Model
		}
		if !e.started {
			e.started = true
			dst = e.emitResponseEvent(dst, "response.created", nil)
			dst = e.emitResponseEvent(dst, "response.in_progress", nil)
		}

	case ir.EvText:
		dst = e.ensureMessageOpen(dst)
		dst = e.appendEvent(dst, "output_text.delta", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"delta":           ev.Text,
		})
		e.textAccum += ev.Text

	case ir.EvReasoningStart:
		dst = e.closeItem(dst)
		e.openItemType = "reasoning"
		e.outputIndex++
		e.itemID = "rs_" + e.opts.Env.NewID("", 16)
		dst = e.appendEvent(dst, "output_item.added", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"item": map[string]any{
				"type": "reasoning",
				"id":   e.itemID,
			},
		})

	case ir.EvReasoning:
		if e.openItemType != "reasoning" {
			dst = e.closeItem(dst)
			e.openItemType = "reasoning"
			e.outputIndex++
			e.itemID = "rs_" + e.opts.Env.NewID("", 16)
			dst = e.appendEvent(dst, "output_item.added", map[string]any{
				"sequence_number": e.nextSeq(),
				"output_index":    e.outputIndex,
				"item": map[string]any{
					"type": "reasoning",
					"id":   e.itemID,
				},
			})
		}
		dst = e.appendEvent(dst, "reasoning_summary_text.delta", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"delta":           ev.Text,
		})

	case ir.EvSignature:
		// Responses reasoning signatures are carried via encrypted_content, which
		// the bridge-level seal codec handles. Ignore bare signature events here.

	case ir.EvSealed:
		// Encrypted reasoning arrives as a reasoning item with encrypted_content.
		if ev.Sealed == nil {
			break
		}
		dst = e.closeItem(dst)
		e.openItemType = "reasoning"
		e.outputIndex++
		e.itemID = "rs_" + e.opts.Env.NewID("", 16)
		var encStr string
		_ = json.Unmarshal(ev.Sealed.Data, &encStr)
		dst = e.appendEvent(dst, "output_item.added", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"item": map[string]any{
				"type":              "reasoning",
				"id":                e.itemID,
				"encrypted_content": encStr,
			},
		})
		dst = e.appendEvent(dst, "output_item.done", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"item": map[string]any{
				"type":              "reasoning",
				"id":                e.itemID,
				"encrypted_content": encStr,
				"status":            "completed",
			},
		})
		e.openItemType = ""

	case ir.EvToolStart:
		dst = e.closeItem(dst)
		e.openItemType = "function_call"
		e.outputIndex++
		e.callName = ev.Name
		e.itemID = ev.ID
		if e.itemID == "" {
			e.itemID = "fc_" + e.opts.Env.NewID("", 16)
		}
		dst = e.appendEvent(dst, "output_item.added", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"item": map[string]any{
				"type":    "function_call",
				"id":      e.itemID,
				"call_id": e.itemID,
				"name":    ev.Name,
				"status":  "in_progress",
			},
		})

	case ir.EvToolArgs:
		if e.openItemType != "function_call" {
			break
		}
		dst = e.appendEvent(dst, "function_call_arguments.delta", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"item_id":         e.itemID,
			"delta":           ev.Text,
		})

	case ir.EvToolEnd:
		if e.openItemType != "function_call" {
			break
		}
		args := "{}"
		if len(ev.Arguments) > 0 {
			args = string(ev.Arguments)
		}
		dst = e.appendEvent(dst, "function_call_arguments.done", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"item_id":         e.itemID,
			"arguments":       args,
		})
		dst = e.appendEvent(dst, "output_item.done", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"item": map[string]any{
				"type":      "function_call",
				"id":        e.itemID,
				"call_id":   e.itemID,
				"name":      e.callName,
				"arguments": args,
				"status":    "completed",
			},
		})
		e.openItemType = ""

	case ir.EvStop:
		e.stop = ir.Stop{}
		if ev.Stop != nil {
			e.stop = *ev.Stop
		}
		dst = e.closeItem(dst)
		status := stopToResponsesStatus(e.stop)
		dst = e.appendEvent(dst, "response.completed", map[string]any{
			"sequence_number": e.nextSeq(),
			"response":        e.buildResponseObject(status, true),
		})

	case ir.EvUsage:
		if ev.Usage != nil {
			e.usage = ev.Usage
		}

	case ir.EvError:
		if ev.Err != nil {
			dst = BuildStreamError(dst, ev.Err)
		}
	}
	return dst, nil
}

// Close finalises the stream (no extra events beyond EvStop).
func (e *StreamEncoder) Close(dst []byte) ([]byte, error) {
	return dst, nil
}

// Keepalive sends a response.in_progress event.
func (e *StreamEncoder) Keepalive(dst []byte) []byte {
	return e.emitResponseEvent(dst, "response.in_progress", nil)
}

func (e *StreamEncoder) ensureMessageOpen(dst []byte) []byte {
	if e.openItemType == "message" {
		return dst
	}
	dst = e.closeItem(dst)
	e.openItemType = "message"
	e.outputIndex++
	e.itemID = "msg_" + e.opts.Env.NewID("", 16)
	e.textAccum = ""
	dst = e.appendEvent(dst, "output_item.added", map[string]any{
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
	dst = e.appendEvent(dst, "content_part.added", map[string]any{
		"sequence_number": e.nextSeq(),
		"output_index":    e.outputIndex,
		"content_index":   0,
		"part": map[string]any{
			"type": "output_text",
			"text": "",
		},
	})
	return dst
}

func (e *StreamEncoder) closeItem(dst []byte) []byte {
	switch e.openItemType {
	case "message":
		dst = e.appendEvent(dst, "output_text.done", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"text":            e.textAccum,
		})
		dst = e.appendEvent(dst, "content_part.done", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"content_index":   0,
			"part": map[string]any{
				"type": "output_text",
				"text": e.textAccum,
			},
		})
		dst = e.appendEvent(dst, "output_item.done", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"item": map[string]any{
				"type":   "message",
				"id":     e.itemID,
				"role":   "assistant",
				"status": "completed",
				"content": []map[string]any{
					{"type": "output_text", "text": e.textAccum},
				},
			},
		})
	case "reasoning":
		// If we got here without a done event, emit one now.
		dst = e.appendEvent(dst, "output_item.done", map[string]any{
			"sequence_number": e.nextSeq(),
			"output_index":    e.outputIndex,
			"item": map[string]any{
				"type":   "reasoning",
				"id":     e.itemID,
				"status": "completed",
			},
		})
	}
	e.openItemType = ""
	e.textAccum = ""
	return dst
}

func (e *StreamEncoder) buildResponseObject(status string, includeUsage bool) map[string]any {
	m := map[string]any{
		"id":     e.id,
		"object": "response",
		"model":  e.model,
		"status": status,
		"output": []any{},
	}
	if includeUsage && e.usage != nil {
		m["usage"] = irUsageToWire(e.usage)
	}
	return m
}

func (e *StreamEncoder) emitResponseEvent(dst []byte, evType string, _ any) []byte {
	return e.appendEvent(dst, evType, map[string]any{
		"sequence_number": e.nextSeq(),
		"response":        e.buildResponseObject("", false),
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

	status := stopToResponsesStatus(respStop(resp))
	wr := wireResponse{
		ID:     id,
		Object: "response",
		Model:  model,
		Status: status,
		Output: out,
		Usage:  irUsageToWire(&resp.Usage),
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
		if p.Reasoning.Sealed != nil {
			var encStr string
			_ = json.Unmarshal(p.Reasoning.Sealed.Data, &encStr)
			item.EncryptedContent = encStr
		} else {
			for _, s := range p.Reasoning.Summary {
				item.Summary = append(item.Summary, wireReasoningPart{Type: "summary_text", Text: s})
			}
			if len(item.Summary) == 0 && p.Reasoning.Text != "" {
				item.Summary = []wireReasoningPart{{Type: "summary_text", Text: p.Reasoning.Text}}
			}
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
				ID:        c.ID,
				CallID:    c.ID,
				Name:      c.Name,
				Namespace: c.Namespace,
				Input:     input,
			}}, nil
		default:
			return []wireOutputItem{{
				Type:      "function_call",
				ID:        c.ID,
				CallID:    c.ID,
				Name:      c.Name,
				Arguments: args,
				Status:    "completed",
			}}, nil
		}

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
