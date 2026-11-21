package anthropic

import (
	"encoding/json"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// StreamDecoder converts Anthropic streaming SSE events into ir.Events.
type StreamDecoder struct {
	env      *ir.Env
	names    map[string]ir.ToolRef
	started  bool
	terminal bool
	// openBlocks maps block index -> type of the open block.
	openBlocks map[int]string
}

// NewStreamDecoder returns an Anthropic stream decoder.
func NewStreamDecoder(opts DecodeOptions) *StreamDecoder {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	env.Style = ir.StyleDot
	return &StreamDecoder{
		env:        env,
		names:      opts.Names,
		openBlocks: map[int]string{},
	}
}

// Decode converts one SSE event into zero or more IR events.
func (d *StreamDecoder) Decode(out []ir.Event, ev sse.Event) ([]ir.Event, error) {
	if d.terminal {
		return out, nil
	}
	if len(ev.Data) == 0 {
		return out, nil
	}

	var se wireStreamEvent
	if json.Unmarshal(ev.Data, &se) != nil {
		return out, nil
	}

	switch se.Type {
	case "ping":
		// heartbeat, ignore

	case "message_start":
		if se.Message == nil {
			return out, nil
		}
		d.started = true
		var usage *ir.Usage
		if se.Message.Usage != nil {
			u := decodeUsage(se.Message.Usage)
			usage = &u
		}
		out = append(out, ir.Event{
			Type:  ir.EvStart,
			ID:    se.Message.ID,
			Model: se.Message.Model,
			Usage: usage,
		})

	case "content_block_start":
		if se.ContentBlock == nil {
			return out, nil
		}
		d.openBlocks[se.Index] = se.ContentBlock.Type
		switch se.ContentBlock.Type {
		case "text":
			// No explicit start event; text deltas produce EvText.
		case "thinking":
			out = append(out, ir.Event{Type: ir.EvReasoningStart, Index: 0, ID: ""})
		case "redacted_thinking":
			out = append(out, ir.Event{Type: ir.EvReasoningStart, Index: 0, ID: ""})
		case "tool_use":
			id := se.ContentBlock.ID
			name := se.ContentBlock.Name
			out = append(out, ir.Event{
				Type:      ir.EvToolStart,
				Index:     0,
				ID:        id,
				Name:      name,
				Namespace: "",
				Kind:      ir.ToolFunction,
			})
		}

	case "content_block_delta":
		if se.Delta == nil {
			return out, nil
		}
		blockType := d.openBlocks[se.Index]
		switch se.Delta.Type {
		case "text_delta":
			if blockType == "text" {
				out = append(out, ir.Event{Type: ir.EvText, Index: 0, Text: se.Delta.Text})
			}
		case "thinking_delta":
			out = append(out, ir.Event{Type: ir.EvReasoning, Index: 0, Text: se.Delta.Thinking})
		case "signature_delta":
			sig := &ir.Signature{
				Value:  se.Delta.Signature,
				Origin: ir.Origin{Format: ir.FormatAnthropic},
			}
			out = append(out, ir.Event{Type: ir.EvSignature, Index: 0, Signature: sig})
		case "input_json_delta":
			out = append(out, ir.Event{Type: ir.EvToolArgs, Index: 0, Text: se.Delta.PartialJSON})
		}

	case "content_block_stop":
		blockType := d.openBlocks[se.Index]
		switch blockType {
		case "tool_use":
			out = append(out, ir.Event{Type: ir.EvToolEnd, Index: 0})
		case "redacted_thinking":
			// The sealed block was delivered at content_block_start (whole).
			// We already emitted EvReasoningStart; emit a sealed event now.
			if se.ContentBlock != nil {
				data, _ := jsonx.Marshal(se.ContentBlock.Data)
				sealed := &ir.Sealed{
					Kind:   ir.SealedAnthropicRedacted,
					Data:   data,
					Origin: ir.Origin{Format: ir.FormatAnthropic},
				}
				out = append(out, ir.Event{Type: ir.EvSealed, Index: 0, Sealed: sealed})
			}
		}
		delete(d.openBlocks, se.Index)

	case "message_delta":
		if se.Delta != nil && se.Delta.StopReason != "" {
			stop := stopFromAnthropic(se.Delta.StopReason, se.Delta.StopSequence)
			out = append(out, ir.Event{Type: ir.EvStop, Index: 0, Stop: &stop})
		}
		if se.Usage != nil {
			u := decodeUsage(se.Usage)
			out = append(out, ir.Event{Type: ir.EvUsage, Usage: &u})
		}

	case "message_stop":
		d.terminal = true

	case "error":
		if se.Error != nil {
			e := &ir.APIError{
				Type:    mapAnthropicErrorType(se.Error.Type),
				Message: se.Error.Message,
				Raw:     append([]byte(nil), ev.Data...),
			}
			out = append(out, ir.Event{Type: ir.EvError, Err: e})
			d.terminal = true
		}
	}

	return out, nil
}

// End finalises the stream.
func (d *StreamDecoder) End(how ir.EndKind) *ir.Event {
	if d.terminal {
		return nil
	}
	return ir.EndErrorEvent(how)
}

// stopFromAnthropic maps Anthropic stop_reason to ir.Stop.
func stopFromAnthropic(reason, seq string) ir.Stop {
	switch reason {
	case "end_turn":
		return ir.Stop{Reason: ir.StopEndTurn, Raw: reason}
	case "max_tokens":
		return ir.Stop{Reason: ir.StopMaxTokens, Raw: reason}
	case "stop_sequence":
		return ir.Stop{Reason: ir.StopSequence, Sequence: seq, Raw: reason}
	case "tool_use":
		return ir.Stop{Reason: ir.StopToolUse, Raw: reason}
	case "pause_turn":
		return ir.Stop{Reason: ir.StopPauseTurn, Raw: reason}
	}
	return ir.Stop{Reason: ir.StopEndTurn, Raw: reason}
}

// decodeUsage converts a wireUsage to ir.Usage.
func decodeUsage(u *wireUsage) ir.Usage {
	out := ir.Usage{
		InputTokens:      u.InputTokens,
		CacheReadTokens:  u.CacheReadInputTokens,
		CacheWriteTokens: u.CacheCreationInputTokens,
		OutputTokens:     u.OutputTokens,
	}
	out.Reported |= ir.UInput | ir.UOutput
	if u.CacheReadInputTokens > 0 {
		out.Reported |= ir.UCacheRead
	}
	if u.CacheCreationInputTokens > 0 {
		out.Reported |= ir.UCacheWrite
	}
	return out
}

// ParseResponse decodes a non-streaming Anthropic Messages response.
func ParseResponse(body []byte, opts DecodeOptions) (*ir.Response, error) {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	env.Style = ir.StyleDot

	var wr wireResponse
	if json.Unmarshal(body, &wr) != nil {
		if e := ParseError(0, body); e.Message != "" {
			return nil, e
		}
		return nil, ir.Protocol("anthropic: response body is not a valid object")
	}
	if wr.Type == "error" {
		e := ParseError(0, body)
		return nil, e
	}

	resp := &ir.Response{ID: wr.ID, Model: wr.Model}

	var parts []ir.Part
	for i, block := range wr.Content {
		fp := ir.Root.Key("content").At(i)
		part, err := responseBlockToPart(block, fp, env)
		if err != nil {
			return nil, err
		}
		if part != nil {
			parts = append(parts, *part)
		}
	}
	stop := stopFromAnthropic(wr.StopReason, wr.StopSequence)
	stop = ir.FinalStop(stop, parts)
	resp.Outputs = []ir.Output{{Index: 0, Parts: parts, Stop: stop}}

	if wr.Usage != nil {
		resp.Usage = decodeUsage(wr.Usage)
	}
	return resp, nil
}

func responseBlockToPart(block wireContentBlock, p ir.Path, env *ir.Env) (*ir.Part, error) {
	switch block.Type {
	case "text":
		part := ir.TextPart(block.Text)
		return &part, nil
	case "thinking":
		part := ir.Part{
			Type: ir.PartReasoning,
			Reasoning: &ir.Reasoning{
				Text: block.Thinking,
				Signature: &ir.Signature{
					Value:  block.Signature,
					Origin: ir.Origin{Format: ir.FormatAnthropic},
				},
			},
		}
		return &part, nil
	case "redacted_thinking":
		data, _ := jsonx.Marshal(block.Data)
		part := ir.Part{
			Type: ir.PartReasoning,
			Reasoning: &ir.Reasoning{
				Sealed: &ir.Sealed{
					Kind:   ir.SealedAnthropicRedacted,
					Data:   data,
					Origin: ir.Origin{Format: ir.FormatAnthropic},
				},
			},
		}
		return &part, nil
	case "tool_use":
		args := block.Input
		if len(args) == 0 {
			args = json.RawMessage(`{}`)
		}
		part := ir.Part{
			Type: ir.PartToolCall,
			Call: &ir.ToolCall{
				ID:        block.ID,
				Name:      block.Name,
				Kind:      ir.ToolFunction,
				Arguments: args,
			},
		}
		return &part, nil
	}
	return nil, nil
}
