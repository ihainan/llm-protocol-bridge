package anthropic

import (
	"encoding/json"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// StreamEncoder converts ir.Events into Anthropic Messages SSE chunks.
type StreamEncoder struct {
	opts    EncodeOptions
	id      string
	model   string
	started bool
	// blockIdx is the next content block index to assign.
	blockIdx int
	// openBlock is the index of the currently open content block (-1 = none).
	openBlock int
	// openBlockType is the type string of the open block.
	openBlockType string
}

// NewStreamEncoder returns an Anthropic stream encoder.
func NewStreamEncoder(opts EncodeOptions) *StreamEncoder {
	return &StreamEncoder{
		opts:      opts,
		openBlock: -1,
	}
}

// Encode serialises one ir.Event to Anthropic SSE bytes appended to dst.
func (e *StreamEncoder) Encode(dst []byte, ev ir.Event) ([]byte, error) {
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
		if !e.started {
			e.started = true
			dst = e.emitMessageStart(dst, ev)
			dst = e.emitPing(dst)
		}

	case ir.EvText:
		dst = e.ensureTextBlock(dst)
		dst = e.appendSSE(dst, "content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": e.openBlock,
			"delta": map[string]any{"type": "text_delta", "text": ev.Text},
		})

	case ir.EvReasoningStart:
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
		dst = e.appendSSE(dst, "content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": e.openBlock,
			"delta": map[string]any{"type": "thinking_delta", "thinking": ev.Text},
		})

	case ir.EvSignature:
		if ev.Signature != nil {
			dst = e.appendSSE(dst, "content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": e.openBlock,
				"delta": map[string]any{"type": "signature_delta", "signature": ev.Signature.Value},
			})
		}

	case ir.EvSealed:
		if ev.Sealed != nil {
			dst = e.closeBlock(dst)
			idx := e.blockIdx
			e.blockIdx++
			var dataStr string
			_ = json.Unmarshal(ev.Sealed.Data, &dataStr)
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
		dst = e.closeBlock(dst)
		idx := e.blockIdx
		e.blockIdx++
		e.openBlock = idx
		e.openBlockType = "tool_use"
		dst = e.appendSSE(dst, "content_block_start", map[string]any{
			"type":  "content_block_start",
			"index": idx,
			"content_block": map[string]any{
				"type":  "tool_use",
				"id":    ev.ID,
				"name":  ev.Name,
				"input": json.RawMessage(`{}`),
			},
		})

	case ir.EvToolArgs:
		dst = e.appendSSE(dst, "content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": e.openBlock,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": ev.Text},
		})

	case ir.EvToolEnd:
		dst = e.closeBlock(dst)

	case ir.EvStop:
		dst = e.closeBlock(dst)
		reason := stopToAnthropic(ev.Stop)
		seq := ""
		if ev.Stop != nil {
			seq = ev.Stop.Sequence
		}
		dst = e.appendSSE(dst, "message_delta", map[string]any{
			"type": "message_delta",
			"delta": map[string]any{
				"stop_reason":   reason,
				"stop_sequence": seqOrNull(seq),
			},
			"usage": map[string]any{"output_tokens": 0},
		})

	case ir.EvUsage:
		if ev.Usage != nil {
			dst = e.appendSSE(dst, "message_delta", map[string]any{
				"type":  "message_delta",
				"delta": map[string]any{},
				"usage": buildAnthropicUsage(*ev.Usage),
			})
		}

	case ir.EvError:
		if ev.Err != nil {
			dst = BuildStreamError(dst, ev.Err)
		}
	}
	return dst, nil
}

// Close writes the message_stop event.
func (e *StreamEncoder) Close(dst []byte) ([]byte, error) {
	dst = e.closeBlock(dst)
	dst = e.appendSSE(dst, "message_stop", map[string]any{"type": "message_stop"})
	return dst, nil
}

// Keepalive sends a ping event.
func (e *StreamEncoder) Keepalive(dst []byte) []byte {
	return e.emitPing(dst)
}

func (e *StreamEncoder) emitMessageStart(dst []byte, ev ir.Event) []byte {
	var promptTokens int
	if ev.Usage != nil {
		promptTokens = ev.Usage.PromptTokens()
	} else if e.opts.PromptEstimate > 0 {
		promptTokens = e.opts.PromptEstimate
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
		"usage": map[string]any{
			"input_tokens":  promptTokens,
			"output_tokens": 0,
		},
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

func (e *StreamEncoder) closeBlock(dst []byte) []byte {
	if e.openBlock < 0 {
		return dst
	}
	idx := e.openBlock
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
			bs, err := partToBlocks(p, fp, env)
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
	m := map[string]any{
		"input_tokens":  u.PromptTokens(),
		"output_tokens": u.OutputTokens,
	}
	if u.Has(ir.UCacheRead) {
		m["cache_read_input_tokens"] = u.CacheReadTokens
	}
	if u.Has(ir.UCacheWrite) {
		m["cache_creation_input_tokens"] = u.CacheWriteTokens
	}
	return m
}

func seqOrNull(seq string) any {
	if seq == "" {
		return nil
	}
	return seq
}
