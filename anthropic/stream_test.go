package anthropic_test

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/anthropic"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

func feedAnthropicChunks(t *testing.T, dec *anthropic.StreamDecoder, raw string) []ir.Event {
	t.Helper()
	r := sse.NewReader(bytes.NewReader([]byte(raw)))
	var events []ir.Event
	for {
		ev, err := r.Next()
		if err != nil {
			break
		}
		if ev.Name == "" && len(ev.Data) == 0 {
			continue
		}
		var err2 error
		events, err2 = dec.Decode(events, ev)
		if err2 != nil {
			t.Fatalf("Decode error: %v", err2)
		}
	}
	return events
}

func newAnthropicDecoder() *anthropic.StreamDecoder {
	return anthropic.NewStreamDecoder(anthropic.DecodeOptions{Env: &ir.Env{Allow: ir.DefaultAllowed}})
}

// TestStreamDecode_Anthropic_HappyPath verifies a complete normal stream.
func TestStreamDecode_Anthropic_HappyPath(t *testing.T) {
	dec := newAnthropicDecoder()
	raw := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg1\",\"model\":\"claude-3\",\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":1}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

	events := feedAnthropicChunks(t, dec, raw)

	hasStart, hasText, hasStop, hasUsage := false, false, false, false
	for _, ev := range events {
		switch ev.Type {
		case ir.EvStart:
			hasStart = true
			if ev.ID != "msg1" {
				t.Errorf("start ID: want msg1, got %q", ev.ID)
			}
		case ir.EvText:
			hasText = true
			if ev.Text != "hello" {
				t.Errorf("text: want hello, got %q", ev.Text)
			}
		case ir.EvStop:
			hasStop = true
		case ir.EvUsage:
			hasUsage = true
		}
	}
	if !hasStart {
		t.Error("missing EvStart")
	}
	if !hasText {
		t.Error("missing EvText")
	}
	if !hasStop {
		t.Error("missing EvStop")
	}
	if !hasUsage {
		t.Error("missing EvUsage")
	}

	// After message_stop, End(EndDone) must return nil.
	ev := dec.End(ir.EndDone)
	if ev != nil {
		t.Errorf("End(EndDone) after message_stop should be nil, got %+v", ev)
	}
}

// TestStreamDecode_Anthropic_Truncated tests End(EndEOF) on an incomplete stream.
func TestStreamDecode_Anthropic_Truncated(t *testing.T) {
	dec := newAnthropicDecoder()
	raw := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg2\",\"model\":\"m\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n"

	feedAnthropicChunks(t, dec, raw)

	ev := dec.End(ir.EndEOF)
	if ev == nil {
		t.Fatal("End(EndEOF) on truncated stream should return an error event")
	}
	if ev.Type != ir.EvError {
		t.Errorf("event type: want EvError, got %v", ev.Type)
	}
}

// TestStreamDecode_Anthropic_InterleavedToolCalls tests tool_use block.
func TestStreamDecode_Anthropic_InterleavedToolCalls(t *testing.T) {
	dec := newAnthropicDecoder()
	raw := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg3\",\"model\":\"m\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tc1\",\"name\":\"Read\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"/x\\\"}\"  }}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":5}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

	events := feedAnthropicChunks(t, dec, raw)

	hasToolStart, hasToolArgs, hasToolEnd := false, false, false
	for _, ev := range events {
		switch ev.Type {
		case ir.EvToolStart:
			hasToolStart = true
			if ev.Name != "Read" {
				t.Errorf("tool name: want Read, got %q", ev.Name)
			}
		case ir.EvToolArgs:
			hasToolArgs = true
		case ir.EvToolEnd:
			hasToolEnd = true
		}
	}
	if !hasToolStart {
		t.Error("missing EvToolStart")
	}
	if !hasToolArgs {
		t.Error("missing EvToolArgs")
	}
	if !hasToolEnd {
		t.Error("missing EvToolEnd")
	}
}

// TestStreamDecode_Anthropic_ErrorMidStream tests an error event inside the stream.
func TestStreamDecode_Anthropic_ErrorMidStream(t *testing.T) {
	dec := newAnthropicDecoder()
	raw := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg4\",\"model\":\"m\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n" +
		"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"server busy\"}}\n\n"

	events := feedAnthropicChunks(t, dec, raw)

	hasError := false
	for _, ev := range events {
		if ev.Type == ir.EvError {
			hasError = true
			if ev.Err == nil {
				t.Error("EvError has nil Err")
			}
		}
	}
	if !hasError {
		t.Error("expected EvError from error event")
	}
	// After terminal error, End(EndDone) must return nil.
	ev := dec.End(ir.EndDone)
	if ev != nil {
		t.Errorf("End(EndDone) after terminal error should be nil")
	}
}

// TestStreamDecode_Anthropic_ThinkingBlock tests redacted_thinking block.
func TestStreamDecode_Anthropic_ThinkingBlock(t *testing.T) {
	dec := newAnthropicDecoder()
	raw := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg5\",\"model\":\"m\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"reasoning text\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"abc123\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

	events := feedAnthropicChunks(t, dec, raw)

	hasReasoning := false
	hasSignature := false
	for _, ev := range events {
		switch ev.Type {
		case ir.EvReasoning:
			hasReasoning = true
		case ir.EvSignature:
			hasSignature = true
		}
	}
	if !hasReasoning {
		t.Error("missing EvReasoning event for thinking block")
	}
	if !hasSignature {
		t.Error("missing EvSignature event for thinking block")
	}
}

// TestStreamDecode_Anthropic_UsageOnlyDelta tests that message_delta usage is emitted.
func TestStreamDecode_Anthropic_UsageOnlyDelta(t *testing.T) {
	dec := newAnthropicDecoder()
	raw := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg6\",\"model\":\"m\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

	events := feedAnthropicChunks(t, dec, raw)

	outputTokensSeen := 0
	for _, ev := range events {
		if ev.Type == ir.EvUsage && ev.Usage != nil {
			outputTokensSeen += ev.Usage.OutputTokens
		}
		if ev.Type == ir.EvStart && ev.Usage != nil {
			outputTokensSeen += ev.Usage.OutputTokens
		}
	}
	if outputTokensSeen == 0 {
		t.Error("output tokens never emitted in usage events")
	}
}

// --- BuildResponse Anthropic ---

func TestBuildResponse_Anthropic_SimpleText(t *testing.T) {
	resp := &ir.Response{
		ID:    "msg1",
		Model: "claude-3",
		Outputs: []ir.Output{{
			Index: 0,
			Parts: []ir.Part{ir.TextPart("hello")},
			Stop:  ir.Stop{Reason: ir.StopEndTurn},
		}},
		Usage: ir.Usage{InputTokens: 5, OutputTokens: 1, Reported: ir.UInput | ir.UOutput},
	}
	body, err := anthropic.BuildResponse(resp, anthropic.EncodeOptions{
		Env:   &ir.Env{Allow: ir.DefaultAllowed},
		Model: "claude-3",
	})
	if err != nil {
		t.Fatalf("BuildResponse: %v", err)
	}
	var v map[string]json.RawMessage
	if err := json.Unmarshal(body, &v); err != nil {
		t.Fatalf("not valid JSON: %v\n%s", err, body)
	}
	if _, ok := v["content"]; !ok {
		t.Error("missing content field")
	}
	if _, ok := v["usage"]; !ok {
		t.Error("missing usage field")
	}
}
