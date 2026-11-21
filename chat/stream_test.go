package chat_test

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/chat"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// feedChunks parses raw SSE text and feeds each event through the decoder.
// Returns all collected ir.Events and any feed error.
func feedChatChunks(t *testing.T, dec *chat.StreamDecoder, raw string) ([]ir.Event, error) {
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
		events, err = dec.Decode(events, ev)
		if err != nil {
			return events, err
		}
	}
	return events, nil
}

func newChatDecoder() *chat.StreamDecoder {
	return chat.NewStreamDecoder(chat.DecodeOptions{Env: &ir.Env{Allow: ir.DefaultAllowed}})
}

// TestStreamDecode_Chat_HappyPath tests a complete normal stream.
func TestStreamDecode_Chat_HappyPath(t *testing.T) {
	dec := newChatDecoder()
	raw := "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":1,\"total_tokens\":6}}\n\n" +
		"data: [DONE]\n\n"

	events, err := feedChatChunks(t, dec, raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	hasStart := false
	hasText := false
	hasStop := false
	hasUsage := false
	for _, ev := range events {
		switch ev.Type {
		case ir.EvStart:
			hasStart = true
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
		t.Error("missing EvStart event")
	}
	if !hasText {
		t.Error("missing EvText event")
	}
	if !hasStop {
		t.Error("missing EvStop event")
	}
	if !hasUsage {
		t.Error("missing EvUsage event")
	}
}

// TestStreamDecode_Chat_Truncated tests End(EndEOF) on an incomplete stream.
func TestStreamDecode_Chat_Truncated(t *testing.T) {
	dec := newChatDecoder()
	raw := "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n"
	_, err := feedChatChunks(t, dec, raw)
	if err != nil {
		t.Fatalf("feed error: %v", err)
	}

	// Stream ended without [DONE] → End(EndEOF) must return an error event.
	ev := dec.End(ir.EndEOF)
	if ev == nil {
		t.Fatal("End(EndEOF) on truncated stream should return an error event, got nil")
	}
	if ev.Type != ir.EvError {
		t.Errorf("event type: want EvError, got %v", ev.Type)
	}
	if ev.Err == nil {
		t.Error("EvError event has nil Err")
	}
}

// TestStreamDecode_Chat_TruncatedTimeout tests End(EndTimeout).
func TestStreamDecode_Chat_TruncatedTimeout(t *testing.T) {
	dec := newChatDecoder()
	raw := "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n"
	_, _ = feedChatChunks(t, dec, raw)
	ev := dec.End(ir.EndTimeout)
	if ev == nil || ev.Type != ir.EvError {
		t.Error("End(EndTimeout) on incomplete stream should return EvError")
	}
	if ev.Err != nil && ev.Err.Status != 504 {
		t.Errorf("timeout error should have status 504, got %d", ev.Err.Status)
	}
}

// TestStreamDecode_Chat_InterleavedToolCalls tests multiple tool-call delta fragments.
func TestStreamDecode_Chat_InterleavedToolCalls(t *testing.T) {
	dec := newChatDecoder()

	// Tool call with arguments split across two chunks.
	raw := "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"tc1\",\"type\":\"function\",\"function\":{\"name\":\"Read\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\":\\\"\"}}]},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"/x\\\"}\"}}]},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n" +
		"data: [DONE]\n\n"

	events, err := feedChatChunks(t, dec, raw)
	if err != nil {
		t.Fatalf("feed error: %v", err)
	}

	hasToolStart := false
	hasToolArgs := false
	hasToolEnd := false
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

// TestStreamDecode_Chat_UsageOnlyFinalChunk tests a stream where the last chunk
// has no choices but carries usage (OpenAI stream_options pattern).
func TestStreamDecode_Chat_UsageOnlyFinalChunk(t *testing.T) {
	dec := newChatDecoder()
	raw := "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12}}\n\n" +
		"data: [DONE]\n\n"

	events, err := feedChatChunks(t, dec, raw)
	if err != nil {
		t.Fatalf("feed error: %v", err)
	}

	var usageEv *ir.Event
	for i := range events {
		if events[i].Type == ir.EvUsage {
			usageEv = &events[i]
		}
	}
	if usageEv == nil {
		t.Fatal("no EvUsage event found")
	}
	if usageEv.Usage.InputTokens+usageEv.Usage.OutputTokens == 0 {
		t.Error("usage event has zero tokens")
	}
}

// TestStreamDecode_Chat_ErrorMidStream tests an error object inside a chunk.
func TestStreamDecode_Chat_ErrorMidStream(t *testing.T) {
	dec := newChatDecoder()
	raw := "data: {\"error\":{\"message\":\"rate limit\",\"type\":\"rate_limit_error\"}}\n\n"

	events, err := feedChatChunks(t, dec, raw)
	if err != nil {
		t.Fatalf("unexpected feed error: %v", err)
	}

	hasError := false
	for _, ev := range events {
		if ev.Type == ir.EvError {
			hasError = true
			if ev.Err == nil {
				t.Error("EvError with nil Err")
			}
		}
	}
	if !hasError {
		t.Error("expected EvError event from error chunk")
	}
}

// TestStreamDecode_Chat_EndDone_Terminal tests that End on a completed stream returns nil.
func TestStreamDecode_Chat_EndDone_Terminal(t *testing.T) {
	dec := newChatDecoder()
	raw := "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n"
	_, _ = feedChatChunks(t, dec, raw)

	// After terminal [DONE], End(EndDone) must return nil.
	ev := dec.End(ir.EndDone)
	if ev != nil {
		t.Errorf("End(EndDone) after terminal [DONE] should return nil, got %+v", ev)
	}
}

// TestStreamDecode_Chat_ParseResponse tests non-streaming response parsing.
func TestStreamDecode_Chat_ParseResponse(t *testing.T) {
	body := []byte(`{"id":"r1","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`)
	resp, err := chat.ParseResponse(body, chat.DecodeOptions{Env: &ir.Env{Allow: ir.DefaultAllowed}})
	if err != nil {
		t.Fatalf("ParseResponse: %v", err)
	}
	if resp.ID != "r1" {
		t.Errorf("ID: want r1, got %q", resp.ID)
	}
	if len(resp.Outputs) == 0 {
		t.Fatal("no outputs")
	}
	if resp.Usage.InputTokens+resp.Usage.OutputTokens == 0 {
		t.Error("usage zero")
	}
}

// TestStreamDecode_Chat_ReasoningContent tests reasoning_content field extraction.
func TestStreamDecode_Chat_ReasoningContent(t *testing.T) {
	dec := newChatDecoder()
	raw := "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"let me think\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n"
	events, err := feedChatChunks(t, dec, raw)
	if err != nil {
		t.Fatalf("feed error: %v", err)
	}
	hasReasoning := false
	for _, ev := range events {
		if ev.Type == ir.EvReasoning {
			hasReasoning = true
		}
	}
	if !hasReasoning {
		t.Error("expected EvReasoning event from reasoning_content delta")
	}
}

// --- BuildResponse tests ---

func TestBuildResponse_Chat_SimpleText(t *testing.T) {
	resp := &ir.Response{
		ID:    "r1",
		Model: "m",
		Outputs: []ir.Output{{
			Index: 0,
			Parts: []ir.Part{ir.TextPart("hello")},
			Stop:  ir.Stop{Reason: ir.StopEndTurn},
		}},
		Usage: ir.Usage{InputTokens: 5, OutputTokens: 1, Reported: ir.UInput | ir.UOutput},
	}
	body, err := chat.BuildResponse(resp, chat.EncodeOptions{
		Env:   &ir.Env{Allow: ir.DefaultAllowed},
		Model: "m",
	})
	if err != nil {
		t.Fatalf("BuildResponse: %v", err)
	}
	var v map[string]json.RawMessage
	if err := json.Unmarshal(body, &v); err != nil {
		t.Fatalf("not valid JSON: %v\n%s", err, body)
	}
	if _, ok := v["choices"]; !ok {
		t.Error("missing choices field")
	}
	if _, ok := v["usage"]; !ok {
		t.Error("missing usage field")
	}
}

func TestBuildResponse_Chat_ToolCall(t *testing.T) {
	resp := &ir.Response{
		ID:    "r1",
		Model: "m",
		Outputs: []ir.Output{{
			Index: 0,
			Parts: []ir.Part{{
				Type: ir.PartToolCall,
				Call: &ir.ToolCall{ID: "tc1", Name: "Read", Kind: ir.ToolFunction, Arguments: json.RawMessage(`{"path":"/x"}`)},
			}},
			Stop: ir.Stop{Reason: ir.StopToolUse},
		}},
	}
	body, err := chat.BuildResponse(resp, chat.EncodeOptions{
		Env:   &ir.Env{Allow: ir.DefaultAllowed},
		Model: "m",
	})
	if err != nil {
		t.Fatalf("BuildResponse: %v", err)
	}
	var v map[string]json.RawMessage
	if err := json.Unmarshal(body, &v); err != nil {
		t.Fatalf("not valid JSON: %v", err)
	}
}
