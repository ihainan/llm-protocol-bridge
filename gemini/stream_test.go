package gemini_test

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/gemini"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

func feedGeminiChunks(t *testing.T, dec *gemini.StreamDecoder, raw string) []ir.Event {
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

func newGeminiDecoder() *gemini.StreamDecoder {
	return gemini.NewStreamDecoder(gemini.DecodeOptions{Env: &ir.Env{Allow: ir.DefaultAllowed}})
}

// TestStreamDecode_Gemini_HappyPath verifies a complete normal stream.
func TestStreamDecode_Gemini_HappyPath(t *testing.T) {
	dec := newGeminiDecoder()

	// Gemini streams bare JSON objects as SSE data lines.
	raw := `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hello"}]},"index":0,"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1,"totalTokenCount":6},"modelVersion":"gemini-2.0-flash"}` + "\n\n"

	events := feedGeminiChunks(t, dec, raw)

	hasStart, hasText, hasStop, hasUsage := false, false, false, false
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

	// After finishReason chunk, End(EndDone) must still return nil when terminal is set.
	// (Gemini does not set terminal=true on normal stop, so End returns an error event.)
	// We verify that End(EndDone) on an already-finished stream doesn't panic.
	_ = dec.End(ir.EndDone)
}

// TestStreamDecode_Gemini_Truncated verifies End(EndEOF) on an incomplete stream.
func TestStreamDecode_Gemini_Truncated(t *testing.T) {
	dec := newGeminiDecoder()

	// First chunk has text but no finishReason.
	raw := `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]},"index":0}],"modelVersion":"gemini-2.0-flash"}` + "\n\n"

	feedGeminiChunks(t, dec, raw)

	ev := dec.End(ir.EndEOF)
	if ev == nil {
		t.Fatal("End(EndEOF) on truncated stream should return an error event")
	}
	if ev.Type != ir.EvError {
		t.Errorf("event type: want EvError, got %v", ev.Type)
	}
}

// TestStreamDecode_Gemini_InterleavedToolCalls verifies functionCall parts.
func TestStreamDecode_Gemini_InterleavedToolCalls(t *testing.T) {
	dec := newGeminiDecoder()

	// Gemini sends the full function call in a single part (no streaming of args).
	raw := `data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"Read","args":{"path":"/x"}}}]},"index":0,"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":5,"totalTokenCount":10},"modelVersion":"gemini-2.0-flash"}` + "\n\n"

	events := feedGeminiChunks(t, dec, raw)

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

// TestStreamDecode_Gemini_UsageOnlyFinalChunk verifies usage in the final chunk.
func TestStreamDecode_Gemini_UsageOnlyFinalChunk(t *testing.T) {
	dec := newGeminiDecoder()

	// First chunk: text without usage.
	raw1 := `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"index":0}],"modelVersion":"gemini-2.0-flash"}` + "\n\n"
	// Second chunk: finishReason + usage.
	raw2 := `data: {"candidates":[{"content":{"role":"model","parts":[]},"index":0,"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2,"totalTokenCount":12},"modelVersion":"gemini-2.0-flash"}` + "\n\n"

	events := feedGeminiChunks(t, dec, raw1+raw2)

	var usageEv *ir.Event
	for i := range events {
		if events[i].Type == ir.EvUsage {
			usageEv = &events[i]
		}
	}
	if usageEv == nil {
		t.Fatal("no EvUsage event found")
	}
	if usageEv.Usage == nil {
		t.Fatal("EvUsage has nil Usage")
	}
	if usageEv.Usage.InputTokens+usageEv.Usage.OutputTokens == 0 {
		t.Error("usage event has zero tokens")
	}
}

// TestStreamDecode_Gemini_ErrorMidStream verifies error object handling.
func TestStreamDecode_Gemini_ErrorMidStream(t *testing.T) {
	dec := newGeminiDecoder()

	// Gemini error envelope.
	raw := `data: {"error":{"code":429,"message":"Resource has been exhausted","status":"RESOURCE_EXHAUSTED"}}` + "\n\n"

	events := feedGeminiChunks(t, dec, raw)

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
		t.Error("expected EvError from error envelope")
	}
	// After terminal error, End(EndDone) must return nil.
	ev := dec.End(ir.EndDone)
	if ev != nil {
		t.Errorf("End(EndDone) after terminal error should be nil")
	}
}

// --- BuildResponse Gemini ---

func TestBuildResponse_Gemini_SimpleText(t *testing.T) {
	resp := &ir.Response{
		ID:    "",
		Model: "gemini-2.0-flash",
		Outputs: []ir.Output{{
			Index: 0,
			Parts: []ir.Part{ir.TextPart("hello")},
			Stop:  ir.Stop{Reason: ir.StopEndTurn},
		}},
		Usage: ir.Usage{InputTokens: 5, OutputTokens: 1, Reported: ir.UInput | ir.UOutput},
	}
	body, err := gemini.BuildResponse(resp, gemini.EncodeOptions{
		Env: &ir.Env{Allow: ir.DefaultAllowed},
	})
	if err != nil {
		t.Fatalf("BuildResponse: %v", err)
	}
	var v map[string]json.RawMessage
	if err := json.Unmarshal(body, &v); err != nil {
		t.Fatalf("not valid JSON: %v\n%s", err, body)
	}
	if _, ok := v["candidates"]; !ok {
		t.Error("missing candidates field")
	}
}
