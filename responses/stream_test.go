package responses_test

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/responses"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

func feedResponsesChunks(t *testing.T, dec *responses.StreamDecoder, raw string) []ir.Event {
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

func newResponsesDecoder() *responses.StreamDecoder {
	return responses.NewStreamDecoder(responses.DecodeOptions{Env: &ir.Env{Allow: ir.DefaultAllowed}})
}

// TestStreamDecode_Responses_HappyPath verifies a complete normal stream.
func TestStreamDecode_Responses_HappyPath(t *testing.T) {
	dec := newResponsesDecoder()

	// Minimal Responses API stream: created → text delta → item done → completed.
	raw := "event: response.created\n" +
		`data: {"type":"response.created","response":{"id":"resp1","model":"gpt-4","status":"in_progress"}}` + "\n\n" +
		"event: response.in_progress\n" +
		`data: {"type":"response.in_progress","response":{"id":"resp1","model":"gpt-4","status":"in_progress"}}` + "\n\n" +
		"event: response.output_item.added\n" +
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"item1","status":"in_progress"}}` + "\n\n" +
		"event: response.content_part.added\n" +
		`data: {"type":"response.content_part.added","item_id":"item1","output_index":0,"content_index":0,"part":{"type":"output_text","text":""}}` + "\n\n" +
		"event: response.output_text.delta\n" +
		`data: {"type":"response.output_text.delta","item_id":"item1","output_index":0,"content_index":0,"delta":"hello"}` + "\n\n" +
		"event: response.content_part.done\n" +
		`data: {"type":"response.content_part.done","item_id":"item1","output_index":0,"content_index":0,"part":{"type":"output_text","text":"hello"}}` + "\n\n" +
		"event: response.output_item.done\n" +
		`data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"item1","status":"completed"}}` + "\n\n" +
		"event: response.completed\n" +
		`data: {"type":"response.completed","response":{"id":"resp1","model":"gpt-4","status":"completed","usage":{"input_tokens":5,"output_tokens":1,"total_tokens":6}}}` + "\n\n"

	events := feedResponsesChunks(t, dec, raw)

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

	// After response.completed, End(EndDone) must return nil.
	ev := dec.End(ir.EndDone)
	if ev != nil {
		t.Errorf("End(EndDone) after response.completed should be nil, got %+v", ev)
	}
}

// TestStreamDecode_Responses_Truncated verifies End(EndEOF) on an incomplete stream.
func TestStreamDecode_Responses_Truncated(t *testing.T) {
	dec := newResponsesDecoder()
	raw := "event: response.created\n" +
		`data: {"type":"response.created","response":{"id":"resp2","model":"gpt-4","status":"in_progress"}}` + "\n\n" +
		"event: response.output_text.delta\n" +
		`data: {"type":"response.output_text.delta","item_id":"item1","output_index":0,"content_index":0,"delta":"partial"}` + "\n\n"

	feedResponsesChunks(t, dec, raw)

	ev := dec.End(ir.EndEOF)
	if ev == nil {
		t.Fatal("End(EndEOF) on truncated stream should return an error event")
	}
	if ev.Type != ir.EvError {
		t.Errorf("event type: want EvError, got %v", ev.Type)
	}
}

// TestStreamDecode_Responses_InterleavedToolCalls verifies function_call_arguments.delta.
func TestStreamDecode_Responses_InterleavedToolCalls(t *testing.T) {
	dec := newResponsesDecoder()
	raw := "event: response.created\n" +
		`data: {"type":"response.created","response":{"id":"resp3","model":"gpt-4","status":"in_progress"}}` + "\n\n" +
		"event: response.output_item.added\n" +
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc1","call_id":"call1","name":"Read","arguments":""}}` + "\n\n" +
		"event: response.function_call_arguments.delta\n" +
		`data: {"type":"response.function_call_arguments.delta","item_id":"fc1","output_index":0,"delta":"{\"path\":"}` + "\n\n" +
		"event: response.function_call_arguments.delta\n" +
		`data: {"type":"response.function_call_arguments.delta","item_id":"fc1","output_index":0,"delta":"\"/x\"}"}` + "\n\n" +
		"event: response.function_call_arguments.done\n" +
		`data: {"type":"response.function_call_arguments.done","item_id":"fc1","output_index":0,"arguments":"{\"path\":\"/x\"}"}` + "\n\n" +
		"event: response.output_item.done\n" +
		`data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc1","call_id":"call1","name":"Read","arguments":"{\"path\":\"/x\"}"}}` + "\n\n" +
		"event: response.completed\n" +
		`data: {"type":"response.completed","response":{"id":"resp3","model":"gpt-4","status":"completed","usage":{"input_tokens":5,"output_tokens":5,"total_tokens":10}}}` + "\n\n"

	events := feedResponsesChunks(t, dec, raw)

	hasToolStart, hasToolArgs := false, false
	for _, ev := range events {
		switch ev.Type {
		case ir.EvToolStart:
			hasToolStart = true
			if ev.Name != "Read" {
				t.Errorf("tool name: want Read, got %q", ev.Name)
			}
		case ir.EvToolArgs:
			hasToolArgs = true
		}
	}
	if !hasToolStart {
		t.Error("missing EvToolStart")
	}
	if !hasToolArgs {
		t.Error("missing EvToolArgs")
	}
}

// TestStreamDecode_Responses_UsageOnlyFinalChunk verifies usage in response.completed.
func TestStreamDecode_Responses_UsageOnlyFinalChunk(t *testing.T) {
	dec := newResponsesDecoder()
	raw := "event: response.created\n" +
		`data: {"type":"response.created","response":{"id":"resp4","model":"gpt-4","status":"in_progress"}}` + "\n\n" +
		"event: response.output_item.added\n" +
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"item1","status":"in_progress"}}` + "\n\n" +
		"event: response.output_text.delta\n" +
		`data: {"type":"response.output_text.delta","item_id":"item1","output_index":0,"content_index":0,"delta":"hi"}` + "\n\n" +
		"event: response.output_item.done\n" +
		`data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"item1","status":"completed"}}` + "\n\n" +
		"event: response.completed\n" +
		`data: {"type":"response.completed","response":{"id":"resp4","model":"gpt-4","status":"completed","usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12}}}` + "\n\n"

	events := feedResponsesChunks(t, dec, raw)

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

// TestStreamDecode_Responses_ErrorMidStream verifies error event handling.
func TestStreamDecode_Responses_ErrorMidStream(t *testing.T) {
	dec := newResponsesDecoder()
	raw := "event: response.created\n" +
		`data: {"type":"response.created","response":{"id":"resp5","model":"gpt-4","status":"in_progress"}}` + "\n\n" +
		"event: error\n" +
		`data: {"type":"error","error":{"type":"server_error","code":"internal_server_error","message":"Something went wrong"}}` + "\n\n"

	events := feedResponsesChunks(t, dec, raw)

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

// --- BuildResponse Responses ---

func TestBuildResponse_Responses_SimpleText(t *testing.T) {
	resp := &ir.Response{
		ID:    "resp1",
		Model: "gpt-4",
		Outputs: []ir.Output{{
			Index: 0,
			Parts: []ir.Part{ir.TextPart("hello")},
			Stop:  ir.Stop{Reason: ir.StopEndTurn},
		}},
		Usage: ir.Usage{InputTokens: 5, OutputTokens: 1, Reported: ir.UInput | ir.UOutput},
	}
	body, err := responses.BuildResponse(resp, responses.EncodeOptions{
		Env:   &ir.Env{Allow: ir.DefaultAllowed},
		Model: "gpt-4",
	})
	if err != nil {
		t.Fatalf("BuildResponse: %v", err)
	}
	var v map[string]json.RawMessage
	if err := json.Unmarshal(body, &v); err != nil {
		t.Fatalf("not valid JSON: %v\n%s", err, body)
	}
	if _, ok := v["output"]; !ok {
		t.Error("missing output field")
	}
	if _, ok := v["usage"]; !ok {
		t.Error("missing usage field")
	}
}
