package bridge_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/anthropic"
	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/chat"
	"github.com/ihainan/llm-protocol-bridge/gemini"
	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/responses"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

func feedSSE(t *testing.T, s *bridge.Stream, raw string) []byte {
	t.Helper()
	r := sse.NewReader(bytes.NewReader([]byte(raw)))
	var out []byte
	for {
		ev, err := r.Next()
		if err != nil {
			break
		}
		if ev.Name == "" && len(ev.Data) == 0 {
			continue
		}
		out, err = s.Feed(out, ev)
		if err != nil {
			t.Fatalf("Feed: %v", err)
		}
	}
	var err error
	out, err = s.End(out, ir.EndEOF)
	if err != nil {
		t.Fatalf("End: %v", err)
	}
	return out
}

// Item 1 & 2: Responses SSE events must use standard "response.*" names,
// emit output_index starting at 0, delay response.completed until after usage,
// populate final response.output, emit response.incomplete on max_tokens,
// and handle text + tool_call without prematurely stopping on message item done.
func TestStage2_ResponsesStreamProtocol(t *testing.T) {
	t.Run("decode_standard_response_prefix_text_and_tool", func(t *testing.T) {
		dec := responses.NewStreamDecoder(responses.DecodeOptions{})
		raw := "event: response.created\n" +
			`data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_1","model":"gpt-5","status":"in_progress"}}` + "\n\n" +
			"event: response.output_item.added\n" +
			`data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"message","id":"msg_1","status":"in_progress","role":"assistant","content":[]}}` + "\n\n" +
			"event: response.output_text.delta\n" +
			`data: {"type":"response.output_text.delta","sequence_number":2,"output_index":0,"content_index":0,"delta":"checking..."}` + "\n\n" +
			"event: response.output_item.done\n" +
			`data: {"type":"response.output_item.done","sequence_number":3,"output_index":0,"item":{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"checking..."}]}}` + "\n\n" +
			"event: response.output_item.added\n" +
			`data: {"type":"response.output_item.added","sequence_number":4,"output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"ping","arguments":""}}` + "\n\n" +
			"event: response.output_item.done\n" +
			`data: {"type":"response.output_item.done","sequence_number":5,"output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"ping","arguments":"{}"}}` + "\n\n" +
			"event: response.completed\n" +
			`data: {"type":"response.completed","sequence_number":6,"response":{"id":"resp_1","model":"gpt-5","status":"completed","usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}` + "\n\n"

		r := sse.NewReader(bytes.NewReader([]byte(raw)))
		c := ir.NewCollector()
		stopCount := 0
		for {
			ev, err := r.Next()
			if err != nil {
				break
			}
			var irEvs []ir.Event
			irEvs, err = dec.Decode(irEvs, ev)
			if err != nil {
				t.Fatalf("Decode: %v", err)
			}
			for _, e := range irEvs {
				if e.Type == ir.EvStop {
					stopCount++
				}
				c.Add(e)
			}
		}
		if stopCount != 1 {
			t.Fatalf("expected exactly 1 EvStop at response.completed, got %d", stopCount)
		}
		resp := c.Response()
		if len(resp.Outputs) != 1 || len(resp.Outputs[0].Parts) != 2 {
			t.Fatalf("expected 2 parts (text + zero-arg tool_call), got %+v", resp.Outputs)
		}
		if resp.Outputs[0].Parts[0].Text != "checking..." {
			t.Errorf("text: got %q", resp.Outputs[0].Parts[0].Text)
		}
		if resp.Outputs[0].Parts[1].Type != ir.PartToolCall || resp.Outputs[0].Parts[1].Call.Name != "ping" {
			t.Errorf("second part: got %+v", resp.Outputs[0].Parts[1])
		}
		if resp.Outputs[0].Stop.Reason != ir.StopToolUse {
			t.Errorf("stop reason: want tool_use, got %q", resp.Outputs[0].Stop.Reason)
		}
	})

	t.Run("encode_uses_response_prefix_zero_index_usage_and_output", func(t *testing.T) {
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat,
			[]byte(`{"model":"gpt-5","input":"hi","stream":true}`), bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		chatSSE := "data: {\"id\":\"c1\",\"model\":\"gpt-5\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"},\"finish_reason\":null}]}\n\n" +
			"data: {\"id\":\"c1\",\"model\":\"gpt-5\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
			"data: {\"id\":\"c1\",\"model\":\"gpt-5\",\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3,\"total_tokens\":15}}\n\n" +
			"data: [DONE]\n\n"

		out := feedSSE(t, plan.NewStream(), chatSSE)
		r := sse.NewReader(bytes.NewReader(out))
		var evNames []string
		var completed map[string]any
		firstOutputIdx := -1
		for {
			ev, err := r.Next()
			if err != nil {
				break
			}
			evNames = append(evNames, ev.Name)
			var m map[string]any
			if err := json.Unmarshal(ev.Data, &m); err != nil {
				t.Fatalf("invalid json for %s: %v", ev.Name, err)
			}
			if !strings.HasPrefix(ev.Name, "response.") {
				t.Errorf("event name %q missing 'response.' prefix", ev.Name)
			}
			if ev.Name == "response.output_item.added" && firstOutputIdx == -1 {
				firstOutputIdx = int(m["output_index"].(float64))
			}
			if ev.Name == "response.completed" {
				completed = m
			}
		}
		if firstOutputIdx != 0 {
			t.Errorf("first output_index: want 0, got %d", firstOutputIdx)
		}
		if completed == nil {
			t.Fatalf("missing response.completed; events=%v", evNames)
		}
		respObj := completed["response"].(map[string]any)
		if respObj["usage"] == nil {
			t.Errorf("response.completed missing usage: %+v", respObj)
		}
		outputs, _ := respObj["output"].([]any)
		if len(outputs) == 0 {
			t.Errorf("response.completed has empty output array: %+v", respObj)
		}
	})

	t.Run("encode_incomplete_event_on_max_tokens", func(t *testing.T) {
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat,
			[]byte(`{"model":"gpt-5","input":"hi","stream":true}`), bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		chatSSE := "data: {\"id\":\"c1\",\"model\":\"gpt-5\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"cut\"},\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":1,\"total_tokens\":6}}\n\n" +
			"data: [DONE]\n\n"
		out := feedSSE(t, plan.NewStream(), chatSSE)
		if !strings.Contains(string(out), "event: response.incomplete\n") {
			t.Errorf("expected response.incomplete event, got:\n%s", out)
		}
		if strings.Contains(string(out), "event: response.completed\n") {
			t.Errorf("must not emit response.completed on max_tokens:\n%s", out)
		}
	})
}

// Item 6: Gemini systemInstruction Content object and generationConfig field order.
func TestStage2_GeminiParseSystemAndSchemaOrder(t *testing.T) {
	body := []byte(`{
		"systemInstruction": {"parts": [{"text": "be concise"}]},
		"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
		"generationConfig": {
			"responseSchema": {"type": "OBJECT", "properties": {"a": {"type": "STRING"}}},
			"responseMimeType": "application/json"
		}
	}`)
	req, err := gemini.ParseRequest(body, gemini.ParseOptions{Model: "gemini-2.5-pro"})
	if err != nil {
		t.Fatalf("ParseRequest failed on standard systemInstruction: %v", err)
	}
	if len(req.Instructions) != 1 || req.Instructions[0].Text != "be concise" {
		t.Fatalf("Instructions: got %+v", req.Instructions)
	}
	if req.Output == nil || req.Output.Type != ir.OutputJSONSchema || len(req.Output.Schema) == 0 {
		t.Fatalf("Output schema lost when responseSchema precedes responseMimeType: %+v", req.Output)
	}
}

// Item 7: Chat buildReasoning pointer + preserve JSON Schema keywords like additionalProperties.
func TestStage2_ChatBuildReasoningAndSchema(t *testing.T) {
	req := &ir.Request{
		Model:     "gpt-5",
		Messages:  []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("hi")}}},
		Reasoning: &ir.ReasoningConfig{Mode: ir.ReasoningOn, Effort: ir.EffortHigh},
		Tools: []ir.Tool{{
			Name:   "lookup",
			Kind:   ir.ToolFunction,
			Strict: ir.Ptr(true),
			Schema: json.RawMessage(`{"type":"object","properties":{"id":{"type":"string"}},"required":["id"],"additionalProperties":false}`),
		}},
	}
	built, err := chat.BuildRequest(req, chat.BuildOptions{})
	if err != nil {
		t.Fatalf("BuildRequest: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(built.Body, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if m["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort lost: body=%s", built.Body)
	}
	if !strings.Contains(string(built.Body), `"additionalProperties":false`) {
		t.Errorf("additionalProperties stripped from Chat tool schema: body=%s", built.Body)
	}
}

// Item 13: Chat error response, Gemini 0 candidates, finish_reason EOF without [DONE],
// wire error types, and Close after error.
func TestStage2_ErrorsAndTerminalStates(t *testing.T) {
	t.Run("chat_parse_response_error_envelope", func(t *testing.T) {
		_, err := chat.ParseResponse([]byte(`{"error":{"message":"bad key","type":"authentication_error"}}`), chat.DecodeOptions{})
		if err == nil {
			t.Fatal("expected error for Chat error envelope, got nil")
		}
	})

	t.Run("chat_decoder_finish_without_done_and_done_without_finish", func(t *testing.T) {
		// 1. finish_reason arrived, then EOF without [DONE] => normal completion.
		d1 := chat.NewStreamDecoder(chat.DecodeOptions{})
		var evs []ir.Event
		var err error
		evs, err = d1.Decode(evs, sse.Event{Data: []byte(`{"id":"1","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}`)})
		if err != nil {
			t.Fatalf("d1.Decode error: %v", err)
		}
		var sawText, sawStop bool
		for _, ev := range evs {
			if ev.Type == ir.EvText && ev.Text == "hi" {
				sawText = true
			}
			if ev.Type == ir.EvStop && ev.Stop.Reason == ir.StopEndTurn {
				sawStop = true
			}
		}
		if !sawText || !sawStop {
			t.Errorf("d1.Decode expected EvText(\"hi\") and EvStop(end_turn), got %+v", evs)
		}
		if ev := d1.End(ir.EndEOF); ev != nil {
			t.Errorf("finish_reason + EOF without [DONE] should be nil, got %+v", ev)
		}

		// 2. [DONE] arrived WITHOUT any finish_reason => truncated error on EOF.
		d2 := chat.NewStreamDecoder(chat.DecodeOptions{})
		evs = nil
		evs, err = d2.Decode(evs, sse.Event{Data: []byte(`{"id":"1","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}`)})
		if err != nil {
			t.Fatalf("d2.Decode chunk error: %v", err)
		}
		evs, err = d2.Decode(evs, sse.Event{Data: []byte(`[DONE]`)})
		if err != nil {
			t.Fatalf("d2.Decode [DONE] error: %v", err)
		}
		sawText, sawStop = false, false
		for _, ev := range evs {
			if ev.Type == ir.EvText && ev.Text == "partial" {
				sawText = true
			}
			if ev.Type == ir.EvStop {
				sawStop = true
			}
		}
		if !sawText || sawStop {
			t.Errorf("d2.Decode expected EvText(\"partial\") and no EvStop, got %+v", evs)
		}
		if ev := d2.End(ir.EndEOF); ev == nil || ev.Type != ir.EvError {
			t.Errorf("[DONE] without finish_reason should return EvError on EndEOF, got %+v", ev)
		}
	})

	t.Run("gemini_zero_candidates_is_error", func(t *testing.T) {
		_, err := gemini.ParseResponse([]byte(`{"candidates":[],"promptFeedback":{"blockReason":"SAFETY"}}`), gemini.DecodeOptions{})
		if err == nil {
			t.Fatal("expected error for Gemini 0 candidates, got nil")
		}
	})

	t.Run("openai_error_wire_type_roundtrip", func(t *testing.T) {
		status, body := chat.BuildError(&ir.APIError{Status: 429, Type: ir.ErrRateLimit, Message: "slow down"})
		if status != 429 {
			t.Fatalf("status: want 429, got %d", status)
		}
		if !strings.Contains(string(body), `"rate_limit_error"`) {
			t.Errorf("expected OpenAI wire type rate_limit_error, got %s", body)
		}
		parsed := chat.ParseError(status, body)
		if parsed.Type != ir.ErrRateLimit {
			t.Errorf("round-trip error type: want rate_limit, got %q", parsed.Type)
		}
	})

	t.Run("anthropic_close_after_error_no_message_stop", func(t *testing.T) {
		enc := anthropic.NewStreamEncoder(anthropic.EncodeOptions{Env: &ir.Env{}, Model: "m"})
		var dst []byte
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStart, Model: "m"})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvError, Err: &ir.APIError{Status: 500, Type: ir.ErrServer, Message: "boom"}})
		dst, _ = enc.Close(dst)
		if strings.Contains(string(dst), "message_stop") {
			t.Errorf("Close after EvError must not emit message_stop:\n%s", dst)
		}
	})
}

// Item 14: jsonx.Obj.key escapes special characters and chat Passthrough stays identity-only.
func TestStage2_JSONKeyEscapingAndPassthroughScope(t *testing.T) {
	o := jsonx.Start(nil)
	o.Str("model", "gpt-5")
	o.Raw(`x":0,"model`, []byte(`"evil"`))
	raw := o.End()
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("escaped key produced invalid JSON: %v\n%s", err, raw)
	}
	if m["model"] != "gpt-5" {
		t.Errorf("key injection overwrote model: %+v", m)
	}
}

// Item 15: Multi-text-block response converted to Chat non-streaming must have string content.
func TestStage2_ChatBuildResponseMultiTextIsString(t *testing.T) {
	resp := &ir.Response{
		ID:    "r1",
		Model: "m",
		Outputs: []ir.Output{{
			Index: 0,
			Parts: []ir.Part{ir.TextPart("part1 "), ir.TextPart("part2")},
			Stop:  ir.Stop{Reason: ir.StopEndTurn},
		}},
	}
	body, err := chat.BuildResponse(resp, chat.EncodeOptions{Env: &ir.Env{}, Model: "m"})
	if err != nil {
		t.Fatalf("BuildResponse: %v", err)
	}
	var parsed struct {
		Choices []struct {
			Message struct {
				Content any `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	s, ok := parsed.Choices[0].Message.Content.(string)
	if !ok || s != "part1 part2" {
		t.Errorf("message.content must be joined string 'part1 part2', got %T (%v)", parsed.Choices[0].Message.Content, parsed.Choices[0].Message.Content)
	}
}

// P2 #1: Collector.Response() must not corrupt open tool args or hang on huge candidate index.
func TestStage2_CollectorSnapshotAndHugeIndex(t *testing.T) {
	c := ir.NewCollector()
	c.Add(ir.Event{Type: ir.EvToolStart, Index: 0, ID: "c1", Name: "f"})
	c.Add(ir.Event{Type: ir.EvToolArgs, Index: 0, Text: `{"a":`})
	_ = c.Response() // mid-stream snapshot must not close open call
	c.Add(ir.Event{Type: ir.EvToolArgs, Index: 0, Text: `1}`})
	c.Add(ir.Event{Type: ir.EvToolEnd, Index: 0})
	resp := c.Response()
	if string(resp.Outputs[0].Parts[0].Call.Arguments) != `{"a":1}` {
		t.Errorf("mid-stream snapshot broke tool args: got %s", resp.Outputs[0].Parts[0].Call.Arguments)
	}

	c2 := ir.NewCollector()
	c2.Add(ir.Event{Type: ir.EvText, Index: 100000000, Text: "hi"})
	r2 := c2.Response()
	if len(r2.Outputs) != 1 || r2.Outputs[0].Index != 100000000 || r2.Outputs[0].Parts[0].Text != "hi" {
		t.Errorf("huge candidate index: got %+v", r2.Outputs)
	}
}
