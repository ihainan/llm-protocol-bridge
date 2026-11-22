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
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/responses"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// Item 3: Parallel / interleaved tool calls & multi-candidate isolation.
func TestStage4A_InterleavedParallelToolCalls(t *testing.T) {
	// Chat upstream sends interleaved deltas for tool_calls[0] and tool_calls[1].
	chatSSE := "" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"Read","arguments":"{\"path\":"}}]}}]}` + "\n\n" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"Write","arguments":"{\"file\":"}}]}}]}` + "\n\n" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"/a\"}"}}]}}]}` + "\n\n" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"/b\"}"}}]}}]}` + "\n\n" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n" +
		"data: [DONE]\n\n"

	t.Run("collector_preserves_both_interleaved_calls", func(t *testing.T) {
		dec := chat.NewStreamDecoder(chat.DecodeOptions{})
		col := ir.NewCollector()
		r := sse.NewReader(bytes.NewReader([]byte(chatSSE)))
		for {
			ev, err := r.Next()
			if err != nil {
				break
			}
			evs, err := dec.Decode(nil, ev)
			if err != nil {
				t.Fatalf("Decode: %v", err)
			}
			for _, e := range evs {
				col.Add(e)
			}
		}
		resp := col.Response()
		if len(resp.Outputs) != 1 || len(resp.Outputs[0].Parts) != 2 {
			t.Fatalf("want 2 tool call parts, got %+v", resp.Outputs)
		}
		c0 := resp.Outputs[0].Parts[0].Call
		c1 := resp.Outputs[0].Parts[1].Call
		if c0.ID != "call_a" || c0.Name != "Read" || string(c0.Arguments) != `{"path":"/a"}` {
			t.Errorf("call 0 corrupted: %+v (%s)", c0, string(c0.Arguments))
		}
		if c1.ID != "call_b" || c1.Name != "Write" || string(c1.Arguments) != `{"file":"/b"}` {
			t.Errorf("call 1 corrupted: %+v (%s)", c1, string(c1.Arguments))
		}
	})

	for _, clientFmt := range []ir.Format{ir.FormatChat, ir.FormatAnthropic, ir.FormatResponses, ir.FormatGemini} {
		clientFmt := clientFmt
		t.Run("chat_interleaved_to_"+string(clientFmt), func(t *testing.T) {
			var reqBody []byte
			switch clientFmt {
			case ir.FormatChat:
				reqBody = []byte(`{"model":"m","messages":[{"role":"user","content":"go"}],"stream":true}`)
			case ir.FormatAnthropic:
				reqBody = []byte(`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"go"}],"stream":true}`)
			case ir.FormatResponses:
				reqBody = []byte(`{"model":"m","input":"go","stream":true}`)
			case ir.FormatGemini:
				reqBody = []byte(`{"contents":[{"role":"user","parts":[{"text":"go"}]}]}`)
			}
			plan, err := bridge.Convert(clientFmt, ir.FormatChat, reqBody, bridge.Options{})
			if err != nil {
				t.Fatalf("Convert: %v", err)
			}
			s := plan.NewStream()
			r := sse.NewReader(bytes.NewReader([]byte(chatSSE)))
			var out []byte
			for {
				ev, err := r.Next()
				if err != nil {
					break
				}
				out, err = s.Feed(out, ev)
				if err != nil {
					t.Fatalf("Feed: %v", err)
				}
			}
			out, err = s.End(out, ir.EndDone)
			if err != nil {
				t.Fatalf("End: %v", err)
			}

			// Re-decode the client output with the client format's decoder (or JSON array for Gemini).
			col := ir.NewCollector()
			switch clientFmt {
			case ir.FormatChat:
				d := chat.NewStreamDecoder(chat.DecodeOptions{})
				rr := sse.NewReader(bytes.NewReader(out))
				for {
					ev, err := rr.Next()
					if err != nil {
						break
					}
					evs, _ := d.Decode(nil, ev)
					for _, e := range evs {
						col.Add(e)
					}
				}
			case ir.FormatAnthropic:
				d := anthropic.NewStreamDecoder(anthropic.DecodeOptions{})
				rr := sse.NewReader(bytes.NewReader(out))
				for {
					ev, err := rr.Next()
					if err != nil {
						break
					}
					evs, _ := d.Decode(nil, ev)
					for _, e := range evs {
						col.Add(e)
					}
				}
			case ir.FormatResponses:
				d := responses.NewStreamDecoder(responses.DecodeOptions{})
				rr := sse.NewReader(bytes.NewReader(out))
				for {
					ev, err := rr.Next()
					if err != nil {
						break
					}
					evs, _ := d.Decode(nil, ev)
					for _, e := range evs {
						col.Add(e)
					}
				}
			case ir.FormatGemini:
				// Default AltSSE=false produces a valid JSON array of chunks.
				var chunks []json.RawMessage
				if err := json.Unmarshal(out, &chunks); err != nil {
					t.Fatalf("Gemini AltSSE=false output is not valid JSON array: %v\n%s", err, out)
				}
				d := gemini.NewStreamDecoder(gemini.DecodeOptions{})
				for _, ch := range chunks {
					evs, _ := d.Decode(nil, sse.Event{Data: ch})
					for _, e := range evs {
						col.Add(e)
					}
				}
			}

			resp := col.Response()
			if len(resp.Outputs) != 1 || len(resp.Outputs[0].Parts) != 2 {
				t.Fatalf("%s: want 2 tool calls, got %+v\nraw:\n%s", clientFmt, resp.Outputs, out)
			}
			p0 := resp.Outputs[0].Parts[0].Call
			p1 := resp.Outputs[0].Parts[1].Call
			if p0.Name != "Read" || string(p0.Arguments) != `{"path":"/a"}` {
				t.Errorf("%s call 0 mismatch: %+v (%s)", clientFmt, p0, string(p0.Arguments))
			}
			if p1.Name != "Write" || string(p1.Arguments) != `{"file":"/b"}` {
				t.Errorf("%s call 1 mismatch: %+v (%s)", clientFmt, p1, string(p1.Arguments))
			}
		})
	}

	t.Run("chat_multi_candidate_isolation", func(t *testing.T) {
		multiSSE := "" +
			`data: {"id":"c2","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"c0","type":"function","function":{"name":"A","arguments":"{\"k\":1}"}}]}},{"index":1,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"B","arguments":"{\"k\":2}"}}]}}]}` + "\n\n" +
			`data: {"id":"c2","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n" +
			`data: {"id":"c2","model":"m","choices":[{"index":1,"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n" +
			"data: [DONE]\n\n"
		dec := chat.NewStreamDecoder(chat.DecodeOptions{})
		col := ir.NewCollector()
		r := sse.NewReader(bytes.NewReader([]byte(multiSSE)))
		for {
			ev, err := r.Next()
			if err != nil {
				break
			}
			evs, _ := dec.Decode(nil, ev)
			for _, e := range evs {
				col.Add(e)
			}
		}
		resp := col.Response()
		if len(resp.Outputs) != 2 {
			t.Fatalf("want 2 candidates, got %d", len(resp.Outputs))
		}
		if len(resp.Outputs[0].Parts) != 1 || resp.Outputs[0].Parts[0].Call.Name != "A" {
			t.Errorf("candidate 0 wrong: %+v", resp.Outputs[0].Parts)
		}
		if len(resp.Outputs[1].Parts) != 1 || resp.Outputs[1].Parts[0].Call.Name != "B" {
			t.Errorf("candidate 1 wrong: %+v", resp.Outputs[1].Parts)
		}
	})
}

// Item 4: Gemini StreamEncoder incremental emission, tool args accumulation, JSON array stream.
func TestStage4A_GeminiStreamEncoder(t *testing.T) {
	t.Run("incremental_text_not_buffered_until_stop", func(t *testing.T) {
		enc := gemini.NewStreamEncoder(gemini.EncodeOptions{Env: &ir.Env{}, Model: "gemini-2.5", AltSSE: true})
		var dst []byte
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStart, Model: "gemini-2.5"})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvText, Index: 0, Text: "hello "})
		if len(dst) == 0 || !strings.Contains(string(dst), "hello ") {
			t.Fatalf("expected immediate chunk for EvText, got %q", string(dst))
		}
		before := len(dst)
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvText, Index: 0, Text: "world"})
		if len(dst) == before {
			t.Fatalf("expected second incremental chunk for EvText")
		}
	})

	t.Run("tool_args_accumulated_across_fragments_and_valid_json_array", func(t *testing.T) {
		enc := gemini.NewStreamEncoder(gemini.EncodeOptions{Env: &ir.Env{}, Model: "gemini-2.5", AltSSE: false})
		var dst []byte
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStart, Model: "gemini-2.5"})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvToolStart, Index: 0, ID: "c1", Name: "Read", Kind: ir.ToolFunction})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvToolArgs, Index: 0, ID: "c1", Text: `{"path":`})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvToolArgs, Index: 0, ID: "c1", Text: `"/tmp/x"}`})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvToolEnd, Index: 0, ID: "c1"})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopToolUse}})
		dst, _ = enc.Close(dst)

		var arr []map[string]any
		if err := json.Unmarshal(dst, &arr); err != nil {
			t.Fatalf("AltSSE=false output is not valid JSON array: %v\n%s", err, dst)
		}
		if !strings.Contains(string(dst), `"/tmp/x"`) || !strings.Contains(string(dst), `"path"`) {
			t.Errorf("accumulated args missing: %s", dst)
		}
	})
}

// Item 10: Tool calls & tool results cross-format rules.
func TestStage4A_ToolCallAndResultCrossFormat(t *testing.T) {
	t.Run("anthropic_merges_consecutive_user_messages_and_encodes_ids", func(t *testing.T) {
		chatReq := []byte(`{
			"model": "claude-4",
			"messages": [
				{"role": "user", "content": "run"},
				{"role": "assistant", "tool_calls": [{"id": "call.1/a:b", "type": "function", "function": {"name": "Read", "arguments": "{\"p\":\"/x\"}"}}]},
				{"role": "tool", "tool_call_id": "call.1/a:b", "content": "ok"},
				{"role": "user", "content": "next"}
			]
		}`)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic, chatReq, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		var body struct {
			Messages []struct {
				Role    string           `json:"role"`
				Content []map[string]any `json:"content"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(plan.Body, &body); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if len(body.Messages) != 3 {
			t.Fatalf("expected 3 alternating messages (merged user), got %d: %s", len(body.Messages), plan.Body)
		}
		if body.Messages[2].Role != "user" || len(body.Messages[2].Content) != 2 {
			t.Fatalf("expected merged user message with tool_result + text, got %+v", body.Messages[2])
		}
		// Encoded ID must match ^[a-zA-Z0-9_-]+$ and decode back via AnthropicRawID.
		tuID, _ := body.Messages[1].Content[0]["id"].(string)
		trID, _ := body.Messages[2].Content[0]["tool_use_id"].(string)
		if tuID != trID || tuID == "call.1/a:b" {
			t.Errorf("expected AnthropicSafeID encoding, got tuID=%q trID=%q", tuID, trID)
		}
		if got := ir.AnthropicRawID(tuID); got != "call.1/a:b" {
			t.Errorf("AnthropicRawID(%q) = %q, want call.1/a:b", tuID, got)
		}
	})

	t.Run("anthropic_tool_choice_none", func(t *testing.T) {
		chatReq := []byte(`{"model":"claude-4","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"tool_choice":"none"}`)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic, chatReq, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		var body struct {
			ToolChoice map[string]any `json:"tool_choice"`
		}
		_ = json.Unmarshal(plan.Body, &body)
		if body.ToolChoice["type"] != "none" {
			t.Errorf("want tool_choice.type=none, got %+v", body.ToolChoice)
		}
	})

	t.Run("gemini_resolves_functionResponse_name_and_no_double_wrap", func(t *testing.T) {
		anthReq := []byte(`{
			"model": "gemini-2.5",
			"max_tokens": 64,
			"messages": [
				{"role": "user", "content": "hi"},
				{"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "GetWeather", "input": {"city": "NYC"}}]},
				{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": [{"type":"text","text":"sunny "},{"type":"text","text":"72F"}]}]}
			]
		}`)
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatGemini, anthReq, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		var body struct {
			Contents []struct {
				Parts []struct {
					FunctionResponse *struct {
						Name     string         `json:"name"`
						Response map[string]any `json:"response"`
					} `json:"functionResponse"`
				} `json:"parts"`
			} `json:"contents"`
		}
		if err := json.Unmarshal(plan.Body, &body); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		fr := body.Contents[2].Parts[0].FunctionResponse
		if fr == nil || fr.Name != "GetWeather" {
			t.Fatalf("expected functionResponse.name=GetWeather, got %+v", fr)
		}
		if fr.Response["output"] != "sunny 72F" {
			t.Errorf("expected concatenated text 'sunny 72F', got %+v", fr.Response)
		}

		// Round-trip Gemini -> Gemini must not double-wrap functionResponse.response.
		plan2, err := bridge.Convert(ir.FormatGemini, ir.FormatGemini, plan.Body, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert round-trip: %v", err)
		}
		var body2 struct {
			Contents []struct {
				Parts []struct {
					FunctionResponse *struct {
						Response map[string]any `json:"response"`
					} `json:"functionResponse"`
				} `json:"parts"`
			} `json:"contents"`
		}
		_ = json.Unmarshal(plan2.Body, &body2)
		if body2.Contents[2].Parts[0].FunctionResponse.Response["output"] != "sunny 72F" {
			t.Errorf("Gemini round-trip double-wrapped response: %s", plan2.Body)
		}
	})

	t.Run("gemini_non_streaming_generates_deterministic_call_id", func(t *testing.T) {
		raw := []byte(`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"functionCall":{"name":"Read","args":{"path":"/x"}}}]},"finishReason":"STOP"}]}`)
		r1, err := gemini.ParseResponse(raw, gemini.DecodeOptions{})
		if err != nil {
			t.Fatalf("ParseResponse: %v", err)
		}
		r2, _ := gemini.ParseResponse(raw, gemini.DecodeOptions{})
		id1 := r1.Outputs[0].Parts[0].Call.ID
		id2 := r2.Outputs[0].Parts[0].Call.ID
		if id1 == "" || id1 != id2 {
			t.Errorf("expected non-empty deterministic call ID, got id1=%q id2=%q", id1, id2)
		}
	})

	t.Run("chat_tool_result_images_and_error_prefix_and_array_parse", func(t *testing.T) {
		anthReq := []byte(`{
			"model": "gpt-4o",
			"max_tokens": 64,
			"messages": [
				{"role": "user", "content": "snap"},
				{"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "Snap", "input": {}}]},
				{"role": "user", "content": [{
					"type": "tool_result",
					"tool_use_id": "tu_1",
					"is_error": true,
					"content": [
						{"type": "text", "text": "partial capture"},
						{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
					]
				}]}
			]
		}`)
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, anthReq, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		var body struct {
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(plan.Body, &body); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if len(body.Messages) != 4 {
			t.Fatalf("expected 4 messages (user, assistant, tool, follow-up user with image), got %d: %s", len(body.Messages), plan.Body)
		}
		var toolStr string
		_ = json.Unmarshal(body.Messages[2].Content, &toolStr)
		if !strings.HasPrefix(toolStr, "[tool error] ") || !strings.Contains(toolStr, "[image: attached in the next message]") {
			t.Errorf("unexpected tool content: %q", toolStr)
		}
		if body.Messages[3].Role != "user" || !strings.Contains(string(body.Messages[3].Content), "[images returned by tool call tu_1]") {
			t.Errorf("expected follow-up user message with lifted image, got %+v", body.Messages[3])
		}

		// Chat parser must accept role:tool with array content.
		chatWithArrTool := []byte(`{"model":"gpt-4o","messages":[{"role":"tool","tool_call_id":"c1","content":[{"type":"text","text":"hello from array"}]}]}`)
		parsed, err := chat.ParseRequest(chatWithArrTool, chat.ParseOptions{})
		if err != nil {
			t.Fatalf("chat.ParseRequest array tool content: %v", err)
		}
		if len(parsed.Messages) != 1 || len(parsed.Messages[0].Parts) != 1 || len(parsed.Messages[0].Parts[0].Result.Content) != 1 || parsed.Messages[0].Parts[0].Result.Content[0].Text != "hello from array" {
			t.Errorf("chat tool array content dropped: %+v", parsed.Messages)
		}
	})

	t.Run("chat_custom_tool_and_namespace_flattening_roundtrip", func(t *testing.T) {
		req := &ir.Request{
			Model: "gpt-4o",
			Messages: []ir.Message{{
				Role:  ir.RoleUser,
				Parts: []ir.Part{ir.TextPart("run")},
			}},
			Tools: []ir.Tool{
				{Name: "exec", Namespace: "shell", Kind: ir.ToolCustom, Description: "Run shell"},
			},
		}
		built, err := chat.BuildRequest(req, chat.BuildOptions{})
		if err != nil {
			t.Fatalf("chat.BuildRequest with ToolCustom+Namespace: %v", err)
		}
		if !strings.Contains(string(built.Body), `"shell__exec"`) {
			t.Errorf("expected flattened name shell__exec, got: %s", built.Body)
		}
		if ref, ok := built.Names["shell__exec"]; !ok || ref.Name != "exec" || ref.Namespace != "shell" || ref.Kind != ir.ToolCustom {
			t.Errorf("Names map missing restoration entry: %+v", built.Names)
		}
		// Decoding Chat response restores Name, Namespace, Kind.
		respRaw := []byte(`{"id":"1","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"shell__exec","arguments":"{\"input\":\"ls -la\"}"}}]},"finish_reason":"tool_calls"}]}`)
		resp, err := chat.ParseResponse(respRaw, chat.DecodeOptions{Names: built.Names})
		if err != nil {
			t.Fatalf("chat.ParseResponse: %v", err)
		}
		call := resp.Outputs[0].Parts[0].Call
		if call.Name != "exec" || call.Namespace != "shell" || call.Kind != ir.ToolCustom || call.Input != "ls -la" {
			t.Errorf("failed to restore custom tool+namespace: %+v", call)
		}
	})

	t.Run("chat_stream_empty_args_emits_empty_object", func(t *testing.T) {
		enc := chat.NewStreamEncoder(chat.EncodeOptions{Env: &ir.Env{}, Model: "m"})
		var dst []byte
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStart, ID: "1", Model: "m"})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvToolStart, Index: 0, ID: "c1", Name: "Ping", Kind: ir.ToolFunction})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvToolEnd, Index: 0, ID: "c1"})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopToolUse}})
		dst, _ = enc.Close(dst)

		dec := chat.NewStreamDecoder(chat.DecodeOptions{})
		col := ir.NewCollector()
		rr := sse.NewReader(bytes.NewReader(dst))
		for {
			ev, err := rr.Next()
			if err != nil {
				break
			}
			evs, _ := dec.Decode(nil, ev)
			for _, e := range evs {
				col.Add(e)
			}
		}
		call := col.Response().Outputs[0].Parts[0].Call
		if string(call.Arguments) != "{}" {
			t.Errorf("empty args tool call in Chat stream: got %q, want {}", string(call.Arguments))
		}
		if !strings.Contains(string(dst), `"{}"`) {
			t.Errorf("Chat stream did not emit \"{}\" arguments delta: %s", dst)
		}
	})
}
