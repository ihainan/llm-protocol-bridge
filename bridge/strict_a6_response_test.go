package bridge_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// Reference: service/responses_strict_test.go:340, :395, :466, :505, :626, :653
// and service/claude_strict_test.go:430, :866.
func TestStrictA6_ResponsesResponseAlignments(t *testing.T) {
	t.Run("reasoning_text_vs_summary_text_and_raw_reasoning_as_summary_flag", func(t *testing.T) {
		// Reference: service/responses_strict_test.go:340 & :653 (未请求 summary 与 RawReasoningAsSummary=false)
		// 1. Request without reasoning.summary -> reasoning_text + summary:[].
		reqNoSum := []byte(`{"model":"m","input":"hi","stream":true}`)
		p1, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, reqNoSum, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert no summary: %v", err)
		}
		rawSSE := `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"think"}}]}` + "\n\n" +
			`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}` + "\n\n" +
			"data: [DONE]\n\n"
		out1 := string(feedSSE(t, p1.NewStream(), rawSSE))
		if !strings.Contains(out1, "event: response.reasoning_text.delta") {
			t.Fatalf("expected reasoning_text.delta when summary not requested:\n%s", out1)
		}
		if strings.Contains(out1, "response.reasoning_summary_text.delta") {
			t.Fatalf("unexpected reasoning_summary_text.delta when summary not requested:\n%s", out1)
		}
		if !strings.Contains(out1, `"summary":[]`) || !strings.Contains(out1, `"type":"reasoning_text"`) {
			t.Fatalf("expected summary:[] and reasoning_text content item:\n%s", out1)
		}

		// 2. Request WITH reasoning.summary:"auto" and default RawReasoningAsSummary (nil=true) -> summary_text.
		reqWithSum := []byte(`{"model":"m","input":"hi","reasoning":{"effort":"medium","summary":"auto"},"stream":true}`)
		p2, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, reqWithSum, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert with summary: %v", err)
		}
		out2 := string(feedSSE(t, p2.NewStream(), rawSSE))
		if !strings.Contains(out2, "event: response.reasoning_summary_text.delta") || !strings.Contains(out2, `"type":"summary_text"`) {
			t.Fatalf("expected summary_text when summary requested and RawReasoningAsSummary default true:\n%s", out2)
		}

		// 3. Request WITH reasoning.summary:"auto" and RawReasoningAsSummary=false -> reasoning_text.
		p3, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, reqWithSum, bridge.Options{
			RawReasoningAsSummary: ir.Ptr(false),
		})
		if err != nil {
			t.Fatalf("Convert with RawReasoningAsSummary=false: %v", err)
		}
		out3 := string(feedSSE(t, p3.NewStream(), rawSSE))
		if !strings.Contains(out3, "event: response.reasoning_text.delta") || strings.Contains(out3, "response.reasoning_summary_text.delta") {
			t.Fatalf("expected reasoning_text when RawReasoningAsSummary=false:\n%s", out3)
		}
	})

	t.Run("non_stream_matches_stream_output_structure", func(t *testing.T) {
		// Reference: service/responses_strict_test.go:466 (TestResponsesNonStreamMatchesStream)
		reqBody := []byte(`{
			"model": "m",
			"store": false,
			"prompt_cache_key": "k1",
			"reasoning": {"effort": "medium", "summary": "auto"},
			"tools": [
				{"type": "function", "name": "exec_command", "parameters": {"type": "object"}},
				{"type": "namespace", "name": "multi_agent_v1", "tools": [{"type": "function", "name": "spawn_agent", "parameters": {"type": "object"}}]}
			],
			"input": "go"
		}`)
		opts := bridge.Options{RawReasoningAsSummary: ir.Ptr(false)}
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, reqBody, opts)
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		upBody := []byte(`{
			"id": "chatcmpl-1",
			"object": "chat.completion",
			"model": "m",
			"choices": [{
				"index": 0,
				"message": {
					"role": "assistant",
					"reasoning_content": "think",
					"content": "run",
					"tool_calls": [
						{"id": "c1", "type": "function", "function": {"name": "exec_command", "arguments": "{\"a\":1}"}},
						{"id": "c2", "type": "function", "function": {"name": "multi_agent_v1__spawn_agent", "arguments": "{\"b\":2}"}}
					]
				},
				"finish_reason": "tool_calls"
			}],
			"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
		}`)
		nonStreamBytes, _, err := plan.ConvertResponse(upBody)
		if err != nil {
			t.Fatalf("ConvertResponse: %v", err)
		}
		rawSSE := `data: {"id":"chatcmpl-1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"think"}}]}` + "\n\n" +
			`data: {"id":"chatcmpl-1","model":"m","choices":[{"index":0,"delta":{"content":"run"}}]}` + "\n\n" +
			`data: {"id":"chatcmpl-1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"exec_command","arguments":"{\"a\":1}"}}]}}]}` + "\n\n" +
			`data: {"id":"chatcmpl-1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"c2","type":"function","function":{"name":"multi_agent_v1__spawn_agent","arguments":"{\"b\":2}"}}]}}]}` + "\n\n" +
			`data: {"id":"chatcmpl-1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}` + "\n\n" +
			"data: [DONE]\n\n"
		streamBytes := feedSSE(t, plan.NewStream(), rawSSE)

		// Extract terminal response.completed object from stream.
		var streamResp map[string]any
		r := sse.NewReader(bytes.NewReader(streamBytes))
		for {
			ev, err := r.Next()
			if err != nil {
				break
			}
			var obj struct {
				Type     string         `json:"type"`
				Response map[string]any `json:"response"`
			}
			if json.Unmarshal(ev.Data, &obj) == nil && obj.Type == "response.completed" {
				streamResp = obj.Response
			}
		}
		if streamResp == nil {
			t.Fatalf("missing response.completed in stream:\n%s", streamBytes)
		}

		var nonStreamResp map[string]any
		_ = json.Unmarshal(nonStreamBytes, &nonStreamResp)

		// Verify store and prompt_cache_key echoed in both (:67 rs.Base 回显).
		if nonStreamResp["store"] != false || nonStreamResp["prompt_cache_key"] != "k1" {
			t.Fatalf("non-stream missing echo fields: %s", nonStreamBytes)
		}
		if streamResp["store"] != false || streamResp["prompt_cache_key"] != "k1" {
			t.Fatalf("stream missing echo fields: %+v", streamResp)
		}

		normalize := func(items any) string {
			arr, _ := items.([]any)
			for _, it := range arr {
				if m, ok := it.(map[string]any); ok {
					delete(m, "id")
					delete(m, "encrypted_content")
				}
			}
			b, _ := json.Marshal(arr)
			return string(b)
		}
		gotNS := normalize(nonStreamResp["output"])
		gotS := normalize(streamResp["output"])
		if gotNS != gotS {
			t.Fatalf("stream vs non-stream output mismatch:\nnon-stream: %s\nstream:     %s", gotNS, gotS)
		}
	})

	t.Run("leading_newlines_stripped_and_newline_only_before_tool_omitted", func(t *testing.T) {
		// Reference: service/responses_strict_test.go:505 (TestResponsesLeadingNewlines)
		reqBody := []byte(`{"model":"m","input":"hi","tools":[{"type":"function","name":"exec_command","parameters":{"type":"object"}}]}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, reqBody, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}

		// 1. Stream leading newlines across chunks stripped; interior \n preserved.
		sse1 := `data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"reasoning_content":"r"}}]}` + "\n\n" +
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"content":"\n\n"}}]}` + "\n\n" +
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"content":"\nHi\n"}}]}` + "\n\n" +
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"content":"there"},"finish_reason":"stop"}]}` + "\n\n" +
			"data: [DONE]\n\n"
		out1 := string(feedSSE(t, plan.NewStream(), sse1))
		if !strings.Contains(out1, `"text":"Hi\nthere"`) {
			t.Fatalf("expected stripped text \"Hi\\nthere\", got:\n%s", out1)
		}

		// 2. Stream newline-only before tool call emits NO message item.
		sse2 := `data: {"id":"2","model":"m","choices":[{"index":0,"delta":{"content":"\n"}}]}` + "\n\n" +
			`data: {"id":"2","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"exec_command","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}` + "\n\n" +
			"data: [DONE]\n\n"
		out2 := string(feedSSE(t, plan.NewStream(), sse2))
		if strings.Contains(out2, `"type":"message"`) {
			t.Fatalf("newline-only text before tool call must not open message item:\n%s", out2)
		}

		// 3. Non-stream leading newlines stripped.
		nsBody := []byte(`{"id":"3","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"\n\nHi"},"finish_reason":"stop"}]}`)
		nsOut, _, err := plan.ConvertResponse(nsBody)
		if err != nil {
			t.Fatalf("ConvertResponse: %v", err)
		}
		if !strings.Contains(string(nsOut), `"text":"Hi"`) {
			t.Fatalf("expected non-stream stripped text \"Hi\", got: %s", nsOut)
		}
	})

	t.Run("duplicate_upstream_call_ids_deduplicated_stream_and_non_stream", func(t *testing.T) {
		// Reference: service/responses_strict_test.go:626 (TestResponsesDuplicateCallIDs)
		reqBody := []byte(`{"model":"m","input":"hi","tools":[{"type":"function","name":"exec_command","parameters":{"type":"object"}}]}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, reqBody, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}

		// Stream
		rawSSE := `data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"dup","type":"function","function":{"name":"exec_command","arguments":"{}"}}]}}]}` + "\n\n" +
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"dup","type":"function","function":{"name":"exec_command","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}` + "\n\n" +
			"data: [DONE]\n\n"
		streamBytes := feedSSE(t, plan.NewStream(), rawSSE)
		var comp struct {
			Response struct {
				Output []struct {
					Type   string `json:"type"`
					CallID string `json:"call_id"`
				} `json:"output"`
			} `json:"response"`
		}
		r := sse.NewReader(bytes.NewReader(streamBytes))
		for {
			ev, err := r.Next()
			if err != nil {
				break
			}
			if ev.Name == "response.completed" {
				_ = json.Unmarshal(ev.Data, &comp)
			}
		}
		if len(comp.Response.Output) != 2 || comp.Response.Output[0].CallID == comp.Response.Output[1].CallID {
			t.Fatalf("stream duplicate call IDs not deduplicated: %+v", comp.Response.Output)
		}

		// Non-stream
		upBody := []byte(`{"id":"1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"dup","type":"function","function":{"name":"exec_command","arguments":"{}"}},{"id":"dup","type":"function","function":{"name":"exec_command","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`)
		nsOut, _, err := plan.ConvertResponse(upBody)
		if err != nil {
			t.Fatalf("ConvertResponse: %v", err)
		}
		var ns struct {
			Output []struct {
				CallID string `json:"call_id"`
			} `json:"output"`
		}
		_ = json.Unmarshal(nsOut, &ns)
		if len(ns.Output) != 2 || ns.Output[0].CallID == ns.Output[1].CallID {
			t.Fatalf("non-stream duplicate call IDs not deduplicated: %+v", ns.Output)
		}
	})

	t.Run("truncated_live_call_omits_output_item_done", func(t *testing.T) {
		// Reference: service/responses_strict_test.go:395 (length 截断的调用不发 output_item.done)
		reqBody := []byte(`{"model":"m","stream":true,"input":"run","tools":[{"type":"function","name":"exec_command","parameters":{"type":"object"}}]}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, reqBody, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		rawSSE := `data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"content":"x"}}]}` + "\n\n" +
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"rm"}}]},"finish_reason":"length"}]}` + "\n\n" +
			"data: [DONE]\n\n"
		streamBytes := feedSSE(t, plan.NewStream(), rawSSE)
		r := sse.NewReader(bytes.NewReader(streamBytes))
		var incOut []map[string]any
		sawIncomplete := false
		for {
			ev, err := r.Next()
			if err != nil {
				break
			}
			var obj struct {
				Type     string         `json:"type"`
				Item     map[string]any `json:"item"`
				Response struct {
					Status            string           `json:"status"`
					IncompleteDetails map[string]any   `json:"incomplete_details"`
					Output            []map[string]any `json:"output"`
				} `json:"response"`
			}
			if json.Unmarshal(ev.Data, &obj) != nil {
				continue
			}
			if obj.Type == "response.output_item.done" && obj.Item["type"] == "function_call" {
				t.Fatalf("truncated live function_call must NOT emit response.output_item.done:\n%s", streamBytes)
			}
			if obj.Type == "response.incomplete" {
				sawIncomplete = true
				incOut = obj.Response.Output
			}
		}
		if !sawIncomplete || len(incOut) != 2 || incOut[1]["status"] != "incomplete" {
			t.Fatalf("expected response.incomplete with 2 output items and incomplete function_call, got saw=%v out=%+v", sawIncomplete, incOut)
		}
	})
}

// Reference: service/claude_strict_test.go:430 (msg_ ID & thinking signature "") and :866 (tool name prefix/repeat merge).
func TestStrictA6_AnthropicResponseAlignments(t *testing.T) {
	t.Run("generate_msg_id_and_empty_thinking_signature_field", func(t *testing.T) {
		// Reference: service/claude_strict_test.go:430
		reqBody := []byte(`{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, reqBody, bridge.Options{
			GenerateResponseID: true,
		})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		upBody := []byte(`{
			"id": "req-1",
			"object": "chat.completion",
			"model": "m",
			"choices": [{"index": 0, "message": {"role": "assistant", "reasoning_content": "think", "content": "ok"}, "finish_reason": "stop"}],
			"usage": {"prompt_tokens": 36, "completion_tokens": 64, "total_tokens": 100}
		}`)
		out, _, err := plan.ConvertResponse(upBody)
		if err != nil {
			t.Fatalf("ConvertResponse: %v", err)
		}
		var resp struct {
			ID      string           `json:"id"`
			Content []map[string]any `json:"content"`
		}
		if err := json.Unmarshal(out, &resp); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if !strings.HasPrefix(resp.ID, "msg_") {
			t.Fatalf("expected msg_ response ID with GenerateResponseID=true, got %q", resp.ID)
		}
		if len(resp.Content) != 2 || resp.Content[0]["type"] != "thinking" {
			t.Fatalf("unexpected content: %s", out)
		}
		sig, hasSig := resp.Content[0]["signature"]
		if !hasSig || sig != "" {
			t.Fatalf("thinking block must include signature field (empty string when unsigned), got %+v", resp.Content[0])
		}
	})

	t.Run("chat_stream_tool_name_prefix_then_full_and_repeated_full", func(t *testing.T) {
		// Reference: service/claude_strict_test.go:866 (前缀后再给全名 & 全名重复)
		reqBody := []byte(`{"model":"m","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"go"}],"tools":[{"name":"get_weather","input_schema":{"type":"object"}}]}`)
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, reqBody, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}

		cases := []struct {
			name   string
			first  string
			second string
		}{
			{"prefix_then_full", "get_", "get_weather"},
			{"repeated_full", "get_weather", "get_weather"},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				rawSSE := `data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"` + tc.first + `"}}]}}]}` + "\n\n" +
					`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"` + tc.second + `","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}` + "\n\n" +
					"data: [DONE]\n\n"
				out := string(feedSSE(t, plan.NewStream(), rawSSE))
				parsed := parseAnthropicSSE(t, out)
				if parsed.Errors != 0 || len(parsed.Blocks) != 1 || parsed.Blocks[0].Name != "get_weather" {
					t.Fatalf("%s: want tool name \"get_weather\", got blocks=%+v errors=%d\n%s", tc.name, parsed.Blocks, parsed.Errors, out)
				}
			})
		}
	})
}
