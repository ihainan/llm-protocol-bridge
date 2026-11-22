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

// Reference: remaining test-only coverage items from service/responses_strict_test.go
// (:67 effort & tool_choice/parallel, :200 effort tier snapping, :569 live custom tool)
// and service/claude_strict_test.go (:264 tool_choice mapping, :294 reasoning mapping,
// :484 non-stream finish reasons, :518 thinking+text+ping order, :559 same-chunk split,
// :607 text after tools, :842 tool missing name, :972 finish/tool consistency).
func TestStrictA6_ResponsesTestOnlyCoverage(t *testing.T) {
	t.Run("effort_normalization_and_tier_snapping_table", func(t *testing.T) {
		// Reference: service/responses_strict_test.go:67 (effort 归一) and :200 (档位折算表)
		profile := &ir.ModelProfile{
			Efforts:                []ir.Effort{ir.EffortLow, ir.EffortHigh, ir.EffortMax},
			EffortInTemplateKwargs: true,
		}
		cases := []struct {
			in   string
			want string
		}{
			{"minimal", "low"},
			{"low", "low"},
			{"medium", "high"},
			{"high", "high"},
			{"xhigh", "max"},
			{"max", "max"},
		}
		for _, tc := range cases {
			body := []byte(`{"model":"m","input":"hi","reasoning":{"effort":"` + tc.in + `"}}`)
			plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, body, bridge.Options{Profile: profile})
			if err != nil {
				t.Fatalf("effort %s: %v", tc.in, err)
			}
			var out struct {
				ReasoningEffort    string         `json:"reasoning_effort"`
				ChatTemplateKwargs map[string]any `json:"chat_template_kwargs"`
			}
			_ = json.Unmarshal(plan.Body, &out)
			if out.ReasoningEffort != tc.want {
				t.Errorf("effort %s: want %q, got %q", tc.in, tc.want, out.ReasoningEffort)
			}
			if out.ChatTemplateKwargs["reasoning_effort"] != tc.want {
				t.Errorf("effort %s template kwargs: want %q, got %+v", tc.in, tc.want, out.ChatTemplateKwargs)
			}
		}
	})

	t.Run("tool_choice_auto_and_parallel_tool_calls_forwarded", func(t *testing.T) {
		// Reference: service/responses_strict_test.go:67 (tool_choice 与并行)
		raw := []byte(`{
			"model": "m",
			"input": "hi",
			"tools": [{"type": "function", "name": "exec_command", "parameters": {"type": "object"}}],
			"tool_choice": "auto",
			"parallel_tool_calls": true
		}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, raw, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		var out struct {
			ToolChoice        any   `json:"tool_choice"`
			ParallelToolCalls *bool `json:"parallel_tool_calls"`
		}
		_ = json.Unmarshal(plan.Body, &out)
		if out.ToolChoice != "auto" || out.ParallelToolCalls == nil || !*out.ParallelToolCalls {
			t.Fatalf("want tool_choice=auto and parallel_tool_calls=true, got %s", plan.Body)
		}
	})

	t.Run("custom_tool_live_streaming_fragmented_utf8_and_late_text", func(t *testing.T) {
		// Reference: service/responses_strict_test.go:569 (TestResponsesStreamCustomToolLive)
		raw := []byte(`{
			"model": "m",
			"stream": true,
			"input": "patch",
			"tools": [{"type": "custom", "name": "apply_patch"}]
		}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, raw, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		chunks := []string{
			`{"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"apply_patch","arguments":"{\"in"}}]}}]}`,
			`{"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"put\": \"*** Begin\\"}}]}}]}`,
			`{"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"n+ \\u4e2"}}]}}]}`,
			`{"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"d \\ud83d\\ude00 \\\"q\\\"\"}"}}]}}]}`,
			`{"id":"1","model":"m","choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"tool_calls"}]}`,
		}
		var rawSSE strings.Builder
		for _, c := range chunks {
			rawSSE.WriteString("data: " + c + "\n\n")
		}
		rawSSE.WriteString("data: [DONE]\n\n")

		streamBytes := feedSSE(t, plan.NewStream(), rawSSE.String())
		r := sse.NewReader(bytes.NewReader(streamBytes))
		var deltas strings.Builder
		deltaCount := 0
		var finalOut []map[string]any
		for {
			ev, err := r.Next()
			if err != nil {
				break
			}
			var obj struct {
				Type     string `json:"type"`
				Delta    string `json:"delta"`
				Response struct {
					Output []map[string]any `json:"output"`
				} `json:"response"`
			}
			if json.Unmarshal(ev.Data, &obj) != nil {
				continue
			}
			if obj.Type == "response.custom_tool_call_input.delta" {
				deltaCount++
				deltas.WriteString(obj.Delta)
			}
			if obj.Type == "response.completed" {
				finalOut = obj.Response.Output
			}
		}
		want := "*** Begin\n+ 中 😀 \"q\""
		if deltaCount < 3 || deltas.String() != want {
			t.Fatalf("custom live deltas mismatch: count=%d got=%q want=%q\n%s", deltaCount, deltas.String(), want, streamBytes)
		}
		if len(finalOut) != 2 || finalOut[0]["type"] != "custom_tool_call" || finalOut[0]["input"] != want || finalOut[1]["type"] != "message" {
			t.Fatalf("unexpected final output order/content: %+v", finalOut)
		}
	})
}

func TestStrictA6_AnthropicTestOnlyCoverage(t *testing.T) {
	t.Run("tool_choice_auto_any_none_tool_and_disable_parallel", func(t *testing.T) {
		// Reference: service/claude_strict_test.go:264 (TestStrictToolChoiceMapping)
		cases := []struct {
			name         string
			tc           string
			wantTC       string
			wantParallel *bool
		}{
			{"auto", `{"type":"auto"}`, `"auto"`, nil},
			{"any", `{"type":"any"}`, `"required"`, nil},
			{"none", `{"type":"none"}`, `"none"`, nil},
			{"tool_disable_parallel", `{"type":"tool","name":"get_weather","disable_parallel_tool_use":true}`, `{"type":"function","function":{"name":"get_weather"}}`, ir.Ptr(false)},
		}
		for _, tc := range cases {
			body := []byte(`{"model":"m","max_tokens":100,"tools":[{"name":"get_weather","input_schema":{"type":"object"}}],"tool_choice":` + tc.tc + `,"messages":[{"role":"user","content":"hi"}]}`)
			plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, body, bridge.Options{})
			if err != nil {
				t.Fatalf("%s: %v", tc.name, err)
			}
			var out struct {
				ToolChoice        json.RawMessage `json:"tool_choice"`
				ParallelToolCalls *bool           `json:"parallel_tool_calls"`
			}
			_ = json.Unmarshal(plan.Body, &out)
			if string(out.ToolChoice) != tc.wantTC {
				t.Errorf("%s: want tool_choice %s, got %s", tc.name, tc.wantTC, out.ToolChoice)
			}
			if tc.wantParallel != nil && (out.ParallelToolCalls == nil || *out.ParallelToolCalls != *tc.wantParallel) {
				t.Errorf("%s: want parallel_tool_calls=%v, got %v", tc.name, *tc.wantParallel, out.ParallelToolCalls)
			}
		}
	})

	t.Run("reasoning_budget_and_effort_mapping_via_profile", func(t *testing.T) {
		// Reference: service/claude_strict_test.go:294 (TestStrictReasoningMapping, 6 rows)
		profile := &ir.ModelProfile{
			Efforts: []ir.Effort{ir.EffortLow, ir.EffortHigh, ir.EffortMax},
			BudgetToEffort: []ir.BudgetStep{
				{Max: 4096, Effort: ir.EffortLow},
				{Max: 16384, Effort: ir.EffortHigh},
				{Max: 0, Effort: ir.EffortMax},
			},
		}
		cases := []struct {
			name string
			body string
			want string
		}{
			{"budget_2048", `{"model":"m","max_tokens":5000,"thinking":{"type":"enabled","budget_tokens":2048},"messages":[{"role":"user","content":"hi"}]}`, "low"},
			{"budget_10000", `{"model":"m","max_tokens":20000,"thinking":{"type":"enabled","budget_tokens":10000},"messages":[{"role":"user","content":"hi"}]}`, "high"},
			{"budget_31999", `{"model":"m","max_tokens":64000,"thinking":{"type":"enabled","budget_tokens":31999},"messages":[{"role":"user","content":"hi"}]}`, "max"},
			{"adaptive", `{"model":"m","max_tokens":5000,"thinking":{"type":"adaptive"},"messages":[{"role":"user","content":"hi"}]}`, ""},
			{"effort_medium", `{"model":"m","max_tokens":5000,"output_config":{"effort":"medium"},"messages":[{"role":"user","content":"hi"}]}`, "high"},
			{"effort_xhigh", `{"model":"m","max_tokens":5000,"output_config":{"effort":"xhigh"},"messages":[{"role":"user","content":"hi"}]}`, "max"},
		}
		for _, tc := range cases {
			plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, []byte(tc.body), bridge.Options{Profile: profile})
			if err != nil {
				t.Fatalf("%s: %v", tc.name, err)
			}
			var out struct {
				ReasoningEffort string `json:"reasoning_effort"`
			}
			_ = json.Unmarshal(plan.Body, &out)
			if out.ReasoningEffort != tc.want {
				t.Errorf("%s: want reasoning_effort=%q, got %q", tc.name, tc.want, out.ReasoningEffort)
			}
		}
	})

	t.Run("non_stream_finish_reason_mapping_and_empty_content_array", func(t *testing.T) {
		// Reference: service/claude_strict_test.go:484 (TestStrictNonStreamEmptyAndStopReasons)
		reqBody := []byte(`{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, reqBody, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		cases := []struct {
			finish string
			want   string
		}{
			{"stop", "end_turn"},
			{"length", "max_tokens"},
			{"content_filter", "refusal"},
			{"", "end_turn"},
			{"weird", "end_turn"},
		}
		for _, tc := range cases {
			up := []byte(`{"id":"c1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":""},"finish_reason":"` + tc.finish + `"}],"usage":{}}`)
			out, _, err := plan.ConvertResponse(up)
			if err != nil {
				t.Fatalf("finish %q: %v", tc.finish, err)
			}
			var resp struct {
				StopReason string          `json:"stop_reason"`
				Content    json.RawMessage `json:"content"`
			}
			_ = json.Unmarshal(out, &resp)
			if resp.StopReason != tc.want || string(resp.Content) != "[]" {
				t.Errorf("finish %q: want stop_reason=%q content=[], got stop_reason=%q content=%s", tc.finish, tc.want, resp.StopReason, resp.Content)
			}
		}
	})

	t.Run("stream_thinking_then_text_ping_second_msg_id_and_usage", func(t *testing.T) {
		// Reference: service/claude_strict_test.go:518 (TestStrictStreamThinkingThenText)
		reqBody := []byte(`{"model":"m","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, reqBody, bridge.Options{GenerateResponseID: true})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		rawSSE := `data: {"id":"up1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"a"}}]}` + "\n\n" +
			`data: {"id":"up1","model":"m","choices":[{"index":0,"delta":{"reasoning_content":"b"}}]}` + "\n\n" +
			`data: {"id":"up1","model":"m","choices":[{"index":0,"delta":{"content":"1591"}}]}` + "\n\n" +
			`data: {"id":"up1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":26,"completion_tokens":81,"prompt_tokens_details":{"cached_tokens":20}}}` + "\n\n" +
			"data: [DONE]\n\n"
		out := feedSSE(t, plan.NewStream(), rawSSE)
		r := sse.NewReader(bytes.NewReader(out))
		var evTypes []string
		var msgID string
		var blockTypes []string
		var deltaUsage map[string]any
		for {
			ev, err := r.Next()
			if err != nil {
				break
			}
			evTypes = append(evTypes, ev.Name)
			var obj map[string]any
			_ = json.Unmarshal(ev.Data, &obj)
			switch ev.Name {
			case "message_start":
				msg, _ := obj["message"].(map[string]any)
				msgID, _ = msg["id"].(string)
			case "content_block_start":
				cb, _ := obj["content_block"].(map[string]any)
				bt, _ := cb["type"].(string)
				blockTypes = append(blockTypes, bt)
			case "message_delta":
				deltaUsage, _ = obj["usage"].(map[string]any)
			}
		}
		if len(evTypes) < 2 || evTypes[0] != "message_start" || evTypes[1] != "ping" {
			t.Fatalf("expected [message_start, ping, ...], got %v", evTypes)
		}
		if !strings.HasPrefix(msgID, "msg_") {
			t.Fatalf("expected msg_ ID, got %q", msgID)
		}
		if strings.Join(blockTypes, ",") != "thinking,text" {
			t.Fatalf("expected block order thinking,text, got %v", blockTypes)
		}
		if deltaUsage["input_tokens"] != float64(6) || deltaUsage["cache_read_input_tokens"] != float64(20) || deltaUsage["output_tokens"] != float64(81) {
			t.Fatalf("unexpected usage in message_delta: %+v", deltaUsage)
		}
	})

	t.Run("same_chunk_reasoning_and_text_split_into_two_blocks", func(t *testing.T) {
		// Reference: service/claude_strict_test.go:559 (TestStrictStreamReasoningAndTextInSameChunk)
		reqBody := []byte(`{"model":"m","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, reqBody, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		rawSSE := `data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"reasoning_content":"think","content":"answer"},"finish_reason":"stop"}]}` + "\n\n" +
			"data: [DONE]\n\n"
		out := string(feedSSE(t, plan.NewStream(), rawSSE))
		parsed := parseAnthropicSSE(t, out)
		if parsed.Errors != 0 || !parsed.MessageStop || len(parsed.Blocks) != 2 ||
			parsed.Blocks[0].Type != "thinking" || parsed.Blocks[1].Type != "text" {
			t.Fatalf("expected [thinking, text] blocks from same chunk, got %+v\n%s", parsed.Blocks, out)
		}
	})

	t.Run("text_after_tool_call_produces_clean_blocks", func(t *testing.T) {
		// Reference: service/claude_strict_test.go:607 (TestStrictStreamTextAfterToolsClosesTools)
		reqBody := []byte(`{"model":"m","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"ls","input_schema":{"type":"object"}}]}`)
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, reqBody, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		rawSSE := `data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"ls","arguments":"{}"}}]}}]}` + "\n\n" +
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"content":"after"}}]}` + "\n\n" +
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n" +
			"data: [DONE]\n\n"
		out := string(feedSSE(t, plan.NewStream(), rawSSE))
		parsed := parseAnthropicSSE(t, out)
		if parsed.Errors != 0 || !parsed.MessageStop || len(parsed.Blocks) != 2 {
			t.Fatalf("expected 2 clean blocks without overlap, got blocks=%+v errors=%d\n%s", parsed.Blocks, parsed.Errors, out)
		}
	})

	t.Run("tool_call_without_name_returns_stream_error", func(t *testing.T) {
		// Reference: service/claude_strict_test.go:842 (TestStrictStreamToolWithoutNameIsError)
		reqBody := []byte(`{"model":"m","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"ls","input_schema":{"type":"object"}}]}`)
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, reqBody, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		rawSSE := `data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]},"finish_reason":"tool_calls"}]}` + "\n\n" +
			"data: [DONE]\n\n"
		out := string(feedSSE(t, plan.NewStream(), rawSSE))
		parsed := parseAnthropicSSE(t, out)
		if parsed.Errors == 0 || len(parsed.Blocks) != 0 {
			t.Fatalf("expected stream error with 0 content blocks, got errors=%d blocks=%+v\n%s", parsed.Errors, parsed.Blocks, out)
		}
	})

	t.Run("finish_reason_tool_calls_without_calls_and_stop_with_calls", func(t *testing.T) {
		// Reference: service/claude_strict_test.go:972 (tool_calls 无调用 -> end_turn; stop 有调用 -> tool_use)
		reqBody := []byte(`{"model":"m","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"ls","input_schema":{"type":"object"}}]}`)
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, reqBody, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}

		sse1 := `data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"tool_calls"}]}` + "\n\n" + "data: [DONE]\n\n"
		out1 := string(feedSSE(t, plan.NewStream(), sse1))
		if !strings.Contains(out1, `"stop_reason":"end_turn"`) {
			t.Fatalf("finish tool_calls without calls must map to end_turn:\n%s", out1)
		}

		sse2 := `data: {"id":"2","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"ls","arguments":"{}"}}]},"finish_reason":"stop"}]}` + "\n\n" + "data: [DONE]\n\n"
		out2 := string(feedSSE(t, plan.NewStream(), sse2))
		if !strings.Contains(out2, `"stop_reason":"tool_use"`) {
			t.Fatalf("finish stop with tool call must map to tool_use:\n%s", out2)
		}
	})
}
