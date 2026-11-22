package bridge_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/chat"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// Reference: service/responses_strict_test.go:67 (system join with \n\n),
// :134 (tool_choice dropped without tools), :200 (mid-conversation developer fold),
// :691 (penalties forwarding & range check).
func TestStrictA6_ResponsesRequestBuildAlignments(t *testing.T) {
	t.Run("instructions_and_developer_joined_with_double_newline", func(t *testing.T) {
		// Reference: service/responses_strict_test.go:67 (system 拼接)
		raw := []byte(`{
			"model": "m",
			"instructions": "You are Codex.",
			"input": [
				{"role": "developer", "content": [{"type": "input_text", "text": "sandbox: workspace-write"}]},
				{"role": "user", "content": "hi"}
			]
		}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, raw, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		var out struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(plan.Body, &out); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		want := "You are Codex.\n\nsandbox: workspace-write"
		if len(out.Messages) < 2 || out.Messages[0].Role != "system" || out.Messages[0].Content != want {
			t.Fatalf("want system %q, got %+v", want, out.Messages)
		}
	})

	t.Run("tool_choice_auto_and_none_dropped_without_tools", func(t *testing.T) {
		// Reference: service/responses_strict_test.go:134 (TestResponsesStrictToolChoiceWithoutTools)
		for _, tc := range []string{
			`{"model":"m","input":"hi","tool_choice":"auto"}`,
			`{"model":"m","input":"hi","tools":[],"tool_choice":"none"}`,
		} {
			plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, []byte(tc), bridge.Options{})
			if err != nil {
				t.Fatalf("Convert %s: %v", tc, err)
			}
			var out map[string]any
			if err := json.Unmarshal(plan.Body, &out); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if _, exists := out["tool_choice"]; exists {
				t.Fatalf("tool_choice must be omitted without tools for %s, got: %s", tc, plan.Body)
			}
		}
	})

	t.Run("mid_conversation_developer_fold_cases", func(t *testing.T) {
		// Reference: service/responses_strict_test.go:200 (中途 developer 折叠 第一例 & 第二例)
		opts := bridge.Options{Chat: chat.Dialect{MidSystem: "fold"}}

		// Case 1: user, assistant, developer, user -> developer prepended to next user.
		raw1 := []byte(`{
			"model": "qwen",
			"input": [
				{"role": "user", "content": "a"},
				{"role": "assistant", "content": "b"},
				{"role": "developer", "content": "note"},
				{"role": "user", "content": "c"}
			]
		}`)
		plan1, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, raw1, opts)
		if err != nil {
			t.Fatalf("Case 1 Convert: %v", err)
		}
		var out1 struct {
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		_ = json.Unmarshal(plan1.Body, &out1)
		if len(out1.Messages) != 3 || out1.Messages[0].Role != "user" || out1.Messages[1].Role != "assistant" || out1.Messages[2].Role != "user" {
			t.Fatalf("Case 1 unexpected roles: %s", plan1.Body)
		}
		var parts1 []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		if err := json.Unmarshal(out1.Messages[2].Content, &parts1); err != nil || len(parts1) != 2 {
			t.Fatalf("Case 1 expected 2 parts on msg[2], got %s", out1.Messages[2].Content)
		}
		if parts1[0].Text != "<system_message>\nnote\n</system_message>" || parts1[1].Text != "c" {
			t.Fatalf("Case 1 parts mismatch: %+v", parts1)
		}

		// Case 2: user U1, developer D, assistant A1, user U2 -> developer appended to U1.
		raw2 := []byte(`{
			"model": "qwen",
			"input": [
				{"role": "user", "content": "U1"},
				{"role": "developer", "content": "D"},
				{"role": "assistant", "content": "A1"},
				{"role": "user", "content": "U2"}
			]
		}`)
		plan2, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, raw2, opts)
		if err != nil {
			t.Fatalf("Case 2 Convert: %v", err)
		}
		var out2 struct {
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		_ = json.Unmarshal(plan2.Body, &out2)
		if len(out2.Messages) != 3 || out2.Messages[0].Role != "user" || out2.Messages[1].Role != "assistant" || out2.Messages[2].Role != "user" {
			t.Fatalf("Case 2 unexpected roles: %s", plan2.Body)
		}
		var parts2 []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		if err := json.Unmarshal(out2.Messages[0].Content, &parts2); err != nil || len(parts2) != 2 {
			t.Fatalf("Case 2 expected 2 parts on msg[0], got %s", out2.Messages[0].Content)
		}
		if parts2[0].Text != "U1" || parts2[1].Text != "<system_message>\nD\n</system_message>" {
			t.Fatalf("Case 2 parts mismatch: %+v", parts2)
		}
	})

	t.Run("penalties_forwarded_and_omitted_when_unset", func(t *testing.T) {
		// Reference: service/responses_strict_test.go:691 (TestResponsesPenalties)
		raw := []byte(`{"model":"m","input":"hi","frequency_penalty":0.5,"presence_penalty":-1}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, raw, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		var out map[string]any
		_ = json.Unmarshal(plan.Body, &out)
		if out["frequency_penalty"] != 0.5 || out["presence_penalty"] != -1.0 {
			t.Fatalf("penalties not forwarded: %s", plan.Body)
		}
		raw2 := []byte(`{"model":"m","input":"hi"}`)
		plan2, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, raw2, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert unset: %v", err)
		}
		var out2 map[string]any
		_ = json.Unmarshal(plan2.Body, &out2)
		if _, ok := out2["frequency_penalty"]; ok {
			t.Fatalf("unset frequency_penalty invented: %s", plan2.Body)
		}
		if _, ok := out2["presence_penalty"]; ok {
			t.Fatalf("unset presence_penalty invented: %s", plan2.Body)
		}
	})
}

// Reference: service/responses_strict_test.go:150 (TestResponsesStrictRejects) and :691 (penalties range).
func TestStrictA6_ResponsesStrictRejects(t *testing.T) {
	cases := []struct {
		name     string
		body     string
		wantPath string
	}{
		{"hosted_file_search", `{"model":"m","input":"hi","tools":[{"type":"file_search"}]}`, "tools[0].type"},
		{"live_web_search", `{"model":"m","input":"hi","tools":[{"type":"web_search"}]}`, "tools"},
		{"bad_effort", `{"model":"m","input":"hi","reasoning":{"effort":"huge"}}`, "reasoning.effort"},
		{"duplicate_tool", `{"model":"m","input":"hi","tools":[{"type":"function","name":"a"},{"type":"function","name":"a"}]}`, "tools[1].name"},
		{"orphan_output", `{"model":"m","input":[{"type":"function_call_output","call_id":"x","output":"o"}]}`, "input[0].call_id"},
		{"input_file", `{"model":"m","input":[{"role":"user","content":[{"type":"input_file","file_id":"f"}]}]}`, "type"},
		{"image_file_id", `{"model":"m","input":[{"role":"user","content":[{"type":"input_image","file_id":"f"}]}]}`, "file_id"},
		{"allowed_required_empty", `{"model":"m","input":"hi","tools":[{"type":"function","name":"a"}],"tool_choice":{"type":"allowed_tools","mode":"required","tools":[]}}`, "tool_choice.tools"},
		{"required_no_tools", `{"model":"m","input":"hi","tool_choice":"required"}`, "tool_choice"},
		{"tool_choice_unknown", `{"model":"m","input":"hi","tools":[{"type":"function","name":"a"}],"tool_choice":{"type":"function","name":"b"}}`, "tool_choice.name"},
		{"developer_only_no_messages", `{"model":"m","input":[{"role":"developer","content":"x"}]}`, "input"},
		{"temperature_out_of_range", `{"model":"m","input":"hi","temperature":3}`, "temperature"},
		{"top_logprobs_positive", `{"model":"m","input":"hi","top_logprobs":2}`, "top_logprobs"},
		{"logprobs_true", `{"model":"m","input":"hi","logprobs":true}`, "logprobs"},
		{"duplicate_call_id", `{"model":"m","input":[{"role":"user","content":"hi"},{"type":"function_call","call_id":"c","name":"f","arguments":"{}"},{"type":"function_call","call_id":"c","name":"f","arguments":"{}"}]}`, "input[2].call_id"},
		{"output_kind_mismatch_custom_to_func", `{"model":"m","input":[{"role":"user","content":"hi"},{"type":"custom_tool_call","call_id":"c","name":"p","input":"x"},{"type":"function_call_output","call_id":"c","output":"o"}]}`, "input[2].type"},
		{"output_kind_mismatch_func_to_custom", `{"model":"m","input":[{"role":"user","content":"hi"},{"type":"function_call","call_id":"c","name":"f","arguments":"{}"},{"type":"custom_tool_call_output","call_id":"c","output":"o"}]}`, "input[2].type"},
		{"output_twice", `{"model":"m","input":[{"role":"user","content":"hi"},{"type":"function_call","call_id":"c","name":"f","arguments":"{}"},{"type":"function_call_output","call_id":"c","output":"1"},{"type":"function_call_output","call_id":"c","output":"2"}]}`, "input[3].call_id"},
		{"arguments_object", `{"model":"m","input":[{"role":"user","content":"hi"},{"type":"function_call","call_id":"c","name":"f","arguments":{}}]}`, "input[1].arguments"},
		{"custom_input_object", `{"model":"m","input":[{"role":"user","content":"hi"},{"type":"custom_tool_call","call_id":"c","name":"p","input":{}}]}`, "input[1].input"},
		{"frequency_penalty_out_of_range", `{"model":"m","input":"hi","frequency_penalty":99}`, "frequency_penalty"},
		{"presence_penalty_out_of_range", `{"model":"m","input":"hi","presence_penalty":-2.5}`, "presence_penalty"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, []byte(tc.body), bridge.Options{})
			if err == nil {
				t.Fatalf("expected error for %s, got nil", tc.name)
			}
			if !strings.Contains(err.Error(), tc.wantPath) {
				t.Errorf("%s: want path %q in error %q", tc.name, tc.wantPath, err.Error())
			}
		})
	}

	t.Run("path_based_reject_include_truncation_context_management_while_allowing_prompt_cache_key", func(t *testing.T) {
		// Reference: service/responses_strict_test.go:150 (include logprobs, unknown include, context_management, truncation auto).
		allowedIncludes := map[string]bool{
			"reasoning.encrypted_content":    true,
			"message.input_image.image_url":  true,
			"web_search_call.action.sources": true,
			"web_search_call.results":        true,
		}
		strictCheck := func(l ir.Loss) error {
			switch l.Kind {
			case ir.LossRequestInclude:
				if !allowedIncludes[l.Detail] {
					return ir.Unsupported(l.Path, "include", "include value not supported: "+l.Detail)
				}
			case ir.LossUnknownField:
				p := l.Path.String()
				if p == "context_management" || p == "truncation" {
					return ir.Unsupported(l.Path, p, l.Detail)
				}
			}
			return nil
		}
		opts := bridge.Options{CheckLoss: strictCheck}

		// Valid Codex request with prompt_cache_key, top_logprobs:0, logprobs:false, truncation:"disabled", allowed include passes.
		valid := `{"model":"m","input":"hi","prompt_cache_key":"k1","top_logprobs":0,"logprobs":false,"truncation":"disabled","include":["reasoning.encrypted_content"]}`
		if _, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, []byte(valid), opts); err != nil {
			t.Fatalf("valid Codex-style request rejected: %v", err)
		}

		rejectCases := []struct {
			name string
			body string
		}{
			{"include_logprobs", `{"model":"m","input":"hi","include":["message.output_text.logprobs"]}`},
			{"unknown_include", `{"model":"m","input":"hi","include":["file_search_call.results"]}`},
			{"context_management", `{"model":"m","input":"hi","context_management":[{"type":"compaction"}]}`},
			{"truncation_auto", `{"model":"m","input":"hi","truncation":"auto"}`},
		}
		for _, rc := range rejectCases {
			if _, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, []byte(rc.body), opts); err == nil {
				t.Fatalf("expected %s to be rejected under strict CheckLoss", rc.name)
			}
		}
	})
}
