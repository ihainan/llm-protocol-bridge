package bridge_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/chat"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// Reference: service/claude_strict_test.go:175 (TestStrictValidation), :254, :316, :334, :754, :914, :1400, :1585, :1640, :1747, :1774.
func TestStrictA6_AnthropicValidationRejects(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		// service/claude_strict_test.go:175
		{"zero_max_tokens", `{"model":"m","max_tokens":0,"messages":[{"role":"user","content":"hi"}]}`},
		{"no_messages", `{"model":"m","max_tokens":10,"messages":[]}`},
		{"bad_role_in_messages", `{"model":"m","max_tokens":10,"messages":[{"role":"tool","content":"x"}]}`},
		{"enabled_without_budget", `{"model":"m","max_tokens":5000,"thinking":{"type":"enabled"},"messages":[{"role":"user","content":"hi"}]}`},
		{"budget_below_1024", `{"model":"m","max_tokens":5000,"thinking":{"type":"enabled","budget_tokens":512},"messages":[{"role":"user","content":"hi"}]}`},
		{"budget_not_below_max", `{"model":"m","max_tokens":2000,"thinking":{"type":"enabled","budget_tokens":2000},"messages":[{"role":"user","content":"hi"}]}`},
		{"bad_thinking_type", `{"model":"m","max_tokens":100,"thinking":{"type":"auto"},"messages":[{"role":"user","content":"hi"}]}`},
		{"bad_effort", `{"model":"m","max_tokens":100,"output_config":{"effort":"extreme"},"messages":[{"role":"user","content":"hi"}]}`},
		{"server_tool_web_search", `{"model":"m","max_tokens":100,"tools":[{"type":"web_search_20250305","name":"web_search"}],"messages":[{"role":"user","content":"hi"}]}`},
		{"tool_without_name", `{"model":"m","max_tokens":100,"tools":[{"input_schema":{"type":"object"}}],"messages":[{"role":"user","content":"hi"}]}`},
		{"tool_choice_unknown_tool", `{"model":"m","max_tokens":100,"tools":[{"name":"a","input_schema":{"type":"object"}}],"tool_choice":{"type":"tool","name":"b"},"messages":[{"role":"user","content":"hi"}]}`},
		{"tool_choice_any_no_tools", `{"model":"m","max_tokens":100,"tool_choice":{"type":"any"},"messages":[{"role":"user","content":"hi"}]}`},
		{"pdf_document", `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"AA=="}}]}]}`},
		{"unknown_top_level_block_search_result", `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":[{"type":"search_result","source":"x"}]}]}`},
		{"tool_result_in_assistant", `{"model":"m","max_tokens":100,"messages":[{"role":"assistant","content":[{"type":"tool_result","tool_use_id":"t1","content":"x"}]}]}`},
		{"structured_output_bad_type", `{"model":"m","max_tokens":100,"output_config":{"format":{"type":"regex"}},"messages":[{"role":"user","content":"hi"}]}`},
		// service/claude_strict_test.go:914
		{"system_object", `{"model":"m","max_tokens":100,"system":{"type":"text","text":"x"},"messages":[{"role":"user","content":"hi"}]}`},
		{"system_number", `{"model":"m","max_tokens":100,"system":42,"messages":[{"role":"user","content":"hi"}]}`},
		// service/claude_strict_test.go:1774 (Round 10 bad cases)
		{"empty_string_content", `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":""}]}`},
		{"empty_content_array", `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":[]}]}`},
		{"empty_text_block", `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":""}]}]}`},
		{"system_block_no_text", `{"model":"m","max_tokens":100,"system":[{"type":"text"}],"messages":[{"role":"user","content":"hi"}]}`},
		{"tool_use_input_string", `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"x"},{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"f","input":"not-object"}]}]}`},
		{"orphan_tool_result", `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"nope","content":"x"}]}]}`},
		{"tool_result_wrong_id", `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"x"},{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"f","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"b","content":"x"}]}]}`},
		{"tool_result_two_turns_on", `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"x"},{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"f","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"ok"}]},{"role":"assistant","content":"done"},{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"late"}]}]}`},
		{"empty_stop_sequence", `{"model":"m","max_tokens":100,"stop_sequences":[""],"messages":[{"role":"user","content":"hi"}]}`},
		{"array_input_schema", `{"model":"m","max_tokens":100,"tools":[{"name":"f","input_schema":{"type":"array","items":{"type":"string"}}}],"messages":[{"role":"user","content":"hi"}]}`},
		{"tool_use_input_null", `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"x"},{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"f","input":null}]}]}`},
		{"tool_use_input_array", `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"x"},{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"f","input":[]}]}]}`},
		{"schema_type_string", `{"model":"m","max_tokens":100,"tools":[{"name":"f","input_schema":{"type":"string"}}],"messages":[{"role":"user","content":"hi"}]}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, []byte(tc.body), bridge.Options{}); err == nil {
				t.Fatalf("expected error for %s, got nil", tc.name)
			}
		})
	}
}

// Reference: service/claude_strict_test.go:1640 (tool_result content validation) and :1747 (search_result kept).
func TestStrictA6_AnthropicToolResultValidationAndSearchResult(t *testing.T) {
	wrap := func(contentJSON string) []byte {
		return []byte(`{"model":"m","max_tokens":100,"messages":[` +
			`{"role":"user","content":"go"},` +
			`{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"f","input":{}}]},` +
			`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":` + contentJSON + `}]}]}`)
	}

	bad := []struct {
		name    string
		content string
	}{
		{"image_without_source", `[{"type":"image"}]`},
		{"element_without_type", `[{}]`},
		{"null_element", `[null]`},
		{"empty_type", `[{"type":""}]`},
		{"image_file_source", `[{"type":"image","source":{"type":"file","file_id":"f1"}}]`},
		{"pdf_document", `[{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"AA=="}}]`},
		{"content_object", `{"type":"text","text":"x"}`},
		{"content_empty_object", `{}`},
		{"content_number", `42`},
		{"nested_tool_result", `[{"type":"tool_result","tool_use_id":"t2","content":"x"}]`},
		{"nested_tool_use", `[{"type":"tool_use","id":"t2","name":"f","input":{}}]`},
		{"nested_thinking", `[{"type":"thinking","thinking":"x"}]`},
		{"image_empty_url", `[{"type":"image","source":{"type":"url","url":""}}]`},
		{"image_non_http_url", `[{"type":"image","source":{"type":"url","url":"file:///etc/passwd"}}]`},
		{"image_base64_no_data", `[{"type":"image","source":{"type":"base64","media_type":"image/png"}}]`},
		{"image_base64_no_mime", `[{"type":"image","source":{"type":"base64","data":"AAAA"}}]`},
	}
	for _, tc := range bad {
		t.Run("bad/"+tc.name, func(t *testing.T) {
			if _, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, wrap(tc.content), bridge.Options{}); err == nil {
				t.Fatalf("expected error for tool_result %s, got nil", tc.name)
			}
		})
	}

	good := []struct {
		name    string
		content string
	}{
		{"string", `"ok"`},
		{"empty_array", `[]`},
		{"null", `null`},
		{"http_url_image", `[{"type":"image","source":{"type":"url","url":"http://example.test/a.png"}}]`},
		{"https_url_image", `[{"type":"image","source":{"type":"url","url":"https://example.test/a.png"}}]`},
		{"search_result", `[{"type":"search_result","source":"https://example.test/doc","title":"Title","content":[{"type":"text","text":"body"}]}]`},
		{"tool_reference_skipped", `[{"type":"tool_reference","tool_name":"Bash"}]`},
	}
	for _, tc := range good {
		t.Run("good/"+tc.name, func(t *testing.T) {
			if _, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, wrap(tc.content), bridge.Options{}); err != nil {
				t.Fatalf("expected valid tool_result %s, got: %v", tc.name, err)
			}
		})
	}

	t.Run("search_result_rendered_text_exact", func(t *testing.T) {
		// Reference: service/claude_strict_test.go:1747 (TestStrictToolResultSearchResultKept)
		body := wrap(`[` +
			`{"type":"search_result","source":"https://example.test/m","title":"Release Notes","content":[{"type":"text","text":"Released on 2026-08-14."}]},` +
			`{"type":"tool_reference","tool_name":"Bash"}` +
			`]`)
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, body, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		var out struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.Unmarshal(plan.Body, &out)
		want := "[search result] Release Notes\nsource: https://example.test/m\nReleased on 2026-08-14."
		if len(out.Messages) != 3 || out.Messages[2].Role != "tool" || out.Messages[2].Content != want {
			t.Fatalf("want tool message %q, got %+v", want, out.Messages)
		}
	})
}

// Reference: service/claude_strict_test.go:254, :316, :334, :754, :914, :1400, :1585, :1774 (11 valid shapes).
func TestStrictA6_AnthropicValidShapesAndClaudeCodeAndFolding(t *testing.T) {
	t.Run("tool_choice_auto_and_none_dropped_without_tools", func(t *testing.T) {
		// Reference: service/claude_strict_test.go:254
		for _, tc := range []string{
			`{"model":"m","max_tokens":100,"tool_choice":{"type":"auto"},"messages":[{"role":"user","content":"hi"}]}`,
			`{"model":"m","max_tokens":100,"tool_choice":{"type":"none"},"messages":[{"role":"user","content":"hi"}]}`,
		} {
			plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, []byte(tc), bridge.Options{})
			if err != nil {
				t.Fatalf("Convert: %v", err)
			}
			var out map[string]any
			_ = json.Unmarshal(plan.Body, &out)
			if _, ok := out["tool_choice"]; ok {
				t.Fatalf("tool_choice should be omitted when no tools exist: %s", plan.Body)
			}
		}
	})

	t.Run("system_newline_join_document_title_and_policy_omissions", func(t *testing.T) {
		// Reference: service/claude_strict_test.go:316 (metadata.user_id) and :334 (system \n, document title+text, omit history thinking)
		raw := []byte(`{
			"model": "m",
			"max_tokens": 100,
			"metadata": {"user_id": "{\"device_id\":\"d\"}"},
			"system": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
			"messages": [
				{"role": "user", "content": [
					{"type": "document", "title": "T", "source": {"type": "text", "media_type": "text/plain", "data": "DOC"}}
				]},
				{"role": "assistant", "content": [
					{"type": "thinking", "thinking": "hmm", "signature": "x"},
					{"type": "text", "text": "ans"}
				]},
				{"role": "user", "content": "next"}
			]
		}`)
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, raw, bridge.Options{
			OmitUserID:           true,
			OmitHistoryReasoning: true,
		})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		var out struct {
			User     string `json:"user"`
			Messages []struct {
				Role             string `json:"role"`
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"messages"`
		}
		_ = json.Unmarshal(plan.Body, &out)
		if out.User != "" || strings.Contains(string(plan.Body), "device_id") {
			t.Fatalf("user_id/device_id leaked: %s", plan.Body)
		}
		if out.Messages[0].Role != "system" || out.Messages[0].Content != "A\nB" {
			t.Fatalf("want system \"A\\nB\", got %+v", out.Messages[0])
		}
		if !strings.Contains(out.Messages[1].Content, "T") || !strings.Contains(out.Messages[1].Content, "DOC") {
			t.Fatalf("document title/text lost: %+v", out.Messages[1])
		}
		if out.Messages[2].ReasoningContent != "" || strings.Contains(string(plan.Body), "hmm") {
			t.Fatalf("historical thinking leaked when OmitHistoryReasoning=true: %s", plan.Body)
		}
	})

	t.Run("real_claude_code_request_shape", func(t *testing.T) {
		// Reference: service/claude_strict_test.go:754 (TestStrictClaudeCodeRequestShape)
		raw := []byte(`{
			"model": "m",
			"max_tokens": 32000,
			"stream": true,
			"thinking": {"type": "adaptive", "display": "omitted"},
			"context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
			"output_config": {"effort": "high"},
			"metadata": {"user_id": "{\"device_id\":\"abc\",\"session_id\":\"s\"}"},
			"safeguards": {"x": 1},
			"system": [{"type": "text", "text": "S1"}, {"type": "text", "text": "S2", "cache_control": {"type": "ephemeral"}}],
			"tools": [{"name": "Read", "description": "read", "input_schema": {"type": "object"}}],
			"messages": [
				{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
				{"role": "system", "content": [{"type": "text", "text": "mid", "cache_control": {"type": "ephemeral"}}]},
				{"role": "user", "content": "go"}
			]
		}`)
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, raw, bridge.Options{
			OmitUserID: true,
		})
		if err != nil {
			t.Fatalf("Claude Code request failed: %v", err)
		}
		var out struct {
			ReasoningEffort string `json:"reasoning_effort"`
			User            string `json:"user"`
			Messages        []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		_ = json.Unmarshal(plan.Body, &out)
		if out.ReasoningEffort != "high" {
			t.Fatalf("want effort high, got %q", out.ReasoningEffort)
		}
		roles := make([]string, len(out.Messages))
		for i, m := range out.Messages {
			roles[i] = m.Role
		}
		if strings.Join(roles, ",") != "system,user,system,user" {
			t.Fatalf("want roles system,user,system,user, got %v", roles)
		}
		if string(out.Messages[2].Content) != `"mid"` {
			t.Fatalf("want mid system content \"mid\", got %s", out.Messages[2].Content)
		}
		if out.User != "" || strings.Contains(string(plan.Body), "device_id") {
			t.Fatalf("device_id leaked: %s", plan.Body)
		}
	})

	t.Run("context_management_valid_vs_unknown_edits_with_check_loss", func(t *testing.T) {
		// Reference: service/claude_strict_test.go:914
		strictCtxCheck := func(l ir.Loss) error {
			if l.Kind == ir.LossUnknownField && strings.HasPrefix(l.Path.String(), "context_management.") {
				return ir.Unsupported(l.Path, "context_management", l.Detail)
			}
			return nil
		}
		opts := bridge.Options{CheckLoss: strictCtxCheck}

		// Valid clear_thinking_YYYYMMDD and clear_tool_uses_YYYYMMDD accepted.
		valid := `{"model":"m","max_tokens":100,"context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"all"},{"type":"clear_tool_uses_20250919"}]},"messages":[{"role":"user","content":"hi"}]}`
		if _, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, []byte(valid), opts); err != nil {
			t.Fatalf("valid context_management rejected: %v", err)
		}
		// Unknown edit rejected.
		bad1 := `{"model":"m","max_tokens":100,"context_management":{"edits":[{"type":"compact_20260101"}]},"messages":[{"role":"user","content":"hi"}]}`
		if _, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, []byte(bad1), opts); err == nil {
			t.Fatalf("unknown context_management edit should be rejected")
		}
		// Edit without 8-digit version rejected.
		bad2 := `{"model":"m","max_tokens":100,"context_management":{"edits":[{"type":"clear_thinking_"}]},"messages":[{"role":"user","content":"hi"}]}`
		if _, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, []byte(bad2), opts); err == nil {
			t.Fatalf("unversioned context_management edit should be rejected")
		}
	})

	t.Run("mid_conversation_system_folding_and_tool_result_images", func(t *testing.T) {
		// Reference: service/claude_strict_test.go:1400 (cases 1-3) and :1585 (tool_result images + mid-system)
		opts := bridge.Options{Chat: chat.Dialect{MidSystem: "fold"}}

		// Case 1: system:"S", [user:"a", assistant:"b", system:"be brief", user:"go"]
		r1 := []byte(`{"model":"qwen","max_tokens":100,"system":"S","messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"system","content":"be brief"},{"role":"user","content":"go"}]}`)
		p1, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, r1, opts)
		if err != nil {
			t.Fatalf("case 1: %v", err)
		}
		var o1 struct {
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		_ = json.Unmarshal(p1.Body, &o1)
		if len(o1.Messages) != 4 || o1.Messages[0].Role != "system" || o1.Messages[3].Role != "user" {
			t.Fatalf("case 1 unexpected messages: %s", p1.Body)
		}
		if !strings.Contains(string(o1.Messages[3].Content), "<system_message>\\nbe brief\\n</system_message>") {
			t.Fatalf("case 1 folded note missing: %s", o1.Messages[3].Content)
		}

		// Case 2: [user:"list", assistant:[tool_use t1], system:"note", user:[tool_result t1 "a.py"]]
		r2 := []byte(`{"model":"qwen","max_tokens":100,"messages":[{"role":"user","content":"list"},{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"ls","input":{}}]},{"role":"system","content":"note"},{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"a.py"}]}]}`)
		p2, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, r2, opts)
		if err != nil {
			t.Fatalf("case 2: %v", err)
		}
		var o2 struct {
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		_ = json.Unmarshal(p2.Body, &o2)
		if len(o2.Messages) != 4 || o2.Messages[2].Role != "tool" || o2.Messages[3].Role != "user" || !strings.Contains(string(o2.Messages[3].Content), "note") {
			t.Fatalf("case 2 mismatch: %s", p2.Body)
		}

		// Case 3: [user:"a", system:"x", assistant:"b", user:"c", system:"y"]
		r3 := []byte(`{"model":"qwen","max_tokens":100,"messages":[{"role":"user","content":"a"},{"role":"system","content":"x"},{"role":"assistant","content":"b"},{"role":"user","content":"c"},{"role":"system","content":"y"}]}`)
		p3, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, r3, opts)
		if err != nil {
			t.Fatalf("case 3: %v", err)
		}
		var o3 struct {
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		_ = json.Unmarshal(p3.Body, &o3)
		if len(o3.Messages) != 3 || !strings.Contains(string(o3.Messages[0].Content), "x\\n</system_message>") || !strings.Contains(string(o3.Messages[2].Content), "y\\n</system_message>") {
			t.Fatalf("case 3 mismatch: %s", p3.Body)
		}

		// Case 4 (:1585): tool_result images + mid-system ordering.
		r4 := []byte(`{"model":"qwen","max_tokens":100,"messages":[
			{"role":"user","content":"look at both"},
			{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"f","input":{}},{"type":"tool_use","id":"t2","name":"g","input":{}}]},
			{"role":"system","content":"be brief"},
			{"role":"user","content":[
				{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"a.png"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}]},
				{"type":"tool_result","tool_use_id":"t2","content":"plain text"},
				{"type":"text","text":"which colour?"}
			]}
		]}`)
		p4, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, r4, opts)
		if err != nil {
			t.Fatalf("case 4: %v", err)
		}
		var o4 struct {
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		_ = json.Unmarshal(p4.Body, &o4)
		if len(o4.Messages) != 5 || o4.Messages[2].Role != "tool" || o4.Messages[3].Role != "tool" || o4.Messages[4].Role != "user" {
			t.Fatalf("case 4 unexpected structure: %s", p4.Body)
		}
		var parts []struct {
			Type     string `json:"type"`
			Text     string `json:"text"`
			ImageURL *struct {
				URL string `json:"url"`
			} `json:"image_url"`
		}
		if err := json.Unmarshal(o4.Messages[4].Content, &parts); err != nil || len(parts) != 4 {
			t.Fatalf("case 4 expected 4 parts in trailing user msg, got %s", o4.Messages[4].Content)
		}
		if parts[0].Text != "<system_message>\nbe brief\n</system_message>" ||
			parts[1].Text != "[images returned by tool call t1]" ||
			parts[2].ImageURL == nil || parts[2].ImageURL.URL != "data:image/png;base64,AAAA" ||
			parts[3].Text != "which colour?" {
			t.Fatalf("case 4 part order mismatch: %+v", parts)
		}
	})

	t.Run("round10_all_11_valid_shapes", func(t *testing.T) {
		// Reference: service/claude_strict_test.go:1774 (11 valid shapes)
		validShapes := []struct {
			name string
			body string
		}{
			{"empty_final_assistant_prefill", `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":""}]}`},
			{"system_between_tool_use_and_tool_result", `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"x"},{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"f","input":{}}]},{"role":"system","content":"note"},{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":""}]}]}`},
			{"empty_tool_output_text_block", `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"x"},{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"f","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":[{"type":"text","text":""}]}]}]}`},
			{"tool_without_input_schema", `{"model":"m","max_tokens":100,"tools":[{"name":"c"}],"messages":[{"role":"user","content":"hi"}]}`},
			{"whitespace_text_block", `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"  \n"}]}]}`},
			{"newline_stop_sequence", `{"model":"m","max_tokens":100,"stop_sequences":["\n\n"],"messages":[{"role":"user","content":"hi"}]}`},
			{"schema_without_type", `{"model":"m","max_tokens":100,"tools":[{"name":"f","input_schema":{"properties":{"a":{"type":"string"}}}}],"messages":[{"role":"user","content":"hi"}]}`},
			{"schema_type_list", `{"model":"m","max_tokens":100,"tools":[{"name":"f","input_schema":{"type":["object","null"],"properties":{}}}],"messages":[{"role":"user","content":"hi"}]}`},
			{"schema_with_draft07_and_anyOf", `{"model":"m","max_tokens":100,"tools":[{"name":"f","input_schema":{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","anyOf":[{"required":["a"]},{"required":["b"]}]}}],"messages":[{"role":"user","content":"hi"}]}`},
			{"results_split_over_consecutive_users", `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"x"},{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"f","input":{}},{"type":"tool_use","id":"b","name":"g","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"1"}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"b","content":"2"}]}]}`},
			{"assistant_thinking_and_tool_use_only", `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"x"},{"role":"assistant","content":[{"type":"thinking","thinking":"t"},{"type":"tool_use","id":"a","name":"f","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":[]},{"type":"text","text":"go on"}]}]}`},
		}
		for _, tc := range validShapes {
			t.Run(tc.name, func(t *testing.T) {
				if _, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, []byte(tc.body), bridge.Options{}); err != nil {
					t.Fatalf("valid shape %s rejected: %v", tc.name, err)
				}
			})
		}
	})
}
