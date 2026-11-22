package bridge_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/anthropic"
	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/responses"
)

// Regression tests for all 7 findings from codex-strictA6.txt (1 P1 + 6 P2).
func TestStrictA6_CodexReviewFixes(t *testing.T) {
	// P1 #1: Responses stateful continuation (previous_response_id) allows tool outputs
	// whose matching calls live in the previous response.
	t.Run("P1_01_responses_previous_response_id_tool_output_allowed", func(t *testing.T) {
		for _, typ := range []string{"function_call_output", "custom_tool_call_output"} {
			body := []byte(`{"model":"m","previous_response_id":"resp_prev","input":[{"type":"` + typ + `","call_id":"c1","output":"ok"}]}`)
			plan, err := bridge.Convert(ir.FormatResponses, ir.FormatResponses, body, bridge.Options{})
			if err != nil {
				t.Fatalf("%s continuation rejected: %v", typ, err)
			}
			if !strings.Contains(string(plan.Body), `"previous_response_id":"resp_prev"`) || !strings.Contains(string(plan.Body), `"call_id":"c1"`) {
				t.Fatalf("%s continuation body unexpected: %s", typ, plan.Body)
			}
		}
	})

	// P2 #2: Responses reasoning item with content:[{type:"reasoning_text"}] roundtrips into Chat history.
	t.Run("P2_02_responses_reasoning_text_content_roundtrips_in_history", func(t *testing.T) {
		body := []byte(`{
			"model": "m",
			"input": [
				{"role": "user", "content": "hi"},
				{"type": "reasoning", "id": "rs_1", "summary": [], "content": [{"type": "reasoning_text", "text": "think"}]},
				{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
				{"role": "user", "content": "next"}
			]
		}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, body, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		if !strings.Contains(string(plan.Body), `"reasoning_content":"think"`) {
			t.Fatalf("expected reasoning_text history preserved in Chat request: %s", plan.Body)
		}
	})

	// FixA7 #1 (r3 #1 & #3): Deterministic Chat streaming tool-name merge matching
	// the four-branch rule without guessing from other declared tool prefixes.
	t.Run("FixA7_01_chat_stream_tool_name_merge_deterministic_and_stream_nonstream_parity", func(t *testing.T) {
		// r3 #1: declaring both foobar and foofoobarbaz; receiving "foo" then "foobar"
		// must produce "foobar" (not "foofoobar"), matching non-streaming.
		reqR3_1 := []byte(`{
			"model": "m",
			"max_tokens": 100,
			"stream": true,
			"messages": [{"role": "user", "content": "go"}],
			"tools": [
				{"name": "foobar", "input_schema": {"type": "object"}},
				{"name": "foofoobarbaz", "input_schema": {"type": "object"}}
			]
		}`)
		plan1, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, reqR3_1, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert r3 #1: %v", err)
		}
		rawSSE1 := `data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"foo"}}]}}]}` + "\n\n" +
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"foobar","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}` + "\n\n" +
			"data: [DONE]\n\n"
		out1 := string(feedSSE(t, plan1.NewStream(), rawSSE1))
		parsed1 := parseAnthropicSSE(t, out1)
		if parsed1.Errors != 0 || len(parsed1.Blocks) != 1 || parsed1.Blocks[0].Name != "foobar" {
			t.Fatalf("r3 #1: want tool name \"foobar\", got blocks=%+v errors=%d\n%s", parsed1.Blocks, parsed1.Errors, out1)
		}
		nsBody1 := []byte(`{"id":"1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"foobar","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`)
		nsOut1, _, err := plan1.ConvertResponse(nsBody1)
		if err != nil {
			t.Fatalf("ConvertResponse r3 #1: %v", err)
		}
		if !strings.Contains(string(nsOut1), `"name":"foobar"`) {
			t.Fatalf("r3 #1 non-stream mismatch: %s", nsOut1)
		}

		// Standard upstream patterns: once full name, prefix then full, split fragments,
		// and repeated full name all match non-streaming "get_weather".
		reqStd := []byte(`{"model":"m","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"go"}],"tools":[{"name":"get_weather","input_schema":{"type":"object"}}]}`)
		planStd, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, reqStd, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert std: %v", err)
		}
		nsStd, _, err := planStd.ConvertResponse([]byte(`{"id":"1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`))
		if err != nil || !strings.Contains(string(nsStd), `"name":"get_weather"`) {
			t.Fatalf("std non-stream failed: err=%v body=%s", err, nsStd)
		}
		patterns := []struct {
			name   string
			chunks []string
		}{
			{"single_full_name", []string{
				`{"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`,
			}},
			{"prefix_then_full", []string{
				`{"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"get_"}}]}}]}`,
				`{"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"get_weather","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`,
			}},
			{"split_fragments", []string{
				`{"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"get_"}}]}}]}`,
				`{"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"weather","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`,
			}},
			{"repeated_full_name", []string{
				`{"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"get_weather"}}]}}]}`,
				`{"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"get_weather","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`,
			}},
		}
		for _, pt := range patterns {
			var sb strings.Builder
			for _, ch := range pt.chunks {
				sb.WriteString("data: " + ch + "\n\n")
			}
			sb.WriteString("data: [DONE]\n\n")
			out := string(feedSSE(t, planStd.NewStream(), sb.String()))
			parsed := parseAnthropicSSE(t, out)
			if parsed.Errors != 0 || len(parsed.Blocks) != 1 || parsed.Blocks[0].Name != "get_weather" {
				t.Fatalf("%s: want \"get_weather\", got blocks=%+v errors=%d\n%s", pt.name, parsed.Blocks, parsed.Errors, out)
			}
		}

		// r3 #3 pinned known limitation (documented in README): identical consecutive
		// fragments ("foo", "foo", "bar") collapse the duplicate "foo" -> "foobar".
		reqLim := []byte(`{
			"model": "m",
			"max_tokens": 100,
			"stream": true,
			"messages": [{"role": "user", "content": "go"}],
			"tools": [
				{"name": "foo", "input_schema": {"type": "object"}},
				{"name": "foofoobar", "input_schema": {"type": "object"}}
			]
		}`)
		planLim, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, reqLim, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert r3 #3: %v", err)
		}
		rawSSELim := `data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"foo"}}]}}]}` + "\n\n" +
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"foo"}}]}}]}` + "\n\n" +
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"bar","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}` + "\n\n" +
			"data: [DONE]\n\n"
		outLim := string(feedSSE(t, planLim.NewStream(), rawSSELim))
		parsedLim := parseAnthropicSSE(t, outLim)
		if parsedLim.Errors != 0 || len(parsedLim.Blocks) != 1 || parsedLim.Blocks[0].Name != "foobar" {
			t.Fatalf("r3 #3 pinned known limitation: want \"foobar\", got blocks=%+v errors=%d\n%s", parsedLim.Blocks, parsedLim.Errors, outLim)
		}
	})

	// P2 #4: Consecutive Anthropic assistant messages with parallel tool calls both remain valid targets.
	t.Run("P2_04_anthropic_consecutive_assistant_tool_uses_accepted", func(t *testing.T) {
		body := []byte(`{
			"model": "m",
			"max_tokens": 100,
			"messages": [
				{"role": "user", "content": "run both"},
				{"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "f", "input": {}}]},
				{"role": "assistant", "content": [{"type": "tool_use", "id": "b", "name": "g", "input": {}}]},
				{"role": "user", "content": [
					{"type": "tool_result", "tool_use_id": "a", "content": "r1"},
					{"type": "tool_result", "tool_use_id": "b", "content": "r2"}
				]}
			]
		}`)
		if _, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, body, bridge.Options{}); err != nil {
			t.Fatalf("consecutive assistant tool_use messages rejected: %v", err)
		}
	})

	// P2 #5: Direct StreamEncoder pendingTools queue bounded even without EvToolArgs.
	t.Run("P2_05_direct_encoders_bound_pending_tool_starts_without_args", func(t *testing.T) {
		rEnc := responses.NewStreamEncoder(responses.EncodeOptions{Env: &ir.Env{}, Model: "m", MaxBufferedBytes: 32})
		var rDst []byte
		rDst, _ = rEnc.Encode(rDst, ir.Event{Type: ir.EvToolStart, Index: 0, ID: "c0", Name: "fn", Kind: ir.ToolFunction})
		for i := 0; i < 10; i++ {
			rDst, _ = rEnc.Encode(rDst, ir.Event{Type: ir.EvToolStart, Index: 0, ID: "call_long_id", Name: "fn_name", Kind: ir.ToolFunction})
		}
		rDst, _ = rEnc.Close(rDst)
		if !strings.Contains(string(rDst), `"response.failed"`) {
			t.Fatalf("Responses encoder failed to bound pending EvToolStart queue:\n%s", rDst)
		}

		aEnc := anthropic.NewStreamEncoder(anthropic.EncodeOptions{Env: &ir.Env{}, Model: "m", MaxBufferedBytes: 32})
		var aDst []byte
		aDst, _ = aEnc.Encode(aDst, ir.Event{Type: ir.EvToolStart, Index: 0, ID: "c0", Name: "fn", Kind: ir.ToolFunction})
		for i := 0; i < 10; i++ {
			aDst, _ = aEnc.Encode(aDst, ir.Event{Type: ir.EvToolStart, Index: 0, ID: "call_long_id", Name: "fn_name", Kind: ir.ToolFunction})
		}
		aDst, _ = aEnc.Close(aDst)
		parsed := parseAnthropicSSE(t, string(aDst))
		if parsed.Errors == 0 || parsed.MessageStop {
			t.Fatalf("Anthropic encoder failed to bound pending EvToolStart queue:\n%s", aDst)
		}
	})

	// P2 #6: Anthropic context_management preserved on identity build, reports loss cross-format,
	// and KeepTurns trims older assistant reasoning in Chat build.
	t.Run("P2_06_context_management_identity_passthrough_loss_and_keep_turns", func(t *testing.T) {
		raw := []byte(`{
			"model": "claude-sonnet-4",
			"max_tokens": 100,
			"context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 1}}]},
			"messages": [
				{"role": "user", "content": "q1"},
				{"role": "assistant", "content": [{"type": "thinking", "thinking": "old_thought", "signature": "s1"}, {"type": "text", "text": "a1"}]},
				{"role": "user", "content": "q2"},
				{"role": "assistant", "content": [{"type": "thinking", "thinking": "recent_thought", "signature": "s2"}, {"type": "text", "text": "a2"}]},
				{"role": "user", "content": "q3"}
			]
		}`)

		// Identity Anthropic -> Anthropic preserves context_management with zero context_management loss.
		pID, err := bridge.Convert(ir.FormatAnthropic, ir.FormatAnthropic, raw, bridge.Options{})
		if err != nil {
			t.Fatalf("Anthropic->Anthropic: %v", err)
		}
		if !strings.Contains(string(pID.Body), `"context_management"`) {
			t.Fatalf("identity Anthropic->Anthropic dropped context_management: %s", pID.Body)
		}

		// Cross-format Anthropic -> Chat honors KeepTurns=1 (drops old_thought, keeps recent_thought)
		// and records LossUnknownField on context_management (so Deny[LossUnknownField] can reject).
		pChat, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, raw, bridge.Options{})
		if err != nil {
			t.Fatalf("Anthropic->Chat: %v", err)
		}
		if strings.Contains(string(pChat.Body), "old_thought") || !strings.Contains(string(pChat.Body), "recent_thought") {
			t.Fatalf("KeepTurns=1 not honored in Chat build: %s", pChat.Body)
		}
		sawLoss := false
		for _, l := range pChat.Losses {
			if l.Kind == ir.LossUnknownField && l.Path.String() == "context_management" {
				sawLoss = true
			}
		}
		if !sawLoss {
			t.Fatalf("expected context_management loss in cross-format conversion, got %+v", pChat.Losses)
		}
		if _, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, raw, bridge.Options{Deny: ir.LossSet{ir.LossUnknownField: true}}); err == nil {
			t.Fatalf("Deny[LossUnknownField] should reject dropped context_management cross-format")
		}
	})

	// P2 #7: TrimLeadingNewlines shifts citation StartIndex/EndIndex in both non-stream and stream.
	t.Run("P2_07_trim_leading_newlines_shifts_citations", func(t *testing.T) {
		resp := &ir.Response{
			ID:    "r1",
			Model: "m",
			Outputs: []ir.Output{{
				Parts: []ir.Part{{
					Type: ir.PartText,
					Text: "\n\nParis",
					Citations: []ir.Citation{{
						Type:       "url_citation",
						URL:        "https://example.test",
						Title:      "Paris",
						StartIndex: 2,
						EndIndex:   7,
					}},
				}},
				Stop: ir.Stop{Reason: ir.StopEndTurn},
			}},
		}
		// Non-stream
		b, err := responses.BuildResponse(resp, responses.EncodeOptions{Env: &ir.Env{}, Model: "m", TrimLeadingNewlines: true})
		if err != nil {
			t.Fatalf("BuildResponse: %v", err)
		}
		var out struct {
			Output []struct {
				Content []struct {
					Text        string `json:"text"`
					Annotations []struct {
						StartIndex int `json:"start_index"`
						EndIndex   int `json:"end_index"`
					} `json:"annotations"`
				} `json:"content"`
			} `json:"output"`
		}
		_ = json.Unmarshal(b, &out)
		if out.Output[0].Content[0].Text != "Paris" ||
			len(out.Output[0].Content[0].Annotations) != 1 ||
			out.Output[0].Content[0].Annotations[0].StartIndex != 0 ||
			out.Output[0].Content[0].Annotations[0].EndIndex != 5 {
			t.Fatalf("non-stream citation not shifted: %s", b)
		}

		// Stream (split leading newlines across chunks)
		enc := responses.NewStreamEncoder(responses.EncodeOptions{Env: &ir.Env{}, Model: "m", TrimLeadingNewlines: true})
		var dst []byte
		for _, ev := range []ir.Event{
			{Type: ir.EvText, Index: 0, Text: "\n"},
			{Type: ir.EvText, Index: 0, Text: "\nParis"},
			{Type: ir.EvCitation, Index: 0, Citation: &ir.Citation{Type: "url_citation", URL: "https://example.test", Title: "Paris", StartIndex: 2, EndIndex: 7}},
			{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopEndTurn}},
		} {
			dst, err = enc.Encode(dst, ev)
			if err != nil {
				t.Fatalf("Encode: %v", err)
			}
		}
		dst, _ = enc.Close(dst)
		s := string(dst)
		if !strings.Contains(s, `"start_index":0`) || !strings.Contains(s, `"end_index":5`) {
			t.Fatalf("stream citation not shifted to [0,5):\n%s", s)
		}

		// Follow-up P2: newline-only chunk before EvBuiltin must NOT shift citations on a later message.
		enc2 := responses.NewStreamEncoder(responses.EncodeOptions{Env: &ir.Env{}, Model: "m", TrimLeadingNewlines: true})
		var dst2 []byte
		for _, ev := range []ir.Event{
			{Type: ir.EvText, Index: 0, Text: "\n\n"},
			{Type: ir.EvBuiltin, Index: 0, Builtin: &ir.BuiltinActivity{Kind: ir.BuiltinWebSearch, ID: "ws_1"}},
			{Type: ir.EvText, Index: 0, Text: "Paris"},
			{Type: ir.EvCitation, Index: 0, Citation: &ir.Citation{Type: "url_citation", URL: "https://example.test", Title: "Paris", StartIndex: 0, EndIndex: 5}},
			{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopEndTurn}},
		} {
			dst2, err = enc2.Encode(dst2, ev)
			if err != nil {
				t.Fatalf("Encode enc2: %v", err)
			}
		}
		dst2, _ = enc2.Close(dst2)
		if !strings.Contains(string(dst2), `"start_index":0`) || !strings.Contains(string(dst2), `"end_index":5`) {
			t.Fatalf("pendingTrimRunes leaked across EvBuiltin boundary:\n%s", dst2)
		}

		// FixA7 #2 (r3 #2): candidate 1 EvStop must NOT clear candidate 0's pendingTrimRunes.
		enc3 := responses.NewStreamEncoder(responses.EncodeOptions{Env: &ir.Env{}, Model: "m", TrimLeadingNewlines: true})
		var dst3 []byte
		for _, ev := range []ir.Event{
			{Type: ir.EvText, Index: 0, Text: "\n\n"},
			{Type: ir.EvStop, Index: 1, Stop: &ir.Stop{Reason: ir.StopEndTurn}},
			{Type: ir.EvText, Index: 0, Text: "Paris"},
			{Type: ir.EvCitation, Index: 0, Citation: &ir.Citation{Type: "url_citation", URL: "https://example.test", Title: "Paris", StartIndex: 2, EndIndex: 7}},
			{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopEndTurn}},
		} {
			dst3, err = enc3.Encode(dst3, ev)
			if err != nil {
				t.Fatalf("Encode enc3: %v", err)
			}
		}
		dst3, _ = enc3.Close(dst3)
		if !strings.Contains(string(dst3), `"start_index":0`) || !strings.Contains(string(dst3), `"end_index":5`) {
			t.Fatalf("r3 #2: candidate 1 EvStop cleared candidate 0 pendingTrimRunes:\n%s", dst3)
		}
	})

	// Follow-up P2: KeepTurns counting merged consecutive assistant turns.
	t.Run("R2_P2_merged_keep_turns", func(t *testing.T) {
		// Consecutive assistant messages in the latest turn both keep thinking under KeepTurns=1.
		ktBody := []byte(`{
			"model": "claude-sonnet-4",
			"max_tokens": 100,
			"context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 1}}]},
			"messages": [
				{"role": "user", "content": "q1"},
				{"role": "assistant", "content": [{"type": "thinking", "thinking": "older_turn", "signature": "s0"}, {"type": "text", "text": "a0"}]},
				{"role": "user", "content": "q2"},
				{"role": "assistant", "content": [{"type": "thinking", "thinking": "latest_part1", "signature": "s1"}, {"type": "text", "text": "p1"}]},
				{"role": "assistant", "content": [{"type": "text", "text": "p2"}]},
				{"role": "user", "content": "q3"}
			]
		}`)
		pChat, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, ktBody, bridge.Options{})
		if err != nil {
			t.Fatalf("KeepTurns merged turn: %v", err)
		}
		if strings.Contains(string(pChat.Body), "older_turn") || !strings.Contains(string(pChat.Body), "latest_part1") {
			t.Fatalf("KeepTurns=1 should drop older_turn and keep latest_part1 across consecutive assistant msgs: %s", pChat.Body)
		}
	})
}
