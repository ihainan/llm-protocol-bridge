package bridge_test

import (
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

func parseChatToolArgs(t *testing.T, raw string) (argsBySlot map[int]string, sawDone bool, sawError bool) {
	t.Helper()
	argsBySlot = map[int]string{}
	for _, line := range strings.Split(raw, "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data: "))
		if payload == "[DONE]" {
			sawDone = true
			continue
		}
		var chunk struct {
			Error   *json.RawMessage `json:"error"`
			Choices []struct {
				Delta struct {
					ToolCalls []struct {
						Index    int `json:"index"`
						Function struct {
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("bad chat SSE JSON %q: %v", payload, err)
		}
		if chunk.Error != nil {
			sawError = true
			continue
		}
		for _, ch := range chunk.Choices {
			for _, tc := range ch.Delta.ToolCalls {
				argsBySlot[tc.Index] += tc.Function.Arguments
			}
		}
	}
	return argsBySlot, sawDone, sawError
}

// FixA5 #1: Responses -> Anthropic function_call with no argument deltas (only
// output_item.done arguments), suffix completion, and non-prefix mismatch error.
func TestReview5_01_AnthropicFunctionCallFinalOnlyArgsAndSuffix(t *testing.T) {
	newReq := func() *ir.Request {
		return &ir.Request{
			Model: "m", Stream: true,
			Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{{Type: ir.PartText, Text: "go"}}}},
			Tools: []ir.Tool{
				{Name: "f", Kind: ir.ToolFunction, Schema: json.RawMessage(`{"type":"object","properties":{"x":{"type":"integer"}}}`)},
				{Name: "g", Kind: ir.ToolFunction, Schema: json.RawMessage(`{"type":"object","properties":{"y":{"type":"integer"}}}`)},
			},
		}
	}

	// Subtest A: No argument deltas; only output_item.done provides arguments (active + queued).
	finalOnlyEvents := []sse.Event{
		{Name: "response.created", Data: []byte(`{"type":"response.created","response":{"id":"r1","status":"in_progress"}}`)},
		{Name: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc1","call_id":"c1","name":"f","arguments":"","status":"in_progress"}}`)},
		{Name: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc2","call_id":"c2","name":"g","arguments":"","status":"in_progress"}}`)},
		{Name: "response.output_item.done", Data: []byte(`{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc1","call_id":"c1","name":"f","arguments":"{\"x\":1}","status":"completed"}}`)},
		{Name: "response.output_item.done", Data: []byte(`{"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"fc2","call_id":"c2","name":"g","arguments":"{\"y\":2}","status":"completed"}}`)},
		{Name: "response.completed", Data: []byte(`{"type":"response.completed","response":{"id":"r1","status":"completed"}}`)},
	}

	for _, rm := range allReleaseModes {
		t.Run("final_only/"+rm.name, func(t *testing.T) {
			plan, err := bridge.Build(newReq(), ir.FormatAnthropic, ir.FormatResponses, bridge.Options{ToolRelease: rm.rel})
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			st := plan.NewStream()
			var out []byte
			for _, ev := range finalOnlyEvents {
				out, err = st.Feed(out, ev)
				if err != nil {
					t.Fatalf("Feed: %v", err)
				}
			}
			out, err = st.End(out, ir.EndEOF)
			if err != nil {
				t.Fatalf("End: %v", err)
			}
			parsed := parseAnthropicSSE(t, string(out))
			if parsed.Errors != 0 || !parsed.MessageStop || len(parsed.Blocks) != 2 {
				t.Fatalf("release %s: unexpected stream errors=%d stop=%v blocks=%+v\n%s", rm.name, parsed.Errors, parsed.MessageStop, parsed.Blocks, string(out))
			}
			var a0, a1 map[string]any
			if err := json.Unmarshal([]byte(parsed.Blocks[0].Input), &a0); err != nil || a0["x"] != float64(1) {
				t.Fatalf("release %s: block 0 want {\"x\":1}, got %q (err=%v)\n%s", rm.name, parsed.Blocks[0].Input, err, string(out))
			}
			if err := json.Unmarshal([]byte(parsed.Blocks[1].Input), &a1); err != nil || a1["y"] != float64(2) {
				t.Fatalf("release %s: block 1 want {\"y\":2}, got %q (err=%v)\n%s", rm.name, parsed.Blocks[1].Input, err, string(out))
			}
		})
	}

	// Subtest B: Function call delta prefix "{\"x\":" completed by final "{\"x\":1}" (active + queued).
	suffixEvents := []sse.Event{
		{Name: "response.created", Data: []byte(`{"type":"response.created","response":{"id":"r2","status":"in_progress"}}`)},
		{Name: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc1","call_id":"c1","name":"f","arguments":"","status":"in_progress"}}`)},
		{Name: "response.function_call_arguments.delta", Data: []byte(`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"x\":"}`)},
		{Name: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc2","call_id":"c2","name":"g","arguments":"","status":"in_progress"}}`)},
		{Name: "response.function_call_arguments.delta", Data: []byte(`{"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"y\":"}`)},
		{Name: "response.output_item.done", Data: []byte(`{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc1","call_id":"c1","name":"f","arguments":"{\"x\":1}","status":"completed"}}`)},
		{Name: "response.output_item.done", Data: []byte(`{"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"fc2","call_id":"c2","name":"g","arguments":"{\"y\":2}","status":"completed"}}`)},
		{Name: "response.completed", Data: []byte(`{"type":"response.completed","response":{"id":"r2","status":"completed"}}`)},
	}

	for _, rm := range allReleaseModes {
		t.Run("prefix_suffix/"+rm.name, func(t *testing.T) {
			plan, err := bridge.Build(newReq(), ir.FormatAnthropic, ir.FormatResponses, bridge.Options{ToolRelease: rm.rel})
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			st := plan.NewStream()
			var out []byte
			for _, ev := range suffixEvents {
				out, err = st.Feed(out, ev)
				if err != nil {
					t.Fatalf("Feed: %v", err)
				}
			}
			out, err = st.End(out, ir.EndEOF)
			if err != nil {
				t.Fatalf("End: %v", err)
			}
			parsed := parseAnthropicSSE(t, string(out))
			if parsed.Errors != 0 || !parsed.MessageStop || len(parsed.Blocks) != 2 {
				t.Fatalf("release %s: unexpected stream errors=%d stop=%v blocks=%+v\n%s", rm.name, parsed.Errors, parsed.MessageStop, parsed.Blocks, string(out))
			}
			var a0, a1 map[string]any
			if err := json.Unmarshal([]byte(parsed.Blocks[0].Input), &a0); err != nil || a0["x"] != float64(1) {
				t.Fatalf("release %s: block 0 want {\"x\":1}, got %q (err=%v)", rm.name, parsed.Blocks[0].Input, err)
			}
			if err := json.Unmarshal([]byte(parsed.Blocks[1].Input), &a1); err != nil || a1["y"] != float64(2) {
				t.Fatalf("release %s: block 1 want {\"y\":2}, got %q (err=%v)", rm.name, parsed.Blocks[1].Input, err)
			}
		})
	}

	// Subtest C: Streamed function args disagreeing with final Arguments must abort with error.
	t.Run("non_prefix_mismatch_aborts", func(t *testing.T) {
		enc := anthropic.NewStreamEncoder(anthropic.EncodeOptions{Env: &ir.Env{}, Model: "m"})
		var dst []byte
		var err error
		for _, ev := range []ir.Event{
			{Type: ir.EvToolStart, Index: 0, ID: "c1", Name: "f", Kind: ir.ToolFunction},
			{Type: ir.EvToolArgs, Index: 0, ID: "c1", Text: `{"a":1}`},
			{Type: ir.EvToolEnd, Index: 0, ID: "c1", Arguments: json.RawMessage(`{"b":2}`)},
			{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopToolUse}},
		} {
			dst, err = enc.Encode(dst, ev)
			if err != nil {
				t.Fatalf("Encode: %v", err)
			}
		}
		dst, _ = enc.Close(dst)
		parsed := parseAnthropicSSE(t, string(dst))
		if parsed.Errors == 0 || parsed.MessageStop {
			t.Fatalf("expected error abort on mismatched function args, got errors=%d stop=%v:\n%s", parsed.Errors, parsed.MessageStop, string(dst))
		}
	})
}

// FixA5 #2: Chat encoder custom tool & function tool suffix completion,
// prefix mismatch error, and unclosed tool error on normal EvStop.
func TestReview5_02_ChatCustomAndFunctionSuffixAndMismatch(t *testing.T) {
	newReq := func() *ir.Request {
		return &ir.Request{
			Model: "m", Stream: true,
			Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{{Type: ir.PartText, Text: "go"}}}},
			Tools: []ir.Tool{
				{Name: "exec", Kind: ir.ToolCustom},
				{Name: "f", Kind: ir.ToolFunction, Schema: json.RawMessage(`{"type":"object","properties":{"x":{"type":"integer"}}}`)},
			},
		}
	}

	// Subtest A: Responses -> Chat custom tool delta "hel", final "hello" across all 4 release modes.
	customSuffixEvents := []sse.Event{
		{Name: "response.created", Data: []byte(`{"type":"response.created","response":{"id":"r1","status":"in_progress"}}`)},
		{Name: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","output_index":0,"item":{"type":"custom_tool_call","id":"ctc1","call_id":"c1","name":"exec","input":"","status":"in_progress"}}`)},
		{Name: "response.custom_tool_call_input.delta", Data: []byte(`{"type":"response.custom_tool_call_input.delta","output_index":0,"delta":"hel"}`)},
		{Name: "response.output_item.done", Data: []byte(`{"type":"response.output_item.done","output_index":0,"item":{"type":"custom_tool_call","id":"ctc1","call_id":"c1","name":"exec","input":"hello","status":"completed"}}`)},
		{Name: "response.completed", Data: []byte(`{"type":"response.completed","response":{"id":"r1","status":"completed"}}`)},
	}

	for _, rm := range allReleaseModes {
		t.Run("custom_suffix/"+rm.name, func(t *testing.T) {
			plan, err := bridge.Build(newReq(), ir.FormatChat, ir.FormatResponses, bridge.Options{ToolRelease: rm.rel})
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			st := plan.NewStream()
			var out []byte
			for _, ev := range customSuffixEvents {
				out, err = st.Feed(out, ev)
				if err != nil {
					t.Fatalf("Feed: %v", err)
				}
			}
			out, err = st.End(out, ir.EndEOF)
			if err != nil {
				t.Fatalf("End: %v", err)
			}
			argsBySlot, sawDone, sawErr := parseChatToolArgs(t, string(out))
			if sawErr || !sawDone {
				t.Fatalf("release %s: expected clean [DONE], got sawErr=%v sawDone=%v\n%s", rm.name, sawErr, sawDone, string(out))
			}
			var got map[string]any
			if err := json.Unmarshal([]byte(argsBySlot[0]), &got); err != nil || got["input"] != "hello" {
				t.Fatalf("release %s: want {\"input\":\"hello\"}, got %q (err=%v)\n%s", rm.name, argsBySlot[0], err, string(out))
			}
		})
	}

	// Subtest B: Responses -> Chat function tool delta "{\"x\":" + final "{\"x\":1}" across all 4 release modes.
	funcSuffixEvents := []sse.Event{
		{Name: "response.created", Data: []byte(`{"type":"response.created","response":{"id":"r2","status":"in_progress"}}`)},
		{Name: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc1","call_id":"c1","name":"f","arguments":"","status":"in_progress"}}`)},
		{Name: "response.function_call_arguments.delta", Data: []byte(`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"x\":"}`)},
		{Name: "response.output_item.done", Data: []byte(`{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc1","call_id":"c1","name":"f","arguments":"{\"x\":1}","status":"completed"}}`)},
		{Name: "response.completed", Data: []byte(`{"type":"response.completed","response":{"id":"r2","status":"completed"}}`)},
	}

	for _, rm := range allReleaseModes {
		t.Run("function_suffix/"+rm.name, func(t *testing.T) {
			plan, err := bridge.Build(newReq(), ir.FormatChat, ir.FormatResponses, bridge.Options{ToolRelease: rm.rel})
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			st := plan.NewStream()
			var out []byte
			for _, ev := range funcSuffixEvents {
				out, err = st.Feed(out, ev)
				if err != nil {
					t.Fatalf("Feed: %v", err)
				}
			}
			out, err = st.End(out, ir.EndEOF)
			if err != nil {
				t.Fatalf("End: %v", err)
			}
			argsBySlot, sawDone, sawErr := parseChatToolArgs(t, string(out))
			if sawErr || !sawDone {
				t.Fatalf("release %s: expected clean [DONE], got sawErr=%v sawDone=%v\n%s", rm.name, sawErr, sawDone, string(out))
			}
			var got map[string]any
			if err := json.Unmarshal([]byte(argsBySlot[0]), &got); err != nil || got["x"] != float64(1) {
				t.Fatalf("release %s: want {\"x\":1}, got %q (err=%v)\n%s", rm.name, argsBySlot[0], err, string(out))
			}
		})
	}

	// Subtest C: Direct Chat encoder non-prefix mismatches and unclosed custom tool on EvStop must error without [DONE].
	t.Run("non_prefix_custom_errors", func(t *testing.T) {
		enc := chat.NewStreamEncoder(chat.EncodeOptions{Env: &ir.Env{}, Model: "m"})
		var dst []byte
		for _, ev := range []ir.Event{
			{Type: ir.EvToolStart, Index: 0, ID: "c1", Name: "exec", Kind: ir.ToolCustom},
			{Type: ir.EvToolArgs, Index: 0, ID: "c1", Text: "xyz"},
			{Type: ir.EvToolEnd, Index: 0, ID: "c1", Arguments: json.RawMessage(`"hello"`)},
			{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopToolUse}},
		} {
			dst, _ = enc.Encode(dst, ev)
		}
		dst, _ = enc.Close(dst)
		_, sawDone, sawErr := parseChatToolArgs(t, string(dst))
		if !sawErr || sawDone {
			t.Fatalf("expected stream error without [DONE], got sawErr=%v sawDone=%v:\n%s", sawErr, sawDone, string(dst))
		}
	})

	t.Run("non_prefix_function_errors", func(t *testing.T) {
		enc := chat.NewStreamEncoder(chat.EncodeOptions{Env: &ir.Env{}, Model: "m"})
		var dst []byte
		for _, ev := range []ir.Event{
			{Type: ir.EvToolStart, Index: 0, ID: "c1", Name: "f", Kind: ir.ToolFunction},
			{Type: ir.EvToolArgs, Index: 0, ID: "c1", Text: `{"a":1}`},
			{Type: ir.EvToolEnd, Index: 0, ID: "c1", Arguments: json.RawMessage(`{"b":2}`)},
			{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopToolUse}},
		} {
			dst, _ = enc.Encode(dst, ev)
		}
		dst, _ = enc.Close(dst)
		_, sawDone, sawErr := parseChatToolArgs(t, string(dst))
		if !sawErr || sawDone {
			t.Fatalf("expected stream error without [DONE], got sawErr=%v sawDone=%v:\n%s", sawErr, sawDone, string(dst))
		}
	})

	t.Run("unclosed_custom_on_stop_errors", func(t *testing.T) {
		enc := chat.NewStreamEncoder(chat.EncodeOptions{Env: &ir.Env{}, Model: "m"})
		var dst []byte
		for _, ev := range []ir.Event{
			{Type: ir.EvToolStart, Index: 0, ID: "c1", Name: "exec", Kind: ir.ToolCustom},
			{Type: ir.EvToolArgs, Index: 0, ID: "c1", Text: "hel"},
			{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopToolUse}},
		} {
			dst, _ = enc.Encode(dst, ev)
		}
		dst, _ = enc.Close(dst)
		_, sawDone, sawErr := parseChatToolArgs(t, string(dst))
		if !sawErr || sawDone {
			t.Fatalf("expected stream error without [DONE] for unclosed custom tool, got sawErr=%v sawDone=%v:\n%s", sawErr, sawDone, string(dst))
		}
	})
}

// FixA5 #3: Responses encoder multi-candidate same call ID routing & EvStop
// candidate isolation across all 4 release modes.
func TestReview5_03_ResponsesMultiCandidateSameIDAndStopIsolation(t *testing.T) {
	// Subtest A: Candidates=2 reusing ID "same" (cand 0 custom "hello", cand 1 function {"x":1}), cand 0 stops first.
	for _, rm := range allReleaseModes {
		t.Run("same_id_isolated/"+rm.name, func(t *testing.T) {
			n := 2
			req := &ir.Request{
				Model: "m", Stream: true, Candidates: &n,
				Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{{Type: ir.PartText, Text: "go"}}}},
				Tools: []ir.Tool{
					{Name: "exec", Kind: ir.ToolCustom},
					{Name: "f", Kind: ir.ToolFunction, Schema: json.RawMessage(`{"type":"object"}`)},
				},
			}
			plan, err := bridge.Build(req, ir.FormatResponses, ir.FormatChat, bridge.Options{ToolRelease: rm.rel})
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			st := plan.NewStream()
			chunks := []string{
				`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"same","type":"function","function":{"name":"exec","arguments":"{\"input\":\"hello\"}"}}]}},{"index":1,"delta":{"tool_calls":[{"index":0,"id":"same","type":"function","function":{"name":"f","arguments":"{\"x\":1}"}}]}}]}`,
				`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
				`{"choices":[{"index":1,"delta":{},"finish_reason":"tool_calls"}]}`,
			}
			var out []byte
			for _, ch := range chunks {
				out, err = st.Feed(out, sse.Event{Data: []byte(ch)})
				if err != nil {
					t.Fatalf("Feed: %v", err)
				}
			}
			out, err = st.End(out, ir.EndDone)
			if err != nil {
				t.Fatalf("End: %v", err)
			}
			s := string(out)
			if !strings.Contains(s, `"response.completed"`) {
				t.Fatalf("release %s: expected response.completed, got:\n%s", rm.name, s)
			}
			if !strings.Contains(s, `"type":"custom_tool_call"`) || !strings.Contains(s, `"input":"hello"`) {
				t.Fatalf("release %s: cand 0 custom_tool_call corrupted:\n%s", rm.name, s)
			}
			if !strings.Contains(s, `"type":"function_call"`) || !strings.Contains(s, `"arguments":"{\"x\":1}"`) {
				t.Fatalf("release %s: cand 1 function_call corrupted:\n%s", rm.name, s)
			}
			if strings.Contains(s, `hello{"x":1}`) || strings.Contains(s, `hello{\"x\":1}`) {
				t.Fatalf("release %s: cross-candidate argument contamination detected:\n%s", rm.name, s)
			}
		})
	}

	// Subtest B: Cand 1 EvStop while Cand 0 is still streaming custom tool must NOT prematurely close Cand 0's tool.
	for _, rm := range allReleaseModes {
		t.Run("other_candidate_stop/"+rm.name, func(t *testing.T) {
			n := 2
			req := &ir.Request{
				Model: "m", Stream: true, Candidates: &n,
				Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{{Type: ir.PartText, Text: "go"}}}},
				Tools:    []ir.Tool{{Name: "exec", Kind: ir.ToolCustom}},
			}
			plan, err := bridge.Build(req, ir.FormatResponses, ir.FormatChat, bridge.Options{ToolRelease: rm.rel})
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			st := plan.NewStream()
			chunks := []string{
				`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"exec","arguments":"{\"input\":\"hel"}}]}}]}`,
				`{"choices":[{"index":1,"delta":{},"finish_reason":"stop"}]}`,
				`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"lo\"}"}}]},"finish_reason":"tool_calls"}]}`,
			}
			var out []byte
			for _, ch := range chunks {
				out, err = st.Feed(out, sse.Event{Data: []byte(ch)})
				if err != nil {
					t.Fatalf("Feed: %v", err)
				}
			}
			out, err = st.End(out, ir.EndDone)
			if err != nil {
				t.Fatalf("End: %v", err)
			}
			s := string(out)
			if !strings.Contains(s, `"response.completed"`) || !strings.Contains(s, `"input":"hello"`) {
				t.Fatalf("release %s: expected completed custom_tool_call with input 'hello', got:\n%s", rm.name, s)
			}
		})
	}
}

// FixA5 Horizontal 4x4 matrix checks: Responses suffix delta / mismatch / early
// close + Gemini unclosed tool / multi-candidate same-ID / final-only args.
func TestReview5_04_HorizontalMatrix_ResponsesAndGeminiGaps(t *testing.T) {
	// 1. Responses encoder emits missing suffix delta so accumulated deltas match final input/arguments.
	t.Run("responses_suffix_delta_emitted", func(t *testing.T) {
		enc := responses.NewStreamEncoder(responses.EncodeOptions{Env: &ir.Env{}, Model: "m"})
		var dst []byte
		for _, ev := range []ir.Event{
			{Type: ir.EvToolStart, Index: 0, ID: "c1", Name: "exec", Kind: ir.ToolCustom},
			{Type: ir.EvToolArgs, Index: 0, ID: "c1", Text: "hel"},
			{Type: ir.EvToolEnd, Index: 0, ID: "c1", Arguments: json.RawMessage(`"hello"`)},
			{Type: ir.EvToolStart, Index: 0, ID: "c2", Name: "f", Kind: ir.ToolFunction},
			{Type: ir.EvToolArgs, Index: 0, ID: "c2", Text: `{"x":`},
			{Type: ir.EvToolEnd, Index: 0, ID: "c2", Arguments: json.RawMessage(`{"x":1}`)},
			{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopToolUse}},
		} {
			dst, _ = enc.Encode(dst, ev)
		}
		dst, _ = enc.Close(dst)
		s := string(dst)
		if !strings.Contains(s, `"delta":"lo"`) {
			t.Fatalf("expected missing custom tool suffix delta 'lo' in Responses SSE:\n%s", s)
		}
		if !strings.Contains(s, `"delta":"1}"`) {
			t.Fatalf("expected missing function tool suffix delta '1}' in Responses SSE:\n%s", s)
		}
		if !strings.Contains(s, `"response.completed"`) {
			t.Fatalf("expected response.completed:\n%s", s)
		}
	})

	// 2. Responses encoder non-prefix mismatch fails stream instead of silently overwriting.
	t.Run("responses_non_prefix_fails", func(t *testing.T) {
		enc := responses.NewStreamEncoder(responses.EncodeOptions{Env: &ir.Env{}, Model: "m"})
		var dst []byte
		for _, ev := range []ir.Event{
			{Type: ir.EvToolStart, Index: 0, ID: "c1", Name: "exec", Kind: ir.ToolCustom},
			{Type: ir.EvToolArgs, Index: 0, ID: "c1", Text: "xyz"},
			{Type: ir.EvToolEnd, Index: 0, ID: "c1", Arguments: json.RawMessage(`"hello"`)},
			{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopToolUse}},
		} {
			dst, _ = enc.Encode(dst, ev)
		}
		dst, _ = enc.Close(dst)
		s := string(dst)
		if !strings.Contains(s, `"response.failed"`) || strings.Contains(s, `"response.completed"`) {
			t.Fatalf("expected response.failed on non-prefix custom tool input, got:\n%s", s)
		}
	})

	// 3. Responses encoder early closeItem by text before EvToolEnd must NOT mark call completed.
	t.Run("responses_early_close_fails", func(t *testing.T) {
		enc := responses.NewStreamEncoder(responses.EncodeOptions{Env: &ir.Env{}, Model: "m"})
		var dst []byte
		for _, ev := range []ir.Event{
			{Type: ir.EvToolStart, Index: 0, ID: "c1", Name: "exec", Kind: ir.ToolCustom},
			{Type: ir.EvToolArgs, Index: 0, ID: "c1", Text: "hel"},
			{Type: ir.EvText, Index: 0, Text: "note"},
			{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopEndTurn}},
		} {
			dst, _ = enc.Encode(dst, ev)
		}
		dst, _ = enc.Close(dst)
		s := string(dst)
		if strings.Contains(s, `"status":"completed","type":"custom_tool_call"`) || strings.Contains(s, `"response.completed"`) {
			t.Fatalf("unfinished tool item must not be marked completed on early close:\n%s", s)
		}
		if !strings.Contains(s, `"response.failed"`) {
			t.Fatalf("expected response.failed on interrupted tool item:\n%s", s)
		}
	})

	// 4. Gemini encoder unclosed tool call at normal EvStop must NOT fabricate completed functionCall.
	t.Run("gemini_unclosed_call_not_completed", func(t *testing.T) {
		enc := gemini.NewStreamEncoder(gemini.EncodeOptions{Env: &ir.Env{}, Model: "m", AltSSE: true})
		var dst []byte
		for _, ev := range []ir.Event{
			{Type: ir.EvToolStart, Index: 0, ID: "c1", Name: "f", Kind: ir.ToolFunction},
			{Type: ir.EvToolArgs, Index: 0, ID: "c1", Text: `{"x":`},
			{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopToolUse}},
		} {
			dst, _ = enc.Encode(dst, ev)
		}
		dst, _ = enc.Close(dst)
		s := string(dst)
		if strings.Contains(s, `"functionCall"`) {
			t.Fatalf("unclosed tool call must not emit completed functionCall in Gemini:\n%s", s)
		}
		if !strings.Contains(s, `"error"`) {
			t.Fatalf("expected Gemini stream error for unclosed tool call:\n%s", s)
		}
	})

	// 5. Gemini encoder final-only args + multi-candidate same call ID across all 4 release modes.
	for _, rm := range allReleaseModes {
		t.Run("gemini_multi_cand_same_id/"+rm.name, func(t *testing.T) {
			n := 2
			req := &ir.Request{
				Model: "m", Stream: true, Candidates: &n,
				Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{{Type: ir.PartText, Text: "go"}}}},
				Tools: []ir.Tool{
					{Name: "exec", Kind: ir.ToolCustom},
					{Name: "f", Kind: ir.ToolFunction, Schema: json.RawMessage(`{"type":"object"}`)},
				},
			}
			plan, err := bridge.Build(req, ir.FormatGemini, ir.FormatChat, bridge.Options{ToolRelease: rm.rel, AltSSE: true})
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			st := plan.NewStream()
			chunks := []string{
				`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"same","type":"function","function":{"name":"exec","arguments":"{\"input\":\"hello\"}"}}]}},{"index":1,"delta":{"tool_calls":[{"index":0,"id":"same","type":"function","function":{"name":"f","arguments":"{\"x\":1}"}}]}}]}`,
				`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"},{"index":1,"delta":{},"finish_reason":"tool_calls"}]}`,
			}
			var out []byte
			for _, ch := range chunks {
				out, err = st.Feed(out, sse.Event{Data: []byte(ch)})
				if err != nil {
					t.Fatalf("Feed: %v", err)
				}
			}
			out, err = st.End(out, ir.EndDone)
			if err != nil {
				t.Fatalf("End: %v", err)
			}
			s := string(out)
			if !strings.Contains(s, `"args":{"input":"hello"}`) || !strings.Contains(s, `"args":{"x":1}`) {
				t.Fatalf("release %s: expected isolated Gemini args for both candidates, got:\n%s", rm.name, s)
			}
		})
	}
}

// Codex fixA5 P2 follow-ups (#1 leading-only whitespace trim, #2 exact big-int
// JSON equality, #3 other-candidate text deferred during open tool call, #4
// Close fails when any queued pending tool is unfinished).
func TestReview5_05_CodexFixA5_P2Fixes(t *testing.T) {
	// #1: Leading whitespace in sent must not strip significant trailing space inside a JSON string.
	t.Run("trailing_space_inside_json_string_not_duplicated", func(t *testing.T) {
		enc := chat.NewStreamEncoder(chat.EncodeOptions{Env: &ir.Env{}, Model: "m"})
		var dst []byte
		for _, ev := range []ir.Event{
			{Type: ir.EvToolStart, Index: 0, ID: "c1", Name: "f", Kind: ir.ToolFunction},
			{Type: ir.EvToolArgs, Index: 0, ID: "c1", Text: ` {"x":"a `},
			{Type: ir.EvToolEnd, Index: 0, ID: "c1", Arguments: json.RawMessage(`{"x":"a b"}`)},
			{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopToolUse}},
		} {
			dst, _ = enc.Encode(dst, ev)
		}
		dst, _ = enc.Close(dst)
		argsBySlot, sawDone, sawErr := parseChatToolArgs(t, string(dst))
		if sawErr || !sawDone {
			t.Fatalf("expected clean [DONE], got sawErr=%v sawDone=%v:\n%s", sawErr, sawDone, string(dst))
		}
		var got map[string]any
		if err := json.Unmarshal([]byte(argsBySlot[0]), &got); err != nil || got["x"] != "a b" {
			t.Fatalf("expected x=\"a b\" (single space), got %q (parsed=%v, err=%v)", argsBySlot[0], got["x"], err)
		}
	})

	// #2: Distinct large integers (2^53 vs 2^53+1) must NOT compare equal via float64.
	t.Run("large_integer_mismatch_not_treated_as_equal", func(t *testing.T) {
		enc := chat.NewStreamEncoder(chat.EncodeOptions{Env: &ir.Env{}, Model: "m"})
		var dst []byte
		for _, ev := range []ir.Event{
			{Type: ir.EvToolStart, Index: 0, ID: "c1", Name: "f", Kind: ir.ToolFunction},
			{Type: ir.EvToolArgs, Index: 0, ID: "c1", Text: `{"x":9007199254740992}`},
			{Type: ir.EvToolEnd, Index: 0, ID: "c1", Arguments: json.RawMessage(`{"x":9007199254740993}`)},
			{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopToolUse}},
		} {
			dst, _ = enc.Encode(dst, ev)
		}
		dst, _ = enc.Close(dst)
		_, sawDone, sawErr := parseChatToolArgs(t, string(dst))
		if !sawErr || sawDone {
			t.Fatalf("expected mismatch error for 9007199254740992 vs 9007199254740993, got sawErr=%v sawDone=%v:\n%s", sawErr, sawDone, string(dst))
		}
	})

	// #3: Candidate 1 text arriving while Candidate 0 has an open tool call must not abort Candidate 0's tool.
	t.Run("other_candidate_text_does_not_abort_open_responses_tool", func(t *testing.T) {
		enc := responses.NewStreamEncoder(responses.EncodeOptions{Env: &ir.Env{}, Model: "m"})
		var dst []byte
		for _, ev := range []ir.Event{
			{Type: ir.EvToolStart, Index: 0, ID: "a", Name: "f", Kind: ir.ToolFunction},
			{Type: ir.EvToolArgs, Index: 0, ID: "a", Text: `{"x":`},
			{Type: ir.EvText, Index: 1, Text: "ok"},
			{Type: ir.EvToolArgs, Index: 0, ID: "a", Text: `1}`},
			{Type: ir.EvToolEnd, Index: 0, ID: "a", Arguments: json.RawMessage(`{"x":1}`)},
			{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopToolUse}},
			{Type: ir.EvStop, Index: 1, Stop: &ir.Stop{Reason: ir.StopEndTurn}},
		} {
			dst, _ = enc.Encode(dst, ev)
		}
		dst, _ = enc.Close(dst)
		s := string(dst)
		if strings.Contains(s, `"response.failed"`) || !strings.Contains(s, `"response.completed"`) {
			t.Fatalf("other candidate text aborted valid Responses tool call:\n%s", s)
		}
		if !strings.Contains(s, `"arguments":"{\"x\":1}"`) || !strings.Contains(s, `"text":"ok"`) {
			t.Fatalf("expected both completed tool call and deferred text, got:\n%s", s)
		}
	})

	// #4: Responses Close() must fail when queue contains [done B, unfinished C] after A is aborted.
	t.Run("responses_close_fails_when_later_queued_tool_unfinished", func(t *testing.T) {
		enc := responses.NewStreamEncoder(responses.EncodeOptions{Env: &ir.Env{}, Model: "m"})
		var dst []byte
		for _, ev := range []ir.Event{
			{Type: ir.EvToolStart, Index: 0, ID: "A", Name: "fa", Kind: ir.ToolFunction},
			{Type: ir.EvToolStart, Index: 0, ID: "B", Name: "fb", Kind: ir.ToolFunction},
			{Type: ir.EvToolStart, Index: 0, ID: "C", Name: "fc", Kind: ir.ToolFunction},
			{Type: ir.EvToolEnd, Index: 0, ID: "B", Arguments: json.RawMessage(`{}`)},
			{Type: ir.EvToolEnd, Index: 0, ID: "A", Err: &ir.APIError{Message: "cancelled"}},
		} {
			dst, _ = enc.Encode(dst, ev)
		}
		dst, _ = enc.Close(dst)
		s := string(dst)
		if strings.Contains(s, `"response.completed"`) {
			t.Fatalf("Close() emitted response.completed with unfinished queued tool C:\n%s", s)
		}
		if !strings.Contains(s, `"response.failed"`) {
			t.Fatalf("expected response.failed when queued tool C never completed:\n%s", s)
		}
	})
}
