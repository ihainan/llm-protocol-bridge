package bridge_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/anthropic"
	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

type parsedAnthropicBlock struct {
	Index   int
	Type    string
	ID      string
	Name    string
	Input   string
	Text    string
	Stopped bool
}

type parsedAnthropicStream struct {
	Blocks      []parsedAnthropicBlock
	Errors      int
	MessageStop bool
}

func parseAnthropicSSE(t *testing.T, raw string) parsedAnthropicStream {
	t.Helper()
	var res parsedAnthropicStream
	byIdx := map[int]*parsedAnthropicBlock{}
	var order []int
	for _, line := range strings.Split(raw, "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		var ev struct {
			Type         string `json:"type"`
			Index        int    `json:"index"`
			ContentBlock struct {
				Type string `json:"type"`
				ID   string `json:"id"`
				Name string `json:"name"`
			} `json:"content_block"`
			Delta struct {
				Type        string `json:"type"`
				PartialJSON string `json:"partial_json"`
				Text        string `json:"text"`
			} `json:"delta"`
		}
		if err := json.Unmarshal([]byte(payload), &ev); err != nil {
			t.Fatalf("bad anthropic SSE JSON %q: %v", payload, err)
		}
		switch ev.Type {
		case "content_block_start":
			b := &parsedAnthropicBlock{
				Index: ev.Index,
				Type:  ev.ContentBlock.Type,
				ID:    ev.ContentBlock.ID,
				Name:  ev.ContentBlock.Name,
			}
			byIdx[ev.Index] = b
			order = append(order, ev.Index)
		case "content_block_delta":
			b := byIdx[ev.Index]
			if b != nil {
				switch ev.Delta.Type {
				case "input_json_delta":
					b.Input += ev.Delta.PartialJSON
				case "text_delta":
					b.Text += ev.Delta.Text
				}
			}
		case "content_block_stop":
			if b := byIdx[ev.Index]; b != nil {
				b.Stopped = true
			}
		case "message_stop":
			res.MessageStop = true
		case "error":
			res.Errors++
		}
	}
	for _, idx := range order {
		res.Blocks = append(res.Blocks, *byIdx[idx])
	}
	return res
}

var allReleaseModes = []struct {
	name string
	rel  ir.ToolRelease
}{
	{"default", ir.ToolReleaseDefault},
	{"buffered", ir.ToolReleaseBuffered},
	{"live_first", ir.ToolReleaseLiveFirst},
	{"stream", ir.ToolReleaseStream},
}

// FixA4 #1 (P1): Anthropic upstream -> Anthropic client (and Responses client)
// custom tool input must be unwrapped by anthropic.StreamDecoder so EvToolArgs
// carries raw text consistently across all 4 decoders and is never double-wrapped.
func TestReview4_01_AnthropicUpstreamCustomToolNotDoubleWrapped(t *testing.T) {
	newReq := func() *ir.Request {
		return &ir.Request{
			Model: "m", Stream: true,
			Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{{Type: ir.PartText, Text: "go"}}}},
			Tools:    []ir.Tool{{Name: "exec", Kind: ir.ToolCustom}},
		}
	}
	events := []sse.Event{
		{Name: "message_start", Data: []byte(`{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"m","content":[]}}`)},
		{Name: "content_block_start", Data: []byte(`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_a","name":"exec","input":{}}}`)},
		{Name: "content_block_delta", Data: []byte(`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"input\":\"hel"}}`)},
		{Name: "content_block_delta", Data: []byte(`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"lo\\nworld\"}"}}`)},
		{Name: "content_block_stop", Data: []byte(`{"type":"content_block_stop","index":0}`)},
		{Name: "message_delta", Data: []byte(`{"type":"message_delta","delta":{"stop_reason":"tool_use"}}`)},
		{Name: "message_stop", Data: []byte(`{"type":"message_stop"}`)},
	}

	for _, rm := range allReleaseModes {
		t.Run("anthropic_to_anthropic/"+rm.name, func(t *testing.T) {
			plan, err := bridge.Build(newReq(), ir.FormatAnthropic, ir.FormatAnthropic, bridge.Options{ToolRelease: rm.rel})
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			st := plan.NewStream()
			var out []byte
			for _, ev := range events {
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
			if parsed.Errors != 0 || !parsed.MessageStop {
				t.Fatalf("expected clean completion, got errors=%d messageStop=%v:\n%s", parsed.Errors, parsed.MessageStop, string(out))
			}
			if len(parsed.Blocks) != 1 || parsed.Blocks[0].Type != "tool_use" || !parsed.Blocks[0].Stopped {
				t.Fatalf("unexpected blocks: %+v\n%s", parsed.Blocks, string(out))
			}
			var args map[string]any
			if err := json.Unmarshal([]byte(parsed.Blocks[0].Input), &args); err != nil {
				t.Fatalf("invalid tool_use JSON %q: %v", parsed.Blocks[0].Input, err)
			}
			if got, _ := args["input"].(string); got != "hello\nworld" {
				t.Fatalf("want input %q, got %q (raw=%q)", "hello\nworld", got, parsed.Blocks[0].Input)
			}
		})

		t.Run("anthropic_to_responses/"+rm.name, func(t *testing.T) {
			plan, err := bridge.Build(newReq(), ir.FormatResponses, ir.FormatAnthropic, bridge.Options{ToolRelease: rm.rel})
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			st := plan.NewStream()
			var out []byte
			for _, ev := range events {
				out, err = st.Feed(out, ev)
				if err != nil {
					t.Fatalf("Feed: %v", err)
				}
			}
			out, err = st.End(out, ir.EndEOF)
			if err != nil {
				t.Fatalf("End: %v", err)
			}
			s := string(out)
			if strings.Contains(s, `"delta":"{\"input\"`) {
				t.Fatalf("raw JSON wrapper leaked into Responses custom_tool_call_input.delta:\n%s", s)
			}
			if !strings.Contains(s, `"input":"hello\nworld"`) || !strings.Contains(s, `"response.completed"`) {
				t.Fatalf("expected unwrapped custom_tool_call input 'hello\\nworld', got:\n%s", s)
			}
		})
	}
}

// FixA4 #2 (P2): When streamed custom tool input is a strict prefix of the
// final Arguments ("hel" -> "hello", or queued "b" -> "beta"), finishCustomInput
// must emit the remaining suffix; when NOT a prefix, it must abort with an error.
func TestReview4_02_AnthropicCustomToolSuffixAndPrefixMismatch(t *testing.T) {
	newReq := func() *ir.Request {
		return &ir.Request{
			Model: "m", Stream: true,
			Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{{Type: ir.PartText, Text: "go"}}}},
			Tools:    []ir.Tool{{Name: "exec", Kind: ir.ToolCustom}},
		}
	}

	// Responses upstream sends partial delta "hel" for call 0 and "b" for call 1,
	// then output_item.done carries full input "hello" and "beta".
	prefixEvents := []sse.Event{
		{Name: "response.created", Data: []byte(`{"type":"response.created","response":{"id":"r1","status":"in_progress"}}`)},
		{Name: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","output_index":0,"item":{"type":"custom_tool_call","id":"ctc_1","call_id":"c1","name":"exec","input":"","status":"in_progress"}}`)},
		{Name: "response.custom_tool_call_input.delta", Data: []byte(`{"type":"response.custom_tool_call_input.delta","output_index":0,"delta":"hel"}`)},
		{Name: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","output_index":1,"item":{"type":"custom_tool_call","id":"ctc_2","call_id":"c2","name":"exec","input":"","status":"in_progress"}}`)},
		{Name: "response.custom_tool_call_input.delta", Data: []byte(`{"type":"response.custom_tool_call_input.delta","output_index":1,"delta":"b"}`)},
		{Name: "response.output_item.done", Data: []byte(`{"type":"response.output_item.done","output_index":0,"item":{"type":"custom_tool_call","id":"ctc_1","call_id":"c1","name":"exec","input":"hello","status":"completed"}}`)},
		{Name: "response.output_item.done", Data: []byte(`{"type":"response.output_item.done","output_index":1,"item":{"type":"custom_tool_call","id":"ctc_2","call_id":"c2","name":"exec","input":"beta","status":"completed"}}`)},
		{Name: "response.completed", Data: []byte(`{"type":"response.completed","response":{"id":"r1","status":"completed"}}`)},
	}

	for _, rm := range allReleaseModes {
		t.Run("prefix_suffix/"+rm.name, func(t *testing.T) {
			plan, err := bridge.Build(newReq(), ir.FormatAnthropic, ir.FormatResponses, bridge.Options{ToolRelease: rm.rel})
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			st := plan.NewStream()
			var out []byte
			for _, ev := range prefixEvents {
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
			if parsed.Errors != 0 || !parsed.MessageStop {
				t.Fatalf("expected clean completion, got errors=%d messageStop=%v:\n%s", parsed.Errors, parsed.MessageStop, string(out))
			}
			if len(parsed.Blocks) != 2 {
				t.Fatalf("expected 2 tool blocks, got %+v\n%s", parsed.Blocks, string(out))
			}
			want := []string{"hello", "beta"}
			for i, w := range want {
				var args map[string]any
				if err := json.Unmarshal([]byte(parsed.Blocks[i].Input), &args); err != nil {
					t.Fatalf("block %d invalid JSON %q: %v", i, parsed.Blocks[i].Input, err)
				}
				if got, _ := args["input"].(string); got != w {
					t.Fatalf("block %d: want %q, got %q (raw=%q)", i, w, got, parsed.Blocks[i].Input)
				}
			}
		})
	}

	// Non-prefix final input must abort with error rather than silently truncating or corrupting.
	t.Run("non_prefix_direct_encoder", func(t *testing.T) {
		enc := anthropic.NewStreamEncoder(anthropic.EncodeOptions{Env: &ir.Env{}, Model: "m"})
		var dst []byte
		var err error
		for _, ev := range []ir.Event{
			{Type: ir.EvToolStart, Index: 0, ID: "c1", Name: "exec", Kind: ir.ToolCustom},
			{Type: ir.EvToolArgs, Index: 0, ID: "c1", Text: "xyz"},
			{Type: ir.EvToolEnd, Index: 0, ID: "c1", Arguments: json.RawMessage(`"hello"`)},
			{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopToolUse}},
		} {
			dst, err = enc.Encode(dst, ev)
			if err != nil {
				t.Fatalf("Encode: %v", err)
			}
		}
		dst, err = enc.Close(dst)
		if err != nil {
			t.Fatalf("Close: %v", err)
		}
		parsed := parseAnthropicSSE(t, string(dst))
		if parsed.Errors == 0 || parsed.MessageStop {
			t.Fatalf("expected stream error abort on non-prefix final input, got errors=%d messageStop=%v:\n%s", parsed.Errors, parsed.MessageStop, string(dst))
		}
		for _, b := range parsed.Blocks {
			if b.Stopped {
				t.Fatalf("aborted tool block must not emit content_block_stop:\n%s", string(dst))
			}
		}
	})

	// Non-prefix final input via bridge.Stream across LiveFirst and Stream modes must abort with error.
	for _, rm := range []struct {
		name string
		rel  ir.ToolRelease
	}{
		{"live_first", ir.ToolReleaseLiveFirst},
		{"stream", ir.ToolReleaseStream},
	} {
		t.Run("non_prefix_bridge/"+rm.name, func(t *testing.T) {
			plan, err := bridge.Build(newReq(), ir.FormatAnthropic, ir.FormatResponses, bridge.Options{ToolRelease: rm.rel})
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			st := plan.NewStream()
			badEvents := []sse.Event{
				{Name: "response.created", Data: []byte(`{"type":"response.created","response":{"id":"r2","status":"in_progress"}}`)},
				{Name: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","output_index":0,"item":{"type":"custom_tool_call","id":"ctc_1","call_id":"c1","name":"exec","input":"","status":"in_progress"}}`)},
				{Name: "response.custom_tool_call_input.delta", Data: []byte(`{"type":"response.custom_tool_call_input.delta","output_index":0,"delta":"xyz"}`)},
				{Name: "response.output_item.done", Data: []byte(`{"type":"response.output_item.done","output_index":0,"item":{"type":"custom_tool_call","id":"ctc_1","call_id":"c1","name":"exec","input":"hello","status":"completed"}}`)},
				{Name: "response.completed", Data: []byte(`{"type":"response.completed","response":{"id":"r2","status":"completed"}}`)},
			}
			var out []byte
			for _, ev := range badEvents {
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
			if parsed.Errors == 0 || parsed.MessageStop {
				t.Fatalf("release %s: expected error abort on non-prefix final input, got errors=%d messageStop=%v:\n%s", rm.name, parsed.Errors, parsed.MessageStop, string(out))
			}
		})
	}
}

// FixA4 #3 (P2): Early closeBlock triggered by interleaved text or another
// candidate's EvStop must never leave a custom tool_use block with invalid JSON.
func TestReview4_03_AnthropicCustomToolEarlyCloseNoInvalidJSON(t *testing.T) {
	// Subtest A: Candidate 1's EvStop while Candidate 0 is streaming a custom tool
	// must NOT close Candidate 0's tool block; all 4 release modes must deliver valid {"input":"hello"}.
	for _, rm := range allReleaseModes {
		t.Run("other_candidate_stop/"+rm.name, func(t *testing.T) {
			n := 2
			req := &ir.Request{
				Model: "m", Stream: true, Candidates: &n,
				Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{{Type: ir.PartText, Text: "go"}}}},
				Tools:    []ir.Tool{{Name: "exec", Kind: ir.ToolCustom}},
			}
			plan, err := bridge.Build(req, ir.FormatAnthropic, ir.FormatChat, bridge.Options{ToolRelease: rm.rel})
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
			parsed := parseAnthropicSSE(t, string(out))
			if parsed.Errors != 0 || !parsed.MessageStop {
				t.Fatalf("release %s: expected clean completion, got errors=%d messageStop=%v:\n%s", rm.name, parsed.Errors, parsed.MessageStop, string(out))
			}
			if len(parsed.Blocks) != 1 || !parsed.Blocks[0].Stopped {
				t.Fatalf("release %s: expected 1 stopped tool block, got %+v\n%s", rm.name, parsed.Blocks, string(out))
			}
			var args map[string]any
			if err := json.Unmarshal([]byte(parsed.Blocks[0].Input), &args); err != nil {
				t.Fatalf("release %s: invalid tool_use JSON %q: %v\n%s", rm.name, parsed.Blocks[0].Input, err, string(out))
			}
			if got, _ := args["input"].(string); got != "hello" {
				t.Fatalf("release %s: want input %q, got %q", rm.name, "hello", got)
			}
		})
	}

	// Subtest B: Text interleaved before custom tool finishes in all 4 release modes.
	// Default/Buffered/LiveFirst queue the text behind the tool and complete cleanly;
	// Stream mode cannot pause/resume an Anthropic tool_use block and must abort with error
	// rather than closing the block with invalid JSON.
	for _, rm := range allReleaseModes {
		t.Run("interleaved_text/"+rm.name, func(t *testing.T) {
			req := &ir.Request{
				Model: "m", Stream: true,
				Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{{Type: ir.PartText, Text: "go"}}}},
				Tools:    []ir.Tool{{Name: "exec", Kind: ir.ToolCustom}},
			}
			plan, err := bridge.Build(req, ir.FormatAnthropic, ir.FormatResponses, bridge.Options{ToolRelease: rm.rel})
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			st := plan.NewStream()
			events := []sse.Event{
				{Name: "response.created", Data: []byte(`{"type":"response.created","response":{"id":"r1","status":"in_progress"}}`)},
				{Name: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","output_index":0,"item":{"type":"custom_tool_call","id":"ctc_1","call_id":"c1","name":"exec","input":"","status":"in_progress"}}`)},
				{Name: "response.custom_tool_call_input.delta", Data: []byte(`{"type":"response.custom_tool_call_input.delta","output_index":0,"delta":"hel"}`)},
				{Name: "response.output_text.delta", Data: []byte(`{"type":"response.output_text.delta","output_index":1,"delta":"note"}`)},
				{Name: "response.custom_tool_call_input.delta", Data: []byte(`{"type":"response.custom_tool_call_input.delta","output_index":0,"delta":"lo"}`)},
				{Name: "response.output_item.done", Data: []byte(`{"type":"response.output_item.done","output_index":0,"item":{"type":"custom_tool_call","id":"ctc_1","call_id":"c1","name":"exec","input":"hello","status":"completed"}}`)},
				{Name: "response.completed", Data: []byte(`{"type":"response.completed","response":{"id":"r1","status":"completed"}}`)},
			}
			var out []byte
			for _, ev := range events {
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
			// Invariant: NO stopped tool_use block may EVER contain invalid JSON.
			for _, b := range parsed.Blocks {
				if b.Type == "tool_use" && b.Stopped {
					if !json.Valid([]byte(b.Input)) {
						t.Fatalf("release %s: stopped tool_use block has invalid JSON %q:\n%s", rm.name, b.Input, string(out))
					}
				}
			}
			if rm.rel == ir.ToolReleaseStream {
				if parsed.Errors == 0 || parsed.MessageStop {
					t.Fatalf("release stream: expected error abort when text interrupts open tool_use, got errors=%d messageStop=%v:\n%s", parsed.Errors, parsed.MessageStop, string(out))
				}
			} else {
				if parsed.Errors != 0 || !parsed.MessageStop {
					t.Fatalf("release %s: expected clean completion, got errors=%d messageStop=%v:\n%s", rm.name, parsed.Errors, parsed.MessageStop, string(out))
				}
			}
		})
	}
}

// FixA4 #4 (P2): Multi-candidate calls reusing the same call ID ("same") must be
// routed by (candidateIndex, id) so arguments never cross-contaminate across tools.
func TestReview4_04_AnthropicMultiCandidateSameCallIDIsolated(t *testing.T) {
	for _, rm := range allReleaseModes {
		t.Run(rm.name, func(t *testing.T) {
			n := 2
			req := &ir.Request{
				Model: "m", Stream: true, Candidates: &n,
				Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{{Type: ir.PartText, Text: "go"}}}},
				Tools: []ir.Tool{
					{Name: "exec", Kind: ir.ToolCustom},
					{Name: "f", Kind: ir.ToolFunction},
				},
			}
			plan, err := bridge.Build(req, ir.FormatAnthropic, ir.FormatChat, bridge.Options{ToolRelease: rm.rel})
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			st := plan.NewStream()
			first := `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"same","type":"function","function":{"name":"exec","arguments":"{\"input\":\"hello\"}"}}]}},{"index":1,"delta":{"tool_calls":[{"index":0,"id":"same","type":"function","function":{"name":"f","arguments":"{\"x\":1}"}}]}}]}`
			last := `{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"},{"index":1,"delta":{},"finish_reason":"tool_calls"}]}`
			var out []byte
			for _, ch := range []string{first, last} {
				out, err = st.Feed(out, sse.Event{Data: []byte(ch)})
				if err != nil {
					t.Fatalf("Feed: %v", err)
				}
			}
			out, err = st.End(out, ir.EndDone)
			if err != nil {
				t.Fatalf("End: %v", err)
			}
			parsed := parseAnthropicSSE(t, string(out))
			if parsed.Errors != 0 || !parsed.MessageStop {
				t.Fatalf("release %s: expected clean completion, got errors=%d messageStop=%v:\n%s", rm.name, parsed.Errors, parsed.MessageStop, string(out))
			}
			if len(parsed.Blocks) != 2 {
				t.Fatalf("release %s: expected 2 tool blocks, got %+v\n%s", rm.name, parsed.Blocks, string(out))
			}
			// Block 0: exec (custom) -> {"input":"hello"}
			var args0 map[string]any
			if err := json.Unmarshal([]byte(parsed.Blocks[0].Input), &args0); err != nil {
				t.Fatalf("release %s: block 0 invalid JSON %q: %v", rm.name, parsed.Blocks[0].Input, err)
			}
			if got, _ := args0["input"].(string); got != "hello" {
				t.Fatalf("release %s: block 0 want input=%q, got %q (raw=%q)", rm.name, "hello", got, parsed.Blocks[0].Input)
			}
			// Block 1: f (function) -> {"x":1}
			var args1 map[string]any
			if err := json.Unmarshal([]byte(parsed.Blocks[1].Input), &args1); err != nil {
				t.Fatalf("release %s: block 1 invalid JSON %q: %v", rm.name, parsed.Blocks[1].Input, err)
			}
			if got, _ := args1["x"].(float64); got != 1 {
				t.Fatalf("release %s: block 1 want x=1, got %v (raw=%q)", rm.name, args1["x"], parsed.Blocks[1].Input)
			}
		})
	}
}

// FixA4 follow-up (#1 P1 & #2 P2): Non-streaming Gemini & Anthropic responses
// (and streaming Gemini responses) must preserve ToolCustom Input (including
// empty string "") wrapped as {"input": "..."} instead of outputting {}.
func TestReview4_05_NonStreamAndGeminiCustomToolPreservesInputAndEmptyWrapper(t *testing.T) {
	newReq := func(stream bool) *ir.Request {
		return &ir.Request{
			Model: "m", Stream: stream,
			Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{{Type: ir.PartText, Text: "go"}}}},
			Tools:    []ir.Tool{{Name: "exec", Kind: ir.ToolCustom}},
		}
	}

	// 1. Non-streaming Gemini client <- Gemini / Anthropic upstream ("hello" and "").
	for _, tc := range []struct {
		name     string
		upstream ir.Format
		body     string
		want     string
	}{
		{"gemini_to_gemini_hello", ir.FormatGemini, `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"exec","args":{"input":"hello"}}}]},"finishReason":"STOP"}]}`, "hello"},
		{"gemini_to_gemini_empty", ir.FormatGemini, `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"exec","args":{"input":""}}}]},"finishReason":"STOP"}]}`, ""},
		{"anthropic_to_gemini_hello", ir.FormatAnthropic, `{"id":"m1","type":"message","role":"assistant","model":"m","content":[{"type":"tool_use","id":"toolu_1","name":"exec","input":{"input":"hello"}}],"stop_reason":"tool_use"}`, "hello"},
		{"anthropic_to_gemini_empty", ir.FormatAnthropic, `{"id":"m1","type":"message","role":"assistant","model":"m","content":[{"type":"tool_use","id":"toolu_1","name":"exec","input":{"input":""}}],"stop_reason":"tool_use"}`, ""},
	} {
		t.Run("nonstream_gemini/"+tc.name, func(t *testing.T) {
			plan, err := bridge.Build(newReq(false), ir.FormatGemini, tc.upstream, bridge.Options{})
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			out, _, err := plan.ConvertResponse([]byte(tc.body))
			if err != nil {
				t.Fatalf("ConvertResponse: %v", err)
			}
			var doc struct {
				Candidates []struct {
					Content struct {
						Parts []struct {
							FunctionCall *struct {
								Name string         `json:"name"`
								Args map[string]any `json:"args"`
							} `json:"functionCall"`
						} `json:"parts"`
					} `json:"content"`
				} `json:"candidates"`
			}
			if err := json.Unmarshal(out, &doc); err != nil {
				t.Fatalf("unmarshal: %v\n%s", err, string(out))
			}
			fc := doc.Candidates[0].Content.Parts[0].FunctionCall
			if fc == nil {
				t.Fatalf("missing functionCall: %s", string(out))
			}
			rawVal, exists := fc.Args["input"]
			if !exists {
				t.Fatalf("missing required 'input' field in Gemini functionCall.args: %s", string(out))
			}
			if got, _ := rawVal.(string); got != tc.want {
				t.Fatalf("want input=%q, got %q (%s)", tc.want, got, string(out))
			}
		})
	}

	// 2. Non-streaming Anthropic client <- Anthropic / Gemini upstream (including empty {"input":""}).
	for _, tc := range []struct {
		name     string
		upstream ir.Format
		body     string
		want     string
	}{
		{"anthropic_to_anthropic_empty", ir.FormatAnthropic, `{"id":"m1","type":"message","role":"assistant","model":"m","content":[{"type":"tool_use","id":"toolu_1","name":"exec","input":{"input":""}}],"stop_reason":"tool_use"}`, ""},
		{"gemini_to_anthropic_empty", ir.FormatGemini, `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"exec","args":{"input":""}}}]},"finishReason":"STOP"}]}`, ""},
		{"gemini_to_anthropic_hello", ir.FormatGemini, `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"exec","args":{"input":"hello"}}}]},"finishReason":"STOP"}]}`, "hello"},
	} {
		t.Run("nonstream_anthropic/"+tc.name, func(t *testing.T) {
			plan, err := bridge.Build(newReq(false), ir.FormatAnthropic, tc.upstream, bridge.Options{})
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			out, _, err := plan.ConvertResponse([]byte(tc.body))
			if err != nil {
				t.Fatalf("ConvertResponse: %v", err)
			}
			var doc struct {
				Content []struct {
					Type  string         `json:"type"`
					Input map[string]any `json:"input"`
				} `json:"content"`
			}
			if err := json.Unmarshal(out, &doc); err != nil {
				t.Fatalf("unmarshal: %v\n%s", err, string(out))
			}
			rawVal, exists := doc.Content[0].Input["input"]
			if !exists {
				t.Fatalf("missing required 'input' field in Anthropic tool_use.input: %s", string(out))
			}
			if got, _ := rawVal.(string); got != tc.want {
				t.Fatalf("want input=%q, got %q (%s)", tc.want, got, string(out))
			}
		})
	}

	// 3. Streaming Gemini client <- Gemini upstream across all 4 release modes.
	for _, rm := range allReleaseModes {
		t.Run("stream_gemini/"+rm.name, func(t *testing.T) {
			plan, err := bridge.Build(newReq(true), ir.FormatGemini, ir.FormatGemini, bridge.Options{ToolRelease: rm.rel, AltSSE: true})
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			st := plan.NewStream()
			chunk := []byte(`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"exec","args":{"input":"hello"}}}]},"finishReason":"STOP"}]}`)
			var out []byte
			out, err = st.Feed(out, sse.Event{Data: chunk})
			if err != nil {
				t.Fatalf("Feed: %v", err)
			}
			out, err = st.End(out, ir.EndEOF)
			if err != nil {
				t.Fatalf("End: %v", err)
			}
			if !strings.Contains(string(out), `"args":{"input":"hello"}`) {
				t.Fatalf("release %s: expected wrapped Gemini functionCall.args, got:\n%s", rm.name, string(out))
			}
		})
	}
}
