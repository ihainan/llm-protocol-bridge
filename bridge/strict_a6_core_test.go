package bridge_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/anthropic"
	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/responses"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// Reference: service/claude_strict_test.go:1050 (buffered bytes: 8 MiB cap; 4 MiB and 5 MiB must pass without double-counting).
func TestStrictA6_01_BufferedToolBytesNotDoubleCounted(t *testing.T) {
	reqBody := []byte(`{
		"model": "m",
		"max_tokens": 1024,
		"stream": true,
		"messages": [{"role": "user", "content": "run"}],
		"tools": [{"name": "write", "input_schema": {"type": "object"}}]
	}`)
	plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, reqBody, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}

	t.Run("5MiB_single_call_accepted_under_8MiB_default", func(t *testing.T) {
		// 5 MiB would previously fail at 10 MiB > 8 MiB because pipeline.go counted
		// both EvToolArgs and EvToolEnd.Arguments for the same call.
		payload := `{"data":"` + strings.Repeat("a", 5*1024*1024) + `"}`
		s := plan.NewStream()
		chunk1, _ := json.Marshal(map[string]any{
			"id":    "c1",
			"model": "m",
			"choices": []any{map[string]any{
				"index": 0,
				"delta": map[string]any{
					"role": "assistant",
					"tool_calls": []any{map[string]any{
						"index": 0,
						"id":    "call_1",
						"type":  "function",
						"function": map[string]any{
							"name":      "write",
							"arguments": payload,
						},
					}},
				},
			}},
		})
		chunk2 := []byte(`{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`)
		var out []byte
		out, err = s.Feed(out, sse.Event{Data: chunk1})
		if err != nil {
			t.Fatalf("Feed chunk1: %v", err)
		}
		out, err = s.Feed(out, sse.Event{Data: chunk2})
		if err != nil {
			t.Fatalf("Feed chunk2: %v", err)
		}
		out, err = s.End(out, ir.EndDone)
		if err != nil {
			t.Fatalf("End: %v", err)
		}
		parsed := parseAnthropicSSE(t, string(out))
		if parsed.Errors != 0 || !parsed.MessageStop || len(parsed.Blocks) != 1 {
			t.Fatalf("5 MiB single tool call should succeed under 8 MiB limit: errors=%d stop=%v blocks=%d", parsed.Errors, parsed.MessageStop, len(parsed.Blocks))
		}
	})

	t.Run("4MiB_accepted_then_second_4MiB_plus_1_rejected", func(t *testing.T) {
		s := plan.NewStream()
		half := `{"data":"` + strings.Repeat("a", 4*1024*1024) + `"}`
		mkChunk := func(idx int, id, args string) []byte {
			b, _ := json.Marshal(map[string]any{
				"id":    "c2",
				"model": "m",
				"choices": []any{map[string]any{
					"index": 0,
					"delta": map[string]any{
						"tool_calls": []any{map[string]any{
							"index": idx,
							"id":    id,
							"type":  "function",
							"function": map[string]any{
								"name":      "write",
								"arguments": args,
							},
						}},
					},
				}},
			})
			return b
		}
		var out []byte
		out, err = s.Feed(out, sse.Event{Data: mkChunk(0, "call_1", half)})
		if err != nil {
			t.Fatalf("Feed call 1: %v", err)
		}
		if s.Done() {
			t.Fatalf("4 MiB first call must not trigger buffer overflow")
		}
		out, err = s.Feed(out, sse.Event{Data: mkChunk(1, "call_2", half)})
		if err != nil {
			t.Fatalf("Feed call 2: %v", err)
		}
		out, _ = s.Feed(out, sse.Event{Data: []byte(`{"id":"c2","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`)})
		out, _ = s.End(out, ir.EndDone)
		parsed := parseAnthropicSSE(t, string(out))
		if parsed.Errors == 0 || parsed.MessageStop {
			t.Fatalf("expected >8 MiB total buffered calls to fail: errors=%d stop=%v", parsed.Errors, parsed.MessageStop)
		}
	})
}

// Carry-over from fixA5 review: deferredOtherCand in responses/encode.go and
// anthropic/encode.go must be bounded by MaxBufferedBytes.
func TestStrictA6_02_DeferredOtherCandBoundedByMaxBufferedBytes(t *testing.T) {
	t.Run("responses_encoder_direct", func(t *testing.T) {
		enc := responses.NewStreamEncoder(responses.EncodeOptions{
			Env:              &ir.Env{},
			Model:            "m",
			MaxBufferedBytes: 32,
		})
		var dst []byte
		var err error
		for _, ev := range []ir.Event{
			{Type: ir.EvToolStart, Index: 0, ID: "c1", Name: "f", Kind: ir.ToolFunction},
			{Type: ir.EvText, Index: 1, Text: strings.Repeat("x", 20)},
			{Type: ir.EvText, Index: 1, Text: strings.Repeat("y", 20)},
			{Type: ir.EvToolEnd, Index: 0, ID: "c1", Arguments: json.RawMessage(`{}`)},
			{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopToolUse}},
		} {
			dst, err = enc.Encode(dst, ev)
			if err != nil {
				t.Fatalf("Encode: %v", err)
			}
		}
		dst, _ = enc.Close(dst)
		s := string(dst)
		if !strings.Contains(s, `"response.failed"`) || strings.Contains(s, `"response.completed"`) {
			t.Fatalf("expected response.failed on deferredOtherCand overflow, got:\n%s", s)
		}
	})

	t.Run("anthropic_encoder_direct", func(t *testing.T) {
		enc := anthropic.NewStreamEncoder(anthropic.EncodeOptions{
			Env:              &ir.Env{},
			Model:            "m",
			MaxBufferedBytes: 32,
		})
		var dst []byte
		var err error
		for _, ev := range []ir.Event{
			{Type: ir.EvToolStart, Index: 0, ID: "c1", Name: "f", Kind: ir.ToolFunction},
			{Type: ir.EvText, Index: 1, Text: strings.Repeat("x", 20)},
			{Type: ir.EvText, Index: 1, Text: strings.Repeat("y", 20)},
			{Type: ir.EvToolEnd, Index: 0, ID: "c1", Arguments: json.RawMessage(`{}`)},
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
			t.Fatalf("expected Anthropic error abort on deferredOtherCand overflow, got errors=%d stop=%v:\n%s", parsed.Errors, parsed.MessageStop, string(dst))
		}
	})

	t.Run("bridge_build_propagates_limit_to_responses_and_anthropic", func(t *testing.T) {
		n := 2
		req := &ir.Request{
			Model: "m", Stream: true, Candidates: &n,
			Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{{Type: ir.PartText, Text: "go"}}}},
			Tools:    []ir.Tool{{Name: "f", Kind: ir.ToolFunction, Schema: json.RawMessage(`{"type":"object"}`)}},
		}
		for _, clientFmt := range []ir.Format{ir.FormatResponses, ir.FormatAnthropic} {
			plan, err := bridge.Build(req, clientFmt, ir.FormatChat, bridge.Options{
				ToolRelease: ir.ToolReleaseStream,
				Limits:      ir.Limits{MaxBufferedBytes: 32},
			})
			if err != nil {
				t.Fatalf("%s Build: %v", clientFmt, err)
			}
			st := plan.NewStream()
			var out []byte
			chunks := []string{
				`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"f","arguments":"{"}}]}}]}`,
				`{"choices":[{"index":1,"delta":{"content":"` + strings.Repeat("z", 40) + `"}}]}`,
				`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]},"finish_reason":"tool_calls"},{"index":1,"delta":{},"finish_reason":"stop"}]}`,
			}
			for _, ch := range chunks {
				out, err = st.Feed(out, sse.Event{Data: []byte(ch)})
				if err != nil {
					t.Fatalf("%s Feed: %v", clientFmt, err)
				}
			}
			out, _ = st.End(out, ir.EndDone)
			s := string(out)
			if clientFmt == ir.FormatResponses {
				if !strings.Contains(s, `"response.failed"`) || strings.Contains(s, `"response.completed"`) {
					t.Fatalf("Responses bridge stream should fail on deferredOtherCand overflow:\n%s", s)
				}
			} else {
				parsed := parseAnthropicSSE(t, s)
				if parsed.Errors == 0 || parsed.MessageStop {
					t.Fatalf("Anthropic bridge stream should fail on deferredOtherCand overflow:\n%s", s)
				}
			}
		}
	})
}

// Reference: service/claude_strict_test.go:1368 (stop_sequences count > 64 and byte length > 16 KiB under StopEmulate)
// and path-based loss rejection via bridge.Options.CheckLoss.
func TestStrictA6_03_CheckLossAndStopEmulateLimits(t *testing.T) {
	t.Run("check_loss_path_based_rejection", func(t *testing.T) {
		denyMCP := func(l ir.Loss) error {
			if l.Kind == ir.LossUnknownField && l.Path.String() == "mcp_servers" {
				return ir.Unsupported(l.Path, "mcp_servers", "mcp_servers is not supported")
			}
			return nil
		}
		// Unknown safeguards allowed, mcp_servers rejected.
		okBody := []byte(`{"model":"m","max_tokens":64,"safeguards":{"x":1},"messages":[{"role":"user","content":"hi"}]}`)
		if _, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, okBody, bridge.Options{CheckLoss: denyMCP}); err != nil {
			t.Fatalf("safeguards should be allowed: %v", err)
		}
		badBody := []byte(`{"model":"m","max_tokens":64,"mcp_servers":[{"type":"url","url":"https://x","name":"x"}],"messages":[{"role":"user","content":"hi"}]}`)
		if _, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, badBody, bridge.Options{CheckLoss: denyMCP}); err == nil {
			t.Fatalf("expected mcp_servers to be rejected by CheckLoss")
		}
	})

	t.Run("stop_emulate_limits_64_count_and_16KiB_bytes", func(t *testing.T) {
		limits := ir.Limits{MaxStopSequences: 64, MaxStopSequenceBytes: 16 << 10}
		opts := bridge.Options{StopMode: ir.StopEmulate, Limits: limits}

		// 64 stop sequences accepted.
		stops64 := make([]string, 64)
		for i := range stops64 {
			stops64[i] = "s"
		}
		b64, _ := json.Marshal(map[string]any{
			"model": "m", "max_tokens": 64, "stop_sequences": stops64,
			"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		})
		if _, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, b64, opts); err != nil {
			t.Fatalf("64 stop sequences should succeed: %v", err)
		}

		// 65 stop sequences rejected.
		stops65 := append(stops64, "extra")
		b65, _ := json.Marshal(map[string]any{
			"model": "m", "max_tokens": 64, "stop_sequences": stops65,
			"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		})
		if _, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, b65, opts); err == nil {
			t.Fatalf("65 stop sequences should be rejected")
		}

		// 16 KiB + 1 byte rejected.
		bLong, _ := json.Marshal(map[string]any{
			"model": "m", "max_tokens": 64, "stop_sequences": []string{strings.Repeat("x", (16<<10)+1)},
			"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		})
		if _, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, bLong, opts); err == nil {
			t.Fatalf("16 KiB + 1 stop sequence should be rejected")
		}
	})
}
