package bridge_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/anthropic"
	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

func feedBridgeSSE(t *testing.T, s *bridge.Stream, raw string) []byte {
	t.Helper()
	var out []byte
	r := sse.NewReader(bytes.NewReader([]byte(raw)))
	for {
		ev, err := r.Next()
		if err != nil {
			break
		}
		if len(ev.Data) == 0 && ev.Name == "" {
			continue
		}
		var err2 error
		out, err2 = s.Feed(out, ev)
		if err2 != nil {
			t.Fatalf("Feed: %v", err2)
		}
	}
	return out
}

// Item 11: bridge Options (Gemini Model/Stream/AltSSE), merged Parse+Build Losses,
// Stream filter pipeline (ToolRelease, Limits, StopEmulate, ReasoningOut, Intercept+NextRound), and Repair.
func TestStage5A_BridgeOptionsAndPipeline(t *testing.T) {
	t.Run("gemini_options_model_stream_altsse_and_merged_losses", func(t *testing.T) {
		gemReq := []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"unknownTopLevel":123,"generationConfig":{"topK":10}}`)
		plan, err := bridge.Convert(ir.FormatGemini, ir.FormatChat, gemReq, bridge.Options{
			Model:  "gemini-2.5-pro",
			Stream: true,
			AltSSE: true,
			Allow:  ir.LossSet{ir.LossSamplingTopK: true},
		})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		if plan.Model != "gemini-2.5-pro" || plan.ClientModel != "gemini-2.5-pro" {
			t.Errorf("Gemini Model lost: Model=%q ClientModel=%q", plan.Model, plan.ClientModel)
		}
		if !plan.ClientStream || !plan.UpstreamStream {
			t.Errorf("Gemini Stream lost: ClientStream=%v UpstreamStream=%v", plan.ClientStream, plan.UpstreamStream)
		}
		// Plan.Losses must include BOTH parse-phase unknown_field AND build-phase sampling.top_k!
		var sawUnknown, sawTopK bool
		for _, l := range plan.Losses {
			if l.Kind == ir.LossUnknownField {
				sawUnknown = true
			}
			if l.Kind == ir.LossSamplingTopK {
				sawTopK = true
			}
		}
		if !sawUnknown || !sawTopK {
			t.Errorf("expected both parse+build losses in Plan.Losses, got %+v", plan.Losses)
		}

		// AltSSE=true must emit SSE data: lines for Gemini client stream.
		s := plan.NewStream()
		out := feedBridgeSSE(t, s, `data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`+"\n\ndata: [DONE]\n\n")
		out, _ = s.End(out, ir.EndDone)
		if !strings.HasPrefix(string(out), "data: ") {
			t.Errorf("AltSSE=true must emit SSE data: frames, got: %s", out)
		}
	})

	t.Run("tool_release_buffered_drops_on_max_tokens_and_rejects_invalid_json", func(t *testing.T) {
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat,
			[]byte(`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"go"}],"stream":true}`),
			bridge.Options{ToolRelease: ir.ToolReleaseBuffered},
		)
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}

		// 1. max_tokens drops buffered tool calls without error.
		s1 := plan.NewStream()
		out1 := feedBridgeSSE(t, s1,
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"Read","arguments":"{\"path\":\"/x"}}]}}]} `+"\n\n"+
				`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`+"\n\n"+
				"data: [DONE]\n\n",
		)
		out1, _ = s1.End(out1, ir.EndDone)
		if strings.Contains(string(out1), `"tool_use"`) {
			t.Errorf("buffered tool call must be dropped on max_tokens:\n%s", out1)
		}
		if !strings.Contains(string(out1), `"stop_reason":"max_tokens"`) {
			t.Errorf("expected max_tokens stop_reason:\n%s", out1)
		}

		// 2. Invalid JSON args on normal stop emits error event and never delivers tool_use.
		s2 := plan.NewStream()
		out2 := feedBridgeSSE(t, s2,
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"Read","arguments":"not-json"}}]}}]} `+"\n\n"+
				`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`+"\n\n"+
				"data: [DONE]\n\n",
		)
		out2, _ = s2.End(out2, ir.EndDone)
		if strings.Contains(string(out2), `"type":"tool_use"`) {
			t.Errorf("invalid tool call must not be delivered:\n%s", out2)
		}
		if !strings.Contains(string(out2), "event: error") {
			t.Errorf("expected error event on invalid tool args:\n%s", out2)
		}
	})

	t.Run("limits_max_tool_calls_and_buffered_bytes", func(t *testing.T) {
		plan, _ := bridge.Convert(ir.FormatAnthropic, ir.FormatChat,
			[]byte(`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"go"}],"stream":true}`),
			bridge.Options{
				ToolRelease: ir.ToolReleaseBuffered,
				Limits:      ir.Limits{MaxToolCalls: 1},
			},
		)
		s := plan.NewStream()
		out := feedBridgeSSE(t, s,
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"A","arguments":"{}"}},{"index":1,"id":"c2","type":"function","function":{"name":"B","arguments":"{}"}}]}}]}`+"\n\n"+
				`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`+"\n\n"+
				"data: [DONE]\n\n",
		)
		out, _ = s.End(out, ir.EndDone)
		if !strings.Contains(string(out), "event: error") {
			t.Errorf("expected error event when exceeding MaxToolCalls=1:\n%s", out)
		}
	})

	t.Run("stop_emulate_utf8_across_chunks_and_ignores_reasoning", func(t *testing.T) {
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat,
			[]byte(`{"model":"m","max_tokens":64,"stop_sequences":["停止!"],"messages":[{"role":"user","content":"go"}],"stream":true}`),
			bridge.Options{StopMode: ir.StopEmulate},
		)
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		s := plan.NewStream()
		// Reasoning contains "停止!" — must NOT trigger stop!
		// Output text splits "停止!" across two chunks — MUST trigger stop and trim suffix!
		out := feedBridgeSSE(t, s,
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"思考停止!继续"}}]}`+"\n\n"+
				`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"content":"你好停"}}]}`+"\n\n"+
				`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"content":"止!多余文字"}}]}`+"\n\n"+
				`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`+"\n\n"+
				"data: [DONE]\n\n",
		)
		out, _ = s.End(out, ir.EndDone)
		if !s.Done() {
			t.Errorf("expected stream Done() after stop sequence match")
		}
		dec := anthropic.NewStreamDecoder(anthropic.DecodeOptions{})
		col := ir.NewCollector()
		r := sse.NewReader(bytes.NewReader(out))
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
		text := ir.TextOf(resp.Outputs[0].Parts, "")
		if text != "你好" {
			t.Errorf("StopEmulate output text: got %q, want %q", text, "你好")
		}
		if resp.Outputs[0].Stop.Reason != ir.StopSequence || resp.Outputs[0].Stop.Sequence != "停止!" {
			t.Errorf("StopEmulate stop reason: got %+v", resp.Outputs[0].Stop)
		}
	})

	t.Run("reasoning_out_hidden_drops_reasoning", func(t *testing.T) {
		plan, _ := bridge.Convert(ir.FormatAnthropic, ir.FormatChat,
			[]byte(`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"go"}],"stream":true}`),
			bridge.Options{Reasoning: ir.ReasoningOutHidden},
		)
		s := plan.NewStream()
		out := feedBridgeSSE(t, s,
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"secret thoughts"}}]}`+"\n\n"+
				`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"content":"public answer"},"finish_reason":"stop"}]}`+"\n\n"+
				"data: [DONE]\n\n",
		)
		out, _ = s.End(out, ir.EndDone)
		if strings.Contains(string(out), "secret thoughts") || strings.Contains(string(out), `"thinking"`) {
			t.Errorf("ReasoningOutHidden leaked reasoning:\n%s", out)
		}
		if !strings.Contains(string(out), "public answer") {
			t.Errorf("public answer missing:\n%s", out)
		}
	})

	t.Run("intercept_and_next_round_accumulates_usage_and_keeps_single_client_stream", func(t *testing.T) {
		plan1, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat,
			[]byte(`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"search"}],"stream":true}`),
			bridge.Options{Intercept: []string{"web_search"}},
		)
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		s := plan1.NewStream()
		// Round 1: upstream calls intercepted web_search + reports usage (100 in, 10 out).
		out := feedBridgeSSE(t, s,
			`data: {"id":"r1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"ws_1","type":"function","function":{"name":"web_search","arguments":"{\"q\":\"go\"}"}}]}}]}`+"\n\n"+
				`data: {"id":"r1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}}`+"\n\n"+
				"data: [DONE]\n\n",
		)
		out, _ = s.End(out, ir.EndDone)
		if s.Done() {
			t.Fatalf("stream must NOT be Done() while intercepted calls are pending")
		}
		ic := s.Intercepted()
		if len(ic) != 1 || ic[0].Name != "web_search" || string(ic[0].Arguments) != `{"q":"go"}` {
			t.Fatalf("expected 1 intercepted web_search call, got %+v", ic)
		}
		if strings.Contains(string(out), "web_search") || strings.Contains(string(out), "message_stop") {
			t.Fatalf("intercepted call or message_stop leaked to client on round 1:\n%s", out)
		}

		// Round 2: caller runs tool, builds plan2, calls NextRound(plan2).
		plan2, _ := bridge.Convert(ir.FormatAnthropic, ir.FormatChat,
			[]byte(`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"search"}],"stream":true}`),
			bridge.Options{Intercept: []string{"web_search"}},
		)
		s.NextRound(plan2)
		out = append(out, feedBridgeSSE(t, s,
			`data: {"id":"r2","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"found it"},"finish_reason":"stop"}],"usage":{"prompt_tokens":150,"completion_tokens":20,"total_tokens":170}}`+"\n\n"+
				"data: [DONE]\n\n",
		)...)
		out, _ = s.End(out, ir.EndDone)
		if !s.Done() {
			t.Errorf("expected stream Done() after round 2")
		}
		if strings.Count(string(out), "event: message_start") != 1 {
			t.Errorf("client must see exactly 1 message_start across rounds:\n%s", out)
		}
		if strings.Count(string(out), "event: message_stop") != 1 {
			t.Errorf("client must see exactly 1 message_stop across rounds:\n%s", out)
		}
		if !strings.Contains(string(out), "found it") {
			t.Errorf("round 2 text missing:\n%s", out)
		}
		// Usage must be SUMMED across rounds (100+150=250 in, 10+20=30 out)!
		u := s.Usage()
		if u.InputTokens != 250 || u.OutputTokens != 30 {
			t.Errorf("usage must be summed across rounds: got %+v, want input=250 output=30", u)
		}
	})

	t.Run("history_repairs", func(t *testing.T) {
		// Assistant calls c1 & c2, user only provides result for c1 twice + orphan result c99 + empty assistant turn.
		raw := []byte(`{
			"model": "claude-4",
			"messages": [
				{"role": "user", "content": "hi"},
				{"role": "assistant", "content": ""},
				{"role": "assistant", "tool_calls": [
					{"id": "c1", "type": "function", "function": {"name": "A", "arguments": "{}"}},
					{"id": "c2", "type": "function", "function": {"name": "B", "arguments": "{}"}}
				]},
				{"role": "tool", "tool_call_id": "c1", "content": "r1"},
				{"role": "tool", "tool_call_id": "c1", "content": "r1-dup"},
				{"role": "tool", "tool_call_id": "c99", "content": "orphan"}
			]
		}`)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic, raw, bridge.Options{
			Repair: ir.RepairSet{
				ir.RepairMissingResults:   true,
				ir.RepairDuplicateResults: true,
				ir.RepairOrphanResults:    true,
				ir.RepairEmptyAssistant:   true,
			},
		})
		if err != nil {
			t.Fatalf("Convert with Repair: %v", err)
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
		// Empty assistant dropped -> 3 messages: user, assistant(c1,c2), user(c1, c2-synthesized, orphan-as-text).
		if len(body.Messages) != 3 {
			t.Fatalf("expected 3 repaired messages, got %d: %s", len(body.Messages), plan.Body)
		}
		c1Count, c2Count, orphanText := 0, 0, false
		for _, blk := range body.Messages[2].Content {
			if blk["type"] == "tool_result" {
				if blk["tool_use_id"] == "c1" {
					c1Count++
				}
				if blk["tool_use_id"] == "c2" {
					c2Count++
				}
				if blk["tool_use_id"] == "c99" {
					t.Errorf("orphan c99 should have been converted to text, not tool_result")
				}
			}
			if blk["type"] == "text" && strings.Contains(blk["text"].(string), "orphan") {
				orphanText = true
			}
		}
		if c1Count != 1 || c2Count != 1 || !orphanText {
			t.Errorf("repairs failed: c1Count=%d c2Count=%d orphanText=%v body=%s", c1Count, c2Count, orphanText, plan.Body)
		}
	})
}
