package bridge_test

import (
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// Anthropic upstreams (Bedrock, several relays) repeat input and cache counters
// as 0 placeholders in message_delta. They must not erase message_start's counts.
func TestAnthropicStreamZeroPlaceholdersKeepStartUsage(t *testing.T) {
	plan, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic,
		[]byte(`{"model":"claude-4","messages":[{"role":"user","content":"hi"}],"stream":true}`), bridge.Options{})
	if err != nil {
		t.Fatal(err)
	}
	s := plan.NewStream()
	sse := "" +
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"model\":\"claude-4\",\"usage\":{\"input_tokens\":100,\"cache_read_input_tokens\":5000,\"cache_creation_input_tokens\":300,\"cache_creation\":{\"ephemeral_5m_input_tokens\":300,\"ephemeral_1h_input_tokens\":0},\"output_tokens\":1}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":0,\"cache_read_input_tokens\":0,\"cache_creation_input_tokens\":0,\"output_tokens\":20}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	out := feedBridgeSSE(t, s, sse)
	out, _ = s.End(out, ir.EndDone)
	u := s.Usage()
	if u.InputTokens != 100 || u.CacheReadTokens != 5000 || u.CacheWriteTokens != 300 || u.CacheWrite5mTokens != 300 || u.OutputTokens != 20 {
		t.Fatalf("zero placeholders erased usage: %+v", u)
	}
	if !u.Has(ir.UInput | ir.UCacheRead | ir.UCacheWrite | ir.UOutput) {
		t.Fatalf("reported bits lost: %+v", u)
	}
	// The client sees the same counts (prompt 5400 = 100 + 5000 + 300).
	if !strings.Contains(string(out), `"prompt_tokens":5400`) || !strings.Contains(string(out), `"cached_tokens":5000`) {
		t.Fatalf("client usage wrong: %s", out)
	}
}

// A positive message_delta value still replaces message_start's (cumulative snapshots).
func TestAnthropicStreamPositiveDeltaStillOverrides(t *testing.T) {
	plan, _ := bridge.Convert(ir.FormatChat, ir.FormatAnthropic,
		[]byte(`{"model":"claude-4","messages":[{"role":"user","content":"hi"}],"stream":true}`), bridge.Options{})
	s := plan.NewStream()
	sse := "" +
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"model\":\"claude-4\",\"usage\":{\"input_tokens\":0,\"output_tokens\":1}}}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":42,\"cache_read_input_tokens\":7,\"output_tokens\":3}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	out := feedBridgeSSE(t, s, sse)
	_, _ = s.End(out, ir.EndDone)
	u := s.Usage()
	if u.InputTokens != 42 || u.CacheReadTokens != 7 || u.OutputTokens != 3 {
		t.Fatalf("usage = %+v", u)
	}
	// An explicit 0 with nothing earlier still counts as reported.
	if !u.Has(ir.UInput) {
		t.Fatalf("input not reported: %+v", u)
	}
}
