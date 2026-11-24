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

func chatUsageVia(t *testing.T, dia func(*bridge.Options), body string) ir.Usage {
	t.Helper()
	opts := bridge.Options{}
	dia(&opts)
	plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat,
		[]byte(`{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`), opts)
	if err != nil {
		t.Fatal(err)
	}
	_, resp, err := plan.ConvertResponse([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	return resp.Usage
}

// Moonshot-style usage: the standard detail says 0, the real count is at the top.
func TestChatCacheReadFallsBackWhenStandardIsZero(t *testing.T) {
	withFields := func(o *bridge.Options) { o.Chat.CacheReadFields = []string{"cached_tokens", "num_cached_tokens"} }
	u := chatUsageVia(t, withFields, `{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":1000,"completion_tokens":5,"cached_tokens":768,"prompt_tokens_details":{"cached_tokens":0}}}`)
	if u.CacheReadTokens != 768 || u.InputTokens != 232 {
		t.Fatalf("moonshot fallback: %+v", u)
	}
	// Standard positive wins; the alias is not added.
	u = chatUsageVia(t, withFields, `{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":1000,"completion_tokens":5,"cached_tokens":768,"num_cached_tokens":768,"prompt_tokens_details":{"cached_tokens":512}}}`)
	if u.CacheReadTokens != 512 || u.InputTokens != 488 {
		t.Fatalf("standard should win: %+v", u)
	}
	// Mistral's num_cached_tokens alone.
	u = chatUsageVia(t, withFields, `{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":1000,"completion_tokens":5,"num_cached_tokens":640}}`)
	if u.CacheReadTokens != 640 || u.InputTokens != 360 {
		t.Fatalf("mistral alias: %+v", u)
	}
	// No positive count anywhere: an explicit 0 stays a reported miss.
	u = chatUsageVia(t, withFields, `{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":1000,"completion_tokens":5,"cached_tokens":0,"prompt_tokens_details":{"cached_tokens":0}}}`)
	if u.CacheReadTokens != 0 || u.InputTokens != 1000 || !u.Has(ir.UCacheRead) {
		t.Fatalf("zero case: %+v", u)
	}
}

// Relays in front of Claude that use Anthropic's names in Chat usage.
func TestChatCacheAnthropicNamesFromRelay(t *testing.T) {
	withFields := func(o *bridge.Options) {
		o.Chat.CacheReadFields = []string{"cache_read_input_tokens"}
		o.Chat.CacheWriteFields = []string{"prompt_tokens_details.cache_write_tokens", "cache_creation_input_tokens"}
	}
	// Exclusive prompt (smaller than the cached counts): kept as uncached input.
	u := chatUsageVia(t, withFields, `{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":100,"completion_tokens":5,"cache_read_input_tokens":5000,"cache_creation_input_tokens":300}}`)
	if u.InputTokens != 100 || u.CacheReadTokens != 5000 || u.CacheWriteTokens != 300 {
		t.Fatalf("exclusive relay usage: %+v", u)
	}
	// Inclusive prompt: the cached counts are taken out of it.
	u = chatUsageVia(t, withFields, `{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":5400,"completion_tokens":5,"cache_read_input_tokens":5000,"cache_creation_input_tokens":300}}`)
	if u.InputTokens != 100 || u.CacheReadTokens != 5000 || u.CacheWriteTokens != 300 {
		t.Fatalf("inclusive relay usage: %+v", u)
	}
}
