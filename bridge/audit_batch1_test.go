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

const cfgUpdateBody = `{"model":"m","reasoning":{"effort":"low"},"input":[
 {"role":"user","content":"draft a plan"},
 {"type":"message","role":"assistant","content":[{"type":"output_text","text":"plan"}]},
 {"type":"configuration_update","reasoning":{"effort":"medium"}},
 {"role":"user","content":"more"},
 {"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]},
 {"type":"configuration_update","reasoning":{"effort":"high"}},
 {"role":"user","content":"analyze failure modes"}]}`

// configuration_update changes reasoning mid-conversation (OpenAI Reasoning
// guide). Other upstreams get the last update as the request-level effort.
func TestConfigurationUpdateAppliedForChatUpstream(t *testing.T) {
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, []byte(cfgUpdateBody), bridge.Options{})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	body := string(plan.Body)
	if !strings.Contains(body, `"reasoning_effort":"high"`) || strings.Contains(body, "configuration_update") {
		t.Fatalf("effort not applied: %s", body)
	}
	found := false
	for _, l := range plan.Losses {
		if l.Kind == ir.LossConfigUpdate {
			found = true
		}
	}
	if !found {
		t.Fatalf("loss not recorded: %+v", plan.Losses)
	}
}

// A Responses upstream receives the items verbatim, in place, with the
// request-level effort unchanged (which keeps the cached prompt prefix).
func TestConfigurationUpdateKeptForResponsesUpstream(t *testing.T) {
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatResponses, []byte(cfgUpdateBody), bridge.Options{})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	body := string(plan.Body)
	if !strings.Contains(body, `"effort":"low"`) {
		t.Fatalf("request-level effort changed: %s", body)
	}
	i1 := strings.Index(body, `"effort":"medium"`)
	i2 := strings.Index(body, `"text":"more"`)
	i3 := strings.Index(body, `"effort":"high"`)
	i4 := strings.Index(body, `"text":"analyze failure modes"`)
	if i1 < 0 || i2 < 0 || i3 < 0 || i4 < 0 || !(i1 < i2 && i2 < i3 && i3 < i4) {
		t.Fatalf("updates not kept in place: %s", body)
	}
}

func TestConfigurationUpdateInvalidEffortRejected(t *testing.T) {
	_, err := bridge.Convert(ir.FormatResponses, ir.FormatChat,
		[]byte(`{"model":"m","input":[{"type":"configuration_update","reasoning":{"effort":"warp"}},{"role":"user","content":"x"}]}`), bridge.Options{})
	if err == nil || !strings.Contains(err.Error(), "effort") {
		t.Fatalf("want effort error, got %v", err)
	}
}

// Codex remote compaction items: kept for Responses upstreams, a clear
// unsupported error (naming the item) for others instead of a parse failure.
func TestCompactionItems(t *testing.T) {
	trigger := `{"model":"m","input":[{"role":"user","content":"hi"},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"yo"}]},{"type":"compaction_trigger"}]}`
	sealed := `{"model":"m","input":[{"type":"compaction","id":"cmp_1","encrypted_content":"gAAAA"},{"role":"user","content":"next"}]}`
	for _, tc := range []struct{ body, want string }{{trigger, "compaction_trigger"}, {sealed, "compaction"}} {
		_, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, []byte(tc.body), bridge.Options{})
		if err == nil || !strings.Contains(err.Error(), tc.want) || strings.Contains(err.Error(), "unknown input item") {
			t.Fatalf("%s: want clear unsupported error, got %v", tc.want, err)
		}
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatResponses, []byte(tc.body), bridge.Options{})
		if err != nil {
			t.Fatalf("%s to Responses: %v", tc.want, err)
		}
		if !strings.Contains(string(plan.Body), `"type":"`+tc.want+`"`) {
			t.Fatalf("%s not forwarded: %s", tc.want, plan.Body)
		}
	}
}
