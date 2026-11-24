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

// Responses input_file.file_data is a data URI; the prefix must not reach
// Anthropic's base64 field (Anthropic rejects it).
func TestResponsesInputFileDataURIToAnthropic(t *testing.T) {
	body := `{"model":"m","input":[{"role":"user","content":[{"type":"input_text","text":"summarize"},{"type":"input_file","filename":"a.pdf","file_data":"data:application/pdf;base64,JVBERi0="}]}]}`
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, []byte(body), bridge.Options{})
	if err != nil {
		t.Fatal(err)
	}
	got := string(plan.Body)
	if strings.Contains(got, "data:application/pdf") || !strings.Contains(got, `"source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}`) {
		t.Fatalf("data URI not split: %s", got)
	}
	// Gemini gets the media type and the bare payload too.
	plan, err = bridge.Convert(ir.FormatResponses, ir.FormatGemini, []byte(body), bridge.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if g := string(plan.Body); strings.Contains(g, "data:application") || !strings.Contains(g, `"mimeType":"application/pdf"`) {
		t.Fatalf("gemini inline data wrong: %s", g)
	}
}

// Non-PDF documents fail in the gateway with the accepted types named,
// instead of an upstream 400; text files become Anthropic text documents.
func TestAnthropicDocumentTypes(t *testing.T) {
	docx := `{"model":"m","input":[{"role":"user","content":[{"type":"input_file","filename":"r.docx","file_data":"data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,UEsDBA=="}]}]}`
	_, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, []byte(docx), bridge.Options{})
	if err == nil || !strings.Contains(err.Error(), "PDF") || !strings.Contains(err.Error(), "r.docx") {
		t.Fatalf("want clear unsupported error, got %v", err)
	}
	// Bare base64 with only a file name to go by.
	bare := `{"model":"m","input":[{"role":"user","content":[{"type":"input_file","filename":"sheet.xlsx","file_data":"UEsDBA=="}]}]}`
	if _, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, []byte(bare), bridge.Options{}); err == nil {
		t.Fatal("xlsx by file name should be rejected")
	}
	txt := `{"model":"m","input":[{"role":"user","content":[{"type":"input_file","filename":"n.txt","file_data":"data:text/plain;base64,aGVsbG8gd29ybGQ="}]}]}`
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, []byte(txt), bridge.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(plan.Body), `"source":{"type":"text","media_type":"text/plain","data":"hello world"}`) {
		t.Fatalf("text file not sent as text document: %s", plan.Body)
	}
	// Unknown type without a name keeps the previous PDF default.
	unknown := `{"model":"m","input":[{"role":"user","content":[{"type":"input_file","file_data":"JVBERi0="}]}]}`
	plan, err = bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, []byte(unknown), bridge.Options{})
	if err != nil || !strings.Contains(string(plan.Body), `"media_type":"application/pdf"`) {
		t.Fatalf("pdf default changed: %v %s", err, plan.Body)
	}
}

// Anthropic -> Responses writes file_data as a data URI.
func TestAnthropicDocumentToResponsesFileData(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}]}]}`
	plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatResponses, []byte(body), bridge.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(plan.Body), `"file_data":"data:application/pdf;base64,JVBERi0="`) {
		t.Fatalf("file_data not a data URI: %s", plan.Body)
	}
}

const forcedToolsBody = `{"model":"m","messages":[{"role":"user","content":"plan"}],
 "tools":[{"type":"function","function":{"name":"todo","parameters":{"type":"object"}}},{"type":"function","function":{"name":"ls","parameters":{"type":"object"}}}],
 "tool_choice":%s}`

// Models that reject forced tool_choice get it relaxed to auto.
func TestNoForcedToolChoiceRelaxes(t *testing.T) {
	strict := &ir.ModelProfile{NoForcedToolChoice: true}
	cases := []struct {
		choice, wantTC string
		wantTools      []string
		dropTools      []string
	}{
		{`{"type":"function","function":{"name":"todo"}}`, `"tool_choice":{"type":"auto"}`, []string{`"name":"todo"`}, []string{`"name":"ls"`}},
		{`"required"`, `"tool_choice":{"type":"auto"}`, []string{`"name":"todo"`, `"name":"ls"`}, nil},
	}
	for _, c := range cases {
		body := strings.Replace(forcedToolsBody, "%s", c.choice, 1)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic, []byte(body), bridge.Options{Profile: strict})
		if err != nil {
			t.Fatalf("%s: %v", c.choice, err)
		}
		got := string(plan.Body)
		if !strings.Contains(got, c.wantTC) {
			t.Fatalf("%s: tool_choice not relaxed: %s", c.choice, got)
		}
		for _, w := range c.wantTools {
			if !strings.Contains(got, w) {
				t.Fatalf("%s: missing tool %s: %s", c.choice, w, got)
			}
		}
		for _, d := range c.dropTools {
			if strings.Contains(got, d) {
				t.Fatalf("%s: tool %s should be dropped: %s", c.choice, d, got)
			}
		}
		lost := false
		for _, l := range plan.Losses {
			lost = lost || l.Kind == ir.LossForcedToolChoice
		}
		if !lost {
			t.Fatalf("%s: loss not recorded", c.choice)
		}
	}
	// Responses allowed_tools in required mode: the list stays, mode becomes auto.
	rbody := `{"model":"m","input":"plan","tools":[{"type":"function","name":"todo","parameters":{"type":"object"}},{"type":"function","name":"ls","parameters":{"type":"object"}}],
	 "tool_choice":{"type":"allowed_tools","mode":"required","tools":[{"type":"function","name":"ls"}]}}`
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, []byte(rbody), bridge.Options{Profile: strict})
	if err != nil {
		t.Fatal(err)
	}
	if got := string(plan.Body); !strings.Contains(got, `"tool_choice":{"type":"auto"}`) || strings.Contains(got, `"name":"todo"`) || !strings.Contains(got, `"name":"ls"`) {
		t.Fatalf("allowed_tools not relaxed: %s", got)
	}
	// Without the flag nothing changes.
	body := strings.Replace(forcedToolsBody, "%s", `"required"`, 1)
	plan, err = bridge.Convert(ir.FormatChat, ir.FormatAnthropic, []byte(body), bridge.Options{})
	if err != nil || !strings.Contains(string(plan.Body), `"tool_choice":{"type":"any"}`) {
		t.Fatalf("unflagged model changed: %v %s", err, plan.Body)
	}
	// auto and none are left alone even with the flag.
	body = strings.Replace(forcedToolsBody, "%s", `"none"`, 1)
	plan, err = bridge.Convert(ir.FormatChat, ir.FormatAnthropic, []byte(body), bridge.Options{Profile: strict})
	if err != nil || !strings.Contains(string(plan.Body), `"tool_choice":{"type":"none"}`) || len(plan.Losses) != 0 {
		t.Fatalf("none changed: %v %s %+v", err, plan.Body, plan.Losses)
	}
}

// Review round 1 regressions.

// The format builders themselves apply or reject extras, so a caller that
// parses and builds without bridge.Convert does not silently drop them.
func TestDirectBuildersHandleExtras(t *testing.T) {
	sealed := `{"model":"m","input":[{"type":"compaction","encrypted_content":"gAAAA"},{"role":"user","content":"next"}]}`
	req, err := responses.ParseRequest([]byte(sealed), responses.ParseOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := anthropic.BuildRequest(req, anthropic.BuildOptions{}); err == nil {
		t.Fatal("anthropic builder dropped a compaction item")
	}
	if _, err := chat.BuildRequest(req, chat.BuildOptions{}); err == nil {
		t.Fatal("chat builder dropped a compaction item")
	}
	if _, err := gemini.BuildRequest(req, gemini.BuildOptions{}); err == nil {
		t.Fatal("gemini builder dropped a compaction item")
	}
	upd := `{"model":"m","input":[{"role":"user","content":"x"},{"type":"configuration_update","reasoning":{"effort":"high"}},{"role":"user","content":"y"}]}`
	req, err = responses.ParseRequest([]byte(upd), responses.ParseOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := chat.BuildRequest(req, chat.BuildOptions{Env: &ir.Env{Deny: ir.LossSet{ir.LossConfigUpdate: true}}}); err == nil {
		t.Fatal("denied configuration_update loss not reported")
	}
}

// Responses -> Responses keeps tool-output file data as a data URI.
func TestToolOutputFileDataURIRoundTrip(t *testing.T) {
	body := `{"model":"m","input":[{"type":"function_call","call_id":"c1","name":"f","arguments":"{}"},{"type":"function_call_output","call_id":"c1","output":[{"type":"input_file","filename":"a.pdf","file_data":"data:application/pdf;base64,JVBERi0="},{"type":"input_file","file_url":"https://example.com/b.pdf"}]}],"tools":[{"type":"function","name":"f","parameters":{"type":"object"}}]}`
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatResponses, []byte(body), bridge.Options{})
	if err != nil {
		t.Fatal(err)
	}
	got := string(plan.Body)
	if !strings.Contains(got, `"file_data":"data:application/pdf;base64,JVBERi0="`) || !strings.Contains(got, `"file_url":"https://example.com/b.pdf"`) {
		t.Fatalf("tool output file changed: %s", got)
	}
}

// A bare tool_choice name selects a namespaced tool; relaxing keeps it.
func TestRelaxForcedToolChoiceNamespacedBareName(t *testing.T) {
	body := `{"model":"m","input":"x","tools":[{"type":"namespace","name":"ns","tools":[{"type":"function","name":"f","parameters":{"type":"object"}}]},{"type":"function","name":"g","parameters":{"type":"object"}}],"tool_choice":{"type":"function","name":"f"}}`
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, []byte(body), bridge.Options{Profile: &ir.ModelProfile{NoForcedToolChoice: true}})
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Tools      []json.RawMessage     `json:"tools"`
		ToolChoice struct{ Type string } `json:"tool_choice"`
	}
	if err := json.Unmarshal(plan.Body, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Tools) != 1 || got.ToolChoice.Type != "auto" {
		t.Fatalf("namespaced tool lost: %s", plan.Body)
	}
}

// Partial updates accumulate in input order.
func TestConfigurationUpdatesAccumulate(t *testing.T) {
	body := `{"model":"m","reasoning":{"effort":"low"},"input":[{"role":"user","content":"x"},{"type":"configuration_update","reasoning":{"effort":"high"}},{"type":"configuration_update","reasoning":{"summary":"detailed"}},{"role":"user","content":"y"}]}`
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, []byte(body), bridge.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(plan.Body), `"reasoning_effort":"high"`) {
		t.Fatalf("earlier effort update lost: %s", plan.Body)
	}
}

// A developer message after an extra stays in place for a Responses
// upstream, and still becomes the system prompt for other upstreams.
func TestDeveloperAfterExtraKeepsOrder(t *testing.T) {
	body := `{"model":"m","input":[{"type":"configuration_update","reasoning":{"effort":"high"}},{"role":"developer","content":"do X"},{"role":"user","content":"next"}]}`
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatResponses, []byte(body), bridge.Options{})
	if err != nil {
		t.Fatal(err)
	}
	var rb struct {
		Instructions *json.RawMessage              `json:"instructions"`
		Input        []struct{ Type, Role string } `json:"input"`
	}
	if err := json.Unmarshal(plan.Body, &rb); err != nil {
		t.Fatal(err)
	}
	if rb.Instructions != nil || len(rb.Input) != 3 || rb.Input[0].Type != "configuration_update" || rb.Input[1].Role != "developer" {
		t.Fatalf("order changed: %s", plan.Body)
	}
	plan, err = bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, []byte(body), bridge.Options{})
	if err != nil {
		t.Fatal(err)
	}
	var ab struct {
		System   json.RawMessage `json:"system"`
		Messages json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(plan.Body, &ab); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(ab.System), "do X") || strings.Contains(string(ab.Messages), "do X") {
		t.Fatalf("developer message not lifted for Anthropic: %s", plan.Body)
	}
}

// An input holding only a compaction item is a valid continuation.
func TestCompactionOnlyInput(t *testing.T) {
	body := `{"model":"m","input":[{"type":"compaction","id":"cmp_1","encrypted_content":"gAAAA"}]}`
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatResponses, []byte(body), bridge.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(plan.Body), `"type":"compaction"`) {
		t.Fatalf("compaction dropped: %s", plan.Body)
	}
}

// Review round 2 regressions.

// The Plan carries the applied update, so the response side hides reasoning
// the update turned off.
func TestConfigurationUpdateReachesResponseSide(t *testing.T) {
	body := `{"model":"m","reasoning":{"effort":"high"},"input":[{"role":"user","content":"x"},{"type":"configuration_update","reasoning":{"effort":"none"}}]}`
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, []byte(body), bridge.Options{Profile: &ir.ModelProfile{
		Style: ir.StyleEffort, Efforts: []ir.Effort{ir.EffortLow, ir.EffortHigh}, Off: ir.OffRule{Mode: ir.OffLowestHidden},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(plan.Body), `"reasoning_effort":"low"`) {
		t.Fatalf("upstream effort: %s", plan.Body)
	}
	out, _, err := plan.ConvertResponse([]byte(`{"id":"c","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":"secret thought","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(out), "secret thought") {
		t.Fatalf("hidden reasoning leaked: %s", out)
	}
}

// A summary-only first update turns reasoning on, as the same object does at
// the top level.
func TestSummaryOnlyUpdateTurnsReasoningOn(t *testing.T) {
	upd := `{"model":"m","input":[{"role":"user","content":"x"},{"type":"configuration_update","reasoning":{"summary":"auto"}},{"role":"user","content":"y"}]}`
	top := `{"model":"m","reasoning":{"summary":"auto"},"input":[{"role":"user","content":"x"},{"role":"user","content":"y"}]}`
	for _, f := range []ir.Format{ir.FormatAnthropic, ir.FormatGemini} {
		a, err := bridge.Convert(ir.FormatResponses, f, []byte(upd), bridge.Options{})
		if err != nil {
			t.Fatal(err)
		}
		b, err := bridge.Convert(ir.FormatResponses, f, []byte(top), bridge.Options{})
		if err != nil {
			t.Fatal(err)
		}
		if string(a.Body) != string(b.Body) {
			t.Fatalf("%s: update %s, top level %s", f, a.Body, b.Body)
		}
	}
}

// A bare name with a top-level tool of that name selects only that tool.
func TestRelaxForcedToolChoicePrefersTopLevelTool(t *testing.T) {
	body := `{"model":"m","input":"x","tools":[{"type":"function","name":"f","parameters":{"type":"object"}},{"type":"namespace","name":"ns","tools":[{"type":"function","name":"f","parameters":{"type":"object"}}]}],"tool_choice":{"type":"function","name":"f"}}`
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, []byte(body), bridge.Options{Profile: &ir.ModelProfile{NoForcedToolChoice: true}})
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Tools []struct{ Name string } `json:"tools"`
	}
	if err := json.Unmarshal(plan.Body, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Tools) != 1 || got.Tools[0].Name != "f" {
		t.Fatalf("tools: %s", plan.Body)
	}
}

// A zero aggregate next to a positive cache_creation split counts the split.
func TestAnthropicStreamZeroAggregatePositiveSplit(t *testing.T) {
	plan, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic,
		[]byte(`{"model":"claude-4","messages":[{"role":"user","content":"hi"}],"stream":true}`), bridge.Options{})
	if err != nil {
		t.Fatal(err)
	}
	s := plan.NewStream()
	sse := "" +
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"m\",\"usage\":{\"input_tokens\":10,\"output_tokens\":1,\"cache_creation_input_tokens\":300,\"cache_creation\":{\"ephemeral_5m_input_tokens\":300,\"ephemeral_1h_input_tokens\":0}}}}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":20,\"cache_creation_input_tokens\":0,\"cache_creation\":{\"ephemeral_5m_input_tokens\":500,\"ephemeral_1h_input_tokens\":0}}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	out := feedBridgeSSE(t, s, sse)
	_, _ = s.End(out, ir.EndDone)
	u := s.Usage()
	if u.CacheWriteTokens != 500 || u.CacheWrite5mTokens != 500 || u.PromptTokens() != 510 {
		t.Fatalf("usage: %+v prompt %d", u, u.PromptTokens())
	}
}

// Review round 3 regressions.

// Responses -> Responses: the items go upstream unchanged, and the response
// side applies the settings they leave in effect.
func TestConfigurationUpdateResponsesUpstreamResponseSide(t *testing.T) {
	body := `{"model":"m","reasoning":{"effort":"none","summary":"auto"},"input":[{"role":"user","content":"x"},{"type":"configuration_update","reasoning":{"effort":"high"}},{"role":"user","content":"y"}]}`
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatResponses, []byte(body), bridge.Options{Profile: &ir.ModelProfile{Off: ir.OffRule{Mode: ir.OffLowestHidden}}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(plan.Body), `"type":"configuration_update"`) {
		t.Fatalf("update not forwarded: %s", plan.Body)
	}
	out, _, err := plan.ConvertResponse([]byte(`{"id":"resp_1","object":"response","status":"completed","model":"m","output":[{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"visible thought"}]},{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hi","annotations":[]}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), "visible thought") {
		t.Fatalf("reasoning the update turned on was hidden: %s", out)
	}
}

// Image data URIs with parameters keep their bytes on Responses -> Responses.
func TestImageDataURIParamsUnchanged(t *testing.T) {
	uri := "data:image/png;foo=bar;base64,aGVsbG8="
	body := `{"model":"m","input":[{"role":"user","content":[{"type":"input_image","image_url":"` + uri + `"}]}]}`
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatResponses, []byte(body), bridge.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(plan.Body), uri) {
		t.Fatalf("image URI changed: %s", plan.Body)
	}
}

// Review round 4 regressions.

// A cache marker on a tool the relaxing drops still disables automatic cache
// placement.
func TestRelaxForcedToolChoiceKeepsClientMarkedState(t *testing.T) {
	body := `{"model":"m","messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}},{"type":"function","function":{"name":"g","parameters":{"type":"object"}},"cache_control":{"type":"ephemeral"}}],"tool_choice":{"type":"function","function":{"name":"f"}}}`
	plan, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic, []byte(body), bridge.Options{
		Profile:   &ir.ModelProfile{NoForcedToolChoice: true},
		Anthropic: anthropic.Dialect{AutoCache: anthropic.AutoCachePrefix},
	})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(plan.Body), "cache_control") {
		t.Fatalf("automatic cache marker added: %s", plan.Body)
	}
}

// An input of only configuration_update items is still an empty input.
func TestConfigurationUpdateOnlyInputRejected(t *testing.T) {
	body := `{"model":"m","input":[{"type":"configuration_update","reasoning":{"effort":"high"}}]}`
	for _, f := range []ir.Format{ir.FormatChat, ir.FormatAnthropic, ir.FormatGemini, ir.FormatResponses} {
		if _, err := bridge.Convert(ir.FormatResponses, f, []byte(body), bridge.Options{}); err == nil {
			t.Fatalf("%s: accepted an input with no messages", f)
		}
	}
}
