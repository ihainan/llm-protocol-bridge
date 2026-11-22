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

// Review2 #31: Buffered mode must preserve relative order of Call A -> Text T -> Call B,
// and emulated stop after a pending call must preserve legitimate text prefix ("abc" before "!").
func TestReview2_P2_31_BufferedOrderAndStopPrefixPreserved(t *testing.T) {
	t.Run("buffered_call_text_call_preserves_order", func(t *testing.T) {
		reqBody := []byte(`{"model":"claude-sonnet-4","stream":true,"max_tokens":1024,"messages":[{"role":"user","content":"go"}],"tools":[{"name":"A","input_schema":{"type":"object"}},{"name":"B","input_schema":{"type":"object"}}]}`)
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, reqBody, bridge.Options{
			ToolRelease: ir.ToolReleaseBuffered,
		})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		s := plan.NewStream()
		raw := "data: " + `{"id":"1","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"cA","type":"function","function":{"name":"A","arguments":"{}"}}]},"finish_reason":null}]}` + "\n\n" +
			"data: " + `{"id":"1","choices":[{"index":0,"delta":{"content":"MID_TEXT"},"finish_reason":null}]}` + "\n\n" +
			"data: " + `{"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"cB","type":"function","function":{"name":"B","arguments":"{}"}}]},"finish_reason":null}]}` + "\n\n" +
			"data: " + `{"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n" +
			"data: [DONE]\n\n"
		out := string(feedSSE(t, s, raw))
		posA := strings.Index(out, `"name":"A"`)
		posT := strings.Index(out, `MID_TEXT`)
		posB := strings.Index(out, `"name":"B"`)
		if posA < 0 || posT < 0 || posB < 0 || !(posA < posT && posT < posB) {
			t.Errorf("expected order A < MID_TEXT < B, got posA=%d posT=%d posB=%d:\n%s", posA, posT, posB, out)
		}
	})

	t.Run("stop_emulate_preserves_prefix_before_stop_after_call", func(t *testing.T) {
		reqBody := []byte(`{"model":"claude-sonnet-4","stream":true,"max_tokens":1024,"stop_sequences":["!"],"messages":[{"role":"user","content":"go"}],"tools":[{"name":"A","input_schema":{"type":"object"}}]}`)
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, reqBody, bridge.Options{
			ToolRelease: ir.ToolReleaseBuffered,
			StopMode:    ir.StopEmulate,
		})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		s := plan.NewStream()
		raw := "data: " + `{"id":"1","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"cA","type":"function","function":{"name":"A","arguments":"{}"}}]},"finish_reason":null}]}` + "\n\n" +
			"data: " + `{"id":"1","choices":[{"index":0,"delta":{"content":"abc"},"finish_reason":null}]}` + "\n\n" +
			"data: " + `{"id":"1","choices":[{"index":0,"delta":{"content":"!rest"},"finish_reason":"stop"}]}` + "\n\n" +
			"data: [DONE]\n\n"
		out := string(feedSSE(t, s, raw))
		if !strings.Contains(out, `"abc"`) {
			t.Errorf("legitimate prefix 'abc' before stop '!' was dropped:\n%s", out)
		}
		if strings.Contains(out, "rest") {
			t.Errorf("text after stop '!' leaked:\n%s", out)
		}
	})
}

// Review2 #32: MaxBufferedBytes must count queued post-call text/reasoning events.
func TestReview2_P2_32_MaxBufferedBytesCoversQueuedEvents(t *testing.T) {
	reqBody := []byte(`{"model":"claude-sonnet-4","stream":true,"max_tokens":1024,"messages":[{"role":"user","content":"go"}],"tools":[{"name":"A","input_schema":{"type":"object"}}]}`)
	plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, reqBody, bridge.Options{
		ToolRelease: ir.ToolReleaseBuffered,
		Limits:      ir.Limits{MaxBufferedBytes: 1024},
	})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	s := plan.NewStream()
	bigText := strings.Repeat("x", 2048)
	var out []byte
	out, err = s.Feed(out, sse.Event{Data: []byte(`{"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"cA","type":"function","function":{"name":"A","arguments":"{}"}}]},"finish_reason":null}]}`)})
	if err != nil {
		t.Fatalf("Feed 1: %v", err)
	}
	chunk2, _ := json.Marshal(map[string]any{
		"id": "1",
		"choices": []any{map[string]any{
			"index": 0,
			"delta": map[string]any{"content": bigText},
		}},
	})
	out, err = s.Feed(out, sse.Event{Data: chunk2})
	if err != nil {
		t.Fatalf("Feed 2: %v", err)
	}
	if !strings.Contains(string(out), "error") {
		t.Errorf("expected buffered byte limit error for 2048B queued text with MaxBufferedBytes=1024, got:\n%s", string(out))
	}
}

// Review2 #39: Responses reasoning summary "S" must not be duplicated to "SS" by StreamResponse.
func TestReview2_P2_39_ResponsesReasoningSummaryNoDuplicate(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","input":"think"}`)
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatResponses, reqBody, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	upstreamResp := []byte(`{
		"id":"resp_1","object":"response","model":"gpt-5","status":"completed",
		"output":[
			{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"S"}]},
			{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok"}]}
		]
	}`)
	sseBytes, err := plan.StreamResponse(nil, upstreamResp)
	if err != nil {
		t.Fatalf("StreamResponse: %v", err)
	}
	if strings.Contains(string(sseBytes), `"text":"SS"`) || strings.Count(string(sseBytes), `"delta":"S"`) > 1 {
		t.Errorf("Responses reasoning summary duplicated in StreamResponse:\n%s", string(sseBytes))
	}
}

// Review2 #40: OffPatch must validate JSON and whitelist across all formats (returning CodeConfig on error)
// and apply valid patches across Chat, Responses, Anthropic, and Gemini.
func TestReview2_P2_40_OffPatchValidationAndCrossFormat(t *testing.T) {
	req := &ir.Request{
		Model:     "m",
		Messages:  []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("hi")}}},
		Reasoning: &ir.ReasoningConfig{Mode: ir.ReasoningOff},
	}

	t.Run("invalid_json_returns_config_error", func(t *testing.T) {
		env := &ir.Env{Profile: &ir.ModelProfile{
			Off: ir.OffRule{Mode: ir.OffPatch, Patch: json.RawMessage(`{bad json`)},
		}}
		for _, fn := range []struct {
			name string
			run  func() error
		}{
			{"chat", func() error { _, err := chat.BuildRequest(req, chat.BuildOptions{Env: env}); return err }},
			{"responses", func() error { _, err := responses.BuildRequest(req, responses.BuildOptions{Env: env}); return err }},
			{"anthropic", func() error { _, err := anthropic.BuildRequest(req, anthropic.BuildOptions{Env: env}); return err }},
			{"gemini", func() error { _, err := gemini.BuildRequest(req, gemini.BuildOptions{Env: env}); return err }},
		} {
			err := fn.run()
			if err == nil {
				t.Errorf("%s accepted invalid OffPatch JSON", fn.name)
			} else if ie, ok := err.(*ir.Error); !ok || ie.Code != ir.CodeConfig {
				t.Errorf("%s expected CodeConfig for invalid OffPatch JSON, got %v", fn.name, err)
			}
		}
	})

	t.Run("non_whitelisted_key_returns_config_error", func(t *testing.T) {
		env := &ir.Env{Profile: &ir.ModelProfile{
			Off: ir.OffRule{Mode: ir.OffPatch, Patch: json.RawMessage(`{"model":"hijacked"}`)},
		}}
		if _, err := chat.BuildRequest(req, chat.BuildOptions{Env: env}); err == nil {
			t.Errorf("chat accepted non-whitelisted OffPatch key")
		}
	})

	t.Run("valid_patch_applied_in_responses_and_anthropic", func(t *testing.T) {
		env := &ir.Env{Profile: &ir.ModelProfile{
			Off: ir.OffRule{Mode: ir.OffPatch, Patch: json.RawMessage(`{"enable_thinking":false}`)},
		}}
		bResp, err := responses.BuildRequest(req, responses.BuildOptions{Env: env})
		if err != nil {
			t.Fatalf("responses BuildRequest: %v", err)
		}
		if !strings.Contains(string(bResp.Body), `"enable_thinking":false`) {
			t.Errorf("responses ignored OffPatch: %s", string(bResp.Body))
		}
		bAnth, err := anthropic.BuildRequest(req, anthropic.BuildOptions{Env: env})
		if err != nil {
			t.Fatalf("anthropic BuildRequest: %v", err)
		}
		if !strings.Contains(string(bAnth.Body), `"enable_thinking":false`) {
			t.Errorf("anthropic ignored OffPatch: %s", string(bAnth.Body))
		}
	})
}

// Review2 #41: When both Effort and Budget are set, Effort takes precedence over Budget.
func TestReview2_P2_41_EffortTakesPrecedenceOverBudget(t *testing.T) {
	b1024 := 1024
	req := &ir.Request{
		Model:     "claude-sonnet-4",
		Messages:  []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("hi")}}},
		Reasoning: &ir.ReasoningConfig{Mode: ir.ReasoningOn, Effort: ir.EffortHigh, Budget: &b1024},
	}
	built, err := anthropic.BuildRequest(req, anthropic.BuildOptions{})
	if err != nil {
		t.Fatalf("BuildRequest: %v", err)
	}
	if strings.Contains(string(built.Body), `"budget_tokens":1024`) {
		t.Errorf("Budget=1024 overrode Effort=high in Anthropic BuildRequest: %s", string(built.Body))
	}
	if !strings.Contains(string(built.Body), `"budget_tokens":24000`) {
		t.Errorf("expected EffortHigh budget 24000, got: %s", string(built.Body))
	}
}

// Review2 #42: Gemini thinkingBudget: -1 (dynamic) must NOT map to minimal effort.
func TestReview2_P2_42_GeminiDynamicBudgetMinusOne(t *testing.T) {
	body := []byte(`{
		"contents":[{"role":"user","parts":[{"text":"hi"}]}],
		"generationConfig":{"thinkingConfig":{"thinkingBudget":-1}}
	}`)
	planChat, err := bridge.Convert(ir.FormatGemini, ir.FormatChat, body, bridge.Options{Model: "gpt-5"})
	if err != nil {
		t.Fatalf("Convert to Chat: %v", err)
	}
	if strings.Contains(string(planChat.Body), `"minimal"`) {
		t.Errorf("Gemini dynamic thinkingBudget:-1 must not downgrade to minimal in Chat: %s", string(planChat.Body))
	}
	planGem, err := bridge.Convert(ir.FormatGemini, ir.FormatGemini, body, bridge.Options{Model: "gemini-2.5-pro"})
	if err != nil {
		t.Fatalf("Convert to Gemini: %v", err)
	}
	if !strings.Contains(string(planGem.Body), `"thinkingBudget":-1`) {
		t.Errorf("Gemini identity conversion lost thinkingBudget:-1: %s", string(planGem.Body))
	}
}

// Review2 #43: Explicit max_tokens too small for Anthropic minimum thinking budget must NOT
// be silently raised above the user's cap.
func TestReview2_P2_43_AnthropicExplicitMaxTokensNotRaised(t *testing.T) {
	body := []byte(`{"model":"claude-sonnet-4","max_tokens":64,"reasoning_effort":"low","messages":[{"role":"user","content":"hi"}]}`)
	plan, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic, body, bridge.Options{})
	if err == nil {
		t.Fatalf("expected error when explicit max_tokens=64 cannot fit Anthropic min budget 1024, got body: %s", string(plan.Body))
	}
}

// Review2 #44: Anthropic manual extended thinking is mutually exclusive with forced tool_choice (any/tool).
func TestReview2_P2_44_AnthropicThinkingForcedToolChoiceConflict(t *testing.T) {
	body := []byte(`{
		"model":"claude-sonnet-4",
		"reasoning_effort":"high",
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],
		"tool_choice":"required"
	}`)
	plan, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic, body, bridge.Options{})
	if err == nil {
		t.Fatalf("expected error for Anthropic manual thinking enabled + forced tool_choice required, got body: %s", string(plan.Body))
	}
}
