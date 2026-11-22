package bridge_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/chat"
	"github.com/ihainan/llm-protocol-bridge/gemini"
	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/responses"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// Regression tests for all 9 P1 findings and key P2 findings from Codex xhigh review (codex-A2.txt).
func TestCodexA2_AllP1Fixes(t *testing.T) {
	// P1 #1: LiveFirst must NOT complete the first live tool call early if text arrives
	// before EvStop validation (e.g. when response truncates on max_tokens or a 2nd call fails).
	t.Run("P1_01_livefirst_does_not_complete_tool_before_validation", func(t *testing.T) {
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat,
			[]byte(`{"model":"m","input":"go","stream":true}`),
			bridge.Options{ToolRelease: ir.ToolReleaseLiveFirst},
		)
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		s := plan.NewStream()
		// First tool call starts live, then intervening text arrives, then a second call has INVALID JSON.
		chatSSE := "" +
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"A","arguments":"{\"x\":1}"}}]}}]}` + "\n\n" +
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"content":"mid text"}}]}` + "\n\n" +
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"c2","type":"function","function":{"name":"B","arguments":"not-json"}}]}}]} ` + "\n\n" +
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n" +
			"data: [DONE]\n\n"
		out := feedBridgeSSE(t, s, chatSSE)
		out, _ = s.End(out, ir.EndDone)
		if strings.Contains(string(out), "response.output_item.done") {
			t.Fatalf("LiveFirst completed tool item early before validation failed:\n%s", out)
		}
		if !strings.Contains(string(out), "response.failed") {
			t.Fatalf("expected response.failed on invalid 2nd call:\n%s", out)
		}
	})

	// P1 #2: Split tool names in Chat stream must be assembled before EvToolStart so Intercept matches.
	t.Run("P1_02_split_tool_name_assembled_for_intercept", func(t *testing.T) {
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat,
			[]byte(`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"search"}],"stream":true}`),
			bridge.Options{Intercept: []string{"web_search"}},
		)
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		s := plan.NewStream()
		// Name split across 3 chunks: ID-only, then "web_", then "search", then arguments.
		chatSSE := "" +
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"","arguments":""}}]}}]}` + "\n\n" +
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"web_"}}]}}]} ` + "\n\n" +
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"search"}}]}}]} ` + "\n\n" +
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":\"go\"}"}}]}}]}` + "\n\n" +
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n" +
			"data: [DONE]\n\n"
		out := feedBridgeSSE(t, s, chatSSE)
		out, _ = s.End(out, ir.EndDone)
		ic := s.Intercepted()
		if len(ic) != 1 || ic[0].Name != "web_search" {
			t.Fatalf("split tool name failed to match Intercept: intercepted=%+v out=%s", ic, out)
		}
		if strings.Contains(string(out), "web_") {
			t.Errorf("partial intercepted tool name leaked to client:\n%s", out)
		}
	})

	// P1 #3: Gemini request history without functionCall.id / functionResponse.id must assign
	// deterministic IDs and correlate results to unanswered calls by name.
	t.Run("P1_03_gemini_history_without_ids_correlated", func(t *testing.T) {
		gemReq := []byte(`{
			"contents": [
				{"role": "user", "parts": [{"text": "read two files"}]},
				{"role": "model", "parts": [
					{"functionCall": {"name": "Read", "args": {"path": "/a"}}},
					{"functionCall": {"name": "Read", "args": {"path": "/b"}}}
				]},
				{"role": "user", "parts": [
					{"functionResponse": {"name": "Read", "response": {"output": "body a"}}},
					{"functionResponse": {"name": "Read", "response": {"output": "body b"}}}
				]}
			]
		}`)
		plan, err := bridge.Convert(ir.FormatGemini, ir.FormatAnthropic, gemReq, bridge.Options{Model: "claude-4"})
		if err != nil {
			t.Fatalf("Convert: %v", err)
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
		tu0, _ := body.Messages[1].Content[0]["id"].(string)
		tu1, _ := body.Messages[1].Content[1]["id"].(string)
		tr0, _ := body.Messages[2].Content[0]["tool_use_id"].(string)
		tr1, _ := body.Messages[2].Content[1]["tool_use_id"].(string)
		if tu0 == "" || tu1 == "" || tu0 == tu1 {
			t.Fatalf("expected distinct non-empty deterministic IDs, got tu0=%q tu1=%q", tu0, tu1)
		}
		if tr0 != tu0 || tr1 != tu1 {
			t.Fatalf("tool_result IDs not correlated to tool_use IDs: tu=(%q,%q) tr=(%q,%q)", tu0, tu1, tr0, tr1)
		}
	})

	// P1 #4: Gemini ToolCall.Signature must survive round-trip through a Chat client via call ID suffix.
	t.Run("P1_04_gemini_tool_call_signature_roundtrips_through_chat_client", func(t *testing.T) {
		seal := ir.NewSealCodec([]byte("k"))
		plan1, err := bridge.Convert(ir.FormatChat, ir.FormatGemini,
			[]byte(`{"model":"gemini-2.5","messages":[{"role":"user","content":"go"}]}`),
			bridge.Options{ProviderID: "google", Seal: seal},
		)
		if err != nil {
			t.Fatalf("Convert 1: %v", err)
		}
		gemResp := []byte(`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"functionCall":{"id":"fc_1","name":"Do","args":{"x":1}},"thoughtSignature":"gem_sig_xyz"}]},"finishReason":"STOP"}]}`)
		chatResp, _, err := plan1.ConvertResponse(gemResp)
		if err != nil {
			t.Fatalf("ConvertResponse: %v", err)
		}
		var parsedChat struct {
			Choices []struct {
				Message struct {
					ToolCalls []struct {
						ID string `json:"id"`
					} `json:"tool_calls"`
				} `json:"message"`
			} `json:"choices"`
		}
		_ = json.Unmarshal(chatResp, &parsedChat)
		wireCallID := parsedChat.Choices[0].Message.ToolCalls[0].ID
		if !strings.Contains(wireCallID, "::") {
			t.Fatalf("expected sealed Gemini signature suffix on Chat tool_call.id, got %q", wireCallID)
		}

		// Next turn: Chat client echoes wireCallID in assistant tool_calls and tool message.
		nextChatReq, _ := json.Marshal(map[string]any{
			"model": "gemini-2.5",
			"messages": []any{
				map[string]any{"role": "user", "content": "go"},
				map[string]any{"role": "assistant", "tool_calls": []any{
					map[string]any{"id": wireCallID, "type": "function", "function": map[string]any{"name": "Do", "arguments": `{"x":1}`}},
				}},
				map[string]any{"role": "tool", "tool_call_id": wireCallID, "content": "ok"},
			},
		})
		plan2, err := bridge.Convert(ir.FormatChat, ir.FormatGemini, nextChatReq, bridge.Options{ProviderID: "google", Seal: seal})
		if err != nil {
			t.Fatalf("Convert 2: %v", err)
		}
		if !strings.Contains(string(plan2.Body), `"thoughtSignature":"gem_sig_xyz"`) {
			t.Fatalf("Gemini thoughtSignature not restored from Chat round-trip: %s", plan2.Body)
		}
	})

	// P1 #5: Same-format responses with ProviderID must seal provider provenance so a second
	// provider cannot receive provider A's signature or encrypted_content.
	t.Run("P1_05_same_format_provider_isolation_via_seal", func(t *testing.T) {
		seal := ir.NewSealCodec([]byte("k"))
		// Turn 1: Anthropic client -> Anthropic provider A
		pA, _ := bridge.Convert(ir.FormatAnthropic, ir.FormatAnthropic,
			[]byte(`{"model":"claude","max_tokens":64,"messages":[{"role":"user","content":"q"}]}`),
			bridge.Options{ProviderID: "prov-a", Seal: seal},
		)
		respA, _, err := pA.ConvertResponse([]byte(`{"id":"m1","type":"message","role":"assistant","model":"claude","content":[{"type":"thinking","thinking":"deep","signature":"sig_from_a"},{"type":"text","text":"ans"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":5}}`))
		if err != nil {
			t.Fatalf("ConvertResponse A: %v", err)
		}
		var outA struct {
			Content []map[string]any `json:"content"`
		}
		_ = json.Unmarshal(respA, &outA)
		sigA, _ := outA.Content[0]["signature"].(string)
		if !ir.IsSealed(sigA) {
			t.Fatalf("expected sealed signature when ProviderID is set, got %q", sigA)
		}

		// Turn 2: Same client history sent to provider B must drop provider A's thinking block!
		nextReq, _ := json.Marshal(map[string]any{
			"model":      "claude",
			"max_tokens": 64,
			"messages": []any{
				map[string]any{"role": "user", "content": "q"},
				map[string]any{"role": "assistant", "content": []any{
					map[string]any{"type": "thinking", "thinking": "deep", "signature": sigA},
					map[string]any{"type": "text", "text": "ans"},
				}},
				map[string]any{"role": "user", "content": "next"},
			},
		})
		pB, err := bridge.Convert(ir.FormatAnthropic, ir.FormatAnthropic, nextReq, bridge.Options{ProviderID: "prov-b", Seal: seal})
		if err != nil {
			t.Fatalf("Convert B: %v", err)
		}
		if strings.Contains(string(pB.Body), "sig_from_a") || strings.Contains(string(pB.Body), `"type":"thinking"`) {
			t.Fatalf("provider A signature leaked to provider B: %s", pB.Body)
		}
		// Sending back to provider A succeeds with raw sig_from_a restored!
		pA2, err := bridge.Convert(ir.FormatAnthropic, ir.FormatAnthropic, nextReq, bridge.Options{ProviderID: "prov-a", Seal: seal})
		if err != nil {
			t.Fatalf("Convert A2: %v", err)
		}
		if !strings.Contains(string(pA2.Body), `"signature":"sig_from_a"`) {
			t.Fatalf("provider A raw signature not restored: %s", pA2.Body)
		}
	})

	// P1 #6 & #7: ConvertResponse, StreamResponse, and OffLowestHidden/OffPatch all honor hidden reasoning.
	t.Run("P1_06_07_hidden_reasoning_in_nonstream_streamresponse_and_off_rules", func(t *testing.T) {
		// 1. Explicit ReasoningOutHidden on ConvertResponse and StreamResponse.
		plan, _ := bridge.Convert(ir.FormatChat, ir.FormatAnthropic,
			[]byte(`{"model":"m","messages":[{"role":"user","content":"q"}]}`),
			bridge.Options{Reasoning: ir.ReasoningOutHidden},
		)
		anthResp := []byte(`{"id":"m1","type":"message","role":"assistant","model":"m","content":[{"type":"thinking","thinking":"secret","signature":"s"},{"type":"text","text":"public"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":5}}`)
		out, _, err := plan.ConvertResponse(anthResp)
		if err != nil {
			t.Fatalf("ConvertResponse: %v", err)
		}
		if strings.Contains(string(out), "secret") || strings.Contains(string(out), "reasoning_content") {
			t.Errorf("ConvertResponse leaked hidden reasoning: %s", out)
		}
		streamOut, err := plan.StreamResponse(nil, anthResp)
		if err != nil {
			t.Fatalf("StreamResponse: %v", err)
		}
		if strings.Contains(string(streamOut), "secret") || strings.Contains(string(streamOut), "reasoning_content") {
			t.Errorf("StreamResponse leaked hidden reasoning: %s", streamOut)
		}

		// 2. OffLowestHidden automatically hides reasoning even without explicit Options.Reasoning.
		planOff, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat,
			[]byte(`{"model":"m","max_tokens":64,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"q"}],"stream":true}`),
			bridge.Options{Profile: &ir.ModelProfile{Off: ir.OffRule{Mode: ir.OffLowestHidden}}},
		)
		if err != nil {
			t.Fatalf("Convert OffLowestHidden: %v", err)
		}
		s := planOff.NewStream()
		sout := feedBridgeSSE(t, s,
			`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"hidden thought"}}]}`+"\n\n"+
				`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"content":"visible"},"finish_reason":"stop"}]}`+"\n\n"+
				"data: [DONE]\n\n",
		)
		sout, _ = s.End(sout, ir.EndDone)
		if strings.Contains(string(sout), "hidden thought") {
			t.Errorf("OffLowestHidden failed to hide reasoning in stream:\n%s", sout)
		}
	})

	// P1 #8: Responses previous_response_id converted to Chat/Anthropic/Gemini must fail with CodeUnsupported.
	t.Run("P1_08_previous_response_id_rejected_across_all_non_responses_targets", func(t *testing.T) {
		respReq := []byte(`{"model":"m","previous_response_id":"resp_prev","input":"continue"}`)
		for _, target := range []ir.Format{ir.FormatChat, ir.FormatAnthropic, ir.FormatGemini} {
			if _, err := bridge.Convert(ir.FormatResponses, target, respReq, bridge.Options{}); err == nil {
				t.Errorf("Responses->%s should reject previous_response_id", target)
			}
		}
	})

	// P1 #9: Namespaced ToolChoiceAllowed must NOT allow same-named tool from a different namespace.
	t.Run("P1_09_namespaced_allowed_tools_strict_match", func(t *testing.T) {
		plan, err := bridge.Build(&ir.Request{
			Model:    "gpt-4o",
			Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("hi")}}},
			Tools: []ir.Tool{
				{Namespace: "a", Name: "f", Kind: ir.ToolFunction},
				{Namespace: "b", Name: "f", Kind: ir.ToolFunction},
			},
			ToolChoice: ir.ToolChoice{
				Mode:    ir.ToolChoiceAllowed,
				Allowed: []ir.ToolRef{{Namespace: "a", Name: "f"}},
			},
		}, ir.FormatResponses, ir.FormatChat, bridge.Options{})
		if err != nil {
			t.Fatalf("Build: %v", err)
		}
		if strings.Contains(string(plan.Body), `"b__f"`) {
			t.Fatalf("namespaced allowed list leaked b__f: %s", plan.Body)
		}
		if !strings.Contains(string(plan.Body), `"a__f"`) {
			t.Fatalf("expected a__f in tools: %s", plan.Body)
		}
	})

	// Selected high-impact P2 fixes verified alongside P1s:
	t.Run("P2_selected_fixes_stop_emulate_single_char_jsonx_int64_and_empty_resp_validation", func(t *testing.T) {
		// P2 #26 & #30: Single-char stop sequence emits incrementally AND works with Responses upstream!
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatResponses,
			[]byte(`{"model":"gpt-5","stop":["."],"messages":[{"role":"user","content":"hi"}],"stream":true}`),
			bridge.Options{StopMode: ir.StopEmulate},
		)
		if err != nil {
			t.Fatalf("StopEmulate with Responses upstream should succeed, got: %v", err)
		}
		s := plan.NewStream()
		var out []byte
		r := sse.NewReader(bytes.NewReader([]byte(
			"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"r1\",\"model\":\"gpt-5\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
				"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":1,\"output_index\":0,\"delta\":\"hello\"}\n\n",
		)))
		for {
			ev, err := r.Next()
			if err != nil {
				break
			}
			out, _ = s.Feed(out, ev)
		}
		if !strings.Contains(string(out), `"hello"`) {
			t.Errorf("single-char StopEmulate buffered non-matching text instead of streaming live: %s", out)
		}

		// P2 #53: 64-bit integer precision in jsonx.Int (seed > 2^53).
		n, ok := jsonx.Int([]byte("9007199254740993"))
		if !ok || n != 9007199254740993 {
			t.Errorf("jsonx.Int lost 64-bit precision: got %d (ok=%v), want 9007199254740993", n, ok)
		}

		// P2 #50: Empty {} non-streaming response rejected as protocol error.
		if _, err := responses.ParseResponse([]byte(`{}`), responses.DecodeOptions{}); err == nil {
			t.Errorf("responses.ParseResponse should reject empty {}")
		}
		if _, err := chat.ParseResponse([]byte(`{}`), chat.DecodeOptions{}); err == nil {
			t.Errorf("chat.ParseResponse should reject empty {}")
		}
		if _, err := gemini.ParseResponse([]byte(`{}`), gemini.DecodeOptions{}); err == nil {
			t.Errorf("gemini.ParseResponse should reject empty {}")
		}
	})
}
