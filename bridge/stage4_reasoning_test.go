package bridge_test

import (
	"bytes"
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

// Item 8: Reasoning effort / budget / off conversion & ModelProfile.Resolve across all 4 builders.
func TestStage4B_ReasoningConfigAndModelProfile(t *testing.T) {
	t.Run("chat_effort_to_anthropic_budget_and_max_tokens_and_sampling_constraints", func(t *testing.T) {
		chatReq := []byte(`{
			"model": "claude-sonnet-4",
			"messages": [{"role": "user", "content": "think"}],
			"reasoning_effort": "high",
			"temperature": 0.7,
			"top_p": 0.9
		}`)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic, chatReq, bridge.Options{
			Allow: ir.LossSet{ir.LossSamplingClamp: true},
		})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		var body struct {
			MaxTokens   int      `json:"max_tokens"`
			Temperature *float64 `json:"temperature"`
			TopP        *float64 `json:"top_p"`
			Thinking    *struct {
				Type         string `json:"type"`
				BudgetTokens int    `json:"budget_tokens"`
			} `json:"thinking"`
		}
		if err := json.Unmarshal(plan.Body, &body); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if body.Thinking == nil || body.Thinking.Type != "enabled" || body.Thinking.BudgetTokens < 1024 {
			t.Fatalf("expected thinking enabled with valid budget_tokens >= 1024, got %+v (body=%s)", body.Thinking, plan.Body)
		}
		if body.MaxTokens <= body.Thinking.BudgetTokens {
			t.Errorf("Anthropic requires max_tokens > budget_tokens: max_tokens=%d budget_tokens=%d", body.MaxTokens, body.Thinking.BudgetTokens)
		}
		if body.Temperature != nil || body.TopP != nil {
			t.Errorf("Anthropic thinking enabled must omit conflicting temperature/top_p, got temp=%v top_p=%v", body.Temperature, body.TopP)
		}
	})

	t.Run("anthropic_budget_to_chat_and_responses_effort", func(t *testing.T) {
		anthReq := []byte(`{
			"model": "o3",
			"max_tokens": 16384,
			"thinking": {"type": "enabled", "budget_tokens": 4096},
			"messages": [{"role": "user", "content": "solve"}]
		}`)
		pChat, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, anthReq, bridge.Options{})
		if err != nil {
			t.Fatalf("Anthropic->Chat: %v", err)
		}
		var chatOut struct {
			ReasoningEffort string `json:"reasoning_effort"`
		}
		_ = json.Unmarshal(pChat.Body, &chatOut)
		if chatOut.ReasoningEffort != "low" {
			t.Errorf("Anthropic budget 4096 -> Chat reasoning_effort: got %q, want low (body=%s)", chatOut.ReasoningEffort, pChat.Body)
		}

		pResp, err := bridge.Convert(ir.FormatAnthropic, ir.FormatResponses, anthReq, bridge.Options{})
		if err != nil {
			t.Fatalf("Anthropic->Responses: %v", err)
		}
		var respOut struct {
			Reasoning *struct {
				Effort string `json:"effort"`
			} `json:"reasoning"`
		}
		_ = json.Unmarshal(pResp.Body, &respOut)
		if respOut.Reasoning == nil || respOut.Reasoning.Effort != "low" {
			t.Errorf("Anthropic budget 4096 -> Responses reasoning.effort: got %+v, want low (body=%s)", respOut.Reasoning, pResp.Body)
		}
	})

	t.Run("gemini_explicit_zero_budget_turns_reasoning_off", func(t *testing.T) {
		chatReq := []byte(`{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none"}`)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatGemini, chatReq, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		if !strings.Contains(string(plan.Body), `"thinkingBudget":0`) {
			t.Fatalf("expected explicit thinkingBudget:0 in Gemini request, got: %s", plan.Body)
		}
		// Round-trip parse back into IR must mark ReasoningOff.
		req2, err := gemini.ParseRequest(plan.Body, gemini.ParseOptions{Model: "gemini-2.5-flash"})
		if err != nil {
			t.Fatalf("gemini.ParseRequest: %v", err)
		}
		if req2.Reasoning == nil || req2.Reasoning.Mode != ir.ReasoningOff {
			t.Errorf("expected ReasoningOff from thinkingBudget:0, got %+v", req2.Reasoning)
		}
	})

	t.Run("responses_parses_all_effort_levels_and_profile_off_rules", func(t *testing.T) {
		for _, eff := range []string{"none", "minimal", "low", "medium", "high", "xhigh", "max"} {
			req, err := responses.ParseRequest([]byte(`{"model":"m","input":"hi","reasoning":{"effort":"`+eff+`"}}`), responses.ParseOptions{})
			if err != nil {
				t.Fatalf("ParseRequest effort=%s: %v", eff, err)
			}
			if req.Reasoning == nil || string(req.Reasoning.Effort) != eff {
				t.Errorf("effort=%s: got %+v", eff, req.Reasoning)
			}
		}

		// OffUnsupported rejects reasoning_effort: none.
		_, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic,
			[]byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none"}`),
			bridge.Options{Profile: &ir.ModelProfile{Off: ir.OffRule{Mode: ir.OffUnsupported}}},
		)
		if err == nil {
			t.Errorf("expected error for OffUnsupported profile")
		}

		// Gemini StyleLevel emits thinkingLevel.
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatGemini,
			[]byte(`{"model":"gemini-3","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}`),
			bridge.Options{Profile: &ir.ModelProfile{Style: ir.StyleLevel, Efforts: []ir.Effort{ir.EffortLow, ir.EffortHigh}}},
		)
		if err != nil {
			t.Fatalf("Gemini StyleLevel: %v", err)
		}
		if !strings.Contains(string(plan.Body), `"thinkingLevel"`) {
			t.Errorf("expected thinkingLevel in Gemini body, got: %s", plan.Body)
		}
	})
}

// Item 9: Provenance (Origin.Matches), SealCodec, Anthropic redacted_thinking & auto thinking block, Gemini thoughtSignature.
func TestStage4B_SignaturesProvenanceAndSeal(t *testing.T) {
	t.Run("foreign_and_unsigned_reasoning_dropped_with_loss_not_sent_to_anthropic_or_responses", func(t *testing.T) {
		// Gemini thought signature in history -> Anthropic must drop unsigned/foreign thinking with LossReasoningSignature.
		gemReq := []byte(`{
			"contents": [
				{"role": "user", "parts": [{"text": "q"}]},
				{"role": "model", "parts": [{"text": "hmm", "thought": true, "thoughtSignature": "gem_sig_1"}, {"text": "ans"}]},
				{"role": "user", "parts": [{"text": "follow"}]}
			]
		}`)
		plan, err := bridge.Convert(ir.FormatGemini, ir.FormatAnthropic, gemReq, bridge.Options{})
		if err != nil {
			t.Fatalf("Gemini->Anthropic: %v", err)
		}
		if strings.Contains(string(plan.Body), "gem_sig_1") || strings.Contains(string(plan.Body), `"type":"thinking"`) {
			t.Errorf("foreign Gemini signature leaked to Anthropic: %s", plan.Body)
		}
		foundSigLoss := false
		for _, l := range plan.Losses {
			if l.Kind == ir.LossReasoningSignature {
				foundSigLoss = true
			}
		}
		if !foundSigLoss {
			t.Errorf("expected LossReasoningSignature in Plan.Losses, got %+v", plan.Losses)
		}

		// Anthropic redacted_thinking -> Responses must NOT send it as encrypted_content!
		anthReq := []byte(`{
			"model": "gpt-5",
			"max_tokens": 64,
			"messages": [
				{"role": "user", "content": "q"},
				{"role": "assistant", "content": [{"type": "redacted_thinking", "data": "anth_redacted_xyz"}, {"type": "text", "text": "ans"}]},
				{"role": "user", "content": "next"}
			]
		}`)
		pResp, err := bridge.Convert(ir.FormatAnthropic, ir.FormatResponses, anthReq, bridge.Options{})
		if err != nil {
			t.Fatalf("Anthropic->Responses: %v", err)
		}
		if strings.Contains(string(pResp.Body), "anth_redacted_xyz") || strings.Contains(string(pResp.Body), `"encrypted_content"`) {
			t.Errorf("Anthropic redacted_thinking leaked as Responses encrypted_content: %s", pResp.Body)
		}
		foundSealedLoss := false
		for _, l := range pResp.Losses {
			if l.Kind == ir.LossReasoningSealed {
				foundSealedLoss = true
			}
		}
		if !foundSealedLoss {
			t.Errorf("expected LossReasoningSealed in Plan.Losses, got %+v", pResp.Losses)
		}
	})

	t.Run("anthropic_stream_redacted_thinking_preserved_and_chat_reasoning_opens_thinking_block", func(t *testing.T) {
		// Anthropic StreamDecoder must decode redacted_thinking from content_block_start.
		dec := anthropic.NewStreamDecoder(anthropic.DecodeOptions{Env: &ir.Env{ProviderID: "prov-a"}})
		raw := "" +
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"model\":\"claude\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"redacted_thinking\",\"data\":\"opaque_data_123\"}}\n\n" +
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n" +
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
		col := ir.NewCollector()
		r := sse.NewReader(bytes.NewReader([]byte(raw)))
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
		if len(resp.Outputs[0].Parts) == 0 || resp.Outputs[0].Parts[0].Reasoning == nil || resp.Outputs[0].Parts[0].Reasoning.Sealed == nil {
			t.Fatalf("redacted_thinking lost in Anthropic StreamDecoder: %+v", resp.Outputs[0].Parts)
		}
		var dataStr string
		_ = json.Unmarshal(resp.Outputs[0].Parts[0].Reasoning.Sealed.Data, &dataStr)
		if dataStr != "opaque_data_123" || resp.Outputs[0].Parts[0].Reasoning.Sealed.Origin.Provider != "prov-a" {
			t.Errorf("unexpected sealed block: %+v (data=%q)", resp.Outputs[0].Parts[0].Reasoning.Sealed, dataStr)
		}

		// Chat reasoning_content stream -> Anthropic StreamEncoder must open a thinking block (never index: -1).
		enc := anthropic.NewStreamEncoder(anthropic.EncodeOptions{Env: &ir.Env{}, Model: "claude"})
		var dst []byte
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStart, ID: "1", Model: "claude"})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvReasoning, Index: 0, Text: "step 1"})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvText, Index: 0, Text: "answer"})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopEndTurn}})
		dst, _ = enc.Close(dst)
		if strings.Contains(string(dst), `"index":-1`) {
			t.Fatalf("Anthropic StreamEncoder emitted invalid index -1:\n%s", dst)
		}
		if !strings.Contains(string(dst), `"type":"thinking"`) || !strings.Contains(string(dst), `"thinking_delta"`) {
			t.Fatalf("expected thinking content_block_start + thinking_delta:\n%s", dst)
		}
	})

	t.Run("gemini_preserves_thought_signature_and_coalesces_stream_thoughts", func(t *testing.T) {
		// ParseRequest preserves thoughtSignature on thought:true part.
		req, err := gemini.ParseRequest([]byte(`{"contents":[{"role":"model","parts":[{"text":"hmm","thought":true,"thoughtSignature":"sig_g1"}]}]}`), gemini.ParseOptions{Env: &ir.Env{ProviderID: "google"}})
		if err != nil {
			t.Fatalf("gemini.ParseRequest: %v", err)
		}
		r := req.Messages[0].Parts[0].Reasoning
		if r == nil || r.Signature == nil || r.Signature.Value != "sig_g1" || r.Signature.Origin.Provider != "google" {
			t.Fatalf("thoughtSignature lost on thought:true part: %+v", req.Messages[0].Parts[0])
		}

		// StreamDecoder coalesces consecutive thought chunks (only 1 EvReasoningStart) and keeps functionCall thoughtSignature.
		dec := gemini.NewStreamDecoder(gemini.DecodeOptions{Env: &ir.Env{ProviderID: "google"}})
		var all []ir.Event
		for _, ch := range []string{
			`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"a","thought":true}]}}]}`,
			`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"b","thought":true,"thoughtSignature":"sig_t"}]}}]}`,
			`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"functionCall":{"name":"F","args":{}},"thoughtSignature":"sig_fc"}]},"finishReason":"STOP"}]}`,
		} {
			evs, _ := dec.Decode(nil, sse.Event{Data: []byte(ch)})
			all = append(all, evs...)
		}
		starts := 0
		var fcSig string
		for _, e := range all {
			if e.Type == ir.EvReasoningStart {
				starts++
			}
			if e.Type == ir.EvToolStart && e.Signature != nil {
				fcSig = e.Signature.Value
			}
		}
		if starts != 1 {
			t.Errorf("expected 1 coalesced EvReasoningStart for consecutive thought chunks, got %d", starts)
		}
		if fcSig != "sig_fc" {
			t.Errorf("functionCall thoughtSignature lost in Gemini StreamDecoder, got %q", fcSig)
		}
	})

	t.Run("seal_codec_roundtrip_across_anthropic_client_and_responses_upstream", func(t *testing.T) {
		seal := ir.NewSealCodec([]byte("secret-key"))
		// 1. Responses upstream returns encrypted_content -> encoded to Anthropic client with SealCodec.
		respBody := []byte(`{"id":"r1","object":"response","model":"o3","status":"completed","output":[{"type":"reasoning","id":"rs_1","encrypted_content":"enc_payload_abc","summary":[{"type":"summary_text","text":"s"}]},{"type":"message","id":"m1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"42"}]}]}`)
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatResponses,
			[]byte(`{"model":"o3","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`),
			bridge.Options{ProviderID: "openai", Seal: seal},
		)
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		anthResp, _, err := plan.ConvertResponse(respBody)
		if err != nil {
			t.Fatalf("ConvertResponse: %v", err)
		}
		var parsedAnth struct {
			Content []map[string]any `json:"content"`
		}
		_ = json.Unmarshal(anthResp, &parsedAnth)
		var sealedData string
		for _, b := range parsedAnth.Content {
			if b["type"] == "redacted_thinking" {
				sealedData, _ = b["data"].(string)
			}
		}
		if !ir.IsSealed(sealedData) {
			t.Fatalf("expected sealed envelope in Anthropic redacted_thinking, got: %s", anthResp)
		}

		// 2. Next turn: Anthropic client echoes redacted_thinking back -> Responses upstream recovers encrypted_content!
		nextReq, _ := json.Marshal(map[string]any{
			"model":      "o3",
			"max_tokens": 64,
			"messages": []any{
				map[string]any{"role": "user", "content": "hi"},
				map[string]any{"role": "assistant", "content": []any{
					map[string]any{"type": "redacted_thinking", "data": sealedData},
					map[string]any{"type": "text", "text": "42"},
				}},
				map[string]any{"role": "user", "content": "more"},
			},
		})
		plan2, err := bridge.Convert(ir.FormatAnthropic, ir.FormatResponses, nextReq, bridge.Options{ProviderID: "openai", Seal: seal})
		if err != nil {
			t.Fatalf("Convert next turn: %v", err)
		}
		if !strings.Contains(string(plan2.Body), `"encrypted_content":"enc_payload_abc"`) {
			t.Fatalf("expected sealed Responses encrypted_content to round-trip back, got: %s", plan2.Body)
		}

		// 3. Same envelope sent to a DIFFERENT provider must be dropped!
		planOther, err := bridge.Convert(ir.FormatAnthropic, ir.FormatResponses, nextReq, bridge.Options{ProviderID: "other-provider", Seal: seal})
		if err != nil {
			t.Fatalf("Convert other provider: %v", err)
		}
		if strings.Contains(string(planOther.Body), "enc_payload_abc") {
			t.Fatalf("sealed encrypted_content leaked to different provider: %s", planOther.Body)
		}
	})
}

var _ = chat.BuildRequest
