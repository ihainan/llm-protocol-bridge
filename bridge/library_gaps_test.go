package bridge_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/anthropic"
	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/chat"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/responses"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

func TestOmitUserIDResponsesAndAnthropic(t *testing.T) {
	anthReq := []byte(`{
		"model":"m1",
		"max_tokens":128,
		"metadata":{"user_id":"secret-user-123"},
		"messages":[{"role":"user","content":"hi"}]
	}`)

	pKeep, err := bridge.Convert(ir.FormatAnthropic, ir.FormatResponses, anthReq, bridge.Options{OmitUserID: false})
	if err != nil {
		t.Fatalf("Convert keep: %v", err)
	}
	if !strings.Contains(string(pKeep.Body), `"secret-user-123"`) {
		t.Fatalf("expected user preserved when OmitUserID=false: %s", pKeep.Body)
	}

	pOmit, err := bridge.Convert(ir.FormatAnthropic, ir.FormatResponses, anthReq, bridge.Options{OmitUserID: true})
	if err != nil {
		t.Fatalf("Convert omit responses: %v", err)
	}
	if strings.Contains(string(pOmit.Body), "secret-user-123") || strings.Contains(string(pOmit.Body), `"user":`) {
		t.Fatalf("expected user omitted in Responses body when OmitUserID=true: %s", pOmit.Body)
	}

	pAnthOmit, err := bridge.Convert(ir.FormatAnthropic, ir.FormatAnthropic, anthReq, bridge.Options{OmitUserID: true})
	if err != nil {
		t.Fatalf("Convert omit anthropic: %v", err)
	}
	if strings.Contains(string(pAnthOmit.Body), "secret-user-123") {
		t.Fatalf("expected metadata.user_id omitted in Anthropic body when OmitUserID=true: %s", pAnthOmit.Body)
	}
}

type customUnsignedCodec struct {
	prefix string
}

func (c customUnsignedCodec) Seal(r ir.SealRecord) string { return ir.DefaultSeal.Seal(r) }
func (c customUnsignedCodec) Unseal(s string) (ir.SealRecord, bool) {
	return ir.DefaultSeal.Unseal(s)
}
func (c customUnsignedCodec) SealUnsigned(client ir.Format, origin ir.Origin, text string) string {
	return c.prefix + string(client) + ":" + text
}
func (c customUnsignedCodec) VerifyUnsigned(client ir.Format, sealed, text string) (ir.Origin, string, bool) {
	wantPrefix := c.prefix + string(client) + ":"
	if !strings.HasPrefix(sealed, wantPrefix) {
		return ir.Origin{}, "", false
	}
	payload := strings.TrimPrefix(sealed, wantPrefix)
	if client == ir.FormatAnthropic && payload != text {
		return ir.Origin{}, "", false
	}
	return ir.Origin{Format: ir.FormatChat}, payload, true
}

func TestSealUnsignedReasoningRoundTripAnthropicAndResponses(t *testing.T) {
	chatResp := []byte(`{
		"id":"chatcmpl-1",
		"object":"chat.completion",
		"created":1,
		"model":"m1",
		"choices":[{"index":0,"message":{"role":"assistant","reasoning_content":"step one","content":"done"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}
	}`)

	anthReq := []byte(`{"model":"m1","max_tokens":128,"messages":[{"role":"user","content":"hi"}]}`)

	// Default SealUnsignedReasoning=false leaves Anthropic signature empty.
	pOff, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, anthReq, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert off: %v", err)
	}
	outOff, _, err := pOff.ConvertResponse(chatResp)
	if err != nil {
		t.Fatalf("ConvertResponse off: %v", err)
	}
	var anthOff struct {
		Content []struct {
			Type      string `json:"type"`
			Thinking  string `json:"thinking"`
			Signature string `json:"signature"`
		} `json:"content"`
	}
	if err := json.Unmarshal(outOff, &anthOff); err != nil {
		t.Fatalf("unmarshal off: %v", err)
	}
	if len(anthOff.Content) < 1 || anthOff.Content[0].Type != "thinking" || anthOff.Content[0].Signature != "" {
		t.Fatalf("expected empty signature when SealUnsignedReasoning=false, got: %s", outOff)
	}

	// SealUnsignedReasoning=true seals non-stream and stream Anthropic thinking blocks.
	pOn, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, anthReq, bridge.Options{
		SealUnsignedReasoning: true,
	})
	if err != nil {
		t.Fatalf("Convert on: %v", err)
	}
	outOn, _, err := pOn.ConvertResponse(chatResp)
	if err != nil {
		t.Fatalf("ConvertResponse on: %v", err)
	}
	var anthOn struct {
		Content []struct {
			Type      string `json:"type"`
			Thinking  string `json:"thinking"`
			Signature string `json:"signature"`
		} `json:"content"`
	}
	if err := json.Unmarshal(outOn, &anthOn); err != nil {
		t.Fatalf("unmarshal on: %v", err)
	}
	sig := anthOn.Content[0].Signature
	if !ir.IsSealed(sig) {
		t.Fatalf("expected lpb1 sealed signature, got %q in %s", sig, outOn)
	}

	// Stream Anthropic sealing emits signature_delta before content_block_stop.
	st := pOn.NewStream()
	var streamOut []byte
	chunks := []string{
		`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m1","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"step one"}}]}`,
		`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m1","choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}]}`,
	}
	for _, ch := range chunks {
		streamOut, err = st.Feed(streamOut, sse.Event{Data: []byte(ch)})
		if err != nil {
			t.Fatalf("Feed: %v", err)
		}
	}
	streamOut, err = st.End(streamOut, ir.EndDone)
	if err != nil {
		t.Fatalf("End: %v", err)
	}
	if !strings.Contains(string(streamOut), `"signature_delta"`) || !strings.Contains(string(streamOut), `"lpb1.`) {
		t.Fatalf("expected signature_delta with sealed lpb1 value in Anthropic stream:\n%s", streamOut)
	}

	// Round-trip back to Chat with RequireVerified=true:
	// valid signature replays; tampered thinking or foreign signature does NOT replay.
	turn2 := []byte(`{
		"model":"m1",
		"max_tokens":128,
		"messages":[
			{"role":"user","content":"q1"},
			{"role":"assistant","content":[
				{"type":"thinking","thinking":"step one","signature":"` + sig + `"},
				{"type":"thinking","thinking":"tampered","signature":"` + sig + `"},
				{"type":"thinking","thinking":"foreign","signature":"raw_anthropic_sig"},
				{"type":"text","text":"done"}
			]},
			{"role":"user","content":"q2"}
		]
	}`)
	pTurn2, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, turn2, bridge.Options{
		Chat: chat.Dialect{
			ReasoningReplay: chat.ReasoningReplay{
				Field:           "reasoning_content",
				RequireVerified: true,
			},
		},
	})
	if err != nil {
		t.Fatalf("Convert turn2: %v", err)
	}
	if !strings.Contains(string(pTurn2.Body), `"reasoning_content":"step one"`) {
		t.Fatalf("expected verified thinking replayed, got: %s", pTurn2.Body)
	}
	if strings.Contains(string(pTurn2.Body), "tampered") || strings.Contains(string(pTurn2.Body), "foreign") {
		t.Fatalf("expected tampered/foreign thinking NOT replayed, got: %s", pTurn2.Body)
	}

	// Responses non-stream + stream sealing only when include has reasoning.encrypted_content.
	respReqNoInc := []byte(`{"model":"m1","input":"hi"}`)
	pRespNoInc, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, respReqNoInc, bridge.Options{
		SealUnsignedReasoning: true,
	})
	if err != nil {
		t.Fatalf("Convert resp no inc: %v", err)
	}
	outNoInc, _, err := pRespNoInc.ConvertResponse(chatResp)
	if err != nil {
		t.Fatalf("ConvertResponse resp no inc: %v", err)
	}
	if strings.Contains(string(outNoInc), "encrypted_content") {
		t.Fatalf("expected no encrypted_content without include, got: %s", outNoInc)
	}

	respReqInc := []byte(`{"model":"m1","input":"hi","include":["reasoning.encrypted_content"]}`)
	pRespInc, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, respReqInc, bridge.Options{
		SealUnsignedReasoning: true,
	})
	if err != nil {
		t.Fatalf("Convert resp inc: %v", err)
	}
	outInc, _, err := pRespInc.ConvertResponse(chatResp)
	if err != nil {
		t.Fatalf("ConvertResponse resp inc: %v", err)
	}
	var respObj struct {
		Output []struct {
			Type             string `json:"type"`
			EncryptedContent string `json:"encrypted_content"`
		} `json:"output"`
	}
	if err := json.Unmarshal(outInc, &respObj); err != nil {
		t.Fatalf("unmarshal resp inc: %v", err)
	}
	enc := ""
	for _, it := range respObj.Output {
		if it.Type == "reasoning" {
			enc = it.EncryptedContent
		}
	}
	if !ir.IsSealed(enc) {
		t.Fatalf("expected sealed encrypted_content in Responses output, got: %s", outInc)
	}

	// Round-trip Responses encrypted_content back to Chat with RequireVerified=true.
	respTurn2 := []byte(`{
		"model":"m1",
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"q1"}]},
			{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"` + enc + `"},
			{"type":"reasoning","id":"rs_2","summary":[{"type":"summary_text","text":"unverified raw"}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"q2"}]}
		]
	}`)
	pRespTurn2, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, respTurn2, bridge.Options{
		Chat: chat.Dialect{
			ReasoningReplay: chat.ReasoningReplay{
				Field:           "reasoning_content",
				RequireVerified: true,
			},
		},
	})
	if err != nil {
		t.Fatalf("Convert resp turn2: %v", err)
	}
	if !strings.Contains(string(pRespTurn2.Body), `"reasoning_content":"step one"`) {
		t.Fatalf("expected verified Responses reasoning replayed, got: %s", pRespTurn2.Body)
	}
	if strings.Contains(string(pRespTurn2.Body), "unverified raw") {
		t.Fatalf("expected unverified Responses reasoning NOT replayed, got: %s", pRespTurn2.Body)
	}

	// Custom UnsignedTextSealer codec round-trip on Anthropic & Responses.
	codec := customUnsignedCodec{prefix: "custom."}
	pCustom, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, anthReq, bridge.Options{
		Seal:                  codec,
		SealUnsignedReasoning: true,
	})
	if err != nil {
		t.Fatalf("Convert custom: %v", err)
	}
	outCustom, _, err := pCustom.ConvertResponse(chatResp)
	if err != nil {
		t.Fatalf("ConvertResponse custom: %v", err)
	}
	if !strings.Contains(string(outCustom), `"signature":"custom.anthropic:step one"`) {
		t.Fatalf("expected custom unsigned signature, got: %s", outCustom)
	}
}

func TestChatReasoningReplayBudgetsAndKeepTurns(t *testing.T) {
	// 3 assistant turns with reasoning: MaxItems=2 keeps only the newest window;
	// MaxChars=10 keeps only items fitting in 10 bytes.
	req := []byte(`{
		"model":"m1",
		"max_tokens":128,
		"messages":[
			{"role":"user","content":"u1"},
			{"role":"assistant","content":[{"type":"thinking","thinking":"aaaa"},{"type":"text","text":"a1"}]},
			{"role":"user","content":"u2"},
			{"role":"assistant","content":[{"type":"thinking","thinking":"bbbb"},{"type":"text","text":"a2"}]},
			{"role":"user","content":"u3"},
			{"role":"assistant","content":[{"type":"thinking","thinking":"cccc"},{"type":"text","text":"a3"}]},
			{"role":"user","content":"u4"}
		]
	}`)

	pItems, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, req, bridge.Options{
		Chat: chat.Dialect{
			ReasoningReplay: chat.ReasoningReplay{
				Field:    "reasoning",
				MaxItems: 2,
			},
		},
	})
	if err != nil {
		t.Fatalf("Convert MaxItems: %v", err)
	}
	// MaxItems=2 keeps the newest two blocks ("bbbb" and "cccc") and drops "aaaa".
	if strings.Contains(string(pItems.Body), "aaaa") {
		t.Fatalf("expected MaxItems=2 to drop oldest item aaaa: %s", pItems.Body)
	}
	if !strings.Contains(string(pItems.Body), `"reasoning":"bbbb"`) || !strings.Contains(string(pItems.Body), `"reasoning":"cccc"`) {
		t.Fatalf("expected newest two items bbbb and cccc in field 'reasoning': %s", pItems.Body)
	}

	pChars, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, req, bridge.Options{
		Chat: chat.Dialect{
			ReasoningReplay: chat.ReasoningReplay{
				Field:    "reasoning_content",
				MaxChars: 8,
			},
		},
	})
	if err != nil {
		t.Fatalf("Convert MaxChars: %v", err)
	}
	if strings.Contains(string(pChars.Body), "aaaa") {
		t.Fatalf("expected MaxChars=8 to drop oldest item aaaa: %s", pChars.Body)
	}
	if !strings.Contains(string(pChars.Body), `"reasoning_content":"bbbb"`) || !strings.Contains(string(pChars.Body), `"reasoning_content":"cccc"`) {
		t.Fatalf("expected newest fitting items bbbb and cccc kept under MaxChars=8: %s", pChars.Body)
	}

	// KeepTurns=1 keeps both assistant turns inside the latest tool loop while dropping an earlier user turn.
	toolLoopReq := []byte(`{
		"model":"m1",
		"max_tokens":128,
		"tools":[{"name":"lookup","input_schema":{"type":"object"}}],
		"context_management":{"edits":[{"type":"clear_thinking_20251015","keep":{"type":"thinking_turns","value":1}}]},
		"messages":[
			{"role":"user","content":"old question"},
			{"role":"assistant","content":[{"type":"thinking","thinking":"old_turn_thought"},{"type":"text","text":"old answer"}]},
			{"role":"user","content":"new question"},
			{"role":"assistant","content":[{"type":"thinking","thinking":"loop_step_1"},{"type":"tool_use","id":"tu_1","name":"lookup","input":{}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":"res"}]},
			{"role":"assistant","content":[{"type":"thinking","thinking":"loop_step_2"},{"type":"text","text":"final"}]},
			{"role":"user","content":"follow up"}
		]
	}`)
	// Add one final assistant message? No, follow-up user ended the tool loop so keep=1 keeps the tool loop!
	// Wait: "follow up" is a user message after the assistant, so the tool loop IS the last assistant turn.
	pKeep, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, toolLoopReq, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert toolLoop KeepTurns=1: %v", err)
	}
	if strings.Contains(string(pKeep.Body), "old_turn_thought") {
		t.Fatalf("expected old_turn_thought dropped under KeepTurns=1: %s", pKeep.Body)
	}
	if !strings.Contains(string(pKeep.Body), "loop_step_1") || !strings.Contains(string(pKeep.Body), "loop_step_2") {
		t.Fatalf("expected both tool-loop steps kept under KeepTurns=1: %s", pKeep.Body)
	}
}

func TestOffLowestHiddenWithPatchAndEffortTemplateKwargs(t *testing.T) {
	profile := &ir.ModelProfile{
		Style:                  ir.StyleEffort,
		Efforts:                []ir.Effort{ir.EffortLow, ir.EffortHigh, ir.EffortMax},
		EffortInTemplateKwargs: true,
		Off: ir.OffRule{
			Mode:  ir.OffLowestHidden,
			Patch: json.RawMessage(`{"chat_template_kwargs":{"enable_thinking":false}}`),
		},
	}

	anthReq := []byte(`{"model":"m1","max_tokens":128,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"hi"}]}`)
	pAnth, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, anthReq, bridge.Options{Profile: profile})
	if err != nil {
		t.Fatalf("Convert anthropic off: %v", err)
	}
	var obj map[string]any
	if err := json.Unmarshal(pAnth.Body, &obj); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if obj["reasoning_effort"] != "low" {
		t.Fatalf("expected reasoning_effort=low, got %v in %s", obj["reasoning_effort"], pAnth.Body)
	}
	kw, _ := obj["chat_template_kwargs"].(map[string]any)
	if kw == nil || kw["enable_thinking"] != false || kw["reasoning_effort"] != "low" {
		t.Fatalf("expected chat_template_kwargs with enable_thinking=false and reasoning_effort=low, got %s", pAnth.Body)
	}

	respReq := []byte(`{"model":"m1","reasoning":{"effort":"none"},"input":"hi"}`)
	pResp, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, respReq, bridge.Options{Profile: profile})
	if err != nil {
		t.Fatalf("Convert responses off: %v", err)
	}
	var obj2 map[string]any
	if err := json.Unmarshal(pResp.Body, &obj2); err != nil {
		t.Fatalf("unmarshal resp: %v", err)
	}
	kw2, _ := obj2["chat_template_kwargs"].(map[string]any)
	if obj2["reasoning_effort"] != "low" || kw2 == nil || kw2["enable_thinking"] != false || kw2["reasoning_effort"] != "low" {
		t.Fatalf("expected Responses off to emit low + chat_template_kwargs, got %s", pResp.Body)
	}
}

func TestOmitHistoryReasoningResponses(t *testing.T) {
	anthReq := []byte(`{
		"model":"m1",
		"max_tokens":128,
		"messages":[
			{"role":"user","content":"q1"},
			{"role":"assistant","content":[{"type":"thinking","thinking":"secret history thought"},{"type":"text","text":"a1"}]},
			{"role":"user","content":"q2"}
		]
	}`)
	p, err := bridge.Convert(ir.FormatAnthropic, ir.FormatResponses, anthReq, bridge.Options{
		OmitHistoryReasoning: true,
	})
	if err != nil {
		t.Fatalf("Convert OmitHistoryReasoning Responses: %v", err)
	}
	if strings.Contains(string(p.Body), "secret history thought") || strings.Contains(string(p.Body), `"type":"reasoning"`) {
		t.Fatalf("expected historical reasoning omitted in Responses body: %s", p.Body)
	}
}

func TestStreamStatsAcrossUpstreamFormats(t *testing.T) {
	// Responses upstream stream without usage -> Stream.Stats() collects text, reasoning, tool call.
	anthReq := []byte(`{"model":"m1","max_tokens":128,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	p, err := bridge.Convert(ir.FormatAnthropic, ir.FormatResponses, anthReq, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	st := p.NewStream()
	events := []sse.Event{
		{Name: "response.created", Data: []byte(`{"type":"response.created","response":{"id":"resp_1","model":"m1","status":"in_progress"}}`)},
		{Name: "response.reasoning_text.delta", Data: []byte(`{"type":"response.reasoning_text.delta","delta":"think "}`)},
		{Name: "response.output_text.delta", Data: []byte(`{"type":"response.output_text.delta","delta":"hello world"}`)},
		{Name: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather"}}`)},
		{Name: "response.function_call_arguments.delta", Data: []byte(`{"type":"response.function_call_arguments.delta","output_index":1,"item_id":"fc_1","delta":"{\"city\":\"NYC\"}"}`)},
		{Name: "response.output_item.done", Data: []byte(`{"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"NYC\"}","status":"completed"}}`)},
		{Name: "response.completed", Data: []byte(`{"type":"response.completed","response":{"id":"resp_1","model":"m1","status":"completed"}}`)},
	}
	var out []byte
	for _, ev := range events {
		out, err = st.Feed(out, ev)
		if err != nil {
			t.Fatalf("Feed %s: %v", ev.Name, err)
		}
	}
	_, err = st.End(out, ir.EndDone)
	if err != nil {
		t.Fatalf("End: %v", err)
	}
	stats := st.Stats()
	if stats.ToolCalls != 1 {
		t.Fatalf("expected 1 tool call, got %d", stats.ToolCalls)
	}
	for _, want := range []string{"think ", "hello world", "get_weather", `{"city":"NYC"}`} {
		if !strings.Contains(stats.Text, want) {
			t.Fatalf("expected stats.Text to contain %q, got %q", want, stats.Text)
		}
	}

	// Done-only Responses tool arguments (no arguments.delta): EvToolEnd still counts final arguments.
	stDoneOnly := p.NewStream()
	doneOnlyEvents := []sse.Event{
		{Name: "response.created", Data: []byte(`{"type":"response.created","response":{"id":"resp_2","model":"m1","status":"in_progress"}}`)},
		{Name: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_2","call_id":"call_2","name":"run_sql"}}`)},
		{Name: "response.function_call_arguments.done", Data: []byte(`{"type":"response.function_call_arguments.done","output_index":0,"item_id":"fc_2","arguments":"{\"query\":\"SELECT 1\"}"}`)},
		{Name: "response.output_item.done", Data: []byte(`{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_2","call_id":"call_2","name":"run_sql","arguments":"{\"query\":\"SELECT 1\"}","status":"completed"}}`)},
		{Name: "response.completed", Data: []byte(`{"type":"response.completed","response":{"id":"resp_2","model":"m1","status":"completed"}}`)},
	}
	var out2 []byte
	for _, ev := range doneOnlyEvents {
		out2, err = stDoneOnly.Feed(out2, ev)
		if err != nil {
			t.Fatalf("Feed done-only %s: %v", ev.Name, err)
		}
	}
	if _, err = stDoneOnly.End(out2, ir.EndDone); err != nil {
		t.Fatalf("End done-only: %v", err)
	}
	stats2 := stDoneOnly.Stats()
	if stats2.ToolCalls != 1 || !strings.Contains(stats2.Text, `SELECT 1`) {
		t.Fatalf("expected done-only tool args captured in Stats(), got %+v", stats2)
	}
}

func TestCodexReviewFixes_CrossFormatTamperNilEnvNextRoundDoneRepairMultiCandCustomStats(t *testing.T) {
	// P1 #1: Cross-format Responses unsigned seal cannot bypass Anthropic thinking text verification.
	t.Run("cross_format_unsigned_seal_cannot_bypass_anthropic_text_check", func(t *testing.T) {
		keyed := ir.NewSealCodec([]byte("test-hmac-key"))
		env := &ir.Env{ProviderID: "provA", Seal: keyed}
		respSeal := ir.SealUnsignedReasoning(env, ir.FormatResponses, ir.Origin{Format: ir.FormatChat, Provider: "provA"}, "original")
		if respSeal == "" {
			t.Fatal("expected non-empty responses seal")
		}
		tamperedReq := []byte(`{
			"model":"m1",
			"max_tokens":128,
			"messages":[
				{"role":"user","content":"q1"},
				{"role":"assistant","content":[
					{"type":"thinking","thinking":"tampered","signature":"` + respSeal + `"},
					{"type":"text","text":"a1"}
				]},
				{"role":"user","content":"q2"}
			]
		}`)
		p, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, tamperedReq, bridge.Options{
			ProviderID: "provA",
			Seal:       keyed,
			Chat: chat.Dialect{
				ReasoningReplay: chat.ReasoningReplay{
					Field:           "reasoning_content",
					RequireVerified: true,
				},
			},
		})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		if strings.Contains(string(p.Body), "tampered") {
			t.Fatalf("cross-format tampered thinking must NOT be replayed: %s", p.Body)
		}
	})

	// P1 #2: StreamEncoders with nil Env and SealUnsignedReasoning=true must not panic.
	t.Run("nil_env_stream_encoders_do_not_panic", func(t *testing.T) {
		aEnc := anthropic.NewStreamEncoder(anthropic.EncodeOptions{SealUnsignedReasoning: true})
		var dst []byte
		var err error
		dst, err = aEnc.Encode(dst, ir.Event{Type: ir.EvReasoning, Text: "x"})
		if err != nil {
			t.Fatalf("anthropic Encode: %v", err)
		}
		dst, err = aEnc.Close(dst)
		if err != nil {
			t.Fatalf("anthropic Close: %v", err)
		}
		if !strings.Contains(string(dst), `"signature_delta"`) {
			t.Fatalf("expected signature_delta with nil Env: %s", dst)
		}

		rEnc := responses.NewStreamEncoder(responses.EncodeOptions{
			SealUnsignedReasoning: true,
			Request:               &ir.Request{Include: []string{"reasoning.encrypted_content"}},
		})
		var rdst []byte
		rdst, err = rEnc.Encode(rdst, ir.Event{Type: ir.EvReasoning, Text: "x"})
		if err != nil {
			t.Fatalf("responses Encode: %v", err)
		}
		rdst, err = rEnc.Close(rdst)
		if err != nil {
			t.Fatalf("responses Close: %v", err)
		}
		if !strings.Contains(string(rdst), `"encrypted_content"`) {
			t.Fatalf("expected encrypted_content with nil Env: %s", rdst)
		}
	})

	// P1 #3: NextRound switching provider A -> provider B stamps each round's reasoning with its own origin.
	t.Run("next_round_provider_switch_preserves_per_round_origin", func(t *testing.T) {
		keyed := ir.NewSealCodec([]byte("round-key"))
		req := []byte(`{
			"model":"m1",
			"max_tokens":128,
			"stream":true,
			"tools":[{"name":"search","input_schema":{"type":"object"}}],
			"messages":[{"role":"user","content":"hi"}]
		}`)
		pA, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, req, bridge.Options{
			ProviderID:            "provA",
			Seal:                  keyed,
			SealUnsignedReasoning: true,
			Intercept:             []string{"search"},
		})
		if err != nil {
			t.Fatalf("Convert round 1: %v", err)
		}
		pB, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, req, bridge.Options{
			ProviderID:            "provB",
			Seal:                  keyed,
			SealUnsignedReasoning: true,
		})
		if err != nil {
			t.Fatalf("Convert round 2: %v", err)
		}

		feedTwoRounds := func(st *bridge.Stream) []byte {
			var out []byte
			r1 := []string{
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m1","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"thoughtA"}}]}`,
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"tc1","type":"function","function":{"name":"search","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`,
			}
			for _, ch := range r1 {
				out, err = st.Feed(out, sse.Event{Data: []byte(ch)})
				if err != nil {
					t.Fatalf("r1 Feed: %v", err)
				}
			}
			out, err = st.End(out, ir.EndDone)
			if err != nil {
				t.Fatalf("r1 End: %v", err)
			}
			if len(st.Intercepted()) != 1 {
				t.Fatalf("expected 1 intercepted call, got %d", len(st.Intercepted()))
			}
			st.NextRound(pB)
			r2 := []string{
				`{"id":"c2","object":"chat.completion.chunk","created":1,"model":"m1","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"thoughtB"}}]}`,
				`{"id":"c2","object":"chat.completion.chunk","created":1,"model":"m1","choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}]}`,
			}
			for _, ch := range r2 {
				out, err = st.Feed(out, sse.Event{Data: []byte(ch)})
				if err != nil {
					t.Fatalf("r2 Feed: %v", err)
				}
			}
			out, err = st.End(out, ir.EndDone)
			if err != nil {
				t.Fatalf("r2 End: %v", err)
			}
			return out
		}

		// Streaming mode: verify both signature_deltas unseal to provA and provB respectively.
		streamBytes := feedTwoRounds(pA.NewStream())
		var sigs []string
		for _, line := range strings.Split(string(streamBytes), "\n") {
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			var ev struct {
				Delta struct {
					Type      string `json:"type"`
					Signature string `json:"signature"`
				} `json:"delta"`
			}
			if json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &ev) == nil && ev.Delta.Type == "signature_delta" {
				sigs = append(sigs, ev.Delta.Signature)
			}
		}
		if len(sigs) != 2 {
			t.Fatalf("expected 2 signature_deltas across 2 rounds, got %d in:\n%s", len(sigs), streamBytes)
		}
		recA, okA := keyed.Unseal(sigs[0])
		recB, okB := keyed.Unseal(sigs[1])
		if !okA || !okB || recA.Origin.Provider != "provA" || recB.Origin.Provider != "provB" {
			t.Fatalf("stream origins mismatch: recA=(%+v,%v) recB=(%+v,%v)", recA.Origin, okA, recB.Origin, okB)
		}

		// Collector mode: verify collected thinking blocks carry provA and provB respectively.
		col := pA.NewCollector()
		feedTwoRounds(col)
		colBytes, _, err := col.Result()
		if err != nil {
			t.Fatalf("collector Result: %v", err)
		}
		var colMsg struct {
			Content []struct {
				Type      string `json:"type"`
				Thinking  string `json:"thinking"`
				Signature string `json:"signature"`
			} `json:"content"`
		}
		if err := json.Unmarshal(colBytes, &colMsg); err != nil {
			t.Fatalf("unmarshal collector: %v", err)
		}
		var colProvs []string
		for _, blk := range colMsg.Content {
			if blk.Type == "thinking" {
				rec, ok := keyed.Unseal(blk.Signature)
				if !ok {
					t.Fatalf("unseal collector thinking %q failed: %q", blk.Thinking, blk.Signature)
				}
				colProvs = append(colProvs, blk.Thinking+":"+rec.Origin.Provider)
			}
		}
		if strings.Join(colProvs, ",") != "thoughtA:provA,thoughtB:provB" {
			t.Fatalf("collector origins mismatch: %v (body=%s)", colProvs, colBytes)
		}
	})

	// P2 #5: Authoritative arguments.done / input.done repairs partial delta when output_item.done omits arguments/input.
	t.Run("arguments_done_repairs_partial_delta", func(t *testing.T) {
		anthReq := []byte(`{"model":"m1","max_tokens":128,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
		p, err := bridge.Convert(ir.FormatAnthropic, ir.FormatResponses, anthReq, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		st := p.NewCollector()
		evs := []sse.Event{
			{Name: "response.created", Data: []byte(`{"type":"response.created","response":{"id":"r1","model":"m1","status":"in_progress"}}`)},
			{Name: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"calc"}}`)},
			{Name: "response.function_call_arguments.delta", Data: []byte(`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"x\":"}`)},
			{Name: "response.function_call_arguments.done", Data: []byte(`{"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\"x\":1}"}`)},
			{Name: "response.output_item.done", Data: []byte(`{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"calc","status":"completed"}}`)},
			{Name: "response.completed", Data: []byte(`{"type":"response.completed","response":{"id":"r1","model":"m1","status":"completed"}}`)},
		}
		var out []byte
		for _, ev := range evs {
			out, err = st.Feed(out, ev)
			if err != nil {
				t.Fatalf("Feed %s: %v", ev.Name, err)
			}
		}
		if _, err = st.End(out, ir.EndDone); err != nil {
			t.Fatalf("End: %v", err)
		}
		body, _, err := st.Result()
		if err != nil {
			t.Fatalf("Result: %v", err)
		}
		if !strings.Contains(string(body), `"input":{"x":1}`) {
			t.Fatalf("expected repaired arguments {\"x\":1}, got: %s", body)
		}
		if !strings.Contains(st.Stats().Text, `{"x":1}`) {
			t.Fatalf("expected repaired arguments in Stats(), got: %+v", st.Stats())
		}
	})

	// P2 #6: Multi-candidate n=2 with identical call_id does not collapse ToolCalls or drop second name.
	t.Run("multi_candidate_same_call_id_stats", func(t *testing.T) {
		chatReq := []byte(`{"model":"m1","n":2,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
		p, err := bridge.Convert(ir.FormatChat, ir.FormatChat, chatReq, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		st := p.NewStream()
		chunks := []string{
			`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m1","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_same","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`,
			`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m1","choices":[{"index":1,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_same","type":"function","function":{"name":"g","arguments":"{}"}}]}}]}`,
			`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"},{"index":1,"delta":{},"finish_reason":"tool_calls"}]}`,
		}
		var out []byte
		for _, ch := range chunks {
			out, err = st.Feed(out, sse.Event{Data: []byte(ch)})
			if err != nil {
				t.Fatalf("Feed: %v", err)
			}
		}
		if _, err = st.End(out, ir.EndDone); err != nil {
			t.Fatalf("End: %v", err)
		}
		stats := st.Stats()
		if stats.ToolCalls != 2 || !strings.Contains(stats.Text, "f") || !strings.Contains(stats.Text, "g") {
			t.Fatalf("expected 2 tool calls with both f and g in Stats(), got %+v", stats)
		}
	})

	// P2 #7: Custom tool Stats().Text is identical whether delivered via delta or done-only.
	t.Run("custom_tool_stats_delta_vs_done_only_identical", func(t *testing.T) {
		req := []byte(`{"model":"m1","stream":true,"input":"hi"}`)
		p, err := bridge.Convert(ir.FormatResponses, ir.FormatResponses, req, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		runEvents := func(evs []sse.Event) bridge.StreamStats {
			st := p.NewStream()
			var out []byte
			for _, ev := range evs {
				out, err = st.Feed(out, ev)
				if err != nil {
					t.Fatalf("Feed %s: %v", ev.Name, err)
				}
			}
			if _, err = st.End(out, ir.EndDone); err != nil {
				t.Fatalf("End: %v", err)
			}
			return st.Stats()
		}
		deltaStats := runEvents([]sse.Event{
			{Name: "response.created", Data: []byte(`{"type":"response.created","response":{"id":"r1","model":"m1","status":"in_progress"}}`)},
			{Name: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","output_index":0,"item":{"type":"custom_tool_call","id":"ctc_1","call_id":"call_c","name":"run"}}`)},
			{Name: "response.custom_tool_call_input.delta", Data: []byte(`{"type":"response.custom_tool_call_input.delta","output_index":0,"delta":"abc\n\"q\""}`)},
			{Name: "response.output_item.done", Data: []byte(`{"type":"response.output_item.done","output_index":0,"item":{"type":"custom_tool_call","id":"ctc_1","call_id":"call_c","name":"run","input":"abc\n\"q\"","status":"completed"}}`)},
			{Name: "response.completed", Data: []byte(`{"type":"response.completed","response":{"id":"r1","model":"m1","status":"completed"}}`)},
		})
		doneOnlyStats := runEvents([]sse.Event{
			{Name: "response.created", Data: []byte(`{"type":"response.created","response":{"id":"r2","model":"m1","status":"in_progress"}}`)},
			{Name: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","output_index":0,"item":{"type":"custom_tool_call","id":"ctc_1","call_id":"call_c","name":"run"}}`)},
			{Name: "response.custom_tool_call_input.done", Data: []byte(`{"type":"response.custom_tool_call_input.done","output_index":0,"input":"abc\n\"q\""}`)},
			{Name: "response.output_item.done", Data: []byte(`{"type":"response.output_item.done","output_index":0,"item":{"type":"custom_tool_call","id":"ctc_1","call_id":"call_c","name":"run","status":"completed"}}`)},
			{Name: "response.completed", Data: []byte(`{"type":"response.completed","response":{"id":"r2","model":"m1","status":"completed"}}`)},
		})
		if deltaStats != doneOnlyStats || deltaStats.Text != "runabc\n\"q\"" {
			t.Fatalf("custom tool Stats mismatch: delta=%+v doneOnly=%+v", deltaStats, doneOnlyStats)
		}
	})
}
