package bridge_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/chat"
	"github.com/ihainan/llm-protocol-bridge/ir"
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
	// Rolling window refills at index 2 (count==2), so window starts at "cccc".
	if strings.Contains(string(pItems.Body), "aaaa") || strings.Contains(string(pItems.Body), "bbbb") {
		t.Fatalf("expected rolling MaxItems=2 window to drop older items: %s", pItems.Body)
	}
	if !strings.Contains(string(pItems.Body), `"reasoning":"cccc"`) {
		t.Fatalf("expected newest item in field 'reasoning': %s", pItems.Body)
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
	if strings.Contains(string(pChars.Body), "aaaa") || strings.Contains(string(pChars.Body), "bbbb") {
		t.Fatalf("expected rolling MaxChars=8 window to reset at cccc: %s", pChars.Body)
	}
	if !strings.Contains(string(pChars.Body), `"reasoning_content":"cccc"`) {
		t.Fatalf("expected cccc kept under MaxChars=8: %s", pChars.Body)
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
