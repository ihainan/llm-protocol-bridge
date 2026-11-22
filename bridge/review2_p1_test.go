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

// Review2 P1 #1: LiveFirst must NOT mark a truncated or stop-emulated unvalidated
// tool call as completed in Responses SSE output.
func TestReview2_P1_LiveFirstTruncatedCall(t *testing.T) {
	t.Run("truncated_by_length", func(t *testing.T) {
		reqBody := []byte(`{"model":"gpt-5","stream":true,"input":"run","tools":[{"type":"function","name":"exec","parameters":{"type":"object","properties":{"cmd":{"type":"string"}}}}]}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, reqBody, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		s := plan.NewStream()
		raw := "data: " + `{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"exec","arguments":"{\"cmd\":"}}]},"finish_reason":null}]}` + "\n\n" +
			"data: " + `{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}` + "\n\n" +
			"data: [DONE]\n\n"
		out := string(feedSSE(t, s, raw))
		if strings.Contains(out, "response.function_call_arguments.done") {
			t.Errorf("truncated LiveFirst tool call must not emit function_call_arguments.done:\n%s", out)
		}
		if strings.Contains(out, `"status":"completed"`) {
			t.Errorf("truncated LiveFirst tool call must not be marked status:completed:\n%s", out)
		}
		if !strings.Contains(out, "event: response.incomplete") {
			t.Errorf("expected response.incomplete terminal event:\n%s", out)
		}
	})

	t.Run("emulated_stop_with_invalid_live_call", func(t *testing.T) {
		reqBody := []byte(`{"model":"gpt-5","stream":true,"input":"run","tools":[{"type":"function","name":"exec","parameters":{"type":"object"}}]}`)
		req, err := responses.ParseRequest(reqBody, responses.ParseOptions{})
		if err != nil {
			t.Fatalf("ParseRequest: %v", err)
		}
		req.Stop = []string{"STOP"}
		plan, err := bridge.Build(req, ir.FormatResponses, ir.FormatChat, bridge.Options{
			StopMode: ir.StopEmulate,
		})
		if err != nil {
			t.Fatalf("Build: %v", err)
		}
		s := plan.NewStream()
		raw := "data: " + `{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"exec","arguments":"{\"cmd\":"}}]},"finish_reason":null}]}` + "\n\n" +
			"data: " + `{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{"content":"hello STOP world"},"finish_reason":null}]}` + "\n\n" +
			"data: " + `{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n" +
			"data: [DONE]\n\n"
		out := string(feedSSE(t, s, raw))
		if strings.Contains(out, "response.function_call_arguments.done") || strings.Contains(out, `"type":"function_call","status":"completed"`) {
			t.Errorf("emulated stop must not complete an unvalidated LiveFirst call:\n%s", out)
		}
		if !strings.Contains(out, `"hello "`) {
			t.Errorf("emulated stop should preserve text prefix before STOP:\n%s", out)
		}
	})
}

// Review2 P1 #2: Gemini mixed explicit-ID and ID-less functionResponse correlation
// must dequeue explicit IDs and scope pending calls to the latest model turn.
func TestReview2_P1_GeminiMixedIDCorrelation(t *testing.T) {
	t.Run("explicit_id_dequeues_from_pendingByName", func(t *testing.T) {
		body := []byte(`{
			"contents": [
				{"role":"user","parts":[{"text":"read both"}]},
				{"role":"model","parts":[
					{"functionCall":{"id":"a","name":"Read","args":{"path":"/a"}}},
					{"functionCall":{"name":"Read","args":{"path":"/b"}}}
				]},
				{"role":"user","parts":[
					{"functionResponse":{"id":"a","name":"Read","response":{"result":"body a"}}},
					{"functionResponse":{"name":"Read","response":{"result":"body b"}}}
				]}
			]
		}`)
		req, err := gemini.ParseRequest(body, gemini.ParseOptions{Model: "gemini-2.5-pro"})
		if err != nil {
			t.Fatalf("ParseRequest: %v", err)
		}
		call1ID := req.Messages[1].Parts[0].Call.ID
		call2ID := req.Messages[1].Parts[1].Call.ID
		res1ID := req.Messages[2].Parts[0].Result.CallID
		res2ID := req.Messages[2].Parts[1].Result.CallID
		if call1ID != "a" || call2ID == "" || call2ID == "a" {
			t.Fatalf("unexpected call IDs: call1=%q call2=%q", call1ID, call2ID)
		}
		if res1ID != "a" {
			t.Errorf("first result should match explicit ID a, got %q", res1ID)
		}
		if res2ID != call2ID {
			t.Errorf("second (ID-less) result should match second call %q, got %q", call2ID, res2ID)
		}
	})

	t.Run("scoped_to_latest_model_turn", func(t *testing.T) {
		body := []byte(`{
			"contents": [
				{"role":"user","parts":[{"text":"turn 1"}]},
				{"role":"model","parts":[{"functionCall":{"id":"old_id","name":"Read","args":{"path":"/old"}}}]},
				{"role":"user","parts":[{"text":"ignore old, try again"}]},
				{"role":"model","parts":[{"functionCall":{"name":"Read","args":{"path":"/new"}}}]},
				{"role":"user","parts":[{"functionResponse":{"name":"Read","response":{"result":"body new"}}}]}
			]
		}`)
		req, err := gemini.ParseRequest(body, gemini.ParseOptions{Model: "gemini-2.5-pro"})
		if err != nil {
			t.Fatalf("ParseRequest: %v", err)
		}
		newCallID := req.Messages[3].Parts[0].Call.ID
		resID := req.Messages[4].Parts[0].Result.CallID
		if resID == "old_id" || resID != newCallID {
			t.Errorf("ID-less functionResponse must match latest model turn %q, not stale %q (got %q)", newCallID, "old_id", resID)
		}
	})
}

// Review2 P1 #3 (and #36): Multi-chunk streaming signatures must be concatenated
// before sealing in both Anthropic and Responses encoders.
func TestReview2_P1_StreamingMultiChunkSignature(t *testing.T) {
	seal := ir.NewSealCodec([]byte("test-secret-key"))
	origin := ir.Origin{Format: ir.FormatAnthropic, Provider: "prov-a"}

	t.Run("anthropic_concatenates_before_sealing", func(t *testing.T) {
		enc := anthropic.NewStreamEncoder(anthropic.EncodeOptions{
			Env:   &ir.Env{ProviderID: "prov-a", Seal: seal},
			Model: "claude-sonnet-4",
		})
		var dst []byte
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStart, Model: "claude-sonnet-4"})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvReasoningStart})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvReasoning, Text: "step 1"})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvSignature, Signature: &ir.Signature{Value: "sig_A", Origin: origin}})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvSignature, Signature: &ir.Signature{Value: "sig_B", Origin: origin}})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvText, Text: "done"})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStop, Stop: &ir.Stop{Reason: ir.StopEndTurn}})
		dst, _ = enc.Close(dst)

		// Decode SSE and collect concatenated signature_delta.
		r := sse.NewReader(bytes.NewReader(dst))
		var fullSig strings.Builder
		for {
			ev, err := r.Next()
			if err != nil {
				break
			}
			var obj struct {
				Type  string `json:"type"`
				Delta struct {
					Type      string `json:"type"`
					Signature string `json:"signature"`
				} `json:"delta"`
			}
			if json.Unmarshal(ev.Data, &obj) == nil && obj.Type == "content_block_delta" && obj.Delta.Type == "signature_delta" {
				fullSig.WriteString(obj.Delta.Signature)
			}
		}
		rec, ok := seal.Unseal(fullSig.String())
		if !ok {
			t.Fatalf("sealed multi-chunk Anthropic signature failed to unseal: %q", fullSig.String())
		}
		if rec.Value != "sig_Asig_B" {
			t.Errorf("expected unsealed value sig_Asig_B, got %q", rec.Value)
		}
	})

	t.Run("responses_concatenates_before_sealing", func(t *testing.T) {
		enc := responses.NewStreamEncoder(responses.EncodeOptions{
			Env:   &ir.Env{ProviderID: "prov-a", Seal: seal},
			Model: "gpt-5",
		})
		var dst []byte
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStart, Model: "gpt-5"})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvReasoningStart})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvReasoning, Text: "step 1"})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvSignature, Signature: &ir.Signature{Value: "sig_A", Origin: origin}})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvSignature, Signature: &ir.Signature{Value: "sig_B", Origin: origin}})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvText, Text: "done"})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStop, Stop: &ir.Stop{Reason: ir.StopEndTurn}})
		dst, _ = enc.Close(dst)

		// Extract encrypted_content from response.output_item.done for reasoning.
		r := sse.NewReader(bytes.NewReader(dst))
		var encContent string
		for {
			ev, err := r.Next()
			if err != nil {
				break
			}
			var obj struct {
				Type string `json:"type"`
				Item struct {
					Type             string `json:"type"`
					EncryptedContent string `json:"encrypted_content"`
				} `json:"item"`
			}
			if json.Unmarshal(ev.Data, &obj) == nil && obj.Type == "response.output_item.done" && obj.Item.Type == "reasoning" {
				encContent = obj.Item.EncryptedContent
			}
		}
		rec, ok := seal.Unseal(encContent)
		if !ok {
			t.Fatalf("Responses sealed multi-chunk signature failed to unseal: %q", encContent)
		}
		if rec.Value != "sig_Asig_B" {
			t.Errorf("expected unsealed value sig_Asig_B, got %q", rec.Value)
		}
	})
}

// Review2 P1 #4: StreamResponse with Intercept must not silently drop calls and emit
// an empty completed response; it must return an explicit error.
func TestReview2_P1_StreamResponseWithIntercept(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","input":"search docs","tools":[{"type":"function","name":"search","parameters":{"type":"object"}}]}`)
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, reqBody, bridge.Options{
		Intercept: []string{"search"},
	})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	upstreamResp := []byte(`{
		"id":"chatcmpl-1",
		"object":"chat.completion",
		"model":"gpt-5",
		"choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"search","arguments":"{}"}}]},"finish_reason":"tool_calls"}]
	}`)
	out, err := plan.StreamResponse(nil, upstreamResp)
	if err == nil {
		t.Fatalf("expected error from StreamResponse when Intercept captures tool calls, got empty/completed SSE:\n%s", string(out))
	}
}

// Review2 P1 #5: NextRound must refresh effectiveReasoningOut so switching to a model
// with OffLowestHidden hides reasoning in round 2.
func TestReview2_P1_NextRoundAutoHiddenReasoning(t *testing.T) {
	req1Body := []byte(`{"model":"m1","stream":true,"reasoning_effort":"none","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"search","parameters":{"type":"object"}}}]}`)
	plan1, err := bridge.Convert(ir.FormatChat, ir.FormatChat, req1Body, bridge.Options{
		Intercept: []string{"search"},
		Profile: &ir.ModelProfile{
			Style: ir.StyleEffort,
			Off:   ir.OffRule{Mode: ir.OffNative},
		},
	})
	if err != nil {
		t.Fatalf("Convert round 1: %v", err)
	}
	s := plan1.NewStream()
	round1SSE := "data: " + `{"id":"c1","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"search","arguments":"{}"}}]},"finish_reason":null}]}` + "\n\n" +
		"data: " + `{"id":"c1","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n" +
		"data: [DONE]\n\n"
	_ = feedSSE(t, s, round1SSE)
	if len(s.Intercepted()) != 1 {
		t.Fatalf("expected 1 intercepted call, got %d", len(s.Intercepted()))
	}

	// Round 2 switches to a model that uses OffLowestHidden.
	plan2, err := bridge.Convert(ir.FormatChat, ir.FormatChat, req1Body, bridge.Options{
		Profile: &ir.ModelProfile{
			Style:   ir.StyleEffort,
			Efforts: []ir.Effort{ir.EffortLow, ir.EffortHigh},
			Off:     ir.OffRule{Mode: ir.OffLowestHidden},
		},
	})
	if err != nil {
		t.Fatalf("Convert round 2: %v", err)
	}
	s.NextRound(plan2)
	round2SSE := "data: " + `{"id":"c2","object":"chat.completion.chunk","model":"m2","choices":[{"index":0,"delta":{"reasoning_content":"secret","content":"visible"},"finish_reason":"stop"}]}` + "\n\n" +
		"data: [DONE]\n\n"
	out2 := string(feedSSE(t, s, round2SSE))
	if strings.Contains(out2, "secret") {
		t.Errorf("NextRound failed to update auto-hidden reasoning policy; leaked secret:\n%s", out2)
	}
	if !strings.Contains(out2, "visible") {
		t.Errorf("expected visible content in round 2:\n%s", out2)
	}
}

// Review2 P1-Verify: Interleaved name and arguments chunks must NOT bypass Intercept
// with a truncated name prefix.
func TestReview2_P1_Verify_InterleavedNameAndArgsIntercept(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","stream":true,"messages":[{"role":"user","content":"search"}],"tools":[{"type":"function","function":{"name":"web_search","parameters":{"type":"object"}}}]}`)
	plan, err := bridge.Convert(ir.FormatChat, ir.FormatChat, reqBody, bridge.Options{
		Intercept: []string{"web_search"},
	})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	s := plan.NewStream()
	raw := "data: " + `{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"web_","arguments":"{"}}]},"finish_reason":null}]}` + "\n\n" +
		"data: " + `{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"search","arguments":"}"}}]},"finish_reason":null}]}` + "\n\n" +
		"data: " + `{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n" +
		"data: [DONE]\n\n"
	out := string(feedSSE(t, s, raw))
	if strings.Contains(out, `"web_"`) || strings.Contains(out, `"web_search"`) {
		t.Errorf("intercepted tool call leaked to client:\n%s", out)
	}
	ic := s.Intercepted()
	if len(ic) != 1 || ic[0].Name != "web_search" {
		t.Errorf("expected intercepted call web_search, got %+v", ic)
	}
}

// Review2 P2 (introduced #1): Deferred EvToolStart must not reorder a zero-arg tool call
// after subsequent text content.
func TestReview2_P2_DelayedToolStartPreservesOrder(t *testing.T) {
	dec := chat.NewStreamDecoder(chat.DecodeOptions{})
	var all []ir.Event
	chunks := []string{
		`{"id":"1","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"f","arguments":""}}]},"finish_reason":null}]}`,
		`{"id":"1","choices":[{"index":0,"delta":{"content":"tail"},"finish_reason":null}]}`,
		`{"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
	}
	for _, ch := range chunks {
		var err error
		all, err = dec.Decode(all, sse.Event{Data: []byte(ch)})
		if err != nil {
			t.Fatalf("Decode: %v", err)
		}
	}
	toolIdx, textIdx := -1, -1
	for i, ev := range all {
		if ev.Type == ir.EvToolStart && toolIdx < 0 {
			toolIdx = i
		}
		if ev.Type == ir.EvText && ev.Text == "tail" && textIdx < 0 {
			textIdx = i
		}
	}
	if toolIdx < 0 || textIdx < 0 || toolIdx > textIdx {
		t.Errorf("expected EvToolStart before EvText(\"tail\"), got toolIdx=%d textIdx=%d events=%+v", toolIdx, textIdx, all)
	}
}

// Review2 P2 (introduced #2): Tool call ID containing "::" must round-trip Gemini
// thoughtSignature through Chat client without corruption.
func TestReview2_P2_CallIDWithDoubleColonSignatureRoundtrip(t *testing.T) {
	seal := ir.NewSealCodec([]byte("secret-key"))
	env := &ir.Env{ProviderID: "google", Seal: seal}
	resp := &ir.Response{
		ID:    "resp_1",
		Model: "gemini-2.5-pro",
		Outputs: []ir.Output{{
			Index: 0,
			Parts: []ir.Part{{
				Type: ir.PartToolCall,
				Call: &ir.ToolCall{
					ID:        "up::42",
					Name:      "lookup",
					Arguments: json.RawMessage(`{"q":"x"}`),
					Signature: &ir.Signature{
						Value:  "sig_gem_99",
						Origin: ir.Origin{Format: ir.FormatGemini, Provider: "google"},
					},
				},
			}},
			Stop: ir.Stop{Reason: ir.StopToolUse},
		}},
	}
	encoded, err := chat.BuildResponse(resp, chat.EncodeOptions{Env: env, Model: "gemini-2.5-pro"})
	if err != nil {
		t.Fatalf("BuildResponse: %v", err)
	}
	var parsedResp struct {
		Choices []struct {
			Message struct {
				ToolCalls []struct {
					ID string `json:"id"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(encoded, &parsedResp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	wireID := parsedResp.Choices[0].Message.ToolCalls[0].ID

	nextChatReq, _ := json.Marshal(map[string]any{
		"model": "gemini-2.5-pro",
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{
				"role": "assistant",
				"tool_calls": []any{
					map[string]any{
						"id":       wireID,
						"type":     "function",
						"function": map[string]any{"name": "lookup", "arguments": `{"q":"x"}`},
					},
				},
			},
			map[string]any{"role": "tool", "tool_call_id": wireID, "content": "ok"},
		},
	})
	irReq, err := chat.ParseRequest(nextChatReq, chat.ParseOptions{Env: env})
	if err != nil {
		t.Fatalf("ParseRequest: %v", err)
	}
	call := irReq.Messages[1].Parts[0].Call
	if call.ID != "up::42" {
		t.Errorf("restored call ID: want up::42, got %q", call.ID)
	}
	if call.Signature == nil || call.Signature.Value != "sig_gem_99" {
		t.Errorf("restored signature: want sig_gem_99, got %+v", call.Signature)
	}
	res := irReq.Messages[2].Parts[0].Result
	if res.CallID != "up::42" {
		t.Errorf("restored tool result CallID: want up::42, got %q", res.CallID)
	}
}

// Review2 P2 (introduced #3): Direct format builders (chat, anthropic, gemini) must
// reject previous_response_id instead of silently dropping it.
func TestReview2_P2_DirectBuilderRejectsPreviousResponseID(t *testing.T) {
	req, err := responses.ParseRequest([]byte(`{"model":"m","previous_response_id":"resp_0","input":"continue"}`), responses.ParseOptions{})
	if err != nil {
		t.Fatalf("ParseRequest: %v", err)
	}
	if _, err := chat.BuildRequest(req, chat.BuildOptions{}); err == nil {
		t.Errorf("chat.BuildRequest should reject previous_response_id")
	}
	if _, err := anthropic.BuildRequest(req, anthropic.BuildOptions{}); err == nil {
		t.Errorf("anthropic.BuildRequest should reject previous_response_id")
	}
	if _, err := gemini.BuildRequest(req, gemini.BuildOptions{}); err == nil {
		t.Errorf("gemini.BuildRequest should reject previous_response_id")
	}
}
