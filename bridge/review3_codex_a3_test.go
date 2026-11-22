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

// Codex A3 P1 #1: LiveFirst with trailing text + finish_reason:"length" must NOT complete the
// open truncated tool call when releasing queued text or on EvRefusal/EvBuiltin.
func TestCodexA3_P1_01_LiveFirstTruncatedWithTrailingText(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","stream":true,"input":"run","tools":[{"type":"function","name":"exec","parameters":{"type":"object"}}]}`)
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, reqBody, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	s := plan.NewStream()
	raw := "data: " + `{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"exec","arguments":"{\"cmd\":"}}]},"finish_reason":null}]}` + "\n\n" +
		"data: " + `{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{"content":"tail"},"finish_reason":null}]}` + "\n\n" +
		"data: " + `{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}` + "\n\n" +
		"data: [DONE]\n\n"
	out := string(feedSSE(t, s, raw))
	if strings.Contains(out, "response.function_call_arguments.done") {
		t.Errorf("truncated LiveFirst call with trailing text must not emit function_call_arguments.done:\n%s", out)
	}
	if strings.Contains(out, `"type":"function_call"`) && strings.Contains(out, `"status":"completed"`) {
		// Check specifically that no function_call item was marked completed.
		r := sse.NewReader(bytes.NewReader([]byte(out)))
		for {
			ev, err := r.Next()
			if err != nil {
				break
			}
			var obj struct {
				Type string `json:"type"`
				Item struct {
					Type   string `json:"type"`
					Status string `json:"status"`
				} `json:"item"`
			}
			if json.Unmarshal(ev.Data, &obj) == nil && obj.Type == "response.output_item.done" && obj.Item.Type == "function_call" && obj.Item.Status == "completed" {
				t.Errorf("truncated LiveFirst function_call marked completed:\n%s", out)
			}
		}
	}
	if !strings.Contains(out, "event: response.incomplete") {
		t.Errorf("expected response.incomplete:\n%s", out)
	}
}

// Codex A3 P1 #2: Emulated stop hit must discard pending buffered/live tool calls while
// preserving legitimate text prefixes (design.md §9.6).
func TestCodexA3_P1_02_EmulatedStopDiscardsPendingCalls(t *testing.T) {
	reqBody := []byte(`{"model":"claude-sonnet-4","stream":true,"max_tokens":1024,"stop_sequences":["END"],"messages":[{"role":"user","content":"go"}],"tools":[{"name":"exec","input_schema":{"type":"object"}}]}`)
	plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, reqBody, bridge.Options{
		ToolRelease: ir.ToolReleaseBuffered,
		StopMode:    ir.StopEmulate,
	})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	s := plan.NewStream()
	raw := "data: " + `{"id":"1","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"exec","arguments":"{}"}}]},"finish_reason":null}]}` + "\n\n" +
		"data: " + `{"id":"1","choices":[{"index":0,"delta":{"content":"prefixENDsuffix"},"finish_reason":"stop"}]}` + "\n\n" +
		"data: [DONE]\n\n"
	out := string(feedSSE(t, s, raw))
	if strings.Contains(out, `"type":"tool_use"`) {
		t.Errorf("emulated stop must discard buffered tool call instead of delivering it:\n%s", out)
	}
	if !strings.Contains(out, `"prefix"`) {
		t.Errorf("emulated stop must preserve legitimate prefix:\n%s", out)
	}
}

// Codex A3 P1 #3: NewCollector() must NOT retain truncated unvalidated tool calls as completed.
func TestCodexA3_P1_03_CollectorDropsTruncatedCall(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","stream":false,"input":"run","tools":[{"type":"function","name":"exec","parameters":{"type":"object"}}]}`)
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, reqBody, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	col := plan.NewCollector()
	chunks := []string{
		`{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"exec","arguments":"{\"cmd\":"}}]},"finish_reason":null}]}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`,
		`[DONE]`,
	}
	for _, c := range chunks {
		if _, err := col.Feed(nil, sse.Event{Data: []byte(c)}); err != nil {
			t.Fatalf("Feed: %v", err)
		}
	}
	if _, err := col.End(nil, ir.EndEOF); err != nil {
		t.Fatalf("End: %v", err)
	}
	body, _, err := col.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if strings.Contains(string(body), `"type":"function_call"`) {
		t.Errorf("NewCollector retained truncated function_call as completed: %s", string(body))
	}
}

// Codex A3 P1 #4: Anthropic server_tool_use input_json_delta must NOT pollute a preceding tool_use call.
func TestCodexA3_P1_04_AnthropicServerToolDeltaNoPollution(t *testing.T) {
	reqBody := []byte(`{"model":"claude-sonnet-4","stream":true,"max_tokens":1024,"messages":[{"role":"user","content":"go"}],"tools":[{"name":"exec","input_schema":{"type":"object"}}]}`)
	plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatAnthropic, reqBody, bridge.Options{
		ToolRelease: ir.ToolReleaseBuffered,
	})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	s := plan.NewStream()
	raw := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"u1","name":"exec","input":{}}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"server_tool_use","id":"s1","name":"web_search","input":{}}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"q\"}"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":1}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n"

	out := string(feedSSE(t, s, raw))
	// Decode tool_use block for u1 and verify its args are still "{}", not {"query":"q"}.
	r := sse.NewReader(bytes.NewReader([]byte(out)))
	inExec := false
	var execArgs strings.Builder
	for {
		ev, err := r.Next()
		if err != nil {
			break
		}
		var obj struct {
			Type         string `json:"type"`
			ContentBlock struct {
				Type string `json:"type"`
				ID   string `json:"id"`
			} `json:"content_block"`
			Delta struct {
				Type        string `json:"type"`
				PartialJSON string `json:"partial_json"`
			} `json:"delta"`
		}
		if json.Unmarshal(ev.Data, &obj) == nil {
			if obj.Type == "content_block_start" {
				inExec = obj.ContentBlock.Type == "tool_use" && obj.ContentBlock.ID == "u1"
			} else if obj.Type == "content_block_delta" && inExec && obj.Delta.Type == "input_json_delta" {
				execArgs.WriteString(obj.Delta.PartialJSON)
			} else if obj.Type == "content_block_stop" {
				inExec = false
			}
		}
	}
	if execArgs.String() != "{}" && execArgs.String() != "" {
		t.Errorf("server_tool_use delta polluted tool_use u1 args: got %q, want {}", execArgs.String())
	}
}

// Codex A3 P1 #5: Gemini ParseRequest must unseal thoughtSignature on functionCall and text/thought parts.
func TestCodexA3_P1_05_GeminiUnsealsFunctionCallAndTextSignature(t *testing.T) {
	seal := ir.NewSealCodec([]byte("secret-key"))
	env := &ir.Env{ProviderID: "prov-g", Seal: seal}
	// Encode a Gemini tool call with ProviderID -> produces sealed thoughtSignature.
	enc := gemini.NewStreamEncoder(gemini.EncodeOptions{Env: env, AltSSE: true})
	var dst []byte
	dst, _ = enc.Encode(dst, ir.Event{
		Type: ir.EvToolStart, ID: "c1", Name: "lookup",
		Signature: &ir.Signature{Value: "raw_sig_123", Origin: ir.Origin{Format: ir.FormatGemini, Provider: "prov-g"}},
	})
	dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvToolArgs, ID: "c1", Text: `{}`})
	dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvToolEnd, ID: "c1", Arguments: json.RawMessage(`{}`)})
	dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStop, Stop: &ir.Stop{Reason: ir.StopToolUse}})
	dst, _ = enc.Close(dst)

	// Extract thoughtSignature from SSE chunk.
	var sealedSig string
	for _, line := range strings.Split(string(dst), "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var resp struct {
			Candidates []struct {
				Content struct {
					Parts []struct {
						ThoughtSignature string `json:"thoughtSignature"`
					} `json:"parts"`
				} `json:"content"`
			} `json:"candidates"`
		}
		if json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &resp) == nil && len(resp.Candidates) > 0 {
			for _, pt := range resp.Candidates[0].Content.Parts {
				if pt.ThoughtSignature != "" {
					sealedSig = pt.ThoughtSignature
				}
			}
		}
	}
	if !ir.IsSealed(sealedSig) {
		t.Fatalf("expected sealed signature, got %q", sealedSig)
	}

	// Replay back into gemini.ParseRequest and build back for prov-g: must unwrap back to "raw_sig_123".
	replayReq, _ := json.Marshal(map[string]any{
		"contents": []any{
			map[string]any{"role": "model", "parts": []any{
				map[string]any{"functionCall": map[string]any{"id": "c1", "name": "lookup", "args": map[string]any{}}, "thoughtSignature": sealedSig},
			}},
		},
	})
	irReq, err := gemini.ParseRequest(replayReq, gemini.ParseOptions{Env: env})
	if err != nil {
		t.Fatalf("ParseRequest: %v", err)
	}
	sig := irReq.Messages[0].Parts[0].Call.Signature
	if sig == nil || sig.Value != "raw_sig_123" || sig.Origin.Provider != "prov-g" {
		t.Fatalf("expected unsealed raw_sig_123 with origin prov-g, got %+v", sig)
	}
}

// Codex A3 P2 #9, #11, #12, #14, #15, #16, #17, #18, #20, #22, #23, #25, #26 regressions.
func TestCodexA3_P2_SelectedFixes(t *testing.T) {
	t.Run("P2_09_candidate_scoped_call_id", func(t *testing.T) {
		reqBody := []byte(`{"model":"gemini-2.5-pro","stream":true,"n":2,"messages":[{"role":"user","content":"go"}],"tools":[{"type":"function","function":{"name":"A","parameters":{"type":"object"}}},{"type":"function","function":{"name":"B","parameters":{"type":"object"}}}]}`)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatGemini, reqBody, bridge.Options{ToolRelease: ir.ToolReleaseBuffered})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		col := plan.NewCollector()
		chunks := []string{
			`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"functionCall":{"id":"x","name":"A","args":{}}}]}},{"index":1,"content":{"role":"model","parts":[{"functionCall":{"id":"x","name":"B","args":{}}}]}}]}`,
			`{"candidates":[{"index":0,"finishReason":"STOP"},{"index":1,"finishReason":"STOP"}]}`,
		}
		for _, c := range chunks {
			if _, err := col.Feed(nil, sse.Event{Data: []byte(c)}); err != nil {
				t.Fatalf("Feed: %v", err)
			}
		}
		_, _ = col.End(nil, ir.EndEOF)
		_, resp, err := col.Result()
		if err != nil {
			t.Fatalf("Result: %v", err)
		}
		if len(resp.Outputs) != 2 || len(resp.Outputs[0].Parts) != 1 || len(resp.Outputs[1].Parts) != 1 ||
			resp.Outputs[0].Parts[0].Call.Name != "A" || resp.Outputs[1].Parts[0].Call.Name != "B" {
			t.Errorf("candidate 0 and 1 same call ID collided: %+v", resp.Outputs)
		}
	})

	t.Run("P2_11_nonstream_stop_across_adjacent_text_parts", func(t *testing.T) {
		reqBody := []byte(`{"model":"gpt-5","stream":false,"stop":["bc"],"messages":[{"role":"user","content":"hi"}]}`)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatResponses, reqBody, bridge.Options{StopMode: ir.StopEmulate})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		upstreamResp := []byte(`{
			"id":"r1","object":"response","model":"gpt-5","status":"completed",
			"output":[{"type":"message","id":"m1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ab"},{"type":"output_text","text":"cd"}]}]
		}`)
		_, irResp, err := plan.ConvertResponse(upstreamResp)
		if err != nil {
			t.Fatalf("ConvertResponse: %v", err)
		}
		if got := ir.TextOf(irResp.Outputs[0].Parts, ""); got != "a" {
			t.Errorf("expected stop match across adjacent text parts -> 'a', got %q", got)
		}
	})

	t.Run("P2_12_mixed_emulated_and_native_finish_no_false_error", func(t *testing.T) {
		reqBody := []byte(`{"model":"gpt-5","stream":true,"n":2,"stop":["END"],"messages":[{"role":"user","content":"hi"}]}`)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatChat, reqBody, bridge.Options{StopMode: ir.StopEmulate})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		s := plan.NewStream()
		var out []byte
		for _, line := range []string{
			`{"id":"1","choices":[{"index":0,"delta":{"content":"aEND"},"finish_reason":null}]}`,
			`{"id":"1","choices":[{"index":1,"delta":{"content":"b"},"finish_reason":"stop"}]}`,
		} {
			out, err = s.Feed(out, sse.Event{Data: []byte(line)})
			if err != nil {
				t.Fatalf("Feed: %v", err)
			}
		}
		out, err = s.End(out, ir.EndEOF)
		if err != nil {
			t.Fatalf("End: %v", err)
		}
		if strings.Contains(string(out), `"error"`) {
			t.Errorf("mixed emulated+native finish for n=2 falsely errored:\n%s", string(out))
		}
	})

	t.Run("P2_13_partial_usage_derived_counts", func(t *testing.T) {
		gdec := gemini.NewStreamDecoder(gemini.DecodeOptions{})
		var gu ir.Usage
		e1, _ := gdec.Decode(nil, sse.Event{Data: []byte(`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"hi"}]}}],"usageMetadata":{"candidatesTokenCount":5}}`)})
		for _, e := range e1 {
			if e.Usage != nil {
				gu.Merge(*e.Usage)
			}
		}
		e2, _ := gdec.Decode(nil, sse.Event{Data: []byte(`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"!"}]},"finishReason":"STOP"}],"usageMetadata":{"thoughtsTokenCount":2}}`)})
		for _, e := range e2 {
			if e.Usage != nil {
				gu.Merge(*e.Usage)
			}
		}
		if gu.OutputTokens != 7 {
			t.Errorf("Gemini partial candidates(5)+thoughts(2) output: want 7, got %d (%+v)", gu.OutputTokens, gu)
		}

		cdec := chat.NewStreamDecoder(chat.DecodeOptions{})
		var cu ir.Usage
		c1, _ := cdec.Decode(nil, sse.Event{Data: []byte(`{"id":"1","choices":[],"usage":{"prompt_tokens":100}}`)})
		for _, e := range c1 {
			if e.Usage != nil {
				cu.Merge(*e.Usage)
			}
		}
		c2, _ := cdec.Decode(nil, sse.Event{Data: []byte(`{"id":"1","choices":[],"usage":{"prompt_tokens_details":{"cached_tokens":20}}}`)})
		for _, e := range c2 {
			if e.Usage != nil {
				cu.Merge(*e.Usage)
			}
		}
		if cu.PromptTokens() != 100 || cu.CacheReadTokens != 20 {
			t.Errorf("Chat partial prompt(100)+cached(20): want PromptTokens=100 CacheRead=20, got %+v", cu)
		}
	})

	t.Run("P2_16_chat_file_data_uri_roundtrip", func(t *testing.T) {
		body := []byte(`{"model":"gpt-5","messages":[{"role":"user","content":[{"type":"file","file":{"filename":"a.pdf","file_data":"data:application/pdf;base64,JVBERi0="}}]}]}`)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatChat, body, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		if !strings.Contains(string(plan.Body), `"data:application/pdf;base64,JVBERi0="`) {
			t.Errorf("Chat->Chat file_data lost data URI prefix: %s", string(plan.Body))
		}
	})

	t.Run("P2_17_responses_stream_preserves_annotations_in_completed", func(t *testing.T) {
		enc := responses.NewStreamEncoder(responses.EncodeOptions{Env: &ir.Env{}, Model: "gpt-5"})
		var dst []byte
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStart, Model: "gpt-5"})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvText, Text: "docs"})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvCitation, Citation: &ir.Citation{Type: "url_citation", URL: "https://example.test", Title: "Docs"}})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStop, Stop: &ir.Stop{Reason: ir.StopEndTurn}})
		dst, _ = enc.Close(dst)
		if !strings.Contains(string(dst), `"annotations":[`) {
			t.Errorf("Responses StreamEncoder dropped annotations from output_item.done/response.completed:\n%s", string(dst))
		}
	})

	t.Run("P2_22_gemini_dynamic_budget_to_anthropic_not_clamped_to_1024", func(t *testing.T) {
		body := []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"thinkingBudget":-1}}}`)
		plan, err := bridge.Convert(ir.FormatGemini, ir.FormatAnthropic, body, bridge.Options{Model: "claude-sonnet-4"})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		if strings.Contains(string(plan.Body), `"budget_tokens":1024`) {
			t.Errorf("Gemini dynamic budget -1 was silently clamped to min 1024 in Anthropic: %s", string(plan.Body))
		}
	})

	t.Run("P2_23_reject_bare_discriminator_objects", func(t *testing.T) {
		if _, err := anthropic.ParseResponse([]byte(`{"type":"message"}`), anthropic.DecodeOptions{}); err == nil {
			t.Errorf("anthropic.ParseResponse accepted bare {\"type\":\"message\"}")
		}
		if _, err := responses.ParseResponse([]byte(`{"object":"response"}`), responses.DecodeOptions{}); err == nil {
			t.Errorf("responses.ParseResponse accepted bare {\"object\":\"response\"}")
		}
	})

	t.Run("P2_25_responses_history_preserves_refusal_part", func(t *testing.T) {
		body := []byte(`{"model":"gpt-5","messages":[{"role":"user","content":"bad"},{"role":"assistant","content":null,"refusal":"cannot assist"}]}`)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatResponses, body, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		if !strings.Contains(string(plan.Body), `"type":"refusal"`) {
			t.Errorf("Responses history downgraded assistant refusal to output_text: %s", string(plan.Body))
		}
	})
}
