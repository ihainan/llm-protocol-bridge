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

// Review2 #11: Gemini streaming inlineData image must decode to EvImage and encode back in Gemini stream.
func TestReview2_P2_11_GeminiStreamInlineImage(t *testing.T) {
	dec := gemini.NewStreamDecoder(gemini.DecodeOptions{})
	chunk := `{"candidates":[{"index":0,"content":{"role":"model","parts":[{"inlineData":{"mimeType":"image/png","data":"AQ=="}}]},"finishReason":"STOP"}]}`
	evs, err := dec.Decode(nil, sse.Event{Data: []byte(chunk)})
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	found := false
	for _, ev := range evs {
		if ev.Type == ir.EvImage && ev.Media != nil && ev.Media.MediaType == "image/png" && ev.Media.Data == "AQ==" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected EvImage(image/png, AQ==), got %+v", evs)
	}

	enc := gemini.NewStreamEncoder(gemini.EncodeOptions{Env: &ir.Env{}, AltSSE: true})
	var dst []byte
	for _, ev := range evs {
		dst, err = enc.Encode(dst, ev)
		if err != nil {
			t.Fatalf("Encode: %v", err)
		}
	}
	dst, _ = enc.Close(dst)
	if !strings.Contains(string(dst), `"inlineData"`) || !strings.Contains(string(dst), `"AQ=="`) {
		t.Errorf("Gemini StreamEncoder dropped EvImage:\n%s", string(dst))
	}
}

// Review2 #20: Multimodal tool results (images/files) in Responses parse/build and Gemini build.
func TestReview2_P2_20_MultimodalToolResults(t *testing.T) {
	t.Run("responses_parse_and_build_array_output", func(t *testing.T) {
		body := []byte(`{
			"model":"gpt-5",
			"input":[
				{"type":"function_call","call_id":"c1","name":"screenshot","arguments":"{}"},
				{"type":"function_call_output","call_id":"c1","output":[
					{"type":"input_text","text":"captured"},
					{"type":"input_image","image_url":"data:image/png;base64,AQ=="}
				]}
			]
		}`)
		req, err := responses.ParseRequest(body, responses.ParseOptions{})
		if err != nil {
			t.Fatalf("ParseRequest: %v", err)
		}
		res := req.Messages[1].Parts[0].Result
		if len(res.Content) != 2 || res.Content[0].Type != ir.PartText || res.Content[0].Text != "captured" || res.Content[1].Type != ir.PartImage {
			t.Fatalf("multimodal tool output lost in Responses ParseRequest: %+v", res.Content)
		}
		built, err := responses.BuildRequest(req, responses.BuildOptions{})
		if err != nil {
			t.Fatalf("BuildRequest: %v", err)
		}
		if !strings.Contains(string(built.Body), `"input_image"`) || !strings.Contains(string(built.Body), `AQ==`) {
			t.Errorf("multimodal tool output image dropped in Responses BuildRequest: %s", string(built.Body))
		}
	})

	t.Run("gemini_build_preserves_multimodal_tool_result", func(t *testing.T) {
		req := &ir.Request{
			Model: "gemini-2.5-pro",
			Messages: []ir.Message{
				{Role: ir.RoleAssistant, Parts: []ir.Part{{Type: ir.PartToolCall, Call: &ir.ToolCall{ID: "c1", Name: "shot", Arguments: json.RawMessage(`{}`)}}}},
				{Role: ir.RoleUser, Parts: []ir.Part{{Type: ir.PartToolResult, Result: &ir.ToolResult{
					CallID: "c1",
					Name:   "shot",
					Content: []ir.Part{
						ir.TextPart("ok"),
						{Type: ir.PartImage, Media: &ir.Media{MediaType: "image/png", Data: "AQ=="}},
					},
				}}}},
			},
		}
		built, err := gemini.BuildRequest(req, gemini.BuildOptions{})
		if err != nil {
			t.Fatalf("gemini BuildRequest: %v", err)
		}
		if !strings.Contains(string(built.Body), `"inlineData"`) || !strings.Contains(string(built.Body), `"AQ=="`) {
			t.Errorf("Gemini BuildRequest dropped image in tool result: %s", string(built.Body))
		}
	})
}

// Review2 #23: Gemini functionDeclarations[].parametersJsonSchema must be preserved (not silently dropped).
func TestReview2_P2_23_GeminiParametersJsonSchema(t *testing.T) {
	body := []byte(`{
		"contents":[{"role":"user","parts":[{"text":"hi"}]}],
		"tools":[{"functionDeclarations":[{
			"name":"search",
			"parametersJsonSchema":{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}
		}]}]
	}`)
	req, err := gemini.ParseRequest(body, gemini.ParseOptions{Model: "gemini-2.5-pro"})
	if err != nil {
		t.Fatalf("ParseRequest: %v", err)
	}
	if len(req.Tools) != 1 || !strings.Contains(string(req.Tools[0].Schema), `"required":["q"]`) {
		t.Fatalf("parametersJsonSchema dropped: %+v", req.Tools)
	}

	// Verify generationConfig.responseJsonSchema is handled by UnknownParam (Reject -> error).
	body2 := []byte(`{
		"contents":[{"role":"user","parts":[{"text":"hi"}]}],
		"generationConfig":{"responseJsonSchema":{"type":"object"}}
	}`)
	if _, err := gemini.ParseRequest(body2, gemini.ParseOptions{Env: &ir.Env{Unknown: ir.UnknownReject}}); err == nil {
		t.Errorf("expected UnknownReject error for generationConfig.responseJsonSchema")
	}
}

// Review2 #24: Media classification & data URI parsing across Gemini, Chat, Anthropic.
func TestReview2_P2_24_MediaClassificationAndDataURI(t *testing.T) {
	t.Run("gemini_image_fileData_to_chat_preserves_url", func(t *testing.T) {
		body := []byte(`{"contents":[{"role":"user","parts":[{"fileData":{"mimeType":"image/png","fileUri":"https://example.test/i.png"}}]}]}`)
		plan, err := bridge.Convert(ir.FormatGemini, ir.FormatChat, body, bridge.Options{Model: "gpt-5"})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		if !strings.Contains(string(plan.Body), "https://example.test/i.png") {
			t.Errorf("Gemini image fileData URL dropped when converting to Chat: %s", string(plan.Body))
		}
	})

	t.Run("chat_file_url_rejected_not_silently_dropped", func(t *testing.T) {
		body := []byte(`{"contents":[{"role":"user","parts":[{"fileData":{"mimeType":"application/pdf","fileUri":"https://example.test/doc.pdf"}}]}]}`)
		if _, err := bridge.Convert(ir.FormatGemini, ir.FormatChat, body, bridge.Options{Model: "gpt-5"}); err == nil {
			t.Errorf("expected unsupported error for PDF file URL toward Chat (Chat file part has no URL field)")
		}
	})

	t.Run("chat_pdf_data_uri_parsed_cleanly_for_anthropic", func(t *testing.T) {
		body := []byte(`{
			"model":"claude-sonnet-4",
			"messages":[{"role":"user","content":[
				{"type":"file","file":{"filename":"a.pdf","file_data":"data:application/pdf;base64,JVBERi0xLjQ="}}
			]}]
		}`)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic, body, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		s := string(plan.Body)
		if strings.Contains(s, "data:application/pdf;base64,") {
			t.Errorf("data URI prefix leaked into Anthropic base64 data: %s", s)
		}
		if strings.Contains(s, "image/jpeg") || !strings.Contains(s, "application/pdf") {
			t.Errorf("PDF media_type corrupted for Anthropic: %s", s)
		}
		if !strings.Contains(s, `"JVBERi0xLjQ="`) {
			t.Errorf("PDF base64 data missing: %s", s)
		}
	})
}

// Review2 #37: Gemini signature-only part decode/parse and foreign signature sealing in encode.
func TestReview2_P2_37_GeminiSignatureOnlyPartAndForeignSeal(t *testing.T) {
	dec := gemini.NewStreamDecoder(gemini.DecodeOptions{Env: &ir.Env{ProviderID: "google"}})
	chunk := `{"candidates":[{"index":0,"content":{"role":"model","parts":[{"thoughtSignature":"sig_only"}]},"finishReason":"STOP"}]}`
	evs, err := dec.Decode(nil, sse.Event{Data: []byte(chunk)})
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	foundSig := false
	for _, ev := range evs {
		if ev.Type == ir.EvSignature && ev.Signature != nil && ev.Signature.Value == "sig_only" {
			foundSig = true
		}
	}
	if !foundSig {
		t.Fatalf("signature-only Gemini part dropped in StreamDecoder: %+v", evs)
	}

	// Replay signature-only part in Gemini ParseRequest must not fail with unrecognised part.
	reqBody := []byte(`{"contents":[{"role":"model","parts":[{"thoughtSignature":"sig_only"}]}]}`)
	req, err := gemini.ParseRequest(reqBody, gemini.ParseOptions{Env: &ir.Env{ProviderID: "google"}})
	if err != nil {
		t.Fatalf("ParseRequest failed on signature-only part: %v", err)
	}
	if len(req.Messages) != 1 || len(req.Messages[0].Parts) != 1 || req.Messages[0].Parts[0].Reasoning == nil || req.Messages[0].Parts[0].Reasoning.Signature == nil || req.Messages[0].Parts[0].Reasoning.Signature.Value != "sig_only" {
		t.Fatalf("unexpected parsed signature-only part: %+v", req.Messages)
	}

	// Foreign signature encoded by Gemini StreamEncoder must be sealed, not emitted raw.
	seal := ir.NewSealCodec([]byte("secret-key"))
	enc := gemini.NewStreamEncoder(gemini.EncodeOptions{Env: &ir.Env{ProviderID: "google", Seal: seal}, AltSSE: true})
	var dst []byte
	dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvSignature, Signature: &ir.Signature{
		Value:  "raw_anthropic_sig",
		Origin: ir.Origin{Format: ir.FormatAnthropic, Provider: "anthropic"},
	}})
	dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStop, Stop: &ir.Stop{Reason: ir.StopEndTurn}})
	dst, _ = enc.Close(dst)
	if strings.Contains(string(dst), "raw_anthropic_sig") {
		t.Errorf("foreign signature leaked unsealed into Gemini stream:\n%s", string(dst))
	}
	if !strings.Contains(string(dst), "lpb1.") {
		t.Errorf("expected sealed foreign signature in Gemini stream:\n%s", string(dst))
	}
}

// Review2 #46: Chat assistant refusal preserved in history; Responses refusal delta keeps refusal semantics.
func TestReview2_P2_46_RefusalHistoryAndStream(t *testing.T) {
	t.Run("chat_history_preserves_assistant_refusal", func(t *testing.T) {
		body := []byte(`{"model":"gpt-5","messages":[{"role":"user","content":"bad"},{"role":"assistant","content":null,"refusal":"cannot assist"}]}`)
		req, err := chat.ParseRequest(body, chat.ParseOptions{})
		if err != nil {
			t.Fatalf("ParseRequest: %v", err)
		}
		if len(req.Messages) != 2 || len(req.Messages[1].Parts) != 1 || req.Messages[1].Parts[0].Type != ir.PartRefusal || req.Messages[1].Parts[0].Text != "cannot assist" {
			t.Fatalf("assistant refusal dropped in Chat ParseRequest: %+v", req.Messages[1].Parts)
		}
	})

	t.Run("responses_refusal_delta_preserves_refusal_semantics", func(t *testing.T) {
		dec := responses.NewStreamDecoder(responses.DecodeOptions{})
		evs, err := dec.Decode(nil, sse.Event{
			Name: "response.refusal.delta",
			Data: []byte(`{"type":"response.refusal.delta","sequence_number":1,"output_index":0,"content_index":0,"delta":"cannot comply"}`),
		})
		if err != nil {
			t.Fatalf("Decode: %v", err)
		}
		if len(evs) != 1 || evs[0].Type != ir.EvRefusal || evs[0].Text != "cannot comply" {
			t.Fatalf("expected EvRefusal(\"cannot comply\"), got %+v", evs)
		}
		// Encoding back to Responses SSE must emit response.refusal.delta, not output_text.delta.
		enc := responses.NewStreamEncoder(responses.EncodeOptions{Env: &ir.Env{}, Model: "gpt-5"})
		var dst []byte
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStart, Model: "gpt-5"})
		dst, _ = enc.Encode(dst, evs[0])
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStop, Stop: &ir.Stop{Reason: ir.StopRefusal}})
		dst, _ = enc.Close(dst)
		if !strings.Contains(string(dst), "response.refusal.delta") || strings.Contains(string(dst), "response.output_text.delta") {
			t.Errorf("expected response.refusal.delta in Responses StreamEncoder:\n%s", string(dst))
		}
	})
}

// Review2 #47: Citations (Responses annotations, Gemini groundingMetadata) & Anthropic server_tool_use
// enter IR and honor Deny[LossCitations] / Deny[LossBuiltinActivity].
func TestReview2_P2_47_CitationsAndHostedToolActivities(t *testing.T) {
	t.Run("responses_annotations_to_chat_honors_deny_citations", func(t *testing.T) {
		respBody := []byte(`{
			"id":"resp_1","object":"response","model":"gpt-5","status":"completed",
			"output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{
				"type":"output_text","text":"See docs",
				"annotations":[{"type":"url_citation","url":"https://example.test","title":"Docs","start_index":0,"end_index":8}]
			}]}]
		}`)
		irResp, err := responses.ParseResponse(respBody, responses.DecodeOptions{})
		if err != nil {
			t.Fatalf("ParseResponse: %v", err)
		}
		if len(irResp.Outputs[0].Parts) == 0 || len(irResp.Outputs[0].Parts[0].Citations) != 1 || irResp.Outputs[0].Parts[0].Citations[0].URL != "https://example.test" {
			t.Fatalf("annotations not parsed into Citations: %+v", irResp.Outputs[0].Parts)
		}
		// Chat BuildResponse cannot represent citations: must record LossCitations and honor Deny.
		if _, err := chat.BuildResponse(irResp, chat.EncodeOptions{Env: &ir.Env{Deny: ir.LossSet{ir.LossCitations: true}}}); err == nil {
			t.Errorf("chat.BuildResponse must fail when Deny[LossCitations]=true")
		}
	})

	t.Run("anthropic_server_tool_use_parsed_to_ir", func(t *testing.T) {
		respBody := []byte(`{
			"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4","stop_reason":"end_turn",
			"content":[
				{"type":"server_tool_use","id":"srv_1","name":"web_search","input":{"query":"golang"}},
				{"type":"web_search_tool_result","tool_use_id":"srv_1","content":[{"type":"web_search_result","url":"https://go.dev","title":"Go"}]},
				{"type":"text","text":"Go is great."}
			]
		}`)
		irResp, err := anthropic.ParseResponse(respBody, anthropic.DecodeOptions{})
		if err != nil {
			t.Fatalf("ParseResponse: %v", err)
		}
		if len(irResp.Outputs[0].Parts) < 3 || irResp.Outputs[0].Parts[0].Type != ir.PartBuiltinCall || irResp.Outputs[0].Parts[1].Type != ir.PartBuiltinResult {
			t.Fatalf("Anthropic server_tool_use / web_search_tool_result dropped: %+v", irResp.Outputs[0].Parts)
		}
	})

	t.Run("gemini_groundingMetadata_emits_citations_and_honors_deny", func(t *testing.T) {
		dec := gemini.NewStreamDecoder(gemini.DecodeOptions{})
		chunk := `{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"fact"}]},"groundingMetadata":{"webSearchQueries":["q"],"groundingChunks":[{"web":{"uri":"https://example.test/g","title":"G"}}]},"finishReason":"STOP"}]}`
		evs, err := dec.Decode(nil, sse.Event{Data: []byte(chunk)})
		if err != nil {
			t.Fatalf("Decode: %v", err)
		}
		hasCit, hasBuiltin := false, false
		for _, ev := range evs {
			if ev.Type == ir.EvCitation && ev.Citation != nil && ev.Citation.URL == "https://example.test/g" {
				hasCit = true
			}
			if ev.Type == ir.EvBuiltin && ev.Builtin != nil && ev.Builtin.Kind == ir.BuiltinWebSearch {
				hasBuiltin = true
			}
		}
		if !hasCit || !hasBuiltin {
			t.Fatalf("expected EvCitation and EvBuiltin from groundingMetadata, got %+v", evs)
		}
		enc := chat.NewStreamEncoder(chat.EncodeOptions{Env: &ir.Env{Deny: ir.LossSet{ir.LossCitations: true}}, Model: "m"})
		var gotErr error
		for _, ev := range evs {
			if ev.Type == ir.EvCitation {
				_, gotErr = enc.Encode(nil, ev)
			}
		}
		if gotErr == nil {
			t.Errorf("chat StreamEncoder must fail on EvCitation when Deny[LossCitations]=true")
		}
	})
}

// Review2 #48: Responses web_search tool config (external_web_access, filters, user_location)
// must be preserved at the top level on same-format conversion and Config builds.
func TestReview2_P2_48_ResponsesWebSearchConfigPreserved(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5",
		"input":"latest news",
		"tools":[{"type":"web_search","external_web_access":false,"user_location":{"type":"approximate","country":"US"},"filters":{"allowed_domains":["example.com"]}}]
	}`)
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatResponses, body, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	s := string(plan.Body)
	if !strings.Contains(s, `"external_web_access":false`) || !strings.Contains(s, `"allowed_domains":["example.com"]`) || !strings.Contains(s, `"country":"US"`) {
		t.Errorf("Responses web_search config dropped on identity conversion: %s", s)
	}
	if strings.Contains(s, `"web_search_options"`) {
		t.Errorf("invalid nested web_search_options emitted: %s", s)
	}

	// Direct Config build must also place fields at top level, not under web_search_options.
	req := &ir.Request{
		Model:    "gpt-5",
		Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("hi")}}},
		Builtins: []ir.BuiltinTool{{
			Kind:   ir.BuiltinWebSearch,
			Config: json.RawMessage(`{"external_web_access":false,"search_context_size":"high"}`),
		}},
	}
	built, err := responses.BuildRequest(req, responses.BuildOptions{})
	if err != nil {
		t.Fatalf("BuildRequest: %v", err)
	}
	if strings.Contains(string(built.Body), `"web_search_options"`) || !strings.Contains(string(built.Body), `"external_web_access":false`) {
		t.Errorf("Responses web_search Config written at wrong nesting level: %s", string(built.Body))
	}
}

// Review2 #52: Chat and Gemini partial usage snapshots must not overwrite previously reported
// input token counts with 0.
func TestReview2_P2_52_PartialUsageSnapshot(t *testing.T) {
	t.Run("chat_partial_usage_preserves_input", func(t *testing.T) {
		dec := chat.NewStreamDecoder(chat.DecodeOptions{})
		var u ir.Usage
		evs1, _ := dec.Decode(nil, sse.Event{Data: []byte(`{"id":"1","choices":[],"usage":{"prompt_tokens":100}}`)})
		for _, e := range evs1 {
			if e.Usage != nil {
				u.Merge(*e.Usage)
			}
		}
		evs2, _ := dec.Decode(nil, sse.Event{Data: []byte(`{"id":"1","choices":[],"usage":{"completion_tokens":5}}`)})
		for _, e := range evs2 {
			if e.Usage != nil {
				u.Merge(*e.Usage)
			}
		}
		if u.InputTokens != 100 || u.OutputTokens != 5 {
			t.Errorf("Chat partial usage overwrote input: want input=100 output=5, got %+v", u)
		}
	})

	t.Run("gemini_partial_usage_preserves_input", func(t *testing.T) {
		dec := gemini.NewStreamDecoder(gemini.DecodeOptions{})
		var u ir.Usage
		evs1, _ := dec.Decode(nil, sse.Event{Data: []byte(`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"hi"}]}}],"usageMetadata":{"promptTokenCount":120}}`)})
		for _, e := range evs1 {
			if e.Usage != nil {
				u.Merge(*e.Usage)
			}
		}
		evs2, _ := dec.Decode(nil, sse.Event{Data: []byte(`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"!"}]},"finishReason":"STOP"}],"usageMetadata":{"candidatesTokenCount":7}}`)})
		for _, e := range evs2 {
			if e.Usage != nil {
				u.Merge(*e.Usage)
			}
		}
		if u.InputTokens != 120 || u.OutputTokens != 7 {
			t.Errorf("Gemini partial usage overwrote input: want input=120 output=7, got %+v", u)
		}
	})
}
