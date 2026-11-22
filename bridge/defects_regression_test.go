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

// Defect 01: Streaming parallel tool calls from Anthropic -> Chat must receive
// distinct sequential tool_calls[].index values (0, 1) and never merge.
func TestDefect01_ParallelToolCallIndexCollision(t *testing.T) {
	plan, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic,
		[]byte(`{"model":"claude-4","messages":[{"role":"user","content":"go"}],"stream":true}`),
		bridge.Options{},
	)
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	anthSSE := "" +
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"model\":\"claude-4\",\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tu_1\",\"name\":\"Read\",\"input\":{}}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"p\\\":\\\"/a\\\"}\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tu_2\",\"name\":\"Write\",\"input\":{}}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"p\\\":\\\"/b\\\"}\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":12}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

	s := plan.NewStream()
	out := feedBridgeSSE(t, s, anthSSE)
	out, _ = s.End(out, ir.EndDone)

	dec := chat.NewStreamDecoder(chat.DecodeOptions{})
	col := ir.NewCollector()
	rr := sse.NewReader(bytes.NewReader(out))
	for {
		ev, err := rr.Next()
		if err != nil {
			break
		}
		evs, _ := dec.Decode(nil, ev)
		for _, e := range evs {
			col.Add(e)
		}
	}
	parts := col.Response().Outputs[0].Parts
	if len(parts) != 2 || parts[0].Call.ID != "tu_1" || parts[1].Call.ID != "tu_2" {
		t.Fatalf("parallel tool calls merged or lost: %+v", parts)
	}
}

// Defect 02: Non-streaming multi-text-block response (Anthropic -> Chat) must
// preserve all text blocks concatenated as one string, never only the last block.
func TestDefect02_MultiTextBlockNonStreamingPreserved(t *testing.T) {
	plan, _ := bridge.Convert(ir.FormatChat, ir.FormatAnthropic,
		[]byte(`{"model":"claude-4","messages":[{"role":"user","content":"q"}]}`),
		bridge.Options{},
	)
	anthResp := []byte(`{"id":"m1","type":"message","role":"assistant","model":"claude-4","content":[{"type":"text","text":"first "},{"type":"text","text":"second"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":4}}`)
	out, _, err := plan.ConvertResponse(anthResp)
	if err != nil {
		t.Fatalf("ConvertResponse: %v", err)
	}
	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if parsed.Choices[0].Message.Content != "first second" {
		t.Errorf("got %q, want %q", parsed.Choices[0].Message.Content, "first second")
	}
}

// Defect 03: Tool without parameters object (Chat -> Anthropic) must be kept
// with an empty object input_schema rather than silently dropped.
func TestDefect03_ZeroParamToolPreserved(t *testing.T) {
	plan, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic,
		[]byte(`{"model":"claude-4","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"GetTime","description":"Current time"}}]}`),
		bridge.Options{},
	)
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	if !strings.Contains(string(plan.Body), `"GetTime"`) || !strings.Contains(string(plan.Body), `"input_schema":{"type":"object","properties":{}}`) {
		t.Errorf("zero-param tool dropped or malformed: %s", plan.Body)
	}
}

// Defect 04: Assistant turn containing both text and tool_use (Anthropic -> Chat)
// must preserve both content and tool_calls.
func TestDefect04_TextAndToolCallCoexistInHistory(t *testing.T) {
	anthReq := []byte(`{
		"model": "gpt-4o",
		"max_tokens": 64,
		"messages": [
			{"role": "user", "content": "read file"},
			{"role": "assistant", "content": [
				{"type": "text", "text": "Let me read that."},
				{"type": "tool_use", "id": "tu_1", "name": "Read", "input": {"p": "/x"}}
			]},
			{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"}]}
		]
	}`)
	plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, anthReq, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	var parsed struct {
		Messages []struct {
			Role      string `json:"role"`
			Content   string `json:"content"`
			ToolCalls []any  `json:"tool_calls"`
		} `json:"messages"`
	}
	_ = json.Unmarshal(plan.Body, &parsed)
	if parsed.Messages[1].Content != "Let me read that." || len(parsed.Messages[1].ToolCalls) != 1 {
		t.Errorf("text+tool_use lost in Anthropic->Chat: %+v", parsed.Messages[1])
	}
}

// Defect 05: Usage cache tokens must never be double-counted across Anthropic/Chat/Gemini.
func TestDefect05_UsageCacheNoDoubleCounting(t *testing.T) {
	// Chat upstream: prompt_tokens=100 (includes cached_tokens=40) -> Anthropic client must see input_tokens=60, cache_read_input_tokens=40.
	pChat, _ := bridge.Convert(ir.FormatAnthropic, ir.FormatChat,
		[]byte(`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`), bridge.Options{})
	chatResp := []byte(`{"id":"1","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110,"prompt_tokens_details":{"cached_tokens":40}}}`)
	out, _, err := pChat.ConvertResponse(chatResp)
	if err != nil {
		t.Fatalf("ConvertResponse: %v", err)
	}
	var anthOut struct {
		Usage struct {
			InputTokens          int `json:"input_tokens"`
			CacheReadInputTokens int `json:"cache_read_input_tokens"`
		} `json:"usage"`
	}
	_ = json.Unmarshal(out, &anthOut)
	if anthOut.Usage.InputTokens != 60 || anthOut.Usage.CacheReadInputTokens != 40 {
		t.Errorf("Anthropic cache double-counted: %+v", anthOut.Usage)
	}
}

// Defect 06: Non-streaming Gemini response with multiple thought parts must
// preserve all reasoning text when converted to Chat.
func TestDefect06_MultiReasoningPartsNonStreamingPreserved(t *testing.T) {
	plan, _ := bridge.Convert(ir.FormatChat, ir.FormatGemini,
		[]byte(`{"model":"gemini-2.5","messages":[{"role":"user","content":"q"}]}`), bridge.Options{})
	gemResp := []byte(`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"step 1; ","thought":true},{"text":"step 2","thought":true},{"text":"answer"}]},"finishReason":"STOP"}]}`)
	out, _, err := plan.ConvertResponse(gemResp)
	if err != nil {
		t.Fatalf("ConvertResponse: %v", err)
	}
	var parsed struct {
		Choices []struct {
			Message struct {
				ReasoningContent string `json:"reasoning_content"`
				Content          string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	_ = json.Unmarshal(out, &parsed)
	if parsed.Choices[0].Message.ReasoningContent != "step 1; step 2" {
		t.Errorf("reasoning truncated: got %q, want %q", parsed.Choices[0].Message.ReasoningContent, "step 1; step 2")
	}
}

// Defect 07: Chat n must map to Gemini generationConfig.candidateCount.
func TestDefect07_CandidatesMappedToCandidateCount(t *testing.T) {
	plan, err := bridge.Convert(ir.FormatChat, ir.FormatGemini,
		[]byte(`{"model":"gemini-2.5","n":3,"messages":[{"role":"user","content":"hi"}]}`), bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	if !strings.Contains(string(plan.Body), `"candidateCount":3`) {
		t.Errorf("candidateCount missing: %s", plan.Body)
	}
}

// Defect 08: Tool strict flag must be preserved in Chat and Responses.
func TestDefect08_ToolStrictPreservedInChatAndResponses(t *testing.T) {
	chatReq := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"F","strict":true,"parameters":{"type":"object","additionalProperties":false}}}]}`)
	pChat, err := bridge.Convert(ir.FormatChat, ir.FormatChat, chatReq, bridge.Options{})
	if err != nil {
		t.Fatalf("Chat->Chat: %v", err)
	}
	if !strings.Contains(string(pChat.Body), `"strict":true`) || !strings.Contains(string(pChat.Body), `"additionalProperties":false`) {
		t.Errorf("strict or additionalProperties lost in Chat: %s", pChat.Body)
	}
	pResp, err := bridge.Convert(ir.FormatChat, ir.FormatResponses, chatReq, bridge.Options{})
	if err != nil {
		t.Fatalf("Chat->Responses: %v", err)
	}
	if !strings.Contains(string(pResp.Body), `"strict":true`) {
		t.Errorf("strict lost in Responses: %s", pResp.Body)
	}
}

// Defect 09: Anthropic URL image must stay a real URL (never wrapped into an invalid data: URI).
func TestDefect09_URLImageNotWrappedAsInvalidDataURL(t *testing.T) {
	anthReq := []byte(`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"https://example.com/cat.png"}}]}]}`)
	pChat, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, anthReq, bridge.Options{})
	if err != nil {
		t.Fatalf("Anthropic->Chat: %v", err)
	}
	if strings.Contains(string(pChat.Body), "data:") || !strings.Contains(string(pChat.Body), "https://example.com/cat.png") {
		t.Errorf("URL image corrupted in Chat: %s", pChat.Body)
	}
	pGem, err := bridge.Convert(ir.FormatAnthropic, ir.FormatGemini, anthReq, bridge.Options{})
	if err != nil {
		t.Fatalf("Anthropic->Gemini: %v", err)
	}
	if !strings.Contains(string(pGem.Body), `"fileUri":"https://example.com/cat.png"`) {
		t.Errorf("URL image corrupted in Gemini: %s", pGem.Body)
	}
}

// Defect 10: Gemini upstream returning 0 candidates must surface an error rather than empty success.
func TestDefect10_GeminiZeroCandidatesReturnsError(t *testing.T) {
	plan, _ := bridge.Convert(ir.FormatAnthropic, ir.FormatGemini,
		[]byte(`{"model":"gemini-2.5","max_tokens":64,"messages":[{"role":"user","content":"blocked"}]}`), bridge.Options{})
	blocked := []byte(`{"promptFeedback":{"blockReason":"SAFETY"}}`)
	if _, _, err := plan.ConvertResponse(blocked); err == nil {
		t.Errorf("expected error for 0-candidate Gemini response")
	}
}

// Defect 11: Anthropic streaming server_tool_use web_search_requests must be
// captured in Stream.Usage() for billing.
func TestDefect11_StreamingWebSearchUsageTracked(t *testing.T) {
	plan, _ := bridge.Convert(ir.FormatChat, ir.FormatAnthropic,
		[]byte(`{"model":"claude-4","messages":[{"role":"user","content":"news"}],"stream":true}`), bridge.Options{})
	s := plan.NewStream()
	anthSSE := "" +
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"model\":\"claude-4\",\"usage\":{\"input_tokens\":50,\"output_tokens\":0}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"latest\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":10,\"server_tool_use\":{\"web_search_requests\":3}}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	out := feedBridgeSSE(t, s, anthSSE)
	_, _ = s.End(out, ir.EndDone)
	if u := s.Usage(); u.WebSearchRequests != 3 || !u.Has(ir.UWebSearch) {
		t.Errorf("streaming web_search_requests lost: %+v", u)
	}
}

// Defect 12: Gemini response containing both text and functionCall must preserve both in Chat.
func TestDefect12_GeminiTextAndFunctionCallBothPreserved(t *testing.T) {
	plan, _ := bridge.Convert(ir.FormatChat, ir.FormatGemini,
		[]byte(`{"model":"gemini-2.5","messages":[{"role":"user","content":"go"}]}`), bridge.Options{})
	gemResp := []byte(`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"calling tool"},{"functionCall":{"id":"c1","name":"Do","args":{"x":1}}}]},"finishReason":"STOP"}]}`)
	out, _, err := plan.ConvertResponse(gemResp)
	if err != nil {
		t.Fatalf("ConvertResponse: %v", err)
	}
	if !strings.Contains(string(out), `"calling tool"`) || !strings.Contains(string(out), `"Do"`) {
		t.Errorf("text or functionCall lost: %s", out)
	}
}

// Defect 13: Responses response containing both text and function_call must preserve both in Chat.
func TestDefect13_ResponsesTextAndToolCallBothPreserved(t *testing.T) {
	plan, _ := bridge.Convert(ir.FormatChat, ir.FormatResponses,
		[]byte(`{"model":"gpt-5","messages":[{"role":"user","content":"go"}]}`), bridge.Options{})
	respBody := []byte(`{"id":"r1","object":"response","model":"gpt-5","status":"completed","output":[{"type":"message","id":"m1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"working"}]},{"type":"function_call","id":"fc1","call_id":"c1","name":"Do","arguments":"{\"x\":1}","status":"completed"}]}`)
	out, _, err := plan.ConvertResponse(respBody)
	if err != nil {
		t.Fatalf("ConvertResponse: %v", err)
	}
	if !strings.Contains(string(out), `"working"`) || !strings.Contains(string(out), `"Do"`) {
		t.Errorf("text or function_call lost: %s", out)
	}
}

// Defect 14: Gemini identity path with FunctionCallIDs dialect preserves functionCall.id and functionResponse.id.
func TestDefect14_GeminiNativePreservesFunctionCallID(t *testing.T) {
	gemReq := []byte(`{"contents":[{"role":"user","parts":[{"text":"go"}]},{"role":"model","parts":[{"functionCall":{"id":"fc_99","name":"Do","args":{"a":1}}}]},{"role":"user","parts":[{"functionResponse":{"id":"fc_99","name":"Do","response":{"output":"ok"}}}]}]}`)
	plan, err := bridge.Convert(ir.FormatGemini, ir.FormatGemini, gemReq, bridge.Options{
		Model:  "gemini-2.5",
		Gemini: gemini.Dialect{FunctionCallIDs: true},
	})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	if strings.Count(string(plan.Body), `"id":"fc_99"`) != 2 {
		t.Errorf("expected both functionCall.id and functionResponse.id preserved: %s", plan.Body)
	}
}

// Defect 15: Gemini clients must receive Gemini error envelopes (code/message/status), never OpenAI error format.
func TestDefect15_GeminiClientReceivesGeminiErrorFormat(t *testing.T) {
	plan, _ := bridge.Convert(ir.FormatGemini, ir.FormatChat,
		[]byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`), bridge.Options{Model: "gemini-2.5"})
	status, body := plan.ConvertError(400, []byte(`{"error":{"type":"invalid_request_error","message":"bad input"}}`))
	if status != 400 {
		t.Errorf("status: got %d, want 400", status)
	}
	var ge struct {
		Error struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
			Status  string `json:"status"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &ge); err != nil {
		t.Fatalf("unmarshal: %v (%s)", err, body)
	}
	if ge.Error.Status != "INVALID_ARGUMENT" || ge.Error.Code != 400 || ge.Error.Message != "bad input" {
		t.Errorf("unexpected Gemini error envelope: %+v (%s)", ge.Error, body)
	}
}

// Defect 16 (library scope): Unknown passthrough fields are only emitted on
// same-format identity builds and never leaked across formats. Proxy-level raw
// body-passthrough guards are enforced in the caller's routing layer.
func TestDefect16_CrossFormatPassthroughIsolated(t *testing.T) {
	chatReq := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"vendor_extra":true}`)
	pSame, err := bridge.Convert(ir.FormatChat, ir.FormatChat, chatReq, bridge.Options{})
	if err != nil {
		t.Fatalf("identity: %v", err)
	}
	if !strings.Contains(string(pSame.Body), `"vendor_extra":true`) {
		t.Errorf("identity build lost passthrough field: %s", pSame.Body)
	}
	pCross, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic, chatReq, bridge.Options{})
	if err != nil {
		t.Fatalf("cross-format: %v", err)
	}
	if strings.Contains(string(pCross.Body), "vendor_extra") {
		t.Errorf("cross-format build leaked unknown field: %s", pCross.Body)
	}
}

// Defect 17 (library scope): System instructions prepended or appended on IR
// are preserved when building for Anthropic and Gemini (and all other targets).
func TestDefect17_SystemInstructionsPreservedAcrossDirections(t *testing.T) {
	req, err := bridge.Parse(ir.FormatChat, []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`), bridge.Options{})
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	req.PrependInstructions("policy prompt")
	for _, target := range []ir.Format{ir.FormatAnthropic, ir.FormatGemini, ir.FormatResponses, ir.FormatChat} {
		plan, err := bridge.Build(req, ir.FormatChat, target, bridge.Options{})
		if err != nil {
			t.Fatalf("Build(%s): %v", target, err)
		}
		if !strings.Contains(string(plan.Body), "policy prompt") {
			t.Errorf("%s dropped prepended system instructions: %s", target, plan.Body)
		}
	}
}

// Strict protocol conversion regressions: strict validation rules for Anthropic
// and Responses request/response conversions.
func TestStrictProtocolRegressions(t *testing.T) {
	t.Run("anthropic_requires_max_tokens_and_temperature_bounds", func(t *testing.T) {
		if _, err := anthropic.ParseRequest([]byte(`{"model":"claude","messages":[{"role":"user","content":"hi"}]}`), anthropic.ParseOptions{}); err == nil {
			t.Errorf("expected error when Anthropic max_tokens is missing")
		}
		if _, err := anthropic.ParseRequest([]byte(`{"model":"claude","max_tokens":64,"temperature":1.5,"messages":[{"role":"user","content":"hi"}]}`), anthropic.ParseOptions{}); err == nil {
			t.Errorf("expected error when Anthropic temperature > 1.0")
		}
	})

	t.Run("responses_rejects_stateful_and_unknown_items", func(t *testing.T) {
		for _, body := range []string{
			`{"model":"gpt-5","conversation":"c1","input":"hi"}`,
			`{"model":"gpt-5","background":true,"input":"hi"}`,
			`{"model":"gpt-5","input":[{"type":"item_reference","id":"i1"}]}`,
		} {
			if _, err := responses.ParseRequest([]byte(body), responses.ParseOptions{}); err == nil {
				t.Errorf("expected error for %s", body)
			}
		}
	})
}
