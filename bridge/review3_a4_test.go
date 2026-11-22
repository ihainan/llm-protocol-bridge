package bridge_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// A4 P1 #1: Non-streaming Chat tool_call with arguments:"null" must be rejected
// by ConvertResponse instead of being cleared to "{}".
func TestReview3_A4_01_NonStreamChatNullArgumentsRejected(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","input":"run","tools":[{"type":"function","name":"exec","parameters":{"type":"object"}}]}`)
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, reqBody, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	upBody := []byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-5","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"exec","arguments":"null"}}]},"finish_reason":"tool_calls"}]}`)
	out, _, err := plan.ConvertResponse(upBody)
	if err == nil {
		t.Fatalf("expected error for arguments:\"null\", got output: %s", string(out))
	}
}

// A4 P1 #2: Responses incomplete function_call item must not be resurrected as
// a completed call when decoded (non-streaming or streaming) and re-encoded.
func TestReview3_A4_02_ResponsesIncompleteToolItemNotResurrected(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","input":"run","tools":[{"type":"function","name":"exec","parameters":{"type":"object"}}]}`)
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatResponses, reqBody, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}

	t.Run("nonstream", func(t *testing.T) {
		upBody := []byte(`{"id":"resp_1","object":"response","model":"gpt-5","status":"completed","output":[{"type":"function_call","id":"fc_1","status":"incomplete","call_id":"c1","name":"exec","arguments":"{}"}]}`)
		out, irResp, err := plan.ConvertResponse(upBody)
		if err != nil {
			t.Fatalf("ConvertResponse: %v", err)
		}
		if len(irResp.Outputs[0].Parts) != 0 || strings.Contains(string(out), `"status":"completed","call_id":"c1"`) || strings.Contains(string(out), `"type":"function_call"`) {
			t.Fatalf("incomplete function_call was resurrected: parts=%+v out=%s", irResp.Outputs[0].Parts, string(out))
		}
	})

	t.Run("stream", func(t *testing.T) {
		sPlan, err := bridge.Convert(ir.FormatResponses, ir.FormatResponses, []byte(`{"model":"gpt-5","stream":true,"input":"run","tools":[{"type":"function","name":"exec","parameters":{"type":"object"}}]}`), bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		raw := "event: response.created\n" +
			`data: {"type":"response.created","sequence_number":0,"response":{"id":"r1","model":"gpt-5","status":"in_progress"}}` + "\n\n" +
			"event: response.output_item.added\n" +
			`data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"c1","name":"exec","arguments":"","status":"in_progress"}}` + "\n\n" +
			"event: response.output_item.done\n" +
			`data: {"type":"response.output_item.done","sequence_number":2,"output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"c1","name":"exec","arguments":"{}","status":"incomplete"}}` + "\n\n" +
			"event: response.completed\n" +
			`data: {"type":"response.completed","sequence_number":3,"response":{"id":"r1","model":"gpt-5","status":"completed","output":[{"type":"function_call","id":"fc_1","call_id":"c1","name":"exec","arguments":"{}","status":"incomplete"}]}}` + "\n\n"
		out := string(feedSSE(t, sPlan.NewStream(), raw))
		if strings.Contains(out, "response.function_call_arguments.done") {
			t.Fatalf("incomplete streamed function_call emitted arguments.done:\n%s", out)
		}
		// Verify no function_call in output_item.done has status:"completed".
		rd := sse.NewReader(strings.NewReader(out))
		for {
			ev, rerr := rd.Next()
			if rerr != nil {
				break
			}
			var m map[string]any
			if json.Unmarshal(ev.Data, &m) == nil && m["type"] == "response.output_item.done" {
				if it, ok := m["item"].(map[string]any); ok && it["type"] == "function_call" && it["status"] == "completed" {
					t.Fatalf("incomplete streamed function_call became completed:\n%s", out)
				}
			}
		}
	})
}

// A4 P2 #3: StreamResponse must not re-run StopEmulate across a removed tool boundary.
func TestReview3_A4_03_StreamResponseNoDoubleStopMatch(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","max_tokens":256,"stop_sequences":["END"],"messages":[{"role":"user","content":"go"}],"tools":[{"name":"exec","input_schema":{"type":"object"}}]}`)
	plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatResponses, reqBody, bridge.Options{
		StopMode: ir.StopEmulate,
	})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	upBody := []byte(`{"id":"resp_1","object":"response","model":"gpt-5","status":"completed","output":[{"type":"message","id":"m1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"E"}]},{"type":"function_call","id":"fc1","call_id":"c1","name":"exec","arguments":"{}","status":"completed"},{"type":"message","id":"m2","role":"assistant","status":"completed","content":[{"type":"output_text","text":"NDxEND"}]}]}`)
	out, err := plan.StreamResponse(nil, upBody)
	if err != nil {
		t.Fatalf("StreamResponse: %v", err)
	}
	var got strings.Builder
	rd := sse.NewReader(strings.NewReader(string(out)))
	for {
		ev, rerr := rd.Next()
		if rerr != nil {
			break
		}
		var m map[string]any
		if json.Unmarshal(ev.Data, &m) == nil && m["type"] == "content_block_delta" {
			if d, ok := m["delta"].(map[string]any); ok && d["type"] == "text_delta" {
				got.WriteString(d["text"].(string))
			}
		}
	}
	if got.String() != "ENDx" {
		t.Fatalf("StreamResponse double-matched stop across removed tool boundary: got %q, want \"ENDx\"\nraw=%s", got.String(), string(out))
	}
}

// A4 P2 #4: Anthropic Stream.Done() must remain false until message_stop arrives,
// so callers that close upstream upon Done()==true do not get a false EndEOF error.
func TestReview3_A4_04_AnthropicDoneWaitsForMessageStop(t *testing.T) {
	reqBody := []byte(`{"model":"claude-sonnet-4","stream":true,"input":"hi"}`)
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, reqBody, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	s := plan.NewStream()
	events := []sse.Event{
		{Name: "message_start", Data: []byte(`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}`)},
		{Name: "content_block_start", Data: []byte(`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)},
		{Name: "content_block_delta", Data: []byte(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`)},
		{Name: "content_block_stop", Data: []byte(`{"type":"content_block_stop","index":0}`)},
		{Name: "message_delta", Data: []byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}`)},
	}
	var out []byte
	for _, ev := range events {
		out, err = s.Feed(out, ev)
		if err != nil {
			t.Fatalf("Feed: %v", err)
		}
	}
	if s.Done() {
		t.Fatalf("Stream.Done() turned true after message_delta before message_stop arrived")
	}
	out, err = s.Feed(out, sse.Event{Name: "message_stop", Data: []byte(`{"type":"message_stop"}`)})
	if err != nil {
		t.Fatalf("Feed message_stop: %v", err)
	}
	if !s.Done() {
		t.Fatalf("Stream.Done() should be true after message_stop")
	}
	out, err = s.End(out, ir.EndEOF)
	if err != nil {
		t.Fatalf("End: %v", err)
	}
	if !strings.Contains(string(out), "event: response.completed") {
		t.Fatalf("expected response.completed after message_stop:\n%s", string(out))
	}
}

// A4 P2 #5: Responses function_call output (streaming and non-streaming) must
// preserve tool Namespace when restored from renamed upstream calls.
func TestReview3_A4_05_ResponsesFunctionCallPreservesNamespace(t *testing.T) {
	reqBody := []byte(`{
		"model":"gpt-5",
		"input":"read both",
		"tools":[
			{"type":"namespace","name":"a","tools":[{"type":"function","name":"read","parameters":{"type":"object"}}]},
			{"type":"namespace","name":"b","tools":[{"type":"function","name":"read","parameters":{"type":"object"}}]}
		]
	}`)
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, reqBody, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	upBody := []byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-5","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"a__read","arguments":"{}"}},{"id":"c2","type":"function","function":{"name":"b__read","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`)
	out, _, err := plan.ConvertResponse(upBody)
	if err != nil {
		t.Fatalf("ConvertResponse: %v", err)
	}
	if !strings.Contains(string(out), `"namespace":"a"`) || !strings.Contains(string(out), `"namespace":"b"`) {
		t.Fatalf("Responses non-streaming function_call dropped namespace:\n%s", string(out))
	}

	// Streaming path
	streamReq := []byte(`{
		"model":"gpt-5",
		"stream":true,
		"input":"read both",
		"tools":[
			{"type":"namespace","name":"a","tools":[{"type":"function","name":"read","parameters":{"type":"object"}}]},
			{"type":"namespace","name":"b","tools":[{"type":"function","name":"read","parameters":{"type":"object"}}]}
		]
	}`)
	sPlan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, streamReq, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert stream: %v", err)
	}
	raw := "data: " + `{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"a__read","arguments":"{}"}},{"index":1,"id":"c2","type":"function","function":{"name":"b__read","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}` + "\n\n" + "data: [DONE]\n\n"
	sOut := string(feedSSE(t, sPlan.NewStream(), raw))
	if !strings.Contains(sOut, `"namespace":"a"`) || !strings.Contains(sOut, `"namespace":"b"`) {
		t.Fatalf("Responses streaming function_call dropped namespace:\n%s", sOut)
	}
}

// A4 P2 #6: Converting Anthropic web_search_tool_result to Responses must record
// LossBuiltinActivity (blockable via Deny) since Responses web_search_call cannot
// carry result hits/encrypted_content/page_age.
func TestReview3_A4_06_AnthropicSearchResultToResponsesRecordsLoss(t *testing.T) {
	reqBody := []byte(`{"model":"claude-sonnet-4","input":"search"}`)
	upBody := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4","content":[{"type":"web_search_tool_result","tool_use_id":"srv_1","content":[{"type":"web_search_result","url":"https://example.com","title":"Ex","encrypted_content":"enc_1","page_age":"1d"}]}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`)

	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, reqBody, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	_, irResp, err := plan.ConvertResponse(upBody)
	if err != nil {
		t.Fatalf("ConvertResponse: %v", err)
	}
	found := false
	for _, l := range irResp.Losses {
		if l.Kind == ir.LossBuiltinActivity {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected LossBuiltinActivity when converting Anthropic search result to Responses, got %v", irResp.Losses)
	}

	denyPlan, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, reqBody, bridge.Options{
		Deny: ir.LossSet{ir.LossBuiltinActivity: true},
	})
	if err != nil {
		t.Fatalf("Convert deny: %v", err)
	}
	if _, _, err := denyPlan.ConvertResponse(upBody); err == nil {
		t.Fatalf("expected Deny[LossBuiltinActivity] to fail on Anthropic search result -> Responses")
	}
}

// A4 P2 #7: Schema downgrade from oneOf -> anyOf and $ref sibling constraints
// must record ir.LossToolSchema (blockable via Deny).
func TestReview3_A4_07_SchemaOneOfAndRefSiblingDowngradeRecordsLoss(t *testing.T) {
	oneOfReq := []byte(`{
		"model":"gemini-2.5-flash",
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"x":{"oneOf":[{"type":"number"},{"type":"integer"}]}}}}}]
	}`)
	plan, err := bridge.Convert(ir.FormatChat, ir.FormatGemini, oneOfReq, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert oneOf: %v", err)
	}
	found := false
	for _, l := range plan.Losses {
		if l.Kind == ir.LossToolSchema {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected LossToolSchema for oneOf -> anyOf downgrade, got %v", plan.Losses)
	}
	if _, err := bridge.Convert(ir.FormatChat, ir.FormatGemini, oneOfReq, bridge.Options{
		Deny: ir.LossSet{ir.LossToolSchema: true},
	}); err == nil {
		t.Fatalf("expected Deny[LossToolSchema] to reject oneOf downgrade")
	}

	refSiblingReq := []byte(`{
		"model":"gemini-2.5-flash",
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","$defs":{"S":{"type":"object","properties":{"a":{"type":"string"}}}},"properties":{"x":{"$ref":"#/$defs/S","additionalProperties":false}}}}}]
	}`)
	refPlan, err := bridge.Convert(ir.FormatChat, ir.FormatGemini, refSiblingReq, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert $ref sibling: %v", err)
	}
	foundRef := false
	for _, l := range refPlan.Losses {
		if l.Kind == ir.LossToolSchema {
			foundRef = true
		}
	}
	if !foundRef {
		t.Fatalf("expected LossToolSchema for dropped $ref sibling constraint, got %v", refPlan.Losses)
	}
}

// A4 P2 #8: StreamResponseWithResult exposes response-phase losses without OnLoss.
func TestReview3_A4_08_StreamResponseWithResultExposesLosses(t *testing.T) {
	reqBody := []byte(`{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"draw"}]}`)
	plan, err := bridge.Convert(ir.FormatChat, ir.FormatGemini, reqBody, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	upBody := []byte(`{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"image/png","data":"AQ=="}},{"text":"done"}]},"finishReason":"STOP"}],"modelVersion":"gemini-2.5-flash"}`)
	_, irResp, err := plan.StreamResponseWithResult(nil, upBody)
	if err != nil {
		t.Fatalf("StreamResponseWithResult: %v", err)
	}
	found := false
	for _, l := range irResp.Losses {
		if l.Kind == ir.LossOutputMedia {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected LossOutputMedia from StreamResponseWithResult, got %v", irResp.Losses)
	}
}

// A4 P2 #9: Anthropic server_tool_use initial content_block_start.input must
// count toward MaxBufferedBytes.
func TestReview3_A4_09_AnthropicServerToolInitialInputCapped(t *testing.T) {
	reqBody := []byte(`{"model":"claude-sonnet-4","stream":true,"max_tokens":256,"messages":[{"role":"user","content":"search"}]}`)
	plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatAnthropic, reqBody, bridge.Options{
		Limits: ir.Limits{MaxBufferedBytes: 1024},
	})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	big := strings.Repeat("a", 2000)
	raw := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srv_1","name":"web_search","input":{"query":"` + big + `"}}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n"
	out := string(feedSSE(t, plan.NewStream(), raw))
	if !strings.Contains(out, "event: error") {
		t.Fatalf("expected buffer limit error for oversized server_tool_use initial input:\n%s", out)
	}
}

// A4 P2 #10: Non-streaming ConvertResponse enforces response-wide MaxToolCalls
// across all candidates and enforces MaxBufferedBytes on tool arguments.
func TestReview3_A4_10_NonStreamResponseWideLimits(t *testing.T) {
	t.Run("max_tool_calls_across_candidates", func(t *testing.T) {
		reqBody := []byte(`{"model":"gemini-2.5-flash","n":2,"messages":[{"role":"user","content":"go"}],"tools":[{"type":"function","function":{"name":"exec","parameters":{"type":"object"}}}]}`)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatGemini, reqBody, bridge.Options{
			Limits: ir.Limits{MaxToolCalls: 1},
		})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		upBody := []byte(`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"functionCall":{"name":"exec","args":{}}}]},"finishReason":"STOP"},{"index":1,"content":{"role":"model","parts":[{"functionCall":{"name":"exec","args":{}}}]},"finishReason":"STOP"}]}`)
		if _, _, err := plan.ConvertResponse(upBody); err == nil {
			t.Fatalf("expected MaxToolCalls=1 to reject 2 tool calls across 2 candidates")
		}
	})

	t.Run("max_buffered_bytes_nonstream", func(t *testing.T) {
		reqBody := []byte(`{"model":"gpt-5","input":"go","tools":[{"type":"function","name":"exec","parameters":{"type":"object"}}]}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, reqBody, bridge.Options{
			Limits: ir.Limits{MaxBufferedBytes: 32},
		})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		upBody := []byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-5","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"exec","arguments":"{\"data\":\"` + strings.Repeat("x", 64) + `\"}"}}]},"finish_reason":"tool_calls"}]}`)
		if _, _, err := plan.ConvertResponse(upBody); err == nil {
			t.Fatalf("expected MaxBufferedBytes=32 to reject 64-byte tool arguments in ConvertResponse")
		}
	})
}

// A4 P2 #11: StopEmulate truncation must not drop Anthropic source-document
// citations (char_location start_char_index:100) by comparing them against output
// byte offsets, and must use rune offsets for output-indexed annotations.
func TestReview3_A4_11_StopEmulatePreservesSourceCitationsAndRuneOffsets(t *testing.T) {
	reqBody := []byte(`{"model":"claude-sonnet-4","max_tokens":256,"stop_sequences":["END"],"messages":[{"role":"user","content":"go"}]}`)
	plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatAnthropic, reqBody, bridge.Options{
		StopMode: ir.StopEmulate,
	})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	upBody := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4","content":[{"type":"text","text":"prefixEND","citations":[{"type":"char_location","cited_text":"prefix","start_char_index":100,"end_char_index":106}]}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`)
	_, irResp, err := plan.ConvertResponse(upBody)
	if err != nil {
		t.Fatalf("ConvertResponse: %v", err)
	}
	if len(irResp.Outputs[0].Parts) != 1 || irResp.Outputs[0].Parts[0].Text != "prefix" {
		t.Fatalf("unexpected parts: %+v", irResp.Outputs[0].Parts)
	}
	if len(irResp.Outputs[0].Parts[0].Citations) != 1 || irResp.Outputs[0].Parts[0].Citations[0].StartIndex != 100 {
		t.Fatalf("source-document citation was dropped or corrupted by stop truncation: %+v", irResp.Outputs[0].Parts[0].Citations)
	}
}

// A4 P2 #12: Gemini web_search build must reject unsupported max_uses limits.
func TestReview3_A4_12_GeminiRejectsWebSearchMaxUses(t *testing.T) {
	reqBody := []byte(`{
		"model":"gemini-2.5-flash",
		"max_tokens":256,
		"messages":[{"role":"user","content":"news"}],
		"tools":[{"type":"web_search_20250305","name":"web_search","max_uses":1}]
	}`)
	if _, err := bridge.Convert(ir.FormatAnthropic, ir.FormatGemini, reqBody, bridge.Options{}); err == nil {
		t.Fatalf("expected Gemini to reject web_search with max_uses limit")
	}
}
