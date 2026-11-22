package bridge_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/chat"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// Review3 P1-1: Cancelled EvToolEnd (Err != nil) must NEVER be emitted as a completed
// tool call by Gemini, Anthropic, Chat, or Responses encoders.
func TestReview3_P1_1_CancelledToolEndNotCompleted(t *testing.T) {
	rawChat := "data: " + `{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"exec","arguments":"{\"cmd\":"}}]},"finish_reason":null}]}` + "\n\n" +
		"data: " + `{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{"content":"END"},"finish_reason":"stop"}]}` + "\n\n" +
		"data: [DONE]\n\n"

	t.Run("gemini_discards_cancelled_functionCall", func(t *testing.T) {
		reqBody := []byte(`{"contents":[{"role":"user","parts":[{"text":"run"}]}],"tools":[{"functionDeclarations":[{"name":"exec","parameters":{"type":"object"}}]}],"generationConfig":{"stopSequences":["END"]}}`)
		plan, err := bridge.Convert(ir.FormatGemini, ir.FormatChat, reqBody, bridge.Options{
			Model:       "gpt-5",
			Stream:      true,
			AltSSE:      true,
			ToolRelease: ir.ToolReleaseLiveFirst,
			StopMode:    ir.StopEmulate,
		})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		out := string(feedSSE(t, plan.NewStream(), rawChat))
		if strings.Contains(out, `"functionCall"`) {
			t.Fatalf("Gemini encoder emitted functionCall on cancelled EvToolEnd:\n%s", out)
		}
	})

	t.Run("anthropic_aborts_open_tool_use_without_block_stop", func(t *testing.T) {
		reqBody := []byte(`{"model":"gpt-5","stream":true,"max_tokens":256,"stop_sequences":["END"],"messages":[{"role":"user","content":"run"}],"tools":[{"name":"exec","input_schema":{"type":"object"}}]}`)
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, reqBody, bridge.Options{
			ToolRelease: ir.ToolReleaseLiveFirst,
			StopMode:    ir.StopEmulate,
		})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		out := string(feedSSE(t, plan.NewStream(), rawChat))
		if strings.Contains(out, `"type":"content_block_stop"`) {
			t.Fatalf("Anthropic encoder emitted content_block_stop for cancelled tool_use:\n%s", out)
		}
		if !strings.Contains(out, "event: error") {
			t.Fatalf("Anthropic encoder must abort open cancelled tool_use with event: error:\n%s", out)
		}
	})

	t.Run("chat_aborts_open_tool_call_without_done_or_empty_args", func(t *testing.T) {
		reqBody := []byte(`{"model":"gpt-5","stream":true,"stop":["END"],"messages":[{"role":"user","content":"run"}],"tools":[{"type":"function","function":{"name":"exec","parameters":{"type":"object"}}}]}`)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatChat, reqBody, bridge.Options{
			ToolRelease: ir.ToolReleaseLiveFirst,
			StopMode:    ir.StopEmulate,
		})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		out := string(feedSSE(t, plan.NewStream(), rawChat))
		if strings.Contains(out, "data: [DONE]") {
			t.Fatalf("Chat encoder emitted [DONE] after cancelled live tool call:\n%s", out)
		}
		if !strings.Contains(out, `"error"`) {
			t.Fatalf("Chat encoder must emit stream error after cancelled live tool call:\n%s", out)
		}
	})
}

// Review3 P1-2: Non-streaming ConvertResponse must enforce tool JSON object
// validation and discard tool calls when StopEmulate triggers or output is truncated.
func TestReview3_P1_2_NonStreamToolValidationAndStopDiscard(t *testing.T) {
	t.Run("invalid_tool_args_rejected_in_ConvertResponse", func(t *testing.T) {
		reqBody := []byte(`{"model":"gpt-5","input":"run","tools":[{"type":"function","name":"exec","parameters":{"type":"object"}}]}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, reqBody, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		upBody := []byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-5","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"exec","arguments":"["}}]},"finish_reason":"tool_calls"}]}`)
		out, _, err := plan.ConvertResponse(upBody)
		if err == nil {
			t.Fatalf("ConvertResponse must reject non-object tool arguments \"[\", got output: %s", string(out))
		}
	})

	t.Run("stop_emulate_discards_collected_tool_calls", func(t *testing.T) {
		reqBody := []byte(`{"model":"gpt-5","max_tokens":256,"stop_sequences":["END"],"messages":[{"role":"user","content":"run"}],"tools":[{"name":"exec","input_schema":{"type":"object"}}]}`)
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatResponses, reqBody, bridge.Options{
			StopMode: ir.StopEmulate,
		})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		upBody := []byte(`{"id":"resp_1","object":"response","model":"gpt-5","status":"completed","output":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"exec","arguments":"{}","status":"completed"},{"type":"message","id":"m_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"prefixENDtail"}]}]}`)
		out, _, err := plan.ConvertResponse(upBody)
		if err != nil {
			t.Fatalf("ConvertResponse: %v", err)
		}
		if strings.Contains(string(out), `"tool_use"`) {
			t.Fatalf("ConvertResponse with StopEmulate hit must discard tool_use, got: %s", string(out))
		}
		if !strings.Contains(string(out), `"prefix"`) || strings.Contains(string(out), `"tail"`) {
			t.Fatalf("expected prefix truncated at END, got: %s", string(out))
		}
		if !strings.Contains(string(out), `"stop_sequence"`) {
			t.Fatalf("expected stop_reason stop_sequence, got: %s", string(out))
		}
	})

	t.Run("truncated_nonstream_discards_tool_calls", func(t *testing.T) {
		reqBody := []byte(`{"model":"gpt-5","input":"run","tools":[{"type":"function","name":"exec","parameters":{"type":"object"}}]}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, reqBody, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		upBody := []byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-5","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"exec","arguments":"{\"cmd\":"}}]},"finish_reason":"length"}]}`)
		out, _, err := plan.ConvertResponse(upBody)
		if err != nil {
			t.Fatalf("ConvertResponse: %v", err)
		}
		if strings.Contains(string(out), `"function_call"`) {
			t.Fatalf("truncated ConvertResponse must drop incomplete function_call, got: %s", string(out))
		}
	})
}

// Review3 P2-6: Cancelling a LiveFirst call in Responses must keep output_index
// aligned with the final response.output array indices.
func TestReview3_P2_6_ResponsesCancelledLiveFirstOutputIndexAligned(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","stream":true,"input":"run","tools":[{"type":"function","name":"exec","parameters":{"type":"object"}}]}`)
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, reqBody, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	raw := "data: " + `{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"exec","arguments":"{\"cmd\":"}}]},"finish_reason":null}]}` + "\n\n" +
		"data: " + `{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{"content":"tail"},"finish_reason":null}]}` + "\n\n" +
		"data: " + `{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}` + "\n\n" +
		"data: [DONE]\n\n"
	out := feedSSE(t, plan.NewStream(), raw)

	r := sse.NewReader(bytes.NewReader(out))
	msgOutIdx := -1
	var finalOutput []map[string]any
	for {
		ev, err := r.Next()
		if err != nil {
			break
		}
		var m map[string]any
		if json.Unmarshal(ev.Data, &m) != nil {
			continue
		}
		if m["type"] == "response.output_item.added" {
			if it, ok := m["item"].(map[string]any); ok && it["type"] == "message" {
				msgOutIdx = int(m["output_index"].(float64))
			}
		}
		if m["type"] == "response.incomplete" || m["type"] == "response.completed" {
			if resp, ok := m["response"].(map[string]any); ok {
				if arr, ok := resp["output"].([]any); ok {
					for _, e := range arr {
						if im, ok := e.(map[string]any); ok {
							finalOutput = append(finalOutput, im)
						}
					}
				}
			}
		}
	}
	if msgOutIdx < 0 {
		t.Fatalf("missing message output_item.added:\n%s", string(out))
	}
	if msgOutIdx >= len(finalOutput) || finalOutput[msgOutIdx]["type"] != "message" {
		t.Fatalf("output_index mismatch: message streamed at output_index=%d, but final response.output=%v", msgOutIdx, finalOutput)
	}
	// Ensure the cancelled call slot (if present) is marked incomplete, never completed.
	for i, it := range finalOutput {
		if it["type"] == "function_call" && it["status"] == "completed" {
			t.Fatalf("finalOutput[%d] cancelled function_call marked completed: %v", i, it)
		}
	}
}

// Review3 P2-7: Non-streaming StopEmulate must match stop sequences spanning
// adjacent text parts even when one part carries citations or a signature.
func TestReview3_P2_7_NonStreamStopAcrossCitationsAndSignature(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","max_tokens":256,"stop_sequences":["bc"],"messages":[{"role":"user","content":"go"}]}`)
	plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatResponses, reqBody, bridge.Options{
		StopMode: ir.StopEmulate,
	})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	upBody := []byte(`{"id":"resp_1","object":"response","model":"gpt-5","status":"completed","output":[{"type":"message","id":"m_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ab","annotations":[{"type":"url_citation","url":"https://example.com","title":"Ex","start_index":0,"end_index":2}]},{"type":"output_text","text":"cd"}]}]}`)
	out, irResp, err := plan.ConvertResponse(upBody)
	if err != nil {
		t.Fatalf("ConvertResponse: %v", err)
	}
	var gotText strings.Builder
	for _, pt := range irResp.Outputs[0].Parts {
		if pt.Type == ir.PartText {
			gotText.WriteString(pt.Text)
		}
	}
	if gotText.String() != "a" {
		t.Fatalf("expected text truncated to \"a\" across cited parts, got %q (raw=%s)", gotText.String(), string(out))
	}
	if irResp.Outputs[0].Stop.Reason != ir.StopSequence || irResp.Outputs[0].Stop.Sequence != "bc" {
		t.Fatalf("expected StopSequence \"bc\", got %+v", irResp.Outputs[0].Stop)
	}
}

// Review3 P2-3: Gemini functionResponse.parts sub-item parse errors must be
// returned rather than silently dropping the entire parts array.
func TestReview3_P2_3_GeminiFunctionResponsePartsErrorPropagated(t *testing.T) {
	reqBody := []byte(`{
		"contents":[
			{"role":"user","parts":[{"text":"hi"}]},
			{"role":"model","parts":[{"functionCall":{"name":"img","args":{}}}]},
			{"role":"user","parts":[{"functionResponse":{
				"name":"img",
				"response":{"ok":true},
				"parts":[
					{"inlineData":{"mimeType":"image/png","data":"AQ=="}},
					{"unsupportedPart":{}}
				]
			}}]}
		]
	}`)
	if _, err := bridge.Convert(ir.FormatGemini, ir.FormatResponses, reqBody, bridge.Options{Model: "gpt-5"}); err == nil {
		t.Fatalf("expected error for invalid part inside Gemini functionResponse.parts, got nil")
	}
}

// Review3 P2-4: Chat dialect CacheReadFields/CacheWriteFields must persist
// across later partial usage chunks so PromptTokens() is never double-counted.
func TestReview3_P2_4_ChatDialectCacheFieldsNotDoubleCountedAcrossChunks(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	plan, err := bridge.Convert(ir.FormatChat, ir.FormatChat, reqBody, bridge.Options{
		Chat: chat.Dialect{
			CacheReadFields: []string{"cache_hit"},
		},
	})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	s := plan.NewStream()
	chunks := []string{
		`{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[],"usage":{"prompt_tokens":100,"cache_hit":20}}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[],"usage":{"completion_tokens":5}}`,
	}
	var out []byte
	for _, c := range chunks {
		out, err = s.Feed(out, sse.Event{Data: []byte(c)})
		if err != nil {
			t.Fatalf("Feed: %v", err)
		}
	}
	out, err = s.End(out, ir.EndDone)
	if err != nil {
		t.Fatalf("End: %v", err)
	}
	if !strings.Contains(string(out), `"prompt_tokens":100`) {
		t.Fatalf("expected final streamed usage prompt_tokens=100, got:\n%s", string(out))
	}
	u := s.Usage()
	if u.PromptTokens() != 100 || u.InputTokens != 80 || u.CacheReadTokens != 20 || u.OutputTokens != 5 {
		t.Fatalf("unexpected usage: %+v (PromptTokens=%d, want 100; Input=80, CacheRead=20, Out=5)", u, u.PromptTokens())
	}
}

// Review3 P2-5: Anthropic stream ending with EOF or timeout after message_delta
// (stop_reason:"end_turn") but WITHOUT message_stop must emit an error, not success.
func TestReview3_P2_5_AnthropicMissingMessageStopNotSuccess(t *testing.T) {
	reqBody := []byte(`{"model":"claude-sonnet-4","stream":true,"input":"hi"}`)
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, reqBody, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	s := plan.NewStream()
	raw := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}` + "\n\n"

	rd := sse.NewReader(strings.NewReader(raw))
	var out []byte
	for {
		ev, rerr := rd.Next()
		if rerr != nil {
			break
		}
		if len(ev.Data) == 0 {
			continue
		}
		out, err = s.Feed(out, ev)
		if err != nil {
			t.Fatalf("Feed: %v", err)
		}
	}
	out, err = s.End(out, ir.EndEOF)
	if err != nil {
		t.Fatalf("End: %v", err)
	}
	outStr := string(out)
	if strings.Contains(outStr, "event: response.completed") {
		t.Fatalf("Anthropic stream missing message_stop must NOT emit response.completed:\n%s", outStr)
	}
	if !strings.Contains(outStr, "event: response.failed") && !strings.Contains(outStr, "event: error") {
		t.Fatalf("expected error/response.failed when message_stop is missing:\n%s", outStr)
	}
}

// Review3 P2-8: Anthropic server_tool_use input_json_delta buffering must
// respect Limits.MaxBufferedBytes (and default 8 MiB cap).
func TestReview3_P2_8_AnthropicServerToolArgsEnforcesMaxBufferedBytes(t *testing.T) {
	reqBody := []byte(`{"model":"claude-sonnet-4","stream":true,"max_tokens":256,"messages":[{"role":"user","content":"search"}]}`)
	plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatAnthropic, reqBody, bridge.Options{
		Limits: ir.Limits{MaxBufferedBytes: 1024},
	})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	s := plan.NewStream()
	bigChunk := strings.Repeat("a", 600)
	raw := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srv_1","name":"web_search","input":{}}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"` + bigChunk + `"}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"` + bigChunk + `"}}` + "\n\n"

	rd := sse.NewReader(strings.NewReader(raw))
	var out []byte
	var feedErr error
	for {
		ev, rerr := rd.Next()
		if rerr != nil {
			break
		}
		if len(ev.Data) == 0 {
			continue
		}
		out, feedErr = s.Feed(out, ev)
		if feedErr != nil {
			break
		}
	}
	if feedErr == nil && !strings.Contains(string(out), "event: error") {
		t.Fatalf("expected buffer limit error when server_tool_use exceeds MaxBufferedBytes=1024, got out:\n%s", string(out))
	}
}

// Review3 A3 #6: Tool wire-name flattening & collision suffixes must stay
// consistent across tool definitions, historical calls, tool results, and
// tool_choice (Gemini allowedFunctionNames, Anthropic forced choice), and
// Responses non-streaming custom_tool_call must apply reverse name mapping.
func TestReview3_A3_06_ToolRenameConsistentAcrossHistoryChoiceAndCustomDecode(t *testing.T) {
	t.Run("gemini_history_result_and_allowedFunctionNames_renamed", func(t *testing.T) {
		req := &ir.Request{
			Model: "gemini-2.5-flash",
			Tools: []ir.Tool{
				{Name: "a__read", Kind: ir.ToolFunction, Schema: json.RawMessage(`{"type":"object"}`)},
				{Namespace: "a", Name: "read", Kind: ir.ToolFunction, Schema: json.RawMessage(`{"type":"object"}`)},
			},
			ToolChoice: ir.ToolChoice{
				Mode:        ir.ToolChoiceAllowed,
				AllowedMode: ir.ToolChoiceRequired,
				Allowed:     []ir.ToolRef{{Namespace: "a", Name: "read"}},
			},
			Messages: []ir.Message{
				{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("read file")}},
				{Role: ir.RoleAssistant, Parts: []ir.Part{{
					Type: ir.PartToolCall,
					Call: &ir.ToolCall{ID: "c1", Namespace: "a", Name: "read", Arguments: json.RawMessage(`{}`)},
				}}},
				{Role: ir.RoleUser, Parts: []ir.Part{{
					Type:   ir.PartToolResult,
					Result: &ir.ToolResult{CallID: "c1", Name: "read", Content: []ir.Part{ir.TextPart("ok")}},
				}}},
			},
		}
		plan, err := bridge.Build(req, ir.FormatResponses, ir.FormatGemini, bridge.Options{})
		if err != nil {
			t.Fatalf("Build: %v", err)
		}
		var doc struct {
			Contents []struct {
				Parts []struct {
					FunctionCall *struct {
						Name string `json:"name"`
					} `json:"functionCall"`
					FunctionResponse *struct {
						Name string `json:"name"`
					} `json:"functionResponse"`
				} `json:"parts"`
			} `json:"contents"`
			ToolConfig struct {
				FunctionCallingConfig struct {
					AllowedFunctionNames []string `json:"allowedFunctionNames"`
				} `json:"functionCallingConfig"`
			} `json:"toolConfig"`
		}
		if err := json.Unmarshal(plan.Body, &doc); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		// a__read takes "a__read", so (a, read) gets collision suffix "a__read_2".
		want := "a__read_2"
		if doc.Contents[1].Parts[0].FunctionCall == nil || doc.Contents[1].Parts[0].FunctionCall.Name != want {
			t.Fatalf("history functionCall.name = %v, want %q\nbody=%s", doc.Contents[1].Parts[0].FunctionCall, want, string(plan.Body))
		}
		if doc.Contents[2].Parts[0].FunctionResponse == nil || doc.Contents[2].Parts[0].FunctionResponse.Name != want {
			t.Fatalf("history functionResponse.name = %v, want %q\nbody=%s", doc.Contents[2].Parts[0].FunctionResponse, want, string(plan.Body))
		}
		if len(doc.ToolConfig.FunctionCallingConfig.AllowedFunctionNames) != 1 || doc.ToolConfig.FunctionCallingConfig.AllowedFunctionNames[0] != want {
			t.Fatalf("allowedFunctionNames = %v, want [%q]", doc.ToolConfig.FunctionCallingConfig.AllowedFunctionNames, want)
		}
	})

	t.Run("anthropic_forced_choice_and_history_renamed", func(t *testing.T) {
		req := &ir.Request{
			Model: "claude-sonnet-4",
			Tools: []ir.Tool{
				{Namespace: "a", Name: "read", Kind: ir.ToolFunction, Schema: json.RawMessage(`{"type":"object"}`)},
			},
			ToolChoice: ir.ToolChoice{Mode: ir.ToolChoiceTool, Namespace: "a", Name: "read"},
			Messages: []ir.Message{
				{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("read")}},
				{Role: ir.RoleAssistant, Parts: []ir.Part{{
					Type: ir.PartToolCall,
					Call: &ir.ToolCall{ID: "c1", Namespace: "a", Name: "read", Arguments: json.RawMessage(`{}`)},
				}}},
				{Role: ir.RoleUser, Parts: []ir.Part{{
					Type:   ir.PartToolResult,
					Result: &ir.ToolResult{CallID: "c1", Content: []ir.Part{ir.TextPart("ok")}},
				}}},
			},
		}
		plan, err := bridge.Build(req, ir.FormatResponses, ir.FormatAnthropic, bridge.Options{})
		if err != nil {
			t.Fatalf("Build: %v", err)
		}
		if !strings.Contains(string(plan.Body), `"tool_choice":{"name":"a__read","type":"tool"}`) && !strings.Contains(string(plan.Body), `"tool_choice":{"type":"tool","name":"a__read"}`) {
			t.Fatalf("Anthropic forced tool_choice did not use renamed wire name a__read:\n%s", string(plan.Body))
		}
		if !strings.Contains(string(plan.Body), `"type":"tool_use","id":"c1","name":"a__read"`) {
			t.Fatalf("Anthropic history tool_use did not use renamed wire name a__read:\n%s", string(plan.Body))
		}
	})

	t.Run("responses_nonstream_custom_tool_reverse_mapped", func(t *testing.T) {
		req := &ir.Request{
			Model: "gpt-5",
			Tools: []ir.Tool{
				{Namespace: "a", Name: "exec", Kind: ir.ToolCustom},
			},
			Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("run")}}},
		}
		plan, err := bridge.Build(req, ir.FormatResponses, ir.FormatResponses, bridge.Options{})
		if err != nil {
			t.Fatalf("Build: %v", err)
		}
		upBody := []byte(`{"id":"resp_1","object":"response","model":"gpt-5","status":"completed","output":[{"type":"custom_tool_call","id":"ctc_1","call_id":"c1","name":"a__exec","input":"ls"}]}`)
		_, irResp, err := plan.ConvertResponse(upBody)
		if err != nil {
			t.Fatalf("ConvertResponse: %v", err)
		}
		call := irResp.Outputs[0].Parts[0].Call
		if call == nil || call.Namespace != "a" || call.Name != "exec" {
			t.Fatalf("Responses non-streaming custom_tool_call reverse mapping failed: %+v", call)
		}
	})
}

// Review3 A3 #10: StopEmulate must strip native stop sequences from the
// upstream body and force UpstreamStream=true (design.md §9.6).
func TestReview3_A3_10_StopEmulateStripsUpstreamStopAndStreamsUpstream(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","max_tokens":256,"stop_sequences":["END"],"messages":[{"role":"user","content":"go"}],"stream":false}`)
	plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, reqBody, bridge.Options{
		StopMode: ir.StopEmulate,
	})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	if strings.Contains(string(plan.Body), `"stop"`) {
		t.Fatalf("StopEmulate must not send native stop to upstream:\n%s", string(plan.Body))
	}
	if !plan.UpstreamStream {
		t.Fatalf("StopEmulate must set Plan.UpstreamStream=true (design.md §9.6)")
	}
	if plan.ClientStream {
		t.Fatalf("Plan.ClientStream should stay false for non-streaming client")
	}
	if len(plan.Request.Stop) != 1 || plan.Request.Stop[0] != "END" {
		t.Fatalf("Plan.Request.Stop must retain original stop sequences for local filter, got %v", plan.Request.Stop)
	}
}

// Review3 A3 #18: Responses SSE output_item.done web_search_call must decode
// to EvBuiltin, and Anthropic search/fetch results must preserve Raw
// (encrypted_content, page_age, error objects) on same-origin round trip while
// reporting loss across formats when fields cannot be represented.
func TestReview3_A3_18_ResponsesSSEWebSearchAndAnthropicRawResultPreserved(t *testing.T) {
	t.Run("responses_sse_web_search_call_decodes", func(t *testing.T) {
		reqBody := []byte(`{"model":"gpt-5","stream":true,"input":"search"}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatResponses, reqBody, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		raw := "event: response.created\n" +
			`data: {"type":"response.created","sequence_number":0,"response":{"id":"r1","model":"gpt-5","status":"in_progress"}}` + "\n\n" +
			"event: response.output_item.done\n" +
			`data: {"type":"response.output_item.done","sequence_number":1,"output_index":0,"item":{"type":"web_search_call","id":"ws_1","status":"completed"}}` + "\n\n" +
			"event: response.completed\n" +
			`data: {"type":"response.completed","sequence_number":2,"response":{"id":"r1","model":"gpt-5","status":"completed","output":[{"type":"web_search_call","id":"ws_1","status":"completed"}]}}` + "\n\n"
		out := string(feedSSE(t, plan.NewStream(), raw))
		if !strings.Contains(out, `"web_search_call"`) || !strings.Contains(out, `"ws_1"`) {
			t.Fatalf("Responses SSE web_search_call was dropped:\n%s", out)
		}
	})

	t.Run("anthropic_same_origin_preserves_raw_encrypted_content_and_page_age", func(t *testing.T) {
		reqBody := []byte(`{"model":"claude-sonnet-4","max_tokens":256,"messages":[{"role":"user","content":"search"}]}`)
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatAnthropic, reqBody, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		upBody := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4","content":[{"type":"web_search_tool_result","tool_use_id":"srv_1","content":[{"type":"web_search_result","url":"https://example.com","title":"Ex","encrypted_content":"enc_abc","page_age":"2 days ago"}]}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`)
		out, _, err := plan.ConvertResponse(upBody)
		if err != nil {
			t.Fatalf("ConvertResponse: %v", err)
		}
		if !strings.Contains(string(out), `"encrypted_content":"enc_abc"`) || !strings.Contains(string(out), `"page_age":"2 days ago"`) {
			t.Fatalf("Anthropic same-origin web_search_tool_result dropped raw fields:\n%s", string(out))
		}
	})
}

// Review3 A3 #19: Responses web_search config (external_web_access:false,
// allowed_domains, user_location) must map where supported and reject/lose
// where unsupported (with Deny respected).
func TestReview3_A3_19_WebSearchConfigMappedOrRejectedCrossFormat(t *testing.T) {
	t.Run("responses_to_anthropic_maps_allowed_domains_and_user_location", func(t *testing.T) {
		reqBody := []byte(`{
			"model":"claude-sonnet-4",
			"input":"news",
			"tools":[{"type":"web_search","filters":{"allowed_domains":["example.com"]},"user_location":{"type":"approximate","country":"US","city":"SF"}}]
		}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, reqBody, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		if !strings.Contains(string(plan.Body), `"allowed_domains":["example.com"]`) {
			t.Fatalf("expected allowed_domains mapped to Anthropic web_search:\n%s", string(plan.Body))
		}
		if !strings.Contains(string(plan.Body), `"user_location"`) {
			t.Fatalf("expected user_location mapped to Anthropic web_search:\n%s", string(plan.Body))
		}
	})

	t.Run("cached_web_search_dropped_with_loss_and_blockable_by_deny", func(t *testing.T) {
		reqBody := []byte(`{
			"model":"claude-sonnet-4",
			"input":"news",
			"tools":[{"type":"web_search","external_web_access":false}]
		}`)
		for _, target := range []ir.Format{ir.FormatAnthropic, ir.FormatGemini} {
			plan, err := bridge.Convert(ir.FormatResponses, target, reqBody, bridge.Options{})
			if err != nil {
				t.Fatalf("Convert->%s: %v", target, err)
			}
			if strings.Contains(string(plan.Body), "web_search") || strings.Contains(string(plan.Body), "googleSearch") {
				t.Fatalf("cached web_search should be dropped towards %s, got:\n%s", target, string(plan.Body))
			}
			hasLoss := false
			for _, l := range plan.Losses {
				if l.Kind == ir.LossCachedWebSearch {
					hasLoss = true
				}
			}
			if !hasLoss {
				t.Fatalf("expected LossCachedWebSearch towards %s, got losses=%v", target, plan.Losses)
			}
			if _, err := bridge.Convert(ir.FormatResponses, target, reqBody, bridge.Options{
				Deny: ir.LossSet{ir.LossCachedWebSearch: true},
			}); err == nil {
				t.Fatalf("expected Deny[LossCachedWebSearch] to reject towards %s", target)
			}
		}
	})

	t.Run("gemini_rejects_web_search_domain_restrictions", func(t *testing.T) {
		reqBody := []byte(`{
			"model":"gemini-2.5-flash",
			"input":"news",
			"tools":[{"type":"web_search","filters":{"allowed_domains":["example.com"]}}]
		}`)
		if _, err := bridge.Convert(ir.FormatResponses, ir.FormatGemini, reqBody, bridge.Options{}); err == nil {
			t.Fatalf("expected Gemini to reject web_search with allowed_domains restriction")
		}
	})
}

// Review3 A3 #20 (Gemini): parametersJsonSchema constraints such as
// additionalProperties:false must be preserved on same-format Gemini builds,
// and OpenAPI downgrade must record a loss when schema constraints are dropped.
func TestReview3_A3_20_GeminiParametersJsonSchemaPreservedAndDowngradeLoss(t *testing.T) {
	t.Run("same_format_gemini_preserves_parametersJsonSchema_additionalProperties", func(t *testing.T) {
		reqBody := []byte(`{
			"contents":[{"role":"user","parts":[{"text":"hi"}]}],
			"tools":[{"functionDeclarations":[{
				"name":"lookup",
				"parametersJsonSchema":{"type":"object","properties":{"q":{"type":"string"}},"additionalProperties":false}
			}]}]
		}`)
		plan, err := bridge.Convert(ir.FormatGemini, ir.FormatGemini, reqBody, bridge.Options{Model: "gemini-2.5-flash"})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		if !strings.Contains(string(plan.Body), `"parametersJsonSchema"`) || !strings.Contains(string(plan.Body), `"additionalProperties":false`) {
			t.Fatalf("same-format Gemini build dropped parametersJsonSchema or additionalProperties:false:\n%s", string(plan.Body))
		}
	})

	t.Run("cross_format_openapi_downgrade_records_loss", func(t *testing.T) {
		reqBody := []byte(`{
			"model":"gemini-2.5-flash",
			"messages":[{"role":"user","content":"hi"}],
			"tools":[{"type":"function","function":{
				"name":"lookup",
				"parameters":{"type":"object","properties":{"q":{"type":"string"}},"additionalProperties":false}
			}}]
		}`)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatGemini, reqBody, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		hasSchemaLoss := false
		for _, l := range plan.Losses {
			if l.Kind == ir.LossToolSchema {
				hasSchemaLoss = true
			}
		}
		if !hasSchemaLoss {
			t.Fatalf("expected LossToolSchema when Gemini ToOpenAPI drops additionalProperties:false, got losses=%v (body=%s)", plan.Losses, string(plan.Body))
		}
		if _, err := bridge.Convert(ir.FormatChat, ir.FormatGemini, reqBody, bridge.Options{
			Deny: ir.LossSet{ir.LossToolSchema: true},
		}); err == nil {
			t.Fatalf("expected Deny[LossToolSchema] to reject OpenAPI schema downgrade")
		}
	})
}

// Review3 A3 #21: Gemini OffPatch must be applied to the built request body,
// and Chat OffPatch merging with a client passthrough object must produce a
// single merged top-level key without duplicate JSON keys.
func TestReview3_A3_21_OffPatchAppliedInGeminiAndMergedInChat(t *testing.T) {
	t.Run("gemini_applies_valid_off_patch", func(t *testing.T) {
		reqBody := []byte(`{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none"}`)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatGemini, reqBody, bridge.Options{
			Profile: &ir.ModelProfile{
				Off: ir.OffRule{
					Mode:  ir.OffPatch,
					Patch: json.RawMessage(`{"enable_thinking":false}`),
				},
			},
		})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		if !strings.Contains(string(plan.Body), `"enable_thinking":false`) {
			t.Fatalf("Gemini build ignored valid OffPatch:\n%s", string(plan.Body))
		}
	})

	t.Run("chat_merges_off_patch_with_passthrough_no_duplicate_keys", func(t *testing.T) {
		reqBody := []byte(`{
			"model":"qwen",
			"messages":[{"role":"user","content":"hi"}],
			"reasoning_effort":"none",
			"chat_template_kwargs":{"custom_flag":true,"enable_thinking":true}
		}`)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatChat, reqBody, bridge.Options{
			Profile: &ir.ModelProfile{
				Off: ir.OffRule{
					Mode:  ir.OffPatch,
					Patch: json.RawMessage(`{"chat_template_kwargs":{"enable_thinking":false}}`),
				},
			},
		})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		bodyStr := string(plan.Body)
		if strings.Count(bodyStr, `"chat_template_kwargs"`) != 1 {
			t.Fatalf("Chat build emitted duplicate chat_template_kwargs key:\n%s", bodyStr)
		}
		var doc struct {
			Kwargs map[string]any `json:"chat_template_kwargs"`
		}
		if err := json.Unmarshal(plan.Body, &doc); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if doc.Kwargs["enable_thinking"] != false || doc.Kwargs["custom_flag"] != true {
			t.Fatalf("expected merged chat_template_kwargs {enable_thinking:false, custom_flag:true}, got %v (body=%s)", doc.Kwargs, bodyStr)
		}
	})
}

// Review3 A3 #24: Response-phase losses must be readable without OnLoss via
// ir.Response.Losses (ConvertResponse / NewCollector) and Stream.Losses().
func TestReview3_A3_24_ResponseLossesReadableWithoutOnLoss(t *testing.T) {
	reqBody := []byte(`{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"draw"}]}`)
	plan, err := bridge.Convert(ir.FormatChat, ir.FormatGemini, reqBody, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}

	t.Run("nonstream_ConvertResponse_populates_irResp_Losses", func(t *testing.T) {
		upBody := []byte(`{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"image/png","data":"AQ=="}},{"text":"done"}]},"finishReason":"STOP"}],"modelVersion":"gemini-2.5-flash"}`)
		_, irResp, err := plan.ConvertResponse(upBody)
		if err != nil {
			t.Fatalf("ConvertResponse: %v", err)
		}
		found := false
		for _, l := range irResp.Losses {
			if l.Kind == ir.LossOutputMedia {
				found = true
			}
		}
		if !found {
			t.Fatalf("expected LossOutputMedia in irResp.Losses without OnLoss, got %v", irResp.Losses)
		}
	})

	t.Run("streaming_Stream_Losses_exposes_response_losses", func(t *testing.T) {
		streamReq := []byte(`{"model":"gemini-2.5-flash","stream":true,"messages":[{"role":"user","content":"draw"}]}`)
		sPlan, err := bridge.Convert(ir.FormatChat, ir.FormatGemini, streamReq, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		s := sPlan.NewStream()
		raw := "data: " + `{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"image/png","data":"AQ=="}},{"text":"done"}]},"finishReason":"STOP"}],"modelVersion":"gemini-2.5-flash"}` + "\n\n"
		_ = feedSSE(t, s, raw)
		losses := s.Losses()
		found := false
		for _, l := range losses {
			if l.Kind == ir.LossOutputMedia {
				found = true
			}
		}
		if !found {
			t.Fatalf("expected LossOutputMedia in s.Losses() without OnLoss, got %v", losses)
		}
	})
}
