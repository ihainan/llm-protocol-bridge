package bridge_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
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

// TestMatrix_4x4_FakeUpstream exercises all 16 (client -> upstream) pairs for
// request conversion, non-streaming response conversion, streaming conversion,
// error conversion, and explicit unsupported-feature rejection.
func TestMatrix_4x4_FakeUpstream(t *testing.T) {
	// Representative client requests per format covering instructions, multi-turn,
	// tool definitions (with strict), tool calls & tool results, reasoning, and stop sequences.
	clientReqBody := func(f ir.Format, stream bool) []byte {
		switch f {
		case ir.FormatChat:
			m := map[string]any{
				"model": "test-model",
				"messages": []any{
					map[string]any{"role": "system", "content": "be concise"},
					map[string]any{"role": "user", "content": "check weather"},
					map[string]any{"role": "assistant", "content": "checking", "tool_calls": []any{
						map[string]any{"id": "call_1", "type": "function", "function": map[string]any{"name": "Weather", "arguments": `{"city":"NYC"}`}},
					}},
					map[string]any{"role": "tool", "tool_call_id": "call_1", "content": "sunny 72F"},
					map[string]any{"role": "user", "content": "summarize"},
				},
				"tools": []any{
					map[string]any{"type": "function", "function": map[string]any{
						"name":        "Weather",
						"description": "Get weather",
						"strict":      true,
						"parameters":  map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}, "required": []string{"city"}},
					}},
				},
				"reasoning_effort": "low",
				"stream":           stream,
			}
			b, _ := json.Marshal(m)
			return b

		case ir.FormatAnthropic:
			m := map[string]any{
				"model":      "test-model",
				"max_tokens": 8192,
				"system":     "be concise",
				"thinking":   map[string]any{"type": "enabled", "budget_tokens": 2048},
				"messages": []any{
					map[string]any{"role": "user", "content": "check weather"},
					map[string]any{"role": "assistant", "content": []any{
						map[string]any{"type": "text", "text": "checking"},
						map[string]any{"type": "tool_use", "id": "call_1", "name": "Weather", "input": map[string]any{"city": "NYC"}},
					}},
					map[string]any{"role": "user", "content": []any{
						map[string]any{"type": "tool_result", "tool_use_id": "call_1", "content": "sunny 72F"},
						map[string]any{"type": "text", "text": "summarize"},
					}},
				},
				"tools": []any{
					map[string]any{
						"name":         "Weather",
						"description":  "Get weather",
						"input_schema": map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}},
					},
				},
				"stream": stream,
			}
			b, _ := json.Marshal(m)
			return b

		case ir.FormatResponses:
			m := map[string]any{
				"model":        "test-model",
				"instructions": "be concise",
				"reasoning":    map[string]any{"effort": "low"},
				"input": []any{
					map[string]any{"type": "message", "role": "user", "content": "check weather"},
					map[string]any{"type": "message", "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": "checking"}}},
					map[string]any{"type": "function_call", "call_id": "call_1", "name": "Weather", "arguments": `{"city":"NYC"}`},
					map[string]any{"type": "function_call_output", "call_id": "call_1", "output": "sunny 72F"},
					map[string]any{"type": "message", "role": "user", "content": "summarize"},
				},
				"tools": []any{
					map[string]any{
						"type":        "function",
						"name":        "Weather",
						"description": "Get weather",
						"strict":      true,
						"parameters":  map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}},
					},
				},
				"stream": stream,
			}
			b, _ := json.Marshal(m)
			return b

		case ir.FormatGemini:
			m := map[string]any{
				"systemInstruction": map[string]any{"parts": []any{map[string]any{"text": "be concise"}}},
				"contents": []any{
					map[string]any{"role": "user", "parts": []any{map[string]any{"text": "check weather"}}},
					map[string]any{"role": "model", "parts": []any{
						map[string]any{"text": "checking"},
						map[string]any{"functionCall": map[string]any{"id": "call_1", "name": "Weather", "args": map[string]any{"city": "NYC"}}},
					}},
					map[string]any{"role": "user", "parts": []any{
						map[string]any{"functionResponse": map[string]any{"id": "call_1", "name": "Weather", "response": map[string]any{"output": "sunny 72F"}}},
						map[string]any{"text": "summarize"},
					}},
				},
				"tools": []any{
					map[string]any{"functionDeclarations": []any{
						map[string]any{
							"name":        "Weather",
							"description": "Get weather",
							"parameters":  map[string]any{"type": "OBJECT", "properties": map[string]any{"city": map[string]any{"type": "STRING"}}},
						},
					}},
				},
				"generationConfig": map[string]any{"thinkingConfig": map[string]any{"thinkingBudget": 2048}},
			}
			b, _ := json.Marshal(m)
			return b
		}
		return nil
	}

	// Fake upstream non-streaming responses with reasoning + text + parallel tool calls + usage.
	fakeUpstreamResp := func(f ir.Format) []byte {
		switch f {
		case ir.FormatChat:
			return []byte(`{"id":"chatcmpl-1","object":"chat.completion","model":"up-model","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":"plan","content":"done","tool_calls":[{"id":"c1","type":"function","function":{"name":"Weather","arguments":"{\"city\":\"LA\"}"}},{"id":"c2","type":"function","function":{"name":"Weather","arguments":"{\"city\":\"SF\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":30,"total_tokens":130,"prompt_tokens_details":{"cached_tokens":20},"completion_tokens_details":{"reasoning_tokens":10}}}`)
		case ir.FormatAnthropic:
			return []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"up-model","content":[{"type":"thinking","thinking":"plan","signature":"sig_a"},{"type":"text","text":"done"},{"type":"tool_use","id":"c1","name":"Weather","input":{"city":"LA"}},{"type":"tool_use","id":"c2","name":"Weather","input":{"city":"SF"}}],"stop_reason":"tool_use","usage":{"input_tokens":80,"cache_read_input_tokens":20,"output_tokens":30}}`)
		case ir.FormatResponses:
			return []byte(`{"id":"resp_1","object":"response","model":"up-model","status":"completed","output":[{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"plan"}]},{"type":"message","id":"m_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done"}]},{"type":"function_call","id":"fc_1","call_id":"c1","name":"Weather","arguments":"{\"city\":\"LA\"}","status":"completed"},{"type":"function_call","id":"fc_2","call_id":"c2","name":"Weather","arguments":"{\"city\":\"SF\"}","status":"completed"}],"usage":{"input_tokens":100,"output_tokens":30,"total_tokens":130,"input_tokens_details":{"cached_tokens":20},"output_tokens_details":{"reasoning_tokens":10}}}`)
		case ir.FormatGemini:
			return []byte(`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"plan","thought":true},{"text":"done"},{"functionCall":{"id":"c1","name":"Weather","args":{"city":"LA"}}},{"functionCall":{"id":"c2","name":"Weather","args":{"city":"SF"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":20,"candidatesTokenCount":20,"thoughtsTokenCount":10,"totalTokenCount":130},"modelVersion":"up-model"}`)
		}
		return nil
	}

	// Fake upstream streaming SSE payloads.
	fakeUpstreamStream := func(f ir.Format) string {
		switch f {
		case ir.FormatChat:
			return "" +
				`data: {"id":"c1","model":"up-model","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"plan"}}]}` + "\n\n" +
				`data: {"id":"c1","model":"up-model","choices":[{"index":0,"delta":{"content":"do"}}]}` + "\n\n" +
				`data: {"id":"c1","model":"up-model","choices":[{"index":0,"delta":{"content":"ne"}}]}` + "\n\n" +
				`data: {"id":"c1","model":"up-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"Weather","arguments":"{\"city\":"}}]}}]}` + "\n\n" +
				`data: {"id":"c1","model":"up-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"c2","type":"function","function":{"name":"Weather","arguments":"{\"city\":"}}]}}]}` + "\n\n" +
				`data: {"id":"c1","model":"up-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"LA\"}"}}]}}]}` + "\n\n" +
				`data: {"id":"c1","model":"up-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"SF\"}"}}]}}]}` + "\n\n" +
				`data: {"id":"c1","model":"up-model","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":30,"total_tokens":130,"prompt_tokens_details":{"cached_tokens":20},"completion_tokens_details":{"reasoning_tokens":10}}}` + "\n\n" +
				"data: [DONE]\n\n"
		case ir.FormatAnthropic:
			return "" +
				"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"model\":\"up-model\",\"usage\":{\"input_tokens\":80,\"cache_read_input_tokens\":20,\"output_tokens\":0}}}\n\n" +
				"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n" +
				"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"plan\"}}\n\n" +
				"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
				"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
				"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"done\"}}\n\n" +
				"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n" +
				"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"c1\",\"name\":\"Weather\",\"input\":{}}}\n\n" +
				"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"LA\\\"}\"}}\n\n" +
				"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":2}\n\n" +
				"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":3,\"content_block\":{\"type\":\"tool_use\",\"id\":\"c2\",\"name\":\"Weather\",\"input\":{}}}\n\n" +
				"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":3,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"SF\\\"}\"}}\n\n" +
				"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":3}\n\n" +
				"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":30}}\n\n" +
				"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
		case ir.FormatResponses:
			return "" +
				"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"r1\",\"model\":\"up-model\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
				"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[]}}\n\n" +
				"event: response.reasoning_summary_text.delta\ndata: {\"type\":\"response.reasoning_summary_text.delta\",\"sequence_number\":2,\"output_index\":0,\"delta\":\"plan\"}\n\n" +
				"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"sequence_number\":3,\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\"}}\n\n" +
				"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":4,\"output_index\":1,\"item\":{\"type\":\"message\",\"id\":\"m1\",\"role\":\"assistant\",\"content\":[]}}\n\n" +
				"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":5,\"output_index\":1,\"delta\":\"done\"}\n\n" +
				"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"sequence_number\":6,\"output_index\":1,\"item\":{\"type\":\"message\",\"id\":\"m1\",\"role\":\"assistant\"}}\n\n" +
				"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":7,\"output_index\":2,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"c1\",\"name\":\"Weather\"}}\n\n" +
				"event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"sequence_number\":8,\"output_index\":2,\"delta\":\"{\\\"city\\\":\\\"LA\\\"}\"}\n\n" +
				"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"sequence_number\":9,\"output_index\":2,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"c1\",\"name\":\"Weather\",\"arguments\":\"{\\\"city\\\":\\\"LA\\\"}\"}}\n\n" +
				"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":10,\"output_index\":3,\"item\":{\"type\":\"function_call\",\"id\":\"fc_2\",\"call_id\":\"c2\",\"name\":\"Weather\"}}\n\n" +
				"event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"sequence_number\":11,\"output_index\":3,\"delta\":\"{\\\"city\\\":\\\"SF\\\"}\"}\n\n" +
				"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"sequence_number\":12,\"output_index\":3,\"item\":{\"type\":\"function_call\",\"id\":\"fc_2\",\"call_id\":\"c2\",\"name\":\"Weather\",\"arguments\":\"{\\\"city\\\":\\\"SF\\\"}\"}}\n\n" +
				"event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":13,\"response\":{\"id\":\"r1\",\"model\":\"up-model\",\"status\":\"completed\",\"usage\":{\"input_tokens\":100,\"output_tokens\":30,\"total_tokens\":130,\"input_tokens_details\":{\"cached_tokens\":20},\"output_tokens_details\":{\"reasoning_tokens\":10}}}}\n\n"
		case ir.FormatGemini:
			return "" +
				`data: {"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"plan","thought":true}]}}],"modelVersion":"up-model"}` + "\n\n" +
				`data: {"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"done"}]}}]}` + "\n\n" +
				`data: {"candidates":[{"index":0,"content":{"role":"model","parts":[{"functionCall":{"id":"c1","name":"Weather","args":{"city":"LA"}}},{"functionCall":{"id":"c2","name":"Weather","args":{"city":"SF"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":20,"candidatesTokenCount":20,"thoughtsTokenCount":10,"totalTokenCount":130}}` + "\n\n"
		}
		return ""
	}

	fakeUpstreamErr := func(f ir.Format) (int, []byte) {
		switch f {
		case ir.FormatChat, ir.FormatResponses:
			return 429, []byte(`{"error":{"type":"rate_limit_error","code":"rate_limit_exceeded","message":"slow down"}}`)
		case ir.FormatAnthropic:
			return 429, []byte(`{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`)
		case ir.FormatGemini:
			return 429, []byte(`{"error":{"code":429,"message":"slow down","status":"RESOURCE_EXHAUSTED"}}`)
		}
		return 500, nil
	}

	decodeClientResp := func(t *testing.T, f ir.Format, body []byte) *ir.Response {
		t.Helper()
		var (
			r   *ir.Response
			err error
		)
		switch f {
		case ir.FormatChat:
			r, err = chat.ParseResponse(body, chat.DecodeOptions{})
		case ir.FormatAnthropic:
			r, err = anthropic.ParseResponse(body, anthropic.DecodeOptions{})
		case ir.FormatResponses:
			r, err = responses.ParseResponse(body, responses.DecodeOptions{})
		case ir.FormatGemini:
			r, err = gemini.ParseResponse(body, gemini.DecodeOptions{})
		}
		if err != nil {
			t.Fatalf("decode client %s non-streaming response: %v\nbody: %s", f, err, body)
		}
		return r
	}

	decodeClientStream := func(t *testing.T, f ir.Format, raw []byte) *ir.Response {
		t.Helper()
		col := ir.NewCollector()
		switch f {
		case ir.FormatChat:
			d := chat.NewStreamDecoder(chat.DecodeOptions{})
			rr := sse.NewReader(bytes.NewReader(raw))
			for {
				ev, err := rr.Next()
				if err != nil {
					break
				}
				evs, _ := d.Decode(nil, ev)
				for _, e := range evs {
					col.Add(e)
				}
			}
		case ir.FormatAnthropic:
			d := anthropic.NewStreamDecoder(anthropic.DecodeOptions{})
			rr := sse.NewReader(bytes.NewReader(raw))
			for {
				ev, err := rr.Next()
				if err != nil {
					break
				}
				evs, _ := d.Decode(nil, ev)
				for _, e := range evs {
					col.Add(e)
				}
			}
		case ir.FormatResponses:
			d := responses.NewStreamDecoder(responses.DecodeOptions{})
			rr := sse.NewReader(bytes.NewReader(raw))
			for {
				ev, err := rr.Next()
				if err != nil {
					break
				}
				evs, _ := d.Decode(nil, ev)
				for _, e := range evs {
					col.Add(e)
				}
			}
		case ir.FormatGemini:
			d := gemini.NewStreamDecoder(gemini.DecodeOptions{})
			rr := sse.NewReader(bytes.NewReader(raw))
			for {
				ev, err := rr.Next()
				if err != nil {
					break
				}
				evs, _ := d.Decode(nil, ev)
				for _, e := range evs {
					col.Add(e)
				}
			}
		}
		return col.Response()
	}

	assertReplySemantics := func(t *testing.T, label string, r *ir.Response) {
		t.Helper()
		if len(r.Outputs) == 0 {
			t.Fatalf("%s: no outputs", label)
		}
		out := r.Outputs[0]
		if got := ir.TextOf(out.Parts, ""); got != "done" {
			t.Errorf("%s: text want %q, got %q", label, "done", got)
		}
		var calls []*ir.ToolCall
		for _, p := range out.Parts {
			if p.Type == ir.PartToolCall && p.Call != nil {
				calls = append(calls, p.Call)
			}
		}
		if len(calls) != 2 {
			t.Fatalf("%s: want 2 parallel tool calls, got %d (%+v)", label, len(calls), out.Parts)
		}
		if calls[0].Name != "Weather" || !strings.Contains(string(calls[0].Arguments), "LA") {
			t.Errorf("%s: call 0 mismatch: %+v", label, calls[0])
		}
		if calls[1].Name != "Weather" || !strings.Contains(string(calls[1].Arguments), "SF") {
			t.Errorf("%s: call 1 mismatch: %+v", label, calls[1])
		}
		if out.Stop.Reason != ir.StopToolUse {
			t.Errorf("%s: stop reason want tool_use, got %+v", label, out.Stop)
		}
		if r.Usage.InputTokens != 80 || r.Usage.CacheReadTokens != 20 || r.Usage.OutputTokens != 30 {
			t.Errorf("%s: usage mismatch: got %+v, want input=80 cache_read=20 output=30", label, r.Usage)
		}
	}

	for _, client := range ir.Formats {
		for _, upstream := range ir.Formats {
			client, upstream := client, upstream
			dir := fmt.Sprintf("%s_to_%s", client, upstream)
			t.Run(dir, func(t *testing.T) {
				o := bridge.Options{
					Model:  "test-model",
					Stream: false,
					AltSSE: true,
				}
				// 1. Non-streaming request + response conversion.
				plan, err := bridge.Convert(client, upstream, clientReqBody(client, false), o)
				if err != nil {
					t.Fatalf("%s non-stream Convert: %v", dir, err)
				}
				if !json.Valid(plan.Body) {
					t.Fatalf("%s non-stream built invalid JSON: %s", dir, plan.Body)
				}
				clientRespBody, _, err := plan.ConvertResponse(fakeUpstreamResp(upstream))
				if err != nil {
					t.Fatalf("%s ConvertResponse: %v", dir, err)
				}
				assertReplySemantics(t, dir+"/nonstream", decodeClientResp(t, client, clientRespBody))

				// 2. Streaming request + SSE conversion + Stream.Usage().
				oStream := o
				oStream.Stream = true
				planS, err := bridge.Convert(client, upstream, clientReqBody(client, true), oStream)
				if err != nil {
					t.Fatalf("%s stream Convert: %v", dir, err)
				}
				s := planS.NewStream()
				out := feedBridgeSSE(t, s, fakeUpstreamStream(upstream))
				out, err = s.End(out, ir.EndDone)
				if err != nil {
					t.Fatalf("%s stream End: %v", dir, err)
				}
				if !s.Done() {
					t.Errorf("%s stream Done() == false after clean end", dir)
				}
				assertReplySemantics(t, dir+"/stream", decodeClientStream(t, client, out))
				su := s.Usage()
				if su.InputTokens != 80 || su.CacheReadTokens != 20 || su.OutputTokens != 30 {
					t.Errorf("%s Stream.Usage(): got %+v, want input=80 cache_read=20 output=30", dir, su)
				}

				// 3. Error conversion produces valid client-format error envelope.
				st, errBody := fakeUpstreamErr(upstream)
				gotSt, gotBody := plan.ConvertError(st, errBody)
				if gotSt != 429 || !strings.Contains(string(gotBody), "slow down") {
					t.Errorf("%s ConvertError: status=%d body=%s", dir, gotSt, gotBody)
				}
				switch client {
				case ir.FormatChat, ir.FormatResponses:
					if !strings.Contains(string(gotBody), `"rate_limit_error"`) {
						t.Errorf("%s OpenAI error missing wire type: %s", dir, gotBody)
					}
				case ir.FormatAnthropic:
					if !strings.Contains(string(gotBody), `"type":"error"`) {
						t.Errorf("%s Anthropic error wrong shape: %s", dir, gotBody)
					}
				case ir.FormatGemini:
					if !strings.Contains(string(gotBody), `"RESOURCE_EXHAUSTED"`) {
						t.Errorf("%s Gemini error missing status: %s", dir, gotBody)
					}
				}
			})
		}
	}

	// 4. Unsupported feature errors include structured Code and Path.
	t.Run("unsupported_feature_error_has_code_and_path", func(t *testing.T) {
		_, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic,
			[]byte(`{"model":"claude-4","n":3,"messages":[{"role":"user","content":"hi"}]}`),
			bridge.Options{},
		)
		var irErr *ir.Error
		if !errors.As(err, &irErr) {
			t.Fatalf("expected *ir.Error, got %T: %v", err, err)
		}
		if irErr.Code != ir.CodeUnsupported || irErr.Path.Render(ir.StyleBracket) == "" {
			t.Errorf("expected CodeUnsupported with non-empty Path, got %+v", irErr)
		}
	})
}
