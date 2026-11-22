package bridge_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// TestLiveRegression_AnthropicThinkingTokens verifies that Anthropic upstream
// usage.output_tokens_details.thinking_tokens (observed on claude-sonnet-4-6
// adaptive thinking) is preserved across non-streaming and streaming conversions
// to Chat, Responses, Gemini, and Anthropic without double-counting output tokens.
func TestLiveRegression_AnthropicThinkingTokens(t *testing.T) {
	// Sanitized fixture modeled on real claude-sonnet-4-6 adaptive thinking output.
	nonStreamBody := []byte(`{
		"id":"msg_live_001",
		"type":"message",
		"role":"assistant",
		"model":"claude-sonnet-4-6",
		"content":[
			{"type":"thinking","thinking":"17 * 19 = 323","signature":"sig_test_001"},
			{"type":"text","text":"323"}
		],
		"stop_reason":"end_turn",
		"stop_sequence":null,
		"usage":{
			"input_tokens":18,
			"cache_creation_input_tokens":0,
			"cache_read_input_tokens":0,
			"output_tokens":29,
			"output_tokens_details":{"thinking_tokens":14}
		}
	}`)

	streamSSE := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_live_002","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":18,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":1}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"17 * 19 = 323"}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_test_002"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"323"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":1}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":29,"output_tokens_details":{"thinking_tokens":14}}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")

	clientReqs := map[ir.Format]string{
		ir.FormatChat:      `{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"17*19"}],"max_completion_tokens":256,"reasoning_effort":"low","stream":%v,"stream_options":{"include_usage":true}}`,
		ir.FormatResponses: `{"model":"claude-sonnet-4-6","input":"17*19","max_output_tokens":256,"reasoning":{"effort":"low"},"stream":%v}`,
		ir.FormatAnthropic: `{"model":"claude-sonnet-4-6","max_tokens":256,"thinking":{"type":"adaptive"},"output_config":{"effort":"low"},"messages":[{"role":"user","content":"17*19"}],"stream":%v}`,
		ir.FormatGemini:    `{"contents":[{"role":"user","parts":[{"text":"17*19"}]}],"generationConfig":{"maxOutputTokens":256,"thinkingConfig":{"thinkingBudget":128,"includeThoughts":true}}}`,
	}

	opts := bridge.Options{
		Profile: &ir.ModelProfile{Style: ir.StyleAdaptive},
	}

	for _, clientFmt := range []ir.Format{ir.FormatChat, ir.FormatResponses, ir.FormatAnthropic, ir.FormatGemini} {
		for _, stream := range []bool{false, true} {
			name := string(clientFmt) + "/non-stream"
			if stream {
				name = string(clientFmt) + "/stream"
			}
			t.Run(name, func(t *testing.T) {
				reqBody := clientReqs[clientFmt]
				if clientFmt != ir.FormatGemini {
					reqBody = strings.Replace(reqBody, "%v", map[bool]string{false: "false", true: "true"}[stream], 1)
				}
				callOpts := opts
				if clientFmt == ir.FormatGemini {
					callOpts.Model = "claude-sonnet-4-6"
					callOpts.Stream = stream
					callOpts.AltSSE = stream
				}
				plan, err := bridge.Convert(clientFmt, ir.FormatAnthropic, []byte(reqBody), callOpts)
				if err != nil {
					t.Fatalf("Convert: %v", err)
				}

				var outBytes []byte
				if !stream {
					outBytes, _, err = plan.ConvertResponse(nonStreamBody)
					if err != nil {
						t.Fatalf("ConvertResponse: %v", err)
					}
				} else {
					st := plan.NewStream()
					rd := sse.NewReader(strings.NewReader(streamSSE))
					for {
						ev, rerr := rd.Next()
						if rerr != nil {
							break
						}
						if len(ev.Data) == 0 && ev.Name == "" {
							continue
						}
						outBytes, err = st.Feed(outBytes, ev)
						if err != nil {
							t.Fatalf("Feed: %v", err)
						}
					}
					outBytes, err = st.End(outBytes, ir.EndEOF)
					if err != nil {
						t.Fatalf("End: %v", err)
					}
				}

				assertUsageThinking14(t, clientFmt, stream, outBytes)
			})
		}
	}
}

func assertUsageThinking14(t *testing.T, fmt ir.Format, stream bool, raw []byte) {
	t.Helper()
	var doc map[string]any
	if !stream {
		if err := json.Unmarshal(raw, &doc); err != nil {
			t.Fatalf("unmarshal %s: %v\n%s", fmt, err, string(raw))
		}
	} else {
		rd := sse.NewReader(strings.NewReader(string(raw)))
		for {
			ev, err := rd.Next()
			if err != nil {
				break
			}
			if len(ev.Data) == 0 || sse.IsDone(ev.Data) {
				continue
			}
			var m map[string]any
			if json.Unmarshal(ev.Data, &m) != nil {
				continue
			}
			switch fmt {
			case ir.FormatChat:
				if m["usage"] != nil {
					doc = m
				}
			case ir.FormatResponses:
				if m["type"] == "response.completed" {
					if r, ok := m["response"].(map[string]any); ok {
						doc = r
					}
				}
			case ir.FormatAnthropic:
				if m["type"] == "message_delta" {
					doc = m
				}
			case ir.FormatGemini:
				if m["usageMetadata"] != nil {
					doc = m
				}
			}
		}
		if doc == nil {
			t.Fatalf("no usage chunk found in %s stream:\n%s", fmt, string(raw))
		}
	}

	switch fmt {
	case ir.FormatChat:
		u, _ := doc["usage"].(map[string]any)
		det, _ := u["completion_tokens_details"].(map[string]any)
		if got := int(u["completion_tokens"].(float64)); got != 29 {
			t.Fatalf("chat completion_tokens = %d, want 29", got)
		}
		if det == nil || int(det["reasoning_tokens"].(float64)) != 14 {
			t.Fatalf("chat reasoning_tokens = %v, want 14 (usage=%v)", det, u)
		}
	case ir.FormatResponses:
		u, _ := doc["usage"].(map[string]any)
		det, _ := u["output_tokens_details"].(map[string]any)
		if got := int(u["output_tokens"].(float64)); got != 29 {
			t.Fatalf("responses output_tokens = %d, want 29", got)
		}
		if det == nil || int(det["reasoning_tokens"].(float64)) != 14 {
			t.Fatalf("responses reasoning_tokens = %v, want 14 (usage=%v)", det, u)
		}
	case ir.FormatAnthropic:
		u, _ := doc["usage"].(map[string]any)
		det, _ := u["output_tokens_details"].(map[string]any)
		if got := int(u["output_tokens"].(float64)); got != 29 {
			t.Fatalf("anthropic output_tokens = %d, want 29", got)
		}
		if det == nil || int(det["thinking_tokens"].(float64)) != 14 {
			t.Fatalf("anthropic thinking_tokens = %v, want 14 (usage=%v)", det, u)
		}
	case ir.FormatGemini:
		u, _ := doc["usageMetadata"].(map[string]any)
		cands := int(u["candidatesTokenCount"].(float64))
		th, _ := u["thoughtsTokenCount"].(float64)
		if int(th) != 14 {
			t.Fatalf("gemini thoughtsTokenCount = %v, want 14 (usage=%v)", u["thoughtsTokenCount"], u)
		}
		if cands+int(th) != 29 {
			t.Fatalf("gemini candidates(%d)+thoughts(%d) = %d, want 29", cands, int(th), cands+int(th))
		}
	}
}

// TestLiveRegression_GeminiToolThoughtSignatureRoundTrip verifies that Gemini 3
// functionCall thoughtSignature survives a round trip through Responses and
// Anthropic clients (both non-streaming and streaming) so round 2 requests back
// to Gemini include thoughtSignature on the historical functionCall part.
func TestLiveRegression_GeminiToolThoughtSignatureRoundTrip(t *testing.T) {
	const testSig = "AY89a1_sanitized_gemini3_thought_signature_example_001=="

	// Sanitized Gemini 3 functionCall response fixtures.
	gemNonStream := []byte(`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"Tokyo"}},"thoughtSignature":"` + testSig + `"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":39,"candidatesTokenCount":16,"thoughtsTokenCount":41,"totalTokenCount":96},"modelVersion":"gemini-3-flash-preview"}`)

	gemStreamSSE := "data: " + string(gemNonStream) + "\n\n"

	for _, clientFmt := range []ir.Format{ir.FormatResponses, ir.FormatAnthropic} {
		for _, stream := range []bool{false, true} {
			name := string(clientFmt) + "/non-stream"
			if stream {
				name = string(clientFmt) + "/stream"
			}
			t.Run(name, func(t *testing.T) {
				var r1Req []byte
				if clientFmt == ir.FormatResponses {
					r1Req = []byte(`{"model":"gemini-3-flash-preview","input":"weather in Tokyo?","tools":[{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],"stream":` + map[bool]string{false: "false", true: "true"}[stream] + `}`)
				} else {
					r1Req = []byte(`{"model":"gemini-3-flash-preview","max_tokens":256,"messages":[{"role":"user","content":"weather in Tokyo?"}],"tools":[{"name":"get_weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],"stream":` + map[bool]string{false: "false", true: "true"}[stream] + `}`)
				}

				plan1, err := bridge.Convert(clientFmt, ir.FormatGemini, r1Req, bridge.Options{})
				if err != nil {
					t.Fatalf("r1 Convert: %v", err)
				}

				var out1 []byte
				if !stream {
					out1, _, err = plan1.ConvertResponse(gemNonStream)
					if err != nil {
						t.Fatalf("r1 ConvertResponse: %v", err)
					}
				} else {
					st := plan1.NewStream()
					rd := sse.NewReader(strings.NewReader(gemStreamSSE))
					for {
						ev, rerr := rd.Next()
						if rerr != nil {
							break
						}
						if len(ev.Data) == 0 {
							continue
						}
						out1, err = st.Feed(out1, ev)
						if err != nil {
							t.Fatalf("r1 Feed: %v", err)
						}
					}
					out1, err = st.End(out1, ir.EndEOF)
					if err != nil {
						t.Fatalf("r1 End: %v", err)
					}
				}

				wireCallID := extractWireToolCallID(t, clientFmt, stream, out1)
				if wireCallID == "" {
					t.Fatalf("missing tool call id in r1 output:\n%s", string(out1))
				}
				if clientFmt == ir.FormatAnthropic {
					for i := 0; i < len(wireCallID); i++ {
						c := wireCallID[i]
						if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-') {
							t.Fatalf("anthropic tool_use.id %q contains illegal character %q", wireCallID, c)
						}
					}
				}

				// Round 2: client echoes wireCallID back in both call and tool result.
				var r2Req []byte
				if clientFmt == ir.FormatResponses {
					r2Req, _ = json.Marshal(map[string]any{
						"model": "gemini-3-flash-preview",
						"input": []any{
							map[string]any{"role": "user", "content": "weather in Tokyo?"},
							map[string]any{"type": "function_call", "call_id": wireCallID, "name": "get_weather", "arguments": `{"city":"Tokyo"}`},
							map[string]any{"type": "function_call_output", "call_id": wireCallID, "output": `{"temp_c":22}`},
						},
						"tools": []any{map[string]any{"type": "function", "name": "get_weather", "parameters": map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}}}},
					})
				} else {
					r2Req, _ = json.Marshal(map[string]any{
						"model":      "gemini-3-flash-preview",
						"max_tokens": 256,
						"messages": []any{
							map[string]any{"role": "user", "content": "weather in Tokyo?"},
							map[string]any{"role": "assistant", "content": []any{
								map[string]any{"type": "tool_use", "id": wireCallID, "name": "get_weather", "input": map[string]any{"city": "Tokyo"}},
							}},
							map[string]any{"role": "user", "content": []any{
								map[string]any{"type": "tool_result", "tool_use_id": wireCallID, "content": `{"temp_c":22}`},
							}},
						},
						"tools": []any{map[string]any{"name": "get_weather", "input_schema": map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}}}},
					})
				}

				plan2, err := bridge.Convert(clientFmt, ir.FormatGemini, r2Req, bridge.Options{})
				if err != nil {
					t.Fatalf("r2 Convert: %v", err)
				}

				var gemReq struct {
					Contents []struct {
						Role  string `json:"role"`
						Parts []struct {
							ThoughtSignature string `json:"thoughtSignature"`
							FunctionCall     *struct {
								ID   string `json:"id"`
								Name string `json:"name"`
							} `json:"functionCall"`
							FunctionResponse *struct {
								ID   string `json:"id"`
								Name string `json:"name"`
							} `json:"functionResponse"`
						} `json:"parts"`
					} `json:"contents"`
				}
				if err := json.Unmarshal(plan2.Body, &gemReq); err != nil {
					t.Fatalf("unmarshal r2 gemini body: %v", err)
				}
				if len(gemReq.Contents) < 3 || len(gemReq.Contents[1].Parts) == 0 {
					t.Fatalf("unexpected gemini r2 contents: %s", string(plan2.Body))
				}
				fcPart := gemReq.Contents[1].Parts[0]
				if fcPart.ThoughtSignature != testSig {
					t.Fatalf("r2 Gemini functionCall thoughtSignature = %q, want %q\nbody=%s", fcPart.ThoughtSignature, testSig, string(plan2.Body))
				}
				if len(gemReq.Contents[2].Parts) == 0 || gemReq.Contents[2].Parts[0].FunctionResponse == nil || gemReq.Contents[2].Parts[0].FunctionResponse.Name != "get_weather" {
					t.Fatalf("r2 Gemini functionResponse missing name get_weather: %s", string(plan2.Body))
				}

				// Switching upstream to Anthropic on round 2 must strip the sealed signature from tool_use.id / tool_use_id.
				planAnth, err := bridge.Convert(clientFmt, ir.FormatAnthropic, r2Req, bridge.Options{})
				if err != nil {
					t.Fatalf("r2 Convert->Anthropic: %v", err)
				}
				if strings.Contains(string(planAnth.Body), "lpb1.") || strings.Contains(string(planAnth.Body), testSig) {
					t.Fatalf("r2 Anthropic body leaked Gemini signature: %s", string(planAnth.Body))
				}
			})
		}
	}
}

func extractWireToolCallID(t *testing.T, fmt ir.Format, stream bool, raw []byte) string {
	t.Helper()
	if !stream {
		var doc map[string]any
		if err := json.Unmarshal(raw, &doc); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if fmt == ir.FormatResponses {
			for _, it := range doc["output"].([]any) {
				m := it.(map[string]any)
				if m["type"] == "function_call" {
					return m["call_id"].(string)
				}
			}
		} else {
			for _, b := range doc["content"].([]any) {
				m := b.(map[string]any)
				if m["type"] == "tool_use" {
					return m["id"].(string)
				}
			}
		}
		return ""
	}
	rd := sse.NewReader(strings.NewReader(string(raw)))
	for {
		ev, err := rd.Next()
		if err != nil {
			break
		}
		if len(ev.Data) == 0 || sse.IsDone(ev.Data) {
			continue
		}
		var m map[string]any
		if json.Unmarshal(ev.Data, &m) != nil {
			continue
		}
		if fmt == ir.FormatResponses && m["type"] == "response.output_item.done" {
			if it, ok := m["item"].(map[string]any); ok && it["type"] == "function_call" {
				return it["call_id"].(string)
			}
		}
		if fmt == ir.FormatAnthropic && m["type"] == "content_block_start" {
			if cb, ok := m["content_block"].(map[string]any); ok && cb["type"] == "tool_use" {
				return cb["id"].(string)
			}
		}
	}
	return ""
}

// TestLiveRegression_CachedWebSearchDroppedForChat verifies design.md §9.8:
// Responses web_search with external_web_access: false (sent by Codex CLI by
// default) is droppable towards Chat Completions via ir.LossCachedWebSearch,
// while active web_search (without external_web_access: false) remains unsupported.
func TestLiveRegression_CachedWebSearchDroppedForChat(t *testing.T) {
	cachedReq := []byte(`{
		"model":"gpt-4.1-mini",
		"input":"Reply with pong",
		"tools":[{"type":"web_search","external_web_access":false}]
	}`)

	var losses []ir.Loss
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, cachedReq, bridge.Options{
		OnLoss: func(l ir.Loss) { losses = append(losses, l) },
	})
	if err != nil {
		t.Fatalf("Convert cached web_search -> chat failed: %v", err)
	}
	foundLoss := false
	for _, l := range losses {
		if l.Kind == ir.LossCachedWebSearch {
			foundLoss = true
		}
	}
	if !foundLoss {
		t.Fatalf("expected LossCachedWebSearch, got losses=%v", losses)
	}
	if strings.Contains(string(plan.Body), "web_search") {
		t.Fatalf("expected cached web_search to be omitted from chat body, got: %s", string(plan.Body))
	}

	// Explicitly denying LossCachedWebSearch must reject the request.
	_, err = bridge.Convert(ir.FormatResponses, ir.FormatChat, cachedReq, bridge.Options{
		Deny: ir.LossSet{ir.LossCachedWebSearch: true},
	})
	if err == nil {
		t.Fatalf("expected error when LossCachedWebSearch is denied")
	}

	// Active web_search (external_web_access not false) must still be unsupported.
	activeReq := []byte(`{
		"model":"gpt-4.1-mini",
		"input":"Reply with pong",
		"tools":[{"type":"web_search"}]
	}`)
	if _, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, activeReq, bridge.Options{}); err == nil {
		t.Fatalf("expected error for active web_search towards Chat Completions")
	}
}
