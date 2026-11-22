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
