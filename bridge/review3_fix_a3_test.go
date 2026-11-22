package bridge_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/bridge"
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
