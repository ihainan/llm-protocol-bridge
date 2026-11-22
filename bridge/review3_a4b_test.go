package bridge_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// A4b P1 #1: Anthropic stream ending with message_stop without message_delta
// stop_reason must fail instead of completing buffered tool calls.
func TestReview3_A4b_01_AnthropicMissingStopReasonFailsStream(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","input":"run","stream":true,"tools":[{"type":"function","name":"exec","parameters":{"type":"object"}}]}`)
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, reqBody, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	st := plan.NewStream()
	events := []sse.Event{
		{Name: "message_start", Data: []byte(`{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"m","content":[]}}`)},
		{Name: "content_block_start", Data: []byte(`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"c","name":"exec","input":{}}}`)},
		{Name: "content_block_stop", Data: []byte(`{"type":"content_block_stop","index":0}`)},
		{Name: "message_stop", Data: []byte(`{"type":"message_stop"}`)},
	}
	var out []byte
	for _, ev := range events {
		out, err = st.Feed(out, ev)
		if err != nil {
			t.Fatalf("Feed: %v", err)
		}
	}
	out, err = st.End(out, ir.EndEOF)
	if err != nil {
		t.Fatalf("End: %v", err)
	}
	s := string(out)
	if strings.Contains(s, `"response.completed"`) || strings.Contains(s, `"status":"completed"`) {
		t.Fatalf("expected failure when message_delta stop_reason is missing, got completed output:\n%s", s)
	}
	if !strings.Contains(s, `"response.failed"`) {
		t.Fatalf("expected response.failed event, got:\n%s", s)
	}
}

// A4b P1 #2: Non-streaming Responses with in_progress/cancelled response status
// or in_progress tool status must not be promoted to completed.
func TestReview3_A4b_02_ResponsesNonStreamInProgressAndCancelledNotCompleted(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","input":"run","tools":[{"type":"function","name":"exec","parameters":{"type":"object"}}]}`)
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatResponses, reqBody, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}

	t.Run("response_in_progress", func(t *testing.T) {
		upBody := []byte(`{"id":"r","object":"response","model":"m","status":"in_progress","output":[{"type":"function_call","id":"fc","call_id":"c","name":"exec","arguments":"{}","status":"in_progress"}]}`)
		out, _, err := plan.ConvertResponse(upBody)
		if err == nil {
			t.Fatalf("expected error for in_progress non-streaming response, got: %s", string(out))
		}
	})

	t.Run("response_cancelled", func(t *testing.T) {
		upBody := []byte(`{"id":"r","object":"response","model":"m","status":"cancelled","output":[{"type":"function_call","id":"fc","call_id":"c","name":"exec","arguments":"{}","status":"completed"}]}`)
		out, _, err := plan.ConvertResponse(upBody)
		if err == nil {
			t.Fatalf("expected error for cancelled non-streaming response, got: %s", string(out))
		}
	})

	t.Run("tool_in_progress_inside_completed_response", func(t *testing.T) {
		upBody := []byte(`{"id":"r","object":"response","model":"m","status":"completed","output":[{"type":"function_call","id":"fc","call_id":"c","name":"exec","arguments":"{}","status":"in_progress"}]}`)
		out, irResp, err := plan.ConvertResponse(upBody)
		if err != nil {
			t.Fatalf("ConvertResponse: %v", err)
		}
		if len(irResp.Outputs[0].Parts) != 0 || strings.Contains(string(out), `"type":"function_call"`) {
			t.Fatalf("in_progress function_call was promoted to completed: %s", string(out))
		}
	})
}

// A4b P2 #3: Cancelling the last buffered tool call must flush queued text
// immediately so subsequent text does not overtake it ("AB", not "BA").
func TestReview3_A4b_03_CancelledBufferedToolPreservesTextOrder(t *testing.T) {
	reqBody := []byte(`{"model":"claude-sonnet-4-6","max_tokens":128,"stream":true,"messages":[{"role":"user","content":"go"}],"tools":[{"name":"f","input_schema":{"type":"object"}}]}`)
	plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatResponses, reqBody, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	st := plan.NewStream()
	events := []sse.Event{
		{Name: "response.created", Data: []byte(`{"type":"response.created","response":{"id":"r","status":"in_progress"}}`)},
		{Name: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc","call_id":"c","name":"f","status":"in_progress","arguments":""}}`)},
		{Name: "response.output_text.delta", Data: []byte(`{"type":"response.output_text.delta","output_index":1,"delta":"A"}`)},
		{Name: "response.output_item.done", Data: []byte(`{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc","call_id":"c","name":"f","status":"cancelled","arguments":"{}"}}`)},
		{Name: "response.output_text.delta", Data: []byte(`{"type":"response.output_text.delta","output_index":1,"delta":"B"}`)},
		{Name: "response.completed", Data: []byte(`{"type":"response.completed","response":{"id":"r","status":"completed"}}`)},
	}
	var out []byte
	for _, ev := range events {
		out, err = st.Feed(out, ev)
		if err != nil {
			t.Fatalf("Feed: %v", err)
		}
	}
	out, err = st.End(out, ir.EndEOF)
	if err != nil {
		t.Fatalf("End: %v", err)
	}
	s := string(out)
	posA := strings.Index(s, `"text":"A"`)
	posB := strings.Index(s, `"text":"B"`)
	if posA < 0 || posB < 0 {
		t.Fatalf("expected both text deltas A and B, got:\n%s", s)
	}
	if posA > posB {
		t.Fatalf("expected text delta A before B, got B before A:\n%s", s)
	}
}

// A4b P2 #4: $ref sibling keywords must go through normal OpenAPI conversion
// (handling type arrays and properties maps without stripping property names).
func TestReview3_A4b_04_SchemaRefSiblingNormalOpenAPIConversion(t *testing.T) {
	t.Run("type_array_sibling", func(t *testing.T) {
		reqBody := []byte(`{"model":"gemini-2.5-pro","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"t","parameters":{"type":"object","$defs":{"S":{"type":"string"}},"properties":{"x":{"$ref":"#/$defs/S","type":["string","null"]}}}}}]}`)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatGemini, reqBody, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		if strings.Contains(string(plan.Body), `["string","null"]`) {
			t.Fatalf("raw JSON Schema type array leaked into Gemini OpenAPI schema:\n%s", string(plan.Body))
		}
	})

	t.Run("properties_sibling_preserves_property_names", func(t *testing.T) {
		reqBody := []byte(`{"model":"gemini-2.5-pro","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"t","parameters":{"type":"object","$defs":{"S":{"type":"object"}},"properties":{"x":{"$ref":"#/$defs/S","properties":{"b":{"type":"integer"}},"required":["b"]}}}}}]}`)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatGemini, reqBody, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		var doc struct {
			Tools []struct {
				FunctionDeclarations []struct {
					Parameters struct {
						Properties map[string]struct {
							Properties map[string]map[string]any `json:"properties"`
							Required   []string                  `json:"required"`
						} `json:"properties"`
					} `json:"parameters"`
				} `json:"functionDeclarations"`
			} `json:"tools"`
		}
		if err := json.Unmarshal(plan.Body, &doc); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		xProp := doc.Tools[0].FunctionDeclarations[0].Parameters.Properties["x"]
		if _, ok := xProp.Properties["b"]; !ok {
			t.Fatalf("sibling property 'b' was stripped from Gemini schema: %s", string(plan.Body))
		}
	})
}

// A4b P2 #5: $ref sibling constraints must intersect (keep stricter bound or
// report LossToolSchema), never silently relax the referenced constraint.
func TestReview3_A4b_05_SchemaRefSiblingDoesNotRelaxConstraint(t *testing.T) {
	reqBody := []byte(`{"model":"gemini-2.5-pro","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"t","parameters":{"type":"object","$defs":{"S":{"type":"string","minLength":5}},"properties":{"x":{"$ref":"#/$defs/S","minLength":1}}}}}]}`)
	plan, err := bridge.Convert(ir.FormatChat, ir.FormatGemini, reqBody, bridge.Options{
		Deny: ir.LossSet{ir.LossToolSchema: true},
	})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	var doc struct {
		Tools []struct {
			FunctionDeclarations []struct {
				Parameters struct {
					Properties map[string]struct {
						MinLength float64 `json:"minLength"`
					} `json:"properties"`
				} `json:"parameters"`
			} `json:"functionDeclarations"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(plan.Body, &doc); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	got := doc.Tools[0].FunctionDeclarations[0].Parameters.Properties["x"].MinLength
	if got != 5 {
		t.Fatalf("expected stricter intersected minLength=5, got %v (body: %s)", got, string(plan.Body))
	}

	// Conflicting pattern sibling cannot be intersected in OpenAPI and must be blocked by Deny[LossToolSchema].
	conflictReq := []byte(`{"model":"gemini-2.5-pro","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"t","parameters":{"type":"object","$defs":{"S":{"type":"string","pattern":"^a"}},"properties":{"x":{"$ref":"#/$defs/S","pattern":"b$"}}}}}]}`)
	if _, err := bridge.Convert(ir.FormatChat, ir.FormatGemini, conflictReq, bridge.Options{
		Deny: ir.LossSet{ir.LossToolSchema: true},
	}); err == nil {
		t.Fatalf("expected Deny[LossToolSchema] to reject conflicting $ref sibling pattern")
	}
}

// A4b P2 #6: Emulated custom tool via Chat with invalid JSON wrapper arguments
// ("null", "[]", "123", "{bad") must be rejected in ConvertResponse.
func TestReview3_A4b_06_EmulatedCustomToolRejectsInvalidJSONArgs(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","input":"go","tools":[{"type":"custom","name":"exec"}]}`)
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, reqBody, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	for _, badArgs := range []string{`null`, `[]`, `123`, `{bad`, `{}`} {
		t.Run(badArgs, func(t *testing.T) {
			rawArg, _ := json.Marshal(badArgs)
			upBody := []byte(`{"id":"x","model":"m","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"exec","arguments":` + string(rawArg) + `}}]},"finish_reason":"tool_calls"}]}`)
			out, _, err := plan.ConvertResponse(upBody)
			if err == nil {
				t.Fatalf("expected error for invalid emulated custom tool arguments %q, got: %s", badArgs, string(out))
			}
		})
	}
}

// A4b P2 #7: Anthropic server_tool_use web_search input query dropped when
// encoding Responses web_search_call must record LossBuiltinActivity and obey Deny.
func TestReview3_A4b_07_ResponsesWebSearchCallReportsDroppedInputLoss(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","input":"search"}`)
	opts := bridge.Options{
		Deny: ir.LossSet{ir.LossBuiltinActivity: true},
	}

	t.Run("nonstream", func(t *testing.T) {
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, reqBody, opts)
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		upBody := []byte(`{"id":"a","type":"message","role":"assistant","model":"m","content":[{"type":"server_tool_use","id":"s","name":"web_search","input":{"query":"secret"}}],"stop_reason":"end_turn"}`)
		out, _, err := plan.ConvertResponse(upBody)
		if err == nil {
			t.Fatalf("expected Deny[LossBuiltinActivity] to fail when dropping web_search input query, got: %s", string(out))
		}
	})

	t.Run("stream", func(t *testing.T) {
		streamReq := []byte(`{"model":"gpt-5","input":"search","stream":true}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, streamReq, opts)
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		st := plan.NewStream()
		events := []sse.Event{
			{Name: "message_start", Data: []byte(`{"type":"message_start","message":{"id":"a","type":"message","role":"assistant","model":"m","content":[]}}`)},
			{Name: "content_block_start", Data: []byte(`{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"s","name":"web_search","input":{"query":"secret"}}}`)},
			{Name: "content_block_stop", Data: []byte(`{"type":"content_block_stop","index":0}`)},
			{Name: "message_delta", Data: []byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn"}}`)},
			{Name: "message_stop", Data: []byte(`{"type":"message_stop"}`)},
		}
		var out []byte
		var feedErr error
		for _, ev := range events {
			out, feedErr = st.Feed(out, ev)
			if feedErr != nil {
				break
			}
		}
		if feedErr == nil {
			t.Fatalf("expected Deny[LossBuiltinActivity] error in stream when dropping web_search input query, got:\n%s", string(out))
		}
	})
}
