package bridge_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// A4c P2 #1: $ref target with nullable:true merged with non-nullable sibling
// type:"string" must clear nullable:true instead of silently allowing null.
func TestReview3_A4c_01_SchemaRefSiblingClearsNullable(t *testing.T) {
	reqBody := []byte(`{"model":"gemini-2.5-pro","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"t","parameters":{"type":"object","$defs":{"S":{"type":["string","null"]}},"properties":{"x":{"$ref":"#/$defs/S","type":"string"}}}}}]}`)
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
						Type     string `json:"type"`
						Nullable bool   `json:"nullable"`
					} `json:"properties"`
				} `json:"parameters"`
			} `json:"functionDeclarations"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(plan.Body, &doc); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	xProp := doc.Tools[0].FunctionDeclarations[0].Parameters.Properties["x"]
	if xProp.Type != "string" || xProp.Nullable {
		t.Fatalf("expected non-nullable string, got type=%q nullable=%v (body: %s)", xProp.Type, xProp.Nullable, string(plan.Body))
	}
}

// A4c P2 #2: Non-streaming emulated custom tool via Chat with empty/omitted
// arguments must be rejected, while valid {"input":""} remains accepted.
func TestReview3_A4c_02_NonStreamEmulatedCustomEmptyArgsRejected(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","input":"go","tools":[{"type":"custom","name":"exec"}]}`)
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, reqBody, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}

	t.Run("empty_string_args", func(t *testing.T) {
		upBody := []byte(`{"id":"r","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"exec","arguments":""}}]},"finish_reason":"tool_calls"}]}`)
		out, _, err := plan.ConvertResponse(upBody)
		if err == nil {
			t.Fatalf("expected error for empty emulated custom tool arguments, got: %s", string(out))
		}
	})

	t.Run("omitted_args", func(t *testing.T) {
		upBody := []byte(`{"id":"r","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"exec"}}]},"finish_reason":"tool_calls"}]}`)
		out, _, err := plan.ConvertResponse(upBody)
		if err == nil {
			t.Fatalf("expected error for omitted emulated custom tool arguments, got: %s", string(out))
		}
	})

	t.Run("valid_empty_input_object", func(t *testing.T) {
		upBody := []byte(`{"id":"r","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"exec","arguments":"{\"input\":\"\"}"}}]},"finish_reason":"tool_calls"}]}`)
		out, _, err := plan.ConvertResponse(upBody)
		if err != nil {
			t.Fatalf("valid {\"input\":\"\"} should succeed, got error: %v (out=%s)", err, string(out))
		}
	})
}

// A4c P2 #3: Cancelling a buffered tool call when an earlier intercepted call
// exists in the timeline must still flush queued text A before subsequent B.
func TestReview3_A4c_03_CancelledToolAfterInterceptedPreservesTextOrder(t *testing.T) {
	reqBody := []byte(`{"model":"claude-sonnet-4-6","max_tokens":128,"stream":true,"messages":[{"role":"user","content":"go"}],"tools":[{"name":"f","input_schema":{"type":"object"}},{"name":"g","input_schema":{"type":"object"}}]}`)
	plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatResponses, reqBody, bridge.Options{
		Intercept: []string{"g"},
	})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	st := plan.NewStream()
	events := []sse.Event{
		{Name: "response.created", Data: []byte(`{"type":"response.created","response":{"id":"r","status":"in_progress"}}`)},
		{Name: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc0","call_id":"g0","name":"g","status":"in_progress","arguments":""}}`)},
		{Name: "response.output_item.done", Data: []byte(`{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc0","call_id":"g0","name":"g","status":"completed","arguments":"{}"}}`)},
		{Name: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc1","call_id":"f1","name":"f","status":"in_progress","arguments":""}}`)},
		{Name: "response.output_text.delta", Data: []byte(`{"type":"response.output_text.delta","output_index":2,"delta":"A"}`)},
		{Name: "response.output_item.done", Data: []byte(`{"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"fc1","call_id":"f1","name":"f","status":"cancelled","arguments":"{}"}}`)},
		{Name: "response.output_text.delta", Data: []byte(`{"type":"response.output_text.delta","output_index":2,"delta":"B"}`)},
		{Name: "response.completed", Data: []byte(`{"type":"response.completed","response":{"id":"r","status":"completed"}}`)},
	}
	var out []byte
	for _, ev := range events {
		out, err = st.Feed(out, ev)
		if err != nil {
			t.Fatalf("Feed: %v", err)
		}
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
	if ic := st.Intercepted(); len(ic) != 1 || ic[0].Name != "g" {
		t.Fatalf("expected intercepted call g, got %+v", ic)
	}
}

// A4c P2 #4: Gemini groundingMetadata webSearchQueries dropped by Responses
// web_search_call must record LossBuiltinActivity and obey Deny.
func TestReview3_A4c_04_GeminiSearchQueryToResponsesReportsLoss(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","input":"search"}`)
	opts := bridge.Options{
		Deny: ir.LossSet{ir.LossBuiltinActivity: true},
	}
	upBody := []byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"No results"}]},"groundingMetadata":{"webSearchQueries":["secret"]},"finishReason":"STOP"}]}`)

	t.Run("nonstream", func(t *testing.T) {
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatGemini, reqBody, opts)
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		out, _, err := plan.ConvertResponse(upBody)
		if err == nil {
			t.Fatalf("expected Deny[LossBuiltinActivity] error when dropping Gemini webSearchQueries, got: %s", string(out))
		}
	})

	t.Run("stream", func(t *testing.T) {
		streamReq := []byte(`{"model":"gpt-5","input":"search","stream":true}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatGemini, streamReq, opts)
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		st := plan.NewStream()
		out, err := st.Feed(nil, sse.Event{Data: upBody})
		if err == nil {
			t.Fatalf("expected Deny[LossBuiltinActivity] error in stream when dropping Gemini webSearchQueries, got:\n%s", string(out))
		}
	})
}

// A4c P2 #5: Responses web_search_call action (query & sources) must be
// preserved on Responses->Responses instead of being dropped during decode.
func TestReview3_A4c_05_ResponsesWebSearchActionPreserved(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","input":"search","stream":true}`)
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatResponses, reqBody, bridge.Options{
		Deny: ir.LossSet{ir.LossBuiltinActivity: true},
	})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	st := plan.NewStream()
	events := []sse.Event{
		{Name: "response.created", Data: []byte(`{"type":"response.created","response":{"id":"r","status":"in_progress"}}`)},
		{Name: "response.output_item.done", Data: []byte(`{"type":"response.output_item.done","output_index":0,"item":{"type":"web_search_call","id":"ws_1","status":"completed","action":{"type":"search","query":"secret","sources":[{"type":"url","url":"https://example.com/"}]}}}`)},
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
	if !strings.Contains(s, `"query":"secret"`) || !strings.Contains(s, `"https://example.com/"`) {
		t.Fatalf("expected Responses web_search_call action query and sources preserved, got:\n%s", s)
	}
}

// A4c P2 #6: Chat streaming emulated custom tool must reject invalid wrapper
// JSON ("null", "[]", "{}", "{bad") instead of encoding raw text as input.
func TestReview3_A4c_06_ChatStreamEmulatedCustomRejectsInvalidWrapper(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","input":"go","stream":true,"tools":[{"type":"custom","name":"exec"}]}`)
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, reqBody, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	for _, badArgs := range []string{`null`, `[]`, `{}`, `{bad`} {
		t.Run(badArgs, func(t *testing.T) {
			st := plan.NewStream()
			rawArg, _ := json.Marshal(badArgs)
			chunk := []byte(`{"id":"r","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c","type":"function","function":{"name":"exec","arguments":` + string(rawArg) + `}}]},"finish_reason":"tool_calls"}]}`)
			var out []byte
			out, err := st.Feed(out, sse.Event{Data: chunk})
			if err != nil {
				t.Fatalf("Feed: %v", err)
			}
			out, err = st.End(out, ir.EndDone)
			if err != nil {
				t.Fatalf("End: %v", err)
			}
			s := string(out)
			if strings.Contains(s, `"response.completed"`) || strings.Contains(s, `"status":"completed"`) {
				t.Fatalf("expected response.failed for invalid streaming custom tool wrapper %q, got:\n%s", badArgs, s)
			}
			if !strings.Contains(s, `"response.failed"`) {
				t.Fatalf("expected response.failed for %q, got:\n%s", badArgs, s)
			}
		})
	}
}
