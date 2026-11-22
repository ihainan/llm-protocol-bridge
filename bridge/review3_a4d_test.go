package bridge_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// A4d P2 #1: Chat streaming emulated custom tool must reject unwrapped JSON
// string arguments (e.g. "\"hello\"", "\"\"") instead of accepting them as
// already-unwrapped custom inputs.
func TestReview3_A4d_01_ChatStreamRawJSONStringCustomArgsRejected(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","input":"go","stream":true,"tools":[{"type":"custom","name":"exec"}]}`)
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, reqBody, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	for _, badArgs := range []string{`"hello"`, `""`} {
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
				t.Fatalf("expected response.failed for raw JSON string custom args %q, got:\n%s", badArgs, s)
			}
			if !strings.Contains(s, `"response.failed"`) {
				t.Fatalf("expected response.failed for %q, got:\n%s", badArgs, s)
			}
		})
	}
}

// A4d P2 #2: Gemini streaming custom tool must emit unwrapped input deltas
// matching the final custom_tool_call_input.done value (never raw JSON wrapper).
func TestReview3_A4d_02_GeminiStreamCustomToolDeltaMatchesDone(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","input":"go","stream":true,"tools":[{"type":"custom","name":"exec"}]}`)
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatGemini, reqBody, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	st := plan.NewStream()
	chunk := []byte(`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"exec","args":{"input":"hello"}}}]},"finishReason":"STOP"}]}`)
	var out []byte
	out, err = st.Feed(out, sse.Event{Data: chunk})
	if err != nil {
		t.Fatalf("Feed: %v", err)
	}
	out, err = st.End(out, ir.EndEOF)
	if err != nil {
		t.Fatalf("End: %v", err)
	}
	s := string(out)
	if strings.Contains(s, `"delta":"{\"input\":\"hello\"}"`) {
		t.Fatalf("raw JSON wrapper leaked into custom_tool_call_input.delta:\n%s", s)
	}
	if !strings.Contains(s, `"delta":"hello"`) || !strings.Contains(s, `"input":"hello"`) || !strings.Contains(s, `"response.completed"`) {
		t.Fatalf("expected unwrapped delta and done 'hello' with response.completed, got:\n%s", s)
	}
}
