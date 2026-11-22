package bridge_test

import (
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

// Review2 #13: Namespaced tools with identical bare names (a.read, b.read) must not
// collide in Responses, Anthropic, or Gemini builders, and Anthropic ToolChoiceAllowed
// must match by (Namespace, Name) tuple.
func TestReview2_P2_13_NamespaceCollisionAndAnthropicAllowed(t *testing.T) {
	req := &ir.Request{
		Model:    "m",
		Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("read")}}},
		Tools: []ir.Tool{
			{Name: "read", Namespace: "a", Kind: ir.ToolFunction, Schema: json.RawMessage(`{"type":"object"}`)},
			{Name: "read", Namespace: "b", Kind: ir.ToolFunction, Schema: json.RawMessage(`{"type":"object"}`)},
		},
	}

	t.Run("responses_distinct_names_and_reverse_map", func(t *testing.T) {
		built, err := responses.BuildRequest(req, responses.BuildOptions{})
		if err != nil {
			t.Fatalf("responses BuildRequest: %v", err)
		}
		if len(built.Names) != 2 {
			t.Fatalf("expected 2 distinct entries in built.Names, got %+v (body=%s)", built.Names, string(built.Body))
		}
	})

	t.Run("anthropic_distinct_names_and_tuple_allowed", func(t *testing.T) {
		built, err := anthropic.BuildRequest(req, anthropic.BuildOptions{})
		if err != nil {
			t.Fatalf("anthropic BuildRequest: %v", err)
		}
		if len(built.Names) != 2 {
			t.Fatalf("expected 2 distinct entries in built.Names, got %+v (body=%s)", built.Names, string(built.Body))
		}
		reqAllowed := *req
		reqAllowed.ToolChoice = ir.ToolChoice{
			Mode:    ir.ToolChoiceAllowed,
			Allowed: []ir.ToolRef{{Name: "read", Namespace: "b"}},
		}
		built2, err := anthropic.BuildRequest(&reqAllowed, anthropic.BuildOptions{})
		if err != nil {
			t.Fatalf("anthropic BuildRequest allowed: %v", err)
		}
		if len(built2.Names) != 1 {
			t.Fatalf("Anthropic allowed_tools must filter strictly by (namespace, name), got %+v (body=%s)", built2.Names, string(built2.Body))
		}
		for _, ref := range built2.Names {
			if ref.Namespace != "b" {
				t.Errorf("wrong namespace allowed: %+v", ref)
			}
		}
	})

	t.Run("gemini_distinct_names_and_reverse_map", func(t *testing.T) {
		built, err := gemini.BuildRequest(req, gemini.BuildOptions{})
		if err != nil {
			t.Fatalf("gemini BuildRequest: %v", err)
		}
		if len(built.Names) != 2 {
			t.Fatalf("expected 2 distinct entries in built.Names, got %+v (body=%s)", built.Names, string(built.Body))
		}
	})
}

// Review2 #14: Freeform custom tools must be wrapped with {input: string} schema
// when building for Anthropic and Gemini.
func TestReview2_P2_14_CustomToolWrapperSchemaAnthropicGemini(t *testing.T) {
	reqBody := []byte(`{
		"model":"m",
		"input":"query",
		"tools":[{"type":"custom","name":"sql","description":"run sql"}]
	}`)
	for _, target := range []ir.Format{ir.FormatAnthropic, ir.FormatGemini} {
		t.Run(string(target), func(t *testing.T) {
			plan, err := bridge.Convert(ir.FormatResponses, target, reqBody, bridge.Options{})
			if err != nil {
				t.Fatalf("Convert to %s: %v", target, err)
			}
			s := string(plan.Body)
			if !strings.Contains(s, `"input"`) || (!strings.Contains(s, `"string"`) && !strings.Contains(s, `"STRING"`)) {
				t.Errorf("%s custom tool missing {input: string} wrapper schema: %s", target, s)
			}
		})
	}
}

// Review2 #15: Chat streaming custom tool encoding produces valid JSON {"input":...},
// and Chat streaming custom tool decoding unwraps {"input":...} incrementally in deltas.
func TestReview2_P2_15_ChatStreamingCustomToolWrapUnwrap(t *testing.T) {
	t.Run("encode_custom_deltas_as_valid_json_object", func(t *testing.T) {
		enc := chat.NewStreamEncoder(chat.EncodeOptions{Env: &ir.Env{}, Model: "gpt-5"})
		var dst []byte
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStart, Model: "gpt-5"})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvToolStart, ID: "c1", Name: "sql", Kind: ir.ToolCustom})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvToolArgs, ID: "c1", Text: "SELECT "})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvToolArgs, ID: "c1", Text: `"1"`})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvToolEnd, ID: "c1"})
		dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStop, Stop: &ir.Stop{Reason: ir.StopToolUse}})
		dst, _ = enc.Close(dst)

		// Re-parse Chat SSE and verify concatenated arguments is valid JSON {"input":"SELECT \"1\""}.
		var argsBuf strings.Builder
		for _, line := range strings.Split(string(dst), "\n") {
			if !strings.HasPrefix(line, "data: ") || strings.TrimSpace(line) == "data: [DONE]" {
				continue
			}
			var ch struct {
				Choices []struct {
					Delta struct {
						ToolCalls []struct {
							Function struct {
								Arguments string `json:"arguments"`
							} `json:"function"`
						} `json:"tool_calls"`
					} `json:"delta"`
				} `json:"choices"`
			}
			if json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &ch) == nil && len(ch.Choices) > 0 {
				for _, tc := range ch.Choices[0].Delta.ToolCalls {
					argsBuf.WriteString(tc.Function.Arguments)
				}
			}
		}
		var obj map[string]string
		if err := json.Unmarshal([]byte(argsBuf.String()), &obj); err != nil {
			t.Fatalf("Chat encoded custom tool arguments are not valid JSON: %q (%v)", argsBuf.String(), err)
		}
		if obj["input"] != `SELECT "1"` {
			t.Errorf("expected unwrapped input SELECT \"1\", got %q", obj["input"])
		}
	})

	t.Run("decode_custom_deltas_unwrapped_consistently", func(t *testing.T) {
		dec := chat.NewStreamDecoder(chat.DecodeOptions{
			Names: map[string]ir.ToolRef{"sql": {Name: "sql", Kind: ir.ToolCustom}},
		})
		chunks := []string{
			`{"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"sql","arguments":"{\"input\":\"SEL"}}]},"finish_reason":null}]}`,
			`{"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ECT \\\"1\\\"\"}"}}]},"finish_reason":null}]}`,
			`{"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		}
		var deltaText strings.Builder
		var endArgs json.RawMessage
		for _, c := range chunks {
			evs, err := dec.Decode(nil, sse.Event{Data: []byte(c)})
			if err != nil {
				t.Fatalf("Decode: %v", err)
			}
			for _, ev := range evs {
				if ev.Type == ir.EvToolArgs {
					deltaText.WriteString(ev.Text)
				}
				if ev.Type == ir.EvToolEnd {
					endArgs = ev.Arguments
				}
			}
		}
		if deltaText.String() != `SELECT "1"` {
			t.Errorf("Chat custom tool deltas must be unwrapped consistently, got %q", deltaText.String())
		}
		var endStr string
		if err := json.Unmarshal(endArgs, &endStr); err != nil || endStr != `SELECT "1"` {
			t.Errorf("Chat custom tool EvToolEnd mismatch: raw=%s err=%v", string(endArgs), err)
		}
	})
}

// Review2 #16: Responses custom tool grammar format must be preserved on build.
func TestReview2_P2_16_ResponsesCustomToolGrammarPreserved(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5",
		"input":"count",
		"tools":[{"type":"custom","name":"digits","format":{"type":"grammar","syntax":"regex","definition":"[0-9]+"}}]
	}`)
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatResponses, body, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	s := string(plan.Body)
	if !strings.Contains(s, `"syntax":"regex"`) || !strings.Contains(s, `"definition":"[0-9]+"`) {
		t.Errorf("Responses custom tool grammar format dropped: %s", s)
	}
}

// Review2 #17: Responses tool_choice.allowed_tools must preserve entry type:"custom".
func TestReview2_P2_17_ResponsesAllowedToolsCustomType(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5",
		"input":"query",
		"tools":[{"type":"custom","name":"sql"}],
		"tool_choice":{"type":"allowed_tools","mode":"auto","tools":[{"type":"custom","name":"sql"}]}
	}`)
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatResponses, body, bridge.Options{})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	var parsed struct {
		ToolChoice struct {
			Type  string `json:"type"`
			Tools []struct {
				Type string `json:"type"`
				Name string `json:"name"`
			} `json:"tools"`
		} `json:"tool_choice"`
	}
	if err := json.Unmarshal(plan.Body, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(parsed.ToolChoice.Tools) != 1 || parsed.ToolChoice.Tools[0].Type != "custom" || parsed.ToolChoice.Tools[0].Name != "sql" {
		t.Errorf("Responses allowed_tools lost custom entry type: %s", string(plan.Body))
	}
}

// Review2 #19: Responses function_call_output must serialize empty output:"" and must not
// emit non-standard is_error field.
func TestReview2_P2_19_ResponsesEmptyOutputAndNoIsError(t *testing.T) {
	req := &ir.Request{
		Model: "gpt-5",
		Messages: []ir.Message{
			{Role: ir.RoleAssistant, Parts: []ir.Part{
				{Type: ir.PartToolCall, Call: &ir.ToolCall{ID: "c1", Name: "noop", Arguments: json.RawMessage(`{}`)}},
				{Type: ir.PartToolCall, Call: &ir.ToolCall{ID: "c2", Name: "fail", Arguments: json.RawMessage(`{}`)}},
			}},
			{Role: ir.RoleUser, Parts: []ir.Part{
				{Type: ir.PartToolResult, Result: &ir.ToolResult{CallID: "c1", Content: []ir.Part{ir.TextPart("")}}},
				{Type: ir.PartToolResult, Result: &ir.ToolResult{CallID: "c2", IsError: true, Content: []ir.Part{ir.TextPart("boom")}}},
			}},
		},
	}
	built, err := responses.BuildRequest(req, responses.BuildOptions{})
	if err != nil {
		t.Fatalf("BuildRequest: %v", err)
	}
	s := string(built.Body)
	if !strings.Contains(s, `"output":""`) {
		t.Errorf("empty function_call_output.output was omitted: %s", s)
	}
	if strings.Contains(s, `"is_error"`) {
		t.Errorf("non-standard is_error emitted on function_call_output: %s", s)
	}
	if !strings.Contains(s, `[tool error] boom`) {
		t.Errorf("expected [tool error] prefix for error tool result: %s", s)
	}
}
