package chat_test

import (
	"encoding/json"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/chat"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

func defaultEnv(style ir.PathStyle) *ir.Env {
	return &ir.Env{Allow: ir.DefaultAllowed, Style: style}
}

// TestParseBuildRoundTrip parses several Chat request bodies and rebuilds them,
// checking that semantic fields survive the round-trip.
func TestParseBuildRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{
			name: "simple",
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			name: "system_user",
			body: `{"model":"m","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hello"}],"temperature":0.5}`,
		},
		{
			name: "tool_call_round",
			body: `{"model":"m","messages":[{"role":"user","content":"read it"},{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"Read","arguments":"{\"path\":\"/x\"}"}}]},{"role":"tool","tool_call_id":"c1","content":"done"}],"tools":[{"type":"function","function":{"name":"Read","description":"Reads","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}]}`,
		},
		{
			name: "image_url",
			body: `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"data:image/png;base64,abc="}}]}]}`,
		},
		{
			name: "stop_sequences",
			body: `{"model":"m","messages":[{"role":"user","content":"hi"}],"stop":["\n\n","END"]}`,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			env := defaultEnv(ir.StyleBracket)
			req, err := chat.ParseRequest([]byte(tc.body), chat.ParseOptions{Env: env})
			if err != nil {
				t.Fatalf("ParseRequest: %v", err)
			}
			if req.Model == "" && tc.name != "no_model" {
				// model may be empty for Gemini-style but Chat requires it
			}

			built, err := chat.BuildRequest(req, chat.BuildOptions{Env: defaultEnv(ir.StyleBracket)})
			if err != nil {
				t.Fatalf("BuildRequest: %v", err)
			}

			// Rebuild must produce valid JSON.
			var v any
			if err := json.Unmarshal(built.Body, &v); err != nil {
				t.Fatalf("BuildRequest output is not valid JSON: %v\n%s", err, built.Body)
			}

			// Re-parse the rebuilt body and check key fields survived.
			req2, err := chat.ParseRequest(built.Body, chat.ParseOptions{Env: defaultEnv(ir.StyleBracket)})
			if err != nil {
				t.Fatalf("re-parse after build: %v\nbody: %s", err, built.Body)
			}
			if req.Model != req2.Model {
				t.Errorf("model: want %q, got %q", req.Model, req2.Model)
			}
			if len(req.Messages) != len(req2.Messages) {
				t.Errorf("messages count: want %d, got %d", len(req.Messages), len(req2.Messages))
			}
			if len(req.Tools) != len(req2.Tools) {
				t.Errorf("tools count: want %d, got %d", len(req.Tools), len(req2.Tools))
			}
		})
	}
}

// TestParseError checks that an error body produces an ir.APIError.
func TestParseError(t *testing.T) {
	body := []byte(`{"error":{"type":"invalid_request_error","message":"bad","code":"bad_code"}}`)
	apiErr := chat.ParseError(400, body)
	if apiErr == nil {
		t.Fatal("expected non-nil APIError")
	}
	if apiErr.Message != "bad" {
		t.Errorf("message: want %q, got %q", "bad", apiErr.Message)
	}
}

// TestBuildRequestStreamOptions checks that streaming requests include stream_options.
func TestBuildRequestStreamOptions(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}`)
	req, err := chat.ParseRequest(body, chat.ParseOptions{Env: defaultEnv(ir.StyleBracket)})
	if err != nil {
		t.Fatalf("ParseRequest: %v", err)
	}
	built, err := chat.BuildRequest(req, chat.BuildOptions{Env: defaultEnv(ir.StyleBracket)})
	if err != nil {
		t.Fatalf("BuildRequest: %v", err)
	}
	var v map[string]json.RawMessage
	if err := json.Unmarshal(built.Body, &v); err != nil {
		t.Fatalf("not valid JSON: %v", err)
	}
	if _, ok := v["stream_options"]; !ok {
		t.Errorf("stream:true request is missing stream_options\n%s", built.Body)
	}
	if built.Stream != true {
		t.Errorf("Built.Stream should be true")
	}
}
