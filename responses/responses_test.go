package responses_test

import (
	"encoding/json"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/responses"
)

func defaultEnv(style ir.PathStyle) *ir.Env {
	return &ir.Env{Allow: ir.DefaultAllowed, Style: style}
}

func TestParseBuildRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{
			name: "simple_string",
			body: `{"model":"m","input":"hi"}`,
		},
		{
			name: "simple_message",
			body: `{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}]}`,
		},
		{
			name: "with_instructions",
			body: `{"model":"m","instructions":"be brief","input":"hello"}`,
		},
		{
			name: "tool_call_round",
			body: `{"model":"m","tools":[{"type":"function","name":"Read","description":"Reads","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}],"input":[{"type":"message","role":"user","content":"read it"},{"type":"function_call","call_id":"c1","name":"Read","arguments":"{\"path\":\"/x\"}"},{"type":"function_call_output","call_id":"c1","output":"done"}]}`,
		},
		{
			name: "reasoning",
			body: `{"model":"m","reasoning":{"effort":"medium","summary":"auto"},"input":"think about this"}`,
		},
		{
			name: "image_input",
			body: `{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"look"},{"type":"input_image","image_url":"data:image/png;base64,abc="}]}]}`,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			env := defaultEnv(ir.StyleBracket)
			req, err := responses.ParseRequest([]byte(tc.body), responses.ParseOptions{Env: env})
			if err != nil {
				t.Fatalf("ParseRequest: %v", err)
			}

			built, err := responses.BuildRequest(req, responses.BuildOptions{Env: defaultEnv(ir.StyleBracket)})
			if err != nil {
				t.Fatalf("BuildRequest: %v", err)
			}

			var v any
			if err := json.Unmarshal(built.Body, &v); err != nil {
				t.Fatalf("BuildRequest output not valid JSON: %v\n%s", err, built.Body)
			}

			req2, err := responses.ParseRequest(built.Body, responses.ParseOptions{Env: defaultEnv(ir.StyleBracket)})
			if err != nil {
				t.Fatalf("re-parse after build: %v\nbody: %s", err, built.Body)
			}
			if req.Model != req2.Model {
				t.Errorf("model: want %q, got %q", req.Model, req2.Model)
			}
		})
	}
}

func TestParseError(t *testing.T) {
	body := []byte(`{"error":{"code":"invalid_request","message":"bad input"}}`)
	apiErr := responses.ParseError(400, body)
	if apiErr == nil {
		t.Fatal("expected non-nil APIError")
	}
	if apiErr.Message != "bad input" {
		t.Errorf("message: want %q, got %q", "bad input", apiErr.Message)
	}
}

func TestBuildRequestStream(t *testing.T) {
	body := []byte(`{"model":"m","input":"hi","stream":true}`)
	req, err := responses.ParseRequest(body, responses.ParseOptions{Env: defaultEnv(ir.StyleBracket)})
	if err != nil {
		t.Fatalf("ParseRequest: %v", err)
	}
	built, err := responses.BuildRequest(req, responses.BuildOptions{Env: defaultEnv(ir.StyleBracket)})
	if err != nil {
		t.Fatalf("BuildRequest: %v", err)
	}
	if !built.Stream {
		t.Error("Built.Stream should be true")
	}
	var v map[string]json.RawMessage
	if err := json.Unmarshal(built.Body, &v); err != nil {
		t.Fatalf("not valid JSON: %v", err)
	}
	if _, ok := v["stream"]; !ok {
		t.Error("stream field missing from output")
	}
}
