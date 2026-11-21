package anthropic_test

import (
	"encoding/json"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/anthropic"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

func defaultEnv(style ir.PathStyle) *ir.Env {
	return &ir.Env{Allow: ir.DefaultAllowed, Style: style}
}

// TestParseBuildRoundTrip parses Anthropic request bodies and rebuilds them.
func TestParseBuildRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{
			name: "simple",
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			name: "system_text",
			body: `{"model":"m","max_tokens":64,"system":"be brief","messages":[{"role":"user","content":"hello"}],"temperature":0.7}`,
		},
		{
			name: "tool_call_round",
			body: `{"model":"m","max_tokens":64,"tools":[{"name":"Read","description":"Reads","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}],"messages":[{"role":"user","content":"read it"},{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Read","input":{"path":"/x"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"done"}]}]}`,
		},
		{
			name: "thinking",
			body: `{"model":"m","max_tokens":2048,"thinking":{"type":"enabled","budget_tokens":1024},"messages":[{"role":"user","content":"think"}]}`,
		},
		{
			name: "image",
			body: `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"abc="}},{"type":"text","text":"what?"}]}]}`,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			env := defaultEnv(ir.StyleDot)
			req, err := anthropic.ParseRequest([]byte(tc.body), anthropic.ParseOptions{Env: env})
			if err != nil {
				t.Fatalf("ParseRequest: %v", err)
			}

			built, err := anthropic.BuildRequest(req, anthropic.BuildOptions{Env: defaultEnv(ir.StyleDot)})
			if err != nil {
				t.Fatalf("BuildRequest: %v", err)
			}

			var v any
			if err := json.Unmarshal(built.Body, &v); err != nil {
				t.Fatalf("BuildRequest output is not valid JSON: %v\n%s", err, built.Body)
			}

			req2, err := anthropic.ParseRequest(built.Body, anthropic.ParseOptions{Env: defaultEnv(ir.StyleDot)})
			if err != nil {
				t.Fatalf("re-parse after build: %v\nbody: %s", err, built.Body)
			}
			if req.Model != req2.Model {
				t.Errorf("model: want %q, got %q", req.Model, req2.Model)
			}
			if len(req.Messages) != len(req2.Messages) {
				t.Errorf("messages count: want %d, got %d", len(req.Messages), len(req2.Messages))
			}
		})
	}
}

// TestParseError verifies Anthropic error parsing.
func TestParseError(t *testing.T) {
	body := []byte(`{"type":"error","error":{"type":"invalid_request_error","message":"bad param"}}`)
	apiErr := anthropic.ParseError(400, body)
	if apiErr == nil {
		t.Fatal("expected non-nil APIError")
	}
	if apiErr.Message != "bad param" {
		t.Errorf("message: want %q, got %q", "bad param", apiErr.Message)
	}
}

// TestBuildRequestMaxTokensDefault checks that max_tokens gets a default when absent.
func TestBuildRequestMaxTokensDefault(t *testing.T) {
	req := &ir.Request{
		Model:    "claude-3",
		Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("hi")}}},
	}
	built, err := anthropic.BuildRequest(req, anthropic.BuildOptions{Env: defaultEnv(ir.StyleDot)})
	if err != nil {
		t.Fatalf("BuildRequest: %v", err)
	}
	var v map[string]json.RawMessage
	if err := json.Unmarshal(built.Body, &v); err != nil {
		t.Fatalf("not valid JSON: %v", err)
	}
	if _, ok := v["max_tokens"]; !ok {
		t.Error("max_tokens missing from output")
	}
}
