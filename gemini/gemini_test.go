package gemini_test

import (
	"encoding/json"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/gemini"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

func defaultEnv(style ir.PathStyle) *ir.Env {
	return &ir.Env{Allow: ir.DefaultAllowed, Style: style}
}

func TestParseBuildRoundTrip(t *testing.T) {
	cases := []struct {
		name  string
		body  string
		model string
	}{
		{
			name:  "simple",
			body:  `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`,
			model: "gemini-1.5-pro",
		},
		{
			name:  "system_instruction",
			body:  `{"systemInstruction":{"parts":[{"text":"be brief"}]},"contents":[{"role":"user","parts":[{"text":"hello"}]}]}`,
			model: "gemini-2.0",
		},
		{
			name:  "tool_call_round",
			body:  `{"tools":[{"functionDeclarations":[{"name":"Read","description":"Reads","parameters":{"type":"OBJECT","properties":{"path":{"type":"STRING"}}}}]}],"contents":[{"role":"user","parts":[{"text":"read it"}]},{"role":"model","parts":[{"functionCall":{"name":"Read","args":{"path":"/x"}}}]},{"role":"user","parts":[{"functionResponse":{"name":"Read","response":{"result":"done"}}}]}]}`,
			model: "gemini-2.0",
		},
		{
			name:  "image",
			body:  `{"contents":[{"role":"user","parts":[{"text":"what?"},{"inlineData":{"mimeType":"image/png","data":"abc="}}]}]}`,
			model: "gemini-1.5-pro",
		},
		{
			name:  "generation_config",
			body:  `{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"temperature":0.7,"maxOutputTokens":64}}`,
			model: "gemini-2.0",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			env := defaultEnv(ir.StyleBracket)
			req, err := gemini.ParseRequest([]byte(tc.body), gemini.ParseOptions{Env: env, Model: tc.model})
			if err != nil {
				t.Fatalf("ParseRequest: %v", err)
			}

			built, err := gemini.BuildRequest(req, gemini.BuildOptions{Env: defaultEnv(ir.StyleBracket)})
			if err != nil {
				t.Fatalf("BuildRequest: %v", err)
			}

			var v any
			if err := json.Unmarshal(built.Body, &v); err != nil {
				t.Fatalf("BuildRequest output not valid JSON: %v\n%s", err, built.Body)
			}

			req2, err := gemini.ParseRequest(built.Body, gemini.ParseOptions{Env: defaultEnv(ir.StyleBracket), Model: tc.model})
			if err != nil {
				t.Fatalf("re-parse after build: %v\nbody: %s", err, built.Body)
			}
			if len(req.Messages) != len(req2.Messages) {
				t.Errorf("messages count: want %d, got %d", len(req.Messages), len(req2.Messages))
			}
			if len(req.Instructions) != len(req2.Instructions) {
				t.Errorf("instructions count: want %d, got %d", len(req.Instructions), len(req2.Instructions))
			}
			if (req.Temperature == nil) != (req2.Temperature == nil) {
				t.Errorf("temperature mismatch: %v vs %v", req.Temperature, req2.Temperature)
			}
			if (req.MaxOutputTokens == nil) != (req2.MaxOutputTokens == nil) {
				t.Errorf("maxOutputTokens mismatch: %v vs %v", req.MaxOutputTokens, req2.MaxOutputTokens)
			}
		})
	}
}

func TestParseError(t *testing.T) {
	body := []byte(`{"error":{"code":400,"message":"bad request","status":"INVALID_ARGUMENT"}}`)
	apiErr := gemini.ParseError(400, body)
	if apiErr == nil {
		t.Fatal("expected non-nil APIError")
	}
	if apiErr.Message != "bad request" {
		t.Errorf("message: want %q, got %q", "bad request", apiErr.Message)
	}
}
