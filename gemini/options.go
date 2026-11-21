package gemini

import "github.com/ihainan/llm-protocol-bridge/ir"

// ParseOptions carries context not in the request body.
type ParseOptions struct {
	Env *ir.Env
	// Model is the upstream model name from the URL.
	Model string
	// Stream indicates whether the request is for streaming (from the URL).
	Stream bool
}

// BuildOptions controls serialisation to Gemini JSON.
type BuildOptions struct {
	Env     *ir.Env
	Dialect Dialect
}

// DecodeOptions controls decoding of an upstream Gemini response.
type DecodeOptions struct {
	Env     *ir.Env
	Dialect Dialect
	Names   map[string]ir.ToolRef
}

// EncodeOptions controls serialisation of an ir.Response to Gemini JSON.
type EncodeOptions struct {
	Env            *ir.Env
	Request        *ir.Request
	Model          string
	PromptEstimate int
	// AltSSE: when true, emit SSE frames; when false, emit a JSON array stream.
	AltSSE bool
}

// Built is the result of BuildRequest.
type Built struct {
	Body   []byte
	Model  string
	Stream bool
	Names  map[string]ir.ToolRef
}
