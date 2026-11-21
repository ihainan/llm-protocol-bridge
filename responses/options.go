package responses

import "github.com/ihainan/llm-protocol-bridge/ir"

// ParseOptions carries context not in the request body.
type ParseOptions struct {
	Env *ir.Env
}

// BuildOptions controls serialisation to Responses JSON.
type BuildOptions struct {
	Env     *ir.Env
	Dialect Dialect
}

// DecodeOptions controls decoding of an upstream Responses response.
type DecodeOptions struct {
	Env     *ir.Env
	Dialect Dialect
	Names   map[string]ir.ToolRef
}

// EncodeOptions controls serialisation of an ir.Response to Responses JSON.
type EncodeOptions struct {
	Env            *ir.Env
	Request        *ir.Request
	Model          string
	PromptEstimate int
}

// Built is the result of BuildRequest.
type Built struct {
	Body   []byte
	Model  string
	Stream bool
	Names  map[string]ir.ToolRef
}
