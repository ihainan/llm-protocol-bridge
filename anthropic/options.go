package anthropic

import "github.com/ihainan/llm-protocol-bridge/ir"

// ParseOptions carries context that is not in the request body.
type ParseOptions struct {
	Env *ir.Env
	// Variant is the endpoint variant: "" (messages) or "count_tokens".
	Variant string
	// AnthropicVersion is the value of the anthropic-version header.
	// An empty string means the header was absent.
	AnthropicVersion string
}

// BuildOptions controls how an ir.Request is serialised to Anthropic Messages JSON.
type BuildOptions struct {
	Env     *ir.Env
	Dialect Dialect
}

// DecodeOptions controls how an upstream Anthropic response is decoded.
type DecodeOptions struct {
	Env     *ir.Env
	Dialect Dialect
	// Names maps sanitised upstream tool names back to their original IR identifiers.
	Names map[string]ir.ToolRef
	// MaxBufferedBytes caps buffered server_tool_use JSON arguments (default 8 MiB).
	MaxBufferedBytes int
}

// EncodeOptions controls how an ir.Response is serialised to Anthropic Messages JSON.
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
