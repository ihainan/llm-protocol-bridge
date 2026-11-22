package chat

import "github.com/ihainan/llm-protocol-bridge/ir"

// ParseOptions carries context that is not in the request body.
type ParseOptions struct {
	Env *ir.Env
}

// BuildOptions controls how an ir.Request is serialised to Chat Completions JSON.
type BuildOptions struct {
	Env     *ir.Env
	Dialect Dialect
	// OmitUserID prevents forwarding Request.User (e.g., Anthropic metadata.user_id)
	// into the upstream Chat Completions "user" field.
	OmitUserID bool
	// OmitHistoryReasoning prevents forwarding historical assistant reasoning
	// blocks into upstream reasoning_content / reasoning message fields.
	OmitHistoryReasoning bool
}

// DecodeOptions controls how an upstream Chat Completions response is decoded.
type DecodeOptions struct {
	Env     *ir.Env
	Dialect Dialect
	// Names maps sanitised upstream tool names back to their original IR identifiers.
	// Populated from Built.Names after BuildRequest.
	Names map[string]ir.ToolRef
}

// EncodeOptions controls how an ir.Response is serialised to Chat Completions JSON.
type EncodeOptions struct {
	Env            *ir.Env
	Request        *ir.Request
	Model          string
	PromptEstimate int
}

// Built is the result of BuildRequest.
type Built struct {
	// Body is the JSON request body.
	Body []byte
	// Model is the upstream model name (extracted from ir.Request.Model).
	Model string
	// Stream is whether the request asks for streaming.
	Stream bool
	// Names maps each sanitised Chat tool name back to the original ir.ToolRef so
	// the response decoder can reverse the sanitisation.
	Names map[string]ir.ToolRef
}
