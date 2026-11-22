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
	// OmitUserID prevents forwarding Request.User into the upstream Responses "user" field.
	OmitUserID bool
	// OmitHistoryReasoning prevents forwarding historical assistant reasoning
	// blocks into upstream Responses input reasoning items.
	OmitHistoryReasoning bool
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
	// MaxBufferedBytes caps deferred multi-candidate events and queued tool
	// buffers inside StreamEncoder (0 = 8 MiB).
	MaxBufferedBytes int
	// GenerateResponseID forces a fresh resp_ response ID instead of reusing the
	// upstream response ID.
	GenerateResponseID bool
	// RawReasoningAsSummary controls whether raw upstream reasoning text is
	// emitted as summary_text when the client requested reasoning.summary.
	// Nil defaults to true; false always emits reasoning_text for raw reasoning.
	RawReasoningAsSummary *bool
	// TrimLeadingNewlines, when true, strips leading \r and \n characters before
	// opening a Responses message item (and omits newline-only messages).
	// Default false preserves model output verbatim.
	TrimLeadingNewlines bool
	// SealUnsignedReasoning, when true and the client requested
	// "reasoning.encrypted_content" in include, seals unsigned reasoning text
	// into encrypted_content using Env.Sealer().
	SealUnsignedReasoning bool
	// UnsignedOrigin is the provenance Origin recorded when SealUnsignedReasoning
	// seals an unsigned reasoning block (defaults to FormatChat + Env.ProviderID).
	UnsignedOrigin ir.Origin
}

// Built is the result of BuildRequest.
type Built struct {
	Body   []byte
	Model  string
	Stream bool
	Names  map[string]ir.ToolRef
}
