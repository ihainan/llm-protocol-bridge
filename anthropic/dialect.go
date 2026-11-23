// Package anthropic implements the Anthropic Messages wire format.
package anthropic

// Dialect describes how an upstream deviates from the standard Anthropic Messages API.
type Dialect struct {
	// MidSystem controls how system/developer messages that appear in the middle
	// of a conversation are handled:
	// "" or "keep": passed through; "fold": merged into the nearest assistant turn.
	MidSystem string

	// AutoCache: when true, the builder inserts cache_control breakpoints
	// automatically on instructions and the last few tool definitions.
	AutoCache bool

	// StrictTools: when true, tool definitions include "disable_parallel_tool_use": true.
	StrictTools bool

	// DefaultMaxTokens is used when ir.Request.MaxOutputTokens is nil.
	// 0 means the downstream will error (Anthropic requires max_tokens).
	DefaultMaxTokens int

	// DisableWebSearch: when true, server-side web_search builtin tools are
	// omitted and recorded via ir.LossWebSearch instead of being emitted as
	// web_search_20250305 (for Anthropic-compatible upstreams such as AWS
	// Bedrock that reject server-side web search tools).
	DisableWebSearch bool
}
