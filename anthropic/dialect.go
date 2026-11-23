// Package anthropic implements the Anthropic Messages wire format.
package anthropic

// Dialect describes how an upstream deviates from the standard Anthropic Messages API.
type Dialect struct {
	// MidSystem controls how system/developer messages that appear in the middle
	// of a conversation are handled:
	// "" or "keep": passed through; "fold": merged into the nearest assistant turn.
	MidSystem string

	// AutoCache selects where the builder places prompt cache breakpoints
	// (cache_control) when the request carries none of its own. The zero
	// value (AutoCacheOff) adds nothing. A request that carries any client
	// cache marker is never modified by AutoCache. See AutoCacheMode.
	AutoCache AutoCacheMode

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

// AutoCacheMode is a prompt cache placement strategy for Dialect.AutoCache.
//
// Anthropic caches the request prefix that ends at each cache_control
// breakpoint (render order: tools, system, messages) and allows at most four
// breakpoints per request. Writing a prefix to the cache costs more than plain
// input (1.25x for the default 5-minute TTL) and reading it back costs a small
// fraction (0.1x on most models), so a breakpoint pays off only when the same
// prefix is sent again before it expires. Prefixes shorter than the model's
// minimum cacheable length are not cached and are not charged as writes.
// Automatic breakpoints always use the default 5-minute TTL.
type AutoCacheMode string

const (
	// AutoCacheOff adds no breakpoints.
	AutoCacheOff AutoCacheMode = ""
	// AutoCachePrefix marks the stable prefix only: the last tool definition
	// and the last system block. Conversation turns are never written to the
	// cache, so one-off requests pay the write premium only on the system
	// prompt and tools, which repeat across requests that share them.
	AutoCachePrefix AutoCacheMode = "prefix"
	// AutoCacheConversation marks the stable prefix plus the conversation:
	// the last cacheable block of the final message and of the user message
	// before it. Each multi-turn or agent-loop request then reads the previous
	// turn from the cache, at the cost of writing every new turn.
	AutoCacheConversation AutoCacheMode = "conversation"
)

// Valid reports whether m is a known AutoCacheMode.
func (m AutoCacheMode) Valid() bool {
	switch m {
	case AutoCacheOff, AutoCachePrefix, AutoCacheConversation:
		return true
	}
	return false
}
