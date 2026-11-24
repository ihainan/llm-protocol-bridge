// Package chat implements the OpenAI Chat Completions wire format.
package chat

// Dialect describes how an upstream deviates from the standard Chat Completions
// format. The zero value is the unmodified OpenAI specification.
type Dialect struct {
	// MaxTokensField selects which JSON field carries the output token limit.
	// "" or "max_tokens" uses max_tokens; "max_completion_tokens" uses that field.
	MaxTokensField string

	// ReasoningParam describes how reasoning configuration is sent upstream.
	// "" or "reasoning_effort": top-level reasoning_effort string.
	// "enable_thinking": top-level enable_thinking bool + budget_tokens.
	// "thinking": top-level thinking object {type, budget_tokens}.
	// "extra_body": extra_body.google.thinking_config object.
	ReasoningParam string

	// EffortInTemplateKwargs: when true, reasoning_effort is sent as
	// extra_body.template_kwargs.reasoning_effort instead of top-level.
	EffortInTemplateKwargs bool

	// DeveloperRole: when true, the leading system prompt uses role "developer"
	// instead of "system".
	DeveloperRole bool

	// StrictTools: when true, tool definitions include "strict": true.
	StrictTools bool

	// FileParts: when true, file content parts are supported.
	FileParts bool

	// StreamUsage: when true, the request includes stream_options.include_usage.
	StreamUsage bool

	// PromptExcludesCache: when true, prompt_tokens already excludes cached tokens
	// (they are not double-counted). The default (false) subtracts them.
	PromptExcludesCache bool

	// CacheReadFields lists alternative field paths for cached input tokens,
	// used when prompt_tokens_details.cached_tokens is absent or 0. The first
	// positive one wins; it is never added to the standard count. When the
	// cached counts exceed prompt_tokens (Anthropic-style names from relays in
	// front of Claude), prompt_tokens is read as excluding them.
	CacheReadFields []string

	// CacheWriteFields lists field paths for cache-write tokens (first positive wins).
	CacheWriteFields []string

	// ReasoningExcludedFromCompletion: when true, reasoning tokens are not
	// already counted inside completion_tokens and must be added separately.
	ReasoningExcludedFromCompletion bool

	// ThinkTags: when true, extract <think>...</think> from output content as
	// reasoning text (Markdown-aware; nested tags are not stripped).
	ThinkTags bool

	// SignaturesInExtraContent: when true, thought signatures arrive as separate
	// content parts of kind "thought" or similar.
	SignaturesInExtraContent bool

	// MidSystem controls system/developer messages that appear in the middle of
	// the conversation (after the first user message):
	// "" or "keep": passed through in place; "fold": wrapped in
	// <system_message>...</system_message> and folded into adjacent user turns.
	MidSystem string

	// ReasoningReplay controls how reasoning is replayed in tool-call rounds.
	ReasoningReplay ReasoningReplay

	// DisableWebSearch: when true, live server-side web_search builtin tools
	// (external_web_access != false) are omitted and recorded via ir.LossWebSearch
	// instead of failing with unsupported. Cached web_search (external_web_access: false)
	// continues to be dropped via ir.LossCachedWebSearch.
	DisableWebSearch bool
}

// ReasoningReplay controls how a previous reasoning block is re-sent to the
// upstream in a tool-call follow-up turn.
type ReasoningReplay struct {
	// Field is the message field name to use for reasoning replay (e.g. "reasoning"
	// or "reasoning_content"). Empty selects based on ReasoningParam.
	Field string
	// Placeholder is a fixed string to send instead of the actual text.
	Placeholder string
	// MaxChars limits how many bytes of reasoning text are replayed; 0 = unlimited.
	MaxChars int
	// MaxItems limits how many reasoning blocks are replayed; 0 = unlimited.
	MaxItems int
	// RequireVerified, when true, replays only reasoning blocks whose provenance
	// signature was verified for the Chat upstream (Origin.Matches(FormatChat, ProviderID)),
	// ignoring unsigned or foreign-provider reasoning blocks.
	RequireVerified bool
}
