package anthropic

import "encoding/json"

// Wire types for the Anthropic Messages request and response.

type wireMetadata struct {
	UserID string `json:"user_id,omitempty"`
}

type wireThinkingConfig struct {
	Type         string `json:"type"` // "enabled" | "disabled"
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

// wireMessage is one conversation turn.
type wireMessage struct {
	Role    string          `json:"role"`    // user | assistant
	Content json.RawMessage `json:"content"` // string | []wireContentBlock
}

// wireContentBlock is one block of content.
type wireContentBlock struct {
	Type string `json:"type"`
	// text
	Text      string            `json:"text,omitempty"`
	Citations []json.RawMessage `json:"citations,omitempty"`
	// image
	Source *wireImageSource `json:"source,omitempty"`
	// document
	Title   string `json:"title,omitempty"`
	Context string `json:"context,omitempty"`
	// tool_use
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
	// tool_result
	ToolUseID string          `json:"tool_use_id,omitempty"`
	IsError   *bool           `json:"is_error,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"` // string | []wireContentBlock
	// thinking / redacted_thinking
	Thinking  string  `json:"thinking,omitempty"`
	Signature *string `json:"signature,omitempty"`
	Data      string  `json:"data,omitempty"`
	// cache_control
	CacheControl *wireCacheControl `json:"cache_control,omitempty"`
}

type wireImageSource struct {
	Type      string `json:"type"` // base64 | url | file
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
	FileID    string `json:"file_id,omitempty"`
}

type wireCacheControl struct {
	Type string `json:"type"`          // ephemeral
	TTL  string `json:"ttl,omitempty"` // 5m | 1h
}

// wireTool is a tool definition.
type wireTool struct {
	Name         string            `json:"name"`
	Description  string            `json:"description,omitempty"`
	InputSchema  json.RawMessage   `json:"input_schema"`
	Strict       *bool             `json:"strict,omitempty"`
	Type         string            `json:"type,omitempty"` // "custom" for custom tools
	CacheControl *wireCacheControl `json:"cache_control,omitempty"`
}

// wireToolChoice shapes.
type wireToolChoiceAuto struct {
	Type                   string `json:"type"` // "auto"
	DisableParallelToolUse bool   `json:"disable_parallel_tool_use,omitempty"`
}
type wireToolChoiceAny struct {
	Type                   string `json:"type"` // "any"
	DisableParallelToolUse bool   `json:"disable_parallel_tool_use,omitempty"`
}
type wireToolChoiceTool struct {
	Type                   string `json:"type"` // "tool"
	Name                   string `json:"name"`
	DisableParallelToolUse bool   `json:"disable_parallel_tool_use,omitempty"`
}

// wireResponse is the non-streaming response body.
type wireResponse struct {
	ID           string             `json:"id"`
	Type         string             `json:"type"` // "message"
	Role         string             `json:"role"`
	Model        string             `json:"model"`
	Content      []wireContentBlock `json:"content"`
	StopReason   string             `json:"stop_reason,omitempty"`
	StopSequence string             `json:"stop_sequence,omitempty"`
	Usage        *wireUsage         `json:"usage,omitempty"`
}

// wireUsage is the usage object. Pointer fields distinguish absent keys
// (such as input_tokens omitted in message_delta) from explicit zeroes.
type wireUsage struct {
	InputTokens              *int                     `json:"input_tokens,omitempty"`
	OutputTokens             *int                     `json:"output_tokens,omitempty"`
	CacheReadInputTokens     *int                     `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens *int                     `json:"cache_creation_input_tokens,omitempty"`
	CacheCreation            *wireCacheCreation       `json:"cache_creation,omitempty"`
	ServerToolUse            *wireServerToolUse       `json:"server_tool_use,omitempty"`
	OutputTokensDetails      *wireOutputTokensDetails `json:"output_tokens_details,omitempty"`
}

type wireOutputTokensDetails struct {
	ThinkingTokens *int `json:"thinking_tokens,omitempty"`
}

type wireCacheCreation struct {
	Ephemeral5mInputTokens int `json:"ephemeral_5m_input_tokens,omitempty"`
	Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens,omitempty"`
}

type wireServerToolUse struct {
	WebSearchRequests int `json:"web_search_requests,omitempty"`
	WebFetchRequests  int `json:"web_fetch_requests,omitempty"`
}

// wireStreamEvent is one SSE event in a streamed Anthropic response.
// The type field selects which other fields are present.
type wireStreamEvent struct {
	Type  string     `json:"type"`
	Index int        `json:"index"`
	Delta *wireDelta `json:"delta,omitempty"`
	// message_start
	Message *wireResponse `json:"message,omitempty"`
	// content_block_start
	ContentBlock *wireContentBlock `json:"content_block,omitempty"`
	// message_delta
	Usage *wireUsage `json:"usage,omitempty"`
	// error
	Error *wireAPIError `json:"error,omitempty"`
}

type wireDelta struct {
	Type         string          `json:"type,omitempty"`
	Text         string          `json:"text,omitempty"`
	Thinking     string          `json:"thinking,omitempty"`
	Signature    string          `json:"signature,omitempty"`
	PartialJSON  string          `json:"partial_json,omitempty"`
	Citation     json.RawMessage `json:"citation,omitempty"`
	StopReason   string          `json:"stop_reason,omitempty"`
	StopSequence string          `json:"stop_sequence,omitempty"`
}

// wireAPIError is the error object in a streaming event or top-level error body.
type wireAPIError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// wireErrorResponse wraps wireAPIError at the top level.
type wireErrorResponse struct {
	Type  string       `json:"type"` // "error"
	Error wireAPIError `json:"error"`
}
