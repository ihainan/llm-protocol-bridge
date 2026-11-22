package chat

import "encoding/json"

// Wire types for the Chat Completions request and response bodies.
// These are internal to the package; callers use ir types.

// wireMessage is one conversation turn in a Chat Completions request or response.
type wireMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content,omitempty"` // string | []wireContentPart | null
	Name       string          `json:"name,omitempty"`
	ToolCalls  []wireToolCall  `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Refusal    string          `json:"refusal,omitempty"`

	// Reasoning fields (dialect-dependent)
	Reasoning        string `json:"reasoning,omitempty"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

// wireContentPart is one element of a multi-part message content array.
type wireContentPart struct {
	Type       string        `json:"type"`
	Text       string        `json:"text,omitempty"`
	ImageURL   *wireImageURL `json:"image_url,omitempty"`
	File       *wireFilePart `json:"file,omitempty"`
	InputAudio *wireAudio    `json:"input_audio,omitempty"`
	Refusal    string        `json:"refusal,omitempty"`
	// Reasoning / thinking (some providers)
	Thinking  string `json:"thinking,omitempty"`
	Reasoning string `json:"reasoning,omitempty"`
	Signature string `json:"signature,omitempty"`
}

type wireImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type wireFilePart struct {
	FileID   string `json:"file_id,omitempty"`
	Filename string `json:"filename,omitempty"`
	FileData string `json:"file_data,omitempty"` // base64
}

type wireAudio struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}

// wireTool is a tool definition in a Chat Completions request.
type wireTool struct {
	Type     string          `json:"type"`
	Function wireFunctionDef `json:"function"`
}

type wireFunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
}

// wireToolCall is a tool call in an assistant message or streaming delta.
type wireToolCall struct {
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function wireFunctionCall `json:"function"`
	Index    *int             `json:"index,omitempty"` // streaming only
}

type wireFunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// wireToolChoice is the structured form of tool_choice.
type wireToolChoice struct {
	Type     string              `json:"type"`
	Function *wireToolChoiceFunc `json:"function,omitempty"`
}

type wireToolChoiceFunc struct {
	Name string `json:"name"`
}

// wireResponseFormat is the response_format object.
type wireResponseFormat struct {
	Type       string          `json:"type"`
	JSONSchema *wireJSONSchema `json:"json_schema,omitempty"`
}

type wireJSONSchema struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Schema      json.RawMessage `json:"schema,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
}

// wireStreamOptions controls stream_options.
type wireStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// wireThinking is the thinking object for some Chat dialects.
type wireThinking struct {
	Type         string `json:"type,omitempty"` // "enabled" | "disabled"
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

// wireResponse is the non-streaming Chat Completions response body.
type wireResponse struct {
	ID                string          `json:"id"`
	Object            string          `json:"object"`
	Created           int64           `json:"created"`
	Model             string          `json:"model"`
	Choices           []wireChoice    `json:"choices"`
	Usage             json.RawMessage `json:"usage,omitempty"`
	SystemFingerprint string          `json:"system_fingerprint,omitempty"`
}

// wireChunk is one streaming chunk.
type wireChunk struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []wireStreamChoice `json:"choices"`
	Usage   json.RawMessage    `json:"usage,omitempty"`
	// Bare error from some providers
	Error *wireError `json:"error,omitempty"`
}

// wireChoice is one candidate in a non-streaming response.
type wireChoice struct {
	Index        int         `json:"index"`
	Message      wireMessage `json:"message"`
	FinishReason string      `json:"finish_reason,omitempty"`
}

// wireStreamChoice is one candidate delta in a streaming chunk.
type wireStreamChoice struct {
	Index        int         `json:"index"`
	Delta        wireMessage `json:"delta"`
	FinishReason string      `json:"finish_reason,omitempty"`
}

// wireUsage is the usage object in a response.
type wireUsage struct {
	PromptTokens            int                          `json:"prompt_tokens"`
	CompletionTokens        int                          `json:"completion_tokens"`
	TotalTokens             int                          `json:"total_tokens"`
	PromptTokensDetails     *wirePromptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *wireCompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

type wirePromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type wireCompletionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

// wireError is the error object inside a Chat Completions error response.
type wireError struct {
	Message string          `json:"message"`
	Type    string          `json:"type,omitempty"`
	Code    json.RawMessage `json:"code,omitempty"` // string or null
	Param   string          `json:"param,omitempty"`
}

// wireErrorResponse wraps wireError in the top-level shape.
type wireErrorResponse struct {
	Error wireError `json:"error"`
}
