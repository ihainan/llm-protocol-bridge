package responses

import "encoding/json"

// Wire types for the OpenAI Responses API.

// wireRequest is the JSON body for POST /v1/responses.
type wireRequest struct {
	Model              string          `json:"model"`
	Input              json.RawMessage `json:"input,omitempty"` // string or []wireInputItem
	Instructions       string          `json:"instructions,omitempty"`
	Tools              []wireTool      `json:"tools,omitempty"`
	ToolChoice         json.RawMessage `json:"tool_choice,omitempty"`
	ParallelToolCalls  *bool           `json:"parallel_tool_calls,omitempty"`
	MaxOutputTokens    *int            `json:"max_output_tokens,omitempty"`
	Temperature        *float64        `json:"temperature,omitempty"`
	TopP               *float64        `json:"top_p,omitempty"`
	Stop               json.RawMessage `json:"stop,omitempty"` // string or []string
	Seed               *int64          `json:"seed,omitempty"`
	PresencePenalty    *float64        `json:"presence_penalty,omitempty"`
	FrequencyPenalty   *float64        `json:"frequency_penalty,omitempty"`
	Reasoning          *wireReasoning  `json:"reasoning,omitempty"`
	Text               *wireTextFormat `json:"text,omitempty"`
	Include            []string        `json:"include,omitempty"`
	Store              *bool           `json:"store,omitempty"`
	ServiceTier        string          `json:"service_tier,omitempty"`
	User               string          `json:"user,omitempty"`
	Metadata           json.RawMessage `json:"metadata,omitempty"`
	Stream             *bool           `json:"stream,omitempty"`
	PreviousResponseID string          `json:"previous_response_id,omitempty"`
}

// wireInputItem is one item in the input array.
type wireInputItem struct {
	Type string `json:"type"`
	// message fields
	Role    string          `json:"role,omitempty"`
	Content json.RawMessage `json:"content,omitempty"` // string or []wireContentPart
	Name    string          `json:"name,omitempty"`
	// function_call fields
	CallID    string `json:"call_id,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	// function_call_output fields
	Output  string `json:"output,omitempty"`
	IsError *bool  `json:"is_error,omitempty"`
	// reasoning fields
	ID               string          `json:"id,omitempty"`
	Summary          json.RawMessage `json:"summary,omitempty"`
	EncryptedContent string          `json:"encrypted_content,omitempty"`
	// custom_tool_call fields
	Namespace string `json:"namespace,omitempty"`
	Input     string `json:"input,omitempty"`
	// custom_tool_call_output fields
	// (uses Output above)
}

// wireContentPart is one part of a message content array.
type wireContentPart struct {
	Type string `json:"type"`
	// input_text / output_text
	Text string `json:"text,omitempty"`
	// input_image
	ImageURL string `json:"image_url,omitempty"`
	Detail   string `json:"detail,omitempty"`
	// input_file
	FileID   string `json:"file_id,omitempty"`
	Filename string `json:"filename,omitempty"`
	FileData string `json:"file_data,omitempty"` // base64
	// refusal
	Refusal string `json:"refusal,omitempty"`
}

// wireReasoning is the reasoning config in a request.
type wireReasoning struct {
	Effort  string `json:"effort,omitempty"`
	Summary string `json:"summary,omitempty"`
}

// wireTextFormat is the text.format config.
type wireTextFormat struct {
	Format *wireResponseFormat `json:"format,omitempty"`
}

// wireResponseFormat describes structured output format.
type wireResponseFormat struct {
	Type        string          `json:"type"` // text | json_object | json_schema
	Name        string          `json:"name,omitempty"`
	Schema      json.RawMessage `json:"schema,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
	Description string          `json:"description,omitempty"`
}

// wireTool is one tool definition.
type wireTool struct {
	Type string `json:"type"` // function | web_search | custom
	// function fields
	Name        string          `json:"name,omitempty"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
	// web_search fields
	WebSearchOptions json.RawMessage `json:"web_search_options,omitempty"`
}

// wireResponse is the non-streaming response body.
type wireResponse struct {
	ID                string                 `json:"id"`
	Object            string                 `json:"object"` // "response"
	Model             string                 `json:"model"`
	Status            string                 `json:"status"` // completed | incomplete | failed
	IncompleteDetails *wireIncompleteDetails `json:"incomplete_details,omitempty"`
	Output            []wireOutputItem       `json:"output"`
	Usage             json.RawMessage        `json:"usage,omitempty"`
	Error             *wireError             `json:"error,omitempty"`
	ServiceTier       string                 `json:"service_tier,omitempty"`
}

type wireIncompleteDetails struct {
	Reason string `json:"reason,omitempty"`
}

// wireOutputItem is one item in the response output array.
type wireOutputItem struct {
	Type string `json:"type"`
	ID   string `json:"id,omitempty"`
	// message fields
	Role    string           `json:"role,omitempty"`
	Content []wireOutputPart `json:"content,omitempty"`
	Status  string           `json:"status,omitempty"`
	// function_call fields
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	// reasoning fields
	Summary          []wireReasoningPart `json:"summary,omitempty"`
	EncryptedContent string              `json:"encrypted_content,omitempty"`
	// custom_tool_call fields
	Namespace string `json:"namespace,omitempty"`
	Input     string `json:"input,omitempty"`
	// web_search_call fields
	// (uses Status and ID above)
}

// wireOutputPart is one content part of a message output item.
type wireOutputPart struct {
	Type string `json:"type"`
	// output_text
	Text string `json:"text,omitempty"`
	// refusal
	Refusal string `json:"refusal,omitempty"`
}

// wireReasoningPart is one part of a reasoning item's summary.
type wireReasoningPart struct {
	Type string `json:"type"` // summary_text
	Text string `json:"text"`
}

// wireUsage is the token usage object in a response.
type wireUsage struct {
	InputTokens         int                `json:"input_tokens"`
	OutputTokens        int                `json:"output_tokens"`
	TotalTokens         int                `json:"total_tokens"`
	InputTokensDetails  *wireInputDetails  `json:"input_tokens_details,omitempty"`
	OutputTokensDetails *wireOutputDetails `json:"output_tokens_details,omitempty"`
}

type wireInputDetails struct {
	CachedTokens     int `json:"cached_tokens,omitempty"`
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
}

type wireOutputDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// wireError is the error object in a response or error response body.
type wireError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Param   string `json:"param,omitempty"`
	Type    string `json:"type,omitempty"`
}

// wireErrorResponse is a top-level error response body (same shape as chat).
type wireErrorResponse struct {
	Error wireError `json:"error"`
}

// --- Streaming event types ---

// wireStreamEvent is one SSE event in a streaming response.
type wireStreamEvent struct {
	Type           string `json:"type"`
	SequenceNumber int    `json:"sequence_number"`
	// response.created / response.in_progress / response.completed / etc.
	Response *wireStreamResponse `json:"response,omitempty"`
	// output_item.added / output_item.done
	OutputIndex int             `json:"output_index,omitempty"`
	Item        *wireOutputItem `json:"item,omitempty"`
	// content_part.added / content_part.done
	ContentIndex int             `json:"content_index,omitempty"`
	Part         *wireOutputPart `json:"part,omitempty"`
	// output_text.delta / output_text.done
	Delta string `json:"delta,omitempty"`
	Text  string `json:"text,omitempty"`
	// function_call_arguments.delta / function_call_arguments.done
	Arguments string `json:"arguments,omitempty"`
	// custom_tool_call_input.delta
	Input string `json:"input,omitempty"`
	// response.failed / error (top-level or nested)
	Error   *wireError `json:"error,omitempty"`
	Code    string     `json:"code,omitempty"`
	Message string     `json:"message,omitempty"`
	Param   string     `json:"param,omitempty"`
	// usage
	Usage *wireUsage `json:"usage,omitempty"`
}

// wireStreamResponse carries the response object in stream events.
type wireStreamResponse struct {
	ID                string                 `json:"id"`
	Object            string                 `json:"object"`
	Model             string                 `json:"model"`
	Status            string                 `json:"status"`
	IncompleteDetails *wireIncompleteDetails `json:"incomplete_details,omitempty"`
	Output            []wireOutputItem       `json:"output"`
	Usage             json.RawMessage        `json:"usage,omitempty"`
	Error             *wireError             `json:"error,omitempty"`
	ServiceTier       string                 `json:"service_tier,omitempty"`
}
