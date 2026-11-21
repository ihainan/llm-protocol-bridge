package gemini

import "encoding/json"

// Wire types for the Gemini generateContent API.

// wireRequest is the JSON body for generateContent / streamGenerateContent.
type wireRequest struct {
	Contents          []wireContent          `json:"contents"`
	SystemInstruction *wireContent           `json:"systemInstruction,omitempty"`
	Tools             []wireToolDef          `json:"tools,omitempty"`
	ToolConfig        *wireToolConfig        `json:"toolConfig,omitempty"`
	GenerationConfig  *wireGenerationConfig  `json:"generationConfig,omitempty"`
	SafetySettings    []wireSafetySetting    `json:"safetySettings,omitempty"`
}

// wireContent is one conversation turn.
type wireContent struct {
	Role  string     `json:"role,omitempty"` // user | model
	Parts []wirePart `json:"parts"`
}

// wirePart is one part of a content.
type wirePart struct {
	// text
	Text string `json:"text,omitempty"`
	// inlineData
	InlineData *wireBlob `json:"inlineData,omitempty"`
	// fileData
	FileData *wireFileData `json:"fileData,omitempty"`
	// functionCall
	FunctionCall *wireFunctionCall `json:"functionCall,omitempty"`
	// functionResponse
	FunctionResponse *wireFunctionResponse `json:"functionResponse,omitempty"`
	// thought (Gemini thinking)
	Thought          bool   `json:"thought,omitempty"`
	ThoughtSignature string `json:"thoughtSignature,omitempty"`
	// executableCode / codeExecutionResult (builtins)
	ExecutableCode       *wireExecutableCode       `json:"executableCode,omitempty"`
	CodeExecutionResult  *wireCodeExecutionResult  `json:"codeExecutionResult,omitempty"`
}

type wireBlob struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"` // base64
}

type wireFileData struct {
	MimeType string `json:"mimeType,omitempty"`
	FileURI  string `json:"fileUri"`
}

type wireFunctionCall struct {
	ID   string          `json:"id,omitempty"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

type wireFunctionResponse struct {
	ID       string          `json:"id,omitempty"`
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
}

type wireExecutableCode struct {
	Language string `json:"language,omitempty"`
	Code     string `json:"code"`
}

type wireCodeExecutionResult struct {
	Outcome string `json:"outcome"`
	Output  string `json:"output,omitempty"`
}

// wireToolDef groups function declarations (and optionally built-in tools).
type wireToolDef struct {
	FunctionDeclarations []wireFunctionDecl `json:"functionDeclarations,omitempty"`
	GoogleSearch         *struct{}          `json:"googleSearch,omitempty"`
	URLContext           *struct{}          `json:"urlContext,omitempty"`
	CodeExecution        *struct{}          `json:"codeExecution,omitempty"`
}

type wireFunctionDecl struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// wireToolConfig controls function calling mode.
type wireToolConfig struct {
	FunctionCallingConfig *wireFunctionCallingConfig `json:"functionCallingConfig,omitempty"`
}

type wireFunctionCallingConfig struct {
	Mode                 string   `json:"mode,omitempty"` // AUTO | NONE | ANY
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

// wireGenerationConfig holds sampling and output format parameters.
type wireGenerationConfig struct {
	Temperature       *float64        `json:"temperature,omitempty"`
	TopP              *float64        `json:"topP,omitempty"`
	TopK              *int            `json:"topK,omitempty"`
	MaxOutputTokens   *int            `json:"maxOutputTokens,omitempty"`
	CandidateCount    *int            `json:"candidateCount,omitempty"`
	StopSequences     []string        `json:"stopSequences,omitempty"`
	Seed              *int64          `json:"seed,omitempty"`
	ResponseMimeType  string          `json:"responseMimeType,omitempty"`
	ResponseSchema    json.RawMessage `json:"responseSchema,omitempty"`
	ThinkingConfig    *wireThinkingConfig `json:"thinkingConfig,omitempty"`
}

type wireThinkingConfig struct {
	ThinkingBudget int    `json:"thinkingBudget,omitempty"`
	IncludeThoughts bool   `json:"includeThoughts,omitempty"`
}

type wireSafetySetting struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

// wireResponse is the non-streaming generateContent response.
type wireResponse struct {
	Candidates     []wireCandidate  `json:"candidates"`
	UsageMetadata  *wireUsage       `json:"usageMetadata,omitempty"`
	ModelVersion   string           `json:"modelVersion,omitempty"`
}

// wireCandidate is one response candidate.
type wireCandidate struct {
	Index          int              `json:"index"`
	Content        wireContent      `json:"content"`
	FinishReason   string           `json:"finishReason,omitempty"`
	SafetyRatings  []wireSafetyRating `json:"safetyRatings,omitempty"`
	GroundingMetadata *wireGroundingMeta `json:"groundingMetadata,omitempty"`
}

type wireSafetyRating struct {
	Category    string `json:"category"`
	Probability string `json:"probability"`
}

type wireGroundingMeta struct {
	WebSearchQueries []string            `json:"webSearchQueries,omitempty"`
	GroundingChunks  []wireGroundChunk   `json:"groundingChunks,omitempty"`
}

type wireGroundChunk struct {
	Web *wireGroundWeb `json:"web,omitempty"`
}

type wireGroundWeb struct {
	URI   string `json:"uri"`
	Title string `json:"title,omitempty"`
}

// wireUsage is the usageMetadata object.
type wireUsage struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	TotalTokenCount         int `json:"totalTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount,omitempty"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount,omitempty"`
	ToolUsePromptTokenCount int `json:"toolUsePromptTokenCount,omitempty"`
}

// wireError is the top-level Gemini error body.
type wireError struct {
	Error wireErrorDetail `json:"error"`
}

type wireErrorDetail struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"` // RPC status code e.g. INVALID_ARGUMENT
}
