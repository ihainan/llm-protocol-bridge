// Package ir defines the format-neutral intermediate representation that every
// conversion in llm-protocol-bridge goes through, together with the helpers shared
// by the format packages: the event collector, usage arithmetic, effort tables,
// id codecs, the signature envelope and the error and loss model.
package ir

import (
	"encoding/json"
)

// Format names a wire format.
type Format string

// The supported wire formats.
const (
	FormatChat      Format = "chat"      // OpenAI Chat Completions
	FormatResponses Format = "responses" // OpenAI Responses
	FormatAnthropic Format = "anthropic" // Anthropic Messages
	FormatGemini    Format = "gemini"    // Gemini generateContent
)

// Formats lists every supported format.
var Formats = []Format{FormatChat, FormatResponses, FormatAnthropic, FormatGemini}

// Valid reports whether f is a supported format.
func (f Format) Valid() bool {
	switch f {
	case FormatChat, FormatResponses, FormatAnthropic, FormatGemini:
		return true
	}
	return false
}

// Origin records who issued an opaque value (a signature or a sealed reasoning
// block), so that it is only ever sent back to its issuer.
type Origin struct {
	Format   Format `json:"format,omitempty"`
	Provider string `json:"provider,omitempty"`
}

// Matches reports whether a value issued by o may be sent to an upstream of the
// given format and provider. An empty provider on either side matches.
func (o Origin) Matches(f Format, provider string) bool {
	if o.Format != f {
		return false
	}
	return o.Provider == "" || provider == "" || o.Provider == provider
}

// Request is a model call in any format.
type Request struct {
	Model  string `json:"model,omitempty"`
	Stream bool   `json:"stream,omitempty"`

	// Instructions holds the leading system / developer content (text parts).
	Instructions []Part    `json:"instructions,omitempty"`
	Messages     []Message `json:"messages,omitempty"`

	Tools             []Tool     `json:"tools,omitempty"`
	ToolChoice        ToolChoice `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool      `json:"parallel_tool_calls,omitempty"`
	MaxToolCalls      *int       `json:"max_tool_calls,omitempty"`

	MaxOutputTokens  *int     `json:"max_output_tokens,omitempty"`
	Temperature      *float64 `json:"temperature,omitempty"`
	TopP             *float64 `json:"top_p,omitempty"`
	TopK             *int     `json:"top_k,omitempty"`
	Stop             []string `json:"stop,omitempty"`
	Candidates       *int     `json:"candidates,omitempty"`
	Seed             *int64   `json:"seed,omitempty"`
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
	Logprobs         *bool    `json:"logprobs,omitempty"`
	TopLogprobs      *int     `json:"top_logprobs,omitempty"`

	Reasoning *ReasoningConfig `json:"reasoning,omitempty"`
	Output    *OutputFormat    `json:"output,omitempty"`
	Verbosity string           `json:"verbosity,omitempty"`

	PromptCacheKey       string          `json:"prompt_cache_key,omitempty"`
	PromptCacheRetention string          `json:"prompt_cache_retention,omitempty"`
	User                 string          `json:"user,omitempty"`
	Metadata             json.RawMessage `json:"metadata,omitempty"`
	ServiceTier          string          `json:"service_tier,omitempty"`

	PreviousResponseID string   `json:"previous_response_id,omitempty"`
	Store              *bool    `json:"store,omitempty"`
	Include            []string `json:"include,omitempty"`

	// Builtins are hosted tools (web search, code execution, ...).
	Builtins []BuiltinTool `json:"builtins,omitempty"`

	// Echo holds client fields a response must repeat back (Responses clients
	// receive their request parameters in every response object).
	Echo json.RawMessage `json:"echo,omitempty"`
	// Passthrough holds parameters the parser did not know, kept verbatim for
	// identity builds only.
	Passthrough map[string]json.RawMessage `json:"passthrough,omitempty"`
	// Source is the format the request was parsed from.
	Source Format `json:"source,omitempty"`
	// DroppedCacheMarkers is set by a parser that discarded a client
	// cache_control marker it could not carry (an invalid shape, or a marker
	// superseded by another). Automatic cache placement treats the request as
	// client-marked and adds no breakpoints of its own.
	DroppedCacheMarkers bool `json:"-"`
}

// Role is a message author.
type Role string

// Message roles.
const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleSystem    Role = "system"
	RoleDeveloper Role = "developer"
)

// Message is one conversation turn. Tool results are parts of user messages.
type Message struct {
	Role  Role   `json:"role"`
	Parts []Part `json:"parts,omitempty"`
	Name  string `json:"name,omitempty"`
	// Phase is a Responses assistant message phase: commentary | final_answer.
	Phase string `json:"phase,omitempty"`
}

// PartType is the kind of a Part.
type PartType string

// Part types.
const (
	PartText          PartType = "text"
	PartRefusal       PartType = "refusal"
	PartImage         PartType = "image"
	PartFile          PartType = "file"
	PartAudio         PartType = "audio"
	PartVideo         PartType = "video"
	PartToolCall      PartType = "tool_call"
	PartToolResult    PartType = "tool_result"
	PartReasoning     PartType = "reasoning"
	PartBuiltinCall   PartType = "builtin_call"
	PartBuiltinResult PartType = "builtin_result"
)

// Part is one block of a message.
type Part struct {
	Type PartType `json:"type"`

	Text      string           `json:"text,omitempty"`
	Media     *Media           `json:"media,omitempty"`
	Call      *ToolCall        `json:"call,omitempty"`
	Result    *ToolResult      `json:"result,omitempty"`
	Reasoning *Reasoning       `json:"reasoning,omitempty"`
	Builtin   *BuiltinActivity `json:"builtin,omitempty"`
	Citations []Citation       `json:"citations,omitempty"`
	// Cache marks a prompt-cache breakpoint after this part.
	Cache *CacheControl `json:"cache,omitempty"`
	// Signature is a Gemini thoughtSignature attached to a text part.
	Signature *Signature `json:"signature,omitempty"`
}

// TextPart returns a text part.
func TextPart(s string) Part { return Part{Type: PartText, Text: s} }

// Media is an image, file, audio or video payload.
type Media struct {
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"` // base64 without a data: prefix
	URL       string `json:"url,omitempty"`
	FileID    string `json:"file_id,omitempty"`
	Filename  string `json:"filename,omitempty"`
	Detail    string `json:"detail,omitempty"`
	Title     string `json:"title,omitempty"`
	Context   string `json:"context,omitempty"`
	// Text holds the content of a plain-text document.
	Text string `json:"text,omitempty"`
	// Source is a search result's source (Anthropic search_result blocks).
	Source string `json:"source,omitempty"`
}

// ToolKind distinguishes JSON-argument functions from freeform custom tools.
type ToolKind string

// Tool kinds.
const (
	ToolFunction ToolKind = "function"
	ToolCustom   ToolKind = "custom"
)

// ToolCall is a call the model made.
type ToolCall struct {
	ID        string   `json:"id,omitempty"`
	Name      string   `json:"name"`
	Namespace string   `json:"namespace,omitempty"`
	Kind      ToolKind `json:"kind,omitempty"`
	// Arguments is the JSON object of a function call, bytes preserved.
	Arguments json.RawMessage `json:"arguments,omitempty"`
	// Input is a custom tool call's freeform input.
	Input string `json:"input,omitempty"`
	// Signature is a Gemini thoughtSignature attached to the call.
	Signature *Signature `json:"signature,omitempty"`
}

// ToolResult answers a ToolCall.
type ToolResult struct {
	CallID  string `json:"call_id"`
	Name    string `json:"name,omitempty"`
	IsError bool   `json:"is_error,omitempty"`
	// Kind is the kind of the call answered (custom tool outputs differ on the
	// Responses wire).
	Kind    ToolKind `json:"kind,omitempty"`
	Content []Part   `json:"content,omitempty"`
}

// Reasoning is model reasoning, possibly signed or sealed by its issuer.
type Reasoning struct {
	ID        string     `json:"id,omitempty"`
	Text      string     `json:"text,omitempty"`
	Summary   []string   `json:"summary,omitempty"`
	Signature *Signature `json:"signature,omitempty"`
	Sealed    *Sealed    `json:"sealed,omitempty"`
	// Origin records the upstream format/provider that produced this reasoning
	// when known (used to seal unsigned reasoning accurately across multi-round
	// streams that switch upstreams).
	Origin Origin `json:"origin,omitempty"`
}

// Signature is an issuer's signature over reasoning (or a Gemini thought
// signature on a text part or call).
type Signature struct {
	Value  string `json:"value"`
	Origin Origin `json:"origin"`
	// Unsigned marks a bridge-verified provenance envelope over originally
	// unsigned reasoning (SealUnsigned), rather than a native issuer signature.
	// Native upstreams (such as Anthropic or Gemini) must never receive it as a
	// wire signature.
	Unsigned bool `json:"unsigned,omitempty"`
}

// SealedKind names the kind of an opaque reasoning block.
type SealedKind string

// Sealed reasoning kinds.
const (
	SealedAnthropicRedacted  SealedKind = "anthropic_redacted"
	SealedResponsesEncrypted SealedKind = "responses_encrypted"
)

// Sealed is reasoning its issuer must receive back unchanged.
type Sealed struct {
	Kind SealedKind `json:"kind"`
	// Data is the block or item exactly as the issuer wrote it.
	Data   json.RawMessage `json:"data"`
	Origin Origin          `json:"origin"`
}

// CacheControl is a prompt-cache breakpoint.
type CacheControl struct {
	Type string `json:"type,omitempty"` // "ephemeral"
	TTL  string `json:"ttl,omitempty"`
}

// Citation annotates output text.
type Citation struct {
	Type       string `json:"type,omitempty"` // url_citation | web_search_result_location | ...
	URL        string `json:"url,omitempty"`
	Title      string `json:"title,omitempty"`
	CitedText  string `json:"cited_text,omitempty"`
	StartIndex int    `json:"start_index,omitempty"`
	EndIndex   int    `json:"end_index,omitempty"`
	// Raw is the citation as its format wrote it, for identity encoding.
	Raw json.RawMessage `json:"raw,omitempty"`
}

// Tool is a tool the model may call.
type Tool struct {
	Name        string          `json:"name"`
	Namespace   string          `json:"namespace,omitempty"`
	Description string          `json:"description,omitempty"`
	Kind        ToolKind        `json:"kind,omitempty"`
	Schema      json.RawMessage `json:"schema,omitempty"`
	SchemaJSON  bool            `json:"schema_json,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
	Grammar     *Grammar        `json:"grammar,omitempty"`
	Deferred    bool            `json:"deferred,omitempty"`
	Cache       *CacheControl   `json:"cache,omitempty"`
}

// Grammar constrains a custom tool's input.
type Grammar struct {
	Type       string `json:"type,omitempty"`   // grammar | text
	Syntax     string `json:"syntax,omitempty"` // lark | regex
	Definition string `json:"definition,omitempty"`
}

// ToolChoiceMode is how the model must use tools.
type ToolChoiceMode string

// Tool choice modes.
const (
	ToolChoiceUnset    ToolChoiceMode = ""
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceRequired ToolChoiceMode = "required"
	ToolChoiceTool     ToolChoiceMode = "tool"
	ToolChoiceAllowed  ToolChoiceMode = "allowed"
)

// ToolRef names a tool, optionally inside a namespace.
type ToolRef struct {
	Name      string   `json:"name"`
	Namespace string   `json:"namespace,omitempty"`
	Kind      ToolKind `json:"kind,omitempty"`
	// Builtin names a hosted tool kind instead of a function.
	Builtin BuiltinKind `json:"builtin,omitempty"`
}

// ToolChoice constrains tool use.
type ToolChoice struct {
	Mode        ToolChoiceMode `json:"mode,omitempty"`
	Name        string         `json:"name,omitempty"`
	Namespace   string         `json:"namespace,omitempty"`
	Builtin     BuiltinKind    `json:"builtin,omitempty"`
	Allowed     []ToolRef      `json:"allowed,omitempty"`
	AllowedMode ToolChoiceMode `json:"allowed_mode,omitempty"`
}

// BuiltinKind names a hosted tool.
type BuiltinKind string

// Hosted tool kinds.
const (
	BuiltinWebSearch     BuiltinKind = "web_search"
	BuiltinWebFetch      BuiltinKind = "web_fetch"
	BuiltinCodeExecution BuiltinKind = "code_execution"
	BuiltinURLContext    BuiltinKind = "url_context"
)

// BuiltinTool is a hosted tool definition.
type BuiltinTool struct {
	Kind BuiltinKind `json:"kind"`
	// Config holds normalised options (max_uses, allowed_domains, ...).
	Config json.RawMessage `json:"config,omitempty"`
	// Raw is the definition as the client wrote it.
	Raw    json.RawMessage `json:"raw,omitempty"`
	Origin Format          `json:"origin,omitempty"`
}

// IsCachedWebSearch reports whether this hosted tool is a Responses cached
// web_search hint (external_web_access: false), which design.md §9.8 marks as
// droppable under LossCachedWebSearch.
func (bt BuiltinTool) IsCachedWebSearch() bool {
	if bt.Kind != BuiltinWebSearch {
		return false
	}
	for _, src := range []json.RawMessage{bt.Raw, bt.Config} {
		if len(src) == 0 {
			continue
		}
		var obj struct {
			ExternalWebAccess *bool `json:"external_web_access"`
		}
		if json.Unmarshal(src, &obj) == nil && obj.ExternalWebAccess != nil && !*obj.ExternalWebAccess {
			return true
		}
	}
	return false
}

// WebSearchConfig holds normalised web_search options parsed from Raw/Config.
type WebSearchConfig struct {
	ExternalWebAccess *bool
	AllowedDomains    []string
	BlockedDomains    []string
	UserLocation      json.RawMessage
	MaxUses           *int
}

// ParseWebSearchConfig extracts known web_search configuration fields from Raw/Config.
func (bt BuiltinTool) ParseWebSearchConfig() WebSearchConfig {
	var cfg WebSearchConfig
	for _, src := range []json.RawMessage{bt.Config, bt.Raw} {
		if len(src) == 0 {
			continue
		}
		var obj struct {
			ExternalWebAccess *bool           `json:"external_web_access"`
			AllowedDomains    []string        `json:"allowed_domains"`
			BlockedDomains    []string        `json:"blocked_domains"`
			UserLocation      json.RawMessage `json:"user_location"`
			MaxUses           *int            `json:"max_uses"`
			Filters           *struct {
				AllowedDomains []string `json:"allowed_domains"`
				BlockedDomains []string `json:"blocked_domains"`
			} `json:"filters"`
		}
		if json.Unmarshal(src, &obj) != nil {
			continue
		}
		if cfg.ExternalWebAccess == nil && obj.ExternalWebAccess != nil {
			cfg.ExternalWebAccess = obj.ExternalWebAccess
		}
		if len(cfg.AllowedDomains) == 0 {
			if len(obj.AllowedDomains) > 0 {
				cfg.AllowedDomains = obj.AllowedDomains
			} else if obj.Filters != nil && len(obj.Filters.AllowedDomains) > 0 {
				cfg.AllowedDomains = obj.Filters.AllowedDomains
			}
		}
		if len(cfg.BlockedDomains) == 0 {
			if len(obj.BlockedDomains) > 0 {
				cfg.BlockedDomains = obj.BlockedDomains
			} else if obj.Filters != nil && len(obj.Filters.BlockedDomains) > 0 {
				cfg.BlockedDomains = obj.Filters.BlockedDomains
			}
		}
		if len(cfg.UserLocation) == 0 && len(obj.UserLocation) > 0 && string(obj.UserLocation) != "null" {
			cfg.UserLocation = append(json.RawMessage(nil), obj.UserLocation...)
		}
		if cfg.MaxUses == nil && obj.MaxUses != nil {
			cfg.MaxUses = obj.MaxUses
		}
	}
	return cfg
}

// BuiltinActivity is a hosted tool call or its result in a conversation or reply.
type BuiltinActivity struct {
	Kind  BuiltinKind     `json:"kind"`
	ID    string          `json:"id,omitempty"`
	Query string          `json:"query,omitempty"`
	URL   string          `json:"url,omitempty"`
	Hits  []Hit           `json:"hits,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
	// IsResult distinguishes a result (hits, page content) from the call itself.
	IsResult bool `json:"is_result,omitempty"`
	// Content is a fetched page's text.
	Content string `json:"content,omitempty"`
	// Error is a provider error code for a failed hosted call.
	Error string `json:"error,omitempty"`
	// Raw is the block or item as its format wrote it.
	Raw    json.RawMessage `json:"raw,omitempty"`
	Origin Format          `json:"origin,omitempty"`
}

// Hit is one search result.
type Hit struct {
	Title   string `json:"title,omitempty"`
	URL     string `json:"url"`
	PageAge string `json:"page_age,omitempty"`
	Snippet string `json:"snippet,omitempty"`
}

// OutputFormatType is a structured-output kind.
type OutputFormatType string

// Structured-output kinds.
const (
	OutputText       OutputFormatType = "text"
	OutputJSONObject OutputFormatType = "json_object"
	OutputJSONSchema OutputFormatType = "json_schema"
)

// OutputFormat is the shape the client asked the answer in.
type OutputFormat struct {
	Type        OutputFormatType `json:"type"`
	Name        string           `json:"name,omitempty"`
	Description string           `json:"description,omitempty"`
	Schema      json.RawMessage  `json:"schema,omitempty"`
	Strict      *bool            `json:"strict,omitempty"`
}

// ReasoningMode is whether the model should reason.
type ReasoningMode string

// Reasoning modes.
const (
	ReasoningDefault  ReasoningMode = ""
	ReasoningOff      ReasoningMode = "off"
	ReasoningOn       ReasoningMode = "on"
	ReasoningAdaptive ReasoningMode = "adaptive"
)

// ReasoningConfig is the client's reasoning request.
type ReasoningConfig struct {
	Mode    ReasoningMode `json:"mode,omitempty"`
	Effort  Effort        `json:"effort,omitempty"`
	Budget  *int          `json:"budget,omitempty"`
	Summary string        `json:"summary,omitempty"`
	Display string        `json:"display,omitempty"`
	// IncludeEncrypted is a Responses client asking for encrypted_content.
	IncludeEncrypted bool `json:"include_encrypted,omitempty"`
	// KeepTurns is how many recent assistant turns keep their reasoning when it
	// is replayed (Anthropic clear_thinking keep); nil keeps all.
	KeepTurns *int `json:"keep_turns,omitempty"`
}

// PrependInstructions adds text before the existing instructions.
func (r *Request) PrependInstructions(text string) {
	if text == "" {
		return
	}
	r.Instructions = append([]Part{TextPart(text)}, r.Instructions...)
}

// AppendInstructions adds text after the existing instructions.
func (r *Request) AppendInstructions(text string) {
	if text == "" {
		return
	}
	r.Instructions = append(r.Instructions, TextPart(text))
}

// HasTools reports whether the request offers any function or hosted tool.
func (r *Request) HasTools() bool { return len(r.Tools) > 0 || len(r.Builtins) > 0 }

// FindTool returns the tool with the given namespace and name.
func (r *Request) FindTool(namespace, name string) *Tool {
	for i := range r.Tools {
		if r.Tools[i].Name == name && r.Tools[i].Namespace == namespace {
			return &r.Tools[i]
		}
	}
	return nil
}

// TextOf joins the text parts of parts.
func TextOf(parts []Part, sep string) string {
	n := 0
	var out string
	for _, p := range parts {
		if p.Type != PartText {
			continue
		}
		if n > 0 {
			out += sep
		}
		out += p.Text
		n++
	}
	return out
}

// Ptr returns a pointer to v.
func Ptr[T any](v T) *T { return &v }
