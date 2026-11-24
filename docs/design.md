# llm-protocol-bridge: Design

Status: draft for review. This document fixes the intermediate representation (IR),
the public API, the error model, the package layout and the porting scope before
implementation starts.

## 1. Purpose and scope

LLM gateways expose one API to their callers and forward each call to providers that
speak a different one. `llm-protocol-bridge` converts requests, streamed responses,
non-streamed responses and errors between four wire formats:

| Name        | Wire format                                                         |
|-------------|---------------------------------------------------------------------|
| `chat`      | OpenAI Chat Completions (`POST /v1/chat/completions`)               |
| `responses` | OpenAI Responses (`POST /v1/responses`)                             |
| `anthropic` | Anthropic Messages (`POST /v1/messages`, `/v1/messages/count_tokens`) |
| `gemini`    | Gemini `generateContent` / `streamGenerateContent`                  |

Terminology:

- **Client format**: the format the gateway's caller used.
- **Upstream format**: the format of the provider the gateway forwards to.
- **IR**: the format-neutral representation every conversion goes through.

### 1.1 Goals

1. Every client format can be served by every upstream format. The twelve
   cross-format directions are the product; the four identity directions exist for
   testing and normalisation, not for relaying.
2. Pure functions: no network I/O, no storage, no hidden clocks or randomness (both
   are injectable). Billing, provider selection, authentication, retries and
   persistence stay in the caller.
3. **No silent loss.** A request feature that cannot be represented in the upstream
   format is rejected with an error that names the JSON path of the offending field.
   The few conversions that are lossy by nature (for example a reasoning signature
   that only its issuer can verify) are documented in the support matrix and reported
   through an optional callback.
4. **Never hand a client an executable tool call that may be truncated or invalid**
   (section 7.3).
5. Exact usage accounting: cache reads, cache writes and reasoning tokens are neither
   double-counted nor dropped in any direction.
6. Deterministic output: the same input and options produce byte-identical upstream
   bodies, so provider-side prompt caches keep hitting across retries. The injectable
   `IDs` and `Clock` sources are used on the response side only; an upstream request
   body never contains a random value or a timestamp, and ids synthesised for history
   repairs are derived deterministically (9.2.2).
7. Small and fast: standard library only, Go 1.24; per-event streaming cost measured by
   benchmarks.

### 1.2 Non-goals

- Executing tools, including hosted tools such as web search. The library represents
  and converts them and supports a caller-driven tool loop (section 7.6); running the
  tools is the caller's job.
- Fetching media by URL (a caller hook resolves URLs, section 9.4).
- Storing conversations for `previous_response_id` (section 9.11).
- Token counting or estimation (the caller may pass an estimate, section 7.2).
- Same-format relaying, which should forward bytes unchanged.
- Provider workarounds keyed by host or model name. Deviations from a format are
  described by **dialect** options the caller chooses (section 10).

## 2. Architecture

```
            client request                                   upstream request
  bytes ──► <client>.ParseRequest ──► ir.Request ──► <upstream>.BuildRequest ──► bytes
                                         │
                                         │ (kept by the Plan: tool-name maps, stop
                                         │  emulation, client model, reasoning provenance)
                                         ▼
  upstream SSE ──► <upstream>.StreamDecoder ─► ir.Event ─► filters ─► <client>.StreamEncoder ─► client SSE
  upstream JSON ─► <upstream>.ParseResponse ─► ir.Response ─────────► <client>.BuildResponse ─► client JSON
  upstream error ► <upstream>.ParseError ────► ir.APIError ─────────► <client>.BuildError ────► client error
```

Each format package implements the same functions; the `bridge` package wires a client
format to an upstream format and carries per-request state from the request side to
the response side. Streamed and non-streamed responses share one code path:

- upstream stream, client non-stream: decoder events go into `ir.Collector`, which
  yields an `ir.Response`;
- upstream non-stream, client stream: `ir.EventsOf(resp)` replays a response as events.

A test asserts that the non-streamed output equals the collected streamed output for
the same upstream answer (ignoring generated ids).

## 3. Package layout

```
go.mod                 module github.com/ihainan/llm-protocol-bridge (go 1.24, no dependencies)
ir/                    IR types, Collector, EventsOf, usage arithmetic, effort tables,
                       ID codecs, signature envelope, Error / Loss types, validation paths
sse/                   SSE reader and writer (CRLF, multi-line data, comments, bare JSON
                       lines, JSON-array streams)
chat/                  OpenAI Chat Completions
responses/             OpenAI Responses
anthropic/             Anthropic Messages
gemini/                Gemini generateContent
bridge/                Format registry, Options, Plan, Stream, stream filters
internal/jsonx/        append-style JSON writers, raw JSON helpers
internal/schema/       JSON Schema adaptation (Gemini subset, Anthropic object root)
internal/testkit/      fake upstream scripts, golden-file helpers
testdata/              request samples, recorded streams, golden outputs
examples/liveproxy/    minimal HTTP shell for live end-to-end tests (not part of the
                       library; never run in CI)
docs/                  this document
```

Format packages sit at the top level so imports stay short. They depend only on `ir`,
`sse` and `internal/...`; `bridge` depends on all of them.

## 4. Public API

### 4.1 Format packages

Every format package exports the same surface:

```go
// Request side.
func ParseRequest(body []byte, opts ParseOptions) (*ir.Request, error)
func BuildRequest(req *ir.Request, opts BuildOptions) (*Built, error)

// Response side, upstream direction (provider -> IR).
func ParseResponse(body []byte, opts DecodeOptions) (*ir.Response, error)
func NewStreamDecoder(opts DecodeOptions) *StreamDecoder
func (d *StreamDecoder) Decode(out []ir.Event, ev sse.Event) ([]ir.Event, error)
func (d *StreamDecoder) End(out []ir.Event, how ir.EndKind) ([]ir.Event, error)

// Response side, client direction (IR -> caller).
func BuildResponse(resp *ir.Response, opts EncodeOptions) ([]byte, error)
func NewStreamEncoder(opts EncodeOptions) *StreamEncoder
func (e *StreamEncoder) Encode(dst []byte, ev ir.Event) ([]byte, error)
func (e *StreamEncoder) Close(dst []byte) ([]byte, error)
func (e *StreamEncoder) Keepalive(dst []byte) []byte

// Errors.
func ParseError(status int, body []byte) *ir.APIError
func BuildError(err *ir.APIError) (status int, body []byte)
func BuildStreamError(dst []byte, err *ir.APIError) []byte
```

- `Built` holds the body plus what goes outside it: the model and streaming flag for
  the Gemini URL, and the request-scoped maps the response side needs (tool names).
- `ParseOptions` carries what is not in the body: the Gemini model and streaming flag
  from the URL, the `anthropic-version` header (validated), and the endpoint variant
  (Anthropic `count_tokens`).
- `End` tells a decoder how the upstream ended: `EndDone` (Chat `[DONE]`), `EndEOF`,
  `EndTimeout`, `EndError`. A stream that ends without a terminal event is a normal end
  only on `EndDone`; otherwise the decoder emits an error event, never a fabricated
  success.
- Append-style signatures let callers reuse buffers; the hot streaming path does not
  allocate per text delta in the common case.
- Decoders take already-split SSE events; `sse.Reader` exists for callers without
  their own reader. Encoders and decoders are single-goroutine state machines.

### 4.2 The `bridge` package

```go
type Format = ir.Format // "chat" | "responses" | "anthropic" | "gemini"

func Convert(client, upstream Format, body []byte, opts Options) (*Plan, error)

// Two-step form for callers that edit the IR in between (add instructions, resolve
// media, expand previous_response_id, rewrite the model, register gateway tools).
func Parse(client Format, body []byte, opts Options) (*ir.Request, error)
func Build(req *ir.Request, client, upstream Format, opts Options) (*Plan, error)

type Plan struct {
	Client, Upstream Format
	Request          *ir.Request // normalised IR the upstream body was built from
	Body             []byte      // upstream request body
	Model            string      // upstream model (Gemini callers put it in the URL)
	ClientModel      string      // model name echoed to the client
	ClientStream     bool
	UpstreamStream   bool        // may differ from ClientStream (stop emulation, 9.6)
	Losses           []ir.Loss
}

func (p *Plan) NewStream() *Stream                                         // stream -> stream
func (p *Plan) NewCollector() *Stream                                      // stream -> client JSON
func (p *Plan) ConvertResponse(body []byte) ([]byte, *ir.Response, error)          // JSON -> JSON
func (p *Plan) StreamResponse(dst, body []byte) ([]byte, error)                    // JSON -> client stream
func (p *Plan) StreamResponseWithResult(dst, body []byte) ([]byte, *ir.Response, error) // JSON -> client stream + ir.Response (with Losses)
func (p *Plan) ConvertError(status int, body []byte) (int, []byte)

type Stream struct{ /* decoder + filters + encoder (or collector) */ }
func (s *Stream) Feed(dst []byte, ev sse.Event) ([]byte, error)
func (s *Stream) End(dst []byte, how ir.EndKind) ([]byte, error)
func (s *Stream) Done() bool              // nothing more is needed from the upstream
func (s *Stream) Keepalive(dst []byte) []byte
func (s *Stream) Usage() ir.Usage         // normalised usage, summed over rounds
func (s *Stream) Losses() []ir.Loss       // response-phase losses recorded across rounds
func (s *Stream) Result() ([]byte, *ir.Response, error) // collector mode: client JSON (ir.Response.Losses holds response losses)
func (s *Stream) Intercepted() []ir.ToolCall             // calls to caller-owned tools (7.6)
func (s *Stream) NextRound(p *Plan)       // continue the same client response (7.6)

func ErrorResponse(client Format, err error) (status int, body []byte)
```

`Plan.Losses` holds request-phase losses (parsing and building); response-phase
losses (decoding and encoding) are readable without setting `OnLoss` via
`ir.Response.Losses` (returned by `Plan.ConvertResponse`, `Plan.StreamResponseWithResult`,
and `Stream.Result`) and `Stream.Losses()` (streaming mode).

`Done()` turns true when the client response is complete before the upstream is (a
matched stop sequence, a terminal upstream error): the caller should stop reading and
close the upstream connection.

`Options`:

```go
type Options struct {
	ProviderID string           // opaque upstream provider id; scopes signatures (9.3)
	Profile    *ir.ModelProfile // capabilities of the upstream model (10.2)
	Chat       chat.Dialect     // upstream dialects (10.1)
	Anthropic  anthropic.Dialect
	Gemini     gemini.Dialect
	Responses  responses.Dialect

	Allow   ir.LossSet          // losses tolerated instead of an error (8.3)
	OnLoss  func(ir.Loss)
	Unknown ir.UnknownPolicy    // unknown request fields (8.4)
	Repair  ir.RepairSet        // optional history repairs (9.2.4)

	ToolRelease ir.ToolRelease  // when tool calls reach the client (7.3); default per client format
	Limits      ir.Limits       // max tool calls / buffered bytes per response
	StopMode    ir.StopMode     // Native | Emulate (9.6)
	Reasoning   ir.ReasoningOut // native | summary | hidden (9.3.5)
	Seal        ir.SealCodec    // signature envelope codec (9.3.4)
	Intercept   []string        // tool names the caller executes itself (7.6)

	IDs   ir.IDSource           // generated ids (default crypto/rand)
	Clock func() time.Time      // timestamps (default time.Now)
	PromptEstimate int          // optional input-token estimate for early usage fields
}
```

## 5. Intermediate representation

The IR is a superset of what the four formats can say, plus provenance needed to route
opaque data back to its issuer. All IR types have stable JSON tags so a caller can
persist them (for example to implement `previous_response_id`).

### 5.1 Request

```go
type Request struct {
	Model  string
	Stream bool

	Instructions []Part    // leading system / developer / instructions / systemInstruction
	Messages     []Message

	Tools             []Tool
	ToolChoice        ToolChoice // zero value = not specified
	ParallelToolCalls *bool
	MaxToolCalls      *int

	MaxOutputTokens  *int
	Temperature      *float64
	TopP             *float64
	TopK             *int
	Stop             []string
	Candidates       *int      // Chat n / Gemini candidateCount
	Seed             *int64
	PresencePenalty  *float64
	FrequencyPenalty *float64
	Logprobs         *bool
	TopLogprobs      *int

	Reasoning *ReasoningConfig
	Output    *OutputFormat
	Verbosity string

	PromptCacheKey       string
	PromptCacheRetention string
	User                 string
	Metadata             json.RawMessage
	ServiceTier          string

	PreviousResponseID string
	Store              *bool
	Include            []string

	Builtins []BuiltinTool         // hosted tools

	Echo        json.RawMessage            // client fields a response must echo (Responses)
	Passthrough map[string]json.RawMessage // kept only for identity builds
	Source      Format
}
```

Pointers mark "not specified", so builders never invent values the client did not ask
for, except where the upstream requires one (Anthropic `max_tokens`; the default then
comes from `ModelProfile`). Optional object and array fields that are `null`, `{}` or
`[]` are treated as absent.

### 5.2 Messages and parts

```go
type Role string // user | assistant | system | developer

type Message struct {
	Role  Role
	Parts []Part
	Name  string
	Phase string // Responses assistant phase: commentary | final_answer
}
```

`system` / `developer` messages appear in `Messages` only for instructions placed in
the middle of the conversation; leading ones are normalised into
`Request.Instructions` (section 9.1).

Tool results are parts of a `user` message (the Anthropic convention). Parsers fold
Chat `tool` messages, Responses `*_call_output` items and Gemini `functionResponse`
parts into it; builders split them out again.

```go
type Part struct {
	Type PartType // text | refusal | image | file | audio | video | tool_call |
	              // tool_result | reasoning | builtin_call | builtin_result

	Text      string
	Media     *Media
	Call      *ToolCall
	Result    *ToolResult
	Reasoning *Reasoning
	Builtin   *BuiltinActivity
	Citations []Citation
	Cache     *CacheControl // prompt-cache breakpoint (type, ttl)
	Signature *Signature    // Gemini thoughtSignature on text parts
}

type Media struct {
	MediaType string
	Data      string // base64, no data: prefix
	URL       string
	FileID    string
	Filename  string
	Detail    string // image detail
	Title     string // document title
	Context   string // document context
	Source    string // search_result source
}

type ToolCall struct {
	ID        string
	Name      string
	Namespace string
	Kind      ToolKind        // function | custom
	Arguments json.RawMessage // function: JSON object, bytes preserved verbatim
	Input     string          // custom tools
	Signature *Signature      // Gemini thoughtSignature on functionCall
}

type ToolResult struct {
	CallID  string
	Name    string // filled from the matching call when the source omits it
	IsError bool
	Content []Part // text, image, file (document / search result) parts
}
```

### 5.3 Reasoning

```go
type Reasoning struct {
	ID        string
	Text      string   // full reasoning text, when exposed
	Summary   []string // Responses summary parts
	Signature *Signature
	Sealed    *Sealed
}

type Signature struct {
	Value  string
	Origin Origin
}

type Sealed struct {
	Kind   SealedKind      // anthropic_redacted | responses_encrypted
	Data   json.RawMessage // block / item exactly as the issuer wrote it
	Origin Origin
}

type Origin struct {
	Format   Format // format of the upstream that issued it
	Provider string // Options.ProviderID at issue time ("" = unknown)
}
```

### 5.4 Tools

```go
type Tool struct {
	Name        string
	Namespace   string
	Description string
	Kind        ToolKind        // function | custom
	Schema      json.RawMessage
	Strict      *bool
	Grammar     *Grammar        // custom tools: syntax (lark | regex), definition
	Deferred    bool            // Anthropic defer_loading
	Cache       *CacheControl
}

type ToolChoice struct {
	Mode        ToolChoiceMode // "" | auto | none | required | tool | allowed
	Name        string
	Namespace   string
	Allowed     []ToolRef
	AllowedMode ToolChoiceMode // auto | required
}

type BuiltinTool struct {
	Kind   BuiltinKind     // web_search | web_fetch | code_execution | url_context | ...
	Config json.RawMessage // normalised options (max_uses, domains, user_location, ...)
	Raw    json.RawMessage // original definition (identity builds)
	Origin Format
}
```

### 5.5 Response, events and usage

```go
type Response struct {
	ID      string
	Model   string
	Outputs []Output // one per candidate
	Usage   Usage
	Created int64
}

type Output struct {
	Index int
	Parts []Part
	Stop  Stop
}

type Stop struct {
	Reason   StopReason // end_turn | max_tokens | stop_sequence | tool_use | content_filter |
	                    // refusal | pause_turn | context_window | other
	Sequence string     // matched stop sequence, when known
	Raw      string     // upstream's own word
}
```

Streaming events are a flat struct with a `Type` and the fields that type uses;
`Index` is the candidate index:

| Type              | Fields                         | Meaning                                    |
|-------------------|--------------------------------|--------------------------------------------|
| `start`           | ID, Model, Usage (prompt side) | response began                             |
| `text`            | Text                           | output text delta                          |
| `citation`        | Citation                       | annotation on the current text part        |
| `reasoning_start` | ID                             | a new reasoning block begins               |
| `reasoning`       | Text, Summary (bool)           | reasoning text or summary delta            |
| `signature`       | Signature                      | signature of the current reasoning / text  |
| `sealed`          | Sealed                         | opaque reasoning block, whole              |
| `tool_start`      | ID, Name, Namespace, Kind      | tool call begins                           |
| `tool_args`       | Text                           | argument / custom input delta              |
| `tool_end`        | Arguments (final, optional)    | tool call complete                         |
| `builtin`         | Builtin                        | hosted tool activity (query, results)      |
| `image`           | Media                          | model-generated image                      |
| `stop`            | Stop                           | candidate finished                         |
| `usage`           | Usage (cumulative snapshot)    | usage so far                               |
| `error`           | Err                            | terminal error                             |

Blocks end implicitly when another block starts or at `stop`. `usage` events are
cumulative snapshots ("last reported value wins"), never deltas. The `Collector`
applies the same rules for non-streamed output.

```go
type Usage struct {
	InputTokens         int // prompt tokens neither read from nor written to a cache
	CacheReadTokens     int
	CacheWriteTokens    int
	CacheWrite5mTokens  int
	CacheWrite1hTokens  int
	OutputTokens        int // reasoning included
	ReasoningTokens     int // subset of OutputTokens
	ToolUsePromptTokens int // subset of InputTokens (Gemini)
	WebSearchRequests   int
	WebFetchRequests    int
	Reported            UsageFields // which counters the upstream actually reported
}

func (u Usage) PromptTokens() int { return u.InputTokens + u.CacheReadTokens + u.CacheWriteTokens }
func (u Usage) TotalTokens() int  { return u.PromptTokens() + u.OutputTokens }
```

`Reported` separates "zero" from "not reported"; encoders omit unreported optional
counters (no misleading `reasoning_tokens: 0` from an upstream that does not count
them).

## 6. Usage mapping

| Format    | Decode (wire to IR) | Encode (IR to wire) |
|-----------|---------------------|---------------------|
| anthropic | `input_tokens` = Input; `cache_read_input_tokens` = CacheRead; `cache_creation_input_tokens` (or 5m + 1h) = CacheWrite; `cache_creation.ephemeral_{5m,1h}_input_tokens` = split; `output_tokens` = Output; `server_tool_use.{web_search,web_fetch}_requests` | the four standard fields; split and `server_tool_use` only when non-zero |
| chat      | Input = `prompt_tokens` - CacheRead - CacheWrite, unless the dialect says the prompt count already excludes cache, or CacheRead + CacheWrite > `prompt_tokens` (then Input = `prompt_tokens`); CacheRead = `prompt_tokens_details.cached_tokens` (dialect fallbacks); CacheWrite from the dialect's cache-write field; Output = `completion_tokens` (plus reasoning when the dialect says it is excluded); Reasoning = `completion_tokens_details.reasoning_tokens` | `prompt_tokens` = PromptTokens(); `completion_tokens`; `total_tokens` = prompt + completion; `cached_tokens`; `reasoning_tokens` when reported |
| responses | as chat, on `input_tokens` / `input_tokens_details` / `output_tokens_details` | `input_tokens` = PromptTokens(); `total_tokens` = input + output; `cached_tokens`, `cache_write_tokens` (dialect switch), `reasoning_tokens` |
| gemini    | Input = `promptTokenCount` - `cachedContentTokenCount` + `toolUsePromptTokenCount`; CacheRead = `cachedContentTokenCount`; Output = `candidatesTokenCount` + `thoughtsTokenCount`; Reasoning = `thoughtsTokenCount` | `promptTokenCount` = PromptTokens() - ToolUsePrompt; `candidatesTokenCount` = Output - Reasoning; `thoughtsTokenCount`; `cachedContentTokenCount` |

The source usage object is never mutated. Billing callers read `Stream.Usage()` or
`Response.Usage` (IR semantics) instead of re-parsing either wire format.

## 7. Streaming model

### 7.1 Decoders

Rules every decoder implements (each with a regression test):

- prompt usage on the first chunk is attached to `start`;
- an error object in any shape (including a bare JSON line without `data:`) ends the
  stream with an `error` event; nothing after it is processed;
- `End(EndEOF | EndTimeout | EndError)` without a terminal event yields an `error`
  event (`timeout` for `EndTimeout`), never a success;
- a reply with no candidates / no output is an error.

Format-specific:

- **chat**: reasoning from `reasoning_content`, else `reasoning`; a chunk carrying both
  reasoning and content yields both; tool-call fragments are keyed by `index` (else
  position) and may interleave; arguments may arrive before the name; names may be
  split, resent whole, or sent as a prefix and then whole, and are merged; repeated
  upstream call ids within a response are replaced; optional `<think>` tag extraction
  (dialect, Markdown-aware). Only the first choice is followed unless `n > 1`.
- **responses**: `function_call` items may all be `added` before any arguments
  (queue, match by `item_id` then `output_index`); final arguments are reconciled with
  streamed deltas; reasoning item ids are kept from `output_item.added`.
- **anthropic**: a `tool_use` block that starts with a non-empty `input` uses it;
  `redacted_thinking` becomes `sealed`; `ping` ignored.
- **gemini**: `functionCall` parts arrive whole; missing ids are synthesised
  deterministically; `groundingMetadata` becomes one `builtin` event; the first
  `finishReason` wins.

### 7.2 Encoders

- **chat**: role chunk; `delta.content`, `delta.reasoning_content`,
  `delta.tool_calls[{index}]` with one index per call in emission order;
  `finish_reason` chunk; usage chunk (`choices: []`) when requested; `data: [DONE]`.
- **responses**: `response.created` and `response.in_progress` first (full response
  object with the client's echoed fields, `output: []`); one output item in progress at
  a time; `sequence_number` from 0; `output_index` = final position; item ids `msg_`,
  `rs_`, `fc_`, `ctc_`, `ws_`; custom tool input streamed with
  `response.custom_tool_call_input.delta` (decoded incrementally from the upstream's
  `{"input": "..."}` arguments, escapes and surrogate pairs split across fragments
  included; the streamed text must be a prefix of the final input or the response
  fails); namespaced calls split back into `namespace` + `name`; assistant `phase`;
  leading newlines after reasoning are trimmed and a newline-only delta never opens a
  message; terminal `response.completed` | `response.incomplete` | `response.failed`
  exactly once.
- **anthropic**: `message_start` (id `msg_...`, client model, `stop_reason` and
  `stop_sequence` null, usage from `PromptEstimate` until exact usage arrives), then
  `ping`; one open block at a time with sequential indices; `thinking` blocks always
  carry `signature`; `message_delta` with `stop_reason`, `stop_sequence` and usage;
  `message_stop`; after an `error` event nothing else.
- **gemini**: one JSON chunk per event group; function calls emitted whole; final
  chunk with `finishReason` and `usageMetadata`; SSE for `alt=sse`, otherwise a JSON
  array stream.

Response envelopes always carry the client's model name and ids generated by the
library, never the upstream's id or model. `Keepalive` writes what each format's
clients tolerate (Chat SSE comment; Anthropic `ping` after `message_start`; Responses
repeated `response.in_progress`; nothing for Gemini). The caller drives keepalive
timing; the Anthropic encoder also emits a `ping` every N buffered tool fragments.

### 7.3 Tool call release

A client starts executing a tool as soon as its format says the call is complete. The
library therefore controls when calls are released (`Options.ToolRelease`):

| Mode        | Behaviour                                                                     | Default for |
|-------------|-------------------------------------------------------------------------------|-------------|
| `Buffered`  | calls are held until the upstream finished; all are validated; then written whole | anthropic, gemini |
| `LiveFirst` | the first call's arguments stream live (keeps clients' idle timers alive); later calls are buffered; no call is completed until the upstream finished and all calls validated | responses |
| `Stream`    | calls are forwarded as they arrive, validated at completion                   | chat        |

Validation: a call needs a name, and function arguments must be a JSON object (empty
means `{}`). One invalid call fails the whole response (`error` /
`response.failed`); no call reaches a completed state. A response that ends with
`max_tokens` drops every buffered call without an error (the calls may be truncated);
a Responses client also has calls dropped on `content_filter`. In `Stream` mode a call
may already be on the wire when the upstream ends with `max_tokens`; the response then
reports the truncation (Chat `finish_reason: length`), never `tool_calls`, so the client
does not execute a truncated call. Text that arrives after
a buffered call is kept in order. `Limits` bounds a response (defaults: 128 calls,
8 MiB of buffered call data); exceeding it is an `error` event.

### 7.4 Stop-reason mapping

Decoders normalise to `Stop.Reason`; encoders map back:

| IR             | chat             | responses                          | anthropic       | gemini      |
|----------------|------------------|------------------------------------|-----------------|-------------|
| end_turn       | `stop`           | `completed`                        | `end_turn`      | `STOP`      |
| tool_use       | `tool_calls`     | `completed`                        | `tool_use`      | `STOP`      |
| max_tokens     | `length`         | `incomplete` / `max_output_tokens` | `max_tokens`    | `MAX_TOKENS`|
| stop_sequence  | `stop`           | `completed`                        | `stop_sequence` | `STOP`      |
| content_filter | `content_filter` | `incomplete` / `content_filter`    | `refusal`       | `SAFETY`    |
| refusal        | `content_filter` | `incomplete` / `content_filter`    | `refusal`       | `SAFETY`    |
| pause_turn     | `stop`           | `completed`                        | `pause_turn`    | `STOP`      |

Consistency rules, in priority order: a `max_tokens` end is always reported as
`max_tokens` (`length`), even when a tool call was already written to the client
(7.3); otherwise `tool_use` is reported whenever a tool call was written to the client,
whatever the upstream said; an upstream `tool_calls` finish with no call
delivered becomes `end_turn`; unknown upstream reasons become `end_turn` (recorded in
`Stop.Raw`).

### 7.5 Filters

Between decoder and encoder `bridge.Stream` runs pure filters: tool-name reverse
mapping (namespaces, sanitised names, custom tools), tool release (7.3), stop-sequence
emulation (9.6), reasoning presentation (9.3.5) and signature sealing (9.3.4).

### 7.6 Caller-executed tools and multi-round responses

A gateway may run some tools itself (for example its own web search) while the client
sees a single response. The caller adds those tools to the IR as ordinary function
tools and lists their names in `Options.Intercept`. The stream then:

1. withholds calls to intercepted names from the client (including a call whose name
   so far is a prefix of an intercepted name, until the name is known) and exposes
   them through `Intercepted()`;
2. at the upstream's end, if intercepted calls exist, completes the client-visible
   calls of that round but does not finish the client response;
3. after the caller runs the tools and builds the next upstream request,
   `NextRound(plan)` swaps in a fresh decoder while keeping the encoder's state
   (indices, sequence numbers, open items), so the same client response continues;
4. sums usage over rounds; the final round finishes the response normally.

The caller can surface its tool activity to the client as `builtin` events
(`server_tool_use` + `web_search_tool_result` for Anthropic, `web_search_call` for
Responses). The library ships no search implementation.

## 8. Errors

### 8.1 Library errors

```go
type Error struct {
	Code    ErrorCode // invalid_request | unsupported | upstream_protocol | config | internal
	Path    Path      // structured path, rendered in the client's style
	Feature string    // stable key: "image.file_id", "tool_choice.allowed_tools"
	Message string
}
```

- `invalid_request` (400): the body violates its own format: missing Anthropic
  `max_tokens`, `budget_tokens` below 1024 or not below `max_tokens`, a
  `tool_result` that answers no call of the latest assistant turn, duplicate Responses
  `call_id`, an output answering a call of another kind, arguments that are not a
  JSON-encoded string, out-of-range sampling values (Anthropic `temperature` in [0, 1],
  OpenAI in [0, 2]), malformed `anthropic-version`, empty text blocks.
- `unsupported` (400): valid in the client format but not expressible upstream:
  `n > 1` towards Anthropic, `input_file` towards an upstream without file input,
  `conversation`, `background: true`, prompt templates, `truncation: "auto"`,
  `item_reference`, MCP servers, containers, server tools without an equivalent.
- `upstream_protocol` (502): the upstream response could not be understood.
- `config` (500): the caller's dialect or profile is inconsistent (for example an
  invalid "reasoning off" patch). The message does not expose configuration details.

`Path` is rendered in the client format's own style: `input[2].call_id` for OpenAI
formats, `messages.0.content.1.source` for Anthropic, `contents[0].parts[1]` for Gemini.
`bridge.ErrorResponse(client, err)` produces the client's error body (OpenAI `param`
carries the path; Anthropic prefixes the message with it).

### 8.2 Upstream errors

```go
type APIError struct {
	Status     int
	Type       ErrorType // invalid_request | authentication | billing | permission | not_found |
	                     // request_too_large | rate_limit | overloaded | server | timeout
	Code       string
	Message    string
	Param      string
	RetryAfter time.Duration
	Raw        json.RawMessage
}
```

| Client    | Non-streamed body                                            | In-stream                                    |
|-----------|--------------------------------------------------------------|----------------------------------------------|
| chat      | `{"error":{"message","type","code","param"}}`                | `data: {"error":{...}}`                      |
| responses | same as chat                                                 | `response.failed` with `error{code,message}` |
| anthropic | `{"type":"error","error":{"type","message"}}`                | `event: error`                               |
| gemini    | `{"error":{"code":<http>,"message","status":<RPC status>}}`  | `data: {"error":{...}}`                      |

Status mapping follows each provider's documentation, for example Anthropic
400 `invalid_request_error`, 401 `authentication_error`, 402 `billing_error`,
403 `permission_error`, 404 `not_found_error`, 413 `request_too_large`,
429 `rate_limit_error`, 408/504 `timeout_error`, 503/529 `overloaded_error`, other 5xx
`api_error`; Gemini `INVALID_ARGUMENT`, `UNAUTHENTICATED`, `PERMISSION_DENIED`,
`NOT_FOUND`, `RESOURCE_EXHAUSTED`, `INTERNAL`, `UNAVAILABLE`, `DEADLINE_EXCEEDED`.
An upstream error type that already belongs to the client's vocabulary is kept.
`Retry-After` is carried in `RetryAfter`. Context-length errors are recognised across
providers and mapped to the client's own context-length error. Upstream messages are
passed through; masking credentials or host names is the caller's job.

### 8.3 Losses

A **loss** keeps the request working but drops or changes information. Each has a stable
key (`ir.LossKind`):

| Key                          | Example                                                    | Default |
|------------------------------|------------------------------------------------------------|---------|
| `reasoning.signature`        | a signature cannot go to another provider                  | allowed |
| `reasoning.sealed`           | encrypted reasoning of another provider                    | allowed |
| `request.user`               | end-user id towards a format without it                    | allowed |
| `request.metadata`           | metadata towards a format without it                       | allowed |
| `request.service_tier`       | service tier towards a format without it                   | allowed |
| `request.prompt_cache`       | `prompt_cache_key` / retention towards a format without it | allowed |
| `cache.marker`               | `cache_control` towards an upstream that caches implicitly | allowed |
| `image.detail`               | image detail towards a format without it                   | allowed |
| `tool.strict`                | `strict` towards an upstream without strict tools          | allowed |
| `tool.parallel`              | `parallel_tool_calls: false` towards Gemini                | allowed |
| `tool.cached_web_search`     | Responses `web_search` with `external_web_access: false` dropped | allowed |
| `request.unknown_field`      | an unknown request parameter (8.4)                         | allowed |
| `sampling.seed`              | `seed` towards an upstream without it (best effort anyway) | allowed |
| `sampling.penalty`           | non-zero penalties towards an upstream without them        | error   |
| `sampling.logprobs`          | `logprobs: true` / `top_logprobs > 0` towards an upstream without them | error |
| `sampling.clamp`             | a sampling value outside the upstream's range, clamped (9.10) | error |
| `stop.limit`                 | more stop sequences than the upstream accepts              | error   |

Neutral values are not losses and are dropped silently: a penalty of 0,
`logprobs: false`, `top_logprobs: 0` (many SDKs send them by default).

Allowed losses are reported through `OnLoss` and `Plan.Losses`; error losses fail
unless listed in `Options.Allow`. Losses that change what the model sees (content,
tools, instructions, tool choice) are never allowed by default. The README matrix lists
every key.

### 8.4 Unknown fields

Each parser knows its format's field list, every field classified as mapped, lossy
(8.3) or unsupported. Unknown input falls into two classes:

- **Unknown parameters** (a key the parser does not know, at the top level or inside a
  parameter object such as `reasoning` or `generationConfig`): governed by
  `Options.Unknown`. The default `Ignore` drops them and reports a
  `request.unknown_field` loss; `Reject` fails with the path.
- **Unknown content**: an unknown content block or part type, message role, input item
  type or tool type. These change what the model sees and are always rejected with the
  path, whatever `Options.Unknown` says. Known extension blocks the library
  deliberately skips are listed in the support matrix.

Identity builds keep unknown parameters verbatim.

## 9. Cross-cutting semantics

### 9.1 Instructions

- Leading system / developer content (Responses `instructions` plus leading system and
  developer items; Anthropic `system`; Chat leading system / developer messages;
  Gemini `systemInstruction`) becomes `Request.Instructions`, emitted as the
  upstream's top-level form. Join separators match the source (`\n\n` between
  Responses items, `\n` between parts of one item and between Anthropic blocks).
- Mid-conversation instructions stay in place by default (Chat `system` messages,
  Responses developer items, Anthropic `{role: system}` turns). For upstreams or
  models that do not accept them, `Dialect.MidSystem = Fold` merges them into the
  adjacent user message as `<system_message>\n...\n</system_message>`. They are never
  hoisted into the top-level instructions (that changes the prompt prefix and breaks
  prompt caching).
- Callers add their own instructions through `Request.PrependInstructions` /
  `AppendInstructions`, which behave the same for every direction.

### 9.2 Tool calls

#### 9.2.1 Definitions

- Function tools keep name, description, schema and `strict`. Absent parameters become
  `{"type":"object","properties":{}}`; a schema whose root cannot be an object is
  `invalid_request`. Duplicate names are `invalid_request`.
- Schemas are adapted in `internal/schema`: Gemini receives `parametersJsonSchema`
  (or the OpenAPI subset in `parameters` when the dialect needs it); Anthropic requires
  an object root.
- Custom (freeform) tools, towards formats without them, become a function with schema
  `{"type":"object","properties":{"input":{"type":"string","description":"The raw input
  for this tool, exactly in the format the tool description requires."}},"required":["input"]}`;
  a grammar is appended to the description. Calls are mapped back to
  `custom_tool_call` for Responses clients.
- Namespaces flatten to `<namespace>__<name>`; the namespace is added to the
  description as `[<namespace>] `. Name collisions get `_2`, `_3`, ... Names are
  restored from a request-scoped map kept in the Plan, never by parsing names.
- Upstream naming rules (Gemini identifier rules, length limits) are applied through
  the same reversible map.

#### 9.2.2 Call ids

- Ids are kept when both sides carry them. Anthropic accepts only `[A-Za-z0-9_-]`;
  other ids are encoded reversibly and decoded on the way back.
- Missing ids are synthesised deterministically from message index, call index, name
  and arguments.
- An id-less Gemini `functionResponse` is matched to the earliest unanswered call of the
  same name in the latest model turn.

#### 9.2.3 Structural rules (always applied)

- Same-role messages are merged where the upstream requires alternation; parallel calls
  split across two assistant messages are joined.
- Tool results are placed directly after the assistant message with the calls.
- Assistant text next to tool calls is kept.
- Chat tool messages are text only. A tool result's images are replaced by
  `[image: attached in the next message]` and sent in a following user message after a
  line `[images returned by tool call <id>]`, merged into the next user message when
  there is one. `is_error` results are prefixed `[tool error] `. Documents and search
  results in tool results are rendered as text.
- Gemini: a user turn mixing `functionResponse` with text is split; `functionResponse`
  never sits in a `model` turn; history does not start or end with `model`.
- `tool_choice` `auto` / `none` is dropped when no tools remain; a forced choice with
  no tools is `invalid_request`.

#### 9.2.4 Optional repairs

Repairs that change content are opt-in (`Options.Repair`); without them the condition
is an `invalid_request` error:

- `synthesize_missing_results`: an unanswered call gets an error result;
- `drop_duplicate_results`: a second result for the same call is dropped;
- `orphan_results_as_text`: a result that answers no call becomes user text;
- `drop_empty_assistant`: assistant messages with no content and no calls are dropped.

#### 9.2.5 tool_choice and parallel calls

| IR             | chat                                | responses                    | anthropic                        | gemini                                |
|----------------|-------------------------------------|------------------------------|----------------------------------|---------------------------------------|
| auto           | `auto`                              | `auto`                       | `{type:auto}`                    | `AUTO`                                |
| none           | `none`                              | `none`                       | `{type:none}`                    | `NONE`                                |
| required       | `required`                          | `required`                   | `{type:any}`                     | `ANY`                                 |
| tool(name)     | `{type:function,function:{name}}`   | `{type:function,name}`       | `{type:tool,name}`               | `ANY` + `allowedFunctionNames:[name]` |
| allowed(list)  | tools filtered, mode auto/required  | `{type:allowed_tools,...}`   | tools filtered, mode auto/any    | `allowedFunctionNames` + mode         |
| parallel=false | `parallel_tool_calls:false`         | `parallel_tool_calls:false`  | `disable_parallel_tool_use:true` | no equivalent (`tool.parallel` loss)  |

When the history contains calls but no tools are declared, the Anthropic builder
declares stub tools and sets `tool_choice` none.

### 9.3 Reasoning

#### 9.3.1 Configuration

```go
type ReasoningConfig struct {
	Mode    ReasoningMode // default | off | on | adaptive
	Effort  Effort        // none | minimal | low | medium | high | xhigh | max
	Budget  *int
	Summary string        // auto | concise | detailed
	Display string        // Anthropic thinking display
}
```

Precedence when reading Anthropic requests: `thinking.disabled` > `output_config.effort`
> `budget_tokens` > profile default. Chat: `reasoning_effort` > `reasoning.effort` >
`thinking`. Gemini: `thinkingLevel` > `thinkingBudget` (`-1` = model default, `0` = off).

#### 9.3.2 Effort and budget

All tables live in the `ModelProfile` (10.2), so a caller can reproduce a given
upstream exactly:

- `Efforts`: levels the upstream accepts; a requested effort is fitted to the nearest
  accepted level, ties going up (with `[low, high, max]`: minimal and low to low,
  medium and high to high, xhigh and max to max);
- `BudgetToEffort`: thresholds for reading a budget as an effort;
- `EffortToBudget`: budgets for budget-style upstreams;
- `DefaultEffort`: sent when the client asked for nothing.

Builders apply the upstream's constraints: Anthropic budget at least 1024 and below
`max_tokens` (raising `max_tokens` when the client did not set it, lowering the budget
when it did), `temperature` / `top_p` omitted while thinking is on when the profile
says so, adaptive thinking with `output_config.effort` for adaptive-style models;
Gemini `thinkingLevel` or `thinkingBudget`; Chat `reasoning_effort` or the dialect's
parameter, optionally mirrored into `chat_template_kwargs.reasoning_effort` without
overwriting client keys.

#### 9.3.3 Turning reasoning off

`ModelProfile.Off` decides what "off" (`effort: none`, `thinking.disabled`,
`thinkingBudget: 0`) becomes:

- `Unsupported`: `unsupported` error;
- `Patch`: a declared set of request fields merged into the upstream body (validated
  against a whitelist: `reasoning_effort`, `reasoning`, `enable_thinking`,
  `chat_template_kwargs`, `think`, `thinking`; an invalid patch is a `config` error);
  reasoning output is hidden from the client;
- `LowestHidden`: for models that always reason, the lowest effort is sent and the
  reasoning is hidden from the client.

#### 9.3.4 Provenance and the signature envelope

Signatures and sealed reasoning go back only to their issuer: a builder emits them only
when `Origin.Format` equals the upstream format and the provider ids match (an empty
id on either side matches). Treating an empty id as a match is only a default for
single-provider callers: a gateway with several upstream providers behind the same
format **must** set `ProviderID`, otherwise a signature issued by one provider is sent
to another and rejected. Otherwise they are dropped (`reasoning.signature` loss);
Anthropic thinking blocks without a usable signature are dropped whole, because
Anthropic rejects them.

Clients send back whatever they received, so provenance has to survive a round trip
through a client of another format. `ir.SealCodec` encodes
`{origin format, provider, kind, value}` into the client format's opaque field:

| Client    | Field                                                     |
|-----------|-----------------------------------------------------------|
| anthropic | `thinking.signature`, `redacted_thinking.data`            |
| responses | reasoning item `encrypted_content` (only when the client asked for it in `include`) |
| gemini    | `thoughtSignature`                                        |
| chat      | tool call id suffix, for Gemini call signatures only      |

The default codec writes `lpb1.<base64url(record)>.<checksum>`; a key turns the
checksum into an HMAC. Parsers recognise the envelope and restore `Origin`; values
without it are attributed to the client's own format with an unknown provider. When the
issuer and the client format are the same and no `ProviderID` is set, raw values pass
through without an envelope. Callers can supply their own codec, for example to accept
an earlier envelope format during a migration.

Reasoning from an upstream that issues no signature (Chat `reasoning_content`) is also
enveloped for Anthropic and Responses clients, so that a later turn
can replay it to an upstream that wants reasoning replay (chat dialect
`ReasoningReplay`, with a character and item budget keeping the newest items, and
Anthropic `context_management` `clear_thinking` `keep` honoured) and can strip it
before an upstream that would reject it. For Anthropic clients the envelope in
`signature` holds only provenance and a checksum of the block's text: the text is
already in the `thinking` block, and copying it would double the history size every
turn. For Responses clients the envelope in `encrypted_content` carries the text,
because the item has no other place for it. `anthropic.StripForeignThinking` removes such
blocks from a native Anthropic request body, for gateways that relay natively after a
converted turn.

#### 9.3.5 Presentation

`Options.Reasoning`: `native` (default) shows reasoning in the client's form; `summary`
shows it as a Responses summary (Responses clients that requested a summary receive the
upstream's raw reasoning as the summary text, otherwise `reasoning_text` content);
`hidden` drops it. A Responses client never receives a reasoning item with `content`
from another provider.

### 9.4 Media

- Images: base64 and URL in all formats; `file_id` only towards the same format.
  Anthropic URL sources must be http(s).
- The library never downloads. `ir.ResolveMedia(ctx, req, fetch)` replaces URLs with
  inline data through a caller function; a URL the upstream cannot take is
  `unsupported` with its path.
- Files: PDFs map to Anthropic `document`, Responses `input_file`, Chat `file` (when
  the chat dialect allows file parts) and Gemini `inlineData` / `fileData`;
  plain-text documents become text (`title`, `context` and body). A caller that turns
  PDFs into text edits the IR before building. Media types are never relabelled.
- Audio and video: Chat `input_audio`, Gemini audio / video parts; other directions
  `unsupported`.

### 9.5 Structured output

`OutputFormat{Type: text | json_object | json_schema, Name, Schema, Strict}`: Chat
`response_format` (`name` defaults to `output`, `strict` to true), Responses
`text.format`, Anthropic `output_config.format` (`json_schema` only; `json_object` is
`unsupported`), Gemini `responseMimeType` + `responseJsonSchema`.

### 9.6 Stop sequences

Native mapping (`stop`, `stop_sequences`, `stopSequences`) with each format's limits
(Anthropic: 64 entries, 16 KiB total, no empty entry; Chat: 4 entries; Gemini: 5).

Anthropic clients expect `stop_reason: stop_sequence` with the matched value, which
Chat and Responses upstreams do not report, and some engines match stop strings inside
reasoning. `StopMode: Emulate` does not send the sequences; a stream filter applies
them:

- only output text is matched (never reasoning or tool arguments); matching does not
  span a reasoning block or tool call;
- the first match is the one that ends earliest; ties go to the longer sequence;
- a suffix that could start a sequence is held back (UTF-8 safe) and flushed at the
  end of the response or before a reasoning block or tool call;
- on a match: the text before it is emitted, buffered tool calls are dropped,
  `stop_sequence` and its value are reported, and `Done()` turns true;
- non-streamed clients are served by streaming from the upstream and collecting
  (`Plan.UpstreamStream = true`); if the upstream answers without streaming, the cut is
  applied to the whole answer.

### 9.7 Prompt caching

Anthropic `cache_control` (type and `ttl`) is kept on instructions, parts (including
`tool_use` and `tool_result`) and tools and reproduced towards Anthropic. Chat clients
may send the same object, following the OpenRouter and LiteLLM conventions: on `text`,
`refusal`, `image_url` and `file` content parts, on assistant `tool_calls` entries, on a
message (applied to its last content part) and on a tool definition next to `function`.
A marker the parser cannot carry is a `cache.marker` loss rather than being ignored: on an
`input_audio` or `thinking` part, on `tool_calls` of a non-assistant message, a
message-level marker that differs from the one the target part already carries, and any
non-null `cache_control` at another position (request level, inside `image_url`, `file` or
`function` objects, ...). That last check walks the request only when it mentions
`cache_control`, reads each object's members last-wins, and skips user data: tool parameter
schemas, `response_format` and `metadata`. Markers on a Chat `tool` message are moved
to the outer `tool_result`, whose breakpoint covers all of its content; the message-level
marker wins over inner ones, otherwise the last inner one, and each superseded marker is a
`cache.marker` loss. A `cache_control` member repeated within one object is read
last-wins, as Anthropic reads it: only the effective member is validated, counted and
checked for TTL order, and a last member of `null` means no marker.

Anthropic accepts at most four breakpoints per request and rejects a marker on an empty
text block. The builder drops markers on empty text, then, above four, keeps the first
marker in render order (tools, system, messages; usually the stable prefix) and the last
three (the most recent turns). Anthropic also requires `1h` breakpoints to precede `5m`
ones; a `1h` marker after a `5m` one is shortened to `5m`. Each dropped or shortened
marker is a `cache.marker` loss.

`anthropic.Dialect.AutoCache` adds breakpoints only when the client wrote none. Any
non-null client `cache_control` disables it, including markers the Chat parser drops as
invalid (`ir.Request.DroppedCacheMarkers`) and markers the builder drops (a folded system
message, a hosted tool removed by `DisableWebSearch`):

| Mode           | Breakpoints                                                              |
| -------------- | ------------------------------------------------------------------------ |
| `""` (off)     | none                                                                     |
| `prefix`       | last function tool; last system block                                    |
| `conversation` | as `prefix`, plus the last cacheable block of the final message and of the user message before it |

A cache write costs more than plain input (1.25x for the 5-minute TTL) and a read costs a
fraction (0.1x on most models), so `prefix` limits the write premium to the shared system
prompt and tools, while `conversation` also writes every new turn so that the next turn
reads it. Automatic breakpoints use the 5-minute TTL and are never placed on thinking
blocks or empty text.

Towards other upstreams marks are a `cache.marker` loss (allowed: those providers cache
implicitly). `prompt_cache_key` and `prompt_cache_retention` are forwarded where the
upstream accepts them.

### 9.8 Hosted tools

`BuiltinTool` represents web search, web fetch, code execution and URL context. Native
equivalents are mapped (Chat `web_search_options`, Responses `web_search*`, Anthropic
`web_search_*` / `web_fetch_*`, Gemini `googleSearch` / `urlContext` /
`codeExecution`), and so is the response side (search activity, citations,
`groundingMetadata`). Without an equivalent the builder returns `unsupported`, unless
the caller replaced the builtin with its own tool (7.6). Responses `web_search` with
`external_web_access: false` is a client hint that the gateway may treat as droppable
(`tool.cached_web_search` loss, allowed). Search counts are reported in
`Usage.WebSearchRequests` for streamed and non-streamed replies alike.

### 9.9 Multiple candidates

`Candidates > 1` maps to Chat `n` and Gemini `candidateCount`; towards Anthropic and
Responses it is `unsupported`.

### 9.10 Sampling parameters

`temperature`, `top_p`, `top_k`, penalties, `seed` and logprobs are mapped where the
upstream has them and range-checked against the client format's rules. Ranges are not
rescaled; a value valid for the client but outside the upstream's range is
`unsupported`, unless the caller allows the `sampling.clamp` loss, in which case the
value is clamped to the upstream's range and the loss is reported.

### 9.11 Responses state

Parsing keeps `previous_response_id`; building for an upstream other than a stateful
Responses upstream fails with `unsupported` unless the caller cleared it after
prepending the stored history. `responses.HistoryOf(req, resp)` returns the IR messages
to store for a completed response. `conversation`, `background`, `item_reference` and
`/v1/responses/compact` are `unsupported` in converted directions.

### 9.12 Anthropic count_tokens

`anthropic.ParseRequest` with the `count_tokens` variant validates the body like a
Messages request without requiring `max_tokens`, so a gateway can validate before
estimating locally or calling a counting endpoint.

## 10. Dialects and model profiles

### 10.1 Dialects

A dialect describes how an upstream deviates from the format it implements. No host
names appear in the library; callers choose dialect values per provider.

- **chat.Dialect**: `MaxTokensField` (`max_tokens` | `max_completion_tokens`);
  `ReasoningParam` (`reasoning_effort` | `enable_thinking` | `thinking` object |
  `extra_body.google.thinking_config` | none); `EffortInTemplateKwargs`;
  `ReasoningReplay` (field, placeholder on tool-call turns, max characters, max
  items); `DeveloperRole`; `StrictTools`; `FileParts`; `StreamUsage`;
  `PromptExcludesCache`, `CacheReadFields`, `CacheWriteFields`,
  `ReasoningExcludedFromCompletion` (usage interpretation); `ThinkTags`;
  `SignaturesInExtraContent`; `MidSystem` (keep | fold).
- **anthropic.Dialect**: `MidSystem`; `AutoCache` (off | prefix | conversation); `StrictTools`; `DefaultMaxTokens`; `DisableWebSearch`.
- **gemini.Dialect**: `SchemaField`; `URLFileData`; `FunctionCallIDs`; `MidSystem`.
- **responses.Dialect**: `StoreDefault`; `CacheWriteField`.

### 10.2 Model profiles

```go
type ModelProfile struct {
	ReasoningStyle   ReasoningStyle // none | effort | budget | adaptive | level
	Efforts          []Effort
	DefaultEffort    Effort
	BudgetToEffort   []BudgetStep
	EffortToBudget   map[Effort]int
	MinBudget        int
	Off              OffRule        // Unsupported | Patch(fields) | LowestHidden
	DefaultMaxTokens int
	SamplingWithReasoning bool
}
```

Profiles are data supplied by the caller; the library ships documented defaults per
upstream format.

## 11. Testing

- **Unit and golden tests** per format function, including the magpie-derived golden
  pairs (section 12) and recorded streams.
- **Combination matrix**: a scripted fake upstream per upstream format; every client
  format against every upstream format (4 x 4, Responses as an upstream included),
  streaming and non-streaming, at least one case
  per capability in section 9; unsupported capabilities asserted to fail with the
  expected `Code`, `Path` and `Feature`.
- **Grammar checkers** shared by all tests: Anthropic (one `message_start`, one open
  block, start/deltas/stop per index, `message_stop` last, nothing after `error`),
  Responses (`created` / `in_progress` first, `sequence_number` equals position, one
  item in progress, `output_index` consistent, terminal event once and last, every
  `output_item.done` equals the final `output` entry), Chat (one index per call,
  `[DONE]` last) and Gemini.
- **Truncation consistency**: for every client format and tool release mode, a
  `max_tokens` end after a tool call has started is reported as truncation and never
  as a completed tool call.
- **Regression list**: every behaviour specified in sections 7 to 9 has a
  named test (validation rules with paths, effort tables via a profile, tool release,
  stop emulation, tool-result images, mid-conversation instructions, usage
  semantics, finish handling), plus the targets in section 13.
- **Round-trip properties** and **fuzzing** of parsers and decoders (no panics, paths
  on errors).
- **Usage invariants** for every direction.
- **Live tests** (`LIVE_TEST=1` plus provider environment variables) through
  `examples/liveproxy`; skipped by default, never in CI; keys only from the environment
  and masked in output. Recorded replies used as fixtures have all headers removed and
  all identifiers other than response and call ids replaced.
- **Benchmarks**: request conversion per direction; per-event decode and encode cost
  and allocations; whole recorded streams.

## 12. Porting scope

The design borrows from magpie (MIT, commit `d4ad1d83afb66328863e3bd53e540a837c8c6892`).
Its gateway code is internal and coupled to routing and subscription logic, so no code
is imported: ideas are re-implemented, and a small set of test samples is copied with a
source note in each file and the MIT notice in `THIRD_PARTY_NOTICES`.

Ported ideas: parts with provenance, reasoning block boundaries, cumulative usage
snapshots, collector-based non-streamed rendering; the decoder robustness rules of 7.1;
id codecs (deterministic ids, Anthropic-safe ids, Gemini name rules, Responses item id
prefixes); tool-history structural rules and optional repairs; Codex-oriented
Responses features (custom tools, namespaces, `allowed_tools`, `phase`).

Samples: the synthetic request/expected-body pairs, and the already scrubbed Claude Code
conversation capture (remaining session UUIDs replaced). Before any copied or adapted
sample is committed it is checked again for user names, local paths, host names,
installation and session identifiers, run through the confidentiality grep and
gitleaks, and given a header naming its source and the magpie commit. Larger client captures are
used only as structure references; any of them added as fixtures is rebuilt with
synthetic content, paths and identifiers.

Not ported: subscription bridging, advisor tools, gateway-side search, learning from
provider refusals, server-side reasoning stores, provider-specific prompt rewriting,
billing extensions, and patches keyed by host or model name (generic ones become dialect
fields).

## 13. Regression targets

Behaviours that converters in this space commonly get wrong; each is a named test:

| #  | Behaviour |
|----|-----------|
| 1  | Parallel tool calls streamed to a Chat client get distinct `index` values |
| 2  | Non-streamed replies keep every text block, not only the last |
| 3  | Tools without parameters are kept; non-object schemas are rejected |
| 4  | Assistant text and tool calls in the same turn are both kept, in every direction |
| 5  | Cache reads and writes are not double-counted in prompt or input tokens |
| 6  | Multiple reasoning segments are concatenated, not overwritten |
| 7  | `n` maps to `candidateCount`, or fails explicitly |
| 8  | Tool `strict` is carried, or reported |
| 9  | Image URLs are sent as URLs, resolved by the caller hook, or rejected; never as broken data URLs |
| 10 | An upstream reply with no candidates is an error, not a success |
| 11 | Hosted-search counts are reported in streamed replies as well |
| 12 | Gemini text and `functionCall` in the same turn are both kept |
| 13 | Text and tool calls in the same Responses reply are both kept |
| 14 | Gemini `functionCall.id` is preserved |
| 15 | Gemini clients receive Gemini-shaped errors |
| 16 | Developer and mid-conversation system messages are mapped, never forwarded with an unknown role |
| 17 | A chunk with both reasoning and text yields both |
| 18 | Response envelopes carry the client's model name and library-generated ids |
| 19 | Non-image files are never sent as images |
| 20 | Every unsupported field fails with a path instead of being dropped |

## 14. Documentation and compatibility

- The README support matrix is generated from (or kept one-to-one with) the loss keys of
  8.3 and the capabilities of section 9; a test fails when a loss key or feature key used
  in code is missing from the matrix.
- The library's defaults are stricter than many existing converters, which drop
  unsupported input silently. An integration should list, per field, the cases where
  the library now returns an error, and decide for each whether to tolerate it through
  `Options.Allow` or accept the new 400.
- Tool-result images use one placeholder and label format for every client format
  (9.2.3); integrations replacing converters with other wording should record this as
  an intentional change.

## 15. Review decisions

1. Unknown parameters are ignored and reported by default; unknown content (block or
   part types, roles, item types, tool types) is always rejected (8.4).
2. Non-zero penalties and requested logprobs towards upstreams without them fail by
   default; neutral values are dropped silently; `seed` is an allowed loss (8.3).
3. Tool release defaults differ per client format (7.3); a `max_tokens` end always
   wins over `tool_use` (7.4).
4. The library's combination tests cover 4 x 4 directions, Responses as an upstream
   included.
5. Tool-result image wording is unified (9.2.3, 14).
