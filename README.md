# llm-protocol-bridge

Pure-Go library for converting LLM API requests and responses between the four
major provider wire formats, with no HTTP or storage dependencies.

## Formats

| Format    | Provider | Endpoint                                |
|-----------|----------|-----------------------------------------|
| chat      | OpenAI   | `/v1/chat/completions`                  |
| responses | OpenAI   | `/v1/responses`                         |
| anthropic | Anthropic| `/v1/messages`                          |
| gemini    | Google   | `:generateContent` / `:streamGenerateContent` |

All 16 conversion directions are supported through a unified intermediate
representation (`ir`): 12 cross-format conversions (each of the four formats to
the other three) plus 4 same-format passthroughs. Each format implements four
operations: parse (wire request → IR), build (IR → wire request), decode
(wire response/SSE → IR), encode (IR → wire response/SSE).

## Design

- **Pure functions.** No HTTP client, no storage, no global state. Bring your
  own transport.
- **No silent drops.** When an upstream field has no target-format equivalent
  and the loss isn't explicitly allowed by the caller, conversion returns an
  error naming the field path, not a silent approximation.
- **Exact usage accounting.** `ir.UsageFields` is a bitfield tracking which
  token categories a provider reported, so cache writes, reasoning tokens, and
  tool-use prompt surcharges are never double-counted across conversions.
- **Deterministic output.** Given the same input and the same `ir.Env`, the
  output is byte-stable. Map iteration is sorted where it affects output.
- **Streaming.** Decoders are state machines fed SSE events; encoders emit
  client-facing SSE. Tool-call release modes (Buffered / LiveFirst / Stream)
  match each provider's streaming contract.

See [`docs/design.md`](docs/design.md) for the full specification.

## Capability Matrix

Support by target upstream format (all 4 client formats parse into the shared IR; unsupported features return a structured `*ir.Error` with `Code` and `Path` unless listed as a configurable loss key):

| Capability | → Chat | → Responses | → Anthropic | → Gemini |
|---|---|---|---|---|
| Text, multi-turn, system/developer | Full (`system`/`developer` via dialect; `MidSystem: "fold"` wraps mid-turn notes in `<system_message>`) | Full (`instructions` + mid-turn roles) | Full (`system` + consecutive role merge + mid-turn `system`) | Full (`systemInstruction` + `contents`) |
| Streaming & Collector | Full SSE + `[DONE]` | Full `response.*` SSE (`sequence_number`, `reasoning_text`/`summary_text`, optional `TrimLeadingNewlines`) | Full `message_*`/`content_block_*` SSE | Full JSON array stream & `alt=sse` |
| Function tools & parallel calls | Full (interleaved + sorted close + name/ID dedup) | Full (sequential item stream; truncated live calls omit `output_item.done`) | Full (`AnthropicSafeID` + sequential blocks) | Full (ID resolution + deterministic IDs) |
| Custom & namespaced tools | Full (wrapped `{input}` + `<ns>__<name>`) | Full (`custom`, `namespace`) | Function mapping | Function mapping |
| `tool_choice` (`auto`/`none`/`required`/`tool`/`allowed`) | Full (`allowed` filters tools; omits `auto`/`none` when no tools) | Full (`allowed_tools` with `mode` + `tools`) | Full (`type: "none"`, `allowed` filters tools) | Full (`AUTO`/`NONE`/`ANY` + `allowedFunctionNames`) |
| Tool result images, documents & `is_error` | Lifted follow-up user turn + `[tool error]`; text docs preserved, binary/PDF gated by `FileParts` | Text output + `is_error` | Native blocks + `search_result` in `tool_result` + `is_error` | `{output}` / `{error}` object |
| Tool `strict` schema | Preserved | Preserved | `tool.strict` loss (unless dialect enables) | `tool.strict` loss |
| Reasoning (effort / budget / off / adaptive) | Full via `ModelProfile.Resolve` | Full via `ModelProfile.Resolve` | Full (`budget_tokens` + `max_tokens` & sampling constraints) | Full (`*int thinkingBudget` incl. `0`, `thinkingLevel`) |
| Signatures & sealed reasoning (`Origin.Matches` + `SealCodec`) | `reasoning.signature` / `reasoning.sealed` loss | Native `encrypted_content` + seal | Native `thinking`/`redacted_thinking` + seal | Native `thoughtSignature` |
| Structured output (`json_schema` / `json_object`) | Full (`response_format`) | Full (`text.format`) | `json_schema` via `output_config.format`; `json_object` → `unsupported` | Full (`responseSchema` + `responseMimeType`) |
| Hosted builtins (`web_search`, etc.) | `unsupported` (use `Options.Intercept`) | `web_search`; others → `unsupported` | `web_search` / `web_fetch`; others → `unsupported` | `googleSearch`, `codeExecution`, `urlContext` |
| Multiple candidates (`n > 1`) | Full | Single candidate | `unsupported` (`CodeUnsupported`) | Full (`candidateCount`) |
| Stop sequences | Up to 4 (`stop.limit` loss) | `stop.limit` loss (or `StopEmulate`) | Full (`stop_sequences`) | Up to 5 (`stop.limit` loss) |
| Usage (input, output, cache, reasoning, server tools) | Full (incl. dialect cache fields) | Full (incl. dialect `CacheWriteField`) | Full (non-inclusive input, 5m/1h split, `server_tool_use`) | Full (`thoughtsTokenCount`, `toolUsePromptTokenCount`) |
| Error conversion (`ConvertError` / `ErrorResponse`) | Standard OpenAI wire error types | Standard OpenAI wire error types + `response.failed` | Anthropic error envelope + SSE `error` | Gemini RPC error envelope (JSON & SSE) |
| Stateful features (`conversation`, `background`, `item_reference`, cross-format `previous_response_id`) | Rejected (`unsupported`) | Identity `previous_response_id` only; others rejected | Rejected (`unsupported`) | Rejected (`unsupported`) |

## Loss keys

When a request or response field has no equivalent in the target format, the
library records a *loss* keyed by a stable identifier. Each loss is either
tolerated (recorded in `Plan.Losses`) or rejected (returned as an error),
depending on the caller's `Options.Allow` / `Options.Deny` sets. The default
column shows the behaviour when the caller passes `bridge.Options{}`.

| Loss key                   | Meaning                                          | Default   |
|----------------------------|--------------------------------------------------|-----------|
| `reasoning.signature`      | Per-block reasoning signature cannot be carried  | allowed   |
| `reasoning.sealed`         | Encrypted reasoning payload cannot be carried    | allowed   |
| `reasoning.summary`        | Reasoning summary style cannot be carried        | allowed   |
| `reasoning.display`        | Reasoning display mode cannot be carried         | allowed   |
| `request.user`             | `user` field cannot be carried                    | allowed   |
| `request.metadata`         | `metadata` field cannot be carried               | allowed   |
| `request.service_tier`     | `service_tier` cannot be carried                 | allowed   |
| `request.prompt_cache`     | Prompt-cache marker cannot be carried            | allowed   |
| `request.store`            | `store` flag cannot be carried                   | allowed   |
| `request.include`          | `include` list cannot be carried                 | allowed   |
| `request.verbosity`        | `verbosity` cannot be carried                     | allowed   |
| `request.max_tool_calls`   | `max_tool_calls` cannot be carried                | allowed   |
| `request.unknown_field`    | Unknown parameter ignored on parse               | allowed   |
| `cache.marker`             | Cache-control marker cannot be carried           | allowed   |
| `image.detail`             | Image `detail` hint cannot be carried            | allowed   |
| `tool.strict`              | Tool `strict` flag cannot be carried             | allowed   |
| `tool.parallel`            | `parallel_tool_calls` cannot be carried          | allowed   |
| `tool.deferred`            | Deferred tool-call mode cannot be carried        | allowed   |
| `tool.cached_web_search`   | Cached web-search state cannot be carried        | allowed   |
| `tool.schema`              | JSON Schema constraints dropped on OpenAPI downgrade | allowed |
| `output.citations`         | Citation metadata cannot be carried              | allowed   |
| `output.builtin_activity`  | Hosted builtin call/result cannot be carried     | allowed   |
| `output.media`             | Non-text output media cannot be carried          | allowed   |
| `sampling.seed`            | `seed` cannot be carried                          | allowed   |
| `sampling.top_k`            | `top_k` cannot be carried                         | denied    |
| `sampling.penalty`         | `presence_penalty` / `frequency_penalty` dropped | denied    |
| `sampling.logprobs`        | `logprobs` cannot be carried                      | denied    |
| `sampling.clamp`           | `temperature` clamped to the target's max        | denied    |
| `stop.limit`               | Extra stop sequences beyond the target's limit   | denied    |

Losses marked *denied* by default return an error unless the caller explicitly
adds the key to `Options.Allow`. Losses marked *allowed* are recorded silently
and can be read without setting `Options.OnLoss`: request-phase losses appear
in `Plan.Losses`, non-streaming / collector / `StreamResponseWithResult`
response-phase losses appear in `ir.Response.Losses`, and streaming
response-phase losses appear in `Stream.Losses()`. Add any key to
`Options.Deny` to turn it into an error instead, or supply `Options.CheckLoss`
for fine-grained path- and detail-based rejection (for example rejecting
`mcp_servers`, `truncation: "auto"`, unsupported `include` entries, or unknown
`context_management` edits while keeping other unknown parameters allowed).

## Policy & Dialect Options

`bridge.Options` exposes explicit switches for stricter upstream or client
policies while keeping zero-value defaults backward-compatible:

| Option | Default | Behaviour when enabled |
|---|---|---|
| `CheckLoss func(ir.Loss) error` | `nil` | Called for every recorded loss before `DefaultAllowed`/`Allow`/`Deny`; returning a non-nil error rejects the conversion with that error. |
| `OmitUserID bool` | `false` | Suppresses forwarding `Request.User` (such as Anthropic `metadata.user_id`) to Chat Completions, Responses, and Anthropic upstreams. |
| `OmitHistoryReasoning bool` | `false` | Omits historical assistant thinking/reasoning text when building Chat Completions requests. |
| `SealUnsignedReasoning bool` | `false` | Uses `Options.Seal` to stamp unsigned upstream reasoning blocks (`signature` in Anthropic, `encrypted_content` in Responses when requested via `include`) so verified reasoning can be replayed on later turns. |
| `GenerateResponseID bool` | `false` | Forces freshly generated format-prefixed response IDs (`msg_` for Anthropic, `resp_` for Responses) instead of passing through upstream IDs. |
| `RawReasoningAsSummary *bool` | `nil` (`true`) | Controls whether raw upstream reasoning text is emitted as Responses `summary_text` when the client requested `reasoning.summary` (`auto`/`concise`/`detailed`); `ir.Ptr(false)` always emits `reasoning_text`. |
| `TrimLeadingNewlines bool` | `false` | Strips leading `\r`/`\n` characters before opening a Responses assistant message item (and omits newline-only messages before tool calls) for upstreams that emit leading blank lines; default `false` preserves model output verbatim. |
| `Chat.MidSystem` (`"fold"`) | `""` (in place) | Wraps mid-conversation `system`/`developer` messages in `<system_message>...</system_message>` and folds them into adjacent `user` turns (after any `tool` messages and before lifted tool-result images and user text). |
| `Chat.ReasoningReplay` | zero value | Configures historical reasoning replay into Chat Completions messages (`Field`, `Placeholder`, rolling `MaxChars` / `MaxItems` budgets, and `RequireVerified` provenance gating). |
| `Chat.FileParts bool` | `false` | Allows cross-format binary/PDF/`file_id` parts to be forwarded as Chat `type: "file"` parts; when `false`, unsupported cross-format binary files return `CodeUnsupported` (plain-text documents are rendered as text). |
| `Limits.MaxBufferedBytes` | `8 MiB` | Caps buffered tool arguments and deferred multi-candidate stream events in the filter pipeline and stream encoders. |
| `Limits.MaxStopSequences` / `MaxStopSequenceBytes` | `0` (unlimited) | Caps stop sequence count and total byte length accepted by `bridge.Build` (e.g. `64` and `16384` under `StopEmulate`). |

## Known Limitations

- **Chat streaming tool-name deduplication**: OpenAI-compatible Chat streaming
  upstreams vary in how they transmit tool call names (single chunk, split
  fragments, repeated full name on every chunk, or prefix followed by full
  name). `chat.StreamDecoder` merges incoming name fragments deterministically
  without guessing from other declared tools: an identical fragment is treated
  as a resend, a fragment prefixed by the current accumulated name replaces it,
  and all other fragments are appended. Consequently, a genuine split into
  identical consecutive fragments (e.g. `"foo"` followed by `"foo"` for a tool
  named `"foofoo"` or `"foofoobar"`) is indistinguishable from a resent chunk on
  the wire and collapses the repeated fragment (`"foo"` / `"foobar"`).

## Install

```
go get github.com/ihainan/llm-protocol-bridge
```

Requires Go 1.24 or later. Standard library only — no external dependencies.

## Quick start

Convert a Chat Completions request to Anthropic Messages:

```go
import (
    "github.com/ihainan/llm-protocol-bridge/bridge"
    "github.com/ihainan/llm-protocol-bridge/ir"
)

// client=chat (the caller's format), upstream=anthropic (where you will send it).
plan, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic, chatBody, bridge.Options{})
if err != nil {
    return err
}
// plan.Body is the Anthropic request body. Send it to Anthropic.
// plan.Losses lists any fields that could not be expressed in the target format.
```

Stream an upstream Anthropic response back to a client that expects Chat SSE:

```go
s := plan.NewStream()
// Feed upstream SSE events (from Anthropic):
out, err = s.Feed(out, upstreamEvent)
// On end:
out, err = s.End(out, ir.EndDone)
```

See [`examples/liveproxy`](examples/liveproxy) for a minimal HTTP proxy that
wires the library to a real transport.

## Status

Phase A (the library) is complete. The conversion core, streaming pipeline,
and test suite are in place. Phase B (packaging, broader tool coverage) is
planned.

## License

MIT. See [LICENSE](LICENSE).
