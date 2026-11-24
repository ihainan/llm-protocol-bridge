# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `sampling.top_k`, `sampling.penalty` and `sampling.clamp` are allowed by
  default (`ir.DefaultAllowed`): a `top_k` towards Chat Completions or
  Responses, a penalty towards Anthropic / Gemini / Responses, a temperature
  above Anthropic's maximum, and `temperature` / `top_p` / `top_k` while
  Anthropic thinking is on are dropped or clamped and reported as losses
  instead of failing the request. Gemini CLI sends `topK` and `topP` on every
  request. Add the keys to `Options.Deny` for the previous behaviour.
  `sampling.logprobs` stays denied.

### Fixed

- Gemini responses no longer carry `"parts": null` or `"candidates": null`:
  an empty `parts` list and an empty `candidates` list are left out, and a
  candidate without parts (the final stream chunk with only `finishReason`)
  has no `content`, as the Gemini API sends it. Gemini CLI read `.length` of
  these and failed at the end of every converted stream.

### Added

- Prompt cache markers from Chat Completions clients: `cache_control` on
  `text` / `refusal` / `image_url` / `file` content parts, on assistant
  `tool_calls` entries, on messages (applied to the
  last content part, LiteLLM convention) and on tool definitions (next to
  `function`) is carried to Anthropic. Markers on a `tool` message move to the
  outer `tool_result`; when several are present only the message-level one (or
  else the last inner one) is kept and the others are `cache.marker` losses.
  Invalid shapes (non-object, type other than `ephemeral`, TTL other than
  `5m` / `1h`) and markers at positions that cannot carry one (`input_audio` /
  `thinking` parts, `tool_calls` of non-assistant messages, a message-level
  marker that differs from the part's own, and any other position such as
  request level or inside `image_url` / `file` / `function` objects; parameter
  schemas, `response_format` and `metadata` are not searched) are dropped as
  `cache.marker` losses. A `cache_control` member
  repeated within one object is read last-wins, and only the effective member
  is validated. Chat, Responses and Gemini upstreams record `cache.marker`
  losses as before.
- `anthropic.Dialect.AutoCache` (`AutoCacheMode`: `""` off, `"prefix"`,
  `"conversation"`) places cache breakpoints automatically when the client
  wrote no `cache_control` of its own; markers on hosted tools the dialect
  drops and markers dropped as invalid (`ir.Request.DroppedCacheMarkers`) also
  count as client markers. It replaces the previously unimplemented boolean
  field of the same name.
- The Anthropic builder enforces the four-breakpoint limit: the first marker in
  render order and the last three are kept, others are dropped as
  `cache.marker` losses. Markers on empty text blocks are dropped the same way,
  and a `1h` marker that follows a `5m` one (rejected by Anthropic) is
  shortened to `5m`; a hosted tool object with a repeated `cache_control`
  member is checked by its last member.
- `bridge/testdata/cache-baseline-v0.1.4.json` pins v0.1.4 request output for
  requests without cache markers.
- Responses `configuration_update` input items (change reasoning effort or
  text format mid-conversation) are accepted. A Responses upstream receives
  them verbatim and in place; other upstreams get the last update as the
  request-level setting, recorded as a `request.configuration_update` loss.
- Codex remote compaction items (`compaction`, `compaction_summary`,
  `compaction_trigger`) are forwarded to Responses upstreams and rejected with
  an `unsupported` error naming the item for other upstreams.
- `ir.ModelProfile.NoForcedToolChoice` relaxes a forced `tool_choice` to
  `auto` for models that reject it (a named tool stays the only tool offered),
  recorded as a `tool.forced_choice` loss.
- `ir.Message.Extras`, `ir.Request.TrailingExtras` and `ir.InputExtra` carry
  Responses input items that have no message equivalent;
  `ir.ApplyInputExtras` folds them into request-level settings and is called by
  the Anthropic, Chat and Gemini builders.

### Fixed

- Anthropic streams: `0` placeholders for input and cache counters in
  `message_delta` no longer erase the counts from `message_start`.
- Chat usage: when `prompt_tokens_details.cached_tokens` is `0` or missing, a
  positive dialect cache field (`Dialect.CacheReadFields`) is used; the two are
  never added.
- Responses `input_file.file_data` data URIs are split into media type and
  data (the prefix used to reach Anthropic's base64 field); `file_url` is
  carried, and the Responses builder writes `file_data` as a data URI (also
  inside tool outputs, which now accept `file_url` too).
- Non-PDF files sent to Anthropic: text files become text documents, other
  types fail with an `unsupported` error naming the accepted types instead of
  an upstream 400.
- Anthropic `tool_use` / `tool_result` `cache_control` and markers on a single
  text block inside `tool_result` were silently dropped; they are now kept.
- Cache markers on Anthropic folded mid-conversation system messages and on
  Responses `instructions` are now reported as `cache.marker` losses instead of
  being dropped silently.

- Conversion core covering all 16 directions across four formats: OpenAI Chat
  Completions (`chat`), OpenAI Responses (`responses`), Anthropic Messages
  (`anthropic`), and Gemini `generateContent` (`gemini`).
- Unified intermediate representation (`ir`) with stable event stream, usage
  bitfield, loss tracking, and collector.
- Streaming pipeline: SSE decoder state machines, filter pipeline, encoder
  state machines. Tool-call release modes (Buffered / LiveFirst / Stream)
  match each provider's contract.
- `bridge.Convert` for non-streaming conversion and `bridge.Plan.NewStream`
  for streaming.
- Unit tests for every format's parse/build round-trip and streaming decode
  path (complete, truncated, interleaved tool-call deltas, usage-only final
  chunk, error mid-stream).
- Golden-file fixtures under `bridge/testdata/` covering cross-format
  conversions.
- Strict client request validation across Responses and Anthropic parsers:
  sampling/penalty ranges, `max_tokens > 0`, non-empty messages/input, duplicate
  tool names, `tool_choice` target checks, `thinking.budget_tokens` bounds,
  `tool_use` JSON object input, `tool_result` role/shape/nested-block/image
  source validation, turn-scoped call/output pairing, object `input_schema`,
  non-empty `stop_sequences`, and versioned `context_management` edit parsing.
- Real client compatibility: Anthropic mid-conversation `role: "system"` support,
  `tool_result` `search_result` rendering and `tool_reference` skipping, Chat
  `MidSystem: "fold"` `<system_message>` folding, text document title/body
  preservation, `\n\n` / `\n` system joining, and omitting `tool_choice` when no
  tools are sent.
- Response alignment: Chat streamed tool name prefix/repeat deduplication and
  per-candidate call ID deduplication; Responses `reasoning_text` vs
  `summary_text` selection, stream/non-stream reasoning item parity,
  `store`/`prompt_cache_key` echoing, and omitting `response.output_item.done`
  on truncated live calls; Anthropic unsigned `thinking` blocks always include
  `"signature": ""`.
- Policy & safety options: `Options.CheckLoss`, `OmitUserID`,
  `OmitHistoryReasoning`, `GenerateResponseID`, `RawReasoningAsSummary`,
  `TrimLeadingNewlines`, and `Limits.MaxStopSequences` / `MaxStopSequenceBytes`.
- Bug fixes: eliminated double-counting of buffered tool bytes in
  `bridge/pipeline.go` and enforced `MaxBufferedBytes` on `deferredOtherCand`
  and queued pending tool buffers in Responses and Anthropic stream encoders.
- Minimal HTTP proxy example under `examples/liveproxy`.
- MIT license, contributing guide, security policy, and CI workflow.
