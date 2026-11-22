# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
