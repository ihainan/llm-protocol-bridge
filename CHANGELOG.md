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
- Minimal HTTP proxy example under `examples/liveproxy`.
- MIT license, contributing guide, security policy, and CI workflow.
