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
| `output.citations`         | Citation metadata cannot be carried              | allowed   |
| `sampling.seed`            | `seed` cannot be carried                          | allowed   |
| `sampling.top_k`            | `top_k` cannot be carried                         | denied    |
| `sampling.penalty`         | `presence_penalty` / `frequency_penalty` dropped | denied    |
| `sampling.logprobs`        | `logprobs` cannot be carried                      | denied    |
| `sampling.clamp`           | `temperature` clamped to the target's max        | denied    |
| `stop.limit`               | Extra stop sequences beyond the target's limit   | denied    |

Losses marked *denied* by default return an error unless the caller explicitly
adds the key to `Options.Allow`. Losses marked *allowed* are recorded in
`Plan.Losses` silently; add them to `Options.Deny` to turn any of them into an
error instead.

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
