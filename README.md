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

All 16 conversion directions (including identities) are supported through a
unified intermediate representation (`ir`). Each format implements four
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

plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, chatBody, bridge.Options{
    Client: ir.ClientOptions{Model: "claude-3-5-sonnet"},
})
if err != nil {
    return err
}
// plan.Body is the Anthropic request body. Send it to Anthropic.
```

Stream an upstream Chat response back to a client that expects Anthropic SSE:

```go
s := plan.NewStream()
// Feed upstream SSE events:
out, err := s.Feed(out, upstreamEvent)
// On end:
out, err := s.End(out, ir.EndDone)
```

See [`examples/liveproxy`](examples/liveproxy) for a minimal HTTP proxy that
wires the library to a real transport.

## Status

Phase A (the library) is complete. The conversion core, streaming pipeline,
and test suite are in place. Phase B (packaging, broader tool coverage) is
planned.

## License

MIT. See [LICENSE](LICENSE).
