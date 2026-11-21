# Security Policy

## Reporting a vulnerability

Email security issues to the maintainer rather than filing a public issue.
Open a GitHub Security Advisory on the repository, or contact the maintainer
directly at the address listed on the GitHub profile.

Please include:

- A description of the issue and its impact.
- Steps to reproduce, including a minimal request body if applicable.
- The format pair(s) affected (e.g. `chat → anthropic`).
- Any suggested fix.

You will receive an acknowledgment within 72 hours. Please do not disclose the
issue publicly until a fix is released.

## Scope

This library is pure-function: it parses and serializes LLM API request and
response bodies, and does not make network calls, read or write storage, or
execute user-supplied code. Security issues in scope are those where the
library mishandles untrusted input in a way that affects a caller — for
example, producing incorrect output that masks a failure, or panicking on
malformed input.

Out of scope:

- How callers transport the converted bodies (use TLS, authentication, and
  rate limiting in your transport layer).
- The security properties of upstream provider APIs themselves.
- Key management: the library never handles API keys. Callers are responsible
  for keeping keys out of logs and request bodies.

## Supported versions

Only the latest release line receives security fixes.
