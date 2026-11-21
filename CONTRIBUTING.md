# Contributing

Contributions are welcome. Please open an issue before starting work on a
non-trivial change so we can align on scope.

## Development

```
go test ./...
go vet ./...
```

Tests live beside the code (`foo_test.go` in the same package). Streaming
decode paths have unit tests covering: a complete stream, a truncated stream,
interleaved tool-call deltas, a usage-only final chunk, and an error event
mid-stream.

When adding a conversion path or changing wire handling, also add or update
the golden-file fixtures under `bridge/testdata/` and run:

```
go test ./bridge/ -update.golden
```

## Confidentiality

This is a public repository. Before pushing:

- No internal hostnames, IPs, cluster names, or internal repo paths.
- No real API keys, Authorization headers, or full prompt content. Test keys
  come from environment variables only — never write them to files, logs, or
  test output.
- No personal names (other than the commit author), employee IDs, or
  departments.
- No wording that implies internal origin ("our gateway", "internal
  evaluation", etc.).
- Run `gitleaks detect --source .` and a plain grep for the patterns above
  before every push. Report both results in the PR description.

Fixtures derived from recorded upstream responses must have request headers
removed and all identifiable information scrubbed except response and call IDs.

## Commit messages

Conventional Commits style (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`,
`perf:`, `chore:`). Keep the subject line under 72 characters.

Commits and PRs authored with Claude Code assistance include the standard
co-author trailer.
