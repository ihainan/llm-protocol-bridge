package ir

import (
	"crypto/rand"
	"encoding/base64"
	"time"
)

// LossKind is a stable key for a lossy conversion.
type LossKind string

// Loss keys. The support matrix lists each with its default.
const (
	LossReasoningSignature LossKind = "reasoning.signature"
	LossReasoningSealed    LossKind = "reasoning.sealed"
	LossRequestUser        LossKind = "request.user"
	LossRequestMetadata    LossKind = "request.metadata"
	LossRequestTier        LossKind = "request.service_tier"
	LossRequestCache       LossKind = "request.prompt_cache"
	LossRequestStore       LossKind = "request.store"
	LossRequestInclude     LossKind = "request.include"
	LossRequestVerbosity   LossKind = "request.verbosity"
	LossRequestMaxCalls    LossKind = "request.max_tool_calls"
	LossUnknownField       LossKind = "request.unknown_field"
	LossCacheMarker        LossKind = "cache.marker"
	LossImageDetail        LossKind = "image.detail"
	LossToolStrict         LossKind = "tool.strict"
	LossToolSchema         LossKind = "tool.schema"
	LossToolParallel       LossKind = "tool.parallel"
	LossToolDeferred       LossKind = "tool.deferred"
	LossCachedWebSearch    LossKind = "tool.cached_web_search"
	LossReasoningSummary   LossKind = "reasoning.summary"
	LossReasoningDisplay   LossKind = "reasoning.display"
	LossSamplingSeed       LossKind = "sampling.seed"
	LossSamplingTopK       LossKind = "sampling.top_k"
	LossSamplingPenalty    LossKind = "sampling.penalty"
	LossSamplingLogprobs   LossKind = "sampling.logprobs"
	LossSamplingClamp      LossKind = "sampling.clamp"
	LossStopLimit          LossKind = "stop.limit"
	LossCitations          LossKind = "output.citations"
	LossBuiltinActivity    LossKind = "output.builtin_activity"
	LossOutputMedia        LossKind = "output.media"
)

// DefaultAllowed lists the losses tolerated unless the caller denies them.
var DefaultAllowed = LossSet{
	LossReasoningSignature: true,
	LossReasoningSealed:    true,
	LossRequestUser:        true,
	LossRequestMetadata:    true,
	LossRequestTier:        true,
	LossRequestCache:       true,
	LossRequestStore:       true,
	LossRequestInclude:     true,
	LossRequestVerbosity:   true,
	LossRequestMaxCalls:    true,
	LossUnknownField:       true,
	LossCacheMarker:        true,
	LossImageDetail:        true,
	LossToolStrict:         true,
	LossToolSchema:         true,
	LossToolParallel:       true,
	LossToolDeferred:       true,
	LossCachedWebSearch:    true,
	LossReasoningSummary:   true,
	LossReasoningDisplay:   true,
	LossSamplingSeed:       true,
	LossCitations:          true,
	LossBuiltinActivity:    true,
	LossOutputMedia:        true,
}

// LossSet is a set of loss keys.
type LossSet map[LossKind]bool

// Loss is one tolerated lossy conversion.
type Loss struct {
	Kind   LossKind
	Path   Path
	Detail string
}

// UnknownPolicy decides what happens to request parameters a parser does not know.
// Unknown content (block types, roles, item types, tool types) is always rejected.
type UnknownPolicy uint8

// Unknown-parameter policies.
const (
	UnknownIgnore UnknownPolicy = iota // drop and report request.unknown_field
	UnknownReject                      // fail with the path
)

// Repair is an optional history repair that changes content.
type Repair string

// Optional repairs.
const (
	RepairMissingResults   Repair = "synthesize_missing_results"
	RepairDuplicateResults Repair = "drop_duplicate_results"
	RepairOrphanResults    Repair = "orphan_results_as_text"
	RepairEmptyAssistant   Repair = "drop_empty_assistant"
)

// RepairSet is a set of enabled repairs.
type RepairSet map[Repair]bool

// IDSource generates ids for response objects. It is never used for upstream
// request bodies, which stay deterministic.
type IDSource interface {
	// NewID returns a random identifier body (no prefix) of about n characters.
	NewID(n int) string
}

type randomIDs struct{}

func (randomIDs) NewID(n int) string {
	b := make([]byte, (n*3+3)/4)
	_, _ = rand.Read(b)
	s := base64.RawURLEncoding.EncodeToString(b)
	// keep ids alphanumeric: some clients validate them
	out := make([]byte, 0, n)
	for i := 0; i < len(s) && len(out) < n; i++ {
		c := s[i]
		if c == '-' || c == '_' {
			c = 'x'
		}
		out = append(out, c)
	}
	return string(out)
}

// RandomIDs is the default IDSource.
var RandomIDs IDSource = randomIDs{}

// Env carries the cross-cutting configuration of one conversion and collects its
// losses. A zero Env is usable; one Env is used by one conversion at a time.
type Env struct {
	// ProviderID scopes signatures and sealed reasoning to an upstream provider.
	ProviderID string
	// Profile describes the upstream model; nil uses the format's defaults.
	Profile *ModelProfile
	// Allow adds tolerated losses to DefaultAllowed; Deny removes them.
	Allow, Deny LossSet
	OnLoss      func(Loss)
	Unknown     UnknownPolicy
	Repair      RepairSet
	// Seal encodes provenance into opaque client fields; nil uses DefaultSeal.
	Seal SealCodec
	// IDs and Clock are used on the response side only.
	IDs   IDSource
	Clock func() time.Time
	// Style is how request paths are rendered (the client format's style).
	Style PathStyle

	losses []Loss
}

// Losses returns the tolerated losses recorded so far.
func (e *Env) Losses() []Loss {
	if e == nil {
		return nil
	}
	return e.losses
}

// Allowed reports whether a loss kind is tolerated.
func (e *Env) Allowed(k LossKind) bool {
	if e != nil {
		if e.Deny[k] {
			return false
		}
		if e.Allow[k] {
			return true
		}
	}
	return DefaultAllowed[k]
}

// Lose records a lossy conversion. It returns nil when the loss is tolerated and
// an unsupported error naming the path otherwise.
func (e *Env) Lose(k LossKind, p Path, detail string) error {
	if !e.Allowed(k) {
		err := Unsupported(p, string(k), detail)
		if e != nil {
			err.Style = e.Style
		}
		return err
	}
	if e == nil {
		return nil
	}
	l := Loss{Kind: k, Path: p, Detail: detail}
	e.losses = append(e.losses, l)
	if e.OnLoss != nil {
		e.OnLoss(l)
	}
	return nil
}

// UnknownParam handles a request parameter the parser does not know.
func (e *Env) UnknownParam(p Path) error {
	if e != nil && e.Unknown == UnknownReject {
		return e.Err(Invalid(p, "unknown parameter"))
	}
	return e.Lose(LossUnknownField, p, "unknown parameter ignored")
}

// Err stamps the client path style on a library error.
func (e *Env) Err(err *Error) *Error {
	if e != nil && err != nil {
		err.Style = e.Style
	}
	return err
}

// Repairs reports whether a repair is enabled.
func (e *Env) Repairs(r Repair) bool { return e != nil && e.Repair[r] }

// Sealer returns the signature codec.
func (e *Env) Sealer() SealCodec {
	if e != nil && e.Seal != nil {
		return e.Seal
	}
	return DefaultSeal
}

// NewID returns a response-side id with the given prefix.
func (e *Env) NewID(prefix string, n int) string {
	src := RandomIDs
	if e != nil && e.IDs != nil {
		src = e.IDs
	}
	return prefix + src.NewID(n)
}

// Now returns the response-side clock's time.
func (e *Env) Now() time.Time {
	if e != nil && e.Clock != nil {
		return e.Clock()
	}
	return time.Now()
}

// ModelProfileOr returns the profile or the given default.
func (e *Env) ModelProfileOr(def *ModelProfile) *ModelProfile {
	if e != nil && e.Profile != nil {
		return e.Profile
	}
	return def
}
