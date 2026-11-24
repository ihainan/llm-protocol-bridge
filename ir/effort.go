package ir

import (
	"bytes"
	"encoding/json"
	"sort"
	"strings"
)

// Effort is a normalised reasoning effort level.
type Effort string

// Effort levels, weakest first. EffortNone means "do not reason".
const (
	EffortUnset   Effort = ""
	EffortNone    Effort = "none"
	EffortMinimal Effort = "minimal"
	EffortLow     Effort = "low"
	EffortMedium  Effort = "medium"
	EffortHigh    Effort = "high"
	EffortXHigh   Effort = "xhigh"
	EffortMax     Effort = "max"
)

// EffortOrder lists the reasoning levels weakest first (EffortNone excluded).
var EffortOrder = []Effort{EffortMinimal, EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax}

// Rank is an effort's position in EffortOrder, -1 if unknown.
func (e Effort) Rank() int {
	for i, x := range EffortOrder {
		if x == e {
			return i
		}
	}
	return -1
}

// ParseEffort reads an effort name case-insensitively.
func ParseEffort(s string) (Effort, bool) {
	e := Effort(strings.ToLower(strings.TrimSpace(s)))
	if e == EffortNone || e.Rank() >= 0 {
		return e, true
	}
	return EffortUnset, false
}

// FitEffort returns the level in levels nearest to want; ties go to the stronger
// level. With no levels, want is returned unchanged.
func FitEffort(want Effort, levels []Effort) Effort {
	if len(levels) == 0 || want == EffortUnset || want == EffortNone {
		return want
	}
	w := want.Rank()
	if w < 0 {
		return want
	}
	best, bestDist := EffortUnset, 1<<30
	for _, l := range levels {
		r := l.Rank()
		if r < 0 {
			continue
		}
		d := r - w
		if d < 0 {
			d = -d
		}
		if d < bestDist || (d == bestDist && r > best.Rank()) {
			best, bestDist = l, d
		}
	}
	if best == EffortUnset {
		return want
	}
	return best
}

// BudgetStep maps budgets up to Max (inclusive) to an effort. Max 0 means no limit.
type BudgetStep struct {
	Max    int    `json:"max"`
	Effort Effort `json:"effort"`
}

// DefaultBudgetToEffort reads a thinking budget as an effort level.
var DefaultBudgetToEffort = []BudgetStep{
	{Max: 1024, Effort: EffortMinimal},
	{Max: 4096, Effort: EffortLow},
	{Max: 10000, Effort: EffortMedium},
	{Max: 24000, Effort: EffortHigh},
	{Max: 32000, Effort: EffortXHigh},
	{Max: 0, Effort: EffortMax},
}

// DefaultEffortToBudget gives budget-style upstreams a budget for each level.
var DefaultEffortToBudget = map[Effort]int{
	EffortMinimal: 1024,
	EffortLow:     4096,
	EffortMedium:  10000,
	EffortHigh:    24000,
	EffortXHigh:   32000,
	EffortMax:     48000,
}

// ReasoningStyle is how an upstream model is asked to reason.
type ReasoningStyle string

// Reasoning styles.
const (
	StyleNone     ReasoningStyle = "none"     // the model does not reason
	StyleEffort   ReasoningStyle = "effort"   // an effort level (reasoning_effort)
	StyleBudget   ReasoningStyle = "budget"   // a token budget (thinking.budget_tokens)
	StyleAdaptive ReasoningStyle = "adaptive" // adaptive thinking plus an effort
	StyleLevel    ReasoningStyle = "level"    // a level name (Gemini thinkingLevel)
)

// OffMode is how "reasoning off" reaches an upstream model.
type OffMode string

// Off modes.
const (
	// OffNative uses the upstream format's own switch (the default).
	OffNative OffMode = ""
	// OffUnsupported rejects "off" for this model.
	OffUnsupported OffMode = "unsupported"
	// OffPatch merges declared fields into the upstream body and hides reasoning.
	OffPatch OffMode = "patch"
	// OffLowestHidden sends the lowest level and hides reasoning (models that
	// always reason).
	OffLowestHidden OffMode = "lowest_hidden"
)

// OffRule says how a model turns reasoning off.
type OffRule struct {
	Mode OffMode `json:"mode,omitempty"`
	// Patch is a JSON object merged into the upstream body for OffPatch; client
	// keys win inside nested objects. An empty object means the model does not
	// reason and nothing needs to be sent.
	Patch json.RawMessage `json:"patch,omitempty"`
}

// ModelProfile describes an upstream model's reasoning and output limits. It is
// data supplied by the caller; nil fields use the defaults above.
type ModelProfile struct {
	Style ReasoningStyle `json:"style,omitempty"`
	// Efforts lists the levels the model accepts; requests are fitted to them.
	Efforts []Effort `json:"efforts,omitempty"`
	// DefaultEffort is sent when the client asked for nothing.
	DefaultEffort  Effort         `json:"default_effort,omitempty"`
	BudgetToEffort []BudgetStep   `json:"budget_to_effort,omitempty"`
	EffortToBudget map[Effort]int `json:"effort_to_budget,omitempty"`
	MinBudget      int            `json:"min_budget,omitempty"`
	Off            OffRule        `json:"off,omitempty"`
	// DefaultMaxTokens is used when the upstream requires max_tokens and the
	// client sent none.
	DefaultMaxTokens int `json:"default_max_tokens,omitempty"`
	// SamplingWithReasoning allows temperature/top_p while reasoning is on.
	SamplingWithReasoning bool `json:"sampling_with_reasoning,omitempty"`
	// EffortInTemplateKwargs mirrors the effort into
	// chat_template_kwargs.reasoning_effort (Chat upstreams).
	EffortInTemplateKwargs bool `json:"effort_in_template_kwargs,omitempty"`
	// NoForcedToolChoice marks a model that rejects a forced tool_choice
	// (Anthropic any/tool). Such a choice is relaxed to auto before sending: a
	// named tool becomes auto with only that tool offered, and the change is
	// recorded as a tool.forced_choice loss.
	NoForcedToolChoice bool `json:"no_forced_tool_choice,omitempty"`
}

// BudgetEffort reads a budget as an effort with the profile's table.
func (p *ModelProfile) BudgetEffort(n int) Effort {
	steps := DefaultBudgetToEffort
	if p != nil && len(p.BudgetToEffort) > 0 {
		steps = p.BudgetToEffort
	}
	for _, s := range steps {
		if s.Max == 0 || n <= s.Max {
			return s.Effort
		}
	}
	return EffortMax
}

// EffortBudget gives the budget for an effort with the profile's table.
func (p *ModelProfile) EffortBudget(e Effort) int {
	if p != nil && p.EffortToBudget != nil {
		if n, ok := p.EffortToBudget[e]; ok {
			return n
		}
	}
	if n, ok := DefaultEffortToBudget[e]; ok {
		return n
	}
	return DefaultEffortToBudget[EffortMedium]
}

// Fit fits an effort to the profile's levels.
func (p *ModelProfile) Fit(e Effort) Effort {
	if p == nil {
		return e
	}
	return FitEffort(e, p.Efforts)
}

// Lowest is the weakest level the model accepts.
func (p *ModelProfile) Lowest() Effort {
	if p != nil && len(p.Efforts) > 0 {
		low := p.Efforts[0]
		for _, e := range p.Efforts[1:] {
			if e.Rank() >= 0 && e.Rank() < low.Rank() {
				low = e
			}
		}
		return low
	}
	return EffortLow
}

// OffPatchField is one validated top-level field from an OffPatch rule.
type OffPatchField struct {
	Key   string
	Value json.RawMessage
}

var offPatchWhitelist = map[string]bool{
	"reasoning_effort":     true,
	"reasoning":            true,
	"enable_thinking":      true,
	"chat_template_kwargs": true,
	"think":                true,
	"thinking":             true,
}

// ParseOffPatch validates an OffPatch JSON object against the design.md §9.3.3
// whitelist and returns its fields in deterministic key order. Invalid JSON or
// non-whitelisted keys return a CodeConfig error.
func ParseOffPatch(raw json.RawMessage) ([]OffPatchField, *Error) {
	b := bytes.TrimSpace(raw)
	if len(b) == 0 {
		return nil, nil
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil || m == nil {
		return nil, &Error{Code: CodeConfig, Path: Root.Key("reasoning"), Message: "invalid reasoning off patch configuration"}
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		if !offPatchWhitelist[k] {
			return nil, &Error{Code: CodeConfig, Path: Root.Key("reasoning"), Message: "reasoning off patch contains disallowed key"}
		}
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]OffPatchField, 0, len(keys))
	for _, k := range keys {
		out = append(out, OffPatchField{Key: k, Value: m[k]})
	}
	return out, nil
}

// ResolvedReasoning is a reasoning request resolved against a model profile.
type ResolvedReasoning struct {
	// Off is true when the client turned reasoning off.
	Off bool
	// Effort is the fitted level to send, "" for none.
	Effort Effort
	// Budget is the client's explicit budget, if any.
	Budget *int
	// Dynamic is true when the client requested dynamic/model-decided thinking budget (e.g. Gemini -1).
	Dynamic bool
	// Asked is true when the client said anything about reasoning.
	Asked bool
	// Adaptive is true when the client asked for adaptive thinking.
	Adaptive bool
}

// Resolve applies precedence (off > effort > budget > profile default) and fits
// the result to the profile's levels.
func (p *ModelProfile) Resolve(c *ReasoningConfig) ResolvedReasoning {
	var r ResolvedReasoning
	if c != nil {
		r.Adaptive = c.Mode == ReasoningAdaptive
		r.Asked = c.Mode != ReasoningDefault || c.Effort != EffortUnset || c.Budget != nil
		if c.Mode == ReasoningOff || c.Effort == EffortNone {
			r.Off = true
			r.Asked = true
			return r
		}
		switch {
		case c.Effort != EffortUnset:
			// Effort takes precedence over Budget (design.md §9.3.1).
			r.Effort = c.Effort
		case c.Budget != nil:
			r.Budget = c.Budget
			if *c.Budget < 0 {
				// Dynamic / model-decided thinking budget (e.g. Gemini thinkingBudget: -1).
				r.Dynamic = true
			} else {
				r.Effort = p.BudgetEffort(*c.Budget)
			}
		}
	}
	if r.Effort == EffortUnset && !r.Dynamic && p != nil {
		r.Effort = p.DefaultEffort
	}
	r.Effort = p.Fit(r.Effort)
	return r
}

// IsForced reports whether the choice makes the model call a tool.
func (tc ToolChoice) IsForced() bool {
	return tc.Mode == ToolChoiceRequired || tc.Mode == ToolChoiceTool ||
		(tc.Mode == ToolChoiceAllowed && tc.AllowedMode == ToolChoiceRequired)
}

// RelaxForcedToolChoice returns req unchanged unless the profile marks the
// model as rejecting forced tool_choice and the request forces one. Then it
// returns a copy with tool_choice auto: a named tool (or hosted tool kind)
// keeps only that tool on offer, required becomes auto, and an allowed list
// in required mode stays a list in auto mode. The change is recorded as a
// LossForcedToolChoice.
func RelaxForcedToolChoice(req *Request, profile *ModelProfile, env *Env) (*Request, error) {
	if profile == nil || !profile.NoForcedToolChoice || !req.ToolChoice.IsForced() {
		return req, nil
	}
	tc := req.ToolChoice
	cp := *req
	var detail string
	switch tc.Mode {
	case ToolChoiceTool:
		cp.Tools, cp.Builtins = nil, nil
		if tc.Builtin != "" {
			for _, b := range req.Builtins {
				if b.Kind == tc.Builtin {
					cp.Builtins = append(cp.Builtins, b)
				}
			}
		} else {
			// The exact tool; a bare name with no such top-level tool selects
			// the namespaced tool of that name, as the tool name map resolves it.
			for _, t := range req.Tools {
				if t.Name == tc.Name && t.Namespace == tc.Namespace {
					cp.Tools = append(cp.Tools, t)
				}
			}
			if len(cp.Tools) == 0 && tc.Namespace == "" {
				for _, t := range req.Tools {
					if t.Name == tc.Name {
						cp.Tools = append(cp.Tools, t)
					}
				}
			}
		}
		// A client cache marker on a dropped tool still marks the request:
		// automatic cache placement must not take over.
		if !cp.DroppedCacheMarkers && droppedToolHasCacheMarker(req, &cp) {
			cp.DroppedCacheMarkers = true
		}
		cp.ToolChoice = ToolChoice{Mode: ToolChoiceAuto}
		detail = "the model rejects a forced tool; sent tool_choice auto with only the named tool"
	case ToolChoiceRequired:
		cp.ToolChoice = ToolChoice{Mode: ToolChoiceAuto}
		detail = "the model rejects tool_choice any/required; sent auto"
	case ToolChoiceAllowed:
		cp.ToolChoice.AllowedMode = ToolChoiceAuto
		detail = "the model rejects a forced tool_choice; the allowed tools are offered with auto"
	}
	if err := env.Lose(LossForcedToolChoice, Root.Key("tool_choice"), detail); err != nil {
		return nil, err
	}
	return &cp, nil
}

// droppedToolHasCacheMarker reports whether a tool or builtin in req but not
// in kept carries a cache marker.
func droppedToolHasCacheMarker(req, kept *Request) bool {
	keptTools := map[[2]string]bool{}
	for _, t := range kept.Tools {
		keptTools[[2]string{t.Namespace, t.Name}] = true
	}
	for _, t := range req.Tools {
		if t.Cache != nil && !keptTools[[2]string{t.Namespace, t.Name}] {
			return true
		}
	}
	keptKinds := map[string]bool{}
	for _, b := range kept.Builtins {
		keptKinds[string(b.Kind)] = true
	}
	for _, b := range req.Builtins {
		if !keptKinds[string(b.Kind)] && bytes.Contains(b.Raw, []byte(`"cache_control"`)) {
			return true
		}
	}
	return false
}
