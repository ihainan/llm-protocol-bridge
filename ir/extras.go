package ir

// ApplyInputExtras prepares a request carrying Responses input extras for a
// builder of another format, which has no such items. Every builder other
// than the Responses one calls it before building.
//
// configuration_update items are folded in input order into request-level
// settings: each field an update sets replaces the earlier value, fields it
// leaves out keep theirs. The result is recorded as a LossConfigUpdate.
// compaction items can only be read by the Responses upstream that sealed
// them, so they are rejected.
//
// It returns req itself when there are no extras, otherwise a shallow copy
// without them (so applying it again is a no-op).
func ApplyInputExtras(req *Request, env *Env) (*Request, error) {
	extras := req.InputExtras()
	if len(extras) == 0 {
		return req, nil
	}
	var rc *ReasoningConfig
	var out *OutputFormat
	lastReasoning, lastOutput := -1, -1
	for _, ex := range extras {
		p := Root.Key("input").At(ex.Index)
		switch ex.Type {
		case ExtraCompactionTrigger:
			return nil, env.Err(Unsupported(p.Key("type"), "compaction_trigger",
				"remote compaction (compaction_trigger) needs an OpenAI Responses upstream; this model cannot produce a compaction item, so configure the client to compact the conversation locally"))
		case ExtraCompaction, ExtraCompactionSummary:
			return nil, env.Err(Unsupported(p.Key("type"), ex.Type,
				"the input carries a compaction item sealed by an OpenAI Responses upstream, which this model cannot read; continue on the model that produced it or start a new conversation"))
		case ExtraConfigurationUpdate:
			if upd := ex.Reasoning; upd != nil {
				if rc == nil {
					rc = &ReasoningConfig{}
					if req.Reasoning != nil {
						*rc = *req.Reasoning
					}
				}
				if upd.Effort != EffortUnset {
					rc.Effort = upd.Effort
					rc.Mode = upd.Mode
					rc.Budget = upd.Budget
				} else if rc.Mode == ReasoningDefault {
					// A summary-only update turns reasoning on, as the same
					// object does at the top level.
					rc.Mode = upd.Mode
				}
				if upd.Summary != "" {
					rc.Summary = upd.Summary
				}
				lastReasoning = ex.Index
			}
			if ex.Output != nil {
				out = ex.Output
				lastOutput = ex.Index
			}
		}
	}
	cp := *req
	cp.TrailingExtras = nil
	cp.Messages = make([]Message, len(req.Messages))
	copy(cp.Messages, req.Messages)
	for i := range cp.Messages {
		cp.Messages[i].Extras = nil
	}
	// The Responses parser stops lifting leading system/developer messages
	// into the instructions at the first extra, to keep them in place for a
	// Responses upstream. With the extras consumed here, lift them as it
	// would have.
	if req.Source == FormatResponses {
		i := 0
		for i < len(cp.Messages) && (cp.Messages[i].Role == RoleSystem || cp.Messages[i].Role == RoleDeveloper) {
			i++
		}
		if i > 0 {
			cp.Instructions = append([]Part(nil), req.Instructions...)
			for _, m := range cp.Messages[:i] {
				if txt := TextOf(m.Parts, "\n"); txt != "" {
					cp.Instructions = append(cp.Instructions, TextPart(txt))
				}
			}
			cp.Messages = cp.Messages[i:]
		}
	}
	if rc != nil {
		if err := env.Lose(LossConfigUpdate, Root.Key("input").At(lastReasoning).Key("reasoning"),
			"configuration_update applied as the request-level reasoning setting"); err != nil {
			return nil, err
		}
		cp.Reasoning = rc
	}
	if out != nil {
		if err := env.Lose(LossConfigUpdate, Root.Key("input").At(lastOutput).Key("text"),
			"configuration_update applied as the request-level text format"); err != nil {
			return nil, err
		}
		cp.Output = out
	}
	return &cp, nil
}
