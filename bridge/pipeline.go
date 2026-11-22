package bridge

import (
	"encoding/json"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

const (
	defaultMaxToolCalls     = 128
	defaultMaxBufferedBytes = 8 * 1024 * 1024
)

type bufferedToolCall struct {
	id          string
	name        string
	ns          string
	kind        ir.ToolKind
	candIdx     int
	sig         *ir.Signature
	args        strings.Builder
	finalArgs   json.RawMessage
	live        bool
	liveEnded   bool
	intercepted bool
	closed      bool
	order       int
}

type queuedEntry struct {
	callKey string
	ev      ir.Event
}

type streamFilter struct {
	client     ir.Format
	release    ir.ToolRelease
	maxCalls   int
	maxBytes   int
	stopMode   ir.StopMode
	stops      []string
	maxStopLen int
	reasoning  ir.ReasoningOut
	intercept  map[string]bool

	// Response-level state (persists across NextRound).
	totalCalls    int
	deliveredCall map[int]bool

	// Per-round state (reset by NextRound).
	stopBuf       map[int]string
	stopTriggered map[int]bool
	errored       bool

	calls         map[string]*bufferedToolCall
	callOrder     []string
	lastCallKey   map[int]string
	callSeq       int
	bufferedBytes int
	liveUsed      bool
	timeline      map[int][]queuedEntry
	intercepted   []ir.ToolCall
}

func effectiveReasoningOut(p *Plan) ir.ReasoningOut {
	if p == nil {
		return ir.ReasoningOutNative
	}
	if p.opts.Reasoning != ir.ReasoningOutNative {
		return p.opts.Reasoning
	}
	if p.opts.Profile != nil && p.Request != nil {
		res := p.opts.Profile.Resolve(p.Request.Reasoning)
		if res.Off && (p.opts.Profile.Off.Mode == ir.OffLowestHidden || p.opts.Profile.Off.Mode == ir.OffPatch) {
			return ir.ReasoningOutHidden
		}
	}
	return ir.ReasoningOutNative
}

func newStreamFilter(p *Plan) *streamFilter {
	rel := p.opts.ToolRelease
	if rel == ir.ToolReleaseDefault {
		switch p.Client {
		case ir.FormatAnthropic, ir.FormatGemini:
			rel = ir.ToolReleaseBuffered
		case ir.FormatResponses:
			rel = ir.ToolReleaseLiveFirst
		default:
			rel = ir.ToolReleaseStream
		}
	}
	maxCalls := p.opts.Limits.MaxToolCalls
	if maxCalls <= 0 {
		maxCalls = defaultMaxToolCalls
	}
	maxBytes := p.opts.Limits.MaxBufferedBytes
	if maxBytes <= 0 {
		maxBytes = defaultMaxBufferedBytes
	}
	var stops []string
	maxStopLen := 0
	if p.opts.StopMode == ir.StopEmulate && p.Request != nil {
		for _, s := range p.Request.Stop {
			if s != "" {
				stops = append(stops, s)
				if len(s) > maxStopLen {
					maxStopLen = len(s)
				}
			}
		}
	}
	ic := map[string]bool{}
	for _, name := range p.opts.Intercept {
		if name != "" {
			ic[name] = true
		}
	}
	f := &streamFilter{
		client:        p.Client,
		release:       rel,
		maxCalls:      maxCalls,
		maxBytes:      maxBytes,
		stopMode:      p.opts.StopMode,
		stops:         stops,
		maxStopLen:    maxStopLen,
		reasoning:     effectiveReasoningOut(p),
		intercept:     ic,
		deliveredCall: map[int]bool{},
	}
	f.resetRound()
	return f
}

func (f *streamFilter) resetRound() {
	f.stopBuf = map[int]string{}
	f.stopTriggered = map[int]bool{}
	f.calls = map[string]*bufferedToolCall{}
	f.callOrder = nil
	f.lastCallKey = map[int]string{}
	f.callSeq = 0
	f.bufferedBytes = 0
	f.liveUsed = false
	f.timeline = map[int][]queuedEntry{}
	f.intercepted = nil
}

func eventByteSize(ev ir.Event) int {
	n := len(ev.Text) + len(ev.Arguments)
	if ev.Signature != nil {
		n += len(ev.Signature.Value)
	}
	if ev.Sealed != nil {
		n += len(ev.Sealed.Data)
	}
	if ev.Media != nil {
		n += len(ev.Media.Data) + len(ev.Media.URL) + len(ev.Media.Text)
	}
	if ev.Builtin != nil {
		n += len(ev.Builtin.Input) + len(ev.Builtin.Content) + len(ev.Builtin.Raw)
	}
	return n
}

func (f *streamFilter) queueEvents(idx int, evs ...ir.Event) []ir.Event {
	for _, e := range evs {
		f.bufferedBytes += eventByteSize(e)
		if f.bufferedBytes > f.maxBytes {
			return f.fail(idx, "buffered byte limit exceeded")
		}
		f.timeline[idx] = append(f.timeline[idx], queuedEntry{ev: e})
	}
	return nil
}

func (f *streamFilter) resetRoundForPlan(p *Plan) {
	f.resetRound()
	if p == nil {
		return
	}
	f.reasoning = effectiveReasoningOut(p)
	f.stopMode = p.opts.StopMode
	f.stops = nil
	f.maxStopLen = 0
	if p.opts.StopMode == ir.StopEmulate && p.Request != nil {
		for _, s := range p.Request.Stop {
			if s != "" {
				f.stops = append(f.stops, s)
				if len(s) > f.maxStopLen {
					f.maxStopLen = len(s)
				}
			}
		}
	}
	ic := map[string]bool{}
	for _, name := range p.opts.Intercept {
		if name != "" {
			ic[name] = true
		}
	}
	f.intercept = ic
}

func (f *streamFilter) allStopsTriggered(expected int) bool {
	if expected <= 0 {
		expected = 1
	}
	count := 0
	for _, v := range f.stopTriggered {
		if v {
			count++
		}
	}
	return count >= expected
}

func (f *streamFilter) isIntercepted(name string) bool {
	return len(f.intercept) > 0 && f.intercept[name]
}

func (f *streamFilter) isInterceptedOrPrefix(name string) bool {
	if len(f.intercept) == 0 {
		return false
	}
	if f.intercept[name] {
		return true
	}
	if name == "" {
		return true
	}
	for target := range f.intercept {
		if strings.HasPrefix(target, name) {
			return true
		}
	}
	return false
}

// process filters one decoded IR event and returns the events to forward to
// the client encoder (or collector).
func (f *streamFilter) process(ev ir.Event) []ir.Event {
	if f.errored {
		return nil
	}
	if ev.Type != ir.EvStart && ev.Type != ir.EvUsage && ev.Type != ir.EvError && f.stopTriggered[ev.Index] {
		return nil
	}

	switch ev.Type {
	case ir.EvStart:
		return []ir.Event{ev}

	case ir.EvUsage:
		return []ir.Event{ev}

	case ir.EvError:
		f.errored = true
		f.calls = map[string]*bufferedToolCall{}
		f.callOrder = nil
		f.timeline = map[int][]queuedEntry{}
		return []ir.Event{ev}

	case ir.EvReasoningStart, ir.EvReasoning, ir.EvSignature, ir.EvSealed:
		var out []ir.Event
		if ev.Type == ir.EvReasoningStart || ev.Type == ir.EvReasoning {
			out = append(out, f.flushStopBuf(ev.Index)...)
		}
		if f.reasoning == ir.ReasoningOutHidden {
			if len(out) > 0 {
				if f.hasPendingCalls(ev.Index) {
					return f.queueEvents(ev.Index, out...)
				}
				return out
			}
			return nil
		}
		if f.reasoning == ir.ReasoningOutSummary && ev.Type == ir.EvReasoning {
			ev.Summary = true
		}
		out = append(out, ev)
		if f.hasPendingCalls(ev.Index) {
			return f.queueEvents(ev.Index, out...)
		}
		return out

	case ir.EvText:
		var out []ir.Event
		if f.stopMode == ir.StopEmulate && len(f.stops) > 0 {
			out = f.processStopText(ev)
		} else {
			out = []ir.Event{ev}
		}
		// Keep text/reasoning queued while buffered or unvalidated LiveFirst calls are pending
		// so encoders never prematurely close/complete a live tool item before EvStop validation.
		if !f.stopTriggered[ev.Index] && f.hasPendingCalls(ev.Index) {
			return f.queueEvents(ev.Index, out...)
		}
		return out

	case ir.EvToolStart:
		var out []ir.Event
		flushed := f.flushStopBuf(ev.Index)
		if len(flushed) > 0 {
			if f.hasPendingCalls(ev.Index) {
				if failEvs := f.queueEvents(ev.Index, flushed...); failEvs != nil {
					return failEvs
				}
			} else {
				out = append(out, flushed...)
			}
		}

		f.callSeq++
		f.totalCalls++
		if f.totalCalls > f.maxCalls {
			return f.fail(ev.Index, "tool call limit exceeded")
		}
		rawID := ev.ID
		if rawID == "" {
			rawID = "__bc_" + strconv.Itoa(f.callSeq)
		}
		key := callMapKey(ev.Index, rawID)
		kind := ev.Kind
		if kind == "" {
			kind = ir.ToolFunction
		}
		bc := &bufferedToolCall{
			id:          ev.ID,
			name:        ev.Name,
			ns:          ev.Namespace,
			kind:        kind,
			candIdx:     ev.Index,
			sig:         ev.Signature,
			intercepted: f.isIntercepted(ev.Name),
			order:       f.callSeq,
		}
		f.calls[key] = bc
		f.callOrder = append(f.callOrder, key)
		f.lastCallKey[ev.Index] = key
		f.timeline[ev.Index] = append(f.timeline[ev.Index], queuedEntry{callKey: key})

		if !bc.intercepted && !f.isInterceptedOrPrefix(ev.Name) {
			switch f.release {
			case ir.ToolReleaseStream:
				bc.live = true
				f.deliveredCall[ev.Index] = true
				out = append(out, ev)
			case ir.ToolReleaseLiveFirst:
				if !f.liveUsed {
					f.liveUsed = true
					bc.live = true
					f.deliveredCall[ev.Index] = true
					out = append(out, ev)
				}
			}
		}
		return out

	case ir.EvToolArgs:
		bc := f.lookupCall(ev.Index, ev.ID)
		if bc == nil {
			return nil
		}
		f.bufferedBytes += len(ev.Text)
		if f.bufferedBytes > f.maxBytes {
			return f.fail(ev.Index, "buffered tool call byte limit exceeded")
		}
		bc.args.WriteString(ev.Text)
		if bc.live {
			return []ir.Event{ev}
		}
		return nil

	case ir.EvToolEnd:
		bc := f.lookupCall(ev.Index, ev.ID)
		if bc == nil {
			return nil
		}
		if ev.Err != nil {
			// Upstream explicitly signalled a cancelled/incomplete tool call:
			// abort any open live call and remove the call so it is never finalized.
			key := callMapKey(bc.candIdx, bc.id)
			var out []ir.Event
			if bc.live && !bc.liveEnded {
				bc.liveEnded = true
				out = append(out, ir.Event{
					Type:  ir.EvToolEnd,
					Index: bc.candIdx,
					ID:    bc.id,
					Err:   ev.Err,
				})
			}
			delete(f.calls, key)
			if f.lastCallKey[bc.candIdx] == key {
				f.lastCallKey[bc.candIdx] = ""
			}
			if tl := f.timeline[bc.candIdx]; len(tl) > 0 {
				kept := tl[:0]
				for _, entry := range tl {
					if entry.callKey != key {
						kept = append(kept, entry)
					}
				}
				f.timeline[bc.candIdx] = kept
			}
			return out
		}
		if ev.Name != "" && ev.Name != bc.name {
			bc.name = ev.Name
			if ev.Namespace != "" {
				bc.ns = ev.Namespace
			}
			if ev.Kind != "" {
				bc.kind = ev.Kind
			}
			bc.intercepted = f.isIntercepted(bc.name)
			if bc.intercepted && bc.live {
				return f.fail(ev.Index, "intercepted tool name changed mid-stream")
			}
		}
		if len(ev.Arguments) > 0 {
			f.bufferedBytes += len(ev.Arguments)
			if f.bufferedBytes > f.maxBytes {
				return f.fail(ev.Index, "buffered tool call byte limit exceeded")
			}
			bc.finalArgs = append(json.RawMessage(nil), ev.Arguments...)
		}
		bc.closed = true
		if bc.live && f.release == ir.ToolReleaseStream {
			// Defer JSON error reporting until EvStop so max_tokens truncation is not turned into 502.
			if finalArgs, _, errMsg := bc.validate(); errMsg == "" {
				bc.liveEnded = true
				ev.Arguments = finalArgs
				return []ir.Event{ev}
			}
			return nil
		}
		return nil

	case ir.EvStop:
		var out []ir.Event
		flushed := f.flushStopBuf(ev.Index)

		stop := ir.Stop{Reason: ir.StopEndTurn}
		if ev.Stop != nil {
			stop = *ev.Stop
		}

		// Collect calls for this candidate in arrival order.
		candCalls := f.candidateCalls(ev.Index)
		dropBuffered := stop.Truncated() || (f.client == ir.FormatResponses && stop.Filtered())
		if dropBuffered {
			// Abort any open live call BEFORE emitting queued non-call events so encoders
			// never close an unvalidated live tool item as completed when switching items.
			for _, bc := range candCalls {
				if bc.live && !bc.liveEnded {
					bc.liveEnded = true
					out = append(out, ir.Event{
						Type:  ir.EvToolEnd,
						Index: bc.candIdx,
						ID:    bc.id,
						Err:   &ir.APIError{Message: "truncated tool call discarded"},
					})
				}
			}
			for _, entry := range f.timeline[ev.Index] {
				if entry.callKey == "" {
					out = append(out, entry.ev)
				}
			}
			out = append(out, flushed...)
			delete(f.timeline, ev.Index)
			f.clearCandidateCalls(ev.Index)
			st := stop
			out = append(out, ir.Event{Type: ir.EvStop, Index: ev.Index, Stop: &st})
			return out
		}

		emitted, failEvs := f.flushValidatedTimeline(ev.Index, candCalls)
		if failEvs != nil {
			return failEvs
		}
		out = append(out, emitted...)
		out = append(out, flushed...)

		// If intercepted calls exist, hold back EvStop so caller can run tools and call NextRound.
		if len(f.intercepted) > 0 {
			return out
		}

		// Consistency rules (design.md §7.4).
		if f.deliveredCall[ev.Index] {
			if !stop.Truncated() && !stop.Filtered() && stop.Reason != ir.StopSequence && stop.Reason != ir.StopPauseTurn {
				stop.Reason = ir.StopToolUse
			}
		} else if stop.Reason == ir.StopToolUse {
			stop.Reason = ir.StopEndTurn
		}
		st := stop
		out = append(out, ir.Event{Type: ir.EvStop, Index: ev.Index, Stop: &st})
		return out
	}

	if f.hasPendingCalls(ev.Index) {
		return f.queueEvents(ev.Index, ev)
	}
	return []ir.Event{ev}
}

func (f *streamFilter) flushValidatedTimeline(idx int, candCalls []*bufferedToolCall) (out []ir.Event, failEvs []ir.Event) {
	type validated struct {
		bc        *bufferedToolCall
		finalArgs json.RawMessage
		customIn  string
	}
	valByPtr := map[*bufferedToolCall]validated{}
	for _, bc := range candCalls {
		args, cin, errMsg := bc.validate()
		if errMsg != "" {
			return nil, f.fail(idx, errMsg)
		}
		if bc.live && bc.closed && f.release == ir.ToolReleaseStream {
			continue
		}
		valByPtr[bc] = validated{bc: bc, finalArgs: args, customIn: cin}
	}

	emitCall := func(v validated) {
		bc := v.bc
		if bc.intercepted {
			tc := ir.ToolCall{
				ID:        bc.id,
				Name:      bc.name,
				Namespace: bc.ns,
				Kind:      bc.kind,
				Signature: bc.sig,
			}
			if bc.kind == ir.ToolCustom {
				tc.Input = v.customIn
			} else {
				tc.Arguments = v.finalArgs
			}
			f.intercepted = append(f.intercepted, tc)
			return
		}
		f.deliveredCall[idx] = true
		if bc.live {
			out = append(out, ir.Event{
				Type:      ir.EvToolEnd,
				Index:     bc.candIdx,
				ID:        bc.id,
				Arguments: v.finalArgs,
			})
			return
		}
		out = append(out, ir.Event{
			Type:      ir.EvToolStart,
			Index:     bc.candIdx,
			ID:        bc.id,
			Name:      bc.name,
			Namespace: bc.ns,
			Kind:      bc.kind,
			Signature: bc.sig,
		})
		if bc.kind == ir.ToolCustom {
			if v.customIn != "" {
				out = append(out, ir.Event{Type: ir.EvToolArgs, Index: bc.candIdx, ID: bc.id, Text: v.customIn})
			}
		} else if len(v.finalArgs) > 0 {
			out = append(out, ir.Event{Type: ir.EvToolArgs, Index: bc.candIdx, ID: bc.id, Text: string(v.finalArgs)})
		}
		out = append(out, ir.Event{Type: ir.EvToolEnd, Index: bc.candIdx, ID: bc.id, Arguments: v.finalArgs})
	}

	seen := map[*bufferedToolCall]bool{}
	for _, entry := range f.timeline[idx] {
		if entry.callKey != "" {
			if bc := f.calls[entry.callKey]; bc != nil {
				if v, ok := valByPtr[bc]; ok {
					emitCall(v)
					seen[bc] = true
				}
			}
			continue
		}
		out = append(out, entry.ev)
	}
	for _, bc := range candCalls {
		if !seen[bc] {
			if v, ok := valByPtr[bc]; ok {
				emitCall(v)
			}
		}
	}
	delete(f.timeline, idx)
	f.clearCandidateCalls(idx)
	return out, nil
}

func (f *streamFilter) processStopText(ev ir.Event) []ir.Event {
	buf := f.stopBuf[ev.Index] + ev.Text
	bestPos := -1
	bestEnd := -1
	bestSeq := ""
	for _, seq := range f.stops {
		if idx := strings.Index(buf, seq); idx >= 0 {
			end := idx + len(seq)
			if bestEnd < 0 || end < bestEnd || (end == bestEnd && idx < bestPos) {
				bestPos = idx
				bestEnd = end
				bestSeq = seq
			}
		}
	}
	if bestPos >= 0 {
		var out []ir.Event
		// Per design.md §9.6, hitting an emulated stop discards pending/buffered calls
		// and aborts any open live call, while preserving legitimate queued text prefixes.
		candCalls := f.candidateCalls(ev.Index)
		for _, bc := range candCalls {
			if bc.live && !bc.liveEnded {
				bc.liveEnded = true
				out = append(out, ir.Event{
					Type:  ir.EvToolEnd,
					Index: bc.candIdx,
					ID:    bc.id,
					Err:   &ir.APIError{Message: "tool call discarded on stop sequence"},
				})
			}
		}
		for _, entry := range f.timeline[ev.Index] {
			if entry.callKey == "" {
				out = append(out, entry.ev)
			}
		}
		delete(f.timeline, ev.Index)
		f.clearCandidateCalls(ev.Index)
		if bestPos > 0 {
			out = append(out, ir.Event{Type: ir.EvText, Index: ev.Index, Text: buf[:bestPos]})
		}
		f.stopBuf[ev.Index] = ""
		f.stopTriggered[ev.Index] = true
		st := ir.Stop{Reason: ir.StopSequence, Sequence: bestSeq}
		out = append(out, ir.Event{Type: ir.EvStop, Index: ev.Index, Stop: &st})
		return out
	}

	// Keep at most maxStopLen - 1 trailing bytes, aligned to a valid UTF-8 rune boundary.
	keep := f.maxStopLen - 1
	if keep <= 0 {
		f.stopBuf[ev.Index] = ""
		if buf == "" {
			return nil
		}
		return []ir.Event{{Type: ir.EvText, Index: ev.Index, Text: buf}}
	}
	if len(buf) <= keep {
		f.stopBuf[ev.Index] = buf
		return nil
	}
	split := len(buf) - keep
	for split > 0 && !utf8.RuneStart(buf[split]) {
		split--
	}
	if split == 0 {
		f.stopBuf[ev.Index] = buf
		return nil
	}
	emit := buf[:split]
	f.stopBuf[ev.Index] = buf[split:]
	return []ir.Event{{Type: ir.EvText, Index: ev.Index, Text: emit}}
}

func (f *streamFilter) flushStopBuf(candIdx int) []ir.Event {
	if s := f.stopBuf[candIdx]; s != "" {
		f.stopBuf[candIdx] = ""
		return []ir.Event{{Type: ir.EvText, Index: candIdx, Text: s}}
	}
	return nil
}

func (f *streamFilter) hasPendingCalls(candIdx int) bool {
	if f.release == ir.ToolReleaseStream {
		return false
	}
	for _, bc := range f.calls {
		if bc.candIdx == candIdx && !bc.intercepted {
			return true
		}
	}
	return false
}

func callMapKey(candIdx int, id string) string {
	return strconv.Itoa(candIdx) + "\x00" + id
}

func (f *streamFilter) lookupCall(candIdx int, id string) *bufferedToolCall {
	if id != "" {
		if bc := f.calls[callMapKey(candIdx, id)]; bc != nil {
			return bc
		}
	}
	if k := f.lastCallKey[candIdx]; k != "" {
		return f.calls[k]
	}
	return nil
}

func (f *streamFilter) candidateCalls(candIdx int) []*bufferedToolCall {
	var out []*bufferedToolCall
	for _, k := range f.callOrder {
		if bc := f.calls[k]; bc != nil && bc.candIdx == candIdx {
			out = append(out, bc)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].order < out[j].order })
	return out
}

func (f *streamFilter) clearCandidateCalls(candIdx int) {
	for k, bc := range f.calls {
		if bc.candIdx == candIdx {
			delete(f.calls, k)
		}
	}
	delete(f.lastCallKey, candIdx)
}

func (f *streamFilter) fail(candIdx int, msg string) []ir.Event {
	f.errored = true
	f.calls = map[string]*bufferedToolCall{}
	f.callOrder = nil
	f.timeline = map[int][]queuedEntry{}
	return []ir.Event{{
		Type:  ir.EvError,
		Index: candIdx,
		Err: &ir.APIError{
			Status:  502,
			Type:    ir.ErrServer,
			Message: msg,
		},
	}}
}

func (bc *bufferedToolCall) validate() (json.RawMessage, string, string) {
	if strings.TrimSpace(bc.name) == "" {
		return nil, "", "tool call missing name"
	}
	if bc.kind == ir.ToolCustom {
		if len(bc.finalArgs) > 0 {
			var s string
			if json.Unmarshal(bc.finalArgs, &s) == nil {
				return bc.finalArgs, s, ""
			}
		}
		s := bc.args.String()
		raw, _ := jsonx.Marshal(s)
		return raw, s, ""
	}
	raw := strings.TrimSpace(string(bc.finalArgs))
	if raw == "" {
		raw = strings.TrimSpace(bc.args.String())
	}
	if raw == "" {
		return json.RawMessage(`{}`), "", ""
	}
	b := []byte(raw)
	if jsonx.Kind(b) != '{' || !json.Valid(b) {
		return nil, "", "tool call arguments must be a valid JSON object"
	}
	return json.RawMessage(b), "", ""
}

func validateToolCall(call *ir.ToolCall) string {
	if call == nil {
		return ""
	}
	if strings.TrimSpace(call.Name) == "" {
		return "tool call missing name"
	}
	if call.Kind == ir.ToolCustom {
		return ""
	}
	raw := strings.TrimSpace(string(call.Arguments))
	if raw == "" {
		return ""
	}
	b := []byte(raw)
	if jsonx.Kind(b) != '{' || !json.Valid(b) {
		return "tool call arguments must be a valid JSON object"
	}
	return ""
}

// repairHistory applies caller-enabled ir.Repair transformations to a copy of req.Messages.
func repairHistory(req *ir.Request, repair ir.RepairSet) {
	if req == nil || len(repair) == 0 || len(req.Messages) == 0 {
		return
	}
	msgs := make([]ir.Message, len(req.Messages))
	copy(msgs, req.Messages)

	// 1. drop_empty_assistant
	if repair[ir.RepairEmptyAssistant] {
		filtered := msgs[:0]
		for _, m := range msgs {
			if m.Role == ir.RoleAssistant && len(m.Parts) == 0 {
				continue
			}
			filtered = append(filtered, m)
		}
		msgs = filtered
	}

	// Track known call IDs from assistant turns.
	knownCalls := map[string]ir.ToolCall{}
	for _, m := range msgs {
		if m.Role == ir.RoleAssistant {
			for _, pt := range m.Parts {
				if pt.Type == ir.PartToolCall && pt.Call != nil && pt.Call.ID != "" {
					knownCalls[pt.Call.ID] = *pt.Call
				}
			}
		}
	}

	// 2. drop_duplicate_results & 3. orphan_results_as_text
	if repair[ir.RepairDuplicateResults] || repair[ir.RepairOrphanResults] {
		seenResultID := map[string]bool{}
		for i := range msgs {
			if msgs[i].Role != ir.RoleUser {
				continue
			}
			var parts []ir.Part
			for _, pt := range msgs[i].Parts {
				if pt.Type != ir.PartToolResult || pt.Result == nil {
					parts = append(parts, pt)
					continue
				}
				cid := pt.Result.CallID
				if repair[ir.RepairOrphanResults] {
					if _, ok := knownCalls[cid]; !ok {
						text := ir.TextOf(pt.Result.Content, "\n")
						if text == "" {
							text = "[orphan tool result]"
						}
						parts = append(parts, ir.TextPart(text))
						continue
					}
				}
				if repair[ir.RepairDuplicateResults] && cid != "" {
					if seenResultID[cid] {
						continue
					}
					seenResultID[cid] = true
				}
				parts = append(parts, pt)
			}
			msgs[i].Parts = parts
		}
		// Drop any user message that became empty after removing duplicate tool results.
		nonEmpty := msgs[:0]
		for _, m := range msgs {
			if m.Role == ir.RoleUser && len(m.Parts) == 0 {
				continue
			}
			nonEmpty = append(nonEmpty, m)
		}
		msgs = nonEmpty
	}

	// 4. synthesize_missing_results
	if repair[ir.RepairMissingResults] {
		var out []ir.Message
		for i := 0; i < len(msgs); i++ {
			m := msgs[i]
			out = append(out, m)
			if m.Role != ir.RoleAssistant {
				continue
			}
			var callIDs []string
			callByID := map[string]*ir.ToolCall{}
			for _, pt := range m.Parts {
				if pt.Type == ir.PartToolCall && pt.Call != nil && pt.Call.ID != "" {
					callIDs = append(callIDs, pt.Call.ID)
					callByID[pt.Call.ID] = pt.Call
				}
			}
			if len(callIDs) == 0 {
				continue
			}
			// Collect answered call IDs in consecutive user messages following this assistant turn.
			answered := map[string]bool{}
			j := i + 1
			for j < len(msgs) && msgs[j].Role == ir.RoleUser {
				for _, pt := range msgs[j].Parts {
					if pt.Type == ir.PartToolResult && pt.Result != nil {
						answered[pt.Result.CallID] = true
					}
				}
				j++
			}
			var missing []ir.Part
			for _, cid := range callIDs {
				if !answered[cid] {
					c := callByID[cid]
					missing = append(missing, ir.Part{
						Type: ir.PartToolResult,
						Result: &ir.ToolResult{
							CallID:  cid,
							Name:    c.Name,
							Kind:    c.Kind,
							IsError: true,
							Content: []ir.Part{ir.TextPart("[missing tool result]")},
						},
					})
				}
			}
			if len(missing) > 0 {
				out = append(out, ir.Message{Role: ir.RoleUser, Parts: missing})
			}
		}
		msgs = out
	}

	req.Messages = msgs
}
