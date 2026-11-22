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
	intercepted bool
	closed      bool
	order       int
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

	// Per-round state (reset by NextRound).
	stopBuf       map[int]string
	stopTriggered bool
	errored       bool

	calls         map[string]*bufferedToolCall
	callOrder     []string
	lastCallKey   map[int]string
	callSeq       int
	bufferedBytes int
	liveUsed      bool
	postCallEvs   map[int][]ir.Event
	deliveredCall map[int]bool
	intercepted   []ir.ToolCall
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
		client:     p.Client,
		release:    rel,
		maxCalls:   maxCalls,
		maxBytes:   maxBytes,
		stopMode:   p.opts.StopMode,
		stops:      stops,
		maxStopLen: maxStopLen,
		reasoning:  p.opts.Reasoning,
		intercept:  ic,
	}
	f.resetRound()
	return f
}

func (f *streamFilter) resetRound() {
	f.stopBuf = map[int]string{}
	f.stopTriggered = false
	f.calls = map[string]*bufferedToolCall{}
	f.callOrder = nil
	f.lastCallKey = map[int]string{}
	f.callSeq = 0
	f.bufferedBytes = 0
	f.liveUsed = false
	f.postCallEvs = map[int][]ir.Event{}
	f.deliveredCall = map[int]bool{}
	f.intercepted = nil
}

func (f *streamFilter) isIntercepted(name string) bool {
	return len(f.intercept) > 0 && f.intercept[name]
}

// process filters one decoded IR event and returns the events to forward to
// the client encoder (or collector).
func (f *streamFilter) process(ev ir.Event) []ir.Event {
	if f.errored {
		return nil
	}
	if f.stopTriggered {
		if ev.Type == ir.EvUsage {
			return []ir.Event{ev}
		}
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
		f.postCallEvs = map[int][]ir.Event{}
		return []ir.Event{ev}

	case ir.EvReasoningStart, ir.EvReasoning, ir.EvSignature, ir.EvSealed:
		if f.reasoning == ir.ReasoningOutHidden {
			return nil
		}
		if f.reasoning == ir.ReasoningOutSummary && ev.Type == ir.EvReasoning {
			ev.Summary = true
		}
		return []ir.Event{ev}

	case ir.EvText:
		var out []ir.Event
		if f.stopMode == ir.StopEmulate && len(f.stops) > 0 {
			out = f.processStopText(ev)
		} else {
			out = []ir.Event{ev}
		}
		// If a call is currently buffered for this candidate and we didn't trigger stop,
		// preserve text order relative to buffered calls.
		if !f.stopTriggered && f.hasBufferedCalls(ev.Index) {
			f.postCallEvs[ev.Index] = append(f.postCallEvs[ev.Index], out...)
			return nil
		}
		return out

	case ir.EvToolStart:
		var out []ir.Event
		out = append(out, f.flushStopBuf(ev.Index)...)

		f.callSeq++
		if f.callSeq > f.maxCalls {
			return f.fail(ev.Index, "tool call limit exceeded")
		}
		key := ev.ID
		if key == "" {
			key = "__bc_" + strconv.Itoa(f.callSeq)
		}
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

		if !bc.intercepted {
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
		if len(ev.Arguments) > 0 {
			if len(ev.Arguments) > f.maxBytes {
				return f.fail(ev.Index, "buffered tool call byte limit exceeded")
			}
			bc.finalArgs = append(json.RawMessage(nil), ev.Arguments...)
		}
		bc.closed = true
		if bc.live && f.release == ir.ToolReleaseStream {
			finalArgs, _, errMsg := bc.validate()
			if errMsg != "" {
				return f.fail(ev.Index, errMsg)
			}
			ev.Arguments = finalArgs
			return []ir.Event{ev}
		}
		return nil

	case ir.EvStop:
		var out []ir.Event
		out = append(out, f.flushStopBuf(ev.Index)...)

		stop := ir.Stop{Reason: ir.StopEndTurn}
		if ev.Stop != nil {
			stop = *ev.Stop
		}

		// Collect calls for this candidate in arrival order.
		candCalls := f.candidateCalls(ev.Index)
		dropBuffered := stop.Truncated() || (f.client == ir.FormatResponses && stop.Filtered())
		if dropBuffered {
			// Drop every buffered call without error.
			f.clearCandidateCalls(ev.Index)
			out = append(out, f.postCallEvs[ev.Index]...)
			delete(f.postCallEvs, ev.Index)
			st := stop
			out = append(out, ir.Event{Type: ir.EvStop, Index: ev.Index, Stop: &st})
			return out
		}

		// Validate all calls before releasing any buffered or liveFirst completion.
		type validated struct {
			bc        *bufferedToolCall
			finalArgs json.RawMessage
			customIn  string
		}
		var vals []validated
		for _, bc := range candCalls {
			if bc.live && bc.closed && f.release == ir.ToolReleaseStream {
				continue
			}
			args, cin, errMsg := bc.validate()
			if errMsg != "" {
				return f.fail(ev.Index, errMsg)
			}
			vals = append(vals, validated{bc: bc, finalArgs: args, customIn: cin})
		}

		for _, v := range vals {
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
				continue
			}
			f.deliveredCall[ev.Index] = true
			if bc.live {
				// LiveFirst already streamed EvToolStart + EvToolArgs; only emit EvToolEnd now.
				out = append(out, ir.Event{
					Type:      ir.EvToolEnd,
					Index:     bc.candIdx,
					ID:        bc.id,
					Arguments: v.finalArgs,
				})
				continue
			}
			// Buffered call: emit Start -> Args -> End whole.
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

		out = append(out, f.postCallEvs[ev.Index]...)
		delete(f.postCallEvs, ev.Index)
		f.clearCandidateCalls(ev.Index)

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

	return []ir.Event{ev}
}

func (f *streamFilter) processStopText(ev ir.Event) []ir.Event {
	buf := f.stopBuf[ev.Index] + ev.Text
	bestPos := -1
	bestSeq := ""
	for _, seq := range f.stops {
		if idx := strings.Index(buf, seq); idx >= 0 {
			if bestPos < 0 || idx < bestPos || (idx == bestPos && len(seq) > len(bestSeq)) {
				bestPos = idx
				bestSeq = seq
			}
		}
	}
	if bestPos >= 0 {
		var out []ir.Event
		if bestPos > 0 {
			out = append(out, ir.Event{Type: ir.EvText, Index: ev.Index, Text: buf[:bestPos]})
		}
		f.stopBuf[ev.Index] = ""
		f.clearCandidateCalls(ev.Index)
		delete(f.postCallEvs, ev.Index)
		f.stopTriggered = true
		st := ir.Stop{Reason: ir.StopSequence, Sequence: bestSeq}
		out = append(out, ir.Event{Type: ir.EvStop, Index: ev.Index, Stop: &st})
		return out
	}

	// Keep at most maxStopLen - 1 trailing bytes, aligned to a valid UTF-8 rune boundary.
	keep := f.maxStopLen - 1
	if keep <= 0 || len(buf) <= keep {
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

func (f *streamFilter) hasBufferedCalls(candIdx int) bool {
	for _, bc := range f.calls {
		if bc.candIdx == candIdx && !bc.live && !bc.intercepted {
			return true
		}
	}
	return false
}

func (f *streamFilter) lookupCall(candIdx int, id string) *bufferedToolCall {
	if id != "" {
		if bc := f.calls[id]; bc != nil {
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
	f.postCallEvs = map[int][]ir.Event{}
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
