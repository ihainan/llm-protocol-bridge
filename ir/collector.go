package ir

import (
	"encoding/json"
	"strings"
)

// Collector assembles a Response from streamed events. Encoders and the bridge
// use it so that streamed and non-streamed output follow the same rules.
type Collector struct {
	res  Response
	outs map[int]*collected
	err  *APIError
	seen bool
}

type collected struct {
	out   Output
	text  strings.Builder // text being added to the last part
	open  int             // index+1 of the part text belongs to, 0 when none
	args  strings.Builder // arguments of the open tool call
	call  int             // index+1 of the open tool call
	fresh bool            // a reasoning_start asked for a new reasoning part
	rid   string          // id given by reasoning_start
}

// NewCollector returns an empty collector.
func NewCollector() *Collector { return &Collector{outs: map[int]*collected{}} }

func (c *Collector) output(i int) *collected {
	o := c.outs[i]
	if o == nil {
		o = &collected{out: Output{Index: i}}
		c.outs[i] = o
	}
	return o
}

func (o *collected) flushText() {
	if o.open > 0 {
		p := &o.out.Parts[o.open-1]
		switch p.Type {
		case PartReasoning:
			p.Reasoning.Text += o.text.String()
		default:
			p.Text += o.text.String()
		}
		o.text.Reset()
		o.open = 0
	}
}

func (o *collected) closeCall() {
	if o.call == 0 {
		return
	}
	p := &o.out.Parts[o.call-1]
	s := o.args.String()
	if p.Call.Kind == ToolCustom {
		p.Call.Input += s
	} else if len(p.Call.Arguments) == 0 {
		t := strings.TrimSpace(s)
		if t == "" {
			t = "{}"
		}
		p.Call.Arguments = json.RawMessage(t)
	}
	o.args.Reset()
	o.call = 0
}

func (o *collected) last() *Part {
	if n := len(o.out.Parts); n > 0 {
		return &o.out.Parts[n-1]
	}
	return nil
}

func (o *collected) add(p Part) *Part {
	o.flushText()
	o.closeCall()
	o.out.Parts = append(o.out.Parts, p)
	return &o.out.Parts[len(o.out.Parts)-1]
}

// openReasoning returns the reasoning part further reasoning belongs to.
func (o *collected) openReasoning() *Part {
	if o.fresh {
		return nil
	}
	if p := o.last(); p != nil && p.Type == PartReasoning && p.Reasoning.Sealed == nil {
		return p
	}
	return nil
}

func (o *collected) newReasoning() *Part {
	p := o.add(Part{Type: PartReasoning, Reasoning: &Reasoning{ID: o.rid}})
	o.fresh, o.rid = false, ""
	return p
}

// Add consumes one event.
func (c *Collector) Add(ev Event) {
	c.seen = true
	switch ev.Type {
	case EvStart:
		if ev.ID != "" {
			c.res.ID = ev.ID
		}
		if ev.Model != "" {
			c.res.Model = ev.Model
		}
		if ev.Usage != nil {
			c.res.Usage.Merge(*ev.Usage)
		}
		return
	case EvUsage:
		if ev.Usage != nil {
			c.res.Usage.Merge(*ev.Usage)
		}
		return
	case EvError:
		c.err = ev.Err
		return
	}
	o := c.output(ev.Index)
	switch ev.Type {
	case EvText:
		if p := o.last(); p == nil || p.Type != PartText || p.Signature != nil || o.fresh {
			o.add(Part{Type: PartText})
			o.fresh = false
		}
		if o.open != len(o.out.Parts) {
			o.flushText()
			o.open = len(o.out.Parts)
		}
		o.text.WriteString(ev.Text)
	case EvCitation:
		if ev.Citation == nil {
			return
		}
		if p := o.last(); p != nil && p.Type == PartText {
			p.Citations = append(p.Citations, *ev.Citation)
		}
	case EvReasoningStart:
		o.flushText()
		o.closeCall()
		o.fresh, o.rid = true, ev.ID
	case EvReasoning:
		p := o.openReasoning()
		if p == nil {
			p = o.newReasoning()
		}
		if ev.Summary {
			o.flushText()
			for len(p.Reasoning.Summary) <= ev.Part {
				p.Reasoning.Summary = append(p.Reasoning.Summary, "")
			}
			p.Reasoning.Summary[ev.Part] += ev.Text
			return
		}
		if o.open != len(o.out.Parts) {
			o.flushText()
			o.open = len(o.out.Parts)
		}
		o.text.WriteString(ev.Text)
	case EvSignature:
		if ev.Signature == nil {
			return
		}
		if p := o.openReasoning(); p != nil {
			o.flushText()
			if p.Reasoning.Signature == nil {
				s := *ev.Signature
				p.Reasoning.Signature = &s
			} else {
				p.Reasoning.Signature.Value += ev.Signature.Value
			}
			return
		}
		if !o.fresh {
			if p := o.last(); p != nil && p.Type == PartText && p.Signature == nil {
				o.flushText()
				s := *ev.Signature
				p.Signature = &s
				return
			}
		}
		p := o.newReasoning()
		s := *ev.Signature
		p.Reasoning.Signature = &s
	case EvSealed:
		if ev.Sealed == nil {
			return
		}
		if p := o.openReasoning(); p != nil && p.Reasoning.Signature == nil && ev.Sealed.Kind == SealedResponsesEncrypted {
			o.flushText()
			s := *ev.Sealed
			p.Reasoning.Sealed = &s
			return
		}
		p := o.newReasoning()
		s := *ev.Sealed
		p.Reasoning.Sealed = &s
	case EvToolStart:
		kind := ev.Kind
		if kind == "" {
			kind = ToolFunction
		}
		o.add(Part{Type: PartToolCall, Call: &ToolCall{ID: ev.ID, Name: ev.Name, Namespace: ev.Namespace, Kind: kind, Signature: ev.Signature}})
		o.call = len(o.out.Parts)
	case EvToolArgs:
		if o.call > 0 {
			o.args.WriteString(ev.Text)
		}
	case EvToolEnd:
		if o.call > 0 {
			p := &o.out.Parts[o.call-1]
			if len(ev.Arguments) > 0 {
				if p.Call.Kind == ToolCustom {
					o.args.Reset()
					var s string
					if json.Unmarshal(ev.Arguments, &s) == nil {
						p.Call.Input = s
					}
				} else {
					o.args.Reset()
					p.Call.Arguments = append(json.RawMessage(nil), ev.Arguments...)
				}
			}
			o.closeCall()
		}
	case EvBuiltin:
		if ev.Builtin == nil {
			return
		}
		t := PartBuiltinCall
		if ev.Builtin.IsResult {
			t = PartBuiltinResult
		}
		b := *ev.Builtin
		o.add(Part{Type: t, Builtin: &b})
	case EvImage:
		if ev.Media != nil {
			m := *ev.Media
			o.add(Part{Type: PartImage, Media: &m})
		}
	case EvStop:
		o.flushText()
		o.closeCall()
		if ev.Stop != nil {
			o.out.Stop = *ev.Stop
		}
	}
}

// Err returns the terminal error the stream reported, if any.
func (c *Collector) Err() *APIError { return c.err }

// Usage returns the usage collected so far.
func (c *Collector) Usage() Usage { return c.res.Usage }

// Response returns the reply assembled so far, with stop reasons finalised.
// The collector may keep receiving events afterwards.
func (c *Collector) Response() *Response {
	r := c.res
	r.Outputs = nil
	max := -1
	for i := range c.outs {
		if i > max {
			max = i
		}
	}
	for i := 0; i <= max; i++ {
		o := c.outs[i]
		if o == nil {
			continue
		}
		o.flushText()
		o.closeCall()
		out := o.out
		out.Parts = append([]Part(nil), o.out.Parts...)
		out.Stop = FinalStop(out.Stop, out.Parts)
		r.Outputs = append(r.Outputs, out)
	}
	if len(r.Outputs) == 0 {
		r.Outputs = []Output{{Stop: FinalStop(Stop{}, nil)}}
	}
	return &r
}

// Empty reports whether no event was added.
func (c *Collector) Empty() bool { return !c.seen }

// HasToolCalls reports whether parts contain a tool call.
func HasToolCalls(parts []Part) bool {
	for _, p := range parts {
		if p.Type == PartToolCall {
			return true
		}
	}
	return false
}

// FinalStop applies the stop consistency rules: a length stop always wins; a
// reply that made tool calls ends with tool_use; an unset reason is end_turn.
func FinalStop(s Stop, parts []Part) Stop {
	if s.Truncated() || s.Filtered() || s.Reason == StopSequence || s.Reason == StopPauseTurn {
		return s
	}
	if HasToolCalls(parts) {
		s.Reason = StopToolUse
		return s
	}
	if s.Reason == StopUnset || s.Reason == StopToolUse || s.Reason == StopOther {
		s.Reason = StopEndTurn
	}
	return s
}

// EventsOf replays a whole response as events.
func EventsOf(r *Response) []Event {
	u := r.Usage
	evs := []Event{{Type: EvStart, ID: r.ID, Model: r.Model}}
	for _, o := range r.Outputs {
		for _, p := range o.Parts {
			evs = appendPartEvents(evs, o.Index, p)
		}
		st := o.Stop
		evs = append(evs, Event{Type: EvStop, Index: o.Index, Stop: &st})
	}
	evs = append(evs, Event{Type: EvUsage, Usage: &u})
	return evs
}

func appendPartEvents(evs []Event, idx int, p Part) []Event {
	switch p.Type {
	case PartText:
		evs = append(evs, Event{Type: EvText, Index: idx, Text: p.Text})
		for i := range p.Citations {
			c := p.Citations[i]
			evs = append(evs, Event{Type: EvCitation, Index: idx, Citation: &c})
		}
		if p.Signature != nil {
			s := *p.Signature
			evs = append(evs, Event{Type: EvSignature, Index: idx, Signature: &s})
		}
	case PartRefusal:
		evs = append(evs, Event{Type: EvText, Index: idx, Text: p.Text})
	case PartReasoning:
		r := p.Reasoning
		if r == nil {
			return evs
		}
		evs = append(evs, Event{Type: EvReasoningStart, Index: idx, ID: r.ID})
		if r.Text != "" {
			evs = append(evs, Event{Type: EvReasoning, Index: idx, Text: r.Text})
		}
		for i, s := range r.Summary {
			evs = append(evs, Event{Type: EvReasoning, Index: idx, Text: s, Summary: true, Part: i})
		}
		if r.Signature != nil {
			s := *r.Signature
			evs = append(evs, Event{Type: EvSignature, Index: idx, Signature: &s})
		}
		if r.Sealed != nil {
			s := *r.Sealed
			evs = append(evs, Event{Type: EvSealed, Index: idx, Sealed: &s})
		}
	case PartToolCall:
		c := p.Call
		evs = append(evs, Event{Type: EvToolStart, Index: idx, ID: c.ID, Name: c.Name, Namespace: c.Namespace, Kind: c.Kind, Signature: c.Signature})
		if c.Kind == ToolCustom {
			if c.Input != "" {
				evs = append(evs, Event{Type: EvToolArgs, Index: idx, Text: c.Input})
			}
		} else if len(c.Arguments) > 0 {
			evs = append(evs, Event{Type: EvToolArgs, Index: idx, Text: string(c.Arguments)})
		}
		evs = append(evs, Event{Type: EvToolEnd, Index: idx})
	case PartBuiltinCall, PartBuiltinResult:
		if p.Builtin != nil {
			b := *p.Builtin
			evs = append(evs, Event{Type: EvBuiltin, Index: idx, Builtin: &b})
		}
	case PartImage:
		if p.Media != nil {
			m := *p.Media
			evs = append(evs, Event{Type: EvImage, Index: idx, Media: &m})
		}
	}
	return evs
}
