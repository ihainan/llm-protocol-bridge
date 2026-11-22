// Package bridge wires a client format to an upstream format.
package bridge

import (
	"strings"
	"time"

	"github.com/ihainan/llm-protocol-bridge/anthropic"
	"github.com/ihainan/llm-protocol-bridge/chat"
	"github.com/ihainan/llm-protocol-bridge/gemini"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/responses"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// Format is the wire format identifier.
type Format = ir.Format

// Options configures one conversion.
type Options struct {
	ProviderID string
	Profile    *ir.ModelProfile

	// Client-side metadata that lives outside the JSON body in some formats
	// (e.g., Gemini URL path model name, stream mode, and alt=sse query param).
	Model  string
	Stream bool
	AltSSE bool

	// Upstream dialect overrides.
	Chat      chat.Dialect
	Anthropic anthropic.Dialect
	Gemini    gemini.Dialect
	Responses responses.Dialect

	Allow   ir.LossSet
	Deny    ir.LossSet
	OnLoss  func(ir.Loss)
	Unknown ir.UnknownPolicy
	Repair  ir.RepairSet

	ToolRelease ir.ToolRelease
	Limits      ir.Limits
	StopMode    ir.StopMode
	Reasoning   ir.ReasoningOut
	Seal        ir.SealCodec
	Intercept   []string

	IDs            ir.IDSource
	Clock          func() time.Time
	PromptEstimate int
}

// Plan is the result of a successful request conversion.
type Plan struct {
	Client, Upstream Format
	Request          *ir.Request
	Body             []byte
	Model            string
	ClientModel      string
	ClientStream     bool
	UpstreamStream   bool
	Losses           []ir.Loss

	opts  Options
	names map[string]ir.ToolRef
}

// Convert parses a client request and builds an upstream request in one step.
func Convert(client, upstream Format, body []byte, opts Options) (*Plan, error) {
	parseEnv := clientEnv(opts, client)
	req, err := parseWithEnv(client, body, opts, parseEnv)
	if err != nil {
		return nil, err
	}
	plan, err := Build(req, client, upstream, opts)
	if err != nil {
		return nil, err
	}
	if pl := parseEnv.Losses(); len(pl) > 0 {
		merged := make([]ir.Loss, 0, len(pl)+len(plan.Losses))
		merged = append(merged, pl...)
		merged = append(merged, plan.Losses...)
		plan.Losses = merged
	}
	return plan, nil
}

// Parse decodes a client request body into an ir.Request.
func Parse(client Format, body []byte, opts Options) (*ir.Request, error) {
	return parseWithEnv(client, body, opts, clientEnv(opts, client))
}

func parseWithEnv(client Format, body []byte, opts Options, env *ir.Env) (*ir.Request, error) {
	var req *ir.Request
	var err error
	switch client {
	case ir.FormatChat:
		req, err = chat.ParseRequest(body, chat.ParseOptions{Env: env})
	case ir.FormatResponses:
		req, err = responses.ParseRequest(body, responses.ParseOptions{Env: env})
	case ir.FormatAnthropic:
		req, err = anthropic.ParseRequest(body, anthropic.ParseOptions{Env: env})
	case ir.FormatGemini:
		req, err = gemini.ParseRequest(body, gemini.ParseOptions{Env: env, Model: opts.Model, Stream: opts.Stream})
	default:
		return nil, ir.Invalid(ir.Root, "unknown client format: "+string(client))
	}
	if err != nil {
		return nil, err
	}
	if opts.Model != "" && req.Model == "" {
		req.Model = opts.Model
	}
	if opts.Stream && !req.Stream {
		req.Stream = true
	}
	return req, nil
}

// Build serialises an ir.Request into an upstream body and returns a Plan.
func Build(req *ir.Request, client, upstream Format, opts Options) (*Plan, error) {
	if len(opts.Repair) > 0 {
		repairHistory(req, opts.Repair)
	}
	env := upstreamEnv(opts, upstream)
	env.Style = ir.StyleFor(client)

	// Reject stateful previous_response_id across non-Responses upstreams.
	if req.PreviousResponseID != "" && upstream != ir.FormatResponses {
		return nil, env.Err(ir.Unsupported(ir.Root.Key("previous_response_id"), "previous_response_id", "previous_response_id is only supported for Responses-to-Responses requests"))
	}

	// When StopEmulate is enabled against Responses (which has no native stop param),
	// tolerate the stop.limit loss during request build so the local filter enforces stops.
	if opts.StopMode == ir.StopEmulate && upstream == ir.FormatResponses {
		allow := ir.LossSet{}
		for k, v := range env.Allow {
			allow[k] = v
		}
		allow[ir.LossStopLimit] = true
		env.Allow = allow
	}
	var body []byte
	var model string
	var stream bool
	var names map[string]ir.ToolRef

	switch upstream {
	case ir.FormatChat:
		built, err := chat.BuildRequest(req, chat.BuildOptions{Env: env, Dialect: opts.Chat})
		if err != nil {
			return nil, err
		}
		body, model, stream, names = built.Body, built.Model, built.Stream, built.Names
	case ir.FormatResponses:
		built, err := responses.BuildRequest(req, responses.BuildOptions{Env: env, Dialect: opts.Responses})
		if err != nil {
			return nil, err
		}
		body, model, stream, names = built.Body, built.Model, built.Stream, built.Names
	case ir.FormatAnthropic:
		built, err := anthropic.BuildRequest(req, anthropic.BuildOptions{Env: env, Dialect: opts.Anthropic})
		if err != nil {
			return nil, err
		}
		body, model, stream, names = built.Body, built.Model, built.Stream, built.Names
	case ir.FormatGemini:
		built, err := gemini.BuildRequest(req, gemini.BuildOptions{Env: env, Dialect: opts.Gemini})
		if err != nil {
			return nil, err
		}
		body, model, stream, names = built.Body, built.Model, built.Stream, built.Names
	default:
		return nil, ir.Invalid(ir.Root, "unknown upstream format: "+string(upstream))
	}

	return &Plan{
		Client:         client,
		Upstream:       upstream,
		Request:        req,
		Body:           body,
		Model:          model,
		ClientModel:    req.Model,
		ClientStream:   req.Stream,
		UpstreamStream: stream,
		Losses:         env.Losses(),
		opts:           opts,
		names:          names,
	}, nil
}

// NewStream returns a stream-to-stream converter.
func (p *Plan) NewStream() *Stream {
	return newStream(p, false)
}

// NewCollector returns a stream-to-JSON converter.
func (p *Plan) NewCollector() *Stream {
	return newStream(p, true)
}

// ConvertResponse converts a non-streaming upstream response body to a client response body.
func (p *Plan) ConvertResponse(body []byte) ([]byte, *ir.Response, error) {
	env := upstreamEnv(p.opts, p.Upstream)
	var irResp *ir.Response
	var err error

	switch p.Upstream {
	case ir.FormatChat:
		irResp, err = chat.ParseResponse(body, chat.DecodeOptions{Env: env, Dialect: p.opts.Chat, Names: p.names})
	case ir.FormatResponses:
		irResp, err = responses.ParseResponse(body, responses.DecodeOptions{Env: env, Dialect: p.opts.Responses, Names: p.names})
	case ir.FormatAnthropic:
		irResp, err = anthropic.ParseResponse(body, anthropic.DecodeOptions{Env: env, Names: p.names})
	case ir.FormatGemini:
		irResp, err = gemini.ParseResponse(body, gemini.DecodeOptions{Env: env, Dialect: p.opts.Gemini, Names: p.names})
	default:
		return nil, nil, ir.Invalid(ir.Root, "unknown upstream format: "+string(p.Upstream))
	}
	if err != nil {
		return nil, nil, err
	}
	irResp, err = applyResponseFilters(irResp, p)
	if err != nil {
		return nil, nil, err
	}

	clientEnv := clientEnv(p.opts, p.Client)
	encOpts := encodeOptions(p, clientEnv)
	var out []byte
	switch p.Client {
	case ir.FormatChat:
		out, err = chat.BuildResponse(irResp, chat.EncodeOptions{
			Env:     clientEnv,
			Request: p.Request,
			Model:   p.ClientModel,
		})
	case ir.FormatResponses:
		out, err = responses.BuildResponse(irResp, responses.EncodeOptions{
			Env:     clientEnv,
			Request: p.Request,
			Model:   p.ClientModel,
		})
	case ir.FormatAnthropic:
		out, err = anthropic.BuildResponse(irResp, anthropic.EncodeOptions{
			Env:     clientEnv,
			Request: p.Request,
			Model:   p.ClientModel,
		})
	case ir.FormatGemini:
		out, err = gemini.BuildResponse(irResp, gemini.EncodeOptions{
			Env:     clientEnv,
			Request: p.Request,
			Model:   p.ClientModel,
		})
	default:
		return nil, nil, ir.Invalid(ir.Root, "unknown client format: "+string(p.Client))
	}
	_ = encOpts
	return out, irResp, err
}

// StreamResponse converts a non-streaming upstream response body to a streaming client response.
func (p *Plan) StreamResponse(dst, body []byte) ([]byte, error) {
	_, irResp, err := p.ConvertResponse(body)
	if err != nil {
		return nil, err
	}
	// Re-encode as a stream using EventsOf (irResp already has response filters applied).
	evs := ir.EventsOf(irResp)
	env := clientEnv(p.opts, p.Client)
	encOpts := encodeOptions(p, env)
	s := newStreamWithEncoder(p, encOpts)
	for _, ev := range evs {
		for _, fe := range s.filter.process(ev) {
			dst, err = s.enc.Encode(dst, fe)
			if err != nil {
				return dst, err
			}
		}
	}
	if len(s.filter.intercepted) > 0 {
		return nil, env.Err(ir.Unsupported(ir.Root, "intercept", "StreamResponse cannot complete intercepted tool calls; use Plan.NewStream with Intercepted and NextRound"))
	}
	dst, err = s.enc.Close(dst)
	return dst, err
}

func applyResponseFilters(resp *ir.Response, p *Plan) (*ir.Response, error) {
	if resp == nil {
		return nil, nil
	}
	ro := effectiveReasoningOut(p)
	var stops []string
	if p != nil && p.opts.StopMode == ir.StopEmulate && p.Request != nil {
		for _, s := range p.Request.Stop {
			if s != "" {
				stops = append(stops, s)
			}
		}
	}
	maxCalls := 0
	if p != nil && p.opts.Limits.MaxToolCalls > 0 {
		maxCalls = p.opts.Limits.MaxToolCalls
	}

	cp := *resp
	cp.Outputs = make([]ir.Output, len(resp.Outputs))
	for i, o := range resp.Outputs {
		oc := o
		var parts []ir.Part
		var textRun []ir.Part
		stopped := false

		// flushTextRun checks StopEmulate across all contiguous PartText entries
		// (including those carrying citations or signatures). On a stop hit it
		// truncates at bestPos, drops all previously collected tool calls per
		// design.md §9.6, and marks stopped=true.
		flushTextRun := func() {
			if len(textRun) == 0 {
				return
			}
			if len(stops) == 0 {
				parts = append(parts, textRun...)
				textRun = nil
				return
			}
			var combined strings.Builder
			for _, tp := range textRun {
				combined.WriteString(tp.Text)
			}
			full := combined.String()
			bestPos, bestEnd, bestSeq := -1, -1, ""
			for _, seq := range stops {
				if idx := strings.Index(full, seq); idx >= 0 {
					end := idx + len(seq)
					if bestEnd < 0 || end < bestEnd || (end == bestEnd && idx < bestPos) {
						bestPos, bestEnd, bestSeq = idx, end, seq
					}
				}
			}
			if bestPos < 0 {
				parts = append(parts, textRun...)
				textRun = nil
				return
			}
			// Drop any tool calls already collected in this candidate when stop hits.
			kept := parts[:0]
			for _, ep := range parts {
				if ep.Type != ir.PartToolCall {
					kept = append(kept, ep)
				}
			}
			parts = kept

			rem := bestPos
			for _, tp := range textRun {
				if rem <= 0 {
					break
				}
				if len(tp.Text) <= rem {
					parts = append(parts, tp)
					rem -= len(tp.Text)
				} else {
					trimmed := tp
					trimmed.Text = tp.Text[:rem]
					if len(tp.Citations) > 0 {
						var cits []ir.Citation
						for _, c := range tp.Citations {
							if c.StartIndex < rem {
								if c.EndIndex > rem {
									c.EndIndex = rem
								}
								cits = append(cits, c)
							}
						}
						trimmed.Citations = cits
					}
					parts = append(parts, trimmed)
					rem = 0
					break
				}
			}
			textRun = nil
			oc.Stop = ir.Stop{Reason: ir.StopSequence, Sequence: bestSeq}
			stopped = true
		}

		for _, pt := range o.Parts {
			if stopped {
				break
			}
			if pt.Type == ir.PartText {
				textRun = append(textRun, pt)
				continue
			}
			flushTextRun()
			if stopped {
				break
			}
			if pt.Type == ir.PartReasoning {
				if ro == ir.ReasoningOutHidden {
					continue
				}
				if ro == ir.ReasoningOutSummary && pt.Reasoning != nil {
					rc := *pt.Reasoning
					if len(rc.Summary) == 0 && rc.Text != "" {
						rc.Summary = []string{rc.Text}
						rc.Text = ""
					}
					pt.Reasoning = &rc
				}
			}
			parts = append(parts, pt)
		}
		if !stopped {
			flushTextRun()
		}

		// Apply the same tool-call truncation/filter drop and JSON object validation
		// rules as the streaming pipeline (design.md §7.3).
		if oc.Stop.Truncated() || oc.Stop.Filtered() {
			kept := parts[:0]
			for _, ep := range parts {
				if ep.Type != ir.PartToolCall {
					kept = append(kept, ep)
				}
			}
			parts = kept
		} else {
			toolCount := 0
			for _, ep := range parts {
				if ep.Type != ir.PartToolCall || ep.Call == nil {
					continue
				}
				toolCount++
				if maxCalls > 0 && toolCount > maxCalls {
					return nil, ir.Protocol("tool call limit exceeded")
				}
				if msg := validateToolCall(ep.Call); msg != "" {
					return nil, ir.Protocol("invalid tool call: " + msg)
				}
			}
		}

		oc.Parts = parts
		cp.Outputs[i] = oc
	}
	return &cp, nil
}

// ConvertError converts an upstream error to a client error response.
func (p *Plan) ConvertError(status int, body []byte) (int, []byte) {
	var apiErr *ir.APIError
	switch p.Upstream {
	case ir.FormatChat:
		apiErr = chat.ParseError(status, body)
	case ir.FormatResponses:
		apiErr = responses.ParseError(status, body)
	case ir.FormatAnthropic:
		apiErr = anthropic.ParseError(status, body)
	case ir.FormatGemini:
		apiErr = gemini.ParseError(status, body)
	default:
		apiErr = &ir.APIError{Status: status, Type: ir.TypeForStatus(status), Raw: body}
	}
	return buildClientError(p.Client, apiErr)
}

// ErrorResponse produces a client-format error body from any error.
func ErrorResponse(client Format, err error) (int, []byte) {
	var apiErr *ir.APIError
	switch e := err.(type) {
	case *ir.APIError:
		apiErr = e
	case *ir.Error:
		apiErr = &ir.APIError{
			Status:  errorCodeToStatus(e.Code),
			Type:    errorCodeToType(e.Code),
			Message: e.Error(),
		}
	default:
		apiErr = &ir.APIError{Status: 500, Type: ir.ErrServer, Message: err.Error()}
	}
	return buildClientError(client, apiErr)
}

func buildClientError(client Format, apiErr *ir.APIError) (int, []byte) {
	switch client {
	case ir.FormatChat, ir.FormatResponses:
		return responses.BuildError(apiErr)
	case ir.FormatAnthropic:
		return anthropic.BuildError(apiErr)
	case ir.FormatGemini:
		return gemini.BuildError(apiErr)
	}
	return responses.BuildError(apiErr)
}

// --- Stream ---

// encoder is the common interface for all client stream encoders.
type encoder interface {
	Encode(dst []byte, ev ir.Event) ([]byte, error)
	Close(dst []byte) ([]byte, error)
	Keepalive(dst []byte) []byte
}

// decoder is the common interface for all upstream stream decoders.
type decoder interface {
	Decode(out []ir.Event, ev sse.Event) ([]ir.Event, error)
	End(how ir.EndKind) *ir.Event
}

// Stream is a live stream converter (decoder + filter pipeline + encoder).
type Stream struct {
	plan          *Plan
	dec           decoder
	enc           encoder
	filter        *streamFilter
	collector     *ir.Collector
	prevUsage     ir.Usage
	roundUsage    ir.Usage
	expectedCands int
	finishedCands map[int]bool
	done          bool
	errored       bool
}

func newStream(p *Plan, collect bool) *Stream {
	env := clientEnv(p.opts, p.Client)
	encOpts := encodeOptions(p, env)
	s := newStreamWithEncoder(p, encOpts)
	if collect {
		s.collector = ir.NewCollector()
		// Collector assembles a whole JSON response; always buffer calls so truncated
		// or invalid live calls never enter the collected output.
		s.filter.release = ir.ToolReleaseBuffered
	}
	return s
}

type encodeOpts struct {
	chat      chat.EncodeOptions
	responses responses.EncodeOptions
	anthropic anthropic.EncodeOptions
	gemini    gemini.EncodeOptions
}

func encodeOptions(p *Plan, env *ir.Env) encodeOpts {
	return encodeOpts{
		chat: chat.EncodeOptions{
			Env: env, Request: p.Request, Model: p.ClientModel,
			PromptEstimate: p.opts.PromptEstimate,
		},
		responses: responses.EncodeOptions{
			Env: env, Request: p.Request, Model: p.ClientModel,
			PromptEstimate: p.opts.PromptEstimate,
		},
		anthropic: anthropic.EncodeOptions{
			Env: env, Request: p.Request, Model: p.ClientModel,
			PromptEstimate: p.opts.PromptEstimate,
		},
		gemini: gemini.EncodeOptions{
			Env: env, Request: p.Request, Model: p.ClientModel,
			PromptEstimate: p.opts.PromptEstimate,
			AltSSE:         p.opts.AltSSE,
		},
	}
}

func newStreamWithEncoder(p *Plan, eo encodeOpts) *Stream {
	var enc encoder
	switch p.Client {
	case ir.FormatChat:
		enc = chat.NewStreamEncoder(eo.chat)
	case ir.FormatResponses:
		enc = responses.NewStreamEncoder(eo.responses)
	case ir.FormatAnthropic:
		enc = anthropic.NewStreamEncoder(eo.anthropic)
	case ir.FormatGemini:
		enc = gemini.NewStreamEncoder(eo.gemini)
	default:
		enc = chat.NewStreamEncoder(eo.chat)
	}

	upstreamEnv := upstreamEnv(p.opts, p.Upstream)
	var dec decoder
	switch p.Upstream {
	case ir.FormatChat:
		dec = chat.NewStreamDecoder(chat.DecodeOptions{Env: upstreamEnv, Dialect: p.opts.Chat, Names: p.names})
	case ir.FormatResponses:
		dec = responses.NewStreamDecoder(responses.DecodeOptions{Env: upstreamEnv, Dialect: p.opts.Responses, Names: p.names})
	case ir.FormatAnthropic:
		dec = anthropic.NewStreamDecoder(anthropic.DecodeOptions{
			Env:              upstreamEnv,
			Names:            p.names,
			MaxBufferedBytes: p.opts.Limits.MaxBufferedBytes,
		})
	case ir.FormatGemini:
		dec = gemini.NewStreamDecoder(gemini.DecodeOptions{Env: upstreamEnv, Dialect: p.opts.Gemini, Names: p.names})
	default:
		dec = chat.NewStreamDecoder(chat.DecodeOptions{Env: upstreamEnv, Dialect: p.opts.Chat, Names: p.names})
	}

	expected := 1
	if p != nil && p.Request != nil && p.Request.Candidates != nil && *p.Request.Candidates > 1 {
		expected = *p.Request.Candidates
	}
	return &Stream{
		plan:          p,
		dec:           dec,
		enc:           enc,
		filter:        newStreamFilter(p),
		expectedCands: expected,
		finishedCands: map[int]bool{},
	}
}

// Feed processes one upstream SSE event and appends any client bytes to dst.
func (s *Stream) Feed(dst []byte, ev sse.Event) ([]byte, error) {
	var events []ir.Event
	var err error
	events, err = s.dec.Decode(events, ev)
	if err != nil {
		return dst, err
	}
	for _, rawEv := range events {
		if rawEv.Usage != nil && (rawEv.Type == ir.EvStart || rawEv.Type == ir.EvUsage) {
			s.roundUsage.Merge(*rawEv.Usage)
		}
		filtered := s.filter.process(rawEv)
		for _, e := range filtered {
			if e.Type == ir.EvError {
				s.errored = true
				s.done = true
			}
			if e.Type == ir.EvStop {
				s.finishedCands[e.Index] = true
				if len(s.finishedCands) >= s.expectedCands {
					s.done = true
				}
			}
			if e.Type == ir.EvUsage && s.prevUsage.Any() {
				tot := s.Usage()
				e.Usage = &tot
			}
			if s.collector != nil {
				s.collector.Add(e)
			} else {
				dst, err = s.enc.Encode(dst, e)
				if err != nil {
					return dst, err
				}
			}
		}
	}
	return dst, nil
}

// End signals how the upstream ended and flushes any final client bytes.
func (s *Stream) End(dst []byte, how ir.EndKind) ([]byte, error) {
	earlyEmulatedDone := s.done && len(s.filter.stopTriggered) > 0
	if !s.errored && !earlyEmulatedDone && !s.filter.allStopsTriggered(s.expectedCands) {
		ev := s.dec.End(how)
		if ev == nil && !s.done && len(s.finishedCands) < s.expectedCands && len(s.filter.intercepted) == 0 {
			ev = ir.EndErrorEvent(how)
		}
		if ev != nil {
			for _, e := range s.filter.process(*ev) {
				if e.Type == ir.EvError {
					s.errored = true
					s.done = true
				}
				if e.Type == ir.EvStop {
					s.finishedCands[e.Index] = true
					if len(s.finishedCands) >= s.expectedCands {
						s.done = true
					}
				}
				if s.collector != nil {
					s.collector.Add(e)
				} else {
					var err error
					dst, err = s.enc.Encode(dst, e)
					if err != nil {
						return dst, err
					}
				}
			}
		}
	}
	// If intercepted tool calls are pending, keep the client stream open for NextRound.
	if len(s.filter.intercepted) > 0 {
		return dst, nil
	}
	if s.collector == nil {
		return s.enc.Close(dst)
	}
	return dst, nil
}

// Done reports whether the client response is complete.
func (s *Stream) Done() bool { return s.done }

// Keepalive sends a format-appropriate keepalive to the client.
func (s *Stream) Keepalive(dst []byte) []byte { return s.enc.Keepalive(dst) }

// Usage returns the cumulative usage collected across all rounds so far.
func (s *Stream) Usage() ir.Usage {
	tot := s.prevUsage
	tot.Add(s.roundUsage)
	return tot
}

// Result returns the collected client JSON response (collector mode only).
func (s *Stream) Result() ([]byte, *ir.Response, error) {
	if s.collector == nil {
		return nil, nil, nil
	}
	if err := s.collector.Err(); err != nil {
		return nil, nil, err
	}
	resp := s.collector.Response()
	resp.Usage = s.Usage()
	var out []byte
	var err error
	env := clientEnv(s.plan.opts, s.plan.Client)
	switch s.plan.Client {
	case ir.FormatChat:
		out, err = chat.BuildResponse(resp, chat.EncodeOptions{Env: env, Request: s.plan.Request, Model: s.plan.ClientModel})
	case ir.FormatResponses:
		out, err = responses.BuildResponse(resp, responses.EncodeOptions{Env: env, Request: s.plan.Request, Model: s.plan.ClientModel})
	case ir.FormatAnthropic:
		out, err = anthropic.BuildResponse(resp, anthropic.EncodeOptions{Env: env, Request: s.plan.Request, Model: s.plan.ClientModel})
	case ir.FormatGemini:
		out, err = gemini.BuildResponse(resp, gemini.EncodeOptions{Env: env, Request: s.plan.Request, Model: s.plan.ClientModel})
	}
	return out, resp, err
}

// Intercepted returns tool calls withheld for caller execution in the current round.
func (s *Stream) Intercepted() []ir.ToolCall {
	if s.filter == nil || len(s.filter.intercepted) == 0 {
		return nil
	}
	out := make([]ir.ToolCall, len(s.filter.intercepted))
	copy(out, s.filter.intercepted)
	return out
}

// NextRound swaps in a fresh decoder for a tool-round continuation while keeping
// the client encoder open and summing usage across rounds.
func (s *Stream) NextRound(p *Plan) {
	s.prevUsage.Add(s.roundUsage)
	s.roundUsage = ir.Usage{}
	s.plan = p
	upstreamEnv := upstreamEnv(p.opts, p.Upstream)
	switch p.Upstream {
	case ir.FormatChat:
		s.dec = chat.NewStreamDecoder(chat.DecodeOptions{Env: upstreamEnv, Dialect: p.opts.Chat, Names: p.names})
	case ir.FormatResponses:
		s.dec = responses.NewStreamDecoder(responses.DecodeOptions{Env: upstreamEnv, Dialect: p.opts.Responses, Names: p.names})
	case ir.FormatAnthropic:
		s.dec = anthropic.NewStreamDecoder(anthropic.DecodeOptions{Env: upstreamEnv, Names: p.names})
	case ir.FormatGemini:
		s.dec = gemini.NewStreamDecoder(gemini.DecodeOptions{Env: upstreamEnv, Dialect: p.opts.Gemini, Names: p.names})
	}
	if s.filter != nil {
		s.filter.resetRoundForPlan(p)
	}
	expected := 1
	if p != nil && p.Request != nil && p.Request.Candidates != nil && *p.Request.Candidates > 1 {
		expected = *p.Request.Candidates
	}
	s.expectedCands = expected
	s.finishedCands = map[int]bool{}
	s.done = false
}

// --- helpers ---

func clientEnv(opts Options, client ir.Format) *ir.Env {
	env := &ir.Env{
		ProviderID: opts.ProviderID,
		Profile:    opts.Profile,
		Allow:      opts.Allow,
		Deny:       opts.Deny,
		OnLoss:     opts.OnLoss,
		Unknown:    opts.Unknown,
		Repair:     opts.Repair,
		Seal:       opts.Seal,
		IDs:        opts.IDs,
		Clock:      opts.Clock,
		Style:      ir.StyleFor(client),
	}
	return env
}

func upstreamEnv(opts Options, upstream ir.Format) *ir.Env {
	env := &ir.Env{
		ProviderID: opts.ProviderID,
		Profile:    opts.Profile,
		Allow:      opts.Allow,
		Deny:       opts.Deny,
		OnLoss:     opts.OnLoss,
		Unknown:    opts.Unknown,
		Repair:     opts.Repair,
		Seal:       opts.Seal,
		IDs:        opts.IDs,
		Clock:      opts.Clock,
		Style:      ir.StyleFor(upstream),
	}
	return env
}

func errorCodeToStatus(c ir.ErrorCode) int {
	switch c {
	case ir.CodeInvalidRequest:
		return 400
	case ir.CodeUnsupported:
		return 400
	case ir.CodeConfig:
		return 500
	case ir.CodeInternal:
		return 500
	case ir.CodeUpstreamProtocol:
		return 502
	}
	return 500
}

func errorCodeToType(c ir.ErrorCode) ir.ErrorType {
	switch c {
	case ir.CodeInvalidRequest, ir.CodeUnsupported:
		return ir.ErrInvalidRequest
	case ir.CodeConfig, ir.CodeInternal:
		return ir.ErrServer
	case ir.CodeUpstreamProtocol:
		return ir.ErrServer
	}
	return ir.ErrServer
}
