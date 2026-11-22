// Package bridge wires a client format to an upstream format.
package bridge

import (
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
	req, err := Parse(client, body, opts)
	if err != nil {
		return nil, err
	}
	return Build(req, client, upstream, opts)
}

// Parse decodes a client request body into an ir.Request.
func Parse(client Format, body []byte, opts Options) (*ir.Request, error) {
	env := clientEnv(opts, client)
	switch client {
	case ir.FormatChat:
		return chat.ParseRequest(body, chat.ParseOptions{Env: env})
	case ir.FormatResponses:
		return responses.ParseRequest(body, responses.ParseOptions{Env: env})
	case ir.FormatAnthropic:
		return anthropic.ParseRequest(body, anthropic.ParseOptions{Env: env})
	case ir.FormatGemini:
		return gemini.ParseRequest(body, gemini.ParseOptions{Env: env})
	}
	return nil, ir.Invalid(ir.Root, "unknown client format: "+string(client))
}

// Build serialises an ir.Request into an upstream body and returns a Plan.
func Build(req *ir.Request, client, upstream Format, opts Options) (*Plan, error) {
	env := upstreamEnv(opts, upstream)
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
	out, irResp, err := p.ConvertResponse(body)
	if err != nil {
		return nil, err
	}
	_ = irResp
	// Re-encode as a stream using EventsOf.
	evs := ir.EventsOf(irResp)
	env := clientEnv(p.opts, p.Client)
	encOpts := encodeOptions(p, env)
	s := newStreamWithEncoder(p, encOpts)
	for _, ev := range evs {
		dst, err = s.enc.Encode(dst, ev)
		if err != nil {
			return dst, err
		}
	}
	dst, err = s.enc.Close(dst)
	_ = out
	return dst, err
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

// Stream is a live stream converter (decoder + encoder).
type Stream struct {
	plan      *Plan
	dec       decoder
	enc       encoder
	collector *ir.Collector
	usage     ir.Usage
	done      bool
}

func newStream(p *Plan, collect bool) *Stream {
	env := clientEnv(p.opts, p.Client)
	encOpts := encodeOptions(p, env)
	s := newStreamWithEncoder(p, encOpts)
	if collect {
		s.collector = ir.NewCollector()
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
		dec = anthropic.NewStreamDecoder(anthropic.DecodeOptions{Env: upstreamEnv, Names: p.names})
	case ir.FormatGemini:
		dec = gemini.NewStreamDecoder(gemini.DecodeOptions{Env: upstreamEnv, Dialect: p.opts.Gemini, Names: p.names})
	default:
		dec = chat.NewStreamDecoder(chat.DecodeOptions{Env: upstreamEnv, Dialect: p.opts.Chat, Names: p.names})
	}

	return &Stream{plan: p, dec: dec, enc: enc}
}

// Feed processes one upstream SSE event and appends any client bytes to dst.
func (s *Stream) Feed(dst []byte, ev sse.Event) ([]byte, error) {
	var events []ir.Event
	var err error
	events, err = s.dec.Decode(events, ev)
	if err != nil {
		return dst, err
	}
	for _, e := range events {
		if e.Usage != nil && (e.Type == ir.EvStart || e.Type == ir.EvUsage) {
			s.usage.Merge(*e.Usage)
		}
		if e.Type == ir.EvError {
			s.done = true
		}
		if e.Type == ir.EvStop {
			s.done = true
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
	return dst, nil
}

// End signals how the upstream ended and flushes any final client bytes.
func (s *Stream) End(dst []byte, how ir.EndKind) ([]byte, error) {
	if ev := s.dec.End(how); ev != nil {
		if s.collector != nil {
			s.collector.Add(*ev)
		} else {
			var err error
			dst, err = s.enc.Encode(dst, *ev)
			if err != nil {
				return dst, err
			}
		}
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

// Usage returns the usage collected so far.
func (s *Stream) Usage() ir.Usage {
	if s.collector != nil {
		return s.collector.Usage()
	}
	return s.usage
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

// Intercepted returns tool calls for caller-owned tools (not yet implemented).
func (s *Stream) Intercepted() []ir.ToolCall { return nil }

// NextRound swaps in a fresh decoder for a tool-round continuation.
func (s *Stream) NextRound(p *Plan) {
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
