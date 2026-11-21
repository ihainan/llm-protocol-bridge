package ir

import (
	"strconv"
	"strings"
	"time"
)

// PathElem is one step of a JSON path: an object key, or an array index when Key
// is empty.
type PathElem struct {
	Key   string
	Index int
}

// Path is a structured JSON path into a request body.
type Path []PathElem

// Root is the empty path.
var Root Path

// Key returns p extended by an object key. p itself is never modified.
func (p Path) Key(k string) Path {
	out := make(Path, len(p), len(p)+1)
	copy(out, p)
	return append(out, PathElem{Key: k})
}

// At returns p extended by an array index.
func (p Path) At(i int) Path {
	out := make(Path, len(p), len(p)+1)
	copy(out, p)
	return append(out, PathElem{Index: i})
}

// Keys returns p extended by several object keys.
func (p Path) Keys(ks ...string) Path {
	out := make(Path, len(p), len(p)+len(ks))
	copy(out, p)
	for _, k := range ks {
		out = append(out, PathElem{Key: k})
	}
	return out
}

// PathStyle selects how a Path is written.
type PathStyle uint8

// Path styles.
const (
	// StyleBracket writes input[2].call_id (OpenAI, Gemini).
	StyleBracket PathStyle = iota
	// StyleDot writes messages.0.content.1 (Anthropic).
	StyleDot
)

// StyleFor returns the path style of a client format.
func StyleFor(f Format) PathStyle {
	if f == FormatAnthropic {
		return StyleDot
	}
	return StyleBracket
}

// Render writes p in the given style.
func (p Path) Render(style PathStyle) string {
	var b strings.Builder
	for i, e := range p {
		if e.Key == "" {
			if style == StyleDot {
				if i > 0 {
					b.WriteByte('.')
				}
				b.WriteString(strconv.Itoa(e.Index))
			} else {
				b.WriteByte('[')
				b.WriteString(strconv.Itoa(e.Index))
				b.WriteByte(']')
			}
			continue
		}
		if i > 0 {
			b.WriteByte('.')
		}
		b.WriteString(e.Key)
	}
	return b.String()
}

func (p Path) String() string { return p.Render(StyleBracket) }

// ErrorCode classifies library errors.
type ErrorCode string

// Error codes.
const (
	// CodeInvalidRequest: the body violates its own format.
	CodeInvalidRequest ErrorCode = "invalid_request"
	// CodeUnsupported: valid in the client format, not expressible upstream.
	CodeUnsupported ErrorCode = "unsupported"
	// CodeUpstreamProtocol: the upstream reply could not be understood.
	CodeUpstreamProtocol ErrorCode = "upstream_protocol"
	// CodeConfig: the caller's options are inconsistent.
	CodeConfig ErrorCode = "config"
	// CodeInternal: a bug in the library.
	CodeInternal ErrorCode = "internal"
)

// Error is a conversion error. Request errors carry the path of the offending
// field in the client body.
type Error struct {
	Code    ErrorCode
	Path    Path
	Feature string
	Message string
	// Style is how Path is rendered in Error(); set from the client format.
	Style PathStyle
}

func (e *Error) Error() string {
	if len(e.Path) == 0 {
		return e.Message
	}
	return e.Path.Render(e.Style) + ": " + e.Message
}

// Param is the rendered path.
func (e *Error) Param() string { return e.Path.Render(e.Style) }

// Status is the HTTP status a gateway should answer with.
func (e *Error) Status() int {
	switch e.Code {
	case CodeInvalidRequest, CodeUnsupported:
		return 400
	case CodeUpstreamProtocol:
		return 502
	default:
		return 500
	}
}

// Invalid returns an invalid_request error.
func Invalid(p Path, msg string) *Error {
	return &Error{Code: CodeInvalidRequest, Path: p, Message: msg}
}

// Unsupported returns an unsupported error for a feature key.
func Unsupported(p Path, feature, msg string) *Error {
	return &Error{Code: CodeUnsupported, Path: p, Feature: feature, Message: msg}
}

// Protocol returns an upstream_protocol error.
func Protocol(msg string) *Error {
	return &Error{Code: CodeUpstreamProtocol, Message: msg}
}

// ConfigError returns a config error. The message is meant for logs.
func ConfigError(msg string) *Error {
	return &Error{Code: CodeConfig, Message: msg}
}

// ErrorType is a normalised upstream error category.
type ErrorType string

// Upstream error categories.
const (
	ErrInvalidRequest  ErrorType = "invalid_request"
	ErrAuthentication  ErrorType = "authentication"
	ErrBilling         ErrorType = "billing"
	ErrPermission      ErrorType = "permission"
	ErrNotFound        ErrorType = "not_found"
	ErrRequestTooLarge ErrorType = "request_too_large"
	ErrRateLimit       ErrorType = "rate_limit"
	ErrOverloaded      ErrorType = "overloaded"
	ErrServer          ErrorType = "server"
	ErrTimeout         ErrorType = "timeout"
)

// APIError is an upstream error in any format.
type APIError struct {
	Status     int
	Type       ErrorType
	Code       string
	Message    string
	Param      string
	RetryAfter time.Duration
	// Raw is the upstream error body as received.
	Raw []byte
}

func (e *APIError) Error() string {
	return strconv.Itoa(e.Status) + " " + string(e.Type) + ": " + e.Message
}

// TypeForStatus maps an HTTP status to an error category.
func TypeForStatus(status int) ErrorType {
	switch status {
	case 400, 422:
		return ErrInvalidRequest
	case 401:
		return ErrAuthentication
	case 402:
		return ErrBilling
	case 403:
		return ErrPermission
	case 404:
		return ErrNotFound
	case 413:
		return ErrRequestTooLarge
	case 429:
		return ErrRateLimit
	case 408, 504:
		return ErrTimeout
	case 503, 529:
		return ErrOverloaded
	}
	if status >= 500 {
		return ErrServer
	}
	if status >= 400 {
		return ErrInvalidRequest
	}
	return ErrServer
}

// StatusForType maps a category back to a representative HTTP status.
func StatusForType(t ErrorType) int {
	switch t {
	case ErrInvalidRequest:
		return 400
	case ErrAuthentication:
		return 401
	case ErrBilling:
		return 402
	case ErrPermission:
		return 403
	case ErrNotFound:
		return 404
	case ErrRequestTooLarge:
		return 413
	case ErrRateLimit:
		return 429
	case ErrOverloaded:
		return 503
	case ErrTimeout:
		return 504
	}
	return 500
}

// IsContextLength reports whether an upstream error says the prompt is longer
// than the model's context window, whatever the provider's wording.
func IsContextLength(e *APIError) bool {
	if e == nil {
		return false
	}
	if e.Code == "context_length_exceeded" || e.Code == "string_above_max_length" {
		return true
	}
	m := strings.ToLower(e.Message)
	for _, s := range []string{
		"context length", "context_length", "context window", "maximum context",
		"prompt is too long", "too many tokens", "input is too long",
		"exceeds the maximum number of tokens", "input token count",
	} {
		if strings.Contains(m, s) {
			return true
		}
	}
	return false
}
