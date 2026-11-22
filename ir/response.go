package ir

import "encoding/json"

// Response is a whole model reply.
type Response struct {
	ID      string   `json:"id,omitempty"`
	Model   string   `json:"model,omitempty"`
	Outputs []Output `json:"outputs"`
	Usage   Usage    `json:"usage"`
	Created int64    `json:"created,omitempty"`
	// Losses records any tolerated lossy conversions during response decoding/encoding.
	Losses []Loss `json:"-"`
}

// Output is one candidate of a reply.
type Output struct {
	Index int    `json:"index"`
	Parts []Part `json:"parts,omitempty"`
	Stop  Stop   `json:"stop"`
}

// First returns the first candidate, creating it if missing.
func (r *Response) First() *Output {
	if len(r.Outputs) == 0 {
		r.Outputs = append(r.Outputs, Output{})
	}
	return &r.Outputs[0]
}

// StopReason is why a candidate ended, normalised across formats.
type StopReason string

// Stop reasons.
const (
	StopUnset         StopReason = ""
	StopEndTurn       StopReason = "end_turn"
	StopMaxTokens     StopReason = "max_tokens"
	StopSequence      StopReason = "stop_sequence"
	StopToolUse       StopReason = "tool_use"
	StopContentFilter StopReason = "content_filter"
	StopRefusal       StopReason = "refusal"
	StopPauseTurn     StopReason = "pause_turn"
	StopContextWindow StopReason = "context_window"
	StopOther         StopReason = "other"
)

// Stop describes how a candidate ended.
type Stop struct {
	Reason   StopReason `json:"reason,omitempty"`
	Sequence string     `json:"sequence,omitempty"`
	// Raw is the upstream's own word for the reason.
	Raw string `json:"raw,omitempty"`
	// Details is provider detail about a refusal, passed through as written.
	Details json.RawMessage `json:"details,omitempty"`
}

// Truncated reports whether the output was cut short by a length limit.
func (s Stop) Truncated() bool {
	return s.Reason == StopMaxTokens || s.Reason == StopContextWindow
}

// Filtered reports whether the output was stopped by a safety filter.
func (s Stop) Filtered() bool {
	return s.Reason == StopContentFilter || s.Reason == StopRefusal
}

// EventType is the kind of a streaming Event.
type EventType uint8

// Event types.
const (
	EvStart          EventType = iota + 1 // ID, Model, Usage (prompt side)
	EvText                                // Text
	EvRefusal                             // Text (model refusal)
	EvCitation                            // Citation
	EvReasoningStart                      // ID
	EvReasoning                           // Text, Summary
	EvSignature                           // Signature
	EvSealed                              // Sealed
	EvToolStart                           // ID, Name, Namespace, Kind
	EvToolArgs                            // Text
	EvToolEnd                             // Arguments (optional final arguments)
	EvBuiltin                             // Builtin
	EvImage                               // Media
	EvStop                                // Stop
	EvUsage                               // Usage (cumulative snapshot)
	EvError                               // Err
)

var eventNames = [...]string{"", "start", "text", "refusal", "citation", "reasoning_start", "reasoning",
	"signature", "sealed", "tool_start", "tool_args", "tool_end", "builtin", "image", "stop",
	"usage", "error"}

func (t EventType) String() string {
	if int(t) < len(eventNames) {
		return eventNames[t]
	}
	return "unknown"
}

// Event is one thing a streamed reply said. Only the fields its Type uses are set.
type Event struct {
	Type EventType
	// Index is the candidate index (0 unless several candidates were asked for).
	Index int

	ID        string
	Name      string
	Namespace string
	Kind      ToolKind
	Model     string
	Text      string
	// Summary marks reasoning text that is a summary rather than full reasoning;
	// Part is then the summary part index.
	Summary bool
	Part    int

	Arguments json.RawMessage
	Usage     *Usage
	Stop      *Stop
	Err       *APIError
	Signature *Signature
	Sealed    *Sealed
	Builtin   *BuiltinActivity
	Media     *Media
	Citation  *Citation
}

// EndKind is how an upstream stream ended.
type EndKind uint8

// Stream end kinds.
const (
	EndDone    EndKind = iota // an explicit end marker ([DONE]) or a terminal event
	EndEOF                    // the connection closed
	EndTimeout                // the caller's read deadline passed
	EndError                  // a transport error
)

// EndError returns the terminal error event for a stream that ended without a
// terminal event, or nil when how is EndDone.
func EndErrorEvent(how EndKind) *Event {
	switch how {
	case EndDone:
		return nil
	case EndTimeout:
		return &Event{Type: EvError, Err: &APIError{Status: 504, Type: ErrTimeout, Message: "upstream stream timed out"}}
	default:
		return &Event{Type: EvError, Err: &APIError{Status: 502, Type: ErrServer, Message: "upstream stream ended without a finish reason"}}
	}
}
