package ir

// ToolRelease controls when tool calls are delivered to the client.
type ToolRelease uint8

// Tool release modes.
const (
	// ToolReleaseDefault uses the default for the client format.
	ToolReleaseDefault ToolRelease = iota
	// ToolReleaseBuffered holds all calls until the upstream finishes, then
	// delivers them all at once after validation. Default for Anthropic, Gemini.
	ToolReleaseBuffered
	// ToolReleaseLiveFirst streams the first call's arguments live; later calls
	// are buffered. Default for Responses.
	ToolReleaseLiveFirst
	// ToolReleaseStream forwards calls as they arrive. Default for Chat.
	ToolReleaseStream
)

// StopMode controls how stop sequences are enforced.
type StopMode uint8

// Stop modes.
const (
	// StopNative sends stop sequences to the upstream and trusts its result.
	StopNative StopMode = iota
	// StopEmulate sends stop sequences to the upstream AND scans output text,
	// stopping as soon as a match is found. Required when the upstream ignores
	// stop sequences (some open-weight models).
	StopEmulate
)

// ReasoningOut controls how reasoning is presented to the client.
type ReasoningOut uint8

// Reasoning output modes.
const (
	// ReasoningOutNative presents reasoning in the client format's native form.
	ReasoningOutNative ReasoningOut = iota
	// ReasoningOutSummary presents reasoning as a Responses summary item.
	ReasoningOutSummary
	// ReasoningOutHidden drops reasoning from the client response.
	ReasoningOutHidden
)

// Limits bounds per-response resource use.
type Limits struct {
	// MaxToolCalls is the maximum number of tool calls per response (0 = no limit).
	MaxToolCalls int
	// MaxBufferedBytes is the maximum bytes of buffered tool arguments and deferred stream events (0 = 8 MiB).
	MaxBufferedBytes int
	// MaxStopSequences, when > 0, caps the number of client stop sequences accepted by Build.
	MaxStopSequences int
	// MaxStopSequenceBytes, when > 0, caps the combined byte length of all client stop sequences.
	MaxStopSequenceBytes int
}
