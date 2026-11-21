// Package gemini implements the Gemini generateContent wire format.
package gemini

// Dialect describes how an upstream deviates from the standard Gemini API.
type Dialect struct {
	// SchemaField selects which field carries the response schema:
	// "" or "responseSchema": uses responseSchema; "response_schema": uses that name.
	SchemaField string

	// URLFileData: when true, file URIs in fileData parts are supported (Vertex AI).
	URLFileData bool

	// FunctionCallIDs: when true, the upstream supports function call IDs in
	// function call and response parts.
	FunctionCallIDs bool

	// MidSystem controls system/developer messages in the middle of a conversation:
	// "" or "keep": passed through; "fold": merged into the nearest assistant turn.
	MidSystem string
}
