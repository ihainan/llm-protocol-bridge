// Package responses implements the OpenAI Responses API wire format.
package responses

// Dialect describes how an upstream deviates from the standard Responses API.
type Dialect struct {
	// StoreDefault is the value assumed for store when not set by the client.
	StoreDefault bool

	// CacheWriteField, when non-empty, is the field name in usage that carries
	// cache-write token counts (non-standard extension).
	CacheWriteField string
}
