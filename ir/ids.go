package ir

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"strings"
)

// DeterministicCallID derives a call id for a call that arrived without one, from
// its position and content, so that the same history always yields the same id.
func DeterministicCallID(msgIndex, callIndex int, name string, args []byte) string {
	h := sha256.New()
	var buf [16]byte
	binary.BigEndian.PutUint64(buf[:8], uint64(msgIndex))
	binary.BigEndian.PutUint64(buf[8:], uint64(callIndex))
	h.Write(buf[:])
	h.Write([]byte(name))
	h.Write([]byte{0})
	h.Write(args)
	return "call_" + hex.EncodeToString(h.Sum(nil)[:12])
}

const anthropicIDPrefix = "lpb_"

func safeIDChar(c byte) bool {
	return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' || c == '-'
}

// AnthropicSafeID encodes a call id so that it matches Anthropic's
// ^[a-zA-Z0-9_-]+$ rule. Ids that already match are kept; others become
// "lpb_" + base64url(id). AnthropicRawID reverses it.
func AnthropicSafeID(id string) string {
	ok := id != "" && !strings.HasPrefix(id, anthropicIDPrefix)
	for i := 0; ok && i < len(id); i++ {
		ok = safeIDChar(id[i])
	}
	if ok {
		return id
	}
	return anthropicIDPrefix + base64.RawURLEncoding.EncodeToString([]byte(id))
}

// AnthropicRawID reverses AnthropicSafeID.
func AnthropicRawID(id string) string {
	if !strings.HasPrefix(id, anthropicIDPrefix) {
		return id
	}
	b, err := base64.RawURLEncoding.DecodeString(id[len(anthropicIDPrefix):])
	if err != nil {
		return id
	}
	return string(b)
}

// SealKind is what a seal envelope carries.
type SealKind string

// Seal kinds.
const (
	// SealSignature carries an issuer's reasoning signature.
	SealSignature SealKind = "sig"
	// SealRedacted carries an Anthropic redacted_thinking block.
	SealRedacted SealKind = "red"
	// SealEncrypted carries a Responses reasoning item with encrypted_content.
	SealEncrypted SealKind = "enc"
	// SealUnsigned marks reasoning whose issuer gave no signature.
	SealUnsigned SealKind = "uns"
)

// SealRecord is the provenance of a reasoning value carried through a client.
type SealRecord struct {
	Origin Origin   `json:"o"`
	Kind   SealKind `json:"k"`
	// Value is the signature, or the sealed block / item as raw JSON text.
	Value string `json:"v,omitempty"`
	// Text is reasoning text carried along (Responses clients, unsigned reasoning).
	Text string `json:"t,omitempty"`
	// TextSum is a checksum of reasoning text the client holds elsewhere (an
	// Anthropic thinking block), so that a modified block is detected.
	TextSum string `json:"s,omitempty"`
}

// SealCodec writes provenance into a client format's opaque reasoning fields and
// reads it back.
type SealCodec interface {
	Seal(r SealRecord) string
	// Unseal decodes an envelope; ok is false for values it did not write.
	Unseal(s string) (r SealRecord, ok bool)
}

// NewSealCodec returns the default envelope codec. With a key, the checksum is an
// HMAC and envelopes written with another key are not recognised.
func NewSealCodec(key []byte) SealCodec { return sealCodec{key: key} }

// DefaultSeal is the keyless default codec.
var DefaultSeal = NewSealCodec(nil)

const sealPrefix = "lpb1."

type sealCodec struct{ key []byte }

func (c sealCodec) sum(b []byte) string {
	var s []byte
	if len(c.key) > 0 {
		m := hmac.New(sha256.New, c.key)
		m.Write(b)
		s = m.Sum(nil)
	} else {
		h := sha256.Sum256(b)
		s = h[:]
	}
	return base64.RawURLEncoding.EncodeToString(s[:16])
}

func (c sealCodec) Seal(r SealRecord) string {
	b, _ := json.Marshal(r)
	body := base64.RawURLEncoding.EncodeToString(b)
	return sealPrefix + body + "." + c.sum([]byte(body))
}

func (c sealCodec) Unseal(s string) (SealRecord, bool) {
	var r SealRecord
	if !strings.HasPrefix(s, sealPrefix) {
		return r, false
	}
	rest := s[len(sealPrefix):]
	dot := strings.LastIndexByte(rest, '.')
	if dot < 0 {
		return r, false
	}
	body, sum := rest[:dot], rest[dot+1:]
	if !hmac.Equal([]byte(sum), []byte(c.sum([]byte(body)))) {
		return r, false
	}
	b, err := base64.RawURLEncoding.DecodeString(body)
	if err != nil || json.Unmarshal(b, &r) != nil {
		return r, false
	}
	return r, true
}

// TextSum is the checksum stored in SealRecord.TextSum.
func TextSum(text string) string {
	h := sha256.Sum256([]byte(text))
	return base64.RawURLEncoding.EncodeToString(h[:12])
}

// IsSealed reports whether s looks like an envelope of the default format.
func IsSealed(s string) bool { return strings.HasPrefix(s, sealPrefix) }
