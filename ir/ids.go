package ir

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"strconv"
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

// EncodeCallSignatureID attaches a sealed Gemini tool-call signature to a call
// id so formats without a dedicated call-signature field (Chat, Responses,
// Anthropic) can carry it back on the next turn.
func EncodeCallSignatureID(env *Env, id string, sig *Signature) string {
	if sig != nil && sig.Value != "" && sig.Origin.Format == FormatGemini {
		sealed := env.Sealer().Seal(SealRecord{
			Origin: sig.Origin,
			Kind:   SealSignature,
			Value:  sig.Value,
		})
		return id + "::" + sealed
	}
	return id
}

// DecodeCallSignatureID reverses EncodeCallSignatureID, returning the original
// call id and any unsealed signature.
func DecodeCallSignatureID(env *Env, id string) (string, *Signature) {
	if idx := strings.LastIndex(id, "::"); idx >= 0 {
		if rec, ok := env.Sealer().Unseal(id[idx+2:]); ok && rec.Kind == SealSignature {
			return id[:idx], &Signature{Value: rec.Value, Origin: rec.Origin}
		}
	}
	return id, nil
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

// UnsignedTextSealer is an optional interface a SealCodec may implement when
// sealing and verifying unsigned reasoning text with a custom wire format (for
// example, a domain-separated digest or compressed envelope).
type UnsignedTextSealer interface {
	SealUnsigned(client Format, origin Origin, text string) string
	VerifyUnsigned(client Format, sealed, text string) (origin Origin, recoveredText string, ok bool)
}

// SealUnsignedReasoning seals unsigned reasoning text for client using env's
// Sealer. For Anthropic clients the default SealRecord stores only TextSum (the
// text already lives in the thinking block); for Responses clients it carries
// Text inside the envelope.
func SealUnsignedReasoning(env *Env, client Format, origin Origin, text string) string {
	if text == "" {
		return ""
	}
	sealer := env.Sealer()
	if u, ok := sealer.(UnsignedTextSealer); ok {
		return u.SealUnsigned(client, origin, text)
	}
	rec := SealRecord{Origin: origin, Kind: SealUnsigned}
	if client == FormatAnthropic {
		rec.TextSum = TextSum(text)
	} else {
		rec.Text = text
	}
	return sealer.Seal(rec)
}

// VerifyUnsignedReasoning checks whether sealed is a verified unsigned
// reasoning envelope produced by env's Sealer for client (and matching text
// when provided).
func VerifyUnsignedReasoning(env *Env, client Format, sealed, text string) (Origin, string, bool) {
	if sealed == "" {
		return Origin{}, "", false
	}
	sealer := env.Sealer()
	if u, ok := sealer.(UnsignedTextSealer); ok {
		return u.VerifyUnsigned(client, sealed, text)
	}
	rec, ok := sealer.Unseal(sealed)
	if !ok || rec.Kind != SealUnsigned {
		return Origin{}, "", false
	}
	if rec.TextSum != "" && rec.TextSum != TextSum(text) {
		return Origin{}, "", false
	}
	outText := text
	if rec.Text != "" {
		outText = rec.Text
	}
	return rec.Origin, outText, true
}

// CustomToolWrapperSchema is the standard JSON Schema used to expose a freeform
// custom tool to formats that only support JSON-schema function tools (design.md §9.2.1).
var CustomToolWrapperSchema = json.RawMessage(`{"type":"object","properties":{"input":{"type":"string","description":"The raw input for this tool, exactly in the format the tool description requires."}},"required":["input"],"additionalProperties":false}`)

// AssignToolWireName flattens a tool's namespace/name, resolves collisions with
// _2/_3 suffixes, records the reverse mapping in names, and returns the wire name
// plus the updated description (with [<namespace>] prefix and optional grammar note).
func AssignToolWireName(names map[string]ToolRef, t Tool) (wireName, desc string) {
	base := t.Name
	if t.Namespace != "" {
		base = t.Namespace + "__" + t.Name
	}
	wireName = base
	for n := 2; ; n++ {
		if _, exists := names[wireName]; !exists {
			break
		}
		wireName = base + "_" + strconv.Itoa(n)
	}
	if names != nil {
		names[wireName] = ToolRef{Name: t.Name, Namespace: t.Namespace, Kind: t.Kind}
	}
	desc = t.Description
	if t.Namespace != "" && !strings.HasPrefix(desc, "["+t.Namespace+"] ") {
		if desc == "" {
			desc = "[" + t.Namespace + "]"
		} else {
			desc = "[" + t.Namespace + "] " + desc
		}
	}
	return wireName, desc
}

// ToolNameMap holds consistent forward and reverse wire-name mappings across
// tool definitions, historical calls, tool results, and tool_choice entries.
type ToolNameMap struct {
	ToWire   map[string]string
	FromWire map[string]ToolRef
	Descs    map[string]string
	BareWire map[string]string
}

// BuildToolNameMap precomputes collision-free wire names for all tools in order.
func BuildToolNameMap(tools []Tool) *ToolNameMap {
	m := &ToolNameMap{
		ToWire:   map[string]string{},
		FromWire: map[string]ToolRef{},
		Descs:    map[string]string{},
		BareWire: map[string]string{},
	}
	bareCount := map[string]int{}
	for _, t := range tools {
		if t.Name == "" {
			continue
		}
		bareCount[t.Name]++
		wire, desc := AssignToolWireName(m.FromWire, t)
		key := t.Namespace + "\x00" + t.Name
		m.ToWire[key] = wire
		m.Descs[key] = desc
	}
	for _, t := range tools {
		if t.Name != "" && bareCount[t.Name] == 1 {
			m.BareWire[t.Name] = m.ToWire[t.Namespace+"\x00"+t.Name]
		}
	}
	return m
}

// WireName returns the assigned wire name for (ns, name).
func (m *ToolNameMap) WireName(ns, name string) string {
	if name == "" {
		return ""
	}
	if m != nil {
		if w, ok := m.ToWire[ns+"\x00"+name]; ok {
			return w
		}
		if ns == "" {
			if w, ok := m.BareWire[name]; ok {
				return w
			}
		}
	}
	if ns != "" {
		return ns + "__" + name
	}
	return name
}

// Description returns the precomputed description for (ns, name) if present.
func (m *ToolNameMap) Description(ns, name, fallback string) string {
	if m != nil {
		if d, ok := m.Descs[ns+"\x00"+name]; ok {
			return d
		}
	}
	return fallback
}
