// Package jsonx holds small JSON helpers shared by the format packages: ordered
// object reading, absent-value checks and append-style writers.
package jsonx

import (
	"bytes"
	"encoding/json"
	"errors"
	"strconv"
	"unicode/utf8"
)

// KV is one member of a JSON object, in document order.
type KV struct {
	Key   string
	Value json.RawMessage
}

// ErrNotObject is returned by Object for values that are not objects.
var ErrNotObject = errors.New("not a JSON object")

// Object splits a JSON object into its members in document order. Duplicate keys
// are kept; callers decide (the last one wins in encoding/json).
func Object(raw []byte) ([]KV, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || raw[0] != '{' {
		return nil, ErrNotObject
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	if _, err := d.Token(); err != nil {
		return nil, err
	}
	var out []KV
	for d.More() {
		t, err := d.Token()
		if err != nil {
			return nil, err
		}
		k, ok := t.(string)
		if !ok {
			return nil, ErrNotObject
		}
		var v json.RawMessage
		if err := d.Decode(&v); err != nil {
			return nil, err
		}
		out = append(out, KV{Key: k, Value: v})
	}
	if _, err := d.Token(); err != nil {
		return nil, err
	}
	return out, nil
}

// Kind returns the first non-space byte of a JSON value ('{', '[', '"', 'n', 't',
// 'f', or a digit / '-'), or 0 for an empty value.
func Kind(raw []byte) byte {
	for _, c := range raw {
		switch c {
		case ' ', '\t', '\r', '\n':
			continue
		}
		return c
	}
	return 0
}

// IsNull reports whether raw is empty or null.
func IsNull(raw []byte) bool {
	t := bytes.TrimSpace(raw)
	return len(t) == 0 || string(t) == "null"
}

// Absent reports whether an optional value counts as not given: empty, null, an
// empty object or an empty array.
func Absent(raw []byte) bool {
	t := bytes.TrimSpace(raw)
	if len(t) == 0 || string(t) == "null" {
		return true
	}
	if t[0] != '{' && t[0] != '[' {
		return false
	}
	inner := bytes.TrimSpace(t[1 : len(t)-1])
	return len(inner) == 0
}

// String decodes a JSON string.
func String(raw []byte) (string, bool) {
	if Kind(raw) != '"' {
		return "", false
	}
	var s string
	if json.Unmarshal(raw, &s) != nil {
		return "", false
	}
	return s, true
}

// Bool decodes a JSON boolean.
func Bool(raw []byte) (bool, bool) {
	switch string(bytes.TrimSpace(raw)) {
	case "true":
		return true, true
	case "false":
		return false, true
	}
	return false, false
}

// Float decodes a JSON number.
func Float(raw []byte) (float64, bool) {
	k := Kind(raw)
	if k != '-' && (k < '0' || k > '9') {
		return 0, false
	}
	f, err := strconv.ParseFloat(string(bytes.TrimSpace(raw)), 64)
	return f, err == nil
}

// Int decodes a JSON integer (a number with an integral value).
func Int(raw []byte) (int64, bool) {
	f, ok := Float(raw)
	if !ok || f != float64(int64(f)) {
		return 0, false
	}
	return int64(f), true
}

// Array splits a JSON array into its elements.
func Array(raw []byte) ([]json.RawMessage, bool) {
	if Kind(raw) != '[' {
		return nil, false
	}
	var out []json.RawMessage
	if json.Unmarshal(raw, &out) != nil {
		return nil, false
	}
	return out, true
}

// Marshal encodes v without HTML escaping and without a trailing newline.
func Marshal(v any) ([]byte, error) {
	var b bytes.Buffer
	e := json.NewEncoder(&b)
	e.SetEscapeHTML(false)
	if err := e.Encode(v); err != nil {
		return nil, err
	}
	out := b.Bytes()
	return out[:len(out)-1], nil
}

// MustMarshal is Marshal for values that cannot fail.
func MustMarshal(v any) []byte {
	b, err := Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}

// Compact returns raw without insignificant whitespace; invalid JSON is returned as is.
func Compact(raw []byte) []byte {
	var b bytes.Buffer
	if json.Compact(&b, raw) != nil {
		return raw
	}
	return b.Bytes()
}

const hex = "0123456789abcdef"

// AppendString appends s as a JSON string, escaping like encoding/json without
// HTML escaping. Invalid UTF-8 is replaced by U+FFFD.
func AppendString(dst []byte, s string) []byte {
	dst = append(dst, '"')
	start := 0
	for i := 0; i < len(s); {
		c := s[i]
		if c < utf8.RuneSelf {
			if c >= 0x20 && c != '"' && c != '\\' {
				i++
				continue
			}
			dst = append(dst, s[start:i]...)
			switch c {
			case '"', '\\':
				dst = append(dst, '\\', c)
			case '\n':
				dst = append(dst, '\\', 'n')
			case '\r':
				dst = append(dst, '\\', 'r')
			case '\t':
				dst = append(dst, '\\', 't')
			default:
				dst = append(dst, '\\', 'u', '0', '0', hex[c>>4], hex[c&0xf])
			}
			i++
			start = i
			continue
		}
		r, size := utf8.DecodeRuneInString(s[i:])
		if r == utf8.RuneError && size == 1 {
			dst = append(dst, s[start:i]...)
			dst = append(dst, `�`...)
			i += size
			start = i
			continue
		}
		if r == ' ' || r == ' ' {
			dst = append(dst, s[start:i]...)
			dst = append(dst, '\\', 'u', '2', '0', '2', hex[r&0xf])
			i += size
			start = i
			continue
		}
		i += size
	}
	dst = append(dst, s[start:]...)
	return append(dst, '"')
}

// Obj is a tiny append-style JSON object writer for hot paths.
type Obj struct {
	B     []byte
	first bool
}

// Start begins an object at the end of dst.
func Start(dst []byte) Obj { return Obj{B: append(dst, '{'), first: true} }

func (o *Obj) key(k string) {
	if !o.first {
		o.B = append(o.B, ',')
	}
	o.first = false
	o.B = AppendString(o.B, k)
	o.B = append(o.B, ':')
}

// Str writes a string member.
func (o *Obj) Str(k, v string) { o.key(k); o.B = AppendString(o.B, v) }

// Int writes an integer member.
func (o *Obj) Int(k string, v int64) { o.key(k); o.B = strconv.AppendInt(o.B, v, 10) }

// Bool writes a boolean member.
func (o *Obj) Bool(k string, v bool) { o.key(k); o.B = strconv.AppendBool(o.B, v) }

// Raw writes a member whose value is already JSON.
func (o *Obj) Raw(k string, v []byte) { o.key(k); o.B = append(o.B, v...) }

// Null writes a null member.
func (o *Obj) Null(k string) { o.key(k); o.B = append(o.B, "null"...) }

// Key writes a key; the caller appends the value to o.B.
func (o *Obj) Key(k string) { o.key(k) }

// End closes the object and returns the buffer.
func (o *Obj) End() []byte { return append(o.B, '}') }
