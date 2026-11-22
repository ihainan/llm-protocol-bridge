package jsonx

import (
	"encoding/json"
	"reflect"
	"strings"
)

// ReconcileCustomInput checks streamed raw custom tool input (sent, when
// hasSent is true) against the final EvToolEnd.Arguments JSON string. It
// returns the full final input and the unsent suffix (which equals full when
// !hasSent). Returns ok=false if final is not a JSON string or sent is not a
// prefix of final.
func ReconcileCustomInput(sent string, hasSent bool, final json.RawMessage) (full, suffix string, ok bool) {
	if len(final) == 0 {
		if !hasSent {
			return sent, sent, true
		}
		return sent, "", true
	}
	if Kind(final) != '"' {
		return "", "", false
	}
	var in string
	if json.Unmarshal(final, &in) != nil {
		return "", "", false
	}
	if !hasSent {
		return in, in, true
	}
	if !strings.HasPrefix(in, sent) {
		return "", "", false
	}
	return in, in[len(sent):], true
}

// ReconcileFunctionArgs reconciles streamed function-call JSON argument text
// (sent) against final EvToolEnd.Arguments. It returns the full validated JSON
// object string (defaulting to "{}" when both are empty) and any unsent suffix
// to append to the wire stream. Returns ok=false if the resulting arguments are
// not a valid JSON object or if sent disagrees with final (neither a prefix nor
// semantically equal valid JSON).
func ReconcileFunctionArgs(sent, final string) (full, suffix string, ok bool) {
	finalTrim := strings.TrimSpace(final)
	if sent == "" {
		if finalTrim == "" {
			return "{}", "{}", true
		}
		b := []byte(finalTrim)
		if Kind(b) != '{' || !json.Valid(b) {
			return "", "", false
		}
		return finalTrim, finalTrim, true
	}
	if finalTrim == "" {
		b := []byte(strings.TrimSpace(sent))
		if Kind(b) != '{' || !json.Valid(b) {
			return "", "", false
		}
		return sent, "", true
	}
	bFinal := []byte(finalTrim)
	if Kind(bFinal) != '{' || !json.Valid(bFinal) {
		return "", "", false
	}
	if strings.HasPrefix(finalTrim, sent) {
		return finalTrim, finalTrim[len(sent):], true
	}
	sentTrim := strings.TrimSpace(sent)
	if strings.HasPrefix(finalTrim, sentTrim) {
		return finalTrim, finalTrim[len(sentTrim):], true
	}
	if json.Valid([]byte(sentTrim)) && equalJSON([]byte(sentTrim), bFinal) {
		return finalTrim, "", true
	}
	return "", "", false
}

func equalJSON(a, b []byte) bool {
	var va, vb any
	if json.Unmarshal(a, &va) != nil || json.Unmarshal(b, &vb) != nil {
		return false
	}
	return reflect.DeepEqual(va, vb)
}

// UnwrapCustomInput extracts the raw string value from a {"input": "..."}
// JSON wrapper object used to emulate custom (freeform) tools over formats
// whose tool arguments must be JSON objects.
func UnwrapCustomInput(raw json.RawMessage) (string, bool) {
	s := strings.TrimSpace(string(raw))
	if s == "" {
		return "", false
	}
	b := []byte(s)
	if Kind(b) != '{' || !json.Valid(b) {
		return "", false
	}
	var obj struct {
		Input *string `json:"input"`
	}
	if json.Unmarshal(b, &obj) != nil || obj.Input == nil {
		return "", false
	}
	return *obj.Input, true
}

// CustomInputDecoder incrementally unwraps streamed JSON fragments of a
// {"input": "..."} object into raw text deltas, holding back incomplete escape
// sequences and UTF-16 surrogate pairs until subsequent fragments arrive.
type CustomInputDecoder struct {
	buf   strings.Builder
	pos   int
	state int // 0=seeking "input":", 1=inside string, 2=closed string
}

// Len returns the number of raw JSON bytes accumulated so far.
func (d *CustomInputDecoder) Len() int {
	return d.buf.Len()
}

// Write appends a JSON fragment and returns any newly decoded raw input text.
func (d *CustomInputDecoder) Write(chunk string) string {
	if chunk != "" {
		d.buf.WriteString(chunk)
	}
	return d.drain()
}

// Final validates the full accumulated JSON wrapper and returns the final
// unwrapped input encoded as a JSON string (or an invalid-wrapper "{}" sentinel
// when the accumulated arguments are not a valid {"input": string} object).
func (d *CustomInputDecoder) Final() (json.RawMessage, bool) {
	if s, ok := UnwrapCustomInput(json.RawMessage(d.buf.String())); ok {
		raw, _ := Marshal(s)
		return raw, true
	}
	return json.RawMessage(`{}`), false
}

func (d *CustomInputDecoder) drain() string {
	s := d.buf.String()
	if d.state == 2 {
		return ""
	}
	if d.state == 0 {
		idx := strings.Index(s[d.pos:], `"input"`)
		if idx < 0 {
			return ""
		}
		i := d.pos + idx + len(`"input"`)
		for i < len(s) && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r') {
			i++
		}
		if i >= len(s) || s[i] != ':' {
			return ""
		}
		i++
		for i < len(s) && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r') {
			i++
		}
		if i >= len(s) || s[i] != '"' {
			return ""
		}
		d.pos = i + 1
		d.state = 1
	}

	var out strings.Builder
	i := d.pos
	for i < len(s) {
		ch := s[i]
		if ch == '"' {
			d.pos = i + 1
			d.state = 2
			break
		}
		if ch == '\\' {
			if i+1 >= len(s) {
				break // incomplete escape
			}
			esc := s[i+1]
			if esc == 'u' {
				if i+6 > len(s) {
					break // incomplete \uXXXX
				}
				// Detect high surrogate hex range D800-DBFF requiring a following \uXXXX low surrogate.
				hexStr := strings.ToUpper(s[i+2 : i+6])
				if len(hexStr) == 4 && hexStr[0] == 'D' && hexStr[1] >= '8' && hexStr[1] <= 'B' {
					if i+12 > len(s) {
						break // wait for low surrogate pair
					}
					if s[i+6] == '\\' && s[i+7] == 'u' {
						var dec string
						if json.Unmarshal([]byte(`"`+s[i:i+12]+`"`), &dec) == nil {
							out.WriteString(dec)
							i += 12
							d.pos = i
							continue
						}
					}
				}
				var dec string
				if json.Unmarshal([]byte(`"`+s[i:i+6]+`"`), &dec) == nil {
					out.WriteString(dec)
				}
				i += 6
				d.pos = i
				continue
			}
			var dec string
			if json.Unmarshal([]byte(`"`+s[i:i+2]+`"`), &dec) == nil {
				out.WriteString(dec)
			}
			i += 2
			d.pos = i
			continue
		}
		out.WriteByte(ch)
		i++
		d.pos = i
	}
	return out.String()
}
