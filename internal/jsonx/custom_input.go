package jsonx

import (
	"encoding/json"
	"strings"
)

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
