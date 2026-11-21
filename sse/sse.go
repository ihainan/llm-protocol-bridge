// Package sse reads and writes Server-Sent Events as LLM APIs use them, including
// the irregular shapes providers send (bare JSON lines without "data:", CRLF line
// endings) and Gemini's JSON-array streams.
package sse

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
)

// Event is one dispatched SSE event. Data has its "data:" lines joined by "\n".
type Event struct {
	Name string
	Data []byte
}

// ErrLineTooLong is returned when a line exceeds the reader's limit.
var ErrLineTooLong = errors.New("sse: line too long")

// DefaultMaxLine is the default limit for one line.
const DefaultMaxLine = 32 << 20

// Reader splits a byte stream into events.
type Reader struct {
	r       *bufio.Reader
	maxLine int
	name    string
	data    []byte
	hasData bool
	line    []byte
	queued  *Event
}

// NewReader returns a Reader over r.
func NewReader(r io.Reader) *Reader {
	return &Reader{r: bufio.NewReaderSize(r, 64<<10), maxLine: DefaultMaxLine}
}

// SetMaxLine changes the line length limit.
func (r *Reader) SetMaxLine(n int) { r.maxLine = n }

func (r *Reader) readLine() ([]byte, error) {
	r.line = r.line[:0]
	for {
		chunk, err := r.r.ReadSlice('\n')
		r.line = append(r.line, chunk...)
		if len(r.line) > r.maxLine {
			return nil, ErrLineTooLong
		}
		if err == bufio.ErrBufferFull {
			continue
		}
		if err != nil {
			if err == io.EOF && len(r.line) > 0 {
				return trimEOL(r.line), nil
			}
			return nil, err
		}
		return trimEOL(r.line), nil
	}
}

func trimEOL(b []byte) []byte {
	if n := len(b); n > 0 && b[n-1] == '\n' {
		b = b[:n-1]
	}
	if n := len(b); n > 0 && b[n-1] == '\r' {
		b = b[:n-1]
	}
	return b
}

func (r *Reader) dispatch() (Event, bool) {
	if !r.hasData && r.name == "" {
		return Event{}, false
	}
	ev := Event{Name: r.name, Data: append([]byte(nil), r.data...)}
	r.name, r.data, r.hasData = "", r.data[:0], false
	return ev, true
}

// Next returns the next event. At the end of the stream it returns io.EOF after
// dispatching any pending event. The returned Data is owned by the caller.
func (r *Reader) Next() (Event, error) {
	if q := r.queued; q != nil {
		r.queued = nil
		return *q, nil
	}
	for {
		line, err := r.readLine()
		if err != nil {
			if err == io.EOF {
				if ev, ok := r.dispatch(); ok {
					return ev, nil
				}
			}
			return Event{}, err
		}
		if len(line) == 0 {
			if ev, ok := r.dispatch(); ok {
				return ev, nil
			}
			continue
		}
		switch line[0] {
		case ':':
			continue // comment / keepalive
		case '{', '[':
			// A bare JSON line (some providers write errors this way). It is an
			// event of its own.
			bare := Event{Data: append([]byte(nil), line...)}
			if ev, ok := r.dispatch(); ok {
				r.queued = &bare
				return ev, nil
			}
			return bare, nil
		}
		field, value := line, []byte(nil)
		if i := bytes.IndexByte(line, ':'); i >= 0 {
			field, value = line[:i], line[i+1:]
			if len(value) > 0 && value[0] == ' ' {
				value = value[1:]
			}
		}
		switch string(field) {
		case "event":
			r.name = string(value)
		case "data":
			if r.hasData {
				r.data = append(r.data, '\n')
			}
			r.data = append(r.data, value...)
			r.hasData = true
		}
	}
}

// IsDone reports whether data is the "[DONE]" end marker.
func IsDone(data []byte) bool { return string(bytes.TrimSpace(data)) == "[DONE]" }

// AppendEvent writes an event with a name. Data containing newlines is split into
// several data lines.
func AppendEvent(dst []byte, name string, data []byte) []byte {
	if name != "" {
		dst = append(dst, "event: "...)
		dst = append(dst, name...)
		dst = append(dst, '\n')
	}
	for {
		i := bytes.IndexByte(data, '\n')
		dst = append(dst, "data: "...)
		if i < 0 {
			dst = append(dst, data...)
			break
		}
		dst = append(dst, data[:i]...)
		dst = append(dst, '\n')
		data = data[i+1:]
	}
	return append(dst, '\n', '\n')
}

// AppendData writes an unnamed event.
func AppendData(dst []byte, data []byte) []byte { return AppendEvent(dst, "", data) }

// AppendComment writes a comment line. With blank, a blank line follows it.
func AppendComment(dst []byte, text string, blank bool) []byte {
	dst = append(dst, ':')
	if text != "" {
		dst = append(dst, ' ')
		dst = append(dst, text...)
	}
	dst = append(dst, '\n')
	if blank {
		dst = append(dst, '\n')
	}
	return dst
}

// ArrayReader reads the elements of a JSON-array stream ([{...},{...}]), the shape
// of Gemini's streamGenerateContent without alt=sse.
type ArrayReader struct {
	d       *json.Decoder
	started bool
	done    bool
}

// NewArrayReader returns an ArrayReader over r.
func NewArrayReader(r io.Reader) *ArrayReader { return &ArrayReader{d: json.NewDecoder(r)} }

// Next returns the next element as an Event with empty Name; io.EOF after the
// closing bracket.
func (a *ArrayReader) Next() (Event, error) {
	if a.done {
		return Event{}, io.EOF
	}
	if !a.started {
		t, err := a.d.Token()
		if err != nil {
			return Event{}, err
		}
		if d, ok := t.(json.Delim); !ok || d != '[' {
			// a single object: return it whole
			return Event{}, errors.New("sse: JSON stream does not start with an array")
		}
		a.started = true
	}
	if !a.d.More() {
		a.done = true
		if _, err := a.d.Token(); err != nil && err != io.EOF {
			return Event{}, err
		}
		return Event{}, io.EOF
	}
	var raw json.RawMessage
	if err := a.d.Decode(&raw); err != nil {
		return Event{}, err
	}
	return Event{Data: raw}, nil
}
