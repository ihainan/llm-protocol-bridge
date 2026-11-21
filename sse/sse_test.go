package sse

import (
	"io"
	"strings"
	"testing"
)

func readAll(t *testing.T, in string) []Event {
	t.Helper()
	r := NewReader(strings.NewReader(in))
	var out []Event
	for {
		ev, err := r.Next()
		if err == io.EOF {
			return out
		}
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, ev)
	}
}

func TestReader(t *testing.T) {
	in := "event: a\r\ndata: 1\r\ndata: 2\r\n\r\n: ping\n\ndata: {\"x\":1}\n\n{\"error\":{\"message\":\"bad\"}}\ndata: tail"
	evs := readAll(t, in)
	if len(evs) != 4 {
		t.Fatalf("got %d events: %+v", len(evs), evs)
	}
	if evs[0].Name != "a" || string(evs[0].Data) != "1\n2" {
		t.Errorf("event 0: %+v", evs[0])
	}
	if string(evs[1].Data) != `{"x":1}` {
		t.Errorf("event 1: %q", evs[1].Data)
	}
	if string(evs[2].Data) != `{"error":{"message":"bad"}}` {
		t.Errorf("bare line: %q", evs[2].Data)
	}
	if string(evs[3].Data) != "tail" {
		t.Errorf("unterminated tail: %q", evs[3].Data)
	}
}

func TestBareLineAfterPendingEvent(t *testing.T) {
	evs := readAll(t, "data: first\n{\"error\":1}\ndata: second\n\n")
	if len(evs) != 3 || string(evs[0].Data) != "first" || string(evs[1].Data) != `{"error":1}` || string(evs[2].Data) != "second" {
		t.Fatalf("%+v", evs)
	}
}

func TestDone(t *testing.T) {
	if !IsDone([]byte(" [DONE] ")) || IsDone([]byte("[DONE]x")) {
		t.Fatal("IsDone")
	}
}

func TestAppendEvent(t *testing.T) {
	got := string(AppendEvent(nil, "message_start", []byte("a\nb")))
	if got != "event: message_start\ndata: a\ndata: b\n\n" {
		t.Fatalf("%q", got)
	}
	if got := string(AppendComment(nil, "keepalive", false)); got != ": keepalive\n" {
		t.Fatalf("%q", got)
	}
}

func TestArrayReader(t *testing.T) {
	r := NewArrayReader(strings.NewReader("[{\"a\":1}\n,\r\n{\"b\":[2]}]"))
	var got []string
	for {
		ev, err := r.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, string(ev.Data))
	}
	if strings.Join(got, "|") != `{"a":1}|{"b":[2]}` {
		t.Fatalf("%v", got)
	}
}

func TestLongLine(t *testing.T) {
	r := NewReader(strings.NewReader("data: " + strings.Repeat("x", 200) + "\n\n"))
	r.SetMaxLine(100)
	if _, err := r.Next(); err != ErrLineTooLong {
		t.Fatalf("want ErrLineTooLong, got %v", err)
	}
}
