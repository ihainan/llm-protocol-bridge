package gemini_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/gemini"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// Gemini CLI reads candidates.length and content.parts.length on every chunk:
// a null in either place crashes it at the end of a converted stream. Empty
// lists are left out instead, as the Gemini API does.

func encodeStreamRaw(t *testing.T, evs []ir.Event) []byte {
	t.Helper()
	enc := gemini.NewStreamEncoder(gemini.EncodeOptions{AltSSE: true, Model: "m", Env: &ir.Env{Allow: ir.DefaultAllowed}})
	var out []byte
	for _, ev := range evs {
		var err error
		if out, err = enc.Encode(out, ev); err != nil {
			t.Fatalf("Encode: %v", err)
		}
	}
	if bytes.Contains(out, []byte("null")) {
		t.Fatalf("stream carries a null:\n%s", out)
	}
	return out
}

func encodeStream(t *testing.T, evs []ir.Event) []map[string]json.RawMessage {
	t.Helper()
	out := encodeStreamRaw(t, evs)
	r := sse.NewReader(bytes.NewReader(out))
	var chunks []map[string]json.RawMessage
	for {
		ev, err := r.Next()
		if err != nil {
			break
		}
		if len(ev.Data) == 0 {
			continue
		}
		var m map[string]json.RawMessage
		if err := json.Unmarshal(ev.Data, &m); err != nil {
			t.Fatalf("chunk is not JSON: %v\n%s", err, ev.Data)
		}
		chunks = append(chunks, m)
	}
	return chunks
}

func TestStreamEncode_NoNullPartsOrCandidates(t *testing.T) {
	chunks := encodeStream(t, []ir.Event{
		{Type: ir.EvStart, Model: "m"},
		{Type: ir.EvText, Text: "hello"},
		{Type: ir.EvStop, Stop: &ir.Stop{Reason: ir.StopEndTurn}},
		{Type: ir.EvUsage, Usage: &ir.Usage{InputTokens: 3, OutputTokens: 1, Reported: ir.UInput | ir.UOutput}},
	})
	if len(chunks) != 3 {
		t.Fatalf("want 3 chunks (text, stop, usage), got %d", len(chunks))
	}
	// the stop chunk: a candidate with finishReason and no content
	var stop []map[string]json.RawMessage
	if err := json.Unmarshal(chunks[1]["candidates"], &stop); err != nil || len(stop) != 1 {
		t.Fatalf("stop chunk candidates: %s (%v)", chunks[1]["candidates"], err)
	}
	if _, ok := stop[0]["content"]; ok {
		t.Errorf("stop candidate has content: %s", stop[0]["content"])
	}
	if string(stop[0]["finishReason"]) != `"STOP"` {
		t.Errorf("finishReason: %s", stop[0]["finishReason"])
	}
	// the usage chunk: no candidates key at all
	if _, ok := chunks[2]["candidates"]; ok {
		t.Errorf("usage chunk has candidates: %s", chunks[2]["candidates"])
	}
	if _, ok := chunks[2]["usageMetadata"]; !ok {
		t.Error("usage chunk lost usageMetadata")
	}
}

func TestBuildResponse_NoNullPartsOrCandidates(t *testing.T) {
	env := &ir.Env{Allow: ir.DefaultAllowed}
	// an output without parts (for example a max-tokens stop while thinking).
	// The Gemini API itself answers this case with a content that carries only
	// the role, so the non-streamed shape keeps it; only the null goes away.
	body, err := gemini.BuildResponse(&ir.Response{
		Model:   "m",
		Outputs: []ir.Output{{Index: 0, Stop: ir.Stop{Reason: ir.StopMaxTokens}}},
	}, gemini.EncodeOptions{Env: env})
	if err != nil {
		t.Fatalf("BuildResponse: %v", err)
	}
	if strings.Contains(string(body), "null") {
		t.Errorf("response carries a null: %s", body)
	}
	if !strings.Contains(string(body), `"content":{"role":"model"}`) {
		t.Errorf("want a content with role and no parts: %s", body)
	}
	// no outputs at all
	body, err = gemini.BuildResponse(&ir.Response{Model: "m"}, gemini.EncodeOptions{Env: env})
	if err != nil {
		t.Fatalf("BuildResponse: %v", err)
	}
	if strings.Contains(string(body), "null") || strings.Contains(string(body), "candidates") {
		t.Errorf("empty response: %s", body)
	}
}

// Requests keep their shape: a content always has parts.
func TestBuildRequest_PartsStillSent(t *testing.T) {
	req := &ir.Request{Model: "m", Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("hi")}}}}
	built, err := gemini.BuildRequest(req, gemini.BuildOptions{Env: &ir.Env{Allow: ir.DefaultAllowed}})
	if err != nil {
		t.Fatalf("BuildRequest: %v", err)
	}
	if !strings.Contains(string(built.Body), `"parts":[{"text":"hi"}]`) {
		t.Errorf("request lost its parts: %s", built.Body)
	}
}

var streamEvents = []ir.Event{
	{Type: ir.EvStart, Model: "m"},
	{Type: ir.EvText, Text: "hello"},
	{Type: ir.EvStop, Stop: &ir.Stop{Reason: ir.StopEndTurn}},
	{Type: ir.EvUsage, Usage: &ir.Usage{InputTokens: 3, OutputTokens: 1, Reported: ir.UInput | ir.UOutput}},
}

// The shapes the encoder now writes (no content on the stop candidate, no
// candidates on the usage chunk) decode back to the same text, stop and usage.
func TestStreamEncode_RoundTripsThroughDecoder(t *testing.T) {
	raw := encodeStreamRaw(t, streamEvents)
	dec := gemini.NewStreamDecoder(gemini.DecodeOptions{Env: &ir.Env{Allow: ir.DefaultAllowed}})
	r := sse.NewReader(bytes.NewReader(raw))
	var evs []ir.Event
	for {
		ev, err := r.Next()
		if err != nil {
			break
		}
		if len(ev.Data) == 0 {
			continue
		}
		if evs, err = dec.Decode(evs, ev); err != nil {
			t.Fatalf("Decode: %v\n%s", err, ev.Data)
		}
	}
	var text string
	var stop *ir.Stop
	var usage *ir.Usage
	for _, ev := range evs {
		switch ev.Type {
		case ir.EvText:
			text += ev.Text
		case ir.EvStop:
			stop = ev.Stop
		case ir.EvUsage:
			usage = ev.Usage
		}
	}
	if text != "hello" || stop == nil || stop.Reason != ir.StopEndTurn {
		t.Errorf("text %q stop %+v", text, stop)
	}
	if usage == nil || usage.InputTokens != 3 || usage.OutputTokens != 1 {
		t.Errorf("usage %+v", usage)
	}
}

// A usage chunk with no candidates key at all (what the encoder writes).
func TestStreamDecode_UsageChunkWithoutCandidates(t *testing.T) {
	dec := gemini.NewStreamDecoder(gemini.DecodeOptions{Env: &ir.Env{Allow: ir.DefaultAllowed}})
	raw := `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"index":0,"finishReason":"STOP"}]}` + "\n\n" +
		`data: {"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2,"totalTokenCount":12}}` + "\n\n"
	r := sse.NewReader(bytes.NewReader([]byte(raw)))
	var evs []ir.Event
	for {
		ev, err := r.Next()
		if err != nil {
			break
		}
		if len(ev.Data) == 0 {
			continue
		}
		if evs, err = dec.Decode(evs, ev); err != nil {
			t.Fatalf("Decode: %v", err)
		}
	}
	var usage *ir.Usage
	for _, ev := range evs {
		if ev.Type == ir.EvUsage {
			usage = ev.Usage
		}
	}
	if usage == nil || usage.InputTokens != 10 || usage.OutputTokens != 2 {
		t.Errorf("usage %+v", usage)
	}
}

// Instructions that are all dropped leave systemInstruction out entirely.
func TestBuildRequest_NoEmptySystemInstruction(t *testing.T) {
	req := &ir.Request{
		Model:        "m",
		Instructions: []ir.Part{{Type: ir.PartReasoning}},
		Messages:     []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("hi")}}},
	}
	built, err := gemini.BuildRequest(req, gemini.BuildOptions{Env: &ir.Env{Allow: ir.DefaultAllowed}})
	if err != nil {
		t.Fatalf("BuildRequest: %v", err)
	}
	if strings.Contains(string(built.Body), "systemInstruction") {
		t.Errorf("empty systemInstruction sent: %s", built.Body)
	}
}
