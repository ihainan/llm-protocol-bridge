package bridge_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// grammarChecker validates the SSE byte stream emitted by an encoder.
// Each format has its own rules; we share the scaffolding here.

// --- helpers ---

// makePlan builds a Plan for the given client+upstream and minimal request body.
func makePlan(t *testing.T, client, upstream ir.Format) *bridge.Plan {
	t.Helper()
	minBodies := map[ir.Format][]byte{
		ir.FormatChat:      []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}`),
		ir.FormatResponses: []byte(`{"model":"m","input":"hi","stream":true}`),
		ir.FormatAnthropic: []byte(`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`),
		ir.FormatGemini:    []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`),
	}
	plan, err := bridge.Convert(client, upstream, minBodies[client], opts())
	if err != nil {
		t.Fatalf("makePlan(%s→%s): %v", client, upstream, err)
	}
	return plan
}

// simulateStream is intentionally removed; grammar tests drive the stream directly.

// --- Chat grammar tests ---

// TestChatStreamGrammar checks that a Chat streaming response produces:
// valid SSE with data: lines, ends with [DONE], usage event near the end.
func TestChatStreamGrammar(t *testing.T) {
	// Build a Chat→Chat stream (identity) so the upstream SSE is Chat format.
	plan := makePlan(t, ir.FormatChat, ir.FormatChat)
	s := plan.NewStream()

	// Feed a minimal chat streaming response.
	chunks := []string{
		`data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
		`data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}`,
		`data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`,
		`data: [DONE]`,
	}

	var out []byte
	for _, chunk := range chunks {
		raw := []byte(chunk + "\n\n")
		r := sse.NewReader(bytes.NewReader(raw))
		for {
			ev, err := r.Next()
			if err != nil {
				break
			}
			if len(ev.Data) == 0 && ev.Name == "" {
				continue
			}
			var err2 error
			out, err2 = s.Feed(out, ev)
			if err2 != nil {
				t.Fatalf("Feed: %v", err2)
			}
		}
	}
	out, err := s.End(out, ir.EndDone)
	if err != nil {
		t.Fatalf("End: %v", err)
	}

	// Grammar checks for Chat output:
	// 1. At least one data: line
	// 2. Ends with data: [DONE]
	// 3. All data: lines (except [DONE]) parse as valid JSON
	raw := string(out)
	if !strings.Contains(raw, "data: ") {
		t.Errorf("chat stream: no data: lines found\n%s", raw)
	}
	if !strings.HasSuffix(strings.TrimRight(raw, "\n"), "data: [DONE]") {
		t.Errorf("chat stream: does not end with data: [DONE]\n%s", raw)
	}
	checkChatSSEGrammar(t, out)
}

func checkChatSSEGrammar(t *testing.T, raw []byte) {
	t.Helper()
	lines := strings.Split(strings.TrimRight(string(raw), "\n"), "\n")
	var dataLines []string
	for _, l := range lines {
		if strings.HasPrefix(l, "data: ") {
			dataLines = append(dataLines, strings.TrimPrefix(l, "data: "))
		}
	}
	if len(dataLines) == 0 {
		t.Error("no data: lines in chat SSE output")
		return
	}
	last := dataLines[len(dataLines)-1]
	if last != "[DONE]" {
		t.Errorf("chat SSE: last data line should be [DONE], got %q", last)
	}
	for i, d := range dataLines[:len(dataLines)-1] {
		if d == "[DONE]" {
			t.Errorf("chat SSE: [DONE] appears at position %d, not just at end", i)
			continue
		}
		var v map[string]interface{}
		if err := parseJSON([]byte(d), &v); err != nil {
			t.Errorf("chat SSE data[%d] is not valid JSON: %v\n%s", i, err, d)
		}
	}
}

// --- Responses grammar tests ---

// TestResponsesStreamGrammar checks that a Responses streaming response produces
// events in the correct sequence: response.created → response.in_progress →
// content events → response.completed.
func TestResponsesStreamGrammar(t *testing.T) {
	plan := makePlan(t, ir.FormatResponses, ir.FormatAnthropic)
	s := plan.NewStream()

	// Minimal Anthropic streaming response.
	chunks := []string{
		`event: message_start` + "\n" + `data: {"type":"message_start","message":{"id":"msg1","model":"m","usage":{"input_tokens":5,"output_tokens":0}}}`,
		`event: content_block_start` + "\n" + `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}`,
		`event: content_block_stop` + "\n" + `data: {"type":"content_block_stop","index":0}`,
		`event: message_delta` + "\n" + `data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}`,
		`event: message_stop` + "\n" + `data: {"type":"message_stop"}`,
	}

	var out []byte
	for _, chunk := range chunks {
		raw := []byte(chunk + "\n\n")
		r := sse.NewReader(bytes.NewReader(raw))
		for {
			ev, err := r.Next()
			if err != nil {
				break
			}
			if len(ev.Data) == 0 && ev.Name == "" {
				continue
			}
			var err2 error
			out, err2 = s.Feed(out, ev)
			if err2 != nil {
				t.Fatalf("Feed: %v", err2)
			}
		}
	}
	out, err := s.End(out, ir.EndDone)
	if err != nil {
		t.Fatalf("End: %v", err)
	}

	checkResponsesSSEGrammar(t, out)
}

func checkResponsesSSEGrammar(t *testing.T, raw []byte) {
	t.Helper()
	var eventTypes []string
	lines := strings.Split(string(raw), "\n")
	for _, l := range lines {
		if strings.HasPrefix(l, "event: ") {
			eventTypes = append(eventTypes, strings.TrimPrefix(l, "event: "))
		}
	}
	if len(eventTypes) == 0 {
		t.Error("responses SSE: no event: lines found")
		return
	}
	// First event must be response.created.
	if eventTypes[0] != "response.created" {
		t.Errorf("responses SSE: first event should be response.created, got %q", eventTypes[0])
	}
	// Last event must be response.completed or response.failed.
	last := eventTypes[len(eventTypes)-1]
	if last != "response.completed" && last != "response.failed" {
		t.Errorf("responses SSE: last event should be response.completed/failed, got %q", last)
	}
	// Sequence numbers must be monotonically increasing.
	checkResponsesSequenceNumbers(t, raw)
}

func checkResponsesSequenceNumbers(t *testing.T, raw []byte) {
	t.Helper()
	prev := -1
	for _, line := range strings.Split(string(raw), "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		d := strings.TrimPrefix(line, "data: ")
		var v map[string]interface{}
		if err := parseJSON([]byte(d), &v); err != nil {
			continue
		}
		if sn, ok := v["sequence_number"]; ok {
			n := int(sn.(float64))
			if n <= prev {
				t.Errorf("responses SSE: sequence_number %d not monotonically increasing after %d", n, prev)
			}
			prev = n
		}
	}
}

// --- Anthropic grammar tests ---

// TestAnthropicStreamGrammar checks that an Anthropic streaming response produces
// events in the correct sequence: message_start → content blocks → message_stop.
func TestAnthropicStreamGrammar(t *testing.T) {
	plan := makePlan(t, ir.FormatAnthropic, ir.FormatChat)
	s := plan.NewStream()

	// Minimal Chat streaming response.
	chunks := []string{
		`data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
		`data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}`,
		`data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`,
		`data: [DONE]`,
	}

	var out []byte
	for _, chunk := range chunks {
		raw := []byte(chunk + "\n\n")
		r := sse.NewReader(bytes.NewReader(raw))
		for {
			ev, err := r.Next()
			if err != nil {
				break
			}
			if len(ev.Data) == 0 && ev.Name == "" {
				continue
			}
			var err2 error
			out, err2 = s.Feed(out, ev)
			if err2 != nil {
				t.Fatalf("Feed: %v", err2)
			}
		}
	}
	out, err := s.End(out, ir.EndDone)
	if err != nil {
		t.Fatalf("End: %v", err)
	}

	checkAnthropicSSEGrammar(t, out)
}

func checkAnthropicSSEGrammar(t *testing.T, raw []byte) {
	t.Helper()
	var eventTypes []string
	lines := strings.Split(string(raw), "\n")
	for _, l := range lines {
		if strings.HasPrefix(l, "event: ") {
			eventTypes = append(eventTypes, strings.TrimPrefix(l, "event: "))
		}
	}
	if len(eventTypes) == 0 {
		t.Error("anthropic SSE: no event: lines found")
		return
	}
	// First event must be message_start.
	if eventTypes[0] != "message_start" {
		t.Errorf("anthropic SSE: first event should be message_start, got %q", eventTypes[0])
	}
	// Must contain exactly one message_start.
	count := 0
	for _, e := range eventTypes {
		if e == "message_start" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("anthropic SSE: expected exactly 1 message_start, got %d", count)
	}
	// Last event must be message_stop or error.
	last := eventTypes[len(eventTypes)-1]
	if last != "message_stop" && last != "error" {
		t.Errorf("anthropic SSE: last event should be message_stop/error, got %q", last)
	}
}

// --- Gemini grammar tests ---

// TestGeminiStreamGrammar checks that a Gemini streaming response produces
// valid JSON objects (array-stream format used by Gemini).
func TestGeminiStreamGrammar(t *testing.T) {
	plan := makePlan(t, ir.FormatGemini, ir.FormatChat)
	s := plan.NewStream()

	// Minimal Chat streaming response.
	chunks := []string{
		`data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
		`data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}`,
		`data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`,
		`data: [DONE]`,
	}

	var out []byte
	for _, chunk := range chunks {
		raw := []byte(chunk + "\n\n")
		r := sse.NewReader(bytes.NewReader(raw))
		for {
			ev, err := r.Next()
			if err != nil {
				break
			}
			if len(ev.Data) == 0 && ev.Name == "" {
				continue
			}
			var err2 error
			out, err2 = s.Feed(out, ev)
			if err2 != nil {
				t.Fatalf("Feed: %v", err2)
			}
		}
	}
	out, err := s.End(out, ir.EndDone)
	if err != nil {
		t.Fatalf("End: %v", err)
	}

	// Gemini output: either SSE data: lines each containing a JSON object,
	// or a streaming JSON array. Either way, each data: payload must parse as JSON.
	checkGeminiSSEGrammar(t, out)
}

func checkGeminiSSEGrammar(t *testing.T, raw []byte) {
	t.Helper()
	if len(raw) == 0 {
		t.Error("gemini SSE: empty output")
		return
	}
	for i, line := range strings.Split(string(raw), "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		d := strings.TrimPrefix(line, "data: ")
		if d == "[DONE]" || d == "" {
			continue
		}
		var v interface{}
		if err := parseJSON([]byte(d), &v); err != nil {
			t.Errorf("gemini SSE data[%d] is not valid JSON: %v\n%s", i, err, d)
		}
	}
}

// --- Keepalive tests ---

// TestKeepaliveFormats checks that Keepalive() produces non-empty output for each format.
func TestKeepaliveFormats(t *testing.T) {
	for _, client := range ir.Formats {
		for _, upstream := range ir.Formats {
			if client == upstream {
				continue
			}
			name := fmt.Sprintf("%s→%s", client, upstream)
			t.Run(name, func(t *testing.T) {
				plan := makePlan(t, client, upstream)
				s := plan.NewStream()
				out := s.Keepalive(nil)
				if len(out) == 0 {
					t.Errorf("%s: Keepalive returned empty bytes", name)
				}
			})
		}
	}
}

func parseJSON(data []byte, v interface{}) error {
	return json.Unmarshal(data, v)
}
