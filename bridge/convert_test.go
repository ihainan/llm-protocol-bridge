package bridge_test

import (
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/internal/testkit"
)

var updateGolden = flag.Bool("update.golden", false, "regenerate golden fixture files from current output")

const testdataDir = "testdata"

// opts returns default bridge options (all default losses allowed, deterministic IDs/clock).
func opts() bridge.Options {
	return bridge.Options{Allow: ir.DefaultAllowed}
}

// goldenCheck compares got against the golden file; when -update.golden is set it writes got instead.
func goldenCheck(t *testing.T, label, file string, got []byte) {
	t.Helper()
	path := filepath.Join(testdataDir, file)
	if *updateGolden {
		var v any
		if err := json.Unmarshal(got, &v); err != nil {
			t.Fatalf("%s: output not valid JSON: %v", label, err)
		}
		pretty, _ := json.MarshalIndent(v, "", "  ")
		if err := os.WriteFile(path, pretty, 0o644); err != nil {
			t.Fatalf("%s: write golden %s: %v", label, path, err)
		}
		t.Logf("%s: wrote golden %s", label, file)
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%s: missing golden file %s (run with -update.golden to create it)", label, path)
	}
	testkit.JSONEqual(t, label, want, got)
}

// convertReq parses body as clientFmt and builds it as upstreamFmt.
// Returns the upstream JSON body.
func convertReq(t *testing.T, clientFmt, upstreamFmt ir.Format, body []byte) []byte {
	t.Helper()
	plan, err := bridge.Convert(clientFmt, upstreamFmt, body, opts())
	if err != nil {
		t.Fatalf("Convert(%s→%s): %v", clientFmt, upstreamFmt, err)
	}
	return plan.Body
}

// TestConvert_ChatToAnthropic parses a Chat request and rebuilds it as Anthropic.
func TestConvert_ChatToAnthropic(t *testing.T) {
	src := testkit.ReadFile(t, filepath.Join(testdataDir, "chat.json"))
	got := convertReq(t, ir.FormatChat, ir.FormatAnthropic, src)
	goldenCheck(t, "chat→anthropic", "chat-to-anthropic.json", got)
}

// TestConvert_ResponsesToChat parses a Responses request and rebuilds it as Chat.
func TestConvert_ResponsesToChat(t *testing.T) {
	src := testkit.ReadFile(t, filepath.Join(testdataDir, "responses.json"))
	got := convertReq(t, ir.FormatResponses, ir.FormatChat, src)
	goldenCheck(t, "responses→chat", "responses-to-chat.json", got)
}

// TestConvert_ResponsesToAnthropic parses a Responses request and rebuilds it as Anthropic.
func TestConvert_ResponsesToAnthropic(t *testing.T) {
	src := testkit.ReadFile(t, filepath.Join(testdataDir, "responses.json"))
	got := convertReq(t, ir.FormatResponses, ir.FormatAnthropic, src)
	goldenCheck(t, "responses→anthropic", "responses-to-anthropic.json", got)
}

// TestConvert_AnthropicToChat parses an Anthropic request and rebuilds it as Chat.
func TestConvert_AnthropicToChat(t *testing.T) {
	src := testkit.ReadFile(t, filepath.Join(testdataDir, "anthropic.json"))
	got := convertReq(t, ir.FormatAnthropic, ir.FormatChat, src)
	goldenCheck(t, "anthropic→chat", "anthropic-to-chat.json", got)
}

// TestConvert_AnthropicImageToResponses parses an Anthropic image request and rebuilds it as Responses.
func TestConvert_AnthropicImageToResponses(t *testing.T) {
	src := testkit.ReadFile(t, filepath.Join(testdataDir, "anthropic-image.json"))
	got := convertReq(t, ir.FormatAnthropic, ir.FormatResponses, src)
	goldenCheck(t, "anthropic-image→responses", "anthropic-to-responses.json", got)
}

// TestConvert_ResponsesStringInput parses a Responses request with a plain string input.
func TestConvert_ResponsesStringInput(t *testing.T) {
	src := testkit.ReadFile(t, filepath.Join(testdataDir, "responses-string.json"))
	// Build as chat to exercise string input handling; just expect no error and
	// that the output parses as valid JSON with messages.
	got := convertReq(t, ir.FormatResponses, ir.FormatChat, src)
	var v map[string]any
	if err := json.Unmarshal(got, &v); err != nil {
		t.Fatalf("responses-string→chat: output is not valid JSON: %v\n%s", err, got)
	}
	if _, ok := v["messages"]; !ok {
		t.Errorf("responses-string→chat: output has no 'messages' field\n%s", got)
	}
}

// TestConvert_AnthropicThinking parses an Anthropic request with thinking enabled.
func TestConvert_AnthropicThinking(t *testing.T) {
	src := testkit.ReadFile(t, filepath.Join(testdataDir, "anthropic-thinking.json"))
	// Build as chat; expect no parse error and a valid JSON body.
	got := convertReq(t, ir.FormatAnthropic, ir.FormatChat, src)
	var v map[string]any
	if err := json.Unmarshal(got, &v); err != nil {
		t.Fatalf("anthropic-thinking→chat: output is not valid JSON: %v\n%s", err, got)
	}
}

// TestConvert_ParseErrors checks that obviously invalid bodies produce errors.
func TestConvert_ParseErrors(t *testing.T) {
	for _, f := range []ir.Format{ir.FormatChat, ir.FormatResponses, ir.FormatAnthropic} {
		f := f
		t.Run(string(f), func(t *testing.T) {
			_, err := bridge.Convert(f, ir.FormatChat, []byte(`not json at all`), opts())
			if err == nil {
				t.Errorf("%s: expected parse error on garbage input, got nil", f)
			}
		})
	}
}

// TestConvert_UnknownFormat checks that an unknown format returns an error.
func TestConvert_UnknownFormat(t *testing.T) {
	_, err := bridge.Convert("bogus", ir.FormatChat, []byte(`{}`), opts())
	if err == nil {
		t.Error("expected error for unknown client format, got nil")
	}
	_, err = bridge.Convert(ir.FormatChat, "bogus", []byte(`{}`), opts())
	if err == nil {
		t.Error("expected error for unknown upstream format, got nil")
	}
}

// TestConvert_AllPairs exercises all 12 cross-format conversions (not identity)
// on the simplest possible valid request to catch compilation/wiring regressions.
func TestConvert_AllPairs(t *testing.T) {
	// A minimal valid request body per format.
	minBodies := map[ir.Format][]byte{
		ir.FormatChat:      []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`),
		ir.FormatResponses: []byte(`{"model":"m","input":"hi"}`),
		ir.FormatAnthropic: []byte(`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`),
		ir.FormatGemini:    []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`),
	}
	for _, client := range ir.Formats {
		for _, upstream := range ir.Formats {
			if client == upstream {
				continue
			}
			name := string(client) + "→" + string(upstream)
			t.Run(name, func(t *testing.T) {
				body := minBodies[client]
				plan, err := bridge.Convert(client, upstream, body, opts())
				if err != nil {
					t.Fatalf("Convert(%s): %v", name, err)
				}
				if len(plan.Body) == 0 {
					t.Errorf("Convert(%s): empty upstream body", name)
				}
				var v any
				if err := json.Unmarshal(plan.Body, &v); err != nil {
					t.Errorf("Convert(%s): output not valid JSON: %v\n%s", name, err, plan.Body)
				}
			})
		}
	}
}

// --- golden update helper (run with -update flag) ---

// TestGoldenUpdate reads the testdata directory and prints missing golden
// files so the developer can generate them manually.
func TestGoldenUpdate(t *testing.T) {
	entries, err := os.ReadDir(testdataDir)
	if err != nil {
		t.Skip("testdata dir not available")
	}
	t.Logf("testdata has %d golden files", len(entries))
}
