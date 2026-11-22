package bridge_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/anthropic"
	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/chat"
	"github.com/ihainan/llm-protocol-bridge/gemini"
	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/responses"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// Review2 #10: Anthropic buildContentBlocks for Instructions must not swallow partToBlocks errors.
func TestReview2_P2_10_AnthropicSystemRejectsAudio(t *testing.T) {
	req := &ir.Request{
		Model: "claude-sonnet-4",
		Instructions: []ir.Part{
			ir.TextPart("be helpful"),
			{Type: ir.PartAudio, Media: &ir.Media{MediaType: "audio/wav", Data: "AAAA"}},
		},
		Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("hi")}}},
	}
	_, err := anthropic.BuildRequest(req, anthropic.BuildOptions{})
	if err == nil {
		t.Fatal("expected error for PartAudio in Anthropic system instructions, got nil")
	}
}

// Review2 #12: Gemini executableCode must preserve language and not silently vanish across encoders;
// encoders without native builtin support must report LossBuiltinActivity (and honor Deny).
func TestReview2_P2_12_GeminiExecutableCodeBuiltinHandling(t *testing.T) {
	t.Run("gemini_preserves_language_and_roundtrips", func(t *testing.T) {
		dec := gemini.NewStreamDecoder(gemini.DecodeOptions{})
		chunk := `{"candidates":[{"index":0,"content":{"role":"model","parts":[{"executableCode":{"language":"PYTHON","code":"print(1)"}}]},"finishReason":"STOP"}]}`
		evs, err := dec.Decode(nil, sse.Event{Data: []byte(chunk)})
		if err != nil {
			t.Fatalf("Decode: %v", err)
		}
		var builtinEv *ir.Event
		for i := range evs {
			if evs[i].Type == ir.EvBuiltin && evs[i].Builtin != nil {
				builtinEv = &evs[i]
				break
			}
		}
		if builtinEv == nil {
			t.Fatalf("expected EvBuiltin event, got %+v", evs)
		}
		if !strings.Contains(string(builtinEv.Builtin.Input), `"PYTHON"`) || !strings.Contains(string(builtinEv.Builtin.Input), `"print(1)"`) {
			t.Errorf("executableCode language or code lost: %s", builtinEv.Builtin.Input)
		}

		// Gemini StreamEncoder should re-emit executableCode with language and code.
		enc := gemini.NewStreamEncoder(gemini.EncodeOptions{Env: &ir.Env{}, AltSSE: true})
		var dst []byte
		for _, ev := range evs {
			dst, err = enc.Encode(dst, ev)
			if err != nil {
				t.Fatalf("gemini Encode: %v", err)
			}
		}
		dst, _ = enc.Close(dst)
		if !strings.Contains(string(dst), `"executableCode"`) || !strings.Contains(string(dst), `"PYTHON"`) || !strings.Contains(string(dst), `"print(1)"`) {
			t.Errorf("Gemini StreamEncoder dropped executableCode:\n%s", string(dst))
		}
	})

	t.Run("unsupported_encoder_records_loss_and_honors_deny", func(t *testing.T) {
		ev := ir.Event{
			Type: ir.EvBuiltin,
			Builtin: &ir.BuiltinActivity{
				Kind:   ir.BuiltinCodeExecution,
				Input:  json.RawMessage(`{"language":"PYTHON","code":"print(1)"}`),
				Origin: ir.FormatGemini,
			},
		}
		for _, name := range []string{"chat", "anthropic", "responses"} {
			t.Run(name, func(t *testing.T) {
				var losses []ir.Loss
				envAllow := &ir.Env{OnLoss: func(l ir.Loss) { losses = append(losses, l) }}
				envDeny := &ir.Env{Deny: ir.LossSet{ir.LossBuiltinActivity: true}}

				switch name {
				case "chat":
					e1 := chat.NewStreamEncoder(chat.EncodeOptions{Env: envAllow, Model: "m"})
					if _, err := e1.Encode(nil, ev); err != nil {
						t.Fatalf("chat allow err: %v", err)
					}
					e2 := chat.NewStreamEncoder(chat.EncodeOptions{Env: envDeny, Model: "m"})
					if _, err := e2.Encode(nil, ev); err == nil {
						t.Errorf("chat expected error when Deny[LossBuiltinActivity]=true")
					}
				case "anthropic":
					e1 := anthropic.NewStreamEncoder(anthropic.EncodeOptions{Env: envAllow, Model: "m"})
					if _, err := e1.Encode(nil, ev); err != nil {
						t.Fatalf("anthropic allow err: %v", err)
					}
					e2 := anthropic.NewStreamEncoder(anthropic.EncodeOptions{Env: envDeny, Model: "m"})
					if _, err := e2.Encode(nil, ev); err == nil {
						t.Errorf("anthropic expected error when Deny[LossBuiltinActivity]=true")
					}
				case "responses":
					e1 := responses.NewStreamEncoder(responses.EncodeOptions{Env: envAllow, Model: "m"})
					if _, err := e1.Encode(nil, ev); err != nil {
						t.Fatalf("responses allow err: %v", err)
					}
					e2 := responses.NewStreamEncoder(responses.EncodeOptions{Env: envDeny, Model: "m"})
					if _, err := e2.Encode(nil, ev); err == nil {
						t.Errorf("responses expected error when Deny[LossBuiltinActivity]=true")
					}
				}
				if len(losses) == 0 || losses[0].Kind != ir.LossBuiltinActivity {
					t.Errorf("%s expected LossBuiltinActivity recorded, got %+v", name, losses)
				}
			})
		}
	})
}

// Review2 #18: Gemini AUTO + allowedFunctionNames must preserve ToolChoiceAllowed with AllowedMode=auto.
func TestReview2_P2_18_GeminiAutoAllowedFunctionNames(t *testing.T) {
	body := []byte(`{
		"contents":[{"role":"user","parts":[{"text":"hi"}]}],
		"tools":[{"functionDeclarations":[{"name":"f"},{"name":"g"}]}],
		"toolConfig":{"functionCallingConfig":{"mode":"AUTO","allowedFunctionNames":["f"]}}
	}`)
	req, err := gemini.ParseRequest(body, gemini.ParseOptions{Model: "gemini-2.5-pro"})
	if err != nil {
		t.Fatalf("ParseRequest: %v", err)
	}
	if req.ToolChoice.Mode != ir.ToolChoiceAllowed {
		t.Fatalf("expected ToolChoiceAllowed, got %q", req.ToolChoice.Mode)
	}
	if req.ToolChoice.AllowedMode != ir.ToolChoiceAuto {
		t.Errorf("expected AllowedMode=auto, got %q", req.ToolChoice.AllowedMode)
	}
	if len(req.ToolChoice.Allowed) != 1 || req.ToolChoice.Allowed[0].Name != "f" {
		t.Errorf("expected Allowed=[f], got %+v", req.ToolChoice.Allowed)
	}
}

// Review2 #27 & #34: Multi-candidate stream must not mark Done() or skip End() error
// when only candidate 0 finished / hit stop while candidate 1 is unfinished.
func TestReview2_P2_27_34_MultiCandidateStopAndDone(t *testing.T) {
	t.Run("stop_emulate_cand0_hit_cand1_incomplete_errors_on_eof", func(t *testing.T) {
		reqBody := []byte(`{"model":"gpt-5","stream":true,"n":2,"stop":["END"],"messages":[{"role":"user","content":"hi"}]}`)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatChat, reqBody, bridge.Options{
			StopMode: ir.StopEmulate,
		})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		s := plan.NewStream()
		var out []byte
		// Candidate 0 hits emulated stop "END"; candidate 1 emits partial text without finish_reason.
		for _, line := range []string{
			`{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{"content":"okEND"},"finish_reason":null}]}`,
			`{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":1,"delta":{"content":"half"},"finish_reason":null}]}`,
		} {
			out, err = s.Feed(out, sse.Event{Data: []byte(line)})
			if err != nil {
				t.Fatalf("Feed: %v", err)
			}
		}
		if s.Done() {
			t.Errorf("Stream.Done() must be false when only 1 of 2 candidates finished")
		}
		out, err = s.End(out, ir.EndEOF)
		if err != nil {
			t.Fatalf("End: %v", err)
		}
		if !strings.Contains(string(out), `"error"`) {
			t.Errorf("expected stream error when candidate 1 ended without finish_reason:\n%s", string(out))
		}
	})

	t.Run("n2_cand0_finish_cand1_unfinished_not_done_and_errors_on_eof", func(t *testing.T) {
		reqBody := []byte(`{"model":"gpt-5","stream":true,"n":2,"messages":[{"role":"user","content":"hi"}]}`)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatChat, reqBody, bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		s := plan.NewStream()
		var out []byte
		for _, line := range []string{
			`{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{"content":"a"},"finish_reason":"stop"}]}`,
			`{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":1,"delta":{"content":"b"},"finish_reason":null}]}`,
		} {
			out, err = s.Feed(out, sse.Event{Data: []byte(line)})
			if err != nil {
				t.Fatalf("Feed: %v", err)
			}
		}
		if s.Done() {
			t.Errorf("Stream.Done() must remain false until all n=2 candidates finish")
		}
		out, err = s.End(out, ir.EndEOF)
		if err != nil {
			t.Fatalf("End: %v", err)
		}
		if !strings.Contains(string(out), `"error"`) {
			t.Errorf("expected error when candidate 1 was truncated before finish_reason:\n%s", string(out))
		}
	})
}

// Review2 #29: Hidden reasoning must still flush pending stop-sequence text tail buffer
// so "ab" -> hidden reasoning -> "c" does not falsely trigger stop "abc".
func TestReview2_P2_29_HiddenReasoningFlushesStopBuffer(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","stream":true,"stop":["abc"],"messages":[{"role":"user","content":"hi"}]}`)
	plan, err := bridge.Convert(ir.FormatChat, ir.FormatChat, reqBody, bridge.Options{
		StopMode:  ir.StopEmulate,
		Reasoning: ir.ReasoningOutHidden,
	})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	s := plan.NewStream()
	raw := "data: " + `{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{"content":"ab"},"finish_reason":null}]}` + "\n\n" +
		"data: " + `{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{"reasoning_content":"secret"},"finish_reason":null}]}` + "\n\n" +
		"data: " + `{"id":"c1","object":"chat.completion.chunk","model":"gpt-5","choices":[{"index":0,"delta":{"content":"c"},"finish_reason":"stop"}]}` + "\n\n" +
		"data: [DONE]\n\n"
	out := string(feedSSE(t, s, raw))
	if strings.Contains(out, "secret") {
		t.Errorf("hidden reasoning leaked:\n%s", out)
	}
	if !strings.Contains(out, `"ab"`) || !strings.Contains(out, `"c"`) {
		t.Errorf("expected both 'ab' and 'c' emitted without false stop match on 'abc':\n%s", out)
	}
}

// Review2 #30: Non-streaming ConvertResponse must enforce StopEmulate truncation.
func TestReview2_P2_30_NonStreamingStopEmulate(t *testing.T) {
	reqBody := []byte(`{"model":"gpt-5","stream":false,"stop":["END"],"messages":[{"role":"user","content":"hi"}]}`)
	plan, err := bridge.Convert(ir.FormatChat, ir.FormatResponses, reqBody, bridge.Options{
		StopMode: ir.StopEmulate,
	})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	upstreamResp := []byte(`{
		"id":"resp_1","object":"response","model":"gpt-5","status":"completed",
		"output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"aENDb"}]}]
	}`)
	clientBody, irResp, err := plan.ConvertResponse(upstreamResp)
	if err != nil {
		t.Fatalf("ConvertResponse: %v", err)
	}
	if got := ir.TextOf(irResp.Outputs[0].Parts, ""); got != "a" {
		t.Errorf("expected non-streaming StopEmulate text 'a', got %q (body=%s)", got, string(clientBody))
	}
	if irResp.Outputs[0].Stop.Reason != ir.StopSequence || irResp.Outputs[0].Stop.Sequence != "END" {
		t.Errorf("expected StopSequence(END), got %+v", irResp.Outputs[0].Stop)
	}
}

// Review2 #38: Anthropic thinking(T,S) -> redacted(A) -> redacted(B) must emit 3 distinct
// reasoning items in Responses stream without overwriting S or A.
func TestReview2_P2_38_ResponsesDistinctReasoningAndRedactedBlocks(t *testing.T) {
	seal := ir.NewSealCodec([]byte("secret-key"))
	origin := ir.Origin{Format: ir.FormatAnthropic, Provider: "anthropic"}
	enc := responses.NewStreamEncoder(responses.EncodeOptions{
		Env:   &ir.Env{ProviderID: "anthropic", Seal: seal},
		Model: "claude-sonnet-4",
	})
	var dst []byte
	dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStart, Model: "claude-sonnet-4"})
	dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvReasoningStart})
	dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvReasoning, Text: "T"})
	dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvSignature, Signature: &ir.Signature{Value: "S", Origin: origin}})
	dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvSealed, Sealed: &ir.Sealed{Kind: ir.SealedAnthropicRedacted, Data: json.RawMessage(`"A"`), Origin: origin}})
	dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvSealed, Sealed: &ir.Sealed{Kind: ir.SealedAnthropicRedacted, Data: json.RawMessage(`"B"`), Origin: origin}})
	dst, _ = enc.Encode(dst, ir.Event{Type: ir.EvStop, Stop: &ir.Stop{Reason: ir.StopEndTurn}})
	dst, _ = enc.Close(dst)

	r := sse.NewReader(bytes.NewReader(dst))
	var reasoningItems []struct {
		Summary          []struct{ Text string } `json:"summary"`
		EncryptedContent string                  `json:"encrypted_content"`
	}
	for {
		ev, err := r.Next()
		if err != nil {
			break
		}
		var obj struct {
			Type string `json:"type"`
			Item struct {
				Type             string                  `json:"type"`
				Summary          []struct{ Text string } `json:"summary"`
				EncryptedContent string                  `json:"encrypted_content"`
			} `json:"item"`
		}
		if json.Unmarshal(ev.Data, &obj) == nil && obj.Type == "response.output_item.done" && obj.Item.Type == "reasoning" {
			reasoningItems = append(reasoningItems, struct {
				Summary          []struct{ Text string } `json:"summary"`
				EncryptedContent string                  `json:"encrypted_content"`
			}{Summary: obj.Item.Summary, EncryptedContent: obj.Item.EncryptedContent})
		}
	}
	if len(reasoningItems) != 3 {
		t.Fatalf("expected 3 distinct reasoning items (thinking + redacted A + redacted B), got %d:\n%s", len(reasoningItems), string(dst))
	}
	r0, ok0 := seal.Unseal(reasoningItems[0].EncryptedContent)
	r1, ok1 := seal.Unseal(reasoningItems[1].EncryptedContent)
	r2, ok2 := seal.Unseal(reasoningItems[2].EncryptedContent)
	if !ok0 || r0.Value != "S" || !ok1 || r1.Value != `"A"` || !ok2 || r2.Value != `"B"` {
		t.Errorf("unexpected unsealed items: r0=(%v,%q) r1=(%v,%q) r2=(%v,%q)", ok0, r0.Value, ok1, r1.Value, ok2, r2.Value)
	}
}

// Review2 #49: Responses streaming response.failed and error events must refine error type
// (e.g., code: insufficient_quota -> ErrBilling) consistently with ParseError.
func TestReview2_P2_49_ResponsesStreamErrorRefinesType(t *testing.T) {
	dec := responses.NewStreamDecoder(responses.DecodeOptions{})
	evData := `{"type":"response.failed","sequence_number":1,"response":{"id":"r1","status":"failed","error":{"type":"rate_limit_error","code":"insufficient_quota","message":"quota exceeded"}}}`
	evs, err := dec.Decode(nil, sse.Event{Name: "response.failed", Data: []byte(evData)})
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if len(evs) != 1 || evs[0].Type != ir.EvError || evs[0].Err == nil {
		t.Fatalf("expected EvError, got %+v", evs)
	}
	if evs[0].Err.Type != ir.ErrBilling {
		t.Errorf("response.failed with insufficient_quota: want ErrBilling, got %q", evs[0].Err.Type)
	}
}

// Review2 #50: Anthropic and Responses ParseResponse must reject structurally incomplete
// objects like {"id":"x"} instead of returning an empty valid response.
func TestReview2_P2_50_RejectIncompleteResponseObjects(t *testing.T) {
	if _, err := anthropic.ParseResponse([]byte(`{"id":"x"}`), anthropic.DecodeOptions{}); err == nil {
		t.Errorf("anthropic.ParseResponse accepted structurally incomplete {\"id\":\"x\"}")
	}
	if _, err := responses.ParseResponse([]byte(`{"id":"x"}`), responses.DecodeOptions{}); err == nil {
		t.Errorf("responses.ParseResponse accepted structurally incomplete {\"id\":\"x\"}")
	}
}

// Review2 #53: jsonx.Int must preserve int64 precision for exact integer scientific notation.
func TestReview2_P2_53_JSONXIntScientificPrecision(t *testing.T) {
	n, ok := jsonx.Int([]byte(`9007199254740993e0`))
	if !ok {
		t.Fatal("jsonx.Int failed to parse 9007199254740993e0")
	}
	if n != 9007199254740993 {
		t.Errorf("jsonx.Int lost precision: want 9007199254740993, got %d", n)
	}
}
