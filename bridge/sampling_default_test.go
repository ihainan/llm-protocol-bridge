package bridge_test

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// Gemini CLI sends topK and topP on every request (with includeThoughts on).
// By default those hints are dropped and reported instead of failing the call.
const geminiCLIRequest = `{"contents":[{"role":"user","parts":[{"text":"hi"}]}],
 "generationConfig":{"temperature":1,"topP":0.95,"topK":64,
  "thinkingConfig":{"includeThoughts":true,"thinkingBudget":8192}}}`

func lossKinds(ls []ir.Loss) map[ir.LossKind]bool {
	m := map[ir.LossKind]bool{}
	for _, l := range ls {
		m[l.Kind] = true
	}
	return m
}

func TestSamplingHintsDroppedByDefault(t *testing.T) {
	cases := []struct {
		name     string
		upstream ir.Format
		absent   []string
		losses   []ir.LossKind
	}{
		// Anthropic with thinking on: top_p and top_k are not allowed
		{"anthropic", ir.FormatAnthropic, []string{"top_p", "top_k"}, []ir.LossKind{ir.LossSamplingClamp, ir.LossSamplingTopK}},
		// Chat Completions has no top_k
		{"chat", ir.FormatChat, []string{"top_k"}, []ir.LossKind{ir.LossSamplingTopK}},
		// Responses has no top_k
		{"responses", ir.FormatResponses, []string{"top_k"}, []ir.LossKind{ir.LossSamplingTopK}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := bridge.Convert(ir.FormatGemini, tc.upstream, []byte(geminiCLIRequest), bridge.Options{Model: "m"})
			if err != nil {
				t.Fatalf("Convert: %v", err)
			}
			var body map[string]json.RawMessage
			if err := json.Unmarshal(plan.Body, &body); err != nil {
				t.Fatalf("body: %v", err)
			}
			for _, k := range tc.absent {
				if _, ok := body[k]; ok {
					t.Errorf("%s still sent: %s", k, plan.Body)
				}
			}
			got := lossKinds(plan.Losses)
			for _, k := range tc.losses {
				if !got[k] {
					t.Errorf("loss %s not reported: %v", k, plan.Losses)
				}
			}
		})
	}
}

func TestSamplingHintsDenyStillRefuses(t *testing.T) {
	_, err := bridge.Convert(ir.FormatGemini, ir.FormatChat, []byte(geminiCLIRequest), bridge.Options{
		Model: "m",
		Deny:  ir.LossSet{ir.LossSamplingTopK: true},
	})
	if err == nil {
		t.Fatal("Deny sampling.top_k: want an error")
	}
}

func TestPenaltyDroppedTowardsAnthropic(t *testing.T) {
	plan, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic, []byte(
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"presence_penalty":0.5,"temperature":1.5}`), bridge.Options{Model: "m"})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	var body map[string]json.RawMessage
	_ = json.Unmarshal(plan.Body, &body)
	if _, ok := body["presence_penalty"]; ok {
		t.Errorf("presence_penalty still sent: %s", plan.Body)
	}
	if string(body["temperature"]) != "1" {
		t.Errorf("temperature not clamped to 1: %s", plan.Body)
	}
	got := lossKinds(plan.Losses)
	if !got[ir.LossSamplingPenalty] || !got[ir.LossSamplingClamp] {
		t.Errorf("losses: %v", plan.Losses)
	}
}

func TestLogprobsStillRefusedByDefault(t *testing.T) {
	_, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic, []byte(
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"logprobs":true}`), bridge.Options{Model: "m"})
	if err == nil {
		t.Fatal("logprobs towards Anthropic: want an error")
	}
}

func TestFrequencyPenaltyDroppedTowardsGeminiAndAnthropic(t *testing.T) {
	for _, up := range []ir.Format{ir.FormatGemini, ir.FormatAnthropic} {
		plan, err := bridge.Convert(ir.FormatChat, up, []byte(
			`{"model":"m","messages":[{"role":"user","content":"hi"}],"frequency_penalty":0.5}`), bridge.Options{Model: "m"})
		if err != nil {
			t.Fatalf("%s: Convert: %v", up, err)
		}
		if !lossKinds(plan.Losses)[ir.LossSamplingPenalty] {
			t.Errorf("%s: penalty loss not reported: %v", up, plan.Losses)
		}
	}
}

// Dropping or clamping is for values the client format accepts. A value outside
// the client format's own range is an invalid request whatever the loss policy.
func TestSamplingOutOfClientRangeIsInvalid(t *testing.T) {
	cases := []struct {
		name   string
		client ir.Format
		up     ir.Format
		body   string
		param  string
	}{
		{"chat presence_penalty", ir.FormatChat, ir.FormatAnthropic,
			`{"model":"m","messages":[{"role":"user","content":"hi"}],"presence_penalty":99}`, "presence_penalty"},
		{"chat frequency_penalty", ir.FormatChat, ir.FormatGemini,
			`{"model":"m","messages":[{"role":"user","content":"hi"}],"frequency_penalty":-3}`, "frequency_penalty"},
		{"chat top_p", ir.FormatChat, ir.FormatAnthropic,
			`{"model":"m","messages":[{"role":"user","content":"hi"}],"top_p":1.5}`, "top_p"},
		{"gemini temperature", ir.FormatGemini, ir.FormatAnthropic,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"temperature":3}}`, "generationConfig.temperature"},
		{"gemini topP", ir.FormatGemini, ir.FormatAnthropic,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"topP":1.2}}`, "generationConfig.topP"},
		{"gemini topK", ir.FormatGemini, ir.FormatChat,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"topK":-1}}`, "generationConfig.topK"},
		{"anthropic top_p", ir.FormatAnthropic, ir.FormatChat,
			`{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"top_p":2}`, "top_p"},
		{"anthropic top_k", ir.FormatAnthropic, ir.FormatChat,
			`{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"top_k":-5}`, "top_k"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := bridge.Convert(tc.client, tc.up, []byte(tc.body), bridge.Options{Model: "m"})
			var e *ir.Error
			if !errors.As(err, &e) {
				t.Fatalf("want an *ir.Error, got %v", err)
			}
			if e.Code != ir.CodeInvalidRequest || e.Param() != tc.param {
				t.Errorf("got code %s param %q, want invalid_request %q", e.Code, e.Param(), tc.param)
			}
		})
	}
}

// The range ends are valid values: zero and the upper bound pass.
func TestSamplingRangeEndsAccepted(t *testing.T) {
	cases := []struct {
		client ir.Format
		up     ir.Format
		body   string
	}{
		{ir.FormatChat, ir.FormatResponses,
			`{"model":"m","messages":[{"role":"user","content":"hi"}],"top_p":0,"temperature":2,"presence_penalty":-2,"frequency_penalty":2}`},
		{ir.FormatChat, ir.FormatChat,
			`{"model":"m","messages":[{"role":"user","content":"hi"}],"top_p":1,"temperature":0}`},
		{ir.FormatAnthropic, ir.FormatGemini,
			`{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"top_p":1,"top_k":0,"temperature":1}`},
		{ir.FormatGemini, ir.FormatGemini,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"temperature":2,"topP":0,"topK":0}}`},
		{ir.FormatGemini, ir.FormatGemini,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"temperature":0,"topP":1}}`},
	}
	for _, tc := range cases {
		if _, err := bridge.Convert(tc.client, tc.up, []byte(tc.body), bridge.Options{Model: "m"}); err != nil {
			t.Errorf("%s -> %s: %v\n%s", tc.client, tc.up, err, tc.body)
		}
	}
}
