package bridge_test

import (
	"encoding/json"
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
