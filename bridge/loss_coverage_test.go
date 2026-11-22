package bridge_test

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// TestLossCoverage checks that every IR request field with a non-neutral value
// is either (a) present in the built upstream body, or (b) present in
// Plan.Losses (or causes an error when the default policy denies it).
//
// A new IR field must be added to the table below to avoid failing this test.
func TestLossCoverage(t *testing.T) {
	boolTrue := true
	boolFalse := false
	intV := func(n int) *int { return &n }
	floatV := func(f float64) *float64 { return &f }
	int64V := func(n int64) *int64 { return &n }

	type result struct {
		bodyKeys map[string]bool
		lossKeys map[string]bool
		isErr    bool
	}

	convert := func(upstream ir.Format, req *ir.Request, opts bridge.Options) result {
		plan, err := bridge.Build(req, ir.FormatChat, upstream, opts)
		if err != nil {
			return result{isErr: true}
		}
		body := map[string]json.RawMessage{}
		json.Unmarshal(plan.Body, &body)
		bk := map[string]bool{}
		for k := range body {
			bk[k] = true
		}
		lk := map[string]bool{}
		for _, l := range plan.Losses {
			lk[string(l.Kind)] = true
		}
		return result{bodyKeys: bk, lossKeys: lk}
	}

	// allow builds Options that explicitly allow a loss kind (so Lose records it
	// instead of returning an error).
	allow := func(k ir.LossKind) bridge.Options {
		return bridge.Options{Allow: ir.LossSet{k: true}}
	}

	// deny builds Options that deny a loss kind (so Lose returns an error).
	deny := func(k ir.LossKind) bridge.Options {
		return bridge.Options{Deny: ir.LossSet{k: true}}
	}

	type cas struct {
		name     string
		upstream ir.Format
		req      *ir.Request
		opts     bridge.Options
		// wantBody: the field must appear in the built body.
		wantBody string
		// wantLoss: the loss key must appear in Plan.Losses.
		wantLoss string
		// wantErr: build must return an error.
		wantErr bool
	}

	base := func() *ir.Request {
		return &ir.Request{
			Model:    "m",
			Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("hi")}}},
		}
	}

	_ = int64V

	cases := []cas{
		// --- temperature ---
		{name: "temperature→anthropic mapped", upstream: ir.FormatAnthropic,
			req:      func() *ir.Request { r := base(); r.Temperature = floatV(0.5); return r }(),
			wantBody: "temperature"},
		{name: "temperature=1.8→anthropic clamp loss", upstream: ir.FormatAnthropic,
			req:  func() *ir.Request { r := base(); r.Temperature = floatV(1.8); return r }(),
			opts: allow(ir.LossSamplingClamp), wantLoss: "sampling.clamp"},
		{name: "temperature=1.8→anthropic clamp error", upstream: ir.FormatAnthropic,
			req:     func() *ir.Request { r := base(); r.Temperature = floatV(1.8); return r }(),
			opts:    deny(ir.LossSamplingClamp),
			wantErr: true},
		{name: "temperature→chat mapped", upstream: ir.FormatChat,
			req:      func() *ir.Request { r := base(); r.Temperature = floatV(0.5); return r }(),
			wantBody: "temperature"},
		{name: "temperature→gemini mapped", upstream: ir.FormatGemini,
			req:      func() *ir.Request { r := base(); r.Temperature = floatV(0.5); return r }(),
			wantBody: "generationConfig"},
		{name: "temperature→responses mapped", upstream: ir.FormatResponses,
			req:      func() *ir.Request { r := base(); r.Temperature = floatV(0.5); return r }(),
			wantBody: "temperature"},

		// --- top_k ---
		{name: "top_k→anthropic mapped", upstream: ir.FormatAnthropic,
			req:      func() *ir.Request { r := base(); r.TopK = intV(5); return r }(),
			wantBody: "top_k"},
		{name: "top_k→chat loss", upstream: ir.FormatChat,
			req:      func() *ir.Request { r := base(); r.TopK = intV(5); return r }(),
			opts:     allow(ir.LossSamplingTopK),
			wantLoss: "sampling.top_k"},
		{name: "top_k→chat error", upstream: ir.FormatChat,
			req:     func() *ir.Request { r := base(); r.TopK = intV(5); return r }(),
			opts:    deny(ir.LossSamplingTopK),
			wantErr: true},
		{name: "top_k→responses loss", upstream: ir.FormatResponses,
			req:      func() *ir.Request { r := base(); r.TopK = intV(5); return r }(),
			opts:     allow(ir.LossSamplingTopK),
			wantLoss: "sampling.top_k"},
		{name: "top_k→gemini mapped", upstream: ir.FormatGemini,
			req:      func() *ir.Request { r := base(); r.TopK = intV(5); return r }(),
			wantBody: "generationConfig"},

		// --- penalties ---
		{name: "penalty→anthropic loss", upstream: ir.FormatAnthropic,
			req:      func() *ir.Request { r := base(); r.PresencePenalty = floatV(0.5); return r }(),
			opts:     allow(ir.LossSamplingPenalty),
			wantLoss: "sampling.penalty"},
		{name: "penalty→anthropic error", upstream: ir.FormatAnthropic,
			req:     func() *ir.Request { r := base(); r.PresencePenalty = floatV(0.5); return r }(),
			opts:    deny(ir.LossSamplingPenalty),
			wantErr: true},
		{name: "freq_penalty→anthropic loss", upstream: ir.FormatAnthropic,
			req:      func() *ir.Request { r := base(); r.FrequencyPenalty = floatV(0.5); return r }(),
			opts:     allow(ir.LossSamplingPenalty),
			wantLoss: "sampling.penalty"},
		{name: "penalty→chat mapped", upstream: ir.FormatChat,
			req:      func() *ir.Request { r := base(); r.PresencePenalty = floatV(0.5); return r }(),
			wantBody: "presence_penalty"},
		{name: "penalty→responses loss", upstream: ir.FormatResponses,
			req:      func() *ir.Request { r := base(); r.PresencePenalty = floatV(0.5); return r }(),
			opts:     allow(ir.LossSamplingPenalty),
			wantLoss: "sampling.penalty"},
		{name: "penalty→gemini loss", upstream: ir.FormatGemini,
			req:      func() *ir.Request { r := base(); r.PresencePenalty = floatV(0.5); return r }(),
			opts:     allow(ir.LossSamplingPenalty),
			wantLoss: "sampling.penalty"},
		{name: "penalty→gemini error", upstream: ir.FormatGemini,
			req:     func() *ir.Request { r := base(); r.PresencePenalty = floatV(0.5); return r }(),
			opts:    deny(ir.LossSamplingPenalty),
			wantErr: true},

		// --- logprobs ---
		{name: "logprobs→anthropic loss", upstream: ir.FormatAnthropic,
			req:      func() *ir.Request { r := base(); r.Logprobs = &boolTrue; return r }(),
			opts:     allow(ir.LossSamplingLogprobs),
			wantLoss: "sampling.logprobs"},
		{name: "logprobs→anthropic error", upstream: ir.FormatAnthropic,
			req:     func() *ir.Request { r := base(); r.Logprobs = &boolTrue; return r }(),
			opts:    deny(ir.LossSamplingLogprobs),
			wantErr: true},
		{name: "logprobs→chat mapped", upstream: ir.FormatChat,
			req:      func() *ir.Request { r := base(); r.Logprobs = &boolTrue; return r }(),
			wantBody: "logprobs"},
		{name: "logprobs→responses loss", upstream: ir.FormatResponses,
			req:      func() *ir.Request { r := base(); r.Logprobs = &boolTrue; return r }(),
			opts:     allow(ir.LossSamplingLogprobs),
			wantLoss: "sampling.logprobs"},
		{name: "logprobs→gemini loss", upstream: ir.FormatGemini,
			req:      func() *ir.Request { r := base(); r.Logprobs = &boolTrue; return r }(),
			opts:     allow(ir.LossSamplingLogprobs),
			wantLoss: "sampling.logprobs"},

		// --- seed ---
		{name: "seed→chat mapped", upstream: ir.FormatChat,
			req:      func() *ir.Request { r := base(); r.Seed = int64V(7); return r }(),
			wantBody: "seed"},
		{name: "seed→anthropic loss", upstream: ir.FormatAnthropic,
			req:      func() *ir.Request { r := base(); r.Seed = int64V(7); return r }(),
			wantLoss: "sampling.seed"},
		{name: "seed→gemini mapped", upstream: ir.FormatGemini,
			req:      func() *ir.Request { r := base(); r.Seed = int64V(7); return r }(),
			wantBody: "generationConfig"},
		{name: "seed→responses loss", upstream: ir.FormatResponses,
			req:      func() *ir.Request { r := base(); r.Seed = int64V(7); return r }(),
			wantLoss: "sampling.seed"},

		// --- stop limit ---
		{name: "6 stops→anthropic mapped", upstream: ir.FormatAnthropic,
			req: func() *ir.Request {
				r := base()
				r.Stop = []string{"a", "b", "c", "d", "e", "f"}
				return r
			}(),
			wantBody: "stop_sequences"},
		{name: "6 stops→chat loss", upstream: ir.FormatChat,
			req: func() *ir.Request {
				r := base()
				r.Stop = []string{"a", "b", "c", "d", "e", "f"}
				return r
			}(),
			opts:     allow(ir.LossStopLimit),
			wantLoss: "stop.limit"},
		{name: "6 stops→chat error", upstream: ir.FormatChat,
			req: func() *ir.Request {
				r := base()
				r.Stop = []string{"a", "b", "c", "d", "e", "f"}
				return r
			}(),
			opts:    deny(ir.LossStopLimit),
			wantErr: true},

		// --- parallel_tool_calls ---
		{name: "parallel=false→anthropic mapped via tool_choice", upstream: ir.FormatAnthropic,
			req: func() *ir.Request {
				r := base()
				r.ParallelToolCalls = &boolFalse
				r.Tools = []ir.Tool{{Name: "f", Kind: ir.ToolFunction}}
				return r
			}(),
			wantBody: "tool_choice"},
		{name: "parallel_tool_calls→chat mapped", upstream: ir.FormatChat,
			req:      func() *ir.Request { r := base(); r.ParallelToolCalls = &boolFalse; return r }(),
			wantBody: "parallel_tool_calls"},
		{name: "parallel_tool_calls→gemini loss", upstream: ir.FormatGemini,
			req:      func() *ir.Request { r := base(); r.ParallelToolCalls = &boolFalse; return r }(),
			wantLoss: "tool.parallel"},
	}

	for _, c := range cases {
		c := c
		t.Run(c.name, func(t *testing.T) {
			res := convert(c.upstream, c.req, c.opts)
			if c.wantErr {
				if !res.isErr {
					t.Errorf("wanted error but got none; losses=%v body=%v", keys(res.lossKeys), keys(res.bodyKeys))
				}
				return
			}
			if res.isErr {
				t.Errorf("unexpected error")
				return
			}
			if c.wantBody != "" && !res.bodyKeys[c.wantBody] {
				t.Errorf("field %q not in body; body keys=%v", c.wantBody, keys(res.bodyKeys))
			}
			if c.wantLoss != "" && !res.lossKeys[c.wantLoss] {
				t.Errorf("loss %q not in Losses; losses=%v", c.wantLoss, keys(res.lossKeys))
			}
		})
	}
}

func keys(m map[string]bool) string {
	var ks []string
	for k := range m {
		ks = append(ks, k)
	}
	return fmt.Sprintf("[%s]", strings.Join(ks, " "))
}
