package anthropic_test

import (
	"encoding/json"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

func decodeBody(t *testing.T, b []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("unmarshal %s: %v", b, err)
	}
	return m
}

func TestAdaptiveThinkingDisplayAndEffort(t *testing.T) {
	adaptiveProfile := &ir.ModelProfile{
		Style:   ir.StyleAdaptive,
		Efforts: []ir.Effort{ir.EffortLow, ir.EffortMedium, ir.EffortHigh, ir.EffortXHigh, ir.EffortMax},
	}

	t.Run("responses_adaptive_default_summarized_and_effort", func(t *testing.T) {
		raw := []byte(`{"model":"claude-sonnet-5-5","input":"hi","reasoning":{"effort":"high"}}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, raw, bridge.Options{Profile: adaptiveProfile})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		m := decodeBody(t, plan.Body)
		th, _ := m["thinking"].(map[string]any)
		if th["type"] != "adaptive" || th["display"] != "summarized" {
			t.Fatalf("want adaptive+summarized, got thinking=%v", th)
		}
		if _, hasBudget := th["budget_tokens"]; hasBudget {
			t.Fatalf("adaptive must not carry budget_tokens: %v", th)
		}
		oc, _ := m["output_config"].(map[string]any)
		if oc["effort"] != "high" {
			t.Fatalf("want output_config.effort=high, got %v", oc)
		}
	})

	t.Run("responses_summary_none_sets_omitted", func(t *testing.T) {
		raw := []byte(`{"model":"claude-sonnet-5-5","input":"hi","reasoning":{"effort":"medium","summary":"none"}}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, raw, bridge.Options{Profile: adaptiveProfile})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		m := decodeBody(t, plan.Body)
		th, _ := m["thinking"].(map[string]any)
		if th["type"] != "adaptive" || th["display"] != "omitted" {
			t.Fatalf("want adaptive+omitted when summary=none, got thinking=%v", th)
		}
		oc, _ := m["output_config"].(map[string]any)
		if oc["effort"] != "medium" {
			t.Fatalf("want output_config.effort=medium, got %v", oc)
		}
	})

	t.Run("responses_minimal_fits_to_low_with_summarized", func(t *testing.T) {
		raw := []byte(`{"model":"claude-sonnet-5-5","input":"hi","reasoning":{"effort":"minimal"}}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, raw, bridge.Options{Profile: adaptiveProfile})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		m := decodeBody(t, plan.Body)
		th, _ := m["thinking"].(map[string]any)
		oc, _ := m["output_config"].(map[string]any)
		if th["type"] != "adaptive" || th["display"] != "summarized" || oc["effort"] != "low" {
			t.Fatalf("want adaptive+summarized+low, got body=%s", plan.Body)
		}
	})

	t.Run("chat_reasoning_effort_sets_summarized_and_effort", func(t *testing.T) {
		raw := []byte(`{"model":"claude-sonnet-5-5","reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)
		plan, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic, raw, bridge.Options{Profile: adaptiveProfile})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		m := decodeBody(t, plan.Body)
		th, _ := m["thinking"].(map[string]any)
		oc, _ := m["output_config"].(map[string]any)
		if th["type"] != "adaptive" || th["display"] != "summarized" || oc["effort"] != "xhigh" {
			t.Fatalf("want adaptive+summarized+xhigh, got body=%s", plan.Body)
		}
	})

	t.Run("gemini_include_thoughts_maps_shown_to_summarized_and_budget_to_effort", func(t *testing.T) {
		raw := []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"thinkingBudget":8000,"includeThoughts":true}}}`)
		plan, err := bridge.Convert(ir.FormatGemini, ir.FormatAnthropic, raw, bridge.Options{Model: "claude-sonnet-5-5", Profile: adaptiveProfile})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		m := decodeBody(t, plan.Body)
		th, _ := m["thinking"].(map[string]any)
		oc, _ := m["output_config"].(map[string]any)
		if th["type"] != "adaptive" || th["display"] != "summarized" || oc["effort"] != "medium" {
			t.Fatalf("want adaptive+summarized+medium, got body=%s", plan.Body)
		}
	})

	t.Run("gemini_include_thoughts_unlisted_manual_model_unchanged", func(t *testing.T) {
		raw := []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"thinkingBudget":8000,"includeThoughts":true}}}`)
		plan, err := bridge.Convert(ir.FormatGemini, ir.FormatAnthropic, raw, bridge.Options{Model: "claude-haiku-4-5"})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		m := decodeBody(t, plan.Body)
		th, _ := m["thinking"].(map[string]any)
		if th["type"] != "enabled" || th["budget_tokens"] != float64(8000) {
			t.Fatalf("want enabled+8000, got thinking=%v", th)
		}
		if _, hasDisplay := th["display"]; hasDisplay {
			t.Fatalf("manual unlisted Gemini->Anthropic must not inject display, got %v", th)
		}
	})

	t.Run("anthropic_explicit_display_wins_over_default", func(t *testing.T) {
		for _, disp := range []string{"omitted", "summarized", "updates"} {
			raw := []byte(`{"model":"claude-sonnet-5-5","max_tokens":4096,"thinking":{"type":"adaptive","display":"` + disp + `"},"output_config":{"effort":"high"},"messages":[{"role":"user","content":"hi"}]}`)
			plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatAnthropic, raw, bridge.Options{Profile: adaptiveProfile})
			if err != nil {
				t.Fatalf("disp=%s Convert: %v", disp, err)
			}
			m := decodeBody(t, plan.Body)
			th, _ := m["thinking"].(map[string]any)
			if th["type"] != "adaptive" || th["display"] != disp {
				t.Fatalf("disp=%s want preserved display, got %v", disp, th)
			}
		}
	})

	t.Run("anthropic_omitted_display_not_injected_on_native", func(t *testing.T) {
		raw := []byte(`{"model":"claude-sonnet-5-5","max_tokens":4096,"thinking":{"type":"adaptive"},"output_config":{"effort":"high"},"messages":[{"role":"user","content":"hi"}]}`)
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatAnthropic, raw, bridge.Options{Profile: adaptiveProfile})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		m := decodeBody(t, plan.Body)
		th, _ := m["thinking"].(map[string]any)
		if th["type"] != "adaptive" {
			t.Fatalf("want adaptive, got %v", th)
		}
		if _, hasDisplay := th["display"]; hasDisplay {
			t.Fatalf("native Anthropic without display must not inject display, got %v", th)
		}
	})

	t.Run("off_lowest_hidden_adaptive_uses_omitted_and_lowest_effort", func(t *testing.T) {
		alwaysAdaptive := &ir.ModelProfile{
			Style:   ir.StyleAdaptive,
			Efforts: []ir.Effort{ir.EffortLow, ir.EffortMedium, ir.EffortHigh, ir.EffortXHigh, ir.EffortMax},
			Off:     ir.OffRule{Mode: ir.OffLowestHidden},
		}
		raw := []byte(`{"model":"claude-opus-5-5","input":"hi","reasoning":{"effort":"none"}}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, raw, bridge.Options{Profile: alwaysAdaptive})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		m := decodeBody(t, plan.Body)
		th, _ := m["thinking"].(map[string]any)
		oc, _ := m["output_config"].(map[string]any)
		if th["type"] != "adaptive" || th["display"] != "omitted" || oc["effort"] != "low" {
			t.Fatalf("want adaptive+omitted+low for OffLowestHidden, got body=%s", plan.Body)
		}
		if _, hasBudget := th["budget_tokens"]; hasBudget {
			t.Fatalf("OffLowestHidden adaptive must not carry budget_tokens: %v", th)
		}
	})

	t.Run("omitted_reasoning_does_not_enable_thinking", func(t *testing.T) {
		raw := []byte(`{"model":"claude-sonnet-5-5","input":"hi"}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, raw, bridge.Options{Profile: adaptiveProfile})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		m := decodeBody(t, plan.Body)
		if _, hasThinking := m["thinking"]; hasThinking {
			t.Fatalf("omitted reasoning must not inject thinking: %s", plan.Body)
		}
		if _, hasOC := m["output_config"]; hasOC {
			t.Fatalf("omitted reasoning must not inject output_config: %s", plan.Body)
		}
	})

	t.Run("explicit_disabled_has_no_display", func(t *testing.T) {
		raw := []byte(`{"model":"claude-sonnet-5-5","input":"hi","reasoning":{"effort":"none"}}`)
		plan, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, raw, bridge.Options{Profile: adaptiveProfile})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		m := decodeBody(t, plan.Body)
		th, _ := m["thinking"].(map[string]any)
		if th["type"] != "disabled" {
			t.Fatalf("want disabled, got %v", th)
		}
		if _, hasDisplay := th["display"]; hasDisplay {
			t.Fatalf("disabled must not carry display: %v", th)
		}
	})
}
