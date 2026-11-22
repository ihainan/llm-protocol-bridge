package bridge_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/anthropic"
	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/gemini"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/responses"
)

// Item 12: Eliminate silent drops, semantic mutations, and non-standard wire fields.
func TestStage5B_NoSilentDrops(t *testing.T) {
	t.Run("anthropic_parse_none_disable_parallel_output_config_adaptive_text_doc_builtins", func(t *testing.T) {
		raw := []byte(`{
			"model": "claude-opus-4-6",
			"max_tokens": 1024,
			"thinking": {"type": "adaptive"},
			"output_config": {
				"effort": "high",
				"format": {"type": "json_schema", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}
			},
			"tools": [
				{"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
				{"name": "f", "description": "fn", "input_schema": {"type": "object"}}
			],
			"tool_choice": {"type": "auto", "disable_parallel_tool_use": true},
			"messages": [{
				"role": "user",
				"content": [
					{"type": "document", "title": "note", "source": {"type": "text", "media_type": "text/plain", "data": "hello doc"}}
				]
			}]
		}`)
		req, err := anthropic.ParseRequest(raw, anthropic.ParseOptions{})
		if err != nil {
			t.Fatalf("anthropic.ParseRequest: %v", err)
		}
		if req.Reasoning == nil || req.Reasoning.Mode != ir.ReasoningAdaptive || req.Reasoning.Effort != ir.EffortHigh {
			t.Errorf("adaptive/output_config.effort lost: %+v", req.Reasoning)
		}
		if req.Output == nil || req.Output.Type != ir.OutputJSONSchema || len(req.Output.Schema) == 0 {
			t.Errorf("output_config.format lost: %+v", req.Output)
		}
		if req.ParallelToolCalls == nil || *req.ParallelToolCalls != false {
			t.Errorf("disable_parallel_tool_use lost: %v", req.ParallelToolCalls)
		}
		if len(req.Builtins) != 1 || req.Builtins[0].Kind != ir.BuiltinWebSearch {
			t.Errorf("Anthropic web_search builtin lost: %+v", req.Builtins)
		}
		if len(req.Messages[0].Parts) != 1 || req.Messages[0].Parts[0].Media == nil || req.Messages[0].Parts[0].Media.Text != "hello doc" {
			t.Errorf("plain-text document lost: %+v", req.Messages[0].Parts)
		}

		// tool_choice.type: "none" must parse cleanly.
		reqNone, err := anthropic.ParseRequest([]byte(`{"model":"claude","max_tokens":64,"tool_choice":{"type":"none"},"messages":[{"role":"user","content":"hi"}]}`), anthropic.ParseOptions{})
		if err != nil {
			t.Fatalf("anthropic tool_choice none failed: %v", err)
		}
		if reqNone.ToolChoice.Mode != ir.ToolChoiceNone {
			t.Errorf("want ToolChoiceNone, got %+v", reqNone.ToolChoice)
		}
	})

	t.Run("anthropic_build_output_candidates_builtins_allowed_tools_strict_loss_midsystem_fold", func(t *testing.T) {
		// 1. json_schema maps to output_config.format.
		pSchema, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic, []byte(`{
			"model": "claude-sonnet-4",
			"messages": [{"role": "user", "content": "hi"}],
			"response_format": {"type": "json_schema", "json_schema": {"name": "res", "schema": {"type": "object"}}}
		}`), bridge.Options{})
		if err != nil {
			t.Fatalf("json_schema->Anthropic: %v", err)
		}
		if !strings.Contains(string(pSchema.Body), `"output_config"`) || !strings.Contains(string(pSchema.Body), `"json_schema"`) {
			t.Errorf("expected output_config.format in Anthropic body: %s", pSchema.Body)
		}

		// 2. json_object -> Anthropic returns ir.Unsupported.
		_, err = bridge.Convert(ir.FormatChat, ir.FormatAnthropic, []byte(`{
			"model": "claude-sonnet-4",
			"messages": [{"role": "user", "content": "hi"}],
			"response_format": {"type": "json_object"}
		}`), bridge.Options{})
		if err == nil {
			t.Errorf("expected error for json_object toward Anthropic")
		}

		// 3. n > 1 -> Anthropic returns ir.Unsupported.
		_, err = bridge.Convert(ir.FormatChat, ir.FormatAnthropic, []byte(`{
			"model": "claude-sonnet-4",
			"n": 2,
			"messages": [{"role": "user", "content": "hi"}]
		}`), bridge.Options{})
		if err == nil {
			t.Errorf("expected error for n=2 toward Anthropic")
		}

		// 4. Builtin web_search maps; unsupported builtin (code_execution) errors.
		pWS, err := bridge.Build(&ir.Request{
			Model:    "claude-sonnet-4",
			Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("search")}}},
			Builtins: []ir.BuiltinTool{{Kind: ir.BuiltinWebSearch}},
		}, ir.FormatResponses, ir.FormatAnthropic, bridge.Options{})
		if err != nil {
			t.Fatalf("web_search->Anthropic: %v", err)
		}
		if !strings.Contains(string(pWS.Body), `"web_search_20250305"`) {
			t.Errorf("expected web_search_20250305 tool in Anthropic body: %s", pWS.Body)
		}
		_, err = bridge.Build(&ir.Request{
			Model:    "claude-sonnet-4",
			Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("run")}}},
			Builtins: []ir.BuiltinTool{{Kind: ir.BuiltinCodeExecution}},
		}, ir.FormatGemini, ir.FormatAnthropic, bridge.Options{})
		if err == nil {
			t.Errorf("expected error for unsupported code_execution builtin toward Anthropic")
		}

		// 5. Tool.Strict records LossToolStrict when !dia.StrictTools; ToolChoiceAllowed filters tools.
		pStrict, err := bridge.Build(&ir.Request{
			Model:    "claude-sonnet-4",
			Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("hi")}}},
			Tools: []ir.Tool{
				{Name: "keep", Kind: ir.ToolFunction, Strict: ir.Ptr(true)},
				{Name: "drop", Kind: ir.ToolFunction},
			},
			ToolChoice: ir.ToolChoice{Mode: ir.ToolChoiceAllowed, Allowed: []ir.ToolRef{{Name: "keep"}}, AllowedMode: ir.ToolChoiceRequired},
		}, ir.FormatChat, ir.FormatAnthropic, bridge.Options{})
		if err != nil {
			t.Fatalf("allowed+strict->Anthropic: %v", err)
		}
		if strings.Contains(string(pStrict.Body), `"drop"`) {
			t.Errorf("ToolChoiceAllowed did not filter disallowed tool in Anthropic: %s", pStrict.Body)
		}
		if !strings.Contains(string(pStrict.Body), `"type":"any"`) {
			t.Errorf("ToolChoiceAllowed(required) should set tool_choice.type=any: %s", pStrict.Body)
		}
		sawStrictLoss := false
		for _, l := range pStrict.Losses {
			if l.Kind == ir.LossToolStrict {
				sawStrictLoss = true
			}
		}
		if !sawStrictLoss {
			t.Errorf("expected LossToolStrict in Plan.Losses, got %+v", pStrict.Losses)
		}

		// 6. MidSystem: "fold" folds into user message as <system_message>...</system_message>.
		pFold, err := bridge.Build(&ir.Request{
			Model: "claude-sonnet-4",
			Messages: []ir.Message{
				{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("u1")}},
				{Role: ir.RoleSystem, Parts: []ir.Part{ir.TextPart("rule")}},
				{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("u2")}},
			},
		}, ir.FormatChat, ir.FormatAnthropic, bridge.Options{Anthropic: anthropic.Dialect{MidSystem: "fold"}})
		if err != nil {
			t.Fatalf("MidSystem fold: %v", err)
		}
		if !strings.Contains(string(pFold.Body), "<system_message>rule</system_message>") {
			t.Errorf("MidSystem fold dropped message: %s", pFold.Body)
		}
	})

	t.Run("responses_shorthand_messages_custom_namespace_allowed_tools_and_unsupported_fields", func(t *testing.T) {
		raw := []byte(`{
			"model": "gpt-5",
			"input": [
				{"role": "developer", "content": "be concise"},
				{"role": "user", "content": "hi"}
			],
			"tools": [
				{"type": "custom", "name": "sql", "description": "Run SQL"},
				{"type": "namespace", "name": "fs", "tools": [{"type": "function", "name": "read", "parameters": {"type": "object"}}]}
			],
			"tool_choice": {"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "read"}]}
		}`)
		req, err := responses.ParseRequest(raw, responses.ParseOptions{})
		if err != nil {
			t.Fatalf("responses.ParseRequest: %v", err)
		}
		if len(req.Instructions) != 1 || req.Instructions[0].Text != "be concise" {
			t.Errorf("leading developer shorthand message not extracted to Instructions: %+v", req.Instructions)
		}
		if len(req.Messages) != 1 || req.Messages[0].Parts[0].Text != "hi" {
			t.Errorf("shorthand user message dropped: %+v", req.Messages)
		}
		if len(req.Tools) != 2 || req.Tools[0].Kind != ir.ToolCustom || req.Tools[1].Namespace != "fs" || req.Tools[1].Name != "read" {
			t.Errorf("custom/namespace tools not parsed: %+v", req.Tools)
		}
		if req.ToolChoice.Mode != ir.ToolChoiceAllowed || req.ToolChoice.AllowedMode != ir.ToolChoiceRequired || len(req.ToolChoice.Allowed) != 1 || req.ToolChoice.Allowed[0].Name != "read" {
			t.Errorf("allowed_tools schema not parsed properly: %+v", req.ToolChoice)
		}

		// Unsupported stateful fields: conversation, background:true, item_reference, unknown content/tool type.
		for name, body := range map[string]string{
			"conversation":      `{"model":"gpt-5","conversation":"conv_1","input":"hi"}`,
			"background":        `{"model":"gpt-5","background":true,"input":"hi"}`,
			"item_reference":    `{"model":"gpt-5","input":[{"type":"item_reference","id":"item_1"}]}`,
			"unknown_part_type": `{"model":"gpt-5","input":[{"type":"message","role":"user","content":[{"type":"bogus_part"}]}]}`,
			"unknown_tool_type": `{"model":"gpt-5","input":"hi","tools":[{"type":"bogus_tool"}]}`,
		} {
			if _, err := responses.ParseRequest([]byte(body), responses.ParseOptions{}); err == nil {
				t.Errorf("%s: expected error, got nil", name)
			}
		}

		// Responses build: preserves mid-conversation developer role, omits seed/penalties/stop (with losses), fixes allowed_tools shape, rejects cross-format previous_response_id.
		pResp, err := bridge.Build(&ir.Request{
			Source: ir.FormatChat,
			Model:  "gpt-5",
			Messages: []ir.Message{
				{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("u1")}},
				{Role: ir.RoleDeveloper, Parts: []ir.Part{ir.TextPart("mid dev")}},
			},
			Seed:            ir.Ptr(int64(42)),
			PresencePenalty: ir.Ptr(0.5),
			Stop:            []string{"END"},
			ToolChoice:      ir.ToolChoice{Mode: ir.ToolChoiceAllowed, AllowedMode: ir.ToolChoiceRequired, Allowed: []ir.ToolRef{{Name: "f"}}},
		}, ir.FormatChat, ir.FormatResponses, bridge.Options{
			Allow: ir.LossSet{ir.LossSamplingPenalty: true, ir.LossStopLimit: true},
		})
		if err != nil {
			t.Fatalf("responses BuildRequest: %v", err)
		}
		bodyStr := string(pResp.Body)
		if strings.Contains(bodyStr, `"seed"`) || strings.Contains(bodyStr, `"presence_penalty"`) || strings.Contains(bodyStr, `"stop"`) {
			t.Errorf("Responses body emitted unsupported fields: %s", bodyStr)
		}
		if !strings.Contains(bodyStr, `"role":"developer"`) {
			t.Errorf("Responses downgraded mid-conversation developer role: %s", bodyStr)
		}
		if !strings.Contains(bodyStr, `"mode":"required"`) || !strings.Contains(bodyStr, `"tools":[{"name":"f","type":"function"}]`) {
			t.Errorf("Responses allowed_tools shape wrong: %s", bodyStr)
		}
		if _, err := bridge.Build(&ir.Request{
			Source:             ir.FormatChat,
			Model:              "gpt-5",
			PreviousResponseID: "resp_1",
			Messages:           []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("hi")}}},
		}, ir.FormatChat, ir.FormatResponses, bridge.Options{}); err == nil {
			t.Errorf("expected error for cross-format previous_response_id")
		}
	})

	t.Run("chat_allowed_tools_and_loss_reporting", func(t *testing.T) {
		pChat, err := bridge.Build(&ir.Request{
			Model: "gpt-4o",
			Messages: []ir.Message{{
				Role: ir.RoleUser,
				Parts: []ir.Part{{
					Type:  ir.PartText,
					Text:  "hi",
					Cache: &ir.CacheControl{Type: "ephemeral"},
				}},
			}},
			Tools: []ir.Tool{
				{Name: "a", Kind: ir.ToolFunction},
				{Name: "b", Kind: ir.ToolFunction},
			},
			ToolChoice: ir.ToolChoice{Mode: ir.ToolChoiceAllowed, AllowedMode: ir.ToolChoiceRequired, Allowed: []ir.ToolRef{{Name: "a"}}},
		}, ir.FormatAnthropic, ir.FormatChat, bridge.Options{})
		if err != nil {
			t.Fatalf("Chat build: %v", err)
		}
		if strings.Contains(string(pChat.Body), `"name":"b"`) {
			t.Errorf("Chat ToolChoiceAllowed did not filter disallowed tool b: %s", pChat.Body)
		}
		if !strings.Contains(string(pChat.Body), `"tool_choice":"required"`) {
			t.Errorf("Chat ToolChoiceAllowed(required) should emit required: %s", pChat.Body)
		}
		sawCache := false
		for _, l := range pChat.Losses {
			if l.Kind == ir.LossCacheMarker {
				sawCache = true
			}
		}
		if !sawCache {
			t.Errorf("expected LossCacheMarker in Chat Plan.Losses, got %+v", pChat.Losses)
		}
	})

	t.Run("gemini_inline_mime_classification_stop_limit_5_and_safety_settings", func(t *testing.T) {
		raw := []byte(`{
			"contents": [{"role": "user", "parts": [
				{"inlineData": {"mimeType": "application/pdf", "data": "AA=="}},
				{"inlineData": {"mimeType": "audio/mp3", "data": "BB=="}},
				{"inlineData": {"mimeType": "video/mp4", "data": "CC=="}},
				{"inlineData": {"mimeType": "image/png", "data": "DD=="}}
			]}],
			"safetySettings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}]
		}`)
		req, err := gemini.ParseRequest(raw, gemini.ParseOptions{Model: "gemini-2.5"})
		if err != nil {
			t.Fatalf("gemini.ParseRequest: %v", err)
		}
		parts := req.Messages[0].Parts
		if parts[0].Type != ir.PartFile || parts[1].Type != ir.PartAudio || parts[2].Type != ir.PartVideo || parts[3].Type != ir.PartImage {
			t.Errorf("MIME classification wrong: got [%s, %s, %s, %s]", parts[0].Type, parts[1].Type, parts[2].Type, parts[3].Type)
		}

		// Identity Gemini -> Gemini preserves safetySettings.
		built, err := gemini.BuildRequest(req, gemini.BuildOptions{})
		if err != nil {
			t.Fatalf("gemini.BuildRequest identity: %v", err)
		}
		if !strings.Contains(string(built.Body), `"safetySettings"`) {
			t.Errorf("Gemini identity build dropped safetySettings: %s", built.Body)
		}

		// Stop limit > 5 triggers LossStopLimit and truncates to 5.
		pGem, err := bridge.Build(&ir.Request{
			Model:    "gemini-2.5",
			Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("hi")}}},
			Stop:     []string{"1", "2", "3", "4", "5", "6"},
		}, ir.FormatAnthropic, ir.FormatGemini, bridge.Options{Allow: ir.LossSet{ir.LossStopLimit: true}})
		if err != nil {
			t.Fatalf("Gemini 6 stops: %v", err)
		}
		var gemOut struct {
			GenerationConfig struct {
				StopSequences []string `json:"stopSequences"`
			} `json:"generationConfig"`
		}
		_ = json.Unmarshal(pGem.Body, &gemOut)
		if len(gemOut.GenerationConfig.StopSequences) != 5 {
			t.Errorf("expected 5 truncated stopSequences, got %v", gemOut.GenerationConfig.StopSequences)
		}
		sawStopLoss := false
		for _, l := range pGem.Losses {
			if l.Kind == ir.LossStopLimit {
				sawStopLoss = true
			}
		}
		if !sawStopLoss {
			t.Errorf("expected LossStopLimit for Gemini >5 stops")
		}
	})
}
