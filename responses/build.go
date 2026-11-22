package responses

import (
	"encoding/json"
	"strings"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// BuildRequest serialises an ir.Request to Responses API JSON.
func BuildRequest(req *ir.Request, opts BuildOptions) (*Built, error) {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	env.Style = ir.StyleBracket

	nameMap := ir.BuildToolNameMap(req.Tools)
	names := nameMap.FromWire
	out := jsonx.Start(nil)
	out.Str("model", req.Model)

	// instructions from leading Instructions parts.
	if len(req.Instructions) > 0 {
		out.Str("instructions", partsToText(req.Instructions))
	}

	// input items.
	items, err := buildInputItems(req.Messages, nameMap, env)
	if err != nil {
		return nil, err
	}
	itemsBytes, err := jsonx.Marshal(items)
	if err != nil {
		return nil, err
	}
	out.Raw("input", itemsBytes)

	// tools.
	if len(req.Tools) > 0 || len(req.Builtins) > 0 {
		var rawTools []json.RawMessage
		for i, t := range req.Tools {
			fp := ir.Root.Key("tools").At(i)
			wt, err := buildTool(t, fp, env, nameMap)
			if err != nil {
				return nil, err
			}
			tb, err := jsonx.Marshal(wt)
			if err != nil {
				return nil, err
			}
			rawTools = append(rawTools, tb)
		}
		for _, bt := range req.Builtins {
			switch bt.Kind {
			case ir.BuiltinWebSearch:
				if bt.Origin == ir.FormatResponses && len(bt.Raw) > 0 {
					rawTools = append(rawTools, append(json.RawMessage(nil), bt.Raw...))
				} else if len(bt.Config) > 0 {
					m := map[string]json.RawMessage{}
					if kvs, err := jsonx.Object(bt.Config); err == nil {
						for _, kv := range kvs {
							m[kv.Key] = kv.Value
						}
					}
					m["type"] = json.RawMessage(`"web_search"`)
					tb, err := jsonx.Marshal(m)
					if err != nil {
						return nil, err
					}
					rawTools = append(rawTools, tb)
				} else {
					rawTools = append(rawTools, json.RawMessage(`{"type":"web_search"}`))
				}
			default:
				if err := env.Err(ir.Unsupported(ir.Root.Key("tools"), string(bt.Kind), "builtin tool not supported by Responses")); err != nil {
					return nil, err
				}
			}
		}
		tb, err := jsonx.Marshal(rawTools)
		if err != nil {
			return nil, err
		}
		out.Raw("tools", tb)
	}

	// tool_choice.
	if req.ToolChoice.Mode != "" {
		tcBytes, err := buildToolChoice(req.ToolChoice, nameMap, env)
		if err != nil {
			return nil, err
		}
		out.Raw("tool_choice", tcBytes)
	}

	// parallel_tool_calls.
	if req.ParallelToolCalls != nil {
		out.Bool("parallel_tool_calls", *req.ParallelToolCalls)
	}

	// Reject multiple candidates (Responses API supports a single candidate).
	if req.Candidates != nil && *req.Candidates > 1 {
		return nil, env.Err(ir.Unsupported(ir.Root.Key("n"), "candidates", "Responses API supports only a single candidate"))
	}

	// sampling.
	if req.Temperature != nil {
		out.Raw("temperature", writeFloat(*req.Temperature))
	}
	if req.TopP != nil {
		out.Raw("top_p", writeFloat(*req.TopP))
	}
	// top_k is not supported by Responses API.
	if req.TopK != nil {
		if err := env.Lose(ir.LossSamplingTopK, ir.Root.Key("top_k"), "not supported by Responses API"); err != nil {
			return nil, err
		}
	}
	if req.MaxOutputTokens != nil {
		out.Int("max_output_tokens", int64(*req.MaxOutputTokens))
	}
	// Responses API does not support seed, presence_penalty, frequency_penalty, logprobs, or stop.
	if req.Seed != nil {
		if err := env.Lose(ir.LossSamplingSeed, ir.Root.Key("seed"), "not supported by Responses API"); err != nil {
			return nil, err
		}
	}
	if req.PresencePenalty != nil && *req.PresencePenalty != 0 {
		if err := env.Lose(ir.LossSamplingPenalty, ir.Root.Key("presence_penalty"), "not supported by Responses API"); err != nil {
			return nil, err
		}
	}
	if req.FrequencyPenalty != nil && *req.FrequencyPenalty != 0 {
		if err := env.Lose(ir.LossSamplingPenalty, ir.Root.Key("frequency_penalty"), "not supported by Responses API"); err != nil {
			return nil, err
		}
	}
	if req.Logprobs != nil && *req.Logprobs {
		if err := env.Lose(ir.LossSamplingLogprobs, ir.Root.Key("logprobs"), "not supported by Responses API"); err != nil {
			return nil, err
		}
	}
	if len(req.Stop) > 0 {
		if err := env.Lose(ir.LossStopLimit, ir.Root.Key("stop"), "not supported by Responses API"); err != nil {
			return nil, err
		}
	}

	// reasoning.
	profile := env.ModelProfileOr(nil)
	if req.Reasoning != nil || (profile != nil && profile.DefaultEffort != ir.EffortUnset) {
		res := profile.Resolve(req.Reasoning)
		if res.Off && profile != nil && profile.Off.Mode == ir.OffPatch {
			fields, cfgErr := ir.ParseOffPatch(profile.Off.Patch)
			if cfgErr != nil {
				return nil, env.Err(cfgErr)
			}
			for _, f := range fields {
				out.Raw(f.Key, f.Value)
			}
		} else {
			rb, err := buildReasoning(req.Reasoning, profile, env)
			if err != nil {
				return nil, err
			}
			if rb != nil {
				out.Raw("reasoning", rb)
			}
		}
	}

	// text.format.
	if req.Output != nil {
		tfBytes, err := buildTextFormat(req.Output)
		if err != nil {
			return nil, err
		}
		if tfBytes != nil {
			out.Raw("text", tfBytes)
		}
	}

	// include.
	if len(req.Include) > 0 {
		inclBytes, _ := jsonx.Marshal(req.Include)
		out.Raw("include", inclBytes)
	}

	// store.
	if req.Store != nil {
		out.Bool("store", *req.Store)
	}

	// previous_response_id is valid only for Responses-to-Responses requests.
	if req.PreviousResponseID != "" {
		if req.Source != "" && req.Source != ir.FormatResponses {
			return nil, env.Err(ir.Unsupported(ir.Root.Key("previous_response_id"), "previous_response_id", "previous_response_id cannot be used across formats"))
		}
		out.Str("previous_response_id", req.PreviousResponseID)
	}

	// service_tier, user, metadata.
	if req.ServiceTier != "" {
		out.Str("service_tier", req.ServiceTier)
	}
	if req.User != "" {
		out.Str("user", req.User)
	}
	if len(req.Metadata) > 0 {
		out.Raw("metadata", req.Metadata)
	}

	// stream.
	if req.Stream {
		out.Bool("stream", true)
	}

	body := out.End()
	return &Built{Body: body, Model: req.Model, Stream: req.Stream, Names: names}, nil
}

func buildInputItems(msgs []ir.Message, nameMap *ir.ToolNameMap, env *ir.Env) ([]wireInputItem, error) {
	var items []wireInputItem
	for i, msg := range msgs {
		fp := ir.Root.Key("input").At(i)
		its, err := messageToItems(msg, fp, nameMap, env)
		if err != nil {
			return nil, err
		}
		items = append(items, its...)
	}
	return items, nil
}

func messageToItems(msg ir.Message, p ir.Path, nameMap *ir.ToolNameMap, env *ir.Env) ([]wireInputItem, error) {
	switch msg.Role {
	case ir.RoleAssistant:
		return assistantPartsToItems(msg.Parts, p, nameMap, env)
	case ir.RoleUser, ir.RoleSystem, ir.RoleDeveloper:
		role := string(msg.Role)
		contentParts, toolResults, err := userPartsToContent(msg.Parts, p, env)
		if err != nil {
			return nil, err
		}
		var items []wireInputItem
		if len(contentParts) > 0 {
			contentBytes, err := jsonx.Marshal(contentParts)
			if err != nil {
				return nil, err
			}
			items = append(items, wireInputItem{
				Type:    "message",
				Role:    role,
				Content: contentBytes,
				Name:    msg.Name,
			})
		}
		items = append(items, toolResults...)
		return items, nil
	}
	return nil, env.Err(ir.Invalid(p.Key("role"), "unknown role: "+string(msg.Role)))
}

func userPartsToContent(parts []ir.Part, p ir.Path, env *ir.Env) ([]wireContentPart, []wireInputItem, error) {
	var content []wireContentPart
	var results []wireInputItem
	for i, part := range parts {
		fp := p.Key("parts").At(i)
		if part.Cache != nil {
			if err := env.Lose(ir.LossCacheMarker, fp.Key("cache_control"), "prompt cache markers not supported by Responses API"); err != nil {
				return nil, nil, err
			}
		}
		switch part.Type {
		case ir.PartText:
			content = append(content, wireContentPart{Type: "input_text", Text: part.Text})
		case ir.PartRefusal:
			content = append(content, wireContentPart{Type: "input_text", Text: part.Text})
		case ir.PartImage:
			if part.Media == nil {
				break
			}
			imageURL := part.Media.URL
			if imageURL == "" && part.Media.Data != "" {
				imageURL = "data:" + part.Media.MediaType + ";base64," + part.Media.Data
			}
			content = append(content, wireContentPart{
				Type:     "input_image",
				ImageURL: imageURL,
				Detail:   part.Media.Detail,
			})
		case ir.PartFile:
			if part.Media == nil {
				break
			}
			content = append(content, wireContentPart{
				Type:     "input_file",
				FileID:   part.Media.FileID,
				Filename: part.Media.Filename,
				FileData: part.Media.Data,
			})
		case ir.PartToolResult:
			if part.Result == nil {
				break
			}
			r := part.Result
			outRaw, err := buildToolOutputValue(r, fp, env)
			if err != nil {
				return nil, nil, err
			}
			switch r.Kind {
			case ir.ToolCustom:
				item := wireInputItem{
					Type:   "custom_tool_call_output",
					CallID: r.CallID,
					Output: outRaw,
				}
				results = append(results, item)
			default:
				item := wireInputItem{
					Type:   "function_call_output",
					CallID: r.CallID,
					Output: outRaw,
				}
				results = append(results, item)
			}
		default:
			if err := env.Err(ir.Unsupported(fp, string(part.Type), "part type not supported in Responses user message")); err != nil {
				return nil, nil, err
			}
		}
	}
	return content, results, nil
}

func assistantPartsToItems(parts []ir.Part, p ir.Path, nameMap *ir.ToolNameMap, env *ir.Env) ([]wireInputItem, error) {
	var items []wireInputItem
	var textParts []wireContentPart
	flushText := func() {
		if len(textParts) > 0 {
			contentBytes, _ := jsonx.Marshal(textParts)
			items = append(items, wireInputItem{
				Type:    "message",
				Role:    "assistant",
				Content: contentBytes,
			})
			textParts = nil
		}
	}

	for i, part := range parts {
		fp := p.Key("parts").At(i)
		if part.Cache != nil {
			if err := env.Lose(ir.LossCacheMarker, fp.Key("cache_control"), "prompt cache markers not supported by Responses API"); err != nil {
				return nil, err
			}
		}
		switch part.Type {
		case ir.PartText:
			textParts = append(textParts, wireContentPart{Type: "output_text", Text: part.Text})
		case ir.PartRefusal:
			textParts = append(textParts, wireContentPart{Type: "refusal", Refusal: part.Text})

		case ir.PartReasoning:
			flushText()
			if part.Reasoning == nil {
				break
			}
			r := part.Reasoning
			item := wireInputItem{Type: "reasoning", ID: r.ID}
			hasContent := false
			if r.Sealed != nil {
				if r.Sealed.Kind == ir.SealedResponsesEncrypted && r.Sealed.Origin.Matches(ir.FormatResponses, env.ProviderID) {
					var enc string
					_ = json.Unmarshal(r.Sealed.Data, &enc)
					if enc != "" {
						item.EncryptedContent = enc
						hasContent = true
					}
				} else {
					if err := env.Lose(ir.LossReasoningSealed, fp, "foreign sealed reasoning block dropped for Responses"); err != nil {
						return nil, err
					}
				}
			}
			if r.Signature != nil && !r.Signature.Origin.Matches(ir.FormatResponses, env.ProviderID) {
				if err := env.Lose(ir.LossReasoningSignature, fp, "foreign reasoning signature dropped for Responses"); err != nil {
					return nil, err
				}
			}
			if len(r.Summary) > 0 {
				sp := make([]map[string]string, 0, len(r.Summary))
				for _, s := range r.Summary {
					sp = append(sp, map[string]string{"type": "summary_text", "text": s})
				}
				summaryBytes, _ := jsonx.Marshal(sp)
				item.Summary = summaryBytes
				hasContent = true
			} else if r.Text != "" {
				summaryBytes, _ := jsonx.Marshal([]map[string]string{{"type": "summary_text", "text": r.Text}})
				item.Summary = summaryBytes
				hasContent = true
			}
			if hasContent {
				items = append(items, item)
			}

		case ir.PartToolCall:
			flushText()
			if part.Call == nil {
				break
			}
			c := part.Call
			switch c.Kind {
			case ir.ToolCustom:
				input := c.Input
				if input == "" {
					// Extract from arguments JSON.
					var m map[string]string
					if json.Unmarshal(c.Arguments, &m) == nil {
						input = m["input"]
					}
				}
				items = append(items, wireInputItem{
					Type:   "custom_tool_call",
					CallID: c.ID,
					Name:   nameMap.WireName(c.Namespace, c.Name),
					Input:  input,
				})
			default:
				argsStr := string(c.Arguments)
				if argsStr == "" {
					argsStr = "{}"
				}
				items = append(items, wireInputItem{
					Type:      "function_call",
					CallID:    c.ID,
					Name:      nameMap.WireName(c.Namespace, c.Name),
					Arguments: argsStr,
				})
			}

		default:
			if err := env.Err(ir.Unsupported(fp, string(part.Type), "part type not supported in Responses assistant message")); err != nil {
				return nil, err
			}
		}
	}
	flushText()
	return items, nil
}

func buildTool(t ir.Tool, p ir.Path, env *ir.Env, nameMap *ir.ToolNameMap) (wireTool, error) {
	if t.Name == "" {
		return wireTool{}, env.Err(ir.Invalid(p.Key("name"), "tool name is required"))
	}
	if t.Cache != nil {
		if err := env.Lose(ir.LossCacheMarker, p.Key("cache_control"), "tool cache markers not supported by Responses API"); err != nil {
			return wireTool{}, err
		}
	}
	wireName := nameMap.WireName(t.Namespace, t.Name)
	desc := nameMap.Description(t.Namespace, t.Name, t.Description)
	if t.Kind == ir.ToolCustom {
		wt := wireTool{
			Type:        "custom",
			Name:        wireName,
			Description: desc,
		}
		if t.Grammar != nil && (t.Grammar.Type != "" || t.Grammar.Syntax != "" || t.Grammar.Definition != "") {
			gt := t.Grammar.Type
			if gt == "" {
				gt = "grammar"
			}
			wt.Format = &wireCustomFormat{
				Type:       gt,
				Syntax:     t.Grammar.Syntax,
				Definition: t.Grammar.Definition,
			}
		}
		return wt, nil
	}
	wt := wireTool{
		Type:        "function",
		Name:        wireName,
		Description: desc,
	}
	if len(t.Schema) > 0 {
		wt.Parameters = t.Schema
	}
	if t.Strict != nil {
		wt.Strict = t.Strict
	}
	return wt, nil
}

func buildToolChoice(tc ir.ToolChoice, nameMap *ir.ToolNameMap, env *ir.Env) (json.RawMessage, error) {
	switch tc.Mode {
	case ir.ToolChoiceAuto:
		return json.RawMessage(`"auto"`), nil
	case ir.ToolChoiceNone:
		return json.RawMessage(`"none"`), nil
	case ir.ToolChoiceRequired:
		return json.RawMessage(`"required"`), nil
	case ir.ToolChoiceTool:
		b, _ := jsonx.Marshal(map[string]any{"type": "function", "name": nameMap.WireName(tc.Namespace, tc.Name)})
		return b, nil
	case ir.ToolChoiceAllowed:
		mode := "auto"
		if tc.AllowedMode == ir.ToolChoiceRequired {
			mode = "required"
		}
		tools := make([]map[string]any, 0, len(tc.Allowed))
		for _, ref := range tc.Allowed {
			typ := "function"
			if ref.Kind == ir.ToolCustom {
				typ = "custom"
			}
			wireName := nameMap.WireName(ref.Namespace, ref.Name)
			entry := map[string]any{"type": typ, "name": wireName}
			tools = append(tools, entry)
		}
		b, _ := jsonx.Marshal(map[string]any{"type": "allowed_tools", "mode": mode, "tools": tools})
		return b, nil
	}
	return nil, env.Err(ir.Unsupported(ir.Root.Key("tool_choice"), string(tc.Mode), "tool choice mode"))
}

func buildReasoning(rc *ir.ReasoningConfig, profile *ir.ModelProfile, env *ir.Env) (json.RawMessage, error) {
	res := profile.Resolve(rc)
	m := map[string]any{}
	if res.Off {
		if profile != nil {
			switch profile.Off.Mode {
			case ir.OffUnsupported:
				return nil, env.Err(ir.Unsupported(ir.Root.Key("reasoning"), "off", "model does not allow disabling reasoning"))
			case ir.OffLowestHidden:
				m["effort"] = string(profile.Lowest())
			default:
				m["effort"] = string(ir.EffortNone)
			}
		} else {
			m["effort"] = string(ir.EffortNone)
		}
	} else {
		if profile != nil && profile.Style == ir.StyleNone {
			return nil, nil
		}
		if res.Effort != ir.EffortUnset {
			m["effort"] = string(res.Effort)
		}
	}
	if rc != nil && rc.Summary != "" {
		m["summary"] = rc.Summary
	}
	if len(m) == 0 {
		return nil, nil
	}
	b, _ := jsonx.Marshal(m)
	return b, nil
}

func buildTextFormat(of *ir.OutputFormat) (json.RawMessage, error) {
	if of == nil {
		return nil, nil
	}
	var fmt map[string]any
	switch of.Type {
	case ir.OutputText:
		fmt = map[string]any{"type": "text"}
	case ir.OutputJSONObject:
		fmt = map[string]any{"type": "json_object"}
	case ir.OutputJSONSchema:
		name := of.Name
		if name == "" {
			name = "output"
		}
		m := map[string]any{"type": "json_schema", "name": name}
		if len(of.Schema) > 0 {
			m["schema"] = json.RawMessage(of.Schema)
		}
		if of.Strict != nil {
			m["strict"] = *of.Strict
		}
		fmt = m
	default:
		return nil, nil
	}
	b, _ := jsonx.Marshal(map[string]any{"format": fmt})
	return b, nil
}

func buildToolOutputValue(r *ir.ToolResult, fp ir.Path, env *ir.Env) (json.RawMessage, error) {
	hasNonText := false
	for _, cp := range r.Content {
		if cp.Type != ir.PartText {
			hasNonText = true
			break
		}
	}
	if !hasNonText {
		text := toolResultText(r.Content)
		if r.IsError && !strings.HasPrefix(text, "[tool error] ") {
			text = "[tool error] " + text
		}
		b, err := jsonx.Marshal(text)
		return b, err
	}
	var cparts []wireContentPart
	if r.IsError {
		cparts = append(cparts, wireContentPart{Type: "input_text", Text: "[tool error]"})
	}
	for i, cp := range r.Content {
		cfp := fp.Key("output").At(i)
		switch cp.Type {
		case ir.PartText:
			cparts = append(cparts, wireContentPart{Type: "input_text", Text: cp.Text})
		case ir.PartImage:
			if cp.Media == nil {
				continue
			}
			imageURL := cp.Media.URL
			if imageURL == "" && cp.Media.Data != "" {
				mt := cp.Media.MediaType
				if mt == "" {
					mt = "image/png"
				}
				imageURL = "data:" + mt + ";base64," + cp.Media.Data
			}
			cparts = append(cparts, wireContentPart{
				Type:     "input_image",
				ImageURL: imageURL,
				Detail:   cp.Media.Detail,
			})
		case ir.PartFile:
			if cp.Media == nil {
				continue
			}
			if cp.Media.URL != "" && cp.Media.FileID == "" && cp.Media.Data == "" {
				return nil, env.Err(ir.Unsupported(cfp, "file.url", "Responses input_file in tool output does not support URLs"))
			}
			cparts = append(cparts, wireContentPart{
				Type:     "input_file",
				FileID:   cp.Media.FileID,
				Filename: cp.Media.Filename,
				FileData: cp.Media.Data,
			})
		default:
			return nil, env.Err(ir.Unsupported(cfp, string(cp.Type), "tool result part type not supported by Responses"))
		}
	}
	b, err := jsonx.Marshal(cparts)
	return b, err
}

func toolResultText(parts []ir.Part) string {
	var sb strings.Builder
	for _, p := range parts {
		if p.Type == ir.PartText {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}

func partsToText(parts []ir.Part) string {
	var sb strings.Builder
	for _, p := range parts {
		if p.Type == ir.PartText {
			if sb.Len() > 0 {
				sb.WriteString("\n\n")
			}
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}

func writeFloat(f float64) json.RawMessage {
	b, _ := json.Marshal(f)
	return b
}
