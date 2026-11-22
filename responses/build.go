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

	names := map[string]ir.ToolRef{}
	out := jsonx.Start(nil)
	out.Str("model", req.Model)

	// instructions from leading Instructions parts.
	if len(req.Instructions) > 0 {
		out.Str("instructions", partsToText(req.Instructions))
	}

	// input items.
	items, err := buildInputItems(req.Messages, names, env)
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
		var wireTools []wireTool
		for i, t := range req.Tools {
			fp := ir.Root.Key("tools").At(i)
			wt, err := buildTool(t, fp, env)
			if err != nil {
				return nil, err
			}
			names[t.Name] = ir.ToolRef{Name: t.Name, Namespace: t.Namespace, Kind: t.Kind}
			wireTools = append(wireTools, wt)
		}
		for _, bt := range req.Builtins {
			switch bt.Kind {
			case ir.BuiltinWebSearch:
				wt := wireTool{Type: "web_search"}
				if len(bt.Config) > 0 {
					wt.WebSearchOptions = bt.Config
				}
				wireTools = append(wireTools, wt)
			default:
				if err := env.Err(ir.Unsupported(ir.Root.Key("tools"), string(bt.Kind), "builtin tool not supported by Responses")); err != nil {
					return nil, err
				}
			}
		}
		tb, err := jsonx.Marshal(wireTools)
		if err != nil {
			return nil, err
		}
		out.Raw("tools", tb)
	}

	// tool_choice.
	if req.ToolChoice.Mode != "" {
		tcBytes, err := buildToolChoice(req.ToolChoice, env)
		if err != nil {
			return nil, err
		}
		out.Raw("tool_choice", tcBytes)
	}

	// parallel_tool_calls.
	if req.ParallelToolCalls != nil {
		out.Bool("parallel_tool_calls", *req.ParallelToolCalls)
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
	if req.Seed != nil {
		out.Int("seed", *req.Seed)
	}
	if req.PresencePenalty != nil {
		out.Raw("presence_penalty", writeFloat(*req.PresencePenalty))
	}
	if req.FrequencyPenalty != nil {
		out.Raw("frequency_penalty", writeFloat(*req.FrequencyPenalty))
	}
	// logprobs is not supported by Responses API.
	if req.Logprobs != nil && *req.Logprobs {
		if err := env.Lose(ir.LossSamplingLogprobs, ir.Root.Key("logprobs"), "not supported by Responses API"); err != nil {
			return nil, err
		}
	}

	// stop.
	if len(req.Stop) == 1 {
		out.Str("stop", req.Stop[0])
	} else if len(req.Stop) > 1 {
		stopBytes, _ := jsonx.Marshal(req.Stop)
		out.Raw("stop", stopBytes)
	}

	// reasoning.
	profile := env.ModelProfileOr(nil)
	if req.Reasoning != nil || (profile != nil && profile.DefaultEffort != ir.EffortUnset) {
		rb, err := buildReasoning(req.Reasoning, profile, env)
		if err != nil {
			return nil, err
		}
		if rb != nil {
			out.Raw("reasoning", rb)
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

	// previous_response_id.
	if req.PreviousResponseID != "" {
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

func buildInputItems(msgs []ir.Message, names map[string]ir.ToolRef, env *ir.Env) ([]wireInputItem, error) {
	var items []wireInputItem
	for i, msg := range msgs {
		fp := ir.Root.Key("input").At(i)
		its, err := messageToItems(msg, fp, env)
		if err != nil {
			return nil, err
		}
		items = append(items, its...)
	}
	return items, nil
}

func messageToItems(msg ir.Message, p ir.Path, env *ir.Env) ([]wireInputItem, error) {
	switch msg.Role {
	case ir.RoleAssistant:
		return assistantPartsToItems(msg.Parts, p, env)
	case ir.RoleUser, ir.RoleSystem, ir.RoleDeveloper:
		// System and developer mid-conversation become user messages.
		role := "user"
		if msg.Role == ir.RoleAssistant {
			role = "assistant"
		}
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
			text := toolResultText(r.Content)
			isErr := r.IsError
			switch r.Kind {
			case ir.ToolCustom:
				item := wireInputItem{
					Type:   "custom_tool_call_output",
					CallID: r.CallID,
					Output: text,
				}
				if isErr {
					t := true
					item.IsError = &t
				}
				results = append(results, item)
			default:
				item := wireInputItem{
					Type:   "function_call_output",
					CallID: r.CallID,
					Output: text,
				}
				if isErr {
					t := true
					item.IsError = &t
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

func assistantPartsToItems(parts []ir.Part, p ir.Path, env *ir.Env) ([]wireInputItem, error) {
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
		switch part.Type {
		case ir.PartText, ir.PartRefusal:
			textParts = append(textParts, wireContentPart{Type: "output_text", Text: part.Text})

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
					Type:      "custom_tool_call",
					CallID:    c.ID,
					Name:      c.Name,
					Namespace: c.Namespace,
					Input:     input,
				})
			default:
				argsStr := string(c.Arguments)
				if argsStr == "" {
					argsStr = "{}"
				}
				items = append(items, wireInputItem{
					Type:      "function_call",
					CallID:    c.ID,
					Name:      c.Name,
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

func buildTool(t ir.Tool, p ir.Path, env *ir.Env) (wireTool, error) {
	if t.Name == "" {
		return wireTool{}, env.Err(ir.Invalid(p.Key("name"), "tool name is required"))
	}
	wt := wireTool{
		Type:        "function",
		Name:        t.Name,
		Description: t.Description,
	}
	if len(t.Schema) > 0 {
		wt.Parameters = t.Schema
	}
	if t.Strict != nil {
		wt.Strict = t.Strict
	}
	return wt, nil
}

func buildToolChoice(tc ir.ToolChoice, env *ir.Env) (json.RawMessage, error) {
	switch tc.Mode {
	case ir.ToolChoiceAuto:
		return json.RawMessage(`"auto"`), nil
	case ir.ToolChoiceNone:
		return json.RawMessage(`"none"`), nil
	case ir.ToolChoiceRequired:
		return json.RawMessage(`"required"`), nil
	case ir.ToolChoiceTool:
		b, _ := jsonx.Marshal(map[string]any{"type": "function", "name": tc.Name})
		return b, nil
	case ir.ToolChoiceAllowed:
		names := make([]string, len(tc.Allowed))
		for i, ref := range tc.Allowed {
			names[i] = ref.Name
		}
		b, _ := jsonx.Marshal(map[string]any{"type": "allowed_tools", "allowed_tools": names})
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
		m := map[string]any{"type": "json_schema"}
		if of.Name != "" {
			m["name"] = of.Name
		}
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
