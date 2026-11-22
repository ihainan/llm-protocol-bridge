package chat

import (
	"encoding/json"
	"sort"
	"strings"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// BuildRequest serialises an ir.Request to Chat Completions JSON.
func BuildRequest(req *ir.Request, opts BuildOptions) (*Built, error) {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	env.Style = ir.StyleBracket

	names := map[string]ir.ToolRef{}
	b := &built{
		env:    env,
		dia:    opts.Dialect,
		names:  names,
		losses: nil,
	}

	// 1. Tools (filter when ToolChoiceAllowed is set)
	if len(req.Builtins) > 0 {
		return nil, b.env.Err(ir.Unsupported(ir.Root.Key("tools"), string(req.Builtins[0].Kind), "hosted builtin tools not supported by Chat Completions"))
	}
	var allowedSet map[string]bool
	if req.ToolChoice.Mode == ir.ToolChoiceAllowed {
		allowedSet = map[string]bool{}
		for _, ref := range req.ToolChoice.Allowed {
			allowedSet[ref.Namespace+"\x00"+ref.Name] = true
		}
	}
	var wireTools []wireTool
	if len(req.Tools) > 0 {
		wireTools = make([]wireTool, 0, len(req.Tools))
		for i, t := range req.Tools {
			if allowedSet != nil && !allowedSet[t.Namespace+"\x00"+t.Name] {
				continue
			}
			fp := ir.Root.Key("tools").At(i)
			wt, err := b.buildTool(t, fp)
			if err != nil {
				return nil, err
			}
			wireTools = append(wireTools, wt)
		}
	}

	// 2. Messages: instructions first, then the conversation.
	var messages []wireMessage
	if len(req.Instructions) > 0 {
		for i, pt := range req.Instructions {
			if pt.Cache != nil {
				if err := b.env.Lose(ir.LossCacheMarker, ir.Root.Key("system").At(i).Key("cache_control"), "prompt cache markers not supported by Chat Completions"); err != nil {
					return nil, err
				}
			}
		}
		role := "system"
		if b.dia.DeveloperRole {
			role = "developer"
		}
		messages = append(messages, wireMessage{
			Role:    role,
			Content: mustMarshalTextParts(req.Instructions),
		})
	}
	for i, m := range req.Messages {
		fp := ir.Root.Key("messages").At(i)
		wms, err := b.buildMessage(m, fp)
		if err != nil {
			return nil, err
		}
		messages = append(messages, wms...)
	}

	// 3. Assemble top-level object using jsonx for field order control.
	out := jsonx.Start(nil)
	out.Str("model", req.Model)

	// messages
	msgsBytes, err := jsonx.Marshal(messages)
	if err != nil {
		return nil, err
	}
	out.Raw("messages", msgsBytes)

	// tools
	if len(wireTools) > 0 {
		toolsBytes, err := jsonx.Marshal(wireTools)
		if err != nil {
			return nil, err
		}
		out.Raw("tools", toolsBytes)
	}

	// tool_choice
	if req.ToolChoice.Mode != "" {
		tcBytes, err := buildToolChoice(req.ToolChoice, b)
		if err != nil {
			return nil, err
		}
		out.Raw("tool_choice", tcBytes)
	}

	// parallel_tool_calls
	if req.ParallelToolCalls != nil {
		out.Bool("parallel_tool_calls", *req.ParallelToolCalls)
	}

	// max_tokens / max_completion_tokens
	if req.MaxOutputTokens != nil {
		field := "max_tokens"
		if b.dia.MaxTokensField == "max_completion_tokens" {
			field = "max_completion_tokens"
		}
		out.Int(field, int64(*req.MaxOutputTokens))
	}

	// sampling
	if req.Temperature != nil {
		writeNum(&out, "temperature", *req.Temperature)
	}
	if req.TopP != nil {
		writeNum(&out, "top_p", *req.TopP)
	}
	if req.Candidates != nil && *req.Candidates > 1 {
		out.Int("n", int64(*req.Candidates))
	}
	if req.PresencePenalty != nil && *req.PresencePenalty != 0 {
		writeNum(&out, "presence_penalty", *req.PresencePenalty)
	}
	if req.FrequencyPenalty != nil && *req.FrequencyPenalty != 0 {
		writeNum(&out, "frequency_penalty", *req.FrequencyPenalty)
	}
	if req.Logprobs != nil && *req.Logprobs {
		out.Bool("logprobs", true)
		if req.TopLogprobs != nil && *req.TopLogprobs > 0 {
			out.Int("top_logprobs", int64(*req.TopLogprobs))
		}
	}
	if req.Seed != nil {
		out.Int("seed", *req.Seed)
	}

	// top_k is not supported by Chat format.
	if req.TopK != nil {
		if err := b.env.Lose(ir.LossSamplingTopK, ir.Root.Key("top_k"), "not supported by Chat Completions"); err != nil {
			return nil, err
		}
	}

	// stop — OpenAI accepts at most 4 stop sequences.
	const chatMaxStop = 4
	if len(req.Stop) > 0 {
		stops := req.Stop
		if len(stops) > chatMaxStop {
			if err := b.env.Lose(ir.LossStopLimit, ir.Root.Key("stop"), "Chat Completions accepts at most 4 stop sequences; extras truncated"); err != nil {
				return nil, err
			}
			stops = stops[:chatMaxStop]
		}
		if len(stops) == 1 {
			out.Str("stop", stops[0])
		} else {
			sBytes, _ := jsonx.Marshal(stops)
			out.Raw("stop", sBytes)
		}
	}

	// stream
	if req.Stream {
		out.Bool("stream", true)
		if b.dia.StreamUsage || true { // always include_usage when streaming
			soBytes, _ := jsonx.Marshal(wireStreamOptions{IncludeUsage: true})
			out.Raw("stream_options", soBytes)
		}
	}

	// response_format
	if req.Output != nil {
		rfBytes, err := buildResponseFormat(req.Output)
		if err != nil {
			return nil, err
		}
		if rfBytes != nil {
			out.Raw("response_format", rfBytes)
		}
	}

	// reasoning
	if err := b.buildReasoning(req, &out); err != nil {
		return nil, err
	}

	// metadata
	if req.User != "" {
		out.Str("user", req.User)
	}
	if len(req.Metadata) > 0 {
		out.Raw("metadata", req.Metadata)
	}
	if req.ServiceTier != "" {
		out.Str("service_tier", req.ServiceTier)
	}
	if req.PromptCacheKey != "" {
		out.Str("prompt_cache_key", req.PromptCacheKey)
	}
	if req.Store != nil {
		if err := b.env.Lose(ir.LossRequestStore, ir.Root.Key("store"), "not supported by Chat Completions"); err != nil {
			return nil, err
		}
	}
	if len(req.Include) > 0 {
		if err := b.env.Lose(ir.LossRequestInclude, ir.Root.Key("include"), "not supported by Chat Completions"); err != nil {
			return nil, err
		}
	}

	// passthrough (identity builds only, deterministic key order)
	if req.Source == ir.FormatChat && len(req.Passthrough) > 0 {
		pkeys := make([]string, 0, len(req.Passthrough))
		for k := range req.Passthrough {
			pkeys = append(pkeys, k)
		}
		sort.Strings(pkeys)
		for _, k := range pkeys {
			out.Raw(k, req.Passthrough[k])
		}
	}

	body := out.End()
	return &Built{
		Body:   body,
		Model:  req.Model,
		Stream: req.Stream,
		Names:  names,
	}, nil
}

// built is the state used during one BuildRequest call.
type built struct {
	env    *ir.Env
	dia    Dialect
	names  map[string]ir.ToolRef
	losses []ir.Loss
}

var customToolSchema = json.RawMessage(`{"type":"object","properties":{"input":{"type":"string"}},"required":["input"],"additionalProperties":false}`)

func flattenToolName(ns, name string) string {
	if ns != "" {
		return ns + "__" + name
	}
	return name
}

// buildTool converts an ir.Tool to wireTool and records the name mapping.
func (b *built) buildTool(t ir.Tool, p ir.Path) (wireTool, error) {
	if t.Name == "" {
		return wireTool{}, b.env.Err(ir.Invalid(p.Key("name"), "tool name is required"))
	}
	kind := t.Kind
	if kind == "" {
		kind = ir.ToolFunction
	}
	if kind != ir.ToolFunction && kind != ir.ToolCustom {
		return wireTool{}, b.env.Err(ir.Invalid(p.Key("kind"), "only function and custom tools are supported"))
	}
	wireName := flattenToolName(t.Namespace, t.Name)
	// Record the name mapping (sanitised name -> original ir.ToolRef).
	ref := ir.ToolRef{Name: t.Name, Namespace: t.Namespace, Kind: kind}
	b.names[wireName] = ref

	if t.Cache != nil {
		if err := b.env.Lose(ir.LossCacheMarker, p.Key("cache_control"), "tool cache markers not supported by Chat Completions"); err != nil {
			return wireTool{}, err
		}
	}
	params := t.Schema
	if kind == ir.ToolCustom {
		params = customToolSchema
	}
	fn := wireFunctionDef{
		Name:        wireName,
		Description: t.Description,
		Parameters:  params,
	}
	if t.Strict != nil && (b.dia.StrictTools || *t.Strict) {
		fn.Strict = t.Strict
	}
	return wireTool{
		Type:     "function",
		Function: fn,
	}, nil
}

// buildMessage converts an ir.Message to one or more wireMessages.
// A user message that mixes tool results with other content is split into
// separate tool-role messages followed by a user-role message.
func (b *built) buildMessage(m ir.Message, p ir.Path) ([]wireMessage, error) {
	wm := wireMessage{Role: string(m.Role), Name: m.Name}
	// Map IR roles back to Chat roles.
	switch m.Role {
	case ir.RoleSystem:
		wm.Role = "system"
	case ir.RoleDeveloper:
		if b.dia.DeveloperRole {
			wm.Role = "developer"
		} else {
			wm.Role = "system"
		}
	case ir.RoleUser:
		wm.Role = "user"
	case ir.RoleAssistant:
		wm.Role = "assistant"
	}

	// Split parts into content parts and tool calls.
	var contentParts []wireContentPart
	var toolCalls []wireToolCall
	var reasoningText string
	var reasoningSignature string
	var refusalText string

	// Collect tool result messages, lifted tool-result images, and remaining (non-result) parts.
	var toolMsgs []wireMessage
	var liftedImages []wireContentPart

	for i, part := range m.Parts {
		fp := p.Key("parts").At(i)
		if part.Cache != nil {
			if err := b.env.Lose(ir.LossCacheMarker, fp.Key("cache_control"), "prompt cache markers not supported by Chat Completions"); err != nil {
				return nil, err
			}
		}
		switch part.Type {
		case ir.PartText:
			contentParts = append(contentParts, wireContentPart{Type: "text", Text: part.Text})

		case ir.PartRefusal:
			refusalText += part.Text
			contentParts = append(contentParts, wireContentPart{Type: "refusal", Refusal: part.Text})

		case ir.PartImage:
			if part.Media == nil {
				continue
			}
			contentParts = append(contentParts, buildImageContentPart(part.Media))

		case ir.PartAudio:
			if part.Media == nil {
				continue
			}
			format := strings.TrimPrefix(part.Media.MediaType, "audio/")
			if format == "" {
				format = "mp3"
			}
			contentParts = append(contentParts, wireContentPart{
				Type:       "input_audio",
				InputAudio: &wireAudio{Data: part.Media.Data, Format: format},
			})

		case ir.PartFile:
			if part.Media == nil {
				continue
			}
			contentParts = append(contentParts, wireContentPart{
				Type: "file",
				File: &wireFilePart{
					FileID:   part.Media.FileID,
					Filename: part.Media.Filename,
					FileData: part.Media.Data,
				},
			})

		case ir.PartReasoning:
			if part.Reasoning != nil {
				if part.Reasoning.Sealed != nil {
					if err := b.env.Lose(ir.LossReasoningSealed, fp, "sealed reasoning block not supported by Chat Completions"); err != nil {
						return nil, err
					}
				}
				if part.Reasoning.Signature != nil && !part.Reasoning.Signature.Origin.Matches(ir.FormatChat, b.env.ProviderID) {
					if err := b.env.Lose(ir.LossReasoningSignature, fp, "reasoning signature not supported by Chat Completions"); err != nil {
						return nil, err
					}
				}
				reasoningText += part.Reasoning.Text
			}
			if part.Signature != nil {
				reasoningSignature = part.Signature.Value
			}

		case ir.PartToolCall:
			if part.Call == nil {
				continue
			}
			args := string(part.Call.Arguments)
			if part.Call.Kind == ir.ToolCustom && args == "" {
				bArgs, _ := jsonx.Marshal(map[string]string{"input": part.Call.Input})
				args = string(bArgs)
			}
			if strings.TrimSpace(args) == "" {
				args = "{}"
			}
			tc := wireToolCall{
				ID:   part.Call.ID,
				Type: "function",
				Function: wireFunctionCall{
					Name:      flattenToolName(part.Call.Namespace, part.Call.Name),
					Arguments: args,
				},
			}
			toolCalls = append(toolCalls, tc)

		case ir.PartToolResult:
			// Each tool result becomes its own tool-role message.
			if part.Result == nil {
				continue
			}
			resultText, imgs := buildToolResultContent(part.Result)
			toolMsgs = append(toolMsgs, wireMessage{
				Role:       "tool",
				ToolCallID: part.Result.CallID,
				Content:    jsonRawString(resultText),
			})
			if len(imgs) > 0 {
				liftedImages = append(liftedImages, wireContentPart{
					Type: "text",
					Text: "[images returned by tool call " + part.Result.CallID + "]",
				})
				liftedImages = append(liftedImages, imgs...)
			}

		case ir.PartBuiltinCall, ir.PartBuiltinResult:
			_ = fp
		}
	}

	if len(liftedImages) > 0 {
		contentParts = append(liftedImages, contentParts...)
	}

	// If the message was entirely tool results (with no lifted images), return just those.
	if len(contentParts) == 0 && len(toolCalls) == 0 && reasoningText == "" && refusalText == "" {
		if len(toolMsgs) > 0 {
			return toolMsgs, nil
		}
		return []wireMessage{wm}, nil
	}

	// Assemble non-result content. A single text part is emitted as a plain
	// string (OpenAI accepts both, but string is the conventional form).
	if len(contentParts) == 1 && contentParts[0].Type == "text" {
		wm.Content = jsonRawString(contentParts[0].Text)
	} else if len(contentParts) > 0 {
		cb, err := jsonx.Marshal(contentParts)
		if err != nil {
			return nil, err
		}
		wm.Content = cb
	}

	if reasoningText != "" {
		if b.dia.ReasoningParam == "reasoning_content" || b.dia.ReasoningParam == "" {
			wm.ReasoningContent = reasoningText
		} else {
			wm.Reasoning = reasoningText
		}
		_ = reasoningSignature
	}

	if len(toolCalls) > 0 {
		wm.ToolCalls = toolCalls
	}

	// Tool result messages (if any) come before the remaining user content.
	out := make([]wireMessage, 0, len(toolMsgs)+1)
	out = append(out, toolMsgs...)
	out = append(out, wm)
	return out, nil
}

// buildReasoning writes reasoning configuration based on ModelProfile and dialect.
func (b *built) buildReasoning(req *ir.Request, out *jsonx.Obj) error {
	profile := b.env.ModelProfileOr(nil)
	if req.Reasoning == nil && (profile == nil || profile.DefaultEffort == ir.EffortUnset) {
		return nil
	}
	res := profile.Resolve(req.Reasoning)
	if res.Off {
		if profile != nil {
			switch profile.Off.Mode {
			case ir.OffUnsupported:
				return b.env.Err(ir.Unsupported(ir.Root.Key("reasoning"), "off", "model does not allow disabling reasoning"))
			case ir.OffPatch:
				if len(profile.Off.Patch) > 0 {
					kvs, err := jsonx.Object(profile.Off.Patch)
					if err == nil {
						for _, kv := range kvs {
							out.Raw(kv.Key, kv.Value)
						}
					}
				}
				return nil
			case ir.OffLowestHidden:
				low := profile.Lowest()
				out.Str("reasoning_effort", string(low))
				return nil
			}
		}
		switch b.dia.ReasoningParam {
		case "", "reasoning_effort":
			out.Str("reasoning_effort", string(ir.EffortNone))
		case "enable_thinking":
			out.Bool("enable_thinking", false)
		case "thinking":
			out.Raw("thinking", jsonRaw(`{"type":"disabled"}`))
		}
		return nil
	}
	if !res.Asked && res.Effort == ir.EffortUnset && res.Budget == nil {
		return nil
	}
	if profile != nil && profile.Style == ir.StyleNone {
		return nil
	}
	effort := res.Effort
	budget := 0
	if res.Budget != nil {
		budget = *res.Budget
	} else if effort != ir.EffortUnset {
		budget = profile.EffortBudget(effort)
	}
	switch b.dia.ReasoningParam {
	case "", "reasoning_effort":
		if effort != ir.EffortUnset {
			out.Str("reasoning_effort", string(effort))
		}
	case "enable_thinking":
		out.Bool("enable_thinking", true)
		if budget > 0 {
			out.Int("budget_tokens", int64(budget))
		}
	case "thinking":
		t := wireThinking{Type: "enabled", BudgetTokens: budget}
		tb, _ := jsonx.Marshal(t)
		out.Raw("thinking", tb)
	case "extra_body":
		t := wireThinking{Type: "enabled", BudgetTokens: budget}
		tb, _ := jsonx.Marshal(t)
		eb, _ := jsonx.Marshal(map[string]any{
			"google": map[string]any{
				"thinking_config": json.RawMessage(tb),
			},
		})
		out.Raw("extra_body", eb)
	}
	if profile != nil && profile.EffortInTemplateKwargs && effort != ir.EffortUnset {
		kw, _ := jsonx.Marshal(map[string]any{"reasoning_effort": string(effort)})
		out.Raw("chat_template_kwargs", kw)
	}
	return nil
}

// buildToolChoice serialises ir.ToolChoice to JSON.
func buildToolChoice(tc ir.ToolChoice, b *built) ([]byte, error) {
	switch tc.Mode {
	case ir.ToolChoiceAuto:
		return jsonRaw(`"auto"`), nil
	case ir.ToolChoiceNone:
		return jsonRaw(`"none"`), nil
	case ir.ToolChoiceRequired:
		return jsonRaw(`"required"`), nil
	case ir.ToolChoiceTool:
		wtc := wireToolChoice{
			Type:     "function",
			Function: &wireToolChoiceFunc{Name: flattenToolName(tc.Namespace, tc.Name)},
		}
		return jsonx.Marshal(wtc)
	case ir.ToolChoiceAllowed:
		if tc.AllowedMode == ir.ToolChoiceRequired {
			return jsonRaw(`"required"`), nil
		}
		return jsonRaw(`"auto"`), nil
	}
	return nil, nil
}

// buildResponseFormat serialises ir.OutputFormat to JSON.
func buildResponseFormat(of *ir.OutputFormat) ([]byte, error) {
	if of == nil {
		return nil, nil
	}
	switch of.Type {
	case ir.OutputText:
		return jsonx.Marshal(wireResponseFormat{Type: "text"})
	case ir.OutputJSONObject:
		return jsonx.Marshal(wireResponseFormat{Type: "json_object"})
	case ir.OutputJSONSchema:
		rf := wireResponseFormat{
			Type: "json_schema",
			JSONSchema: &wireJSONSchema{
				Name:   of.Name,
				Schema: of.Schema,
				Strict: of.Strict,
			},
		}
		if rf.JSONSchema.Name == "" {
			rf.JSONSchema.Name = "output"
		}
		if rf.JSONSchema.Strict == nil {
			t := true
			rf.JSONSchema.Strict = &t
		}
		return jsonx.Marshal(rf)
	}
	return nil, nil
}

func buildImageContentPart(m *ir.Media) wireContentPart {
	url := m.URL
	if url == "" && m.Data != "" {
		mt := m.MediaType
		if mt == "" {
			mt = "image/jpeg"
		}
		url = "data:" + mt + ";base64," + m.Data
	}
	return wireContentPart{
		Type:     "image_url",
		ImageURL: &wireImageURL{URL: url, Detail: m.Detail},
	}
}

// buildToolResultContent builds the text payload for a Chat tool message and
// extracts any images to be lifted into a follow-up user message.
func buildToolResultContent(r *ir.ToolResult) (string, []wireContentPart) {
	var sb strings.Builder
	var imgs []wireContentPart
	for _, p := range r.Content {
		switch p.Type {
		case ir.PartText:
			sb.WriteString(p.Text)
		case ir.PartImage:
			if p.Media != nil {
				imgs = append(imgs, buildImageContentPart(p.Media))
			}
		}
	}
	if len(imgs) > 0 {
		if sb.Len() > 0 {
			sb.WriteByte('\n')
		}
		sb.WriteString("[image: attached in the next message]")
	}
	text := sb.String()
	if r.IsError {
		if text == "" {
			text = "[tool error]"
		} else {
			text = "[tool error] " + text
		}
	}
	return text, imgs
}

// mustMarshalTextParts concatenates text parts into a single string.
func mustMarshalTextParts(parts []ir.Part) json.RawMessage {
	var sb strings.Builder
	for _, p := range parts {
		if p.Type == ir.PartText {
			sb.WriteString(p.Text)
		}
	}
	return jsonRawString(sb.String())
}

// jsonRawString returns a JSON-encoded string value.
func jsonRawString(s string) json.RawMessage {
	b, _ := jsonx.Marshal(s)
	return b
}

// jsonRaw wraps a string as json.RawMessage.
func jsonRaw(s string) json.RawMessage {
	return json.RawMessage(s)
}

// writeNum writes a float64 field.
func writeNum(out *jsonx.Obj, k string, v float64) {
	out.Raw(k, mustFloat(v))
}

func mustFloat(v float64) json.RawMessage {
	b, err := jsonx.Marshal(v)
	if err != nil {
		return jsonRaw("0")
	}
	return b
}
