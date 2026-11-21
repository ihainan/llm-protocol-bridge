package chat

import (
	"encoding/json"
	"strings"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/internal/schema"
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

	// 1. Tools
	var wireTools []wireTool
	if len(req.Tools) > 0 {
		wireTools = make([]wireTool, 0, len(req.Tools))
		for i, t := range req.Tools {
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
	if err := b.buildReasoning(req, out); err != nil {
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
	// store is a Responses-API-only field; never emit it in Chat format.

	// passthrough
	for k, v := range req.Passthrough {
		out.Raw(k, v)
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

// buildTool converts an ir.Tool to wireTool and records the name mapping.
func (b *built) buildTool(t ir.Tool, p ir.Path) (wireTool, error) {
	if t.Name == "" {
		return wireTool{}, b.env.Err(ir.Invalid(p.Key("name"), "tool name is required"))
	}
	if t.Kind != "" && t.Kind != ir.ToolFunction {
		return wireTool{}, b.env.Err(ir.Invalid(p.Key("kind"), "only function tools are supported"))
	}
	// Record the name mapping (sanitised name -> original ir.ToolRef).
	ref := ir.ToolRef{Name: t.Name, Namespace: t.Namespace, Kind: t.Kind}
	b.names[t.Name] = ref

	params := t.Schema
	if !jsonx.Absent(params) {
		params = schema.ToOpenAPI(params)
	}
	fn := wireFunctionDef{
		Name:        t.Name,
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

	// Collect tool result messages and remaining (non-result) parts.
	var toolMsgs []wireMessage

	for i, part := range m.Parts {
		fp := p.Key("parts").At(i)
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
			url := part.Media.URL
			if url == "" && part.Media.Data != "" {
				mt := part.Media.MediaType
				if mt == "" {
					mt = "image/jpeg"
				}
				url = "data:" + mt + ";base64," + part.Media.Data
			}
			contentParts = append(contentParts, wireContentPart{
				Type:     "image_url",
				ImageURL: &wireImageURL{URL: url, Detail: part.Media.Detail},
			})

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
				reasoningText += part.Reasoning.Text
			}
			if part.Signature != nil {
				reasoningSignature = part.Signature.Value
			}

		case ir.PartToolCall:
			if part.Call == nil {
				continue
			}
			tc := wireToolCall{
				ID:   part.Call.ID,
				Type: "function",
				Function: wireFunctionCall{
					Name:      part.Call.Name,
					Arguments: string(part.Call.Arguments),
				},
			}
			toolCalls = append(toolCalls, tc)

		case ir.PartToolResult:
			// Each tool result becomes its own tool-role message.
			if part.Result == nil {
				continue
			}
			resultText := toolResultText(part.Result)
			toolMsgs = append(toolMsgs, wireMessage{
				Role:       "tool",
				ToolCallID: part.Result.CallID,
				Content:    jsonRawString(resultText),
			})

		case ir.PartBuiltinCall, ir.PartBuiltinResult:
			_ = fp
		}
	}

	// If the message was entirely tool results, return just those.
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

// buildReasoning writes reasoning configuration based on the dialect.
func (b *built) buildReasoning(req *ir.Request, out jsonx.Obj) error {
	if req.Reasoning == nil {
		return nil
	}
	r := req.Reasoning
	switch b.dia.ReasoningParam {
	case "", "reasoning_effort":
		if r.Effort != "" {
			out.Str("reasoning_effort", string(r.Effort))
		}
	case "enable_thinking":
		enable := r.Mode != ir.ReasoningOff
		out.Bool("enable_thinking", enable)
		if r.Budget != nil {
			out.Int("budget_tokens", int64(*r.Budget))
		}
	case "thinking":
		if r.Mode == ir.ReasoningOff {
			out.Raw("thinking", jsonRaw(`{"type":"disabled"}`))
		} else {
			t := wireThinking{Type: "enabled"}
			if r.Budget != nil {
				t.BudgetTokens = *r.Budget
			}
			tb, _ := jsonx.Marshal(t)
			out.Raw("thinking", tb)
		}
	case "extra_body":
		if r.Mode != ir.ReasoningOff {
			t := wireThinking{Type: "enabled"}
			if r.Budget != nil {
				t.BudgetTokens = *r.Budget
			}
			tb, _ := jsonx.Marshal(t)
			// extra_body.google.thinking_config
			eb, _ := jsonx.Marshal(map[string]any{
				"google": map[string]any{
					"thinking_config": json.RawMessage(tb),
				},
			})
			out.Raw("extra_body", eb)
		}
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
			Function: &wireToolChoiceFunc{Name: tc.Name},
		}
		return jsonx.Marshal(wtc)
	case ir.ToolChoiceAllowed:
		// Chat does not have an allowed-only mode; fall back to auto.
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

// toolResultText extracts the text content of a tool result.
func toolResultText(r *ir.ToolResult) string {
	var sb strings.Builder
	for _, p := range r.Content {
		if p.Type == ir.PartText {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
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
