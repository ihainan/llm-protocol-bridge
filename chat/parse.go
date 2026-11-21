package chat

import (
	"encoding/json"
	"strings"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/internal/schema"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// ParseRequest decodes a Chat Completions request body into an ir.Request.
func ParseRequest(body []byte, opts ParseOptions) (*ir.Request, error) {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	env.Style = ir.StyleBracket

	kvs, err := jsonx.Object(body)
	if err != nil {
		return nil, ir.Invalid(ir.Root, "request body is not a JSON object")
	}

	req := &ir.Request{Source: ir.FormatChat}
	p := ir.Root

	for _, kv := range kvs {
		k, v := kv.Key, kv.Value
		fp := p.Key(k)
		switch k {
		case "model":
			s, ok := jsonx.String(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be a string"))
			}
			req.Model = s

		case "messages":
			msgs, err := parseMessages(v, fp, env)
			if err != nil {
				return nil, err
			}
			req.Messages = msgs

		case "tools":
			tools, err := parseTools(v, fp, env)
			if err != nil {
				return nil, err
			}
			req.Tools = tools

		case "tool_choice":
			tc, err := parseToolChoice(v, fp, env)
			if err != nil {
				return nil, err
			}
			req.ToolChoice = tc

		case "parallel_tool_calls":
			b, ok := jsonx.Bool(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be a boolean"))
			}
			req.ParallelToolCalls = &b

		case "max_tokens", "max_completion_tokens":
			n, ok := jsonx.Int(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be an integer"))
			}
			ni := int(n)
			req.MaxOutputTokens = &ni

		case "temperature":
			f, ok := jsonx.Float(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be a number"))
			}
			if f < 0 || f > 2 {
				return nil, env.Err(ir.Invalid(fp, "must be between 0 and 2"))
			}
			req.Temperature = &f

		case "top_p":
			f, ok := jsonx.Float(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be a number"))
			}
			req.TopP = &f

		case "n":
			n, ok := jsonx.Int(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be an integer"))
			}
			ni := int(n)
			req.Candidates = &ni

		case "stream":
			b, ok := jsonx.Bool(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be a boolean"))
			}
			req.Stream = b

		case "stream_options":
			// accepted but not stored; include_usage handled at build time

		case "stop":
			stops, err := parseStop(v, fp, env)
			if err != nil {
				return nil, err
			}
			req.Stop = stops

		case "presence_penalty":
			f, ok := jsonx.Float(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be a number"))
			}
			req.PresencePenalty = &f

		case "frequency_penalty":
			f, ok := jsonx.Float(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be a number"))
			}
			req.FrequencyPenalty = &f

		case "logprobs":
			b, ok := jsonx.Bool(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be a boolean"))
			}
			req.Logprobs = &b

		case "top_logprobs":
			n, ok := jsonx.Int(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be an integer"))
			}
			ni := int(n)
			req.TopLogprobs = &ni

		case "seed":
			n, ok := jsonx.Int(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be an integer"))
			}
			req.Seed = &n

		case "response_format":
			of, err := parseResponseFormat(v, fp, env)
			if err != nil {
				return nil, err
			}
			req.Output = of

		case "reasoning_effort":
			if jsonx.IsNull(v) {
				break // null means unset
			}
			s, ok := jsonx.String(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be a string"))
			}
			effort, ok := ir.ParseEffort(s)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "unrecognised reasoning effort"))
			}
			if req.Reasoning == nil {
				req.Reasoning = &ir.ReasoningConfig{}
			}
			req.Reasoning.Effort = effort

		case "user":
			s, ok := jsonx.String(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be a string"))
			}
			req.User = s

		case "metadata":
			req.Metadata = append(json.RawMessage(nil), v...)

		case "service_tier":
			s, ok := jsonx.String(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be a string"))
			}
			req.ServiceTier = s

		case "store":
			b, ok := jsonx.Bool(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be a boolean"))
			}
			req.Store = &b

		default:
			if err := env.UnknownParam(fp); err != nil {
				return nil, err
			}
			if req.Passthrough == nil {
				req.Passthrough = map[string]json.RawMessage{}
			}
			req.Passthrough[k] = append(json.RawMessage(nil), v...)
		}
	}

	// Extract leading system/developer messages into Instructions.
	req.Instructions, req.Messages = extractInstructions(req.Messages)

	return req, nil
}

// parseMessages parses the messages array.
func parseMessages(raw json.RawMessage, p ir.Path, env *ir.Env) ([]ir.Message, error) {
	elems, ok := jsonx.Array(raw)
	if !ok {
		return nil, env.Err(ir.Invalid(p, "must be an array"))
	}
	msgs := make([]ir.Message, 0, len(elems))
	for i, elem := range elems {
		fp := p.At(i)
		msg, err := parseMessage(elem, fp, env)
		if err != nil {
			return nil, err
		}
		msgs = append(msgs, msg)
	}
	return msgs, nil
}

func parseMessage(raw json.RawMessage, p ir.Path, env *ir.Env) (ir.Message, error) {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return ir.Message{}, env.Err(ir.Invalid(p, "must be an object"))
	}

	var msg ir.Message
	var role string
	var contentRaw json.RawMessage
	var toolCalls []wireToolCall
	var toolCallID string
	var nameStr string
	var reasoningText string

	for _, kv := range kvs {
		k, v := kv.Key, kv.Value
		fp := p.Key(k)
		switch k {
		case "role":
			s, ok := jsonx.String(v)
			if !ok {
				return ir.Message{}, env.Err(ir.Invalid(fp, "must be a string"))
			}
			role = s
		case "content":
			contentRaw = v
		case "name":
			s, ok := jsonx.String(v)
			if !ok {
				return ir.Message{}, env.Err(ir.Invalid(fp, "must be a string"))
			}
			nameStr = s
		case "tool_calls":
			tc, err := parseWireToolCalls(v, fp, env)
			if err != nil {
				return ir.Message{}, err
			}
			toolCalls = tc
		case "tool_call_id":
			s, ok := jsonx.String(v)
			if !ok {
				return ir.Message{}, env.Err(ir.Invalid(fp, "must be a string"))
			}
			toolCallID = s
		case "reasoning", "reasoning_content":
			s, ok := jsonx.String(v)
			if !ok {
				break
			}
			reasoningText = s
		case "refusal":
			// handled via content
		}
	}

	msg.Name = nameStr

	switch role {
	case "system":
		msg.Role = ir.RoleSystem
	case "developer":
		msg.Role = ir.RoleDeveloper
	case "user":
		msg.Role = ir.RoleUser
	case "assistant":
		msg.Role = ir.RoleAssistant
	case "tool":
		// Tool message: convert to a user message with tool results.
		// The parts of a tool message are added below.
		msg.Role = ir.RoleUser
	default:
		return ir.Message{}, env.Err(ir.Invalid(p.Key("role"), "unknown role: "+role))
	}

	switch role {
	case "tool":
		// A tool message carries a single text result.
		if !jsonx.IsNull(contentRaw) {
			text := ""
			if s, ok := jsonx.String(contentRaw); ok {
				text = s
			}
			msg.Parts = []ir.Part{{
				Type: ir.PartToolResult,
				Result: &ir.ToolResult{
					CallID:  toolCallID,
					IsError: false,
					Content: []ir.Part{ir.TextPart(text)},
				},
			}}
		}

	case "assistant":
		if reasoningText != "" {
			msg.Parts = append(msg.Parts, ir.Part{
				Type:      ir.PartReasoning,
				Reasoning: &ir.Reasoning{Text: reasoningText},
			})
		}
		if !jsonx.IsNull(contentRaw) {
			parts, err := parseContent(contentRaw, p.Key("content"), env)
			if err != nil {
				return ir.Message{}, err
			}
			msg.Parts = append(msg.Parts, parts...)
		}
		for _, tc := range toolCalls {
			part, err := wireToolCallToPart(tc, p.Key("tool_calls"), env)
			if err != nil {
				return ir.Message{}, err
			}
			msg.Parts = append(msg.Parts, part)
		}

	default:
		if !jsonx.IsNull(contentRaw) {
			parts, err := parseContent(contentRaw, p.Key("content"), env)
			if err != nil {
				return ir.Message{}, err
			}
			msg.Parts = append(msg.Parts, parts...)
		}
	}

	return msg, nil
}

// parseContent decodes a content value: string or array of parts.
func parseContent(raw json.RawMessage, p ir.Path, env *ir.Env) ([]ir.Part, error) {
	switch jsonx.Kind(raw) {
	case '"':
		s, _ := jsonx.String(raw)
		if s == "" {
			return nil, nil
		}
		return []ir.Part{ir.TextPart(s)}, nil
	case '[':
		elems, ok := jsonx.Array(raw)
		if !ok {
			return nil, env.Err(ir.Invalid(p, "must be a string or array"))
		}
		var parts []ir.Part
		for i, elem := range elems {
			fp := p.At(i)
			part, err := parseContentPart(elem, fp, env)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		}
		return parts, nil
	case 'n':
		return nil, nil
	}
	return nil, env.Err(ir.Invalid(p, "must be a string, array, or null"))
}

// parseContentPart decodes one element of a content array.
func parseContentPart(raw json.RawMessage, p ir.Path, env *ir.Env) (ir.Part, error) {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return ir.Part{}, env.Err(ir.Invalid(p, "content part must be an object"))
	}

	var partType string
	var fields = map[string]json.RawMessage{}
	for _, kv := range kvs {
		fields[kv.Key] = kv.Value
		if kv.Key == "type" {
			partType, _ = jsonx.String(kv.Value)
		}
	}

	switch partType {
	case "text":
		text, _ := jsonx.String(fields["text"])
		return ir.TextPart(text), nil

	case "refusal":
		text, _ := jsonx.String(fields["refusal"])
		return ir.Part{Type: ir.PartRefusal, Text: text}, nil

	case "image_url":
		imgRaw, ok := fields["image_url"]
		if !ok || jsonx.IsNull(imgRaw) {
			return ir.Part{}, env.Err(ir.Invalid(p.Key("image_url"), "must be an object"))
		}
		var iu wireImageURL
		if json.Unmarshal(imgRaw, &iu) != nil {
			return ir.Part{}, env.Err(ir.Invalid(p.Key("image_url"), "invalid image_url"))
		}
		media := &ir.Media{Detail: iu.Detail}
		if strings.HasPrefix(iu.URL, "data:") {
			// data:image/jpeg;base64,...
			rest := iu.URL[5:]
			if i := strings.Index(rest, ";base64,"); i >= 0 {
				media.MediaType = rest[:i]
				media.Data = rest[i+8:]
			} else {
				media.URL = iu.URL
			}
		} else {
			media.URL = iu.URL
		}
		return ir.Part{Type: ir.PartImage, Media: media}, nil

	case "input_audio":
		audioRaw, ok := fields["input_audio"]
		if !ok || jsonx.IsNull(audioRaw) {
			return ir.Part{}, env.Err(ir.Invalid(p.Key("input_audio"), "must be an object"))
		}
		var a wireAudio
		if json.Unmarshal(audioRaw, &a) != nil {
			return ir.Part{}, env.Err(ir.Invalid(p.Key("input_audio"), "invalid input_audio"))
		}
		return ir.Part{Type: ir.PartAudio, Media: &ir.Media{
			MediaType: "audio/" + a.Format,
			Data:      a.Data,
		}}, nil

	case "file":
		fileRaw, ok := fields["file"]
		if !ok || jsonx.IsNull(fileRaw) {
			return ir.Part{}, env.Err(ir.Invalid(p.Key("file"), "must be an object"))
		}
		var f wireFilePart
		if json.Unmarshal(fileRaw, &f) != nil {
			return ir.Part{}, env.Err(ir.Invalid(p.Key("file"), "invalid file"))
		}
		return ir.Part{Type: ir.PartFile, Media: &ir.Media{
			FileID:   f.FileID,
			Filename: f.Filename,
			Data:     f.FileData,
		}}, nil

	case "thinking", "thought":
		// Reasoning block in some dialects
		thinking, _ := jsonx.String(fields["thinking"])
		sig, _ := jsonx.String(fields["signature"])
		r := &ir.Reasoning{Text: thinking}
		p := ir.Part{Type: ir.PartReasoning, Reasoning: r}
		if sig != "" {
			p.Signature = &ir.Signature{Value: sig}
		}
		return p, nil

	default:
		return ir.Part{}, env.Err(ir.Invalid(p.Key("type"), "unknown content part type: "+partType))
	}
}

// parseWireToolCalls decodes a tool_calls array.
func parseWireToolCalls(raw json.RawMessage, p ir.Path, env *ir.Env) ([]wireToolCall, error) {
	elems, ok := jsonx.Array(raw)
	if !ok {
		return nil, env.Err(ir.Invalid(p, "must be an array"))
	}
	tcs := make([]wireToolCall, 0, len(elems))
	for i, elem := range elems {
		var tc wireToolCall
		if json.Unmarshal(elem, &tc) != nil {
			return nil, env.Err(ir.Invalid(p.At(i), "invalid tool_call"))
		}
		tcs = append(tcs, tc)
	}
	return tcs, nil
}

func wireToolCallToPart(tc wireToolCall, p ir.Path, env *ir.Env) (ir.Part, error) {
	if tc.Function.Name == "" {
		return ir.Part{}, env.Err(ir.Invalid(p, "tool_call missing function name"))
	}
	var args json.RawMessage
	if s := strings.TrimSpace(tc.Function.Arguments); s != "" && s != "null" {
		args = json.RawMessage(s)
	}
	return ir.Part{
		Type: ir.PartToolCall,
		Call: &ir.ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Kind:      ir.ToolFunction,
			Arguments: args,
		},
	}, nil
}

// parseTools decodes the top-level tools array.
func parseTools(raw json.RawMessage, p ir.Path, env *ir.Env) ([]ir.Tool, error) {
	if jsonx.IsNull(raw) {
		return nil, nil
	}
	elems, ok := jsonx.Array(raw)
	if !ok {
		return nil, env.Err(ir.Invalid(p, "must be an array"))
	}
	tools := make([]ir.Tool, 0, len(elems))
	for i, elem := range elems {
		fp := p.At(i)
		tool, err := parseTool(elem, fp, env)
		if err != nil {
			return nil, err
		}
		tools = append(tools, tool)
	}
	return tools, nil
}

func parseTool(raw json.RawMessage, p ir.Path, env *ir.Env) (ir.Tool, error) {
	var wt wireTool
	if json.Unmarshal(raw, &wt) != nil {
		return ir.Tool{}, env.Err(ir.Invalid(p, "invalid tool"))
	}
	if wt.Type != "function" {
		return ir.Tool{}, env.Err(ir.Invalid(p.Key("type"), "unknown tool type: "+wt.Type))
	}
	if wt.Function.Name == "" {
		return ir.Tool{}, env.Err(ir.Invalid(p.Keys("function", "name"), "tool name is required"))
	}
	paramSchema := wt.Function.Parameters
	if !jsonx.Absent(paramSchema) {
		s, ok := schema.ObjectRoot(paramSchema)
		if !ok {
			return ir.Tool{}, env.Err(ir.Invalid(p.Keys("function", "parameters"), "must be an object schema"))
		}
		paramSchema = s
	}
	t := ir.Tool{
		Name:        wt.Function.Name,
		Description: wt.Function.Description,
		Kind:        ir.ToolFunction,
		Schema:      paramSchema,
	}
	if wt.Function.Strict != nil {
		t.Strict = wt.Function.Strict
	}
	return t, nil
}

// parseToolChoice decodes the tool_choice field.
func parseToolChoice(raw json.RawMessage, p ir.Path, env *ir.Env) (ir.ToolChoice, error) {
	if jsonx.IsNull(raw) {
		return ir.ToolChoice{}, nil
	}
	switch jsonx.Kind(raw) {
	case '"':
		s, _ := jsonx.String(raw)
		switch s {
		case "auto":
			return ir.ToolChoice{Mode: ir.ToolChoiceAuto}, nil
		case "none":
			return ir.ToolChoice{Mode: ir.ToolChoiceNone}, nil
		case "required":
			return ir.ToolChoice{Mode: ir.ToolChoiceRequired}, nil
		default:
			return ir.ToolChoice{}, env.Err(ir.Invalid(p, "unknown tool_choice: "+s))
		}
	case '{':
		var wtc wireToolChoice
		if json.Unmarshal(raw, &wtc) != nil {
			return ir.ToolChoice{}, env.Err(ir.Invalid(p, "invalid tool_choice"))
		}
		if wtc.Type == "function" && wtc.Function != nil {
			return ir.ToolChoice{
				Mode: ir.ToolChoiceTool,
				Name: wtc.Function.Name,
			}, nil
		}
		switch wtc.Type {
		case "auto":
			return ir.ToolChoice{Mode: ir.ToolChoiceAuto}, nil
		case "none":
			return ir.ToolChoice{Mode: ir.ToolChoiceNone}, nil
		case "required":
			return ir.ToolChoice{Mode: ir.ToolChoiceRequired}, nil
		}
		return ir.ToolChoice{}, env.Err(ir.Invalid(p.Key("type"), "unknown tool_choice type: "+wtc.Type))
	}
	return ir.ToolChoice{}, env.Err(ir.Invalid(p, "must be a string or object"))
}

// parseStop decodes the stop field (string or array).
func parseStop(raw json.RawMessage, p ir.Path, env *ir.Env) ([]string, error) {
	if jsonx.IsNull(raw) {
		return nil, nil
	}
	switch jsonx.Kind(raw) {
	case '"':
		s, _ := jsonx.String(raw)
		return []string{s}, nil
	case '[':
		elems, ok := jsonx.Array(raw)
		if !ok {
			return nil, env.Err(ir.Invalid(p, "must be a string or array"))
		}
		stops := make([]string, 0, len(elems))
		for i, elem := range elems {
			s, ok := jsonx.String(elem)
			if !ok {
				return nil, env.Err(ir.Invalid(p.At(i), "must be a string"))
			}
			stops = append(stops, s)
		}
		return stops, nil
	}
	return nil, env.Err(ir.Invalid(p, "must be a string or array"))
}

// parseResponseFormat decodes the response_format object.
func parseResponseFormat(raw json.RawMessage, p ir.Path, env *ir.Env) (*ir.OutputFormat, error) {
	if jsonx.IsNull(raw) {
		return nil, nil
	}
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return nil, env.Err(ir.Invalid(p, "must be an object"))
	}
	var typStr string
	var jsRaw json.RawMessage
	for _, kv := range kvs {
		switch kv.Key {
		case "type":
			typStr, _ = jsonx.String(kv.Value)
		case "json_schema":
			jsRaw = kv.Value
		}
	}
	of := &ir.OutputFormat{}
	switch typStr {
	case "text":
		of.Type = ir.OutputText
	case "json_object":
		of.Type = ir.OutputJSONObject
	case "json_schema":
		of.Type = ir.OutputJSONSchema
		if !jsonx.IsNull(jsRaw) {
			var js wireJSONSchema
			if json.Unmarshal(jsRaw, &js) != nil {
				return nil, env.Err(ir.Invalid(p.Key("json_schema"), "invalid json_schema"))
			}
			of.Name = js.Name
			of.Schema = js.Schema
			of.Strict = js.Strict
		}
	default:
		return nil, env.Err(ir.Invalid(p.Key("type"), "unknown response_format type: "+typStr))
	}
	return of, nil
}

// extractInstructions moves leading system/developer messages into Instructions.
func extractInstructions(msgs []ir.Message) ([]ir.Part, []ir.Message) {
	var instParts []ir.Part
	i := 0
	for i < len(msgs) {
		m := msgs[i]
		if m.Role != ir.RoleSystem && m.Role != ir.RoleDeveloper {
			break
		}
		instParts = append(instParts, m.Parts...)
		i++
	}
	return instParts, msgs[i:]
}
