package anthropic

import (
	"encoding/json"
	"strings"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/internal/schema"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// ParseRequest decodes an Anthropic Messages request body into an ir.Request.
func ParseRequest(body []byte, opts ParseOptions) (*ir.Request, error) {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	env.Style = ir.StyleDot

	kvs, err := jsonx.Object(body)
	if err != nil {
		return nil, ir.Invalid(ir.Root, "request body is not a JSON object")
	}

	req := &ir.Request{Source: ir.FormatAnthropic}
	p := ir.Root
	var hasMaxTokens bool

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

		case "system":
			parts, err := parseSystem(v, fp, env)
			if err != nil {
				return nil, err
			}
			req.Instructions = parts

		case "max_tokens":
			n, ok := jsonx.Int(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be an integer"))
			}
			ni := int(n)
			req.MaxOutputTokens = &ni
			hasMaxTokens = true

		case "temperature":
			f, ok := jsonx.Float(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be a number"))
			}
			if f < 0 || f > 1 {
				return nil, env.Err(ir.Invalid(fp, "must be between 0 and 1"))
			}
			req.Temperature = &f

		case "top_p":
			f, ok := jsonx.Float(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be a number"))
			}
			req.TopP = &f

		case "top_k":
			n, ok := jsonx.Int(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be an integer"))
			}
			ni := int(n)
			req.TopK = &ni

		case "stop_sequences":
			stops, err := parseStopSeqs(v, fp, env)
			if err != nil {
				return nil, err
			}
			req.Stop = stops

		case "stream":
			b, ok := jsonx.Bool(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be a boolean"))
			}
			req.Stream = b

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

		case "thinking":
			rc, err := parseThinkingConfig(v, fp, env)
			if err != nil {
				return nil, err
			}
			req.Reasoning = rc

		case "metadata":
			// Extract user_id.
			kvs2, e2 := jsonx.Object(v)
			if e2 == nil {
				for _, kv2 := range kvs2 {
					if kv2.Key == "user_id" {
						s, ok := jsonx.String(kv2.Value)
						if ok {
							req.User = s
						}
					}
				}
			}

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

	// Anthropic requires max_tokens unless this is a count_tokens call.
	if !hasMaxTokens && opts.Variant != "count_tokens" {
		return nil, env.Err(ir.Invalid(p.Key("max_tokens"), "max_tokens is required"))
	}

	return req, nil
}

func parseSystem(raw json.RawMessage, p ir.Path, env *ir.Env) ([]ir.Part, error) {
	switch jsonx.Kind(raw) {
	case '"':
		s, _ := jsonx.String(raw)
		return []ir.Part{ir.TextPart(s)}, nil
	case '[':
		elems, ok := jsonx.Array(raw)
		if !ok {
			return nil, env.Err(ir.Invalid(p, "must be a string or array"))
		}
		var parts []ir.Part
		for i, elem := range elems {
			part, err := parseContentBlock(elem, p.At(i), env)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		}
		return parts, nil
	}
	return nil, env.Err(ir.Invalid(p, "must be a string or array"))
}

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
	var role string
	var contentRaw json.RawMessage
	for _, kv := range kvs {
		switch kv.Key {
		case "role":
			role, _ = jsonx.String(kv.Value)
		case "content":
			contentRaw = kv.Value
		}
	}

	var irRole ir.Role
	switch role {
	case "user":
		irRole = ir.RoleUser
	case "assistant":
		irRole = ir.RoleAssistant
	default:
		return ir.Message{}, env.Err(ir.Invalid(p.Key("role"), "unknown role: "+role))
	}

	var parts []ir.Part
	switch jsonx.Kind(contentRaw) {
	case '"':
		s, _ := jsonx.String(contentRaw)
		parts = []ir.Part{ir.TextPart(s)}
	case '[':
		elems, ok := jsonx.Array(contentRaw)
		if !ok {
			return ir.Message{}, env.Err(ir.Invalid(p.Key("content"), "must be a string or array"))
		}
		for i, elem := range elems {
			part, err := parseContentBlock(elem, p.Key("content").At(i), env)
			if err != nil {
				return ir.Message{}, err
			}
			parts = append(parts, part)
		}
	case 'n':
		// null content is allowed; empty message
	}

	return ir.Message{Role: irRole, Parts: parts}, nil
}

func parseContentBlock(raw json.RawMessage, p ir.Path, env *ir.Env) (ir.Part, error) {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return ir.Part{}, env.Err(ir.Invalid(p, "must be an object"))
	}
	fields := map[string]json.RawMessage{}
	var typ string
	for _, kv := range kvs {
		fields[kv.Key] = kv.Value
		if kv.Key == "type" {
			typ, _ = jsonx.String(kv.Value)
		}
	}

	switch typ {
	case "text":
		text, _ := jsonx.String(fields["text"])
		part := ir.TextPart(text)
		part.Cache = parseCacheControl(fields["cache_control"])
		return part, nil

	case "image":
		src := fields["source"]
		if jsonx.IsNull(src) {
			return ir.Part{}, env.Err(ir.Invalid(p.Key("source"), "source is required"))
		}
		media, err := parseImageSource(src, p.Key("source"), env)
		if err != nil {
			return ir.Part{}, err
		}
		part := ir.Part{Type: ir.PartImage, Media: media}
		part.Cache = parseCacheControl(fields["cache_control"])
		return part, nil

	case "document":
		part := ir.Part{Type: ir.PartFile}
		src := fields["source"]
		if !jsonx.IsNull(src) {
			media, err := parseImageSource(src, p.Key("source"), env)
			if err != nil {
				return ir.Part{}, err
			}
			if t, ok := jsonx.String(fields["title"]); ok {
				media.Title = t
			}
			if c, ok := jsonx.String(fields["context"]); ok {
				media.Context = c
			}
			part.Media = media
		}
		part.Cache = parseCacheControl(fields["cache_control"])
		return part, nil

	case "tool_use":
		id, _ := jsonx.String(fields["id"])
		name, _ := jsonx.String(fields["name"])
		if name == "" {
			return ir.Part{}, env.Err(ir.Invalid(p.Key("name"), "tool name is required"))
		}
		args := fields["input"]
		return ir.Part{
			Type: ir.PartToolCall,
			Call: &ir.ToolCall{ID: id, Name: name, Kind: ir.ToolFunction, Arguments: append(json.RawMessage(nil), args...)},
		}, nil

	case "tool_result":
		callID, _ := jsonx.String(fields["tool_use_id"])
		isErr := false
		if b, ok := jsonx.Bool(fields["is_error"]); ok {
			isErr = b
		}
		var content []ir.Part
		contentRaw := fields["content"]
		switch jsonx.Kind(contentRaw) {
		case '"':
			s, _ := jsonx.String(contentRaw)
			content = []ir.Part{ir.TextPart(s)}
		case '[':
			elems, ok := jsonx.Array(contentRaw)
			if !ok {
				return ir.Part{}, env.Err(ir.Invalid(p.Key("content"), "must be a string or array"))
			}
			for i, elem := range elems {
				part, err := parseContentBlock(elem, p.Key("content").At(i), env)
				if err != nil {
					return ir.Part{}, err
				}
				content = append(content, part)
			}
		}
		return ir.Part{
			Type: ir.PartToolResult,
			Result: &ir.ToolResult{
				CallID:  callID,
				IsError: isErr,
				Content: content,
			},
		}, nil

	case "thinking":
		thinking, _ := jsonx.String(fields["thinking"])
		sig, _ := jsonx.String(fields["signature"])
		return ir.Part{
			Type: ir.PartReasoning,
			Reasoning: &ir.Reasoning{
				Text:      thinking,
				Signature: &ir.Signature{Value: sig, Origin: ir.Origin{Format: ir.FormatAnthropic}},
			},
		}, nil

	case "redacted_thinking":
		data, _ := jsonx.String(fields["data"])
		rawData, _ := jsonx.Marshal(data)
		return ir.Part{
			Type: ir.PartReasoning,
			Reasoning: &ir.Reasoning{
				Sealed: &ir.Sealed{
					Kind:   ir.SealedAnthropicRedacted,
					Data:   rawData,
					Origin: ir.Origin{Format: ir.FormatAnthropic, Provider: env.ProviderID},
				},
			},
		}, nil

	default:
		return ir.Part{}, env.Err(ir.Invalid(p.Key("type"), "unknown content block type: "+typ))
	}
}

func parseImageSource(raw json.RawMessage, p ir.Path, env *ir.Env) (*ir.Media, error) {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return nil, env.Err(ir.Invalid(p, "must be an object"))
	}
	fields := map[string]json.RawMessage{}
	for _, kv := range kvs {
		fields[kv.Key] = kv.Value
	}
	typ, _ := jsonx.String(fields["type"])
	media := &ir.Media{}
	switch typ {
	case "base64":
		media.MediaType, _ = jsonx.String(fields["media_type"])
		media.Data, _ = jsonx.String(fields["data"])
	case "url":
		media.URL, _ = jsonx.String(fields["url"])
	case "file":
		media.FileID, _ = jsonx.String(fields["file_id"])
	default:
		return nil, env.Err(ir.Invalid(p.Key("type"), "unknown source type: "+typ))
	}
	return media, nil
}

func parseCacheControl(raw json.RawMessage) *ir.CacheControl {
	if jsonx.IsNull(raw) {
		return nil
	}
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return nil
	}
	cc := &ir.CacheControl{}
	for _, kv := range kvs {
		switch kv.Key {
		case "type":
			cc.Type, _ = jsonx.String(kv.Value)
		case "ttl":
			cc.TTL, _ = jsonx.String(kv.Value)
		}
	}
	return cc
}

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
		t, err := parseTool(elem, fp, env)
		if err != nil {
			return nil, err
		}
		tools = append(tools, t)
	}
	return tools, nil
}

func parseTool(raw json.RawMessage, p ir.Path, env *ir.Env) (ir.Tool, error) {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return ir.Tool{}, env.Err(ir.Invalid(p, "must be an object"))
	}
	fields := map[string]json.RawMessage{}
	for _, kv := range kvs {
		fields[kv.Key] = kv.Value
	}
	name, _ := jsonx.String(fields["name"])
	if name == "" {
		return ir.Tool{}, env.Err(ir.Invalid(p.Key("name"), "tool name is required"))
	}
	desc, _ := jsonx.String(fields["description"])
	inputSchema := fields["input_schema"]
	if jsonx.IsNull(inputSchema) {
		inputSchema = schema.EmptyObject
	}
	t := ir.Tool{
		Name:        name,
		Description: desc,
		Kind:        ir.ToolFunction,
		Schema:      inputSchema,
	}
	t.Cache = parseCacheControl(fields["cache_control"])
	return t, nil
}

func parseToolChoice(raw json.RawMessage, p ir.Path, env *ir.Env) (ir.ToolChoice, error) {
	if jsonx.IsNull(raw) {
		return ir.ToolChoice{}, nil
	}
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return ir.ToolChoice{}, env.Err(ir.Invalid(p, "must be an object"))
	}
	var typ, name string
	for _, kv := range kvs {
		switch kv.Key {
		case "type":
			typ, _ = jsonx.String(kv.Value)
		case "name":
			name, _ = jsonx.String(kv.Value)
		}
	}
	switch typ {
	case "auto":
		return ir.ToolChoice{Mode: ir.ToolChoiceAuto}, nil
	case "any":
		return ir.ToolChoice{Mode: ir.ToolChoiceRequired}, nil
	case "tool":
		return ir.ToolChoice{Mode: ir.ToolChoiceTool, Name: name}, nil
	}
	return ir.ToolChoice{}, env.Err(ir.Invalid(p.Key("type"), "unknown tool_choice type: "+typ))
}

func parseThinkingConfig(raw json.RawMessage, p ir.Path, env *ir.Env) (*ir.ReasoningConfig, error) {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return nil, env.Err(ir.Invalid(p, "must be an object"))
	}
	rc := &ir.ReasoningConfig{}
	for _, kv := range kvs {
		switch kv.Key {
		case "type":
			t, _ := jsonx.String(kv.Value)
			switch t {
			case "enabled":
				rc.Mode = ir.ReasoningOn
			case "disabled":
				rc.Mode = ir.ReasoningOff
			}
		case "budget_tokens":
			n, ok := jsonx.Int(kv.Value)
			if ok {
				ni := int(n)
				rc.Budget = &ni
			}
		}
	}
	return rc, nil
}

func parseStopSeqs(raw json.RawMessage, p ir.Path, env *ir.Env) ([]string, error) {
	if jsonx.IsNull(raw) {
		return nil, nil
	}
	elems, ok := jsonx.Array(raw)
	if !ok {
		return nil, env.Err(ir.Invalid(p, "must be an array"))
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

// toolResultText returns the concatenated text content of a ToolResult.
func toolResultText(r *ir.ToolResult) string {
	var sb strings.Builder
	for _, p := range r.Content {
		if p.Type == ir.PartText {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}
