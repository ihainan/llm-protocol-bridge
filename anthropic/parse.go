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
	var hasMessages bool
	var rawContextMgmt json.RawMessage
	var thinkingKeepTurns *int

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
			hasMessages = true

		case "system":
			if jsonx.IsNull(v) {
				break
			}
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
			if n <= 0 && opts.Variant != "count_tokens" {
				return nil, env.Err(ir.Invalid(fp, "max_tokens must be greater than 0"))
			}
			ni := int(n)
			req.MaxOutputTokens = &ni
			hasMaxTokens = true

		case "temperature":
			if jsonx.IsNull(v) {
				break
			}
			f, ok := jsonx.Float(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be a number"))
			}
			if f < 0 || f > 1 {
				return nil, env.Err(ir.Invalid(fp, "must be between 0 and 1"))
			}
			req.Temperature = &f

		case "top_p":
			if jsonx.IsNull(v) {
				break
			}
			f, ok := jsonx.Float(v)
			if !ok {
				return nil, env.Err(ir.Invalid(fp, "must be a number"))
			}
			req.TopP = &f

		case "top_k":
			if jsonx.IsNull(v) {
				break
			}
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
			tools, builtins, err := parseTools(v, fp, env)
			if err != nil {
				return nil, err
			}
			req.Tools = tools
			req.Builtins = builtins

		case "tool_choice":
			tc, disableParallel, err := parseToolChoice(v, fp, env)
			if err != nil {
				return nil, err
			}
			req.ToolChoice = tc
			if disableParallel != nil {
				req.ParallelToolCalls = disableParallel
			}

		case "thinking":
			if jsonx.IsNull(v) {
				break
			}
			rc, err := parseThinkingConfig(v, fp, env)
			if err != nil {
				return nil, err
			}
			if req.Reasoning != nil && req.Reasoning.Effort != ir.EffortUnset && rc.Effort == ir.EffortUnset {
				rc.Effort = req.Reasoning.Effort
			}
			req.Reasoning = rc

		case "output_config":
			if err := parseOutputConfig(v, fp, req, env); err != nil {
				return nil, err
			}

		case "context_management":
			if jsonx.IsNull(v) {
				break
			}
			kt, err := parseContextManagement(v, fp, env)
			if err != nil {
				return nil, err
			}
			thinkingKeepTurns = kt
			rawContextMgmt = append(json.RawMessage(nil), v...)

		case "mcp_servers":
			if jsonx.IsNull(v) {
				break
			}
			if elems, ok := jsonx.Array(v); ok && len(elems) == 0 {
				break
			}
			if err := env.UnknownParam(fp); err != nil {
				return nil, err
			}
			if req.Passthrough == nil {
				req.Passthrough = map[string]json.RawMessage{}
			}
			req.Passthrough[k] = append(json.RawMessage(nil), v...)

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

	if len(rawContextMgmt) > 0 {
		if req.Passthrough == nil {
			req.Passthrough = map[string]json.RawMessage{}
		}
		req.Passthrough["context_management"] = rawContextMgmt
	}
	if thinkingKeepTurns != nil {
		if req.Reasoning == nil {
			req.Reasoning = &ir.ReasoningConfig{}
		}
		req.Reasoning.KeepTurns = thinkingKeepTurns
	}

	// Anthropic requires max_tokens unless this is a count_tokens call.
	if !hasMaxTokens && opts.Variant != "count_tokens" {
		return nil, env.Err(ir.Invalid(p.Key("max_tokens"), "max_tokens is required"))
	}

	// Anthropic requires a non-empty messages array.
	if !hasMessages || len(req.Messages) == 0 {
		return nil, env.Err(ir.Invalid(p.Key("messages"), "messages must not be empty"))
	}

	// Validate thinking.budget_tokens against Anthropic minimum (1024) and max_tokens.
	if req.Reasoning != nil && req.Reasoning.Mode == ir.ReasoningOn {
		if req.Reasoning.Budget == nil {
			return nil, env.Err(ir.Invalid(p.Key("thinking").Key("budget_tokens"), "budget_tokens is required when thinking is enabled"))
		}
		if *req.Reasoning.Budget < 1024 {
			return nil, env.Err(ir.Invalid(p.Key("thinking").Key("budget_tokens"), "budget_tokens must be at least 1024"))
		}
		if opts.Variant != "count_tokens" && req.MaxOutputTokens != nil && *req.Reasoning.Budget >= *req.MaxOutputTokens {
			return nil, env.Err(ir.Invalid(p.Key("thinking").Key("budget_tokens"), "budget_tokens must be less than max_tokens"))
		}
	}

	// Validate tool_choice against declared tools.
	if req.ToolChoice.Mode != ir.ToolChoiceUnset {
		toolNames := make(map[string]bool, len(req.Tools))
		for _, t := range req.Tools {
			toolNames[t.Name] = true
		}
		switch req.ToolChoice.Mode {
		case ir.ToolChoiceRequired:
			if len(req.Tools) == 0 && len(req.Builtins) == 0 {
				return nil, env.Err(ir.Invalid(p.Key("tool_choice"), `tool_choice "any" requires at least one tool`))
			}
		case ir.ToolChoiceTool:
			if req.ToolChoice.Name == "" || !toolNames[req.ToolChoice.Name] {
				return nil, env.Err(ir.Invalid(p.Key("tool_choice").Key("name"), "tool_choice names a tool that is not in tools: "+req.ToolChoice.Name))
			}
		}
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
			ep := p.At(i)
			kvs, err := jsonx.Object(elem)
			if err != nil {
				return nil, env.Err(ir.Invalid(ep, "must be an object"))
			}
			fields := map[string]json.RawMessage{}
			for _, kv := range kvs {
				fields[kv.Key] = kv.Value
			}
			typ, _ := jsonx.String(fields["type"])
			if typ != "text" {
				return nil, env.Err(ir.Invalid(ep.Key("type"), "system blocks must have type text"))
			}
			text, ok := jsonx.String(fields["text"])
			if !ok || text == "" {
				return nil, env.Err(ir.Invalid(ep.Key("text"), "system text block requires non-empty text"))
			}
			part := ir.TextPart(text)
			part.Cache = parseCacheControl(fields["cache_control"])
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
	var lastToolUses map[string]bool
	var prevNonSystem ir.Role
	for i, elem := range elems {
		fp := p.At(i)
		isLast := i == len(elems)-1
		msg, err := parseMessage(elem, fp, isLast, env)
		if err != nil {
			return nil, err
		}
		switch msg.Role {
		case ir.RoleAssistant:
			if prevNonSystem != ir.RoleAssistant || lastToolUses == nil {
				lastToolUses = map[string]bool{}
			}
			for _, pt := range msg.Parts {
				if pt.Type == ir.PartToolCall && pt.Call != nil && pt.Call.ID != "" {
					lastToolUses[pt.Call.ID] = true
				}
			}
			prevNonSystem = ir.RoleAssistant
		case ir.RoleUser:
			if !env.Repairs(ir.RepairOrphanResults) {
				for j, pt := range msg.Parts {
					if pt.Type == ir.PartToolResult && pt.Result != nil {
						if !lastToolUses[pt.Result.CallID] {
							return nil, env.Err(ir.Invalid(fp.Key("content").At(j).Key("tool_use_id"), "tool_result does not match any tool_use in previous assistant turn: "+pt.Result.CallID))
						}
					}
				}
			}
			prevNonSystem = ir.RoleUser
		case ir.RoleSystem:
			// Mid-conversation system messages do not reset lastToolUses or turn state.
		}
		msgs = append(msgs, msg)
	}
	return msgs, nil
}

func parseMessage(raw json.RawMessage, p ir.Path, isLast bool, env *ir.Env) (ir.Message, error) {
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
	case "system":
		irRole = ir.RoleSystem
	default:
		return ir.Message{}, env.Err(ir.Invalid(p.Key("role"), "unknown role: "+role))
	}

	finalAssistant := isLast && irRole == ir.RoleAssistant

	var parts []ir.Part
	switch jsonx.Kind(contentRaw) {
	case '"':
		s, _ := jsonx.String(contentRaw)
		if s == "" && !finalAssistant {
			return ir.Message{}, env.Err(ir.Invalid(p.Key("content"), "message content must not be empty"))
		}
		if s != "" {
			parts = []ir.Part{ir.TextPart(s)}
		}
	case '[':
		elems, ok := jsonx.Array(contentRaw)
		if !ok {
			return ir.Message{}, env.Err(ir.Invalid(p.Key("content"), "must be a string or array"))
		}
		if len(elems) == 0 && !finalAssistant {
			return ir.Message{}, env.Err(ir.Invalid(p.Key("content"), "message content must not be empty"))
		}
		for i, elem := range elems {
			part, err := parseContentBlock(elem, p.Key("content").At(i), irRole, finalAssistant, env)
			if err != nil {
				return ir.Message{}, err
			}
			parts = append(parts, part)
		}
	case 'n', 0:
		if !finalAssistant {
			return ir.Message{}, env.Err(ir.Invalid(p.Key("content"), "message content must not be empty"))
		}
	default:
		return ir.Message{}, env.Err(ir.Invalid(p.Key("content"), "must be a string or array"))
	}

	return ir.Message{Role: irRole, Parts: parts}, nil
}

func parseContentBlock(raw json.RawMessage, p ir.Path, role ir.Role, allowEmptyText bool, env *ir.Env) (ir.Part, error) {
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

	if role == ir.RoleSystem && typ != "text" {
		return ir.Part{}, env.Err(ir.Invalid(p.Key("type"), "system messages only allow text blocks"))
	}

	switch typ {
	case "text":
		text, ok := jsonx.String(fields["text"])
		if !allowEmptyText && (!ok || text == "") {
			return ir.Part{}, env.Err(ir.Invalid(p.Key("text"), "text block requires non-empty text"))
		}
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
		if jsonx.IsNull(src) {
			return ir.Part{}, env.Err(ir.Invalid(p.Key("source"), "source is required"))
		}
		media, err := parseDocumentSource(src, p.Key("source"), env)
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
		part.Cache = parseCacheControl(fields["cache_control"])
		return part, nil

	case "tool_use":
		rawID, _ := jsonx.String(fields["id"])
		id, sig := ir.DecodeCallSignatureID(env, ir.AnthropicRawID(rawID))
		name, _ := jsonx.String(fields["name"])
		if name == "" {
			return ir.Part{}, env.Err(ir.Invalid(p.Key("name"), "tool name is required"))
		}
		var args json.RawMessage
		if rawIn, exists := fields["input"]; exists {
			if jsonx.Kind(rawIn) != '{' {
				return ir.Part{}, env.Err(ir.Invalid(p.Key("input"), "tool_use input must be a JSON object"))
			}
			args = append(json.RawMessage(nil), rawIn...)
		} else {
			args = json.RawMessage(`{}`)
		}
		return ir.Part{
			Type: ir.PartToolCall,
			Call: &ir.ToolCall{ID: id, Name: name, Kind: ir.ToolFunction, Arguments: args, Signature: sig},
		}, nil

	case "tool_result":
		if role != ir.RoleUser {
			return ir.Part{}, env.Err(ir.Invalid(p, "tool_result is only allowed in user messages"))
		}
		rawCallID, _ := jsonx.String(fields["tool_use_id"])
		callID, _ := ir.DecodeCallSignatureID(env, ir.AnthropicRawID(rawCallID))
		isErr := false
		if b, ok := jsonx.Bool(fields["is_error"]); ok {
			isErr = b
		}
		var content []ir.Part
		contentRaw, hasContent := fields["content"]
		if hasContent && !jsonx.IsNull(contentRaw) {
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
					part, okPart, err := parseToolResultContentBlock(elem, p.Key("content").At(i), env)
					if err != nil {
						return ir.Part{}, err
					}
					if okPart {
						content = append(content, part)
					}
				}
			default:
				return ir.Part{}, env.Err(ir.Invalid(p.Key("content"), "tool_result content must be a string or array"))
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
		r := &ir.Reasoning{Text: thinking}
		if sig != "" {
			if rec, ok := env.Sealer().Unseal(sig); ok {
				if rec.TextSum == "" || rec.TextSum == ir.TextSum(thinking) {
					switch rec.Kind {
					case ir.SealSignature:
						r.Signature = &ir.Signature{Value: rec.Value, Origin: rec.Origin}
					case ir.SealEncrypted:
						r.Sealed = &ir.Sealed{Kind: ir.SealedResponsesEncrypted, Data: json.RawMessage(rec.Value), Origin: rec.Origin}
					}
				}
			} else {
				r.Signature = &ir.Signature{Value: sig, Origin: ir.Origin{Format: ir.FormatAnthropic, Provider: env.ProviderID}}
			}
		}
		return ir.Part{
			Type:      ir.PartReasoning,
			Reasoning: r,
		}, nil

	case "redacted_thinking":
		data, _ := jsonx.String(fields["data"])
		if rec, ok := env.Sealer().Unseal(data); ok {
			switch rec.Kind {
			case ir.SealEncrypted:
				return ir.Part{
					Type: ir.PartReasoning,
					Reasoning: &ir.Reasoning{
						Sealed: &ir.Sealed{
							Kind:   ir.SealedResponsesEncrypted,
							Data:   json.RawMessage(rec.Value),
							Origin: rec.Origin,
						},
					},
				}, nil
			case ir.SealRedacted:
				return ir.Part{
					Type: ir.PartReasoning,
					Reasoning: &ir.Reasoning{
						Sealed: &ir.Sealed{
							Kind:   ir.SealedAnthropicRedacted,
							Data:   json.RawMessage(rec.Value),
							Origin: rec.Origin,
						},
					},
				}, nil
			}
		}
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

func parseToolResultContentBlock(raw json.RawMessage, p ir.Path, env *ir.Env) (ir.Part, bool, error) {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return ir.Part{}, false, env.Err(ir.Invalid(p, "must be an object"))
	}
	fields := map[string]json.RawMessage{}
	var typ string
	for _, kv := range kvs {
		fields[kv.Key] = kv.Value
		if kv.Key == "type" {
			typ, _ = jsonx.String(kv.Value)
		}
	}
	if typ == "" {
		return ir.Part{}, false, env.Err(ir.Invalid(p.Key("type"), "tool_result content block type is required"))
	}
	switch typ {
	case "text", "image", "document":
		pt, err := parseContentBlock(raw, p, ir.RoleUser, true, env)
		if err != nil {
			return ir.Part{}, false, err
		}
		return pt, true, nil
	case "search_result":
		txt, err := parseSearchResultBlock(fields, p, env)
		if err != nil {
			return ir.Part{}, false, err
		}
		return ir.TextPart(txt), true, nil
	case "tool_result", "tool_use", "server_tool_use", "thinking", "redacted_thinking":
		return ir.Part{}, false, env.Err(ir.Invalid(p.Key("type"), typ+" is not allowed inside a tool_result"))
	default:
		// Extension blocks inside tool_result (e.g. tool_reference) are accepted and skipped.
		return ir.Part{}, false, nil
	}
}

func parseSearchResultBlock(fields map[string]json.RawMessage, p ir.Path, env *ir.Env) (string, error) {
	title, _ := jsonx.String(fields["title"])
	source, _ := jsonx.String(fields["source"])
	lines := []string{"[search result] " + title}
	if source != "" {
		lines = append(lines, "source: "+source)
	}
	if cr, ok := fields["content"]; ok && !jsonx.IsNull(cr) {
		elems, okArr := jsonx.Array(cr)
		if !okArr {
			return "", env.Err(ir.Invalid(p.Key("content"), "search_result content must be an array"))
		}
		for i, elem := range elems {
			ekvs, err := jsonx.Object(elem)
			if err != nil {
				return "", env.Err(ir.Invalid(p.Key("content").At(i), "must be an object"))
			}
			var etyp, etext string
			for _, kv := range ekvs {
				switch kv.Key {
				case "type":
					etyp, _ = jsonx.String(kv.Value)
				case "text":
					etext, _ = jsonx.String(kv.Value)
				}
			}
			if etyp == "text" && etext != "" {
				lines = append(lines, etext)
			}
		}
	}
	return strings.Join(lines, "\n"), nil
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
		if media.MediaType == "" {
			return nil, env.Err(ir.Invalid(p.Key("media_type"), "base64 source requires media_type"))
		}
		if media.Data == "" {
			return nil, env.Err(ir.Invalid(p.Key("data"), "base64 source requires data"))
		}
	case "url":
		media.URL, _ = jsonx.String(fields["url"])
		low := strings.ToLower(media.URL)
		if !strings.HasPrefix(low, "https://") && !strings.HasPrefix(low, "http://") {
			return nil, env.Err(ir.Invalid(p.Key("url"), "url source must start with http:// or https://"))
		}
	case "file":
		media.FileID, _ = jsonx.String(fields["file_id"])
		if media.FileID == "" {
			return nil, env.Err(ir.Invalid(p.Key("file_id"), "file source requires file_id"))
		}
	default:
		return nil, env.Err(ir.Invalid(p.Key("type"), "unknown source type: "+typ))
	}
	return media, nil
}

func parseDocumentSource(raw json.RawMessage, p ir.Path, env *ir.Env) (*ir.Media, error) {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return nil, env.Err(ir.Invalid(p, "must be an object"))
	}
	fields := map[string]json.RawMessage{}
	for _, kv := range kvs {
		fields[kv.Key] = kv.Value
	}
	typ, _ := jsonx.String(fields["type"])
	switch typ {
	case "text":
		media := &ir.Media{}
		media.MediaType, _ = jsonx.String(fields["media_type"])
		if media.MediaType == "" {
			media.MediaType = "text/plain"
		}
		media.Text, _ = jsonx.String(fields["data"])
		return media, nil
	case "content":
		media := &ir.Media{MediaType: "text/plain"}
		cr := fields["content"]
		if jsonx.Kind(cr) == '"' {
			media.Text, _ = jsonx.String(cr)
			return media, nil
		}
		elems, ok := jsonx.Array(cr)
		if !ok {
			return nil, env.Err(ir.Invalid(p.Key("content"), "content source must be a string or array"))
		}
		var lines []string
		for _, elem := range elems {
			ekvs, err := jsonx.Object(elem)
			if err != nil {
				continue
			}
			var etyp, etext string
			for _, kv := range ekvs {
				switch kv.Key {
				case "type":
					etyp, _ = jsonx.String(kv.Value)
				case "text":
					etext, _ = jsonx.String(kv.Value)
				}
			}
			if etyp == "text" {
				lines = append(lines, etext)
			}
		}
		media.Text = strings.Join(lines, "\n")
		return media, nil
	default:
		return parseImageSource(raw, p, env)
	}
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

func parseTools(raw json.RawMessage, p ir.Path, env *ir.Env) ([]ir.Tool, []ir.BuiltinTool, error) {
	if jsonx.IsNull(raw) {
		return nil, nil, nil
	}
	elems, ok := jsonx.Array(raw)
	if !ok {
		return nil, nil, env.Err(ir.Invalid(p, "must be an array"))
	}
	var tools []ir.Tool
	var builtins []ir.BuiltinTool
	seenNames := map[string]bool{}
	for i, elem := range elems {
		fp := p.At(i)
		t, bt, err := parseTool(elem, fp, env)
		if err != nil {
			return nil, nil, err
		}
		if bt != nil {
			builtins = append(builtins, *bt)
		} else {
			if seenNames[t.Name] {
				return nil, nil, env.Err(ir.Invalid(fp.Key("name"), "duplicate tool name: "+t.Name))
			}
			seenNames[t.Name] = true
			tools = append(tools, t)
		}
	}
	return tools, builtins, nil
}

func parseTool(raw json.RawMessage, p ir.Path, env *ir.Env) (ir.Tool, *ir.BuiltinTool, error) {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return ir.Tool{}, nil, env.Err(ir.Invalid(p, "must be an object"))
	}
	fields := map[string]json.RawMessage{}
	for _, kv := range kvs {
		fields[kv.Key] = kv.Value
	}
	typ, _ := jsonx.String(fields["type"])
	switch {
	case strings.HasPrefix(typ, "web_search"):
		return ir.Tool{}, &ir.BuiltinTool{
			Kind:   ir.BuiltinWebSearch,
			Origin: ir.FormatAnthropic,
			Raw:    append(json.RawMessage(nil), raw...),
		}, nil
	case strings.HasPrefix(typ, "web_fetch"):
		return ir.Tool{}, &ir.BuiltinTool{
			Kind:   ir.BuiltinWebFetch,
			Origin: ir.FormatAnthropic,
			Raw:    append(json.RawMessage(nil), raw...),
		}, nil
	case typ != "" && typ != "custom":
		return ir.Tool{}, nil, env.Err(ir.Unsupported(p.Key("type"), typ, "unsupported Anthropic tool type"))
	}
	name, _ := jsonx.String(fields["name"])
	if strings.TrimSpace(name) == "" {
		return ir.Tool{}, nil, env.Err(ir.Invalid(p.Key("name"), "tool name is required"))
	}
	desc, _ := jsonx.String(fields["description"])
	inputSchema := fields["input_schema"]
	if len(inputSchema) == 0 || jsonx.IsNull(inputSchema) {
		inputSchema = schema.EmptyObject
	} else if !schemaAllowsObject(inputSchema) {
		return ir.Tool{}, nil, env.Err(ir.Invalid(p.Key("input_schema"), "input_schema top-level type must allow object"))
	}
	t := ir.Tool{
		Name:        name,
		Description: desc,
		Kind:        ir.ToolFunction,
		Schema:      inputSchema,
	}
	t.Cache = parseCacheControl(fields["cache_control"])
	if v, ok := fields["strict"]; ok {
		if b, ok2 := jsonx.Bool(v); ok2 {
			t.Strict = &b
		}
	}
	return t, nil, nil
}

func parseToolChoice(raw json.RawMessage, p ir.Path, env *ir.Env) (ir.ToolChoice, *bool, error) {
	if jsonx.IsNull(raw) {
		return ir.ToolChoice{}, nil, nil
	}
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return ir.ToolChoice{}, nil, env.Err(ir.Invalid(p, "must be an object"))
	}
	var typ, name string
	var parallelPtr *bool
	for _, kv := range kvs {
		switch kv.Key {
		case "type":
			typ, _ = jsonx.String(kv.Value)
		case "name":
			name, _ = jsonx.String(kv.Value)
		case "disable_parallel_tool_use":
			if b, ok := jsonx.Bool(kv.Value); ok && b {
				f := false
				parallelPtr = &f
			}
		}
	}
	switch typ {
	case "auto":
		return ir.ToolChoice{Mode: ir.ToolChoiceAuto}, parallelPtr, nil
	case "any":
		return ir.ToolChoice{Mode: ir.ToolChoiceRequired}, parallelPtr, nil
	case "tool":
		return ir.ToolChoice{Mode: ir.ToolChoiceTool, Name: name}, parallelPtr, nil
	case "none":
		return ir.ToolChoice{Mode: ir.ToolChoiceNone}, parallelPtr, nil
	}
	return ir.ToolChoice{}, nil, env.Err(ir.Invalid(p.Key("type"), "unknown tool_choice type: "+typ))
}

func parseOutputConfig(raw json.RawMessage, p ir.Path, req *ir.Request, env *ir.Env) error {
	if jsonx.IsNull(raw) {
		return nil
	}
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return env.Err(ir.Invalid(p, "must be an object"))
	}
	for _, kv := range kvs {
		switch kv.Key {
		case "effort":
			s, ok := jsonx.String(kv.Value)
			if !ok {
				return env.Err(ir.Invalid(p.Key("effort"), "must be a string"))
			}
			eff, ok := ir.ParseEffort(s)
			if !ok {
				return env.Err(ir.Invalid(p.Key("effort"), "unrecognised effort: "+s))
			}
			if req.Reasoning == nil {
				req.Reasoning = &ir.ReasoningConfig{}
			}
			req.Reasoning.Effort = eff
		case "format":
			fkvs, err := jsonx.Object(kv.Value)
			if err != nil {
				return env.Err(ir.Invalid(p.Key("format"), "must be an object"))
			}
			var ftype string
			var fschema json.RawMessage
			for _, fkv := range fkvs {
				switch fkv.Key {
				case "type":
					ftype, _ = jsonx.String(fkv.Value)
				case "schema":
					fschema = append(json.RawMessage(nil), fkv.Value...)
				}
			}
			switch ftype {
			case "json_schema":
				req.Output = &ir.OutputFormat{Type: ir.OutputJSONSchema, Schema: fschema}
			case "text":
				req.Output = &ir.OutputFormat{Type: ir.OutputText}
			default:
				return env.Err(ir.Invalid(p.Keys("format", "type"), "unknown output_config.format type: "+ftype))
			}
		}
	}
	return nil
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
			case "adaptive":
				rc.Mode = ir.ReasoningAdaptive
			default:
				return nil, env.Err(ir.Invalid(p.Key("type"), "unknown thinking type: "+t))
			}
		case "budget_tokens":
			if jsonx.IsNull(kv.Value) {
				break
			}
			n, ok := jsonx.Int(kv.Value)
			if !ok {
				return nil, env.Err(ir.Invalid(p.Key("budget_tokens"), "must be an integer"))
			}
			ni := int(n)
			rc.Budget = &ni
		case "display":
			if s, ok := jsonx.String(kv.Value); ok {
				rc.Display = s
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
		if s == "" {
			return nil, env.Err(ir.Invalid(p.At(i), "stop sequence must not be empty"))
		}
		stops = append(stops, s)
	}
	return stops, nil
}

func schemaAllowsObject(raw json.RawMessage) bool {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return false
	}
	for _, kv := range kvs {
		if kv.Key != "type" || jsonx.IsNull(kv.Value) {
			continue
		}
		switch jsonx.Kind(kv.Value) {
		case '"':
			s, _ := jsonx.String(kv.Value)
			return s == "object"
		case '[':
			elems, ok := jsonx.Array(kv.Value)
			if !ok {
				return false
			}
			hasObj := false
			for _, e := range elems {
				s, okStr := jsonx.String(e)
				if !okStr {
					return false
				}
				if s == "object" {
					hasObj = true
				}
			}
			return hasObj
		default:
			return false
		}
	}
	return true
}

func parseContextManagement(raw json.RawMessage, p ir.Path, env *ir.Env) (*int, error) {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return nil, env.Err(ir.Invalid(p, "must be an object"))
	}
	var keepTurns *int
	for _, kv := range kvs {
		if kv.Key != "edits" || jsonx.IsNull(kv.Value) {
			continue
		}
		elems, ok := jsonx.Array(kv.Value)
		if !ok {
			return nil, env.Err(ir.Invalid(p.Key("edits"), "must be an array"))
		}
		for i, elem := range elems {
			ep := p.Key("edits").At(i)
			ekvs, err := jsonx.Object(elem)
			if err != nil {
				return nil, env.Err(ir.Invalid(ep, "must be an object"))
			}
			var editType string
			var keepRaw json.RawMessage
			for _, ekv := range ekvs {
				switch ekv.Key {
				case "type":
					editType, _ = jsonx.String(ekv.Value)
				case "keep":
					keepRaw = ekv.Value
				}
			}
			if isVersionedContextEdit(editType, "clear_thinking_") {
				if n, ok := jsonx.Int(keepRaw); ok {
					ni := int(n)
					keepTurns = &ni
				} else if kkvs, err := jsonx.Object(keepRaw); err == nil {
					for _, kkv := range kkvs {
						if kkv.Key == "value" {
							if n, ok := jsonx.Int(kkv.Value); ok {
								ni := int(n)
								keepTurns = &ni
							}
						}
					}
				}
				continue
			}
			if isVersionedContextEdit(editType, "clear_tool_uses_") {
				continue
			}
			if err := env.Lose(ir.LossUnknownField, ep.Key("type"), "unsupported context_management edit type: "+editType); err != nil {
				return nil, err
			}
		}
	}
	return keepTurns, nil
}

func isVersionedContextEdit(typ, prefix string) bool {
	if !strings.HasPrefix(typ, prefix) {
		return false
	}
	ver := typ[len(prefix):]
	if len(ver) != 8 {
		return false
	}
	for i := 0; i < 8; i++ {
		if ver[i] < '0' || ver[i] > '9' {
			return false
		}
	}
	return true
}
