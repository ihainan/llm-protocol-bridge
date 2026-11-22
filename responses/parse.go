package responses

import (
	"encoding/json"
	"strings"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// ParseRequest decodes a Responses API request body into an ir.Request.
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

	req := &ir.Request{Source: ir.FormatResponses}
	p := ir.Root

	// First pass: collect raw fields.
	fields := map[string]json.RawMessage{}
	for _, kv := range kvs {
		fields[kv.Key] = kv.Value
	}

	// model
	if v, ok := fields["model"]; ok {
		req.Model, _ = jsonx.String(v)
	}

	// stream
	if v, ok := fields["stream"]; ok {
		if b, ok2 := jsonx.Bool(v); ok2 {
			req.Stream = b
		}
	}

	// instructions → prepend as a system message in Instructions
	if v, ok := fields["instructions"]; ok && !jsonx.IsNull(v) {
		s, ok2 := jsonx.String(v)
		if !ok2 {
			return nil, env.Err(ir.Invalid(p.Key("instructions"), "must be a string"))
		}
		if s != "" {
			req.Instructions = []ir.Part{ir.TextPart(s)}
		}
	}

	// Reject stateful/asynchronous Responses features that cannot be converted statelessly.
	if v, ok := fields["conversation"]; ok && !jsonx.IsNull(v) {
		return nil, env.Err(ir.Unsupported(p.Key("conversation"), "conversation", "stateful conversation is not supported"))
	}
	if v, ok := fields["background"]; ok && !jsonx.IsNull(v) {
		if b, ok2 := jsonx.Bool(v); ok2 && b {
			return nil, env.Err(ir.Unsupported(p.Key("background"), "background", "background execution is not supported"))
		}
	}

	// input: string or array
	if v, ok := fields["input"]; ok && !jsonx.IsNull(v) {
		msgs, err := parseInput(v, p.Key("input"), env)
		if err != nil {
			return nil, err
		}
		// Extract leading system/developer messages into Instructions.
		i := 0
		for i < len(msgs) && (msgs[i].Role == ir.RoleSystem || msgs[i].Role == ir.RoleDeveloper) {
			req.Instructions = append(req.Instructions, msgs[i].Parts...)
			i++
		}
		req.Messages = msgs[i:]
	}

	// tools
	if v, ok := fields["tools"]; ok && !jsonx.IsNull(v) {
		tools, builtins, err := parseTools(v, p.Key("tools"), env)
		if err != nil {
			return nil, err
		}
		req.Tools = tools
		req.Builtins = builtins
	}

	// tool_choice
	if v, ok := fields["tool_choice"]; ok && !jsonx.IsNull(v) {
		tc, err := parseToolChoice(v, p.Key("tool_choice"), env)
		if err != nil {
			return nil, err
		}
		req.ToolChoice = tc
	}

	// parallel_tool_calls
	if v, ok := fields["parallel_tool_calls"]; ok {
		if b, ok2 := jsonx.Bool(v); ok2 {
			req.ParallelToolCalls = &b
		}
	}

	// sampling
	if v, ok := fields["temperature"]; ok {
		if f, ok2 := jsonx.Float(v); ok2 {
			req.Temperature = &f
		}
	}
	if v, ok := fields["top_p"]; ok {
		if f, ok2 := jsonx.Float(v); ok2 {
			req.TopP = &f
		}
	}
	if v, ok := fields["max_output_tokens"]; ok {
		if n, ok2 := jsonx.Int(v); ok2 {
			ni := int(n)
			req.MaxOutputTokens = &ni
		}
	}
	if v, ok := fields["seed"]; ok {
		if n, ok2 := jsonx.Int(v); ok2 {
			req.Seed = &n
		}
	}
	if v, ok := fields["presence_penalty"]; ok {
		if f, ok2 := jsonx.Float(v); ok2 {
			req.PresencePenalty = &f
		}
	}
	if v, ok := fields["frequency_penalty"]; ok {
		if f, ok2 := jsonx.Float(v); ok2 {
			req.FrequencyPenalty = &f
		}
	}

	// stop: string or []string
	if v, ok := fields["stop"]; ok && !jsonx.IsNull(v) {
		stops, err := parseStop(v, p.Key("stop"), env)
		if err != nil {
			return nil, err
		}
		req.Stop = stops
	}

	// reasoning
	if v, ok := fields["reasoning"]; ok && !jsonx.IsNull(v) {
		rc, err := parseReasoning(v, p.Key("reasoning"), env)
		if err != nil {
			return nil, err
		}
		req.Reasoning = rc
	}

	// text.format → OutputFormat
	if v, ok := fields["text"]; ok && !jsonx.IsNull(v) {
		of, err := parseTextFormat(v, p.Key("text"), env)
		if err != nil {
			return nil, err
		}
		req.Output = of
	}

	// include
	if v, ok := fields["include"]; ok && !jsonx.IsNull(v) {
		elems, ok2 := jsonx.Array(v)
		if !ok2 {
			return nil, env.Err(ir.Invalid(p.Key("include"), "must be an array"))
		}
		for _, e := range elems {
			s, _ := jsonx.String(e)
			if s != "" {
				req.Include = append(req.Include, s)
			}
		}
	}

	// store
	if v, ok := fields["store"]; ok {
		if b, ok2 := jsonx.Bool(v); ok2 {
			req.Store = &b
		}
	}

	// previous_response_id
	if v, ok := fields["previous_response_id"]; ok {
		req.PreviousResponseID, _ = jsonx.String(v)
	}

	// service_tier, user, metadata
	if v, ok := fields["service_tier"]; ok {
		req.ServiceTier, _ = jsonx.String(v)
	}
	if v, ok := fields["user"]; ok {
		req.User, _ = jsonx.String(v)
	}
	if v, ok := fields["metadata"]; ok {
		req.Metadata = append(json.RawMessage(nil), v...)
	}

	// Echo: preserve client fields for response echoing.
	echoFields := map[string]json.RawMessage{}
	for _, kv := range kvs {
		switch kv.Key {
		case "model", "input", "instructions", "tools", "tool_choice",
			"parallel_tool_calls", "temperature", "top_p", "max_output_tokens",
			"seed", "stop", "reasoning", "text", "include", "store",
			"previous_response_id", "stream", "service_tier", "user", "metadata",
			"presence_penalty", "frequency_penalty":
			// known, don't echo separately
		default:
			if err := env.UnknownParam(p.Key(kv.Key)); err != nil {
				return nil, err
			}
			echoFields[kv.Key] = kv.Value
		}
	}
	if len(echoFields) > 0 {
		if req.Passthrough == nil {
			req.Passthrough = map[string]json.RawMessage{}
		}
		for k, v := range echoFields {
			req.Passthrough[k] = v
		}
	}

	return req, nil
}

// parseInput handles the input field (string or array).
func parseInput(raw json.RawMessage, p ir.Path, env *ir.Env) ([]ir.Message, error) {
	switch jsonx.Kind(raw) {
	case '"':
		s, _ := jsonx.String(raw)
		return []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart(s)}}}, nil
	case '[':
		return parseInputItems(raw, p, env)
	}
	return nil, env.Err(ir.Invalid(p, "must be a string or array"))
}

// parseInputItems parses a Responses input array into ir.Messages.
// It groups consecutive function_call and reasoning items into one assistant
// message, and groups consecutive function_call_output items into one user message.
func parseInputItems(raw json.RawMessage, p ir.Path, env *ir.Env) ([]ir.Message, error) {
	elems, ok := jsonx.Array(raw)
	if !ok {
		return nil, env.Err(ir.Invalid(p, "must be an array"))
	}

	var msgs []ir.Message

	type pending struct {
		role  ir.Role
		parts []ir.Part
	}
	var cur *pending

	flush := func() {
		if cur != nil && len(cur.parts) > 0 {
			msgs = append(msgs, ir.Message{Role: cur.role, Parts: cur.parts})
		}
		cur = nil
	}
	ensure := func(role ir.Role) *pending {
		if cur == nil || cur.role != role {
			flush()
			cur = &pending{role: role}
		}
		return cur
	}

	for i, elem := range elems {
		fp := p.At(i)
		kvs, err := jsonx.Object(elem)
		if err != nil {
			return nil, env.Err(ir.Invalid(fp, "must be an object"))
		}
		fields := map[string]json.RawMessage{}
		for _, kv := range kvs {
			fields[kv.Key] = kv.Value
		}
		itemType, _ := jsonx.String(fields["type"])
		if itemType == "" && fields["role"] != nil {
			itemType = "message"
		}

		switch itemType {
		case "message":
			flush()
			msg, err := parseMessageItem(fields, fp, env)
			if err != nil {
				return nil, err
			}
			msgs = append(msgs, msg)

		case "reasoning":
			c := ensure(ir.RoleAssistant)
			part, err := parseReasoningItem(fields, fp, env)
			if err != nil {
				return nil, err
			}
			c.parts = append(c.parts, part)

		case "function_call":
			c := ensure(ir.RoleAssistant)
			part, err := parseFunctionCallItem(fields, fp, env)
			if err != nil {
				return nil, err
			}
			c.parts = append(c.parts, part)

		case "function_call_output":
			c := ensure(ir.RoleUser)
			part, err := parseFunctionCallOutputItem(fields, fp, env)
			if err != nil {
				return nil, err
			}
			c.parts = append(c.parts, part)

		case "custom_tool_call":
			c := ensure(ir.RoleAssistant)
			part, err := parseCustomToolCallItem(fields, fp, env)
			if err != nil {
				return nil, err
			}
			c.parts = append(c.parts, part)

		case "custom_tool_call_output":
			c := ensure(ir.RoleUser)
			part, err := parseCustomToolCallOutputItem(fields, fp, env)
			if err != nil {
				return nil, err
			}
			c.parts = append(c.parts, part)

		case "item_reference":
			return nil, env.Err(ir.Unsupported(fp.Key("type"), "item_reference", "item_reference is not supported"))

		default:
			return nil, env.Err(ir.Invalid(fp.Key("type"), "unknown input item type: "+itemType))
		}
	}
	flush()
	return msgs, nil
}

func parseMessageItem(fields map[string]json.RawMessage, p ir.Path, env *ir.Env) (ir.Message, error) {
	roleStr, _ := jsonx.String(fields["role"])
	var role ir.Role
	switch roleStr {
	case "user", "":
		role = ir.RoleUser
	case "assistant":
		role = ir.RoleAssistant
	case "system":
		role = ir.RoleSystem
	case "developer":
		role = ir.RoleDeveloper
	default:
		return ir.Message{}, env.Err(ir.Invalid(p.Key("role"), "unknown role: "+roleStr))
	}

	var parts []ir.Part
	if cv, ok := fields["content"]; ok && !jsonx.IsNull(cv) {
		var err error
		parts, err = parseMessageContent(cv, p.Key("content"), env)
		if err != nil {
			return ir.Message{}, err
		}
	}

	name, _ := jsonx.String(fields["name"])
	return ir.Message{Role: role, Parts: parts, Name: name}, nil
}

func parseMessageContent(raw json.RawMessage, p ir.Path, env *ir.Env) ([]ir.Part, error) {
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
			part, err := parseContentPart(elem, p.At(i), env)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		}
		return parts, nil
	}
	return nil, env.Err(ir.Invalid(p, "must be a string or array"))
}

func parseContentPart(raw json.RawMessage, p ir.Path, env *ir.Env) (ir.Part, error) {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return ir.Part{}, env.Err(ir.Invalid(p, "must be an object"))
	}
	fields := map[string]json.RawMessage{}
	for _, kv := range kvs {
		fields[kv.Key] = kv.Value
	}
	partType, _ := jsonx.String(fields["type"])

	switch partType {
	case "input_text", "output_text", "text":
		text, _ := jsonx.String(fields["text"])
		return ir.TextPart(text), nil

	case "refusal":
		text, _ := jsonx.String(fields["refusal"])
		return ir.Part{Type: ir.PartRefusal, Text: text}, nil

	case "input_image":
		imageURL, _ := jsonx.String(fields["image_url"])
		detail, _ := jsonx.String(fields["detail"])
		media := &ir.Media{URL: imageURL, Detail: detail}
		// Detect inline data URIs.
		if strings.HasPrefix(imageURL, "data:") {
			mediaType, data, ok := parseDataURI(imageURL)
			if ok {
				media.MediaType = mediaType
				media.Data = data
				media.URL = ""
			}
		}
		return ir.Part{Type: ir.PartImage, Media: media}, nil

	case "input_file":
		fileID, _ := jsonx.String(fields["file_id"])
		filename, _ := jsonx.String(fields["filename"])
		fileData, _ := jsonx.String(fields["file_data"])
		media := &ir.Media{FileID: fileID, Filename: filename, Data: fileData}
		return ir.Part{Type: ir.PartFile, Media: media}, nil

	default:
		return ir.Part{}, env.Err(ir.Invalid(p.Key("type"), "unknown content part type: "+partType))
	}
}

func parseReasoningItem(fields map[string]json.RawMessage, p ir.Path, env *ir.Env) (ir.Part, error) {
	id, _ := jsonx.String(fields["id"])
	enc, _ := jsonx.String(fields["encrypted_content"])

	if enc != "" {
		if rec, ok := env.Sealer().Unseal(enc); ok {
			switch rec.Kind {
			case ir.SealRedacted:
				return ir.Part{
					Type: ir.PartReasoning,
					Reasoning: &ir.Reasoning{
						ID: id,
						Sealed: &ir.Sealed{
							Kind:   ir.SealedAnthropicRedacted,
							Data:   json.RawMessage(rec.Value),
							Origin: rec.Origin,
						},
					},
				}, nil
			case ir.SealSignature:
				return ir.Part{
					Type: ir.PartReasoning,
					Reasoning: &ir.Reasoning{
						ID:        id,
						Text:      rec.Text,
						Signature: &ir.Signature{Value: rec.Value, Origin: rec.Origin},
					},
				}, nil
			case ir.SealEncrypted:
				return ir.Part{
					Type: ir.PartReasoning,
					Reasoning: &ir.Reasoning{
						ID: id,
						Sealed: &ir.Sealed{
							Kind:   ir.SealedResponsesEncrypted,
							Data:   json.RawMessage(rec.Value),
							Origin: rec.Origin,
						},
					},
				}, nil
			}
		}
		// Sealed reasoning: stash the encrypted_content in Reasoning.Sealed.
		data, _ := json.Marshal(enc)
		return ir.Part{
			Type: ir.PartReasoning,
			Reasoning: &ir.Reasoning{
				ID: id,
				Sealed: &ir.Sealed{
					Kind:   ir.SealedResponsesEncrypted,
					Data:   data,
					Origin: ir.Origin{Format: ir.FormatResponses, Provider: env.ProviderID},
				},
			},
		}, nil
	}

	// Summary parts.
	var summaryText strings.Builder
	if sv, ok := fields["summary"]; ok {
		elems, _ := jsonx.Array(sv)
		for _, e := range elems {
			kvs2, err := jsonx.Object(e)
			if err != nil {
				continue
			}
			for _, kv := range kvs2 {
				if kv.Key == "text" {
					s, _ := jsonx.String(kv.Value)
					summaryText.WriteString(s)
				}
			}
		}
	}

	return ir.Part{
		Type:      ir.PartReasoning,
		Reasoning: &ir.Reasoning{ID: id, Text: summaryText.String()},
	}, nil
}

func parseFunctionCallItem(fields map[string]json.RawMessage, p ir.Path, env *ir.Env) (ir.Part, error) {
	callID, _ := jsonx.String(fields["call_id"])
	name, _ := jsonx.String(fields["name"])
	argsStr, _ := jsonx.String(fields["arguments"])
	args := json.RawMessage(argsStr)
	if len(args) == 0 {
		args = json.RawMessage(`{}`)
	}
	return ir.Part{
		Type: ir.PartToolCall,
		Call: &ir.ToolCall{
			ID:        callID,
			Name:      name,
			Kind:      ir.ToolFunction,
			Arguments: args,
		},
	}, nil
}

func parseFunctionCallOutputItem(fields map[string]json.RawMessage, p ir.Path, env *ir.Env) (ir.Part, error) {
	callID, _ := jsonx.String(fields["call_id"])
	parts, err := parseToolOutputContent(fields["output"], p.Key("output"), env)
	if err != nil {
		return ir.Part{}, err
	}
	var isErr bool
	if v, ok := fields["is_error"]; ok {
		b, _ := jsonx.Bool(v)
		isErr = b
	}
	return ir.Part{
		Type: ir.PartToolResult,
		Result: &ir.ToolResult{
			CallID:  callID,
			IsError: isErr,
			Content: parts,
		},
	}, nil
}

func parseToolOutputContent(raw json.RawMessage, p ir.Path, env *ir.Env) ([]ir.Part, error) {
	if len(raw) == 0 || jsonx.IsNull(raw) {
		return []ir.Part{ir.TextPart("")}, nil
	}
	if jsonx.Kind(raw) == '[' {
		elems, ok := jsonx.Array(raw)
		if !ok {
			return nil, env.Err(ir.Invalid(p, "output must be a string or array"))
		}
		var parts []ir.Part
		for i, elem := range elems {
			pt, err := parseContentPart(elem, p.At(i), env)
			if err != nil {
				return nil, err
			}
			parts = append(parts, pt)
		}
		return parts, nil
	}
	s, _ := jsonx.String(raw)
	return []ir.Part{ir.TextPart(s)}, nil
}

func parseCustomToolCallItem(fields map[string]json.RawMessage, p ir.Path, env *ir.Env) (ir.Part, error) {
	callID, _ := jsonx.String(fields["call_id"])
	name, _ := jsonx.String(fields["name"])
	ns, _ := jsonx.String(fields["namespace"])
	input, _ := jsonx.String(fields["input"])
	args, _ := json.Marshal(map[string]string{"input": input})
	return ir.Part{
		Type: ir.PartToolCall,
		Call: &ir.ToolCall{
			ID:        callID,
			Name:      name,
			Namespace: ns,
			Kind:      ir.ToolCustom,
			Arguments: args,
		},
	}, nil
}

func parseCustomToolCallOutputItem(fields map[string]json.RawMessage, p ir.Path, env *ir.Env) (ir.Part, error) {
	callID, _ := jsonx.String(fields["call_id"])
	parts, err := parseToolOutputContent(fields["output"], p.Key("output"), env)
	if err != nil {
		return ir.Part{}, err
	}
	var isErr bool
	if v, ok := fields["is_error"]; ok {
		b, _ := jsonx.Bool(v)
		isErr = b
	}
	return ir.Part{
		Type: ir.PartToolResult,
		Result: &ir.ToolResult{
			CallID:  callID,
			Kind:    ir.ToolCustom,
			IsError: isErr,
			Content: parts,
		},
	}, nil
}

func parseTools(raw json.RawMessage, p ir.Path, env *ir.Env) ([]ir.Tool, []ir.BuiltinTool, error) {
	elems, ok := jsonx.Array(raw)
	if !ok {
		return nil, nil, env.Err(ir.Invalid(p, "must be an array"))
	}
	var tools []ir.Tool
	var builtins []ir.BuiltinTool
	for i, elem := range elems {
		fp := p.At(i)
		kvs, err := jsonx.Object(elem)
		if err != nil {
			return nil, nil, env.Err(ir.Invalid(fp, "must be an object"))
		}
		fields := map[string]json.RawMessage{}
		for _, kv := range kvs {
			fields[kv.Key] = kv.Value
		}
		toolType, _ := jsonx.String(fields["type"])
		switch toolType {
		case "function":
			t, err := parseFunctionTool(fields, fp, env)
			if err != nil {
				return nil, nil, err
			}
			tools = append(tools, t)
		case "custom":
			t, err := parseCustomTool(fields, fp, env)
			if err != nil {
				return nil, nil, err
			}
			tools = append(tools, t)
		case "namespace":
			nsName, _ := jsonx.String(fields["name"])
			if nsName == "" {
				return nil, nil, env.Err(ir.Invalid(fp.Key("name"), "namespace name is required"))
			}
			subRaw, ok := fields["tools"]
			if !ok || jsonx.IsNull(subRaw) {
				break
			}
			subTools, subBuiltins, err := parseTools(subRaw, fp.Key("tools"), env)
			if err != nil {
				return nil, nil, err
			}
			for j := range subTools {
				subTools[j].Namespace = nsName
			}
			tools = append(tools, subTools...)
			builtins = append(builtins, subBuiltins...)
		case "web_search", "web_search_preview":
			builtins = append(builtins, ir.BuiltinTool{
				Kind:   ir.BuiltinWebSearch,
				Origin: ir.FormatResponses,
				Raw:    append(json.RawMessage(nil), elem...),
			})
		case "computer_use":
			return nil, nil, env.Err(ir.Unsupported(fp.Key("type"), toolType, "computer_use tool"))
		default:
			return nil, nil, env.Err(ir.Unsupported(fp.Key("type"), toolType, "unsupported tool type: "+toolType))
		}
	}
	return tools, builtins, nil
}

func parseCustomTool(fields map[string]json.RawMessage, p ir.Path, env *ir.Env) (ir.Tool, error) {
	name, _ := jsonx.String(fields["name"])
	if name == "" {
		return ir.Tool{}, env.Err(ir.Invalid(p.Key("name"), "custom tool name is required"))
	}
	desc, _ := jsonx.String(fields["description"])
	t := ir.Tool{
		Kind:        ir.ToolCustom,
		Name:        name,
		Description: desc,
	}
	if fv, ok := fields["format"]; ok && !jsonx.IsNull(fv) {
		if kvs, err := jsonx.Object(fv); err == nil {
			g := &ir.Grammar{}
			for _, kv := range kvs {
				switch kv.Key {
				case "type":
					g.Type, _ = jsonx.String(kv.Value)
				case "syntax":
					g.Syntax, _ = jsonx.String(kv.Value)
				case "definition":
					g.Definition, _ = jsonx.String(kv.Value)
				}
			}
			t.Grammar = g
		}
	}
	return t, nil
}

func parseFunctionTool(fields map[string]json.RawMessage, p ir.Path, env *ir.Env) (ir.Tool, error) {
	name, _ := jsonx.String(fields["name"])
	if name == "" {
		return ir.Tool{}, env.Err(ir.Invalid(p.Key("name"), "tool name is required"))
	}
	desc, _ := jsonx.String(fields["description"])
	t := ir.Tool{
		Kind:        ir.ToolFunction,
		Name:        name,
		Description: desc,
	}
	if v, ok := fields["parameters"]; ok {
		t.Schema = append(json.RawMessage(nil), v...)
	}
	if v, ok := fields["strict"]; ok {
		if b, ok2 := jsonx.Bool(v); ok2 {
			t.Strict = &b
		}
	}
	return t, nil
}

func parseToolChoice(raw json.RawMessage, p ir.Path, env *ir.Env) (ir.ToolChoice, error) {
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
		}
		return ir.ToolChoice{}, env.Err(ir.Invalid(p, "unknown tool_choice string: "+s))
	case '{':
		kvs, err := jsonx.Object(raw)
		if err != nil {
			return ir.ToolChoice{}, env.Err(ir.Invalid(p, "must be a string or object"))
		}
		fields := map[string]json.RawMessage{}
		for _, kv := range kvs {
			fields[kv.Key] = kv.Value
		}
		tcType, _ := jsonx.String(fields["type"])
		switch tcType {
		case "auto":
			return ir.ToolChoice{Mode: ir.ToolChoiceAuto}, nil
		case "none":
			return ir.ToolChoice{Mode: ir.ToolChoiceNone}, nil
		case "required":
			return ir.ToolChoice{Mode: ir.ToolChoiceRequired}, nil
		case "function":
			name, _ := jsonx.String(fields["name"])
			return ir.ToolChoice{Mode: ir.ToolChoiceTool, Name: name}, nil
		case "allowed_tools":
			var allowed []ir.ToolRef
			var allowedMode ir.ToolChoiceMode
			if mv, ok := fields["mode"]; ok {
				ms, _ := jsonx.String(mv)
				switch ms {
				case "auto":
					allowedMode = ir.ToolChoiceAuto
				case "required":
					allowedMode = ir.ToolChoiceRequired
				}
			}
			// Official schema: "tools": [{"type":"function","name":"..."}]
			if v, ok := fields["tools"]; ok && !jsonx.IsNull(v) {
				elems, ok2 := jsonx.Array(v)
				if !ok2 {
					return ir.ToolChoice{}, env.Err(ir.Invalid(p.Key("tools"), "must be an array"))
				}
				for _, e := range elems {
					if kvs2, err := jsonx.Object(e); err == nil {
						var ref ir.ToolRef
						for _, kv2 := range kvs2 {
							switch kv2.Key {
							case "type":
								if s, ok := jsonx.String(kv2.Value); ok {
									if s == "custom" {
										ref.Kind = ir.ToolCustom
									} else if s == "function" {
										ref.Kind = ir.ToolFunction
									}
								}
							case "name":
								ref.Name, _ = jsonx.String(kv2.Value)
							case "namespace":
								ref.Namespace, _ = jsonx.String(kv2.Value)
							}
						}
						if ref.Name != "" {
							allowed = append(allowed, ref)
						}
					}
				}
			} else if v, ok := fields["allowed_tools"]; ok {
				elems, ok2 := jsonx.Array(v)
				if !ok2 {
					return ir.ToolChoice{}, env.Err(ir.Invalid(p.Key("allowed_tools"), "must be an array"))
				}
				for _, e := range elems {
					s, _ := jsonx.String(e)
					if s != "" {
						allowed = append(allowed, ir.ToolRef{Name: s})
					}
				}
			}
			return ir.ToolChoice{Mode: ir.ToolChoiceAllowed, Allowed: allowed, AllowedMode: allowedMode}, nil
		}
		return ir.ToolChoice{}, env.Err(ir.Invalid(p.Key("type"), "unknown tool_choice type: "+tcType))
	}
	return ir.ToolChoice{}, env.Err(ir.Invalid(p, "must be a string or object"))
}

func parseStop(raw json.RawMessage, p ir.Path, env *ir.Env) ([]string, error) {
	switch jsonx.Kind(raw) {
	case '"':
		s, _ := jsonx.String(raw)
		if s != "" {
			return []string{s}, nil
		}
		return nil, nil
	case '[':
		elems, ok := jsonx.Array(raw)
		if !ok {
			return nil, env.Err(ir.Invalid(p, "must be a string or array"))
		}
		var stops []string
		for i, e := range elems {
			s, ok2 := jsonx.String(e)
			if !ok2 {
				return nil, env.Err(ir.Invalid(p.At(i), "must be a string"))
			}
			stops = append(stops, s)
		}
		return stops, nil
	}
	return nil, env.Err(ir.Invalid(p, "must be a string or array"))
}

func parseReasoning(raw json.RawMessage, p ir.Path, env *ir.Env) (*ir.ReasoningConfig, error) {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return nil, env.Err(ir.Invalid(p, "must be an object"))
	}
	rc := &ir.ReasoningConfig{Mode: ir.ReasoningOn}
	for _, kv := range kvs {
		switch kv.Key {
		case "effort":
			if jsonx.IsNull(kv.Value) {
				break
			}
			s, ok := jsonx.String(kv.Value)
			if !ok {
				return nil, env.Err(ir.Invalid(p.Key("effort"), "must be a string"))
			}
			eff, ok := ir.ParseEffort(s)
			if !ok {
				return nil, env.Err(ir.Invalid(p.Key("effort"), "unrecognised reasoning effort: "+s))
			}
			rc.Effort = eff
			if eff == ir.EffortNone {
				rc.Mode = ir.ReasoningOff
			}
		case "summary":
			s, _ := jsonx.String(kv.Value)
			rc.Summary = s
		}
	}
	return rc, nil
}

func parseTextFormat(raw json.RawMessage, p ir.Path, env *ir.Env) (*ir.OutputFormat, error) {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return nil, env.Err(ir.Invalid(p, "must be an object"))
	}
	for _, kv := range kvs {
		if kv.Key == "format" && !jsonx.IsNull(kv.Value) {
			return parseResponseFormat(kv.Value, p.Key("format"), env)
		}
	}
	return nil, nil
}

func parseResponseFormat(raw json.RawMessage, p ir.Path, env *ir.Env) (*ir.OutputFormat, error) {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return nil, env.Err(ir.Invalid(p, "must be an object"))
	}
	fields := map[string]json.RawMessage{}
	for _, kv := range kvs {
		fields[kv.Key] = kv.Value
	}
	fmtType, _ := jsonx.String(fields["type"])
	of := &ir.OutputFormat{}
	switch fmtType {
	case "text":
		of.Type = ir.OutputText
	case "json_object":
		of.Type = ir.OutputJSONObject
	case "json_schema":
		of.Type = ir.OutputJSONSchema
		if v, ok := fields["schema"]; ok {
			of.Schema = append(json.RawMessage(nil), v...)
		}
		if v, ok := fields["name"]; ok {
			of.Name, _ = jsonx.String(v)
		}
		if v, ok := fields["strict"]; ok {
			if b, ok2 := jsonx.Bool(v); ok2 {
				of.Strict = &b
			}
		}
	default:
		return nil, env.Err(ir.Invalid(p.Key("type"), "unknown format type: "+fmtType))
	}
	return of, nil
}

// parseDataURI splits "data:<mediaType>;base64,<data>" into its parts.
func parseDataURI(uri string) (mediaType, data string, ok bool) {
	rest, found := strings.CutPrefix(uri, "data:")
	if !found {
		return "", "", false
	}
	semi := strings.Index(rest, ";")
	if semi < 0 {
		return "", "", false
	}
	mt := rest[:semi]
	rest = rest[semi+1:]
	rest, found = strings.CutPrefix(rest, "base64,")
	if !found {
		return "", "", false
	}
	return mt, rest, true
}
