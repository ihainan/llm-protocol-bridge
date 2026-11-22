package gemini

import (
	"encoding/json"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/internal/schema"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// ParseRequest decodes a Gemini generateContent request body into an ir.Request.
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

	req := &ir.Request{Source: ir.FormatGemini, Model: opts.Model, Stream: opts.Stream}
	p := ir.Root

	for _, kv := range kvs {
		k, v := kv.Key, kv.Value
		fp := p.Key(k)
		switch k {
		case "contents":
			msgs, err := parseContents(v, fp, env)
			if err != nil {
				return nil, err
			}
			req.Messages = msgs

		case "systemInstruction":
			parts, err := parseSystemInstruction(v, fp, env)
			if err != nil {
				return nil, err
			}
			req.Instructions = parts

		case "tools":
			tools, builtins, err := parseTools(v, fp, env)
			if err != nil {
				return nil, err
			}
			req.Tools = append(req.Tools, tools...)
			req.Builtins = append(req.Builtins, builtins...)

		case "toolConfig":
			tc, err := parseToolConfig(v, fp, env)
			if err != nil {
				return nil, err
			}
			req.ToolChoice = tc

		case "generationConfig":
			if err := parseGenerationConfig(v, fp, req, env); err != nil {
				return nil, err
			}

		case "safetySettings":
			// Accepted but not stored in IR (pass through).
			if req.Passthrough == nil {
				req.Passthrough = map[string]json.RawMessage{}
			}
			req.Passthrough[k] = append(json.RawMessage(nil), v...)

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

	return req, nil
}

func parseContents(raw json.RawMessage, p ir.Path, env *ir.Env) ([]ir.Message, error) {
	elems, ok := jsonx.Array(raw)
	if !ok {
		return nil, env.Err(ir.Invalid(p, "must be an array"))
	}
	msgs := make([]ir.Message, 0, len(elems))
	for i, elem := range elems {
		fp := p.At(i)
		msg, err := parseContent(elem, fp, env)
		if err != nil {
			return nil, err
		}
		msgs = append(msgs, msg)
	}
	return msgs, nil
}

func parseContent(raw json.RawMessage, p ir.Path, env *ir.Env) (ir.Message, error) {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return ir.Message{}, env.Err(ir.Invalid(p, "must be an object"))
	}
	var role string
	var partsRaw json.RawMessage
	for _, kv := range kvs {
		switch kv.Key {
		case "role":
			role, _ = jsonx.String(kv.Value)
		case "parts":
			partsRaw = kv.Value
		}
	}

	var irRole ir.Role
	switch role {
	case "user", "":
		irRole = ir.RoleUser
	case "model":
		irRole = ir.RoleAssistant
	default:
		return ir.Message{}, env.Err(ir.Invalid(p.Key("role"), "unknown role: "+role))
	}

	parts, err := parseContentParts(partsRaw, p.Key("parts"), env)
	if err != nil {
		return ir.Message{}, err
	}
	return ir.Message{Role: irRole, Parts: parts}, nil
}

func parseSystemInstruction(raw json.RawMessage, p ir.Path, env *ir.Env) ([]ir.Part, error) {
	if jsonx.IsNull(raw) {
		return nil, nil
	}
	switch jsonx.Kind(raw) {
	case '{':
		kvs, err := jsonx.Object(raw)
		if err != nil {
			return nil, env.Err(ir.Invalid(p, "must be an object"))
		}
		for _, kv := range kvs {
			if kv.Key == "parts" {
				return parseContentParts(kv.Value, p.Key("parts"), env)
			}
		}
		// Tolerate a single Part object directly (e.g. {"text": "..."}).
		part, err := parsePart(raw, p, env)
		if err != nil {
			return nil, err
		}
		return []ir.Part{part}, nil
	case '[':
		return parseContentParts(raw, p, env)
	}
	return nil, env.Err(ir.Invalid(p, "must be a content object"))
}

func parseContentParts(raw json.RawMessage, p ir.Path, env *ir.Env) ([]ir.Part, error) {
	switch jsonx.Kind(raw) {
	case '[':
		elems, ok := jsonx.Array(raw)
		if !ok {
			return nil, env.Err(ir.Invalid(p, "must be an array"))
		}
		parts := make([]ir.Part, 0, len(elems))
		for i, elem := range elems {
			part, err := parsePart(elem, p.At(i), env)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		}
		return parts, nil
	case '{':
		// systemInstruction may be a single content object.
		part, err := parsePart(raw, p, env)
		if err != nil {
			return nil, err
		}
		return []ir.Part{part}, nil
	}
	return nil, nil
}

func parsePart(raw json.RawMessage, p ir.Path, env *ir.Env) (ir.Part, error) {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return ir.Part{}, env.Err(ir.Invalid(p, "must be an object"))
	}
	fields := map[string]json.RawMessage{}
	for _, kv := range kvs {
		fields[kv.Key] = kv.Value
	}

	switch {
	case fields["text"] != nil:
		text, _ := jsonx.String(fields["text"])
		var sig *ir.Signature
		if s, ok := jsonx.String(fields["thoughtSignature"]); ok && s != "" {
			sig = &ir.Signature{Value: s, Origin: ir.Origin{Format: ir.FormatGemini, Provider: env.ProviderID}}
		}
		if b, ok := jsonx.Bool(fields["thought"]); ok && b {
			// A thought part: becomes reasoning, preserving thoughtSignature.
			return ir.Part{
				Type:      ir.PartReasoning,
				Reasoning: &ir.Reasoning{Text: text, Signature: sig},
			}, nil
		}
		part := ir.TextPart(text)
		part.Signature = sig
		return part, nil

	case fields["inlineData"] != nil:
		kvs2, err := jsonx.Object(fields["inlineData"])
		if err != nil {
			return ir.Part{}, env.Err(ir.Invalid(p.Key("inlineData"), "must be an object"))
		}
		media := &ir.Media{}
		for _, kv := range kvs2 {
			switch kv.Key {
			case "mimeType":
				media.MediaType, _ = jsonx.String(kv.Value)
			case "data":
				media.Data, _ = jsonx.String(kv.Value)
			}
		}
		return ir.Part{Type: ir.PartImage, Media: media}, nil

	case fields["fileData"] != nil:
		kvs2, err := jsonx.Object(fields["fileData"])
		if err != nil {
			return ir.Part{}, env.Err(ir.Invalid(p.Key("fileData"), "must be an object"))
		}
		media := &ir.Media{}
		for _, kv := range kvs2 {
			switch kv.Key {
			case "mimeType":
				media.MediaType, _ = jsonx.String(kv.Value)
			case "fileUri":
				media.URL, _ = jsonx.String(kv.Value)
			}
		}
		return ir.Part{Type: ir.PartFile, Media: media}, nil

	case fields["functionCall"] != nil:
		kvs2, err := jsonx.Object(fields["functionCall"])
		if err != nil {
			return ir.Part{}, env.Err(ir.Invalid(p.Key("functionCall"), "must be an object"))
		}
		call := &ir.ToolCall{Kind: ir.ToolFunction}
		for _, kv := range kvs2 {
			switch kv.Key {
			case "id":
				call.ID, _ = jsonx.String(kv.Value)
			case "name":
				call.Name, _ = jsonx.String(kv.Value)
			case "args":
				call.Arguments = append(json.RawMessage(nil), kv.Value...)
			}
		}
		sig, _ := jsonx.String(fields["thoughtSignature"])
		if sig != "" {
			call.Signature = &ir.Signature{Value: sig, Origin: ir.Origin{Format: ir.FormatGemini, Provider: env.ProviderID}}
		}
		return ir.Part{Type: ir.PartToolCall, Call: call}, nil

	case fields["functionResponse"] != nil:
		kvs2, err := jsonx.Object(fields["functionResponse"])
		if err != nil {
			return ir.Part{}, env.Err(ir.Invalid(p.Key("functionResponse"), "must be an object"))
		}
		result := &ir.ToolResult{}
		var resp json.RawMessage
		for _, kv := range kvs2 {
			switch kv.Key {
			case "id":
				result.CallID, _ = jsonx.String(kv.Value)
			case "name":
				result.Name, _ = jsonx.String(kv.Value)
			case "response":
				resp = kv.Value
			}
		}
		// Gemini wraps the result in {output: ...} or {error: ...} or a structured JSON object.
		if !jsonx.IsNull(resp) {
			if kvs3, err := jsonx.Object(resp); err == nil && len(kvs3) == 1 {
				if kvs3[0].Key == "output" {
					if s, ok := jsonx.String(kvs3[0].Value); ok {
						result.Content = []ir.Part{ir.TextPart(s)}
					} else {
						result.Content = []ir.Part{ir.TextPart(string(kvs3[0].Value))}
					}
				} else if kvs3[0].Key == "error" {
					result.IsError = true
					if s, ok := jsonx.String(kvs3[0].Value); ok {
						result.Content = []ir.Part{ir.TextPart(s)}
					} else {
						result.Content = []ir.Part{ir.TextPart(string(kvs3[0].Value))}
					}
				} else {
					result.Content = []ir.Part{ir.TextPart(string(resp))}
				}
			} else {
				result.Content = []ir.Part{ir.TextPart(string(resp))}
			}
		}
		return ir.Part{Type: ir.PartToolResult, Result: result}, nil
	}

	return ir.Part{}, env.Err(ir.Invalid(p, "unrecognised part type"))
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
	for i, elem := range elems {
		fp := p.At(i)
		kvs, err := jsonx.Object(elem)
		if err != nil {
			return nil, nil, env.Err(ir.Invalid(fp, "must be an object"))
		}
		for _, kv := range kvs {
			switch kv.Key {
			case "functionDeclarations":
				decls, ok := jsonx.Array(kv.Value)
				if !ok {
					return nil, nil, env.Err(ir.Invalid(fp.Key(kv.Key), "must be an array"))
				}
				for j, decl := range decls {
					t, err := parseFunctionDecl(decl, fp.Key(kv.Key).At(j), env)
					if err != nil {
						return nil, nil, err
					}
					tools = append(tools, t)
				}
			case "googleSearch":
				builtins = append(builtins, ir.BuiltinTool{
					Kind:   ir.BuiltinWebSearch,
					Origin: ir.FormatGemini,
					Raw:    append(json.RawMessage(nil), elem...),
				})
			case "codeExecution":
				builtins = append(builtins, ir.BuiltinTool{
					Kind:   ir.BuiltinCodeExecution,
					Origin: ir.FormatGemini,
					Raw:    append(json.RawMessage(nil), elem...),
				})
			case "urlContext":
				builtins = append(builtins, ir.BuiltinTool{
					Kind:   ir.BuiltinURLContext,
					Origin: ir.FormatGemini,
					Raw:    append(json.RawMessage(nil), elem...),
				})
			}
		}
	}
	return tools, builtins, nil
}

func parseFunctionDecl(raw json.RawMessage, p ir.Path, env *ir.Env) (ir.Tool, error) {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return ir.Tool{}, env.Err(ir.Invalid(p, "must be an object"))
	}
	var t ir.Tool
	t.Kind = ir.ToolFunction
	for _, kv := range kvs {
		switch kv.Key {
		case "name":
			t.Name, _ = jsonx.String(kv.Value)
		case "description":
			t.Description, _ = jsonx.String(kv.Value)
		case "parameters":
			s := schema.FromOpenAPI(kv.Value)
			t.Schema = s
		}
	}
	if t.Name == "" {
		return ir.Tool{}, env.Err(ir.Invalid(p.Key("name"), "function name is required"))
	}
	return t, nil
}

func parseToolConfig(raw json.RawMessage, p ir.Path, env *ir.Env) (ir.ToolChoice, error) {
	if jsonx.IsNull(raw) {
		return ir.ToolChoice{}, nil
	}
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return ir.ToolChoice{}, env.Err(ir.Invalid(p, "must be an object"))
	}
	for _, kv := range kvs {
		if kv.Key == "functionCallingConfig" {
			return parseFunctionCallingConfig(kv.Value, p.Key(kv.Key), env)
		}
	}
	return ir.ToolChoice{}, nil
}

func parseFunctionCallingConfig(raw json.RawMessage, p ir.Path, env *ir.Env) (ir.ToolChoice, error) {
	if jsonx.IsNull(raw) {
		return ir.ToolChoice{}, nil
	}
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return ir.ToolChoice{}, env.Err(ir.Invalid(p, "must be an object"))
	}
	var mode string
	var allowed []string
	for _, kv := range kvs {
		switch kv.Key {
		case "mode":
			mode, _ = jsonx.String(kv.Value)
		case "allowedFunctionNames":
			elems, ok := jsonx.Array(kv.Value)
			if !ok {
				break
			}
			for _, e := range elems {
				s, _ := jsonx.String(e)
				if s != "" {
					allowed = append(allowed, s)
				}
			}
		}
	}
	switch mode {
	case "AUTO", "":
		return ir.ToolChoice{Mode: ir.ToolChoiceAuto}, nil
	case "NONE":
		return ir.ToolChoice{Mode: ir.ToolChoiceNone}, nil
	case "ANY":
		if len(allowed) > 0 {
			refs := make([]ir.ToolRef, len(allowed))
			for i, n := range allowed {
				refs[i] = ir.ToolRef{Name: n}
			}
			return ir.ToolChoice{Mode: ir.ToolChoiceAllowed, Allowed: refs}, nil
		}
		return ir.ToolChoice{Mode: ir.ToolChoiceRequired}, nil
	}
	return ir.ToolChoice{}, env.Err(ir.Invalid(p.Key("mode"), "unknown mode: "+mode))
}

func parseGenerationConfig(raw json.RawMessage, p ir.Path, req *ir.Request, env *ir.Env) error {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return env.Err(ir.Invalid(p, "must be an object"))
	}
	for _, kv := range kvs {
		k, v := kv.Key, kv.Value
		fp := p.Key(k)
		switch k {
		case "temperature":
			f, ok := jsonx.Float(v)
			if !ok {
				return env.Err(ir.Invalid(fp, "must be a number"))
			}
			req.Temperature = &f
		case "topP":
			f, ok := jsonx.Float(v)
			if !ok {
				return env.Err(ir.Invalid(fp, "must be a number"))
			}
			req.TopP = &f
		case "topK":
			n, ok := jsonx.Int(v)
			if !ok {
				return env.Err(ir.Invalid(fp, "must be an integer"))
			}
			ni := int(n)
			req.TopK = &ni
		case "maxOutputTokens":
			n, ok := jsonx.Int(v)
			if !ok {
				return env.Err(ir.Invalid(fp, "must be an integer"))
			}
			ni := int(n)
			req.MaxOutputTokens = &ni
		case "candidateCount":
			n, ok := jsonx.Int(v)
			if !ok {
				return env.Err(ir.Invalid(fp, "must be an integer"))
			}
			ni := int(n)
			req.Candidates = &ni
		case "stopSequences":
			stops, err := parseStopSeqs(v, fp, env)
			if err != nil {
				return err
			}
			req.Stop = stops
		case "seed":
			n, ok := jsonx.Int(v)
			if !ok {
				return env.Err(ir.Invalid(fp, "must be an integer"))
			}
			req.Seed = &n
		case "responseMimeType":
			s, ok := jsonx.String(v)
			if !ok {
				return env.Err(ir.Invalid(fp, "must be a string"))
			}
			if req.Output == nil {
				req.Output = &ir.OutputFormat{}
			}
			switch s {
			case "text/plain":
				if req.Output.Type != ir.OutputJSONSchema {
					req.Output.Type = ir.OutputText
				}
			case "application/json":
				// Do not downgrade OutputJSONSchema if responseSchema was already parsed.
				if req.Output.Type != ir.OutputJSONSchema {
					req.Output.Type = ir.OutputJSONObject
				}
			}
		case "responseSchema":
			if req.Output == nil {
				req.Output = &ir.OutputFormat{}
			}
			req.Output.Type = ir.OutputJSONSchema
			req.Output.Schema = schema.FromOpenAPI(v)
		case "thinkingConfig":
			tc, err := parseThinkingConfig(v, fp, env)
			if err != nil {
				return err
			}
			req.Reasoning = tc
		default:
			if err := env.UnknownParam(fp); err != nil {
				return err
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
	rc := &ir.ReasoningConfig{Mode: ir.ReasoningOn}
	for _, kv := range kvs {
		switch kv.Key {
		case "thinkingBudget":
			n, ok := jsonx.Int(kv.Value)
			if ok {
				ni := int(n)
				rc.Budget = &ni
				if ni == 0 {
					rc.Mode = ir.ReasoningOff
				}
			}
		case "thinkingLevel":
			s, ok := jsonx.String(kv.Value)
			if ok && s != "" {
				if eff, ok2 := ir.ParseEffort(s); ok2 {
					rc.Effort = eff
					if eff == ir.EffortNone {
						rc.Mode = ir.ReasoningOff
					}
				}
			}
		case "includeThoughts":
			b, ok := jsonx.Bool(kv.Value)
			if ok && b {
				rc.Display = "shown"
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
	for i, e := range elems {
		s, ok := jsonx.String(e)
		if !ok {
			return nil, env.Err(ir.Invalid(p.At(i), "must be a string"))
		}
		stops = append(stops, s)
	}
	return stops, nil
}
