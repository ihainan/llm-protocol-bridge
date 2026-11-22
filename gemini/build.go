package gemini

import (
	"encoding/json"
	"strings"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/internal/schema"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// BuildRequest serialises an ir.Request to Gemini generateContent JSON.
func BuildRequest(req *ir.Request, opts BuildOptions) (*Built, error) {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	env.Style = ir.StyleBracket
	if req.PreviousResponseID != "" {
		return nil, env.Err(ir.Unsupported(ir.Root.Key("previous_response_id"), "previous_response_id", "previous_response_id is only supported by the Responses format"))
	}
	dia := opts.Dialect
	nameMap := ir.BuildToolNameMap(req.Tools)
	names := nameMap.FromWire

	// Map tool call IDs to wire function names so functionResponse.name is always populated consistently.
	callNames := map[string]string{}
	for _, msg := range req.Messages {
		for _, pt := range msg.Parts {
			if pt.Type == ir.PartToolCall && pt.Call != nil && pt.Call.ID != "" && pt.Call.Name != "" {
				callNames[pt.Call.ID] = nameMap.WireName(pt.Call.Namespace, pt.Call.Name)
			}
		}
	}

	// System instruction.
	var sysContent *wireContent
	if len(req.Instructions) > 0 {
		parts, err := irPartsToWireParts(req.Instructions, ir.Root.Key("systemInstruction"), env, dia, nameMap, callNames, false)
		if err != nil {
			return nil, err
		}
		sysContent = &wireContent{Parts: parts}
	}

	// Contents.
	var contents []wireContent
	for i, msg := range req.Messages {
		fp := ir.Root.Key("contents").At(i)
		wc, err := irMessageToWireContent(msg, fp, env, dia, nameMap, callNames)
		if err != nil {
			return nil, err
		}
		contents = append(contents, wc)
	}

	// Tools.
	var toolDefs []wireToolDef
	if len(req.Tools) > 0 || len(req.Builtins) > 0 {
		var decls []wireFunctionDecl
		for i, t := range req.Tools {
			fp := ir.Root.Key("tools").At(i)
			d, err := buildFunctionDecl(t, fp, req.Source == ir.FormatGemini, dia, env, nameMap)
			if err != nil {
				return nil, err
			}
			decls = append(decls, d)
		}
		if len(decls) > 0 {
			toolDefs = append(toolDefs, wireToolDef{FunctionDeclarations: decls})
		}
		for i, bt := range req.Builtins {
			fp := ir.Root.Key("tools").At(len(req.Tools) + i)
			switch bt.Kind {
			case ir.BuiltinWebSearch:
				if bt.IsCachedWebSearch() {
					if err := env.Lose(ir.LossCachedWebSearch, fp, "cached web_search (external_web_access: false) dropped for Gemini"); err != nil {
						return nil, err
					}
					continue
				}
				wsCfg := bt.ParseWebSearchConfig()
				if len(wsCfg.AllowedDomains) > 0 || len(wsCfg.BlockedDomains) > 0 {
					return nil, env.Err(ir.Unsupported(fp, "web_search.allowed_domains", "Gemini googleSearch does not support domain restrictions"))
				}
				if len(wsCfg.UserLocation) > 0 {
					return nil, env.Err(ir.Unsupported(fp, "web_search.user_location", "Gemini googleSearch does not support user_location"))
				}
				toolDefs = append(toolDefs, wireToolDef{GoogleSearch: &struct{}{}})
			case ir.BuiltinCodeExecution:
				toolDefs = append(toolDefs, wireToolDef{CodeExecution: &struct{}{}})
			case ir.BuiltinURLContext:
				toolDefs = append(toolDefs, wireToolDef{URLContext: &struct{}{}})
			default:
				return nil, env.Err(ir.Unsupported(fp, string(bt.Kind), "builtin tool not supported by Gemini"))
			}
		}
	}

	// Tool config.
	var toolCfg *wireToolConfig
	if req.ToolChoice.Mode != "" {
		tc, err := buildToolConfig(req.ToolChoice, nameMap, env)
		if err != nil {
			return nil, err
		}
		toolCfg = tc
	}

	// Generation config.
	gc, offPatchFields, err := buildGenerationConfig(req, dia, env)
	if err != nil {
		return nil, err
	}
	// parallel_tool_calls is not supported by Gemini.
	if req.ParallelToolCalls != nil {
		if err := env.Lose(ir.LossToolParallel, ir.Root.Key("parallel_tool_calls"), "not supported by Gemini"); err != nil {
			return nil, err
		}
	}

	// Assemble wire request.
	wr := wireRequest{
		Contents:          contents,
		SystemInstruction: sysContent,
		Tools:             toolDefs,
		ToolConfig:        toolCfg,
	}
	if gc != nil {
		wr.GenerationConfig = gc
	}
	if req.Source == ir.FormatGemini && req.Passthrough != nil {
		if ss, ok := req.Passthrough["safetySettings"]; ok && !jsonx.IsNull(ss) {
			var settings []wireSafetySetting
			if json.Unmarshal(ss, &settings) == nil {
				wr.SafetySettings = settings
			}
		}
	}
	if req.User != "" {
		if err := env.Lose(ir.LossRequestUser, ir.Root.Key("user"), "not supported by Gemini"); err != nil {
			return nil, err
		}
	}
	if len(req.Metadata) > 0 {
		if err := env.Lose(ir.LossRequestMetadata, ir.Root.Key("metadata"), "not supported by Gemini"); err != nil {
			return nil, err
		}
	}
	if req.ServiceTier != "" {
		if err := env.Lose(ir.LossRequestTier, ir.Root.Key("service_tier"), "not supported by Gemini"); err != nil {
			return nil, err
		}
	}

	body, err := json.Marshal(wr)
	if err != nil {
		return nil, err
	}
	if len(offPatchFields) > 0 {
		body, err = mergeTopLevelPatch(body, offPatchFields)
		if err != nil {
			return nil, err
		}
	}

	return &Built{
		Body:   body,
		Model:  req.Model,
		Stream: req.Stream,
		Names:  names,
	}, nil
}

func mergeTopLevelPatch(base []byte, patch []ir.OffPatchField) ([]byte, error) {
	kvs, err := jsonx.Object(base)
	if err != nil {
		return nil, err
	}
	m := map[string]json.RawMessage{}
	order := make([]string, 0, len(kvs)+len(patch))
	for _, kv := range kvs {
		m[kv.Key] = kv.Value
		order = append(order, kv.Key)
	}
	for _, f := range patch {
		if existing, ok := m[f.Key]; ok {
			if jsonx.Kind(existing) == '{' && jsonx.Kind(f.Value) == '{' {
				m[f.Key] = mergeJSONObjects(existing, f.Value)
			} else {
				m[f.Key] = f.Value
			}
		} else {
			m[f.Key] = f.Value
			order = append(order, f.Key)
		}
	}
	out := jsonx.Start(nil)
	for _, k := range order {
		out.Raw(k, m[k])
	}
	return out.End(), nil
}

func mergeJSONObjects(base, overlay json.RawMessage) json.RawMessage {
	bkvs, err1 := jsonx.Object(base)
	okvs, err2 := jsonx.Object(overlay)
	if err1 != nil || err2 != nil {
		return overlay
	}
	m := map[string]json.RawMessage{}
	order := make([]string, 0, len(bkvs)+len(okvs))
	for _, kv := range bkvs {
		m[kv.Key] = kv.Value
		order = append(order, kv.Key)
	}
	for _, kv := range okvs {
		if prev, ok := m[kv.Key]; ok {
			if jsonx.Kind(prev) == '{' && jsonx.Kind(kv.Value) == '{' {
				m[kv.Key] = mergeJSONObjects(prev, kv.Value)
			} else {
				m[kv.Key] = kv.Value
			}
		} else {
			m[kv.Key] = kv.Value
			order = append(order, kv.Key)
		}
	}
	out := jsonx.Start(nil)
	for _, k := range order {
		out.Raw(k, m[k])
	}
	return out.End()
}

func irMessageToWireContent(msg ir.Message, p ir.Path, env *ir.Env, dia Dialect, nameMap *ir.ToolNameMap, callNames map[string]string) (wireContent, error) {
	var role string
	switch msg.Role {
	case ir.RoleUser, ir.RoleSystem, ir.RoleDeveloper:
		role = "user"
	case ir.RoleAssistant:
		role = "model"
	}
	parts, err := irPartsToWireParts(msg.Parts, p.Key("parts"), env, dia, nameMap, callNames, true)
	if err != nil {
		return wireContent{}, err
	}
	return wireContent{Role: role, Parts: parts}, nil
}

func irPartsToWireParts(parts []ir.Part, p ir.Path, env *ir.Env, dia Dialect, nameMap *ir.ToolNameMap, callNames map[string]string, inMessage bool) ([]wirePart, error) {
	var out []wirePart
	for i, part := range parts {
		fp := p.At(i)
		wp, err := irPartToWirePart(part, fp, env, dia, nameMap, callNames, inMessage)
		if err != nil {
			return nil, err
		}
		out = append(out, wp...)
	}
	return out, nil
}

func irPartToWirePart(part ir.Part, p ir.Path, env *ir.Env, dia Dialect, nameMap *ir.ToolNameMap, callNames map[string]string, inMessage bool) ([]wirePart, error) {
	if part.Cache != nil {
		if err := env.Lose(ir.LossCacheMarker, p.Key("cache_control"), "prompt cache markers not supported by Gemini"); err != nil {
			return nil, err
		}
	}
	switch part.Type {
	case ir.PartText:
		wp := wirePart{Text: part.Text}
		if part.Signature != nil && part.Signature.Value != "" {
			if part.Signature.Origin.Matches(ir.FormatGemini, env.ProviderID) {
				wp.ThoughtSignature = part.Signature.Value
			} else if err := env.Lose(ir.LossReasoningSignature, p, "foreign thought signature dropped for Gemini"); err != nil {
				return nil, err
			}
		}
		return []wirePart{wp}, nil

	case ir.PartReasoning:
		if part.Reasoning == nil {
			return nil, nil
		}
		if part.Reasoning.Sealed != nil {
			if err := env.Lose(ir.LossReasoningSealed, p, "sealed reasoning block not supported by Gemini"); err != nil {
				return nil, err
			}
			if part.Reasoning.Text == "" {
				return nil, nil
			}
		}
		wp := wirePart{
			Text:    part.Reasoning.Text,
			Thought: true,
		}
		sig := part.Reasoning.Signature
		if sig == nil {
			sig = part.Signature
		}
		if sig != nil && sig.Value != "" {
			if sig.Origin.Matches(ir.FormatGemini, env.ProviderID) {
				wp.ThoughtSignature = sig.Value
			} else if err := env.Lose(ir.LossReasoningSignature, p, "foreign reasoning signature dropped for Gemini"); err != nil {
				return nil, err
			}
		}
		return []wirePart{wp}, nil

	case ir.PartImage:
		if part.Media == nil {
			return nil, nil
		}
		if part.Media.Detail != "" {
			if err := env.Lose(ir.LossImageDetail, p.Key("detail"), "image detail not supported by Gemini"); err != nil {
				return nil, err
			}
		}
		if part.Media.URL != "" && dia.URLFileData {
			return []wirePart{{FileData: &wireFileData{
				MimeType: part.Media.MediaType,
				FileURI:  part.Media.URL,
			}}}, nil
		}
		if part.Media.Data != "" {
			return []wirePart{{InlineData: &wireBlob{
				MimeType: part.Media.MediaType,
				Data:     part.Media.Data,
			}}}, nil
		}
		if part.Media.URL != "" {
			return []wirePart{{FileData: &wireFileData{
				MimeType: part.Media.MediaType,
				FileURI:  part.Media.URL,
			}}}, nil
		}
		return nil, env.Err(ir.Unsupported(p, "image", "image with no data or URL"))

	case ir.PartFile, ir.PartAudio, ir.PartVideo:
		if part.Media == nil {
			return nil, nil
		}
		if part.Media.Text != "" {
			return []wirePart{{Text: part.Media.Text}}, nil
		}
		if part.Media.URL != "" {
			return []wirePart{{FileData: &wireFileData{
				MimeType: part.Media.MediaType,
				FileURI:  part.Media.URL,
			}}}, nil
		}
		if part.Media.Data != "" {
			return []wirePart{{InlineData: &wireBlob{
				MimeType: part.Media.MediaType,
				Data:     part.Media.Data,
			}}}, nil
		}
		return nil, env.Err(ir.Unsupported(p, string(part.Type), "media with no data or URL"))

	case ir.PartToolCall:
		if part.Call == nil {
			return nil, nil
		}
		call := part.Call
		args := call.Arguments
		if len(args) == 0 {
			args = json.RawMessage(`{}`)
		}
		fc := &wireFunctionCall{
			Name: nameMap.WireName(call.Namespace, call.Name),
			Args: args,
		}
		if dia.FunctionCallIDs && call.ID != "" {
			fc.ID = call.ID
		}
		wp := wirePart{FunctionCall: fc}
		if call.Signature != nil && call.Signature.Value != "" {
			if call.Signature.Origin.Matches(ir.FormatGemini, env.ProviderID) {
				wp.ThoughtSignature = call.Signature.Value
			} else if err := env.Lose(ir.LossReasoningSignature, p, "foreign tool call signature dropped for Gemini"); err != nil {
				return nil, err
			}
		}
		return []wirePart{wp}, nil

	case ir.PartToolResult:
		if part.Result == nil {
			return nil, nil
		}
		result := part.Result
		fnName := ""
		if result.CallID != "" && callNames != nil {
			fnName = callNames[result.CallID]
		}
		if fnName == "" && result.Name != "" {
			fnName = nameMap.WireName("", result.Name)
		}
		// Wrap content as {output: <text>} or {error: <text>}, preserving JSON objects directly when not an error.
		var respRaw json.RawMessage
		text := toolResultText(result.Content)
		trimmed := strings.TrimSpace(text)
		if result.IsError {
			b, _ := jsonx.Marshal(map[string]any{"error": text})
			respRaw = b
		} else if len(trimmed) > 0 && jsonx.Kind([]byte(trimmed)) == '{' && json.Valid([]byte(trimmed)) {
			respRaw = json.RawMessage(trimmed)
		} else {
			b, _ := jsonx.Marshal(map[string]any{"output": text})
			respRaw = b
		}
		fr := &wireFunctionResponse{
			Name:     fnName,
			Response: respRaw,
		}
		for i, cp := range result.Content {
			if cp.Type == ir.PartText {
				continue
			}
			sub, err := irPartToWirePart(cp, p.Key("content").At(i), env, dia, nameMap, callNames, false)
			if err != nil {
				return nil, err
			}
			fr.Parts = append(fr.Parts, sub...)
		}
		if dia.FunctionCallIDs && result.CallID != "" {
			fr.ID = result.CallID
		}
		return []wirePart{{FunctionResponse: fr}}, nil

	case ir.PartRefusal:
		return []wirePart{{Text: part.Text}}, nil

	case ir.PartBuiltinCall:
		if part.Builtin != nil && part.Builtin.Kind == ir.BuiltinCodeExecution {
			var payload struct {
				Language string `json:"language"`
				Code     string `json:"code"`
			}
			_ = json.Unmarshal(part.Builtin.Input, &payload)
			return []wirePart{{ExecutableCode: &wireExecutableCode{Language: payload.Language, Code: payload.Code}}}, nil
		}
		if part.Call != nil && len(part.Call.Arguments) > 0 {
			return []wirePart{{ExecutableCode: &wireExecutableCode{Code: string(part.Call.Arguments)}}}, nil
		}
		if err := env.Lose(ir.LossBuiltinActivity, p, "builtin call not supported by Gemini"); err != nil {
			return nil, err
		}
		return nil, nil

	case ir.PartBuiltinResult:
		if part.Builtin != nil && part.Builtin.Kind == ir.BuiltinCodeExecution {
			outcome := "OUTCOME_OK"
			if part.Builtin.Error != "" {
				outcome = part.Builtin.Error
			}
			return []wirePart{{CodeExecutionResult: &wireCodeExecutionResult{Outcome: outcome, Output: part.Builtin.Content}}}, nil
		}
		if part.Result != nil {
			text := toolResultText(part.Result.Content)
			return []wirePart{{CodeExecutionResult: &wireCodeExecutionResult{
				Outcome: "OUTCOME_OK",
				Output:  text,
			}}}, nil
		}
		if err := env.Lose(ir.LossBuiltinActivity, p, "builtin result not supported by Gemini"); err != nil {
			return nil, err
		}
		return nil, nil

	default:
		return nil, env.Err(ir.Unsupported(p, string(part.Type), "part type not supported by Gemini"))
	}
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

func buildFunctionDecl(t ir.Tool, p ir.Path, sameFormat bool, dia Dialect, env *ir.Env, nameMap *ir.ToolNameMap) (wireFunctionDecl, error) {
	if t.Name == "" {
		return wireFunctionDecl{}, env.Err(ir.Invalid(p.Key("name"), "tool name is required"))
	}
	if t.Strict != nil && *t.Strict {
		if err := env.Lose(ir.LossToolStrict, p.Key("strict"), "strict tool schemas not supported by Gemini"); err != nil {
			return wireFunctionDecl{}, err
		}
	}
	if t.Cache != nil {
		if err := env.Lose(ir.LossCacheMarker, p.Key("cache_control"), "tool cache markers not supported by Gemini"); err != nil {
			return wireFunctionDecl{}, err
		}
	}
	wireName := nameMap.WireName(t.Namespace, t.Name)
	desc := nameMap.Description(t.Namespace, t.Name, t.Description)
	params := t.Schema
	if t.Kind == ir.ToolCustom {
		params = ir.CustomToolWrapperSchema
		if t.Grammar != nil && t.Grammar.Definition != "" {
			note := "Input grammar (" + t.Grammar.Syntax + "): " + t.Grammar.Definition
			if desc == "" {
				desc = note
			} else {
				desc += "\n" + note
			}
		}
	}
	d := wireFunctionDecl{
		Name:        wireName,
		Description: desc,
	}
	if len(params) > 0 {
		if (sameFormat && t.SchemaJSON) || dia.SchemaField == "parametersJsonSchema" {
			d.ParametersJsonSchema = params
		} else {
			converted, dropped := schema.ToOpenAPIWithLoss(params)
			if len(dropped) > 0 {
				if err := env.Lose(ir.LossToolSchema, p.Key("parameters"), "OpenAPI schema downgrade dropped constraints: "+strings.Join(dropped, ", ")); err != nil {
					return wireFunctionDecl{}, err
				}
			}
			d.Parameters = converted
		}
	}
	return d, nil
}

func buildToolConfig(tc ir.ToolChoice, nameMap *ir.ToolNameMap, env *ir.Env) (*wireToolConfig, error) {
	cfg := &wireToolConfig{FunctionCallingConfig: &wireFunctionCallingConfig{}}
	switch tc.Mode {
	case ir.ToolChoiceAuto, "":
		cfg.FunctionCallingConfig.Mode = "AUTO"
	case ir.ToolChoiceNone:
		cfg.FunctionCallingConfig.Mode = "NONE"
	case ir.ToolChoiceRequired:
		cfg.FunctionCallingConfig.Mode = "ANY"
	case ir.ToolChoiceAllowed:
		if tc.AllowedMode == ir.ToolChoiceAuto {
			cfg.FunctionCallingConfig.Mode = "AUTO"
		} else {
			cfg.FunctionCallingConfig.Mode = "ANY"
		}
		for _, ref := range tc.Allowed {
			cfg.FunctionCallingConfig.AllowedFunctionNames = append(
				cfg.FunctionCallingConfig.AllowedFunctionNames, nameMap.WireName(ref.Namespace, ref.Name))
		}
	case ir.ToolChoiceTool:
		cfg.FunctionCallingConfig.Mode = "ANY"
		if tc.Name != "" {
			cfg.FunctionCallingConfig.AllowedFunctionNames = []string{nameMap.WireName(tc.Namespace, tc.Name)}
		}
	default:
		return nil, env.Err(ir.Unsupported(ir.Root.Key("tool_choice"), string(tc.Mode), "tool choice mode"))
	}
	return cfg, nil
}

func buildGenerationConfig(req *ir.Request, dia Dialect, env *ir.Env) (*wireGenerationConfig, []ir.OffPatchField, error) {
	gc := &wireGenerationConfig{}
	var offPatchFields []ir.OffPatchField
	any := false

	if req.Temperature != nil {
		gc.Temperature = req.Temperature
		any = true
	}
	if req.TopP != nil {
		gc.TopP = req.TopP
		any = true
	}
	if req.TopK != nil {
		gc.TopK = req.TopK
		any = true
	}
	if req.MaxOutputTokens != nil {
		gc.MaxOutputTokens = req.MaxOutputTokens
		any = true
	}
	if req.Candidates != nil {
		gc.CandidateCount = req.Candidates
		any = true
	}
	const geminiMaxStop = 5
	if len(req.Stop) > 0 {
		stops := req.Stop
		if len(stops) > geminiMaxStop {
			if err := env.Lose(ir.LossStopLimit, ir.Root.Key("stop"), "Gemini accepts at most 5 stop sequences; extras truncated"); err != nil {
				return nil, nil, err
			}
			stops = stops[:geminiMaxStop]
		}
		gc.StopSequences = stops
		any = true
	}
	if req.Seed != nil {
		gc.Seed = req.Seed
		any = true
	}
	// Gemini does not support presence/frequency penalty.
	if req.PresencePenalty != nil && *req.PresencePenalty != 0 {
		if err := env.Lose(ir.LossSamplingPenalty, ir.Root.Key("presence_penalty"), "not supported by Gemini"); err != nil {
			return nil, nil, err
		}
	}
	if req.FrequencyPenalty != nil && *req.FrequencyPenalty != 0 {
		if err := env.Lose(ir.LossSamplingPenalty, ir.Root.Key("frequency_penalty"), "not supported by Gemini"); err != nil {
			return nil, nil, err
		}
	}
	// Gemini does not support logprobs.
	if req.Logprobs != nil && *req.Logprobs {
		if err := env.Lose(ir.LossSamplingLogprobs, ir.Root.Key("logprobs"), "not supported by Gemini"); err != nil {
			return nil, nil, err
		}
	}
	if req.Output != nil {
		switch req.Output.Type {
		case ir.OutputJSONObject:
			gc.ResponseMimeType = "application/json"
			any = true
		case ir.OutputJSONSchema:
			gc.ResponseMimeType = "application/json"
			if len(req.Output.Schema) > 0 {
				schemaField := dia.SchemaField
				if schemaField == "" {
					schemaField = "responseSchema"
				}
				// Always set ResponseSchema regardless of field name variant;
				// the wire struct uses responseSchema.
				gc.ResponseSchema = schema.ToOpenAPI(req.Output.Schema)
			}
			any = true
		default:
			gc.ResponseMimeType = "text/plain"
			any = true
		}
	}
	profile := env.ModelProfileOr(nil)
	if req.Reasoning != nil || (profile != nil && profile.DefaultEffort != ir.EffortUnset) {
		res := profile.Resolve(req.Reasoning)
		if res.Off {
			if profile != nil {
				switch profile.Off.Mode {
				case ir.OffUnsupported:
					return nil, nil, env.Err(ir.Unsupported(ir.Root.Key("reasoning"), "off", "model does not allow disabling reasoning"))
				case ir.OffPatch:
					fields, cfgErr := ir.ParseOffPatch(profile.Off.Patch)
					if cfgErr != nil {
						return nil, nil, env.Err(cfgErr)
					}
					offPatchFields = fields
				case ir.OffLowestHidden:
					low := profile.Lowest()
					tc := &wireThinkingConfig{}
					if profile.Style == ir.StyleLevel {
						tc.ThinkingLevel = strings.ToUpper(string(low))
					} else {
						b := profile.EffortBudget(low)
						tc.ThinkingBudget = &b
					}
					gc.ThinkingConfig = tc
					any = true
				default:
					zero := 0
					gc.ThinkingConfig = &wireThinkingConfig{ThinkingBudget: &zero}
					any = true
				}
			} else {
				zero := 0
				gc.ThinkingConfig = &wireThinkingConfig{ThinkingBudget: &zero}
				any = true
			}
		} else if (res.Asked || res.Effort != ir.EffortUnset || res.Budget != nil) && (profile == nil || profile.Style != ir.StyleNone) {
			tc := &wireThinkingConfig{}
			if profile != nil && profile.Style == ir.StyleLevel {
				if res.Effort != ir.EffortUnset {
					tc.ThinkingLevel = strings.ToUpper(string(res.Effort))
				}
			} else {
				if res.Dynamic || (res.Budget != nil && *res.Budget < 0) {
					dyn := -1
					tc.ThinkingBudget = &dyn
				} else if res.Budget != nil {
					b := *res.Budget
					if profile != nil && profile.MinBudget > 0 && b < profile.MinBudget {
						b = profile.MinBudget
					}
					tc.ThinkingBudget = &b
				} else if res.Effort != ir.EffortUnset {
					b := profile.EffortBudget(res.Effort)
					if profile != nil && profile.MinBudget > 0 && b < profile.MinBudget {
						b = profile.MinBudget
					}
					tc.ThinkingBudget = &b
				}
			}
			if req.Reasoning != nil && (req.Reasoning.Display == "shown" || req.Reasoning.Summary != "") {
				tc.IncludeThoughts = true
			}
			gc.ThinkingConfig = tc
			any = true
		}
	}

	if !any {
		return nil, offPatchFields, nil
	}
	return gc, offPatchFields, nil
}
