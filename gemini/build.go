package gemini

import (
	"encoding/json"

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
	dia := opts.Dialect
	names := map[string]ir.ToolRef{}

	// System instruction.
	var sysContent *wireContent
	if len(req.Instructions) > 0 {
		parts, err := irPartsToWireParts(req.Instructions, ir.Root.Key("systemInstruction"), env, dia, names, false)
		if err != nil {
			return nil, err
		}
		sysContent = &wireContent{Parts: parts}
	}

	// Contents.
	var contents []wireContent
	for i, msg := range req.Messages {
		fp := ir.Root.Key("contents").At(i)
		wc, err := irMessageToWireContent(msg, fp, env, dia, names)
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
			d, err := buildFunctionDecl(t, fp, env)
			if err != nil {
				return nil, err
			}
			names[t.Name] = ir.ToolRef{Name: t.Name, Namespace: t.Namespace, Kind: t.Kind}
			decls = append(decls, d)
		}
		if len(decls) > 0 {
			toolDefs = append(toolDefs, wireToolDef{FunctionDeclarations: decls})
		}
		for _, bt := range req.Builtins {
			switch bt.Kind {
			case ir.BuiltinWebSearch:
				toolDefs = append(toolDefs, wireToolDef{GoogleSearch: &struct{}{}})
			case ir.BuiltinCodeExecution:
				toolDefs = append(toolDefs, wireToolDef{CodeExecution: &struct{}{}})
			case ir.BuiltinURLContext:
				toolDefs = append(toolDefs, wireToolDef{URLContext: &struct{}{}})
			}
		}
	}

	// Tool config.
	var toolCfg *wireToolConfig
	if req.ToolChoice.Mode != "" {
		tc, err := buildToolConfig(req.ToolChoice, env)
		if err != nil {
			return nil, err
		}
		toolCfg = tc
	}

	// Generation config.
	gc, err := buildGenerationConfig(req, dia, env)
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

	body, err := json.Marshal(wr)
	if err != nil {
		return nil, err
	}

	return &Built{
		Body:   body,
		Model:  req.Model,
		Stream: req.Stream,
		Names:  names,
	}, nil
}

func irMessageToWireContent(msg ir.Message, p ir.Path, env *ir.Env, dia Dialect, names map[string]ir.ToolRef) (wireContent, error) {
	var role string
	switch msg.Role {
	case ir.RoleUser, ir.RoleSystem, ir.RoleDeveloper:
		role = "user"
	case ir.RoleAssistant:
		role = "model"
	}
	parts, err := irPartsToWireParts(msg.Parts, p.Key("parts"), env, dia, names, true)
	if err != nil {
		return wireContent{}, err
	}
	return wireContent{Role: role, Parts: parts}, nil
}

func irPartsToWireParts(parts []ir.Part, p ir.Path, env *ir.Env, dia Dialect, names map[string]ir.ToolRef, inMessage bool) ([]wirePart, error) {
	var out []wirePart
	for i, part := range parts {
		fp := p.At(i)
		wp, err := irPartToWirePart(part, fp, env, dia, names, inMessage)
		if err != nil {
			return nil, err
		}
		out = append(out, wp...)
	}
	return out, nil
}

func irPartToWirePart(part ir.Part, p ir.Path, env *ir.Env, dia Dialect, names map[string]ir.ToolRef, inMessage bool) ([]wirePart, error) {
	switch part.Type {
	case ir.PartText:
		wp := wirePart{Text: part.Text}
		if part.Signature != nil && part.Signature.Value != "" {
			wp.ThoughtSignature = part.Signature.Value
		}
		return []wirePart{wp}, nil

	case ir.PartReasoning:
		if part.Reasoning == nil {
			return nil, nil
		}
		wp := wirePart{
			Text:    part.Reasoning.Text,
			Thought: true,
		}
		if part.Signature != nil {
			wp.ThoughtSignature = part.Signature.Value
		}
		return []wirePart{wp}, nil

	case ir.PartImage:
		if part.Media == nil {
			return nil, nil
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

	case ir.PartFile:
		if part.Media == nil {
			return nil, nil
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
		return nil, env.Err(ir.Unsupported(p, "file", "file with no data or URL"))

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
			Name: call.Name,
			Args: args,
		}
		if dia.FunctionCallIDs && call.ID != "" {
			fc.ID = call.ID
		}
		wp := wirePart{FunctionCall: fc}
		if call.Signature != nil {
			wp.ThoughtSignature = call.Signature.Value
		}
		return []wirePart{wp}, nil

	case ir.PartToolResult:
		if part.Result == nil {
			return nil, nil
		}
		result := part.Result
		// Wrap content as {output: <text>} or {error: <text>}.
		var respRaw json.RawMessage
		text := toolResultText(result.Content)
		if result.IsError {
			b, _ := jsonx.Marshal(map[string]any{"error": text})
			respRaw = b
		} else {
			b, _ := jsonx.Marshal(map[string]any{"output": text})
			respRaw = b
		}
		fr := &wireFunctionResponse{
			Name:     result.Name,
			Response: respRaw,
		}
		if dia.FunctionCallIDs && result.CallID != "" {
			fr.ID = result.CallID
		}
		return []wirePart{{FunctionResponse: fr}}, nil

	case ir.PartRefusal:
		return []wirePart{{Text: part.Text}}, nil

	case ir.PartBuiltinCall:
		// Pass through raw if available; otherwise skip.
		if part.Call != nil && len(part.Call.Arguments) > 0 {
			// Represent as executable code for code execution results.
			return []wirePart{{ExecutableCode: &wireExecutableCode{Code: string(part.Call.Arguments)}}}, nil
		}
		return nil, nil

	case ir.PartBuiltinResult:
		if part.Result != nil {
			text := toolResultText(part.Result.Content)
			return []wirePart{{CodeExecutionResult: &wireCodeExecutionResult{
				Outcome: "OUTCOME_OK",
				Output:  text,
			}}}, nil
		}
		return nil, nil

	default:
		return nil, env.Err(ir.Unsupported(p, string(part.Type), "part type not supported by Gemini"))
	}
}

func toolResultText(parts []ir.Part) string {
	for _, p := range parts {
		if p.Type == ir.PartText {
			return p.Text
		}
	}
	return ""
}

func buildFunctionDecl(t ir.Tool, p ir.Path, env *ir.Env) (wireFunctionDecl, error) {
	if t.Name == "" {
		return wireFunctionDecl{}, env.Err(ir.Invalid(p.Key("name"), "tool name is required"))
	}
	d := wireFunctionDecl{
		Name:        t.Name,
		Description: t.Description,
	}
	if len(t.Schema) > 0 {
		d.Parameters = schema.ToOpenAPI(t.Schema)
	}
	return d, nil
}

func buildToolConfig(tc ir.ToolChoice, env *ir.Env) (*wireToolConfig, error) {
	cfg := &wireToolConfig{FunctionCallingConfig: &wireFunctionCallingConfig{}}
	switch tc.Mode {
	case ir.ToolChoiceAuto, "":
		cfg.FunctionCallingConfig.Mode = "AUTO"
	case ir.ToolChoiceNone:
		cfg.FunctionCallingConfig.Mode = "NONE"
	case ir.ToolChoiceRequired:
		cfg.FunctionCallingConfig.Mode = "ANY"
	case ir.ToolChoiceAllowed:
		cfg.FunctionCallingConfig.Mode = "ANY"
		for _, ref := range tc.Allowed {
			cfg.FunctionCallingConfig.AllowedFunctionNames = append(
				cfg.FunctionCallingConfig.AllowedFunctionNames, ref.Name)
		}
	case ir.ToolChoiceTool:
		cfg.FunctionCallingConfig.Mode = "ANY"
		if tc.Name != "" {
			cfg.FunctionCallingConfig.AllowedFunctionNames = []string{tc.Name}
		}
	default:
		return nil, env.Err(ir.Unsupported(ir.Root.Key("tool_choice"), string(tc.Mode), "tool choice mode"))
	}
	return cfg, nil
}

func buildGenerationConfig(req *ir.Request, dia Dialect, env *ir.Env) (*wireGenerationConfig, error) {
	gc := &wireGenerationConfig{}
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
	if len(req.Stop) > 0 {
		gc.StopSequences = req.Stop
		any = true
	}
	if req.Seed != nil {
		gc.Seed = req.Seed
		any = true
	}
	// Gemini does not support presence/frequency penalty.
	if req.PresencePenalty != nil && *req.PresencePenalty != 0 {
		if err := env.Lose(ir.LossSamplingPenalty, ir.Root.Key("presence_penalty"), "not supported by Gemini"); err != nil {
			return nil, err
		}
	}
	if req.FrequencyPenalty != nil && *req.FrequencyPenalty != 0 {
		if err := env.Lose(ir.LossSamplingPenalty, ir.Root.Key("frequency_penalty"), "not supported by Gemini"); err != nil {
			return nil, err
		}
	}
	// Gemini does not support logprobs.
	if req.Logprobs != nil && *req.Logprobs {
		if err := env.Lose(ir.LossSamplingLogprobs, ir.Root.Key("logprobs"), "not supported by Gemini"); err != nil {
			return nil, err
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
	if req.Reasoning != nil {
		tc := &wireThinkingConfig{}
		if req.Reasoning.Budget != nil {
			tc.ThinkingBudget = *req.Reasoning.Budget
		}
		if req.Reasoning.Display == "shown" {
			tc.IncludeThoughts = true
		}
		gc.ThinkingConfig = tc
		any = true
	}

	if !any {
		return nil, nil
	}
	return gc, nil
}
