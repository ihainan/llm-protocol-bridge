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

	// previous_response_id (read before input so stateful continuations allow outputs
	// referencing tool calls from the previous response).
	if v, ok := fields["previous_response_id"]; ok && !jsonx.IsNull(v) {
		req.PreviousResponseID, _ = jsonx.String(v)
	}

	// input: string or array (may include additional_tools items declaring extra tools).
	var extraTools []additionalToolBlock
	if v, ok := fields["input"]; ok && !jsonx.IsNull(v) {
		var trailing []ir.InputExtra
		msgs, et, err := parseInput(v, p.Key("input"), env, req.PreviousResponseID != "", &trailing)
		if err != nil {
			return nil, err
		}
		extraTools = et
		req.TrailingExtras = trailing
		// Extract leading system/developer messages into Instructions.
		// Within a single leading message, join multiple text parts with "\n"
		// so Instructions holds one part per instructions/message entry (joined
		// with "\n\n" when building Chat Completions).
		i := 0
		// Lifting stops at an input extra (configuration_update, compaction):
		// a system message after one stays in place, keeping the input order.
		for i < len(msgs) && (msgs[i].Role == ir.RoleSystem || msgs[i].Role == ir.RoleDeveloper) && len(msgs[i].Extras) == 0 {
			txt := ir.TextOf(msgs[i].Parts, "\n")
			if txt != "" {
				req.Instructions = append(req.Instructions, ir.TextPart(txt))
			}
			i++
		}
		req.Messages = msgs[i:]
	}

	// tools: top-level tools merged with any input[].additional_tools blocks.
	// Precedence: duplicate (Namespace, Name) function/custom declarations within a
	// single tools array are rejected (matching v0.1.0); duplicate/aliased builtins
	// within a tools array are preserved verbatim as in v0.1.0. Across top-level
	// tools and subsequent additional_tools items in input[] order, a later block's
	// declaration overrides an earlier block's entry with the same (Namespace, Name)
	// or builtin kind.
	if v, ok := fields["tools"]; ok && !jsonx.IsNull(v) {
		tools, builtins, err := parseTools(v, p.Key("tools"), env)
		if err != nil {
			return nil, err
		}
		req.Tools = tools
		req.Builtins = builtins
	}
	if len(extraTools) > 0 {
		toolIndex := make(map[string]int, len(req.Tools))
		for i, t := range req.Tools {
			toolIndex[t.Namespace+"\x00"+t.Name] = i
		}
		for _, blk := range extraTools {
			for _, t := range blk.tools {
				key := t.Namespace + "\x00" + t.Name
				if idx, exists := toolIndex[key]; exists {
					req.Tools[idx] = t
				} else {
					toolIndex[key] = len(req.Tools)
					req.Tools = append(req.Tools, t)
				}
			}
			if len(blk.builtins) > 0 {
				overridden := make(map[ir.BuiltinKind]bool, len(blk.builtins))
				for _, bt := range blk.builtins {
					overridden[bt.Kind] = true
				}
				kept := req.Builtins[:0]
				for _, bt := range req.Builtins {
					if !overridden[bt.Kind] {
						kept = append(kept, bt)
					}
				}
				req.Builtins = append(kept, blk.builtins...)
			}
		}
	}

	// tool_choice
	if v, ok := fields["tool_choice"]; ok && !jsonx.IsNull(v) {
		tc, err := parseToolChoice(v, p.Key("tool_choice"), env)
		if err != nil {
			return nil, err
		}
		req.ToolChoice = tc
	}

	// Validate tool_choice references against declared tools.
	if req.ToolChoice.Mode != ir.ToolChoiceUnset {
		toolSet := make(map[string]bool, len(req.Tools))
		bareNameSet := make(map[string]bool, len(req.Tools))
		for _, t := range req.Tools {
			toolSet[t.Namespace+"\x00"+t.Name] = true
			bareNameSet[t.Name] = true
		}
		hasTool := func(ns, name string) bool {
			if name == "" {
				return false
			}
			if toolSet[ns+"\x00"+name] {
				return true
			}
			return ns == "" && bareNameSet[name]
		}
		hasAnyTools := len(req.Tools) > 0 || len(req.Builtins) > 0
		switch req.ToolChoice.Mode {
		case ir.ToolChoiceRequired:
			if !hasAnyTools {
				return nil, env.Err(ir.Invalid(p.Key("tool_choice"), `tool_choice "required" needs at least one tool`))
			}
		case ir.ToolChoiceTool:
			if !hasTool(req.ToolChoice.Namespace, req.ToolChoice.Name) {
				return nil, env.Err(ir.Invalid(p.Key("tool_choice").Key("name"), "tool_choice names a tool that is not in tools: "+req.ToolChoice.Name))
			}
		case ir.ToolChoiceAllowed:
			for i, ref := range req.ToolChoice.Allowed {
				if !hasTool(ref.Namespace, ref.Name) {
					return nil, env.Err(ir.Invalid(p.Key("tool_choice").Key("tools").At(i), "allowed tool not found in tools: "+ref.Name))
				}
			}
		}
	}

	// parallel_tool_calls
	if v, ok := fields["parallel_tool_calls"]; ok && !jsonx.IsNull(v) {
		b, ok2 := jsonx.Bool(v)
		if !ok2 {
			return nil, env.Err(ir.Invalid(p.Key("parallel_tool_calls"), "must be a boolean"))
		}
		req.ParallelToolCalls = &b
	}

	// sampling
	if v, ok := fields["temperature"]; ok && !jsonx.IsNull(v) {
		f, ok2 := jsonx.Float(v)
		if !ok2 {
			return nil, env.Err(ir.Invalid(p.Key("temperature"), "must be a number"))
		}
		if f < 0 || f > 2 {
			return nil, env.Err(ir.Invalid(p.Key("temperature"), "must be between 0 and 2"))
		}
		req.Temperature = &f
	}
	if v, ok := fields["top_p"]; ok && !jsonx.IsNull(v) {
		f, ok2 := jsonx.Float(v)
		if !ok2 {
			return nil, env.Err(ir.Invalid(p.Key("top_p"), "must be a number"))
		}
		if f < 0 || f > 1 {
			return nil, env.Err(ir.Invalid(p.Key("top_p"), "must be between 0 and 1"))
		}
		req.TopP = &f
	}
	if v, ok := fields["max_output_tokens"]; ok && !jsonx.IsNull(v) {
		n, ok2 := jsonx.Int(v)
		if !ok2 {
			return nil, env.Err(ir.Invalid(p.Key("max_output_tokens"), "must be an integer"))
		}
		if n <= 0 {
			return nil, env.Err(ir.Invalid(p.Key("max_output_tokens"), "must be greater than 0"))
		}
		ni := int(n)
		req.MaxOutputTokens = &ni
	}
	if v, ok := fields["seed"]; ok && !jsonx.IsNull(v) {
		if n, ok2 := jsonx.Int(v); ok2 {
			req.Seed = &n
		}
	}
	if v, ok := fields["presence_penalty"]; ok && !jsonx.IsNull(v) {
		f, ok2 := jsonx.Float(v)
		if !ok2 {
			return nil, env.Err(ir.Invalid(p.Key("presence_penalty"), "must be a number"))
		}
		if f < -2 || f > 2 {
			return nil, env.Err(ir.Invalid(p.Key("presence_penalty"), "must be between -2 and 2"))
		}
		req.PresencePenalty = &f
	}
	if v, ok := fields["frequency_penalty"]; ok && !jsonx.IsNull(v) {
		f, ok2 := jsonx.Float(v)
		if !ok2 {
			return nil, env.Err(ir.Invalid(p.Key("frequency_penalty"), "must be a number"))
		}
		if f < -2 || f > 2 {
			return nil, env.Err(ir.Invalid(p.Key("frequency_penalty"), "must be between -2 and 2"))
		}
		req.FrequencyPenalty = &f
	}
	if v, ok := fields["top_logprobs"]; ok && !jsonx.IsNull(v) {
		n, ok2 := jsonx.Int(v)
		if !ok2 || n < 0 {
			return nil, env.Err(ir.Invalid(p.Key("top_logprobs"), "must be a non-negative integer"))
		}
		if n > 0 {
			ni := int(n)
			req.TopLogprobs = &ni
			if err := env.Lose(ir.LossSamplingLogprobs, p.Key("top_logprobs"), "top_logprobs is not supported"); err != nil {
				return nil, err
			}
		}
	}
	if v, ok := fields["logprobs"]; ok && !jsonx.IsNull(v) {
		b, ok2 := jsonx.Bool(v)
		if !ok2 {
			return nil, env.Err(ir.Invalid(p.Key("logprobs"), "must be a boolean"))
		}
		if b {
			req.Logprobs = &b
			if err := env.Lose(ir.LossSamplingLogprobs, p.Key("logprobs"), "logprobs is not supported"); err != nil {
				return nil, err
			}
		}
	}
	if v, ok := fields["truncation"]; ok && !jsonx.IsNull(v) {
		s, ok2 := jsonx.String(v)
		if !ok2 {
			return nil, env.Err(ir.Invalid(p.Key("truncation"), "must be a string"))
		}
		switch s {
		case "disabled":
			// accepted no-op
		case "auto":
			if err := env.Lose(ir.LossUnknownField, p.Key("truncation"), `truncation "auto" is not supported`); err != nil {
				return nil, err
			}
		default:
			return nil, env.Err(ir.Invalid(p.Key("truncation"), "unknown truncation mode: "+s))
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
		for i, e := range elems {
			s, ok3 := jsonx.String(e)
			if !ok3 {
				return nil, env.Err(ir.Invalid(p.Key("include").At(i), "must be a string"))
			}
			if s != "" {
				req.Include = append(req.Include, s)
				if s == "reasoning.encrypted_content" {
					if req.Reasoning == nil {
						req.Reasoning = &ir.ReasoningConfig{}
					}
					req.Reasoning.IncludeEncrypted = true
				}
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

	// Input must contain at least one non-system/developer message unless continuing
	// from previous_response_id.
	// A compaction item alone carries the history; a configuration_update
	// alone does not.
	if req.PreviousResponseID == "" && len(req.Messages) == 0 && !hasCompactionExtra(req.TrailingExtras) {
		return nil, env.Err(ir.Invalid(p.Key("input"), "input must contain at least one user or assistant item"))
	}

	// service_tier, user, metadata, prompt_cache_key, prompt_cache_retention
	if v, ok := fields["service_tier"]; ok {
		req.ServiceTier, _ = jsonx.String(v)
	}
	if v, ok := fields["user"]; ok {
		req.User, _ = jsonx.String(v)
	}
	if v, ok := fields["metadata"]; ok {
		req.Metadata = append(json.RawMessage(nil), v...)
	}
	if v, ok := fields["prompt_cache_key"]; ok && !jsonx.IsNull(v) {
		req.PromptCacheKey, _ = jsonx.String(v)
	}
	if v, ok := fields["prompt_cache_retention"]; ok && !jsonx.IsNull(v) {
		req.PromptCacheRetention, _ = jsonx.String(v)
	}

	// Echo: preserve client fields for response echoing.
	echoFields := map[string]json.RawMessage{}
	for _, kv := range kvs {
		switch kv.Key {
		case "model", "input", "instructions", "tools", "tool_choice",
			"parallel_tool_calls", "temperature", "top_p", "max_output_tokens",
			"seed", "stop", "reasoning", "text", "include", "store",
			"previous_response_id", "stream", "service_tier", "user", "metadata",
			"presence_penalty", "frequency_penalty", "top_logprobs", "logprobs",
			"truncation", "prompt_cache_key", "prompt_cache_retention":
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

type additionalToolBlock struct {
	tools    []ir.Tool
	builtins []ir.BuiltinTool
}

// parseInput handles the input field (string or array).
func parseInput(raw json.RawMessage, p ir.Path, env *ir.Env, hasPrevRespID bool, trailing *[]ir.InputExtra) ([]ir.Message, []additionalToolBlock, error) {
	switch jsonx.Kind(raw) {
	case '"':
		s, _ := jsonx.String(raw)
		return []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart(s)}}}, nil, nil
	case '[':
		return parseInputItems(raw, p, env, hasPrevRespID, trailing)
	}
	return nil, nil, env.Err(ir.Invalid(p, "must be a string or array"))
}

// parseInputItems parses a Responses input array into ir.Messages and any
// additional_tools declarations embedded in input[].
// It groups consecutive function_call and reasoning items into one assistant
// message, and groups consecutive function_call_output items into one user message.
// Items without a message equivalent (configuration_update, compaction, ...)
// are attached as ir.InputExtra to the message that follows them, or returned
// through trailing when no message follows.
func parseInputItems(raw json.RawMessage, p ir.Path, env *ir.Env, hasPrevRespID bool, trailing *[]ir.InputExtra) ([]ir.Message, []additionalToolBlock, error) {
	elems, ok := jsonx.Array(raw)
	if !ok {
		return nil, nil, env.Err(ir.Invalid(p, "must be an array"))
	}

	var msgs []ir.Message
	var extraTools []additionalToolBlock
	knownCalls := map[string]ir.ToolKind{}
	answeredCalls := map[string]bool{}

	type pending struct {
		role  ir.Role
		parts []ir.Part
	}
	var cur *pending
	var pendingExtras []ir.InputExtra
	appendMsg := func(m ir.Message) {
		if len(pendingExtras) > 0 {
			m.Extras = append(pendingExtras, m.Extras...)
			pendingExtras = nil
		}
		msgs = append(msgs, m)
	}

	flush := func() {
		if cur != nil && len(cur.parts) > 0 {
			appendMsg(ir.Message{Role: cur.role, Parts: cur.parts})
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
			return nil, nil, env.Err(ir.Invalid(fp, "must be an object"))
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
				return nil, nil, err
			}
			appendMsg(msg)

		case "reasoning":
			c := ensure(ir.RoleAssistant)
			part, err := parseReasoningItem(fields, fp, env)
			if err != nil {
				return nil, nil, err
			}
			c.parts = append(c.parts, part)

		case "function_call":
			c := ensure(ir.RoleAssistant)
			part, err := parseFunctionCallItem(fields, fp, env)
			if err != nil {
				return nil, nil, err
			}
			if _, dup := knownCalls[part.Call.ID]; dup {
				return nil, nil, env.Err(ir.Invalid(fp.Key("call_id"), "duplicate call_id: "+part.Call.ID))
			}
			knownCalls[part.Call.ID] = ir.ToolFunction
			c.parts = append(c.parts, part)

		case "function_call_output":
			c := ensure(ir.RoleUser)
			part, err := parseFunctionCallOutputItem(fields, fp, env)
			if err != nil {
				return nil, nil, err
			}
			callID := part.Result.CallID
			callKind, known := knownCalls[callID]
			if !known {
				if !hasPrevRespID && !env.Repairs(ir.RepairOrphanResults) {
					return nil, nil, env.Err(ir.Invalid(fp.Key("call_id"), "no tool call found for call_id: "+callID))
				}
			} else if callKind != ir.ToolFunction {
				return nil, nil, env.Err(ir.Invalid(fp.Key("type"), "call_id "+callID+" belongs to a custom_tool call, not function"))
			}
			if answeredCalls[callID] && !env.Repairs(ir.RepairDuplicateResults) {
				return nil, nil, env.Err(ir.Invalid(fp.Key("call_id"), "call_id already has an output: "+callID))
			}
			answeredCalls[callID] = true
			c.parts = append(c.parts, part)

		case "custom_tool_call":
			c := ensure(ir.RoleAssistant)
			part, err := parseCustomToolCallItem(fields, fp, env)
			if err != nil {
				return nil, nil, err
			}
			if _, dup := knownCalls[part.Call.ID]; dup {
				return nil, nil, env.Err(ir.Invalid(fp.Key("call_id"), "duplicate call_id: "+part.Call.ID))
			}
			knownCalls[part.Call.ID] = ir.ToolCustom
			c.parts = append(c.parts, part)

		case "custom_tool_call_output":
			c := ensure(ir.RoleUser)
			part, err := parseCustomToolCallOutputItem(fields, fp, env)
			if err != nil {
				return nil, nil, err
			}
			callID := part.Result.CallID
			callKind, known := knownCalls[callID]
			if !known {
				if !hasPrevRespID && !env.Repairs(ir.RepairOrphanResults) {
					return nil, nil, env.Err(ir.Invalid(fp.Key("call_id"), "no tool call found for call_id: "+callID))
				}
			} else if callKind != ir.ToolCustom {
				return nil, nil, env.Err(ir.Invalid(fp.Key("type"), "call_id "+callID+" belongs to a function call, not custom_tool"))
			}
			if answeredCalls[callID] && !env.Repairs(ir.RepairDuplicateResults) {
				return nil, nil, env.Err(ir.Invalid(fp.Key("call_id"), "call_id already has an output: "+callID))
			}
			answeredCalls[callID] = true
			c.parts = append(c.parts, part)

		case "additional_tools":
			if roleRaw, ok := fields["role"]; ok && !jsonx.IsNull(roleRaw) {
				if _, okStr := jsonx.String(roleRaw); !okStr {
					return nil, nil, env.Err(ir.Invalid(fp.Key("role"), "role must be a string"))
				}
			}
			if toolsRaw, ok := fields["tools"]; ok && !jsonx.IsNull(toolsRaw) {
				subTools, subBuiltins, err := parseToolsWithSeen(toolsRaw, fp.Key("tools"), env, map[string]bool{}, "")
				if err != nil {
					return nil, nil, err
				}
				extraTools = append(extraTools, additionalToolBlock{
					tools:    subTools,
					builtins: subBuiltins,
				})
			}

		case "item_reference":
			return nil, nil, env.Err(ir.Unsupported(fp.Key("type"), "item_reference", "item_reference is not supported"))

		case ir.ExtraConfigurationUpdate:
			flush()
			ex, err := parseConfigurationUpdate(fields, elem, i, fp, env)
			if err != nil {
				return nil, nil, err
			}
			pendingExtras = append(pendingExtras, ex)

		case ir.ExtraCompaction, ir.ExtraCompactionSummary, ir.ExtraCompactionTrigger:
			// Codex remote compaction: kept verbatim; only a Responses upstream
			// can act on it (the build reports it to any other upstream).
			flush()
			pendingExtras = append(pendingExtras, ir.InputExtra{Type: itemType, Index: i, Raw: append(json.RawMessage(nil), elem...)})

		default:
			return nil, nil, env.Err(ir.Invalid(fp.Key("type"), "unknown input item type: "+itemType))
		}
	}
	flush()
	if len(pendingExtras) > 0 && trailing != nil {
		*trailing = append(*trailing, pendingExtras...)
	}
	return msgs, extraTools, nil
}

// parseConfigurationUpdate reads a configuration_update input item, which
// changes reasoning (and optionally the text format) for this response and
// the following ones until another update (OpenAI Reasoning guide, "Change
// reasoning mid-conversation").
func parseConfigurationUpdate(fields map[string]json.RawMessage, elem json.RawMessage, index int, p ir.Path, env *ir.Env) (ir.InputExtra, error) {
	ex := ir.InputExtra{Type: ir.ExtraConfigurationUpdate, Index: index, Raw: append(json.RawMessage(nil), elem...)}
	if v, ok := fields["reasoning"]; ok && !jsonx.IsNull(v) {
		rc, err := parseReasoning(v, p.Key("reasoning"), env)
		if err != nil {
			return ex, err
		}
		ex.Reasoning = rc
	}
	if v, ok := fields["text"]; ok && !jsonx.IsNull(v) {
		of, err := parseTextFormat(v, p.Key("text"), env)
		if err != nil {
			return ex, err
		}
		ex.Output = of
	}
	return ex, nil
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
		fileID, _ := jsonx.String(fields["file_id"])
		imageURL, _ := jsonx.String(fields["image_url"])
		if imageURL == "" && len(fields["image_url"]) > 0 && jsonx.Kind(fields["image_url"]) == '{' {
			if ikvs, err := jsonx.Object(fields["image_url"]); err == nil {
				for _, ikv := range ikvs {
					if ikv.Key == "url" {
						imageURL, _ = jsonx.String(ikv.Value)
					}
				}
			}
		}
		if imageURL == "" && fileID == "" {
			return ir.Part{}, env.Err(ir.Invalid(p.Key("image_url"), "input_image requires image_url"))
		}
		detail, _ := jsonx.String(fields["detail"])
		media := &ir.Media{URL: imageURL, FileID: fileID, Detail: detail}
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
		fileURL, _ := jsonx.String(fields["file_url"])
		media := &ir.Media{FileID: fileID, Filename: filename, Data: fileData, URL: fileURL}
		// OpenAI sends file_data as a data URI ("data:application/pdf;base64,...");
		// one with other parameters is kept whole, as before.
		if mediaType, data, ok := parseDataURI(fileData); ok {
			media.MediaType = mediaType
			media.Data = data
		}
		return ir.Part{Type: ir.PartFile, Media: media}, nil

	default:
		return ir.Part{}, env.Err(ir.Invalid(p.Key("type"), "unknown content part type: "+partType))
	}
}

func parseReasoningItem(fields map[string]json.RawMessage, p ir.Path, env *ir.Env) (ir.Part, error) {
	id, _ := jsonx.String(fields["id"])
	enc, _ := jsonx.String(fields["encrypted_content"])

	if enc != "" {
		if origin, text, ok := ir.VerifyUnsignedReasoning(env, ir.FormatResponses, enc, ""); ok {
			return ir.Part{
				Type: ir.PartReasoning,
				Reasoning: &ir.Reasoning{
					ID:        id,
					Text:      text,
					Signature: &ir.Signature{Value: enc, Origin: origin, Unsigned: true},
				},
			}, nil
		}
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
	var summaryList []string
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
					if s != "" {
						summaryList = append(summaryList, s)
						summaryText.WriteString(s)
					}
				}
			}
		}
	}

	// Raw reasoning content parts (e.g. reasoning_text when summary was not requested).
	var contentText strings.Builder
	if cv, ok := fields["content"]; ok {
		elems, _ := jsonx.Array(cv)
		for _, e := range elems {
			kvs2, err := jsonx.Object(e)
			if err != nil {
				continue
			}
			var ptype, ptext string
			for _, kv := range kvs2 {
				switch kv.Key {
				case "type":
					ptype, _ = jsonx.String(kv.Value)
				case "text":
					ptext, _ = jsonx.String(kv.Value)
				}
			}
			if (ptype == "" || ptype == "reasoning_text") && ptext != "" {
				contentText.WriteString(ptext)
			}
		}
	}

	rtext := summaryText.String()
	if rtext == "" {
		rtext = contentText.String()
	}

	return ir.Part{
		Type:      ir.PartReasoning,
		Reasoning: &ir.Reasoning{ID: id, Text: rtext, Summary: summaryList},
	}, nil
}

func parseFunctionCallItem(fields map[string]json.RawMessage, p ir.Path, env *ir.Env) (ir.Part, error) {
	rawCallID, _ := jsonx.String(fields["call_id"])
	if rawCallID == "" {
		return ir.Part{}, env.Err(ir.Invalid(p.Key("call_id"), "call_id is required"))
	}
	callID, sig := ir.DecodeCallSignatureID(env, rawCallID)
	name, _ := jsonx.String(fields["name"])
	if name == "" {
		return ir.Part{}, env.Err(ir.Invalid(p.Key("name"), "name is required"))
	}
	ns, _ := jsonx.String(fields["namespace"])
	var argsStr string
	if rawArgs, ok := fields["arguments"]; ok && !jsonx.IsNull(rawArgs) {
		var okStr bool
		argsStr, okStr = jsonx.String(rawArgs)
		if !okStr {
			return ir.Part{}, env.Err(ir.Invalid(p.Key("arguments"), "arguments must be a JSON-encoded string"))
		}
	}
	args := json.RawMessage(argsStr)
	if strings.TrimSpace(argsStr) == "" {
		args = json.RawMessage(`{}`)
	}
	return ir.Part{
		Type: ir.PartToolCall,
		Call: &ir.ToolCall{
			ID:        callID,
			Name:      name,
			Namespace: ns,
			Kind:      ir.ToolFunction,
			Arguments: args,
			Signature: sig,
		},
	}, nil
}

func parseFunctionCallOutputItem(fields map[string]json.RawMessage, p ir.Path, env *ir.Env) (ir.Part, error) {
	rawCallID, _ := jsonx.String(fields["call_id"])
	if rawCallID == "" {
		return ir.Part{}, env.Err(ir.Invalid(p.Key("call_id"), "call_id is required"))
	}
	callID, _ := ir.DecodeCallSignatureID(env, rawCallID)
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
	rawCallID, _ := jsonx.String(fields["call_id"])
	if rawCallID == "" {
		return ir.Part{}, env.Err(ir.Invalid(p.Key("call_id"), "call_id is required"))
	}
	callID, sig := ir.DecodeCallSignatureID(env, rawCallID)
	name, _ := jsonx.String(fields["name"])
	if name == "" {
		return ir.Part{}, env.Err(ir.Invalid(p.Key("name"), "name is required"))
	}
	ns, _ := jsonx.String(fields["namespace"])
	var input string
	if rawIn, ok := fields["input"]; ok && !jsonx.IsNull(rawIn) {
		var okStr bool
		input, okStr = jsonx.String(rawIn)
		if !okStr {
			return ir.Part{}, env.Err(ir.Invalid(p.Key("input"), "input must be a string"))
		}
	}
	args, _ := json.Marshal(map[string]string{"input": input})
	return ir.Part{
		Type: ir.PartToolCall,
		Call: &ir.ToolCall{
			ID:        callID,
			Name:      name,
			Namespace: ns,
			Kind:      ir.ToolCustom,
			Arguments: args,
			Input:     input,
			Signature: sig,
		},
	}, nil
}

func parseCustomToolCallOutputItem(fields map[string]json.RawMessage, p ir.Path, env *ir.Env) (ir.Part, error) {
	rawCallID, _ := jsonx.String(fields["call_id"])
	if rawCallID == "" {
		return ir.Part{}, env.Err(ir.Invalid(p.Key("call_id"), "call_id is required"))
	}
	callID, _ := ir.DecodeCallSignatureID(env, rawCallID)
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
	return parseToolsWithSeen(raw, p, env, map[string]bool{}, "")
}

func parseToolsWithSeen(raw json.RawMessage, p ir.Path, env *ir.Env, seen map[string]bool, namespace string) ([]ir.Tool, []ir.BuiltinTool, error) {
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
			t.Namespace = namespace
			key := t.Namespace + "\x00" + t.Name
			if seen[key] {
				return nil, nil, env.Err(ir.Invalid(fp.Key("name"), "duplicate tool name: "+t.Name))
			}
			seen[key] = true
			tools = append(tools, t)
		case "custom":
			t, err := parseCustomTool(fields, fp, env)
			if err != nil {
				return nil, nil, err
			}
			t.Namespace = namespace
			key := t.Namespace + "\x00" + t.Name
			if seen[key] {
				return nil, nil, env.Err(ir.Invalid(fp.Key("name"), "duplicate tool name: "+t.Name))
			}
			seen[key] = true
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
			subTools, subBuiltins, err := parseToolsWithSeen(subRaw, fp.Key("tools"), env, seen, nsName)
			if err != nil {
				return nil, nil, err
			}
			tools = append(tools, subTools...)
			builtins = append(builtins, subBuiltins...)
		case "web_search", "web_search_preview":
			var cfgBytes json.RawMessage
			cfgMap := map[string]json.RawMessage{}
			for k, v := range fields {
				if k != "type" {
					cfgMap[k] = v
				}
			}
			if len(cfgMap) > 0 {
				cfgBytes, _ = jsonx.Marshal(cfgMap)
			}
			builtins = append(builtins, ir.BuiltinTool{
				Kind:   ir.BuiltinWebSearch,
				Origin: ir.FormatResponses,
				Config: cfgBytes,
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
		case "function", "custom":
			name, _ := jsonx.String(fields["name"])
			ns, _ := jsonx.String(fields["namespace"])
			return ir.ToolChoice{Mode: ir.ToolChoiceTool, Name: name, Namespace: ns}, nil
		case "allowed_tools":
			var allowed []ir.ToolRef
			var allowedMode ir.ToolChoiceMode
			if mv, ok := fields["mode"]; ok && !jsonx.IsNull(mv) {
				ms, okStr := jsonx.String(mv)
				if !okStr {
					return ir.ToolChoice{}, env.Err(ir.Invalid(p.Key("mode"), "must be a string"))
				}
				switch ms {
				case "", "auto":
					allowedMode = ir.ToolChoiceAuto
				case "required":
					allowedMode = ir.ToolChoiceRequired
				default:
					return ir.ToolChoice{}, env.Err(ir.Invalid(p.Key("mode"), "unknown allowed_tools mode: "+ms))
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
			if allowedMode == ir.ToolChoiceRequired && len(allowed) == 0 {
				return ir.ToolChoice{}, env.Err(ir.Invalid(p.Key("tools"), `tool_choice "required" needs at least one allowed tool`))
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
			if jsonx.IsNull(kv.Value) {
				break
			}
			s, ok := jsonx.String(kv.Value)
			if !ok {
				return nil, env.Err(ir.Invalid(p.Key("summary"), "must be a string"))
			}
			switch s {
			case "", "none", "auto", "concise", "detailed":
				rc.Summary = s
			default:
				return nil, env.Err(ir.Invalid(p.Key("summary"), "unrecognised reasoning summary: "+s))
			}
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

func hasCompactionExtra(extras []ir.InputExtra) bool {
	for _, ex := range extras {
		if ex.Type == ir.ExtraCompaction || ex.Type == ir.ExtraCompactionSummary {
			return true
		}
	}
	return false
}
