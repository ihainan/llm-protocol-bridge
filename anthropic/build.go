package anthropic

import (
	"encoding/json"
	"strings"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// BuildRequest serialises an ir.Request to Anthropic Messages JSON.
func BuildRequest(req *ir.Request, opts BuildOptions) (*Built, error) {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	env.Style = ir.StyleDot
	dia := opts.Dialect
	names := map[string]ir.ToolRef{}

	// max_tokens: required by Anthropic.
	profile := env.ModelProfileOr(nil)
	maxTok := 0
	if req.MaxOutputTokens != nil {
		maxTok = *req.MaxOutputTokens
	} else if profile != nil && profile.DefaultMaxTokens > 0 {
		maxTok = profile.DefaultMaxTokens
	} else if dia.DefaultMaxTokens > 0 {
		maxTok = dia.DefaultMaxTokens
	} else {
		maxTok = 16384 // cross-format conversion default
	}

	// System prompt.
	var sysRaw json.RawMessage
	if len(req.Instructions) > 0 {
		sysRaw = buildContentBlocks(req.Instructions, env)
	}

	// Messages.
	var err error
	var tcRaw json.RawMessage
	var thinkingRaw json.RawMessage
	var thinkingEnabled bool
	var adaptiveEffort ir.Effort
	messages, err := buildMessages(req.Messages, names, env, dia)
	if err != nil {
		return nil, err
	}

	// Reject multiple candidates (Anthropic only produces 1 candidate).
	if req.Candidates != nil && *req.Candidates > 1 {
		return nil, env.Err(ir.Unsupported(ir.Root.Key("n"), "candidates", "Anthropic supports only a single candidate"))
	}

	// Tools (filtering by ToolChoiceAllowed if set) + Builtins.
	var allowedSet map[string]bool
	if req.ToolChoice.Mode == ir.ToolChoiceAllowed {
		allowedSet = map[string]bool{}
		for _, ref := range req.ToolChoice.Allowed {
			allowedSet[ref.Name] = true
		}
	}
	var rawTools []json.RawMessage
	for i, t := range req.Tools {
		if allowedSet != nil && !allowedSet[t.Name] {
			continue
		}
		fp := ir.Root.Key("tools").At(i)
		wt, err := buildTool(t, fp, dia, env)
		if err != nil {
			return nil, err
		}
		names[t.Name] = ir.ToolRef{Name: t.Name, Namespace: t.Namespace, Kind: t.Kind}
		tb, err := jsonx.Marshal(wt)
		if err != nil {
			return nil, err
		}
		rawTools = append(rawTools, tb)
	}
	for i, bt := range req.Builtins {
		fp := ir.Root.Key("tools").At(len(req.Tools) + i)
		switch bt.Kind {
		case ir.BuiltinWebSearch:
			if bt.Origin == ir.FormatAnthropic && len(bt.Raw) > 0 {
				rawTools = append(rawTools, bt.Raw)
			} else {
				tb, _ := jsonx.Marshal(map[string]any{"type": "web_search_20250305", "name": "web_search"})
				rawTools = append(rawTools, tb)
			}
		case ir.BuiltinWebFetch:
			if bt.Origin == ir.FormatAnthropic && len(bt.Raw) > 0 {
				rawTools = append(rawTools, bt.Raw)
			} else {
				tb, _ := jsonx.Marshal(map[string]any{"type": "web_fetch_20250910", "name": "web_fetch"})
				rawTools = append(rawTools, tb)
			}
		default:
			return nil, env.Err(ir.Unsupported(fp, string(bt.Kind), "builtin tool not supported by Anthropic"))
		}
	}

	// Structured output format.
	var outputFormat map[string]any
	if req.Output != nil {
		switch req.Output.Type {
		case ir.OutputJSONSchema:
			outputFormat = map[string]any{
				"type":   "json_schema",
				"schema": json.RawMessage(req.Output.Schema),
			}
		case ir.OutputJSONObject:
			return nil, env.Err(ir.Unsupported(ir.Root.Key("response_format"), "json_object", "Anthropic requires json_schema rather than json_object"))
		case ir.OutputText, "":
			// default text output
		default:
			return nil, env.Err(ir.Unsupported(ir.Root.Key("response_format"), string(req.Output.Type), "unsupported output format for Anthropic"))
		}
	}

	// Tool choice. Emitted when ToolChoice.Mode is set, or when
	// ParallelToolCalls is set to false (Anthropic expresses it via
	// disable_parallel_tool_use on the tool_choice object).
	if req.ToolChoice.Mode != "" || (req.ParallelToolCalls != nil && !*req.ParallelToolCalls) {
		tcRaw, err = buildToolChoice(req.ToolChoice, req.ParallelToolCalls, env)
		if err != nil {
			return nil, err
		}
	}

	// Thinking config resolved via ModelProfile.
	if req.Reasoning != nil || (profile != nil && profile.DefaultEffort != ir.EffortUnset) {
		var budget int
		thinkingRaw, thinkingEnabled, budget, adaptiveEffort, err = buildThinking(req.Reasoning, profile, env)
		if err != nil {
			return nil, err
		}
		if thinkingEnabled && budget > 0 && maxTok <= budget {
			minBudget := 1024
			if profile != nil && profile.MinBudget > 0 {
				minBudget = profile.MinBudget
			}
			if req.MaxOutputTokens == nil {
				maxTok = budget + 4096
			} else if maxTok > minBudget {
				budget = maxTok - 1
				thinkingRaw, _ = jsonx.Marshal(wireThinkingConfig{Type: "enabled", BudgetTokens: budget})
			} else {
				maxTok = budget + 1024
			}
		}
	}

	// Assemble.
	out := jsonx.Start(nil)
	out.Str("model", req.Model)

	if sysRaw != nil {
		out.Raw("system", sysRaw)
	}

	msgsBytes, err := jsonx.Marshal(messages)
	if err != nil {
		return nil, err
	}
	out.Raw("messages", msgsBytes)
	out.Int("max_tokens", int64(maxTok))

	if len(rawTools) > 0 {
		tb, err := jsonx.Marshal(rawTools)
		if err != nil {
			return nil, err
		}
		out.Raw("tools", tb)
	}
	if tcRaw != nil {
		out.Raw("tool_choice", tcRaw)
	}
	if thinkingRaw != nil {
		out.Raw("thinking", thinkingRaw)
	}
	if adaptiveEffort != ir.EffortUnset || outputFormat != nil {
		oc := map[string]any{}
		if adaptiveEffort != ir.EffortUnset {
			oc["effort"] = string(adaptiveEffort)
		}
		if outputFormat != nil {
			oc["format"] = outputFormat
		}
		ocBytes, _ := jsonx.Marshal(oc)
		out.Raw("output_config", ocBytes)
	}

	// Sampling parameters (Anthropic forbids non-default temperature/top_p/top_k when thinking is enabled).
	allowSampling := !thinkingEnabled || (profile != nil && profile.SamplingWithReasoning)
	if req.Temperature != nil {
		t := *req.Temperature
		if !allowSampling && t != 1.0 {
			if err := env.Lose(ir.LossSamplingClamp, ir.Root.Key("temperature"), "omitted because Anthropic thinking forbids non-default temperature"); err != nil {
				return nil, err
			}
		} else {
			if t > 1.0 {
				if err := env.Lose(ir.LossSamplingClamp, ir.Root.Key("temperature"), "clamped to 1.0 (Anthropic max)"); err != nil {
					return nil, err
				}
				t = 1.0
			}
			tb, _ := jsonx.Marshal(t)
			out.Raw("temperature", tb)
		}
	}
	if req.TopP != nil {
		if !allowSampling {
			if err := env.Lose(ir.LossSamplingClamp, ir.Root.Key("top_p"), "omitted when Anthropic thinking is enabled"); err != nil {
				return nil, err
			}
		} else {
			tb, _ := jsonx.Marshal(*req.TopP)
			out.Raw("top_p", tb)
		}
	}
	if req.TopK != nil {
		if !allowSampling {
			if err := env.Lose(ir.LossSamplingTopK, ir.Root.Key("top_k"), "omitted when Anthropic thinking is enabled"); err != nil {
				return nil, err
			}
		} else {
			out.Int("top_k", int64(*req.TopK))
		}
	}
	// Anthropic does not support presence/frequency penalty.
	if req.PresencePenalty != nil && *req.PresencePenalty != 0 {
		if err := env.Lose(ir.LossSamplingPenalty, ir.Root.Key("presence_penalty"), "not supported by Anthropic"); err != nil {
			return nil, err
		}
	}
	if req.FrequencyPenalty != nil && *req.FrequencyPenalty != 0 {
		if err := env.Lose(ir.LossSamplingPenalty, ir.Root.Key("frequency_penalty"), "not supported by Anthropic"); err != nil {
			return nil, err
		}
	}
	// Anthropic does not support logprobs.
	if req.Logprobs != nil && *req.Logprobs {
		if err := env.Lose(ir.LossSamplingLogprobs, ir.Root.Key("logprobs"), "not supported by Anthropic"); err != nil {
			return nil, err
		}
	}
	// Seed is an allowed-by-default loss; record it so it appears in Plan.Losses.
	if req.Seed != nil {
		if err := env.Lose(ir.LossSamplingSeed, ir.Root.Key("seed"), "not supported by Anthropic"); err != nil {
			return nil, err
		}
	}
	if len(req.Stop) > 0 {
		sb, _ := jsonx.Marshal(req.Stop)
		out.Raw("stop_sequences", sb)
	}
	if req.Stream {
		out.Bool("stream", true)
	}
	if req.User != "" {
		mb, _ := jsonx.Marshal(wireMetadata{UserID: req.User})
		out.Raw("metadata", mb)
	}
	if len(req.Metadata) > 0 {
		if err := env.Lose(ir.LossRequestMetadata, ir.Root.Key("metadata"), "arbitrary metadata map not supported by Anthropic"); err != nil {
			return nil, err
		}
	}
	if req.ServiceTier != "" {
		if err := env.Lose(ir.LossRequestTier, ir.Root.Key("service_tier"), "not supported by Anthropic"); err != nil {
			return nil, err
		}
	}
	if req.PromptCacheKey != "" {
		if err := env.Lose(ir.LossRequestCache, ir.Root.Key("prompt_cache_key"), "not supported by Anthropic"); err != nil {
			return nil, err
		}
	}
	if req.Store != nil {
		if err := env.Lose(ir.LossRequestStore, ir.Root.Key("store"), "not supported by Anthropic"); err != nil {
			return nil, err
		}
	}
	if len(req.Include) > 0 {
		if err := env.Lose(ir.LossRequestInclude, ir.Root.Key("include"), "not supported by Anthropic"); err != nil {
			return nil, err
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

// buildMessages converts ir.Messages to wireMessages, merging consecutive
// messages with the same role (required by the Anthropic Messages API).
func buildMessages(msgs []ir.Message, names map[string]ir.ToolRef, env *ir.Env, dia Dialect) ([]wireMessage, error) {
	type msgBlocks struct {
		role   string
		blocks []wireContentBlock
	}
	var merged []msgBlocks
	appendBlocks := func(role string, blocks []wireContentBlock) {
		if len(blocks) == 0 {
			return
		}
		if len(merged) > 0 && merged[len(merged)-1].role == role {
			merged[len(merged)-1].blocks = append(merged[len(merged)-1].blocks, blocks...)
			return
		}
		merged = append(merged, msgBlocks{role: role, blocks: blocks})
	}

	for i, m := range msgs {
		fp := ir.Root.Key("messages").At(i)
		switch m.Role {
		case ir.RoleUser:
			blocks, err := partsToBlocks(m.Parts, fp.Key("content"), env)
			if err != nil {
				return nil, err
			}
			appendBlocks("user", blocks)
		case ir.RoleAssistant:
			blocks, err := partsToBlocks(m.Parts, fp.Key("content"), env)
			if err != nil {
				return nil, err
			}
			appendBlocks("assistant", blocks)
		case ir.RoleSystem, ir.RoleDeveloper:
			switch dia.MidSystem {
			case "fold":
				// Handled in Stage 5 (fold into adjacent user message).
				var sb strings.Builder
				for _, pt := range m.Parts {
					if pt.Type == ir.PartText {
						sb.WriteString(pt.Text)
					}
				}
				if sb.Len() > 0 {
					appendBlocks("user", []wireContentBlock{{Type: "text", Text: "<system_message>" + sb.String() + "</system_message>"}})
				}
			default: // "keep"
				blocks, err := partsToBlocks(m.Parts, fp.Key("content"), env)
				if err != nil {
					return nil, err
				}
				appendBlocks("user", blocks)
			}
		default:
			return nil, env.Err(ir.Invalid(fp.Key("role"), "unsupported role: "+string(m.Role)))
		}
	}

	out := make([]wireMessage, 0, len(merged))
	for _, mb := range merged {
		content, err := jsonx.Marshal(mb.blocks)
		if err != nil {
			return nil, err
		}
		out = append(out, wireMessage{Role: mb.role, Content: content})
	}
	return out, nil
}

func partsToBlocks(parts []ir.Part, p ir.Path, env *ir.Env) ([]wireContentBlock, error) {
	var blocks []wireContentBlock
	for i, part := range parts {
		bs, err := partToBlocks(part, p.At(i), env)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, bs...)
	}
	return blocks, nil
}

func partToBlocks(part ir.Part, p ir.Path, env *ir.Env) ([]wireContentBlock, error) {
	switch part.Type {
	case ir.PartText:
		b := wireContentBlock{Type: "text", Text: part.Text}
		b.CacheControl = irCacheToWire(part.Cache)
		return []wireContentBlock{b}, nil

	case ir.PartRefusal:
		b := wireContentBlock{Type: "text", Text: part.Text}
		return []wireContentBlock{b}, nil

	case ir.PartImage:
		if part.Media == nil {
			return nil, nil
		}
		if part.Media.Detail != "" {
			if err := env.Lose(ir.LossImageDetail, p.Key("detail"), "image detail not supported by Anthropic"); err != nil {
				return nil, err
			}
		}
		src, err := buildImageSource(part.Media, p, env)
		if err != nil {
			return nil, err
		}
		b := wireContentBlock{Type: "image", Source: src}
		b.CacheControl = irCacheToWire(part.Cache)
		return []wireContentBlock{b}, nil

	case ir.PartFile:
		if part.Media == nil {
			return nil, nil
		}
		var src *wireImageSource
		var err error
		if part.Media.Text != "" {
			mt := part.Media.MediaType
			if mt == "" {
				mt = "text/plain"
			}
			src = &wireImageSource{Type: "text", MediaType: mt, Data: part.Media.Text}
		} else {
			src, err = buildImageSource(part.Media, p, env)
			if err != nil {
				return nil, err
			}
		}
		b := wireContentBlock{
			Type:    "document",
			Source:  src,
			Title:   part.Media.Title,
			Context: part.Media.Context,
		}
		b.CacheControl = irCacheToWire(part.Cache)
		return []wireContentBlock{b}, nil

	case ir.PartReasoning:
		if part.Reasoning == nil {
			return nil, nil
		}
		r := part.Reasoning
		if r.Sealed != nil {
			if r.Sealed.Kind == ir.SealedAnthropicRedacted && r.Sealed.Origin.Matches(ir.FormatAnthropic, env.ProviderID) {
				var dataStr string
				_ = json.Unmarshal(r.Sealed.Data, &dataStr)
				return []wireContentBlock{{Type: "redacted_thinking", Data: dataStr}}, nil
			}
			if err := env.Lose(ir.LossReasoningSealed, p, "foreign sealed reasoning block dropped for Anthropic"); err != nil {
				return nil, err
			}
			return nil, nil
		}
		if r.Text != "" {
			if r.Signature != nil && r.Signature.Value != "" && r.Signature.Origin.Matches(ir.FormatAnthropic, env.ProviderID) {
				return []wireContentBlock{{Type: "thinking", Thinking: r.Text, Signature: r.Signature.Value}}, nil
			}
			if err := env.Lose(ir.LossReasoningSignature, p, "unsigned or foreign reasoning block dropped for Anthropic"); err != nil {
				return nil, err
			}
			return nil, nil
		}
		return nil, nil

	case ir.PartToolCall:
		if part.Call == nil {
			return nil, nil
		}
		call := part.Call
		args := call.Arguments
		if len(args) == 0 {
			args = json.RawMessage(`{}`)
		}
		return []wireContentBlock{{
			Type:  "tool_use",
			ID:    ir.AnthropicSafeID(call.ID),
			Name:  call.Name,
			Input: args,
		}}, nil

	case ir.PartToolResult:
		if part.Result == nil {
			return nil, nil
		}
		r := part.Result
		var content json.RawMessage
		// Build content blocks from result content.
		var inner []wireContentBlock
		for i, cp := range r.Content {
			fp := p.Key("content").At(i)
			bs, err := partToBlocks(cp, fp, env)
			if err != nil {
				return nil, err
			}
			inner = append(inner, bs...)
		}
		if len(inner) == 1 && inner[0].Type == "text" {
			// Use plain string form when it's a single text block.
			content, _ = jsonx.Marshal(inner[0].Text)
		} else if len(inner) > 0 {
			content, _ = jsonx.Marshal(inner)
		} else {
			content = json.RawMessage(`""`)
		}
		isErr := r.IsError
		b := wireContentBlock{
			Type:      "tool_result",
			ToolUseID: ir.AnthropicSafeID(r.CallID),
			Content:   content,
		}
		if isErr {
			b.IsError = &isErr
		}
		return []wireContentBlock{b}, nil

	case ir.PartBuiltinCall, ir.PartBuiltinResult:
		return nil, nil

	default:
		return nil, env.Err(ir.Unsupported(p, string(part.Type), "part type not supported by Anthropic"))
	}
}

func buildImageSource(m *ir.Media, p ir.Path, env *ir.Env) (*wireImageSource, error) {
	if m.FileID != "" {
		return &wireImageSource{Type: "file", FileID: m.FileID}, nil
	}
	if m.URL != "" {
		return &wireImageSource{Type: "url", URL: m.URL}, nil
	}
	if m.Data != "" {
		mt := m.MediaType
		if mt == "" {
			mt = "image/jpeg"
		}
		return &wireImageSource{Type: "base64", MediaType: mt, Data: m.Data}, nil
	}
	return nil, env.Err(ir.Invalid(p, "media has no source (url, data, or file_id)"))
}

func buildContentBlocks(parts []ir.Part, env *ir.Env) json.RawMessage {
	var blocks []wireContentBlock
	for i, p := range parts {
		fp := ir.Root.Key("system").At(i)
		bs, err := partToBlocks(p, fp, env)
		if err != nil {
			continue
		}
		blocks = append(blocks, bs...)
	}
	// If a single text block, use plain string.
	if len(blocks) == 1 && blocks[0].Type == "text" && blocks[0].CacheControl == nil {
		b, _ := jsonx.Marshal(blocks[0].Text)
		return b
	}
	b, _ := jsonx.Marshal(blocks)
	return b
}

func buildTool(t ir.Tool, p ir.Path, dia Dialect, env *ir.Env) (wireTool, error) {
	if t.Name == "" {
		return wireTool{}, env.Err(ir.Invalid(p.Key("name"), "tool name is required"))
	}
	if t.Strict != nil && *t.Strict && !dia.StrictTools {
		if err := env.Lose(ir.LossToolStrict, p.Key("strict"), "strict tool schemas not supported by Anthropic dialect"); err != nil {
			return wireTool{}, err
		}
	}
	params := t.Schema
	if jsonx.Absent(params) {
		params = schema_emptyObj
	}
	wt := wireTool{
		Name:        t.Name,
		Description: t.Description,
		InputSchema: params,
	}
	if t.Strict != nil && dia.StrictTools {
		wt.Strict = t.Strict
	}
	wt.CacheControl = irCacheToWire(t.Cache)
	return wt, nil
}

var schema_emptyObj = json.RawMessage(`{"type":"object","properties":{}}`)

func buildToolChoice(tc ir.ToolChoice, parallelToolCalls *bool, env *ir.Env) (json.RawMessage, error) {
	disableParallel := parallelToolCalls != nil && !*parallelToolCalls
	switch tc.Mode {
	case ir.ToolChoiceAuto, "":
		return jsonx.Marshal(wireToolChoiceAuto{Type: "auto", DisableParallelToolUse: disableParallel})
	case ir.ToolChoiceRequired:
		return jsonx.Marshal(wireToolChoiceAny{Type: "any", DisableParallelToolUse: disableParallel})
	case ir.ToolChoiceTool:
		return jsonx.Marshal(wireToolChoiceTool{Type: "tool", Name: tc.Name, DisableParallelToolUse: disableParallel})
	case ir.ToolChoiceAllowed:
		if tc.AllowedMode == ir.ToolChoiceRequired {
			return jsonx.Marshal(wireToolChoiceAny{Type: "any", DisableParallelToolUse: disableParallel})
		}
		return jsonx.Marshal(wireToolChoiceAuto{Type: "auto", DisableParallelToolUse: disableParallel})
	case ir.ToolChoiceNone:
		return jsonx.Marshal(map[string]any{"type": "none"})
	}
	return nil, nil
}

func buildThinking(rc *ir.ReasoningConfig, profile *ir.ModelProfile, env *ir.Env) (raw json.RawMessage, enabled bool, budget int, adaptiveEffort ir.Effort, err error) {
	res := profile.Resolve(rc)
	if res.Off {
		if profile != nil {
			switch profile.Off.Mode {
			case ir.OffUnsupported:
				return nil, false, 0, "", env.Err(ir.Unsupported(ir.Root.Key("reasoning"), "off", "model does not allow disabling reasoning"))
			case ir.OffLowestHidden:
				low := profile.Lowest()
				b := profile.EffortBudget(low)
				if b < 1024 {
					b = 1024
				}
				rb, _ := jsonx.Marshal(wireThinkingConfig{Type: "enabled", BudgetTokens: b})
				return rb, true, b, "", nil
			}
		}
		rb, _ := jsonx.Marshal(wireThinkingConfig{Type: "disabled"})
		return rb, false, 0, "", nil
	}
	if !res.Asked && res.Effort == ir.EffortUnset && res.Budget == nil {
		return nil, false, 0, "", nil
	}
	if profile != nil && profile.Style == ir.StyleNone {
		return nil, false, 0, "", nil
	}
	if res.Adaptive || (profile != nil && profile.Style == ir.StyleAdaptive) {
		rb, _ := jsonx.Marshal(wireThinkingConfig{Type: "adaptive"})
		return rb, true, 0, res.Effort, nil
	}
	b := 0
	if res.Budget != nil {
		b = *res.Budget
	} else if res.Effort != ir.EffortUnset {
		b = profile.EffortBudget(res.Effort)
	} else {
		b = ir.DefaultEffortToBudget[ir.EffortMedium]
	}
	minBudget := 1024
	if profile != nil && profile.MinBudget > 0 {
		minBudget = profile.MinBudget
	}
	if b < minBudget {
		b = minBudget
	}
	rb, _ := jsonx.Marshal(wireThinkingConfig{Type: "enabled", BudgetTokens: b})
	return rb, true, b, "", nil
}

func irCacheToWire(cc *ir.CacheControl) *wireCacheControl {
	if cc == nil {
		return nil
	}
	wcc := &wireCacheControl{Type: cc.Type}
	if wcc.Type == "" {
		wcc.Type = "ephemeral"
	}
	if cc.TTL != "" {
		wcc.TTL = cc.TTL
	}
	return wcc
}

// suppress unused import
var _ = strings.Builder{}
