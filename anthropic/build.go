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
	maxTok := 0
	if req.MaxOutputTokens != nil {
		maxTok = *req.MaxOutputTokens
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
	messages, err := buildMessages(req.Messages, names, env, dia)
	if err != nil {
		return nil, err
	}

	// Tools.
	var wireTools []wireTool
	for i, t := range req.Tools {
		fp := ir.Root.Key("tools").At(i)
		wt, err := buildTool(t, fp, dia, env)
		if err != nil {
			return nil, err
		}
		names[t.Name] = ir.ToolRef{Name: t.Name, Namespace: t.Namespace, Kind: t.Kind}
		wireTools = append(wireTools, wt)
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

	// Thinking config.
	if req.Reasoning != nil {
		thinkingRaw, err = buildThinking(req.Reasoning, env)
		if err != nil {
			return nil, err
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

	if len(wireTools) > 0 {
		tb, err := jsonx.Marshal(wireTools)
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

	// Sampling parameters.
	if req.Temperature != nil {
		t := *req.Temperature
		if t > 1.0 {
			if err := env.Lose(ir.LossSamplingClamp, ir.Root.Key("temperature"), "clamped to 1.0 (Anthropic max)"); err != nil {
				return nil, err
			}
			t = 1.0
		}
		tb, _ := jsonx.Marshal(t)
		out.Raw("temperature", tb)
	}
	if req.TopP != nil {
		tb, _ := jsonx.Marshal(*req.TopP)
		out.Raw("top_p", tb)
	}
	if req.TopK != nil {
		out.Int("top_k", int64(*req.TopK))
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
		src, err := buildImageSource(part.Media, p, env)
		if err != nil {
			return nil, err
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
		if r.Sealed != nil && r.Sealed.Kind == ir.SealedAnthropicRedacted {
			// Re-send the redacted_thinking block as received.
			var dataStr string
			_ = json.Unmarshal(r.Sealed.Data, &dataStr)
			return []wireContentBlock{{Type: "redacted_thinking", Data: dataStr}}, nil
		}
		if r.Text != "" {
			sig := ""
			if r.Signature != nil {
				sig = r.Signature.Value
			}
			return []wireContentBlock{{Type: "thinking", Thinking: r.Text, Signature: sig}}, nil
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
	}
	return nil, nil
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
	params := t.Schema
	if jsonx.Absent(params) {
		params = schema_emptyObj
	}
	wt := wireTool{
		Name:        t.Name,
		Description: t.Description,
		InputSchema: params,
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
	case ir.ToolChoiceNone:
		return jsonx.Marshal(map[string]any{"type": "none"})
	}
	return nil, nil
}

func buildThinking(rc *ir.ReasoningConfig, env *ir.Env) (json.RawMessage, error) {
	if rc.Mode == ir.ReasoningOff {
		return jsonx.Marshal(wireThinkingConfig{Type: "disabled"})
	}
	t := wireThinkingConfig{Type: "enabled"}
	if rc.Budget != nil {
		t.BudgetTokens = *rc.Budget
	}
	return jsonx.Marshal(t)
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
