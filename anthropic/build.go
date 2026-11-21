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

	// Tool choice.
	var tcRaw json.RawMessage
	if req.ToolChoice.Mode != "" {
		tcRaw, err = buildToolChoice(req.ToolChoice, env)
		if err != nil {
			return nil, err
		}
	}

	// Thinking config.
	var thinkingRaw json.RawMessage
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

	if req.Temperature != nil {
		tb, _ := jsonx.Marshal(*req.Temperature)
		out.Raw("temperature", tb)
	}
	if req.TopP != nil {
		tb, _ := jsonx.Marshal(*req.TopP)
		out.Raw("top_p", tb)
	}
	if req.TopK != nil {
		out.Int("top_k", int64(*req.TopK))
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

// buildMessages converts ir.Messages to wireMessages.
// Tool results (PartToolResult) in a user message become their own tool_result blocks.
func buildMessages(msgs []ir.Message, names map[string]ir.ToolRef, env *ir.Env, dia Dialect) ([]wireMessage, error) {
	out := make([]wireMessage, 0, len(msgs))
	for i, m := range msgs {
		fp := ir.Root.Key("messages").At(i)
		wms, err := buildMessage(m, fp, names, env, dia)
		if err != nil {
			return nil, err
		}
		out = append(out, wms...)
	}
	return out, nil
}

func buildMessage(m ir.Message, p ir.Path, names map[string]ir.ToolRef, env *ir.Env, dia Dialect) ([]wireMessage, error) {
	switch m.Role {
	case ir.RoleUser:
		return buildUserMessage(m, p, env)
	case ir.RoleAssistant:
		return buildAssistantMessage(m, p, env)
	case ir.RoleSystem, ir.RoleDeveloper:
		// Mid-conversation system messages.
		switch dia.MidSystem {
		case "fold":
			// Skip: merged by caller into nearest assistant turn (not yet implemented).
			return nil, nil
		default: // "keep"
			blocks := buildContentBlocks(m.Parts, env)
			return []wireMessage{{Role: "user", Content: blocks}}, nil
		}
	}
	return nil, env.Err(ir.Invalid(p.Key("role"), "unsupported role: "+string(m.Role)))
}

func buildUserMessage(m ir.Message, p ir.Path, env *ir.Env) ([]wireMessage, error) {
	var blocks []wireContentBlock
	for i, part := range m.Parts {
		fp := p.Key("content").At(i)
		bs, err := partToBlocks(part, fp, env)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, bs...)
	}
	content, err := jsonx.Marshal(blocks)
	if err != nil {
		return nil, err
	}
	return []wireMessage{{Role: "user", Content: content}}, nil
}

func buildAssistantMessage(m ir.Message, p ir.Path, env *ir.Env) ([]wireMessage, error) {
	var blocks []wireContentBlock
	for i, part := range m.Parts {
		fp := p.Key("content").At(i)
		bs, err := partToBlocks(part, fp, env)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, bs...)
	}
	content, err := jsonx.Marshal(blocks)
	if err != nil {
		return nil, err
	}
	return []wireMessage{{Role: "assistant", Content: content}}, nil
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
			ID:    call.ID,
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
			ToolUseID: r.CallID,
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

func buildToolChoice(tc ir.ToolChoice, env *ir.Env) (json.RawMessage, error) {
	switch tc.Mode {
	case ir.ToolChoiceAuto:
		return jsonx.Marshal(wireToolChoiceAuto{Type: "auto"})
	case ir.ToolChoiceRequired:
		return jsonx.Marshal(wireToolChoiceAny{Type: "any"})
	case ir.ToolChoiceTool:
		return jsonx.Marshal(wireToolChoiceTool{Type: "tool", Name: tc.Name})
	case ir.ToolChoiceNone:
		// Anthropic has no "none" mode; closest is auto (model may still use tools).
		return jsonx.Marshal(wireToolChoiceAuto{Type: "auto"})
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
