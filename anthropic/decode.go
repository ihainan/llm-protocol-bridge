package anthropic

import (
	"encoding/json"
	"strings"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// StreamDecoder converts Anthropic streaming SSE events into ir.Events.
type StreamDecoder struct {
	env      *ir.Env
	names    map[string]ir.ToolRef
	started  bool
	terminal bool
	// openBlocks maps block index -> type of the open block.
	openBlocks      map[int]string
	blockCallID     map[int]string
	serverToolBlock map[int]*ir.BuiltinActivity
	serverToolArgs  map[int]*strings.Builder
	serverToolBytes int
	maxBuf          int
}

const defaultAnthropicMaxBufferedBytes = 8 << 20

// NewStreamDecoder returns an Anthropic stream decoder.
func NewStreamDecoder(opts DecodeOptions) *StreamDecoder {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	env.Style = ir.StyleDot
	maxBuf := opts.MaxBufferedBytes
	if maxBuf <= 0 {
		maxBuf = defaultAnthropicMaxBufferedBytes
	}
	return &StreamDecoder{
		env:             env,
		names:           opts.Names,
		openBlocks:      map[int]string{},
		blockCallID:     map[int]string{},
		serverToolBlock: map[int]*ir.BuiltinActivity{},
		serverToolArgs:  map[int]*strings.Builder{},
		maxBuf:          maxBuf,
	}
}

// Decode converts one SSE event into zero or more IR events.
func (d *StreamDecoder) Decode(out []ir.Event, ev sse.Event) ([]ir.Event, error) {
	if d.terminal {
		return out, nil
	}
	if len(ev.Data) == 0 {
		return out, nil
	}

	var se wireStreamEvent
	if json.Unmarshal(ev.Data, &se) != nil {
		return out, nil
	}

	switch se.Type {
	case "ping":
		// heartbeat, ignore

	case "message_start":
		if se.Message == nil {
			return out, nil
		}
		d.started = true
		var usage *ir.Usage
		if se.Message.Usage != nil {
			u := decodeUsage(se.Message.Usage)
			usage = &u
		}
		out = append(out, ir.Event{
			Type:  ir.EvStart,
			ID:    se.Message.ID,
			Model: se.Message.Model,
			Usage: usage,
		})

	case "content_block_start":
		if se.ContentBlock == nil {
			return out, nil
		}
		d.openBlocks[se.Index] = se.ContentBlock.Type
		switch se.ContentBlock.Type {
		case "text":
			// No explicit start event; text deltas produce EvText.
		case "thinking":
			out = append(out, ir.Event{Type: ir.EvReasoningStart, Index: 0, ID: ""})
		case "redacted_thinking":
			data, _ := jsonx.Marshal(se.ContentBlock.Data)
			sealed := &ir.Sealed{
				Kind:   ir.SealedAnthropicRedacted,
				Data:   data,
				Origin: ir.Origin{Format: ir.FormatAnthropic, Provider: d.env.ProviderID},
			}
			out = append(out, ir.Event{Type: ir.EvSealed, Index: 0, Sealed: sealed})
		case "tool_use":
			id := ir.AnthropicRawID(se.ContentBlock.ID)
			name := se.ContentBlock.Name
			ns := ""
			kind := ir.ToolFunction
			if ref, ok := d.names[name]; ok {
				name = ref.Name
				ns = ref.Namespace
				if ref.Kind != "" {
					kind = ref.Kind
				}
			}
			d.blockCallID[se.Index] = id
			out = append(out, ir.Event{
				Type:      ir.EvToolStart,
				Index:     0,
				Part:      se.Index,
				ID:        id,
				Name:      name,
				Namespace: ns,
				Kind:      kind,
			})
			if len(se.ContentBlock.Input) > 0 && string(se.ContentBlock.Input) != "{}" && !jsonx.IsNull(se.ContentBlock.Input) {
				out = append(out, ir.Event{
					Type:  ir.EvToolArgs,
					Index: 0,
					Part:  se.Index,
					ID:    id,
					Text:  string(se.ContentBlock.Input),
				})
			}
		case "server_tool_use":
			kind := ir.BuiltinKind(se.ContentBlock.Name)
			if kind == "" {
				kind = ir.BuiltinWebSearch
			}
			d.serverToolBlock[se.Index] = &ir.BuiltinActivity{
				Kind:   kind,
				ID:     se.ContentBlock.ID,
				Input:  append(json.RawMessage(nil), se.ContentBlock.Input...),
				Origin: ir.FormatAnthropic,
			}
			d.serverToolArgs[se.Index] = &strings.Builder{}
		case "web_search_tool_result", "web_fetch_tool_result":
			kind := ir.BuiltinWebSearch
			if se.ContentBlock.Type == "web_fetch_tool_result" {
				kind = ir.BuiltinWebFetch
			}
			b := parseAnthropicBuiltinResult(kind, se.ContentBlock.ToolUseID, se.ContentBlock.Content)
			out = append(out, ir.Event{Type: ir.EvBuiltin, Index: 0, Builtin: b})
		}

	case "content_block_delta":
		if se.Delta == nil {
			return out, nil
		}
		blockType := d.openBlocks[se.Index]
		switch se.Delta.Type {
		case "text_delta":
			if blockType == "text" {
				out = append(out, ir.Event{Type: ir.EvText, Index: 0, Text: se.Delta.Text})
			}
		case "citations_delta":
			if len(se.Delta.Citation) > 0 {
				cit := parseAnthropicCitation(se.Delta.Citation)
				out = append(out, ir.Event{Type: ir.EvCitation, Index: 0, Citation: &cit})
			}
		case "thinking_delta":
			out = append(out, ir.Event{Type: ir.EvReasoning, Index: 0, Text: se.Delta.Thinking})
		case "signature_delta":
			sig := &ir.Signature{
				Value:  se.Delta.Signature,
				Origin: ir.Origin{Format: ir.FormatAnthropic, Provider: d.env.ProviderID},
			}
			out = append(out, ir.Event{Type: ir.EvSignature, Index: 0, Signature: sig})
		case "input_json_delta":
			if blockType == "tool_use" {
				out = append(out, ir.Event{Type: ir.EvToolArgs, Index: 0, Part: se.Index, ID: d.blockCallID[se.Index], Text: se.Delta.PartialJSON})
			} else if blockType == "server_tool_use" {
				if sb := d.serverToolArgs[se.Index]; sb != nil {
					if d.serverToolBytes+len(se.Delta.PartialJSON) > d.maxBuf {
						d.serverToolBlock = map[int]*ir.BuiltinActivity{}
						d.serverToolArgs = map[int]*strings.Builder{}
						d.serverToolBytes = 0
						d.terminal = true
						out = append(out, ir.Event{
							Type: ir.EvError,
							Err:  &ir.APIError{Status: 502, Type: ir.ErrServer, Message: "buffered server_tool_use arguments exceeded max_buffered_bytes"},
						})
						return out, nil
					}
					d.serverToolBytes += len(se.Delta.PartialJSON)
					sb.WriteString(se.Delta.PartialJSON)
				}
			}
		}

	case "content_block_stop":
		blockType := d.openBlocks[se.Index]
		switch blockType {
		case "tool_use":
			out = append(out, ir.Event{Type: ir.EvToolEnd, Index: 0, Part: se.Index, ID: d.blockCallID[se.Index]})
			delete(d.blockCallID, se.Index)
		case "server_tool_use":
			if b := d.serverToolBlock[se.Index]; b != nil {
				if sb := d.serverToolArgs[se.Index]; sb != nil && sb.Len() > 0 {
					d.serverToolBytes -= sb.Len()
					if d.serverToolBytes < 0 {
						d.serverToolBytes = 0
					}
					b.Input = json.RawMessage(sb.String())
				}
				out = append(out, ir.Event{Type: ir.EvBuiltin, Index: 0, Builtin: b})
			}
			delete(d.serverToolBlock, se.Index)
			delete(d.serverToolArgs, se.Index)
		case "redacted_thinking":
			// EvSealed was already emitted at content_block_start where data arrives.
		}
		delete(d.openBlocks, se.Index)

	case "message_delta":
		if se.Delta != nil && se.Delta.StopReason != "" {
			stop := stopFromAnthropic(se.Delta.StopReason, se.Delta.StopSequence)
			out = append(out, ir.Event{Type: ir.EvStop, Index: 0, Stop: &stop})
		}
		if se.Usage != nil {
			u := decodeUsage(se.Usage)
			out = append(out, ir.Event{Type: ir.EvUsage, Usage: &u})
		}

	case "message_stop":
		d.terminal = true

	case "error":
		if se.Error != nil {
			e := &ir.APIError{
				Type:    mapAnthropicErrorType(se.Error.Type),
				Message: se.Error.Message,
				Raw:     append([]byte(nil), ev.Data...),
			}
			out = append(out, ir.Event{Type: ir.EvError, Err: e})
			d.terminal = true
		}
	}

	return out, nil
}

// End finalises the stream.
func (d *StreamDecoder) End(how ir.EndKind) *ir.Event {
	if d.terminal {
		return nil
	}
	return ir.EndErrorEvent(how)
}

// stopFromAnthropic maps Anthropic stop_reason to ir.Stop.
func stopFromAnthropic(reason, seq string) ir.Stop {
	switch reason {
	case "end_turn":
		return ir.Stop{Reason: ir.StopEndTurn, Raw: reason}
	case "max_tokens":
		return ir.Stop{Reason: ir.StopMaxTokens, Raw: reason}
	case "stop_sequence":
		return ir.Stop{Reason: ir.StopSequence, Sequence: seq, Raw: reason}
	case "tool_use":
		return ir.Stop{Reason: ir.StopToolUse, Raw: reason}
	case "pause_turn":
		return ir.Stop{Reason: ir.StopPauseTurn, Raw: reason}
	case "refusal":
		return ir.Stop{Reason: ir.StopRefusal, Raw: reason}
	case "model_context_window_exceeded":
		return ir.Stop{Reason: ir.StopContextWindow, Raw: reason}
	}
	return ir.Stop{Reason: ir.StopEndTurn, Raw: reason}
}

// decodeUsage converts a wireUsage to ir.Usage, marking only fields actually
// present in JSON so partial message_delta updates never zero earlier counters.
func decodeUsage(u *wireUsage) ir.Usage {
	var out ir.Usage
	if u.InputTokens != nil {
		out.InputTokens = *u.InputTokens
		out.Reported |= ir.UInput
	}
	if u.OutputTokens != nil {
		out.OutputTokens = *u.OutputTokens
		out.Reported |= ir.UOutput
	}
	if u.CacheReadInputTokens != nil {
		out.CacheReadTokens = *u.CacheReadInputTokens
		out.Reported |= ir.UCacheRead
	}
	if u.CacheCreationInputTokens != nil {
		out.CacheWriteTokens = *u.CacheCreationInputTokens
		out.Reported |= ir.UCacheWrite
	}
	if u.CacheCreation != nil {
		out.CacheWrite5mTokens = u.CacheCreation.Ephemeral5mInputTokens
		out.CacheWrite1hTokens = u.CacheCreation.Ephemeral1hInputTokens
		if out.CacheWrite5mTokens > 0 || out.CacheWrite1hTokens > 0 {
			out.Reported |= ir.UCacheWriteSplit
			if !out.Has(ir.UCacheWrite) {
				out.CacheWriteTokens = out.CacheWrite5mTokens + out.CacheWrite1hTokens
				out.Reported |= ir.UCacheWrite
			}
		}
	}
	if u.ServerToolUse != nil {
		if u.ServerToolUse.WebSearchRequests > 0 {
			out.WebSearchRequests = u.ServerToolUse.WebSearchRequests
			out.Reported |= ir.UWebSearch
		}
		if u.ServerToolUse.WebFetchRequests > 0 {
			out.WebFetchRequests = u.ServerToolUse.WebFetchRequests
			out.Reported |= ir.UWebFetch
		}
	}
	if u.OutputTokensDetails != nil && u.OutputTokensDetails.ThinkingTokens != nil {
		out.ReasoningTokens = *u.OutputTokensDetails.ThinkingTokens
		out.Reported |= ir.UReasoning
	}
	return out
}

// ParseResponse decodes a non-streaming Anthropic Messages response.
func ParseResponse(body []byte, opts DecodeOptions) (*ir.Response, error) {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	env.Style = ir.StyleDot

	var wr wireResponse
	if json.Unmarshal(body, &wr) != nil {
		if e := ParseError(0, body); e.Message != "" {
			return nil, e
		}
		return nil, ir.Protocol("anthropic: response body is not a valid object")
	}
	if wr.Type == "error" {
		e := ParseError(0, body)
		return nil, e
	}
	if (wr.Type != "" && wr.Type != "message") || (wr.Role == "" && len(wr.Content) == 0 && wr.StopReason == "") {
		return nil, ir.Protocol("anthropic: response missing required fields")
	}

	resp := &ir.Response{ID: wr.ID, Model: wr.Model}

	var parts []ir.Part
	for i, block := range wr.Content {
		fp := ir.Root.Key("content").At(i)
		part, err := responseBlockToPart(block, fp, opts.Names, env)
		if err != nil {
			return nil, err
		}
		if part != nil {
			parts = append(parts, *part)
		}
	}
	stop := stopFromAnthropic(wr.StopReason, wr.StopSequence)
	stop = ir.FinalStop(stop, parts)
	resp.Outputs = []ir.Output{{Index: 0, Parts: parts, Stop: stop}}

	if wr.Usage != nil {
		resp.Usage = decodeUsage(wr.Usage)
	}
	return resp, nil
}

func responseBlockToPart(block wireContentBlock, p ir.Path, names map[string]ir.ToolRef, env *ir.Env) (*ir.Part, error) {
	switch block.Type {
	case "text":
		part := ir.TextPart(block.Text)
		for _, rawCit := range block.Citations {
			part.Citations = append(part.Citations, parseAnthropicCitation(rawCit))
		}
		return &part, nil
	case "server_tool_use":
		kind := ir.BuiltinKind(block.Name)
		if kind == "" {
			kind = ir.BuiltinWebSearch
		}
		part := ir.Part{
			Type: ir.PartBuiltinCall,
			Builtin: &ir.BuiltinActivity{
				Kind:   kind,
				ID:     block.ID,
				Input:  append(json.RawMessage(nil), block.Input...),
				Origin: ir.FormatAnthropic,
			},
		}
		return &part, nil
	case "web_search_tool_result", "web_fetch_tool_result":
		kind := ir.BuiltinWebSearch
		if block.Type == "web_fetch_tool_result" {
			kind = ir.BuiltinWebFetch
		}
		part := ir.Part{
			Type:    ir.PartBuiltinResult,
			Builtin: parseAnthropicBuiltinResult(kind, block.ToolUseID, block.Content),
		}
		return &part, nil
	case "thinking":
		part := ir.Part{
			Type: ir.PartReasoning,
			Reasoning: &ir.Reasoning{
				Text: block.Thinking,
				Signature: &ir.Signature{
					Value:  block.Signature,
					Origin: ir.Origin{Format: ir.FormatAnthropic, Provider: env.ProviderID},
				},
			},
		}
		return &part, nil
	case "redacted_thinking":
		data, _ := jsonx.Marshal(block.Data)
		part := ir.Part{
			Type: ir.PartReasoning,
			Reasoning: &ir.Reasoning{
				Sealed: &ir.Sealed{
					Kind:   ir.SealedAnthropicRedacted,
					Data:   data,
					Origin: ir.Origin{Format: ir.FormatAnthropic, Provider: env.ProviderID},
				},
			},
		}
		return &part, nil
	case "tool_use":
		args := block.Input
		if len(args) == 0 {
			args = json.RawMessage(`{}`)
		}
		name := block.Name
		ns := ""
		kind := ir.ToolFunction
		if ref, ok := names[name]; ok {
			name = ref.Name
			ns = ref.Namespace
			if ref.Kind != "" {
				kind = ref.Kind
			}
		}
		part := ir.Part{
			Type: ir.PartToolCall,
			Call: &ir.ToolCall{
				ID:        ir.AnthropicRawID(block.ID),
				Name:      name,
				Namespace: ns,
				Kind:      kind,
				Arguments: args,
			},
		}
		return &part, nil
	}
	return nil, nil
}

func parseAnthropicCitation(raw json.RawMessage) ir.Citation {
	var c struct {
		Type       string `json:"type"`
		URL        string `json:"url"`
		Title      string `json:"title"`
		CitedText  string `json:"cited_text"`
		StartIndex int    `json:"start_char_index"`
		EndIndex   int    `json:"end_char_index"`
	}
	_ = json.Unmarshal(raw, &c)
	return ir.Citation{
		Type:       c.Type,
		URL:        c.URL,
		Title:      c.Title,
		CitedText:  c.CitedText,
		StartIndex: c.StartIndex,
		EndIndex:   c.EndIndex,
		Raw:        append(json.RawMessage(nil), raw...),
	}
}

func parseAnthropicBuiltinResult(kind ir.BuiltinKind, id string, raw json.RawMessage) *ir.BuiltinActivity {
	b := &ir.BuiltinActivity{
		Kind:     kind,
		ID:       id,
		IsResult: true,
		Origin:   ir.FormatAnthropic,
		Raw:      append(json.RawMessage(nil), raw...),
	}
	var items []struct {
		Type    string `json:"type"`
		URL     string `json:"url"`
		Title   string `json:"title"`
		PageAge string `json:"page_age"`
	}
	if json.Unmarshal(raw, &items) == nil && len(items) > 0 {
		for _, it := range items {
			if it.URL != "" {
				b.Hits = append(b.Hits, ir.Hit{URL: it.URL, Title: it.Title, PageAge: it.PageAge})
			}
		}
	} else if s, ok := jsonx.String(raw); ok {
		b.Content = s
	}
	return b
}
