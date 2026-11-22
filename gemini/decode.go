package gemini

import (
	"encoding/json"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// StreamDecoder converts Gemini streamGenerateContent chunks into ir.Events.
type StreamDecoder struct {
	env          *ir.Env
	dia          Dialect
	names        map[string]ir.ToolRef
	started      bool
	terminal     bool
	hasToolCalls map[int]bool
	callSeqBy    map[int]int
	inThought    map[int]bool
}

// NewStreamDecoder returns a Gemini stream decoder.
func NewStreamDecoder(opts DecodeOptions) *StreamDecoder {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	env.Style = ir.StyleBracket
	return &StreamDecoder{
		env:          env,
		dia:          opts.Dialect,
		names:        opts.Names,
		hasToolCalls: map[int]bool{},
		callSeqBy:    map[int]int{},
		inThought:    map[int]bool{},
	}
}

// Decode converts one SSE event (or bare JSON line) into zero or more IR events.
func (d *StreamDecoder) Decode(out []ir.Event, ev sse.Event) ([]ir.Event, error) {
	if sse.IsDone(ev.Data) {
		return out, nil
	}
	data := ev.Data
	if len(data) == 0 {
		return out, nil
	}

	// Gemini streaming may arrive as bare JSON objects (no event: field).
	if jsonx.Kind(data) != '{' {
		return out, nil
	}

	// Try error envelope first.
	if isGeminiError(data) {
		e := ParseError(0, data)
		out = append(out, ir.Event{Type: ir.EvError, Err: e})
		d.terminal = true
		return out, nil
	}

	var resp wireResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return out, nil
	}

	if !d.started {
		d.started = true
		model := resp.ModelVersion
		ev := ir.Event{Type: ir.EvStart, Model: model}
		if resp.UsageMetadata != nil {
			u := decodeGeminiUsage(resp.UsageMetadata)
			ev.Usage = &u
		}
		out = append(out, ev)
	}

	if len(resp.Candidates) == 0 && resp.PromptFeedback != nil && resp.PromptFeedback.BlockReason != "" {
		msg := resp.PromptFeedback.BlockReasonMessage
		if msg == "" {
			msg = "prompt blocked: " + resp.PromptFeedback.BlockReason
		}
		out = append(out, ir.Event{Type: ir.EvError, Err: &ir.APIError{
			Status:  400,
			Type:    ir.ErrInvalidRequest,
			Code:    resp.PromptFeedback.BlockReason,
			Message: msg,
			Raw:     append([]byte(nil), data...),
		}})
		d.terminal = true
		return out, nil
	}

	for _, cand := range resp.Candidates {
		idx := cand.Index
		for _, part := range cand.Content.Parts {
			out = d.decodePart(out, part, idx)
		}
		if cand.FinishReason != "" && cand.FinishReason != "FINISH_REASON_UNSPECIFIED" {
			stop := geminiStop(cand.FinishReason)
			if d.hasToolCalls[idx] && stop.Reason == ir.StopEndTurn {
				stop.Reason = ir.StopToolUse
			}
			out = append(out, ir.Event{Type: ir.EvStop, Index: idx, Stop: &stop})
			d.terminal = true
		}
	}

	if resp.UsageMetadata != nil {
		u := decodeGeminiUsage(resp.UsageMetadata)
		out = append(out, ir.Event{Type: ir.EvUsage, Usage: &u})
	}

	return out, nil
}

// End signals the end of the stream; returns a terminal event when the stream
// ended without a finish reason.
func (d *StreamDecoder) End(how ir.EndKind) *ir.Event {
	if d.terminal {
		return nil
	}
	return ir.EndErrorEvent(how)
}

func (d *StreamDecoder) decodePart(out []ir.Event, part wirePart, idx int) []ir.Event {
	switch {
	case part.FunctionCall != nil:
		d.inThought[idx] = false
		fc := part.FunctionCall
		d.hasToolCalls[idx] = true
		callIdx := d.callSeqBy[idx]
		d.callSeqBy[idx] = callIdx + 1
		args := fc.Args
		if len(args) == 0 {
			args = json.RawMessage(`{}`)
		}
		id := fc.ID
		if id == "" {
			id = ir.DeterministicCallID(idx, callIdx, fc.Name, args)
		}
		name := fc.Name
		ns := ""
		kind := ir.ToolFunction
		if ref, ok := d.names[name]; ok {
			name = ref.Name
			ns = ref.Namespace
			if ref.Kind != "" {
				kind = ref.Kind
			}
		}
		var sig *ir.Signature
		if part.ThoughtSignature != "" {
			sig = &ir.Signature{Value: part.ThoughtSignature, Origin: ir.Origin{Format: ir.FormatGemini, Provider: d.env.ProviderID}}
		}
		out = append(out,
			ir.Event{Type: ir.EvToolStart, Index: idx, Part: callIdx, ID: id, Name: name, Namespace: ns, Kind: kind, Signature: sig},
			ir.Event{Type: ir.EvToolArgs, Index: idx, Part: callIdx, ID: id, Text: string(args)},
			ir.Event{Type: ir.EvToolEnd, Index: idx, Part: callIdx, ID: id, Arguments: args},
		)

	case part.FunctionResponse != nil:
		// Function responses come from user turns; they don't appear in model output.

	case part.ExecutableCode != nil:
		d.inThought[idx] = false
		inMap := map[string]any{"code": part.ExecutableCode.Code}
		if part.ExecutableCode.Language != "" {
			inMap["language"] = part.ExecutableCode.Language
		}
		inRaw, _ := jsonx.Marshal(inMap)
		out = append(out, ir.Event{
			Type:  ir.EvBuiltin,
			Index: idx,
			Builtin: &ir.BuiltinActivity{
				Kind:   ir.BuiltinCodeExecution,
				Input:  inRaw,
				Origin: ir.FormatGemini,
			},
		})

	case part.CodeExecutionResult != nil:
		// Code execution results show as text.
		out = append(out, ir.Event{Type: ir.EvText, Index: idx, Text: part.CodeExecutionResult.Output})

	case part.Thought:
		if !d.inThought[idx] {
			d.inThought[idx] = true
			out = append(out, ir.Event{Type: ir.EvReasoningStart, Index: idx})
		}
		if part.Text != "" {
			out = append(out, ir.Event{Type: ir.EvReasoning, Index: idx, Text: part.Text})
		}
		if part.ThoughtSignature != "" {
			sig := &ir.Signature{Value: part.ThoughtSignature, Origin: ir.Origin{Format: ir.FormatGemini, Provider: d.env.ProviderID}}
			out = append(out, ir.Event{Type: ir.EvSignature, Index: idx, Signature: sig})
		}

	case part.Text != "":
		d.inThought[idx] = false
		out = append(out, ir.Event{Type: ir.EvText, Index: idx, Text: part.Text})
		if part.ThoughtSignature != "" {
			sig := &ir.Signature{Value: part.ThoughtSignature, Origin: ir.Origin{Format: ir.FormatGemini, Provider: d.env.ProviderID}}
			out = append(out, ir.Event{Type: ir.EvSignature, Index: idx, Signature: sig})
		}
	}
	return out
}

// ParseResponse decodes a non-streaming Gemini response body into ir.Response.
func ParseResponse(body []byte, opts DecodeOptions) (*ir.Response, error) {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	env.Style = ir.StyleBracket

	if isGeminiError(body) {
		return nil, ParseError(0, body)
	}

	var wr wireResponse
	if err := json.Unmarshal(body, &wr); err != nil {
		return nil, ir.Protocol("failed to decode Gemini response: " + err.Error())
	}

	resp := &ir.Response{
		Model: wr.ModelVersion,
	}
	if wr.UsageMetadata != nil {
		resp.Usage = decodeGeminiUsage(wr.UsageMetadata)
	}

	for _, cand := range wr.Candidates {
		out := ir.Output{Index: cand.Index}
		callIdx := 0
		for i, part := range cand.Content.Parts {
			fp := ir.Root.Key("candidates").At(cand.Index).Key("content").Key("parts").At(i)
			p, err := wirePartToIR(part, fp, cand.Index, &callIdx, opts.Names, env)
			if err != nil {
				return nil, err
			}
			if p != nil {
				out.Parts = append(out.Parts, *p)
			}
		}
		out.Stop = ir.FinalStop(geminiStop(cand.FinishReason), out.Parts)
		resp.Outputs = append(resp.Outputs, out)
	}
	if len(resp.Outputs) == 0 {
		if wr.PromptFeedback != nil && wr.PromptFeedback.BlockReason != "" {
			msg := wr.PromptFeedback.BlockReasonMessage
			if msg == "" {
				msg = "prompt blocked: " + wr.PromptFeedback.BlockReason
			}
			return nil, &ir.APIError{
				Status:  400,
				Type:    ir.ErrInvalidRequest,
				Code:    wr.PromptFeedback.BlockReason,
				Message: msg,
				Raw:     append([]byte(nil), body...),
			}
		}
		return nil, ir.Protocol("gemini: response has no candidates")
	}
	return resp, nil
}

func wirePartToIR(part wirePart, p ir.Path, candIdx int, callIdx *int, names map[string]ir.ToolRef, env *ir.Env) (*ir.Part, error) {
	switch {
	case part.FunctionCall != nil:
		fc := part.FunctionCall
		args := fc.Args
		if len(args) == 0 {
			args = json.RawMessage(`{}`)
		}
		id := fc.ID
		ci := 0
		if callIdx != nil {
			ci = *callIdx
			*callIdx = ci + 1
		}
		if id == "" {
			id = ir.DeterministicCallID(candIdx, ci, fc.Name, args)
		}
		name := fc.Name
		ns := ""
		kind := ir.ToolFunction
		if ref, ok := names[fc.Name]; ok {
			name = ref.Name
			ns = ref.Namespace
			if ref.Kind != "" {
				kind = ref.Kind
			}
		}
		call := &ir.ToolCall{
			ID:        id,
			Name:      name,
			Namespace: ns,
			Kind:      kind,
			Arguments: args,
		}
		if part.ThoughtSignature != "" {
			call.Signature = &ir.Signature{Value: part.ThoughtSignature, Origin: ir.Origin{Format: ir.FormatGemini, Provider: env.ProviderID}}
		}
		return &ir.Part{Type: ir.PartToolCall, Call: call}, nil

	case part.ExecutableCode != nil:
		inMap := map[string]any{"code": part.ExecutableCode.Code}
		if part.ExecutableCode.Language != "" {
			inMap["language"] = part.ExecutableCode.Language
		}
		inRaw, _ := jsonx.Marshal(inMap)
		return &ir.Part{
			Type: ir.PartBuiltinCall,
			Builtin: &ir.BuiltinActivity{
				Kind:   ir.BuiltinCodeExecution,
				Input:  inRaw,
				Origin: ir.FormatGemini,
			},
		}, nil

	case part.CodeExecutionResult != nil:
		b := &ir.BuiltinActivity{
			Kind:     ir.BuiltinCodeExecution,
			IsResult: true,
			Content:  part.CodeExecutionResult.Output,
			Origin:   ir.FormatGemini,
		}
		if part.CodeExecutionResult.Outcome != "" && part.CodeExecutionResult.Outcome != "OUTCOME_OK" {
			b.Error = part.CodeExecutionResult.Outcome
		}
		return &ir.Part{Type: ir.PartBuiltinResult, Builtin: b}, nil

	case part.Thought:
		r := &ir.Reasoning{Text: part.Text}
		if part.ThoughtSignature != "" {
			r.Signature = &ir.Signature{Value: part.ThoughtSignature, Origin: ir.Origin{Format: ir.FormatGemini, Provider: env.ProviderID}}
		}
		return &ir.Part{
			Type:      ir.PartReasoning,
			Reasoning: r,
		}, nil

	case part.Text != "":
		tp := ir.TextPart(part.Text)
		if part.ThoughtSignature != "" {
			tp.Signature = &ir.Signature{Value: part.ThoughtSignature, Origin: ir.Origin{Format: ir.FormatGemini, Provider: env.ProviderID}}
		}
		return &tp, nil

	case part.InlineData != nil:
		return &ir.Part{Type: ir.PartImage, Media: &ir.Media{
			MediaType: part.InlineData.MimeType,
			Data:      part.InlineData.Data,
		}}, nil

	case part.FileData != nil:
		return &ir.Part{Type: ir.PartFile, Media: &ir.Media{
			MediaType: part.FileData.MimeType,
			URL:       part.FileData.FileURI,
		}}, nil
	}
	return nil, nil
}

func geminiStop(reason string) ir.Stop {
	switch reason {
	case "STOP":
		return ir.Stop{Reason: ir.StopEndTurn, Raw: reason}
	case "MAX_TOKENS":
		return ir.Stop{Reason: ir.StopMaxTokens, Raw: reason}
	case "SAFETY", "RECITATION", "PROHIBITED_CONTENT", "SPII", "BLOCKLIST", "IMAGE_SAFETY":
		return ir.Stop{Reason: ir.StopContentFilter, Raw: reason}
	case "TOOL_CALLS", "FUNCTION_CALL":
		return ir.Stop{Reason: ir.StopToolUse, Raw: reason}
	case "OTHER", "MALFORMED_FUNCTION_CALL":
		return ir.Stop{Reason: ir.StopOther, Raw: reason}
	}
	return ir.Stop{Reason: ir.StopEndTurn, Raw: reason}
}

func decodeGeminiUsage(u *wireUsage) ir.Usage {
	// Per design.md section 6:
	// Input = promptTokenCount - cachedContentTokenCount + toolUsePromptTokenCount
	// Output = candidatesTokenCount + thoughtsTokenCount
	out := ir.Usage{
		OutputTokens:        u.CandidatesTokenCount + u.ThoughtsTokenCount,
		ReasoningTokens:     u.ThoughtsTokenCount,
		ToolUsePromptTokens: u.ToolUsePromptTokenCount,
		CacheReadTokens:     u.CachedContentTokenCount,
		Reported:            ir.UOutput,
	}
	if u.CachedContentTokenCount > 0 {
		out.Reported |= ir.UCacheRead
	}
	out.FromInclusivePrompt(u.PromptTokenCount)
	out.InputTokens += u.ToolUsePromptTokenCount
	if u.ThoughtsTokenCount > 0 {
		out.Reported |= ir.UReasoning
	}
	if u.ToolUsePromptTokenCount > 0 {
		out.Reported |= ir.UToolUsePrompt
	}
	return out
}

func isGeminiError(data []byte) bool {
	kvs, err := jsonx.Object(data)
	if err != nil {
		return false
	}
	for _, kv := range kvs {
		if kv.Key == "error" {
			return true
		}
	}
	return false
}
