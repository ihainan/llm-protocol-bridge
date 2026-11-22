package anthropic

import (
	"encoding/json"
	"strings"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

type toolCallKey struct {
	cand int
	id   string
}

const defaultMaxBufferedBytes = 8 * 1024 * 1024

type pendingTool struct {
	cand         int
	id           string
	name         string
	sig          *ir.Signature
	kind         ir.ToolKind
	args         strings.Builder
	finalArgs    json.RawMessage
	countedBytes int
	argCounted   int
	done         bool
}

// StreamEncoder converts ir.Events into Anthropic Messages SSE chunks.
type StreamEncoder struct {
	opts    EncodeOptions
	id      string
	model   string
	started bool
	errored bool
	// blockIdx is the next content block index to assign.
	blockIdx int
	// openBlock is the index of the currently open content block (-1 = none).
	openBlock int
	// openBlockType is the type string of the open block.
	openBlockType     string
	openBlockCand     int
	thinkingTextAccum strings.Builder
	thinkingSigVal    strings.Builder
	thinkingSigOrigin ir.Origin
	activeCallCand    int
	activeCallID      string
	// activeKind is the kind of the open tool_use block's call. A custom tool's
	// input arrives as raw text and is wrapped into {"input": "..."} here;
	// activeCustomOpened records that the wrapper prefix was already written and
	// activeCustomSent tracks the raw text already streamed for prefix validation.
	activeKind         ir.ToolKind
	activeCustomOpened bool
	activeCustomSent   strings.Builder
	activeFuncSent     strings.Builder
	activeToolFinished bool
	pendingTools       []*pendingTool
	pendingByKey       map[toolCallKey]*pendingTool
	pendingBytes       int
	deferredOtherCand  []ir.Event
	deferredBytes      int
	stop               *ir.Stop
	usage              *ir.Usage
}

// NewStreamEncoder returns an Anthropic stream encoder.
func NewStreamEncoder(opts EncodeOptions) *StreamEncoder {
	return &StreamEncoder{
		opts:         opts,
		openBlock:    -1,
		pendingByKey: map[toolCallKey]*pendingTool{},
	}
}

func (e *StreamEncoder) ensureStarted(dst []byte) []byte {
	if e.started {
		return dst
	}
	e.started = true
	if e.id == "" {
		e.id = "msg_" + e.opts.Env.NewID("", 24)
	}
	if e.model == "" {
		e.model = e.opts.Model
	}
	dst = e.emitMessageStart(dst, ir.Event{Usage: e.usage})
	dst = e.emitPing(dst)
	return dst
}

// Encode serialises one ir.Event to Anthropic SSE bytes appended to dst.
func (e *StreamEncoder) Encode(dst []byte, ev ir.Event) ([]byte, error) {
	if e.errored {
		return dst, nil
	}
	if e.openBlock >= 0 && e.openBlockType == "tool_use" && ev.Index != e.activeCallCand {
		switch ev.Type {
		case ir.EvText, ir.EvRefusal, ir.EvCitation, ir.EvReasoningStart, ir.EvReasoning, ir.EvSignature, ir.EvSealed, ir.EvBuiltin:
			sz := anthropicEventByteSize(ev)
			if sz == 0 {
				sz = 1
			}
			e.deferredBytes += sz
			if e.deferredBytes+e.pendingBytes > e.maxBufferedBytes() {
				return e.abortOpenTool(dst, &ir.APIError{
					Status:  502,
					Type:    ir.ErrServer,
					Message: "buffered stream byte limit exceeded",
				}), nil
			}
			e.deferredOtherCand = append(e.deferredOtherCand, ev)
			return dst, nil
		}
	}
	switch ev.Type {
	case ir.EvStart:
		if e.opts.GenerateResponseID {
			e.id = "msg_" + e.opts.Env.NewID("", 24)
		} else {
			e.id = ev.ID
			if e.id == "" {
				e.id = "msg_" + e.opts.Env.NewID("", 24)
			}
		}
		e.model = e.opts.Model
		if e.model == "" && ev.Model != "" {
			e.model = ev.Model
		}
		if ev.Usage != nil {
			if e.usage == nil {
				u := *ev.Usage
				e.usage = &u
			} else {
				e.usage.Merge(*ev.Usage)
			}
		}
		if !e.started {
			e.started = true
			dst = e.emitMessageStart(dst, ev)
			dst = e.emitPing(dst)
		}

	case ir.EvText, ir.EvRefusal:
		dst = e.ensureStarted(dst)
		dst = e.ensureTextBlock(dst, ev.Index)
		if e.errored {
			break
		}
		dst = e.appendSSE(dst, "content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": e.openBlock,
			"delta": map[string]any{"type": "text_delta", "text": ev.Text},
		})

	case ir.EvCitation:
		if ev.Citation != nil {
			if e.openBlock >= 0 && e.openBlockType == "text" {
				var citObj any
				if len(ev.Citation.Raw) > 0 {
					citObj = json.RawMessage(ev.Citation.Raw)
				} else {
					citObj = map[string]any{
						"type":       "web_search_result_location",
						"url":        ev.Citation.URL,
						"title":      ev.Citation.Title,
						"cited_text": ev.Citation.CitedText,
					}
				}
				dst = e.appendSSE(dst, "content_block_delta", map[string]any{
					"type":  "content_block_delta",
					"index": e.openBlock,
					"delta": map[string]any{"type": "citations_delta", "citation": citObj},
				})
			} else if err := e.opts.Env.Lose(ir.LossCitations, ir.Root.Key("content"), "citation without open text block in Anthropic stream"); err != nil {
				return dst, err
			}
		}

	case ir.EvImage:
		if ev.Media != nil {
			if err := e.opts.Env.Lose(ir.LossOutputMedia, ir.Root.Key("content"), "output media not supported in Anthropic assistant stream"); err != nil {
				return dst, err
			}
		}

	case ir.EvReasoningStart:
		dst = e.ensureStarted(dst)
		dst = e.closeBlock(dst)
		if e.errored {
			break
		}
		idx := e.blockIdx
		e.blockIdx++
		e.openBlock = idx
		e.openBlockType = "thinking"
		e.openBlockCand = ev.Index
		dst = e.appendSSE(dst, "content_block_start", map[string]any{
			"type":          "content_block_start",
			"index":         idx,
			"content_block": map[string]any{"type": "thinking", "thinking": ""},
		})

	case ir.EvReasoning:
		dst = e.ensureStarted(dst)
		if e.openBlock < 0 || e.openBlockType != "thinking" || e.openBlockCand != ev.Index {
			dst = e.closeBlock(dst)
			if e.errored {
				break
			}
			idx := e.blockIdx
			e.blockIdx++
			e.openBlock = idx
			e.openBlockType = "thinking"
			e.openBlockCand = ev.Index
			dst = e.appendSSE(dst, "content_block_start", map[string]any{
				"type":          "content_block_start",
				"index":         idx,
				"content_block": map[string]any{"type": "thinking", "thinking": ""},
			})
		}
		e.thinkingTextAccum.WriteString(ev.Text)
		dst = e.appendSSE(dst, "content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": e.openBlock,
			"delta": map[string]any{"type": "thinking_delta", "thinking": ev.Text},
		})

	case ir.EvSignature:
		if ev.Signature != nil && e.openBlock >= 0 && e.openBlockType == "thinking" {
			e.thinkingSigVal.WriteString(ev.Signature.Value)
			if e.thinkingSigOrigin.Format == "" {
				e.thinkingSigOrigin = ev.Signature.Origin
			}
		}

	case ir.EvSealed:
		if ev.Sealed != nil {
			dst = e.ensureStarted(dst)
			dst = e.closeBlock(dst)
			if e.errored {
				break
			}
			idx := e.blockIdx
			e.blockIdx++
			var dataStr string
			if ev.Sealed.Kind == ir.SealedAnthropicRedacted && !needsAnthropicSeal(ev.Sealed.Origin, e.opts.Env) {
				_ = json.Unmarshal(ev.Sealed.Data, &dataStr)
			} else {
				sk := ir.SealEncrypted
				if ev.Sealed.Kind == ir.SealedAnthropicRedacted {
					sk = ir.SealRedacted
				}
				dataStr = e.opts.Env.Sealer().Seal(ir.SealRecord{
					Origin: ev.Sealed.Origin,
					Kind:   sk,
					Value:  string(ev.Sealed.Data),
				})
			}
			dst = e.appendSSE(dst, "content_block_start", map[string]any{
				"type":  "content_block_start",
				"index": idx,
				"content_block": map[string]any{
					"type": "redacted_thinking",
					"data": dataStr,
				},
			})
			dst = e.appendSSE(dst, "content_block_stop", map[string]any{
				"type":  "content_block_stop",
				"index": idx,
			})
		}

	case ir.EvToolStart:
		dst = e.ensureStarted(dst)
		if e.openBlock >= 0 && e.openBlockType == "tool_use" {
			// Another tool call is already streaming; queue this one until the active block closes.
			hdrBytes := len(ev.ID) + len(ev.Name)
			if ev.Signature != nil {
				hdrBytes += len(ev.Signature.Value)
			}
			if hdrBytes == 0 {
				hdrBytes = 1
			}
			e.pendingBytes += hdrBytes
			if e.deferredBytes+e.pendingBytes > e.maxBufferedBytes() {
				return e.abortOpenTool(dst, &ir.APIError{
					Status:  502,
					Type:    ir.ErrServer,
					Message: "buffered stream byte limit exceeded",
				}), nil
			}
			pt := &pendingTool{cand: ev.Index, id: ev.ID, name: ev.Name, sig: ev.Signature, kind: ev.Kind, countedBytes: hdrBytes}
			e.pendingTools = append(e.pendingTools, pt)
			if ev.ID != "" {
				e.pendingByKey[toolCallKey{cand: ev.Index, id: ev.ID}] = pt
			}
			break
		}
		dst = e.openToolBlock(dst, ev.Index, ev.ID, ev.Name, ev.Signature, ev.Kind)

	case ir.EvToolArgs:
		isActive := e.openBlock >= 0 && e.openBlockType == "tool_use" &&
			ev.Index == e.activeCallCand &&
			(ev.ID == "" || ev.ID == e.activeCallID)
		if !isActive {
			if pt := e.lookupPendingTool(ev.Index, ev.ID); pt != nil {
				e.pendingBytes += len(ev.Text)
				pt.countedBytes += len(ev.Text)
				pt.argCounted += len(ev.Text)
				if e.deferredBytes+e.pendingBytes > e.maxBufferedBytes() {
					return e.abortOpenTool(dst, &ir.APIError{
						Status:  502,
						Type:    ir.ErrServer,
						Message: "buffered stream byte limit exceeded",
					}), nil
				}
				pt.args.WriteString(ev.Text)
			}
			break
		}
		if ev.Text != "" {
			partial := ev.Text
			if e.activeKind == ir.ToolCustom {
				partial = customInputDelta(ev.Text, !e.activeCustomOpened)
				e.activeCustomOpened = true
				e.activeCustomSent.WriteString(ev.Text)
			} else {
				e.activeFuncSent.WriteString(ev.Text)
			}
			dst = e.appendSSE(dst, "content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": e.openBlock,
				"delta": map[string]any{"type": "input_json_delta", "partial_json": partial},
			})
		}

	case ir.EvToolEnd:
		isActive := e.openBlock >= 0 && e.openBlockType == "tool_use" &&
			ev.Index == e.activeCallCand &&
			(ev.ID == "" || ev.ID == e.activeCallID)
		if ev.Err != nil {
			if !isActive {
				if pt := e.lookupPendingTool(ev.Index, ev.ID); pt != nil {
					e.pendingBytes -= pt.countedBytes
					if e.pendingBytes < 0 {
						e.pendingBytes = 0
					}
					if pt.id != "" {
						delete(e.pendingByKey, toolCallKey{cand: pt.cand, id: pt.id})
					}
					for i, item := range e.pendingTools {
						if item == pt {
							e.pendingTools = append(e.pendingTools[:i], e.pendingTools[i+1:]...)
							break
						}
					}
				}
				break
			}
			// An already-opened tool_use block cannot be retracted in Anthropic SSE:
			// abort the stream with an error event instead of emitting content_block_stop.
			dst = e.abortOpenTool(dst, ev.Err)
			break
		}
		if !isActive {
			if pt := e.lookupPendingTool(ev.Index, ev.ID); pt != nil {
				if extra := len(ev.Arguments) - pt.argCounted; extra > 0 {
					e.pendingBytes += extra
					pt.countedBytes += extra
					pt.argCounted += extra
					if e.deferredBytes+e.pendingBytes > e.maxBufferedBytes() {
						return e.abortOpenTool(dst, &ir.APIError{
							Status:  502,
							Type:    ir.ErrServer,
							Message: "buffered stream byte limit exceeded",
						}), nil
					}
				}
				pt.finalArgs = ev.Arguments
				pt.done = true
			}
			break
		}
		if e.activeKind == ir.ToolCustom {
			var ok bool
			dst, ok = e.finishCustomInput(dst, ev.Arguments)
			if !ok {
				break
			}
		} else {
			var ok bool
			dst, ok = e.finishFunctionArgs(dst, ev.Arguments)
			if !ok {
				break
			}
		}
		e.activeToolFinished = true
		dst = e.closeBlock(dst)
		dst = e.drainPendingTools(dst)
		if !e.errored && e.openBlock < 0 && len(e.pendingTools) == 0 && len(e.deferredOtherCand) > 0 {
			var err error
			dst, err = e.replayDeferredOtherCand(dst)
			if err != nil {
				return dst, err
			}
		}

	case ir.EvBuiltin:
		if ev.Builtin == nil {
			break
		}
		if ev.Builtin.Kind == ir.BuiltinWebSearch || ev.Builtin.Kind == ir.BuiltinWebFetch {
			dst = e.ensureStarted(dst)
			dst = e.closeBlock(dst)
			if e.errored {
				break
			}
			idx := e.blockIdx
			e.blockIdx++
			id := ev.Builtin.ID
			if id == "" {
				id = "srvtoolu_" + e.opts.Env.NewID("", 16)
			}
			var cb map[string]any
			if !ev.Builtin.IsResult {
				if ev.Builtin.Origin != ir.FormatAnthropic && len(ev.Builtin.Raw) > 0 {
					if err := e.opts.Env.Lose(ir.LossBuiltinActivity, ir.Root.Key("content"), "cross-format builtin call metadata cannot be fully represented in Anthropic server_tool_use"); err != nil {
						return dst, err
					}
				}
				var inObj any = map[string]any{}
				if len(ev.Builtin.Input) > 0 {
					inObj = json.RawMessage(ev.Builtin.Input)
				} else if ev.Builtin.Query != "" {
					inObj = map[string]any{"query": ev.Builtin.Query}
				} else if ev.Builtin.URL != "" {
					inObj = map[string]any{"url": ev.Builtin.URL}
				}
				cb = map[string]any{
					"type":  "server_tool_use",
					"id":    id,
					"name":  string(ev.Builtin.Kind),
					"input": inObj,
				}
			} else {
				resType := "web_search_tool_result"
				if ev.Builtin.Kind == ir.BuiltinWebFetch {
					resType = "web_fetch_tool_result"
				}
				var content any = []any{}
				if ev.Builtin.Origin == ir.FormatAnthropic && len(ev.Builtin.Raw) > 0 {
					content = json.RawMessage(ev.Builtin.Raw)
				} else if len(ev.Builtin.Hits) > 0 {
					hits := make([]map[string]any, 0, len(ev.Builtin.Hits))
					for _, h := range ev.Builtin.Hits {
						item := map[string]any{
							"type":  "web_search_result",
							"url":   h.URL,
							"title": h.Title,
						}
						if h.PageAge != "" {
							item["page_age"] = h.PageAge
						}
						hits = append(hits, item)
					}
					content = hits
				} else if ev.Builtin.Content != "" {
					content = ev.Builtin.Content
				} else if ev.Builtin.Error != "" {
					if err := e.opts.Env.Lose(ir.LossBuiltinActivity, ir.Root.Key("content"), "cross-format builtin error result cannot be represented natively in Anthropic"); err != nil {
						return dst, err
					}
					content = ev.Builtin.Error
				}
				cb = map[string]any{
					"type":        resType,
					"tool_use_id": id,
					"content":     content,
				}
			}
			dst = e.appendSSE(dst, "content_block_start", map[string]any{
				"type":          "content_block_start",
				"index":         idx,
				"content_block": cb,
			})
			dst = e.appendSSE(dst, "content_block_stop", map[string]any{
				"type":  "content_block_stop",
				"index": idx,
			})
			break
		}
		if err := e.opts.Env.Lose(ir.LossBuiltinActivity, ir.Root.Key("content"), "builtin activity not supported by Anthropic"); err != nil {
			return dst, err
		}

	case ir.EvStop:
		if e.openBlock >= 0 && e.openBlockCand == ev.Index {
			dst = e.closeBlock(dst)
			if e.errored {
				break
			}
		}
		if e.openBlock < 0 {
			dst = e.drainPendingTools(dst)
			if e.errored {
				break
			}
		}
		if ev.Index == 0 || e.stop == nil {
			if ev.Stop != nil {
				st := *ev.Stop
				e.stop = &st
			} else {
				st := ir.Stop{Reason: ir.StopEndTurn}
				e.stop = &st
			}
		}

	case ir.EvUsage:
		if ev.Usage != nil {
			if e.usage == nil {
				u := *ev.Usage
				e.usage = &u
			} else {
				e.usage.Merge(*ev.Usage)
			}
		}

	case ir.EvError:
		if ev.Err != nil {
			e.errored = true
			dst = BuildStreamError(dst, ev.Err)
		}
	}
	return dst, nil
}

// Close writes the merged message_delta and message_stop events unless the stream ended with an error.
func (e *StreamEncoder) Close(dst []byte) ([]byte, error) {
	if e.errored {
		return dst, nil
	}
	if !e.started && e.stop == nil && e.usage == nil {
		return dst, nil
	}
	dst = e.ensureStarted(dst)
	dst = e.closeBlock(dst)
	if e.errored {
		return dst, nil
	}
	dst = e.flushAllPendingTools(dst)
	if e.errored {
		return dst, nil
	}
	if len(e.deferredOtherCand) > 0 {
		var err error
		dst, err = e.replayDeferredOtherCand(dst)
		if err != nil {
			return dst, err
		}
		if e.errored {
			return dst, nil
		}
		dst = e.closeBlock(dst)
		if e.errored {
			return dst, nil
		}
	}
	reason := stopToAnthropic(e.stop)
	seq := ""
	if e.stop != nil {
		seq = e.stop.Sequence
	}
	var usageMap map[string]any
	if e.usage != nil {
		usageMap = buildAnthropicUsage(*e.usage)
	} else {
		usageMap = map[string]any{"output_tokens": 0}
	}
	dst = e.appendSSE(dst, "message_delta", map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   reason,
			"stop_sequence": seqOrNull(seq),
		},
		"usage": usageMap,
	})
	dst = e.appendSSE(dst, "message_stop", map[string]any{"type": "message_stop"})
	return dst, nil
}

// Keepalive sends a ping event.
func (e *StreamEncoder) Keepalive(dst []byte) []byte {
	return e.emitPing(dst)
}

func (e *StreamEncoder) emitMessageStart(dst []byte, ev ir.Event) []byte {
	usageMap := map[string]any{
		"input_tokens":  0,
		"output_tokens": 0,
	}
	if ev.Usage != nil {
		usageMap["input_tokens"] = ev.Usage.InputTokens
		if ev.Usage.Has(ir.UCacheRead) {
			usageMap["cache_read_input_tokens"] = ev.Usage.CacheReadTokens
		}
		if ev.Usage.Has(ir.UCacheWrite) {
			usageMap["cache_creation_input_tokens"] = ev.Usage.CacheWriteTokens
		}
	} else if e.opts.PromptEstimate > 0 {
		usageMap["input_tokens"] = e.opts.PromptEstimate
	}
	model := e.model
	if model == "" {
		model = e.opts.Model
	}
	msg := map[string]any{
		"id":            e.id,
		"type":          "message",
		"role":          "assistant",
		"model":         model,
		"content":       []any{},
		"stop_reason":   nil,
		"stop_sequence": nil,
		"usage":         usageMap,
	}
	return e.appendSSE(dst, "message_start", map[string]any{
		"type":    "message_start",
		"message": msg,
	})
}

func (e *StreamEncoder) emitPing(dst []byte) []byte {
	return e.appendSSE(dst, "ping", map[string]any{"type": "ping"})
}

func (e *StreamEncoder) ensureTextBlock(dst []byte, cand int) []byte {
	if e.openBlock >= 0 && e.openBlockType == "text" && e.openBlockCand == cand {
		return dst
	}
	dst = e.closeBlock(dst)
	if e.errored {
		return dst
	}
	idx := e.blockIdx
	e.blockIdx++
	e.openBlock = idx
	e.openBlockType = "text"
	e.openBlockCand = cand
	return e.appendSSE(dst, "content_block_start", map[string]any{
		"type":          "content_block_start",
		"index":         idx,
		"content_block": map[string]any{"type": "text", "text": ""},
	})
}

func (e *StreamEncoder) openToolBlock(dst []byte, cand int, id, name string, sig *ir.Signature, kind ir.ToolKind) []byte {
	dst = e.closeBlock(dst)
	if e.errored {
		return dst
	}
	idx := e.blockIdx
	e.blockIdx++
	e.openBlock = idx
	e.openBlockType = "tool_use"
	e.openBlockCand = cand
	e.activeCallCand = cand
	e.activeCallID = id
	e.activeKind = kind
	e.activeCustomOpened = false
	e.activeCustomSent.Reset()
	e.activeFuncSent.Reset()
	e.activeToolFinished = false
	wireID := ir.AnthropicSafeID(ir.EncodeCallSignatureID(e.opts.Env, id, sig))
	return e.appendSSE(dst, "content_block_start", map[string]any{
		"type":  "content_block_start",
		"index": idx,
		"content_block": map[string]any{
			"type":  "tool_use",
			"id":    wireID,
			"name":  name,
			"input": json.RawMessage(`{}`),
		},
	})
}

func (e *StreamEncoder) lookupPendingTool(cand int, id string) *pendingTool {
	if id != "" {
		return e.pendingByKey[toolCallKey{cand: cand, id: id}]
	}
	for i := len(e.pendingTools) - 1; i >= 0; i-- {
		if pt := e.pendingTools[i]; pt.cand == cand && !pt.done {
			return pt
		}
	}
	return nil
}

func (e *StreamEncoder) maxBufferedBytes() int {
	if e.opts.MaxBufferedBytes > 0 {
		return e.opts.MaxBufferedBytes
	}
	return defaultMaxBufferedBytes
}

func anthropicEventByteSize(ev ir.Event) int {
	n := len(ev.Text) + len(ev.Arguments)
	if ev.Signature != nil {
		n += len(ev.Signature.Value)
	}
	if ev.Sealed != nil {
		n += len(ev.Sealed.Data)
	}
	if ev.Media != nil {
		n += len(ev.Media.Data) + len(ev.Media.URL) + len(ev.Media.Text)
	}
	if ev.Builtin != nil {
		n += len(ev.Builtin.Input) + len(ev.Builtin.Content) + len(ev.Builtin.Raw)
	}
	if ev.Citation != nil {
		n += len(ev.Citation.URL) + len(ev.Citation.Title) + len(ev.Citation.CitedText) + len(ev.Citation.Raw)
	}
	return n
}

func (e *StreamEncoder) abortOpenTool(dst []byte, err *ir.APIError) []byte {
	e.openBlock = -1
	e.openBlockType = ""
	e.openBlockCand = 0
	e.activeCallCand = 0
	e.activeCallID = ""
	e.activeKind = ""
	e.activeCustomOpened = false
	e.activeCustomSent.Reset()
	e.activeFuncSent.Reset()
	e.activeToolFinished = false
	e.pendingTools = nil
	e.pendingByKey = map[toolCallKey]*pendingTool{}
	e.pendingBytes = 0
	e.deferredOtherCand = nil
	e.deferredBytes = 0
	e.errored = true
	if err == nil {
		err = &ir.APIError{Status: 502, Type: ir.ErrServer, Message: "tool call aborted"}
	}
	return BuildStreamError(dst, err)
}

func (e *StreamEncoder) drainPendingTools(dst []byte) []byte {
	for len(e.pendingTools) > 0 {
		pt := e.pendingTools[0]
		e.pendingTools = e.pendingTools[1:]
		e.pendingBytes -= pt.countedBytes
		if e.pendingBytes < 0 {
			e.pendingBytes = 0
		}
		if pt.id != "" {
			delete(e.pendingByKey, toolCallKey{cand: pt.cand, id: pt.id})
		}
		dst = e.openToolBlock(dst, pt.cand, pt.id, pt.name, pt.sig, pt.kind)
		if e.errored {
			return dst
		}
		if pt.kind == ir.ToolCustom {
			if s := pt.args.String(); s != "" {
				dst = e.appendSSE(dst, "content_block_delta", map[string]any{
					"type":  "content_block_delta",
					"index": e.openBlock,
					"delta": map[string]any{"type": "input_json_delta", "partial_json": customInputDelta(s, true)},
				})
				e.activeCustomOpened = true
				e.activeCustomSent.WriteString(s)
			}
			if pt.done {
				var ok bool
				dst, ok = e.finishCustomInput(dst, pt.finalArgs)
				if !ok {
					return dst
				}
			}
		} else {
			if s := pt.args.String(); s != "" {
				dst = e.appendSSE(dst, "content_block_delta", map[string]any{
					"type":  "content_block_delta",
					"index": e.openBlock,
					"delta": map[string]any{"type": "input_json_delta", "partial_json": s},
				})
				e.activeFuncSent.WriteString(s)
			}
			if pt.done {
				var ok bool
				dst, ok = e.finishFunctionArgs(dst, pt.finalArgs)
				if !ok {
					return dst
				}
			}
		}
		if pt.done {
			e.activeToolFinished = true
			dst = e.closeBlock(dst)
			continue
		}
		// Keep this tool open to receive further live deltas.
		break
	}
	return dst
}

// customInputDelta escapes a chunk of a custom tool's raw input for an
// input_json_delta, starting the {"input":" wrapper when first is true.
func customInputDelta(text string, first bool) string {
	var sb strings.Builder
	if first {
		sb.WriteString(`{"input":"`)
	}
	if b, err := jsonx.Marshal(text); err == nil && len(b) >= 2 {
		sb.Write(b[1 : len(b)-1])
	}
	return sb.String()
}

// finishFunctionArgs reconciles streamed function arguments against EvToolEnd.Arguments,
// emitting missing prefix suffixes (or full final arguments when no delta was sent) and
// aborting with a stream error if streamed text disagrees with the final JSON object.
func (e *StreamEncoder) finishFunctionArgs(dst []byte, final json.RawMessage) ([]byte, bool) {
	sent := e.activeFuncSent.String()
	full, suffix, ok := jsonx.ReconcileFunctionArgs(sent, string(final))
	if !ok {
		return e.abortOpenTool(dst, &ir.APIError{
			Status:  502,
			Type:    ir.ErrServer,
			Message: "streamed tool arguments do not match final arguments",
		}), false
	}
	if sent == "" {
		// content_block_start already seeds "input": {}, so omit a redundant "{}" delta.
		if full == "" || full == "{}" {
			return dst, true
		}
		suffix = full
	}
	if suffix == "" {
		return dst, true
	}
	e.activeFuncSent.WriteString(suffix)
	return e.appendSSE(dst, "content_block_delta", map[string]any{
		"type":  "content_block_delta",
		"index": e.openBlock,
		"delta": map[string]any{"type": "input_json_delta", "partial_json": suffix},
	}), true
}

// finishCustomInput completes the open custom tool block's input. When part of
// the wrapper was already streamed it verifies the streamed text is a prefix of
// the final input (emitting any missing suffix before closing `"}`) and aborts
// with a stream error if the streamed text is not a prefix. When nothing was
// streamed it writes the whole {"input": ...} object from final.
func (e *StreamEncoder) finishCustomInput(dst []byte, final json.RawMessage) ([]byte, bool) {
	full, suffix, ok := jsonx.ReconcileCustomInput(e.activeCustomSent.String(), e.activeCustomOpened, final)
	if !ok {
		return e.abortOpenTool(dst, &ir.APIError{
			Status:  502,
			Type:    ir.ErrServer,
			Message: "streamed custom tool input does not match final input",
		}), false
	}
	var tail string
	if e.activeCustomOpened {
		tail = customInputDelta(suffix, false) + `"}`
		e.activeCustomSent.WriteString(suffix)
	} else {
		b, _ := jsonx.Marshal(map[string]string{"input": full})
		tail = string(b)
		e.activeCustomSent.WriteString(full)
	}
	e.activeCustomOpened = true
	return e.appendSSE(dst, "content_block_delta", map[string]any{
		"type":  "content_block_delta",
		"index": e.openBlock,
		"delta": map[string]any{"type": "input_json_delta", "partial_json": tail},
	}), true
}

func (e *StreamEncoder) replayDeferredOtherCand(dst []byte) ([]byte, error) {
	evs := e.deferredOtherCand
	e.deferredOtherCand = nil
	e.deferredBytes = 0
	for _, ev := range evs {
		var err error
		dst, err = e.Encode(dst, ev)
		if err != nil || e.errored {
			return dst, err
		}
	}
	return dst, nil
}

func (e *StreamEncoder) flushAllPendingTools(dst []byte) []byte {
	for _, pt := range e.pendingTools {
		if !pt.done {
			// Uncompleted pending tool call at stream close: abort rather than
			// fabricating completion of truncated arguments.
			return e.abortOpenTool(dst, &ir.APIError{
				Status:  502,
				Type:    ir.ErrServer,
				Message: "pending tool_use block never completed",
			})
		}
	}
	if len(e.pendingTools) > 0 {
		dst = e.drainPendingTools(dst)
	}
	return dst
}

func (e *StreamEncoder) closeBlock(dst []byte) []byte {
	if e.openBlock < 0 {
		return dst
	}
	if e.openBlockType == "tool_use" && !e.activeToolFinished {
		// Never emit content_block_stop for an unfinished tool_use block (which
		// could leave unclosed or truncated JSON on the wire); abort instead.
		return e.abortOpenTool(dst, &ir.APIError{
			Status:  502,
			Type:    ir.ErrServer,
			Message: "tool_use block closed before completion",
		})
	}
	idx := e.openBlock
	if e.openBlockType == "thinking" {
		if e.thinkingSigVal.Len() > 0 {
			sigVal := e.thinkingSigVal.String()
			if needsAnthropicSeal(e.thinkingSigOrigin, e.opts.Env) {
				sigVal = e.opts.Env.Sealer().Seal(ir.SealRecord{
					Origin:  e.thinkingSigOrigin,
					Kind:    ir.SealSignature,
					Value:   sigVal,
					TextSum: ir.TextSum(e.thinkingTextAccum.String()),
				})
			}
			dst = e.appendSSE(dst, "content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": idx,
				"delta": map[string]any{"type": "signature_delta", "signature": sigVal},
			})
		} else if e.opts.SealUnsignedReasoning && e.thinkingTextAccum.Len() > 0 {
			origin := e.opts.UnsignedOrigin
			if origin.Format == "" {
				origin = ir.Origin{Format: ir.FormatChat, Provider: e.opts.Env.ProviderID}
			}
			if sigVal := ir.SealUnsignedReasoning(e.opts.Env, ir.FormatAnthropic, origin, e.thinkingTextAccum.String()); sigVal != "" {
				dst = e.appendSSE(dst, "content_block_delta", map[string]any{
					"type":  "content_block_delta",
					"index": idx,
					"delta": map[string]any{"type": "signature_delta", "signature": sigVal},
				})
			}
		}
	}
	e.thinkingTextAccum.Reset()
	e.thinkingSigVal.Reset()
	e.thinkingSigOrigin = ir.Origin{}
	e.openBlock = -1
	e.openBlockType = ""
	e.openBlockCand = 0
	e.activeCallCand = 0
	e.activeCallID = ""
	e.activeKind = ""
	e.activeCustomOpened = false
	e.activeCustomSent.Reset()
	e.activeFuncSent.Reset()
	e.activeToolFinished = false
	return e.appendSSE(dst, "content_block_stop", map[string]any{
		"type":  "content_block_stop",
		"index": idx,
	})
}

func (e *StreamEncoder) appendSSE(dst []byte, eventName string, v any) []byte {
	b, err := jsonx.Marshal(v)
	if err != nil {
		return dst
	}
	return sse.AppendEvent(dst, eventName, b)
}

// BuildResponse serialises a whole ir.Response to a non-streaming Anthropic body.
func BuildResponse(resp *ir.Response, opts EncodeOptions) ([]byte, error) {
	env := opts.Env
	if env == nil {
		env = &ir.Env{}
	}
	id := resp.ID
	if opts.GenerateResponseID || id == "" {
		id = "msg_" + env.NewID("", 24)
	}
	model := opts.Model
	if model == "" {
		model = resp.Model
	}

	var out *ir.Output
	if len(resp.Outputs) > 0 {
		out = &resp.Outputs[0]
	}

	var blocks []wireContentBlock
	if out != nil {
		for i, p := range out.Parts {
			fp := ir.Root.Key("content").At(i)
			bs, err := responsePartToBlocks(p, fp, env, opts)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, bs...)
		}
	}
	if blocks == nil {
		blocks = []wireContentBlock{}
	}

	var stopReason, stopSeq string
	if out != nil {
		stopReason = stopToAnthropic(&out.Stop)
		stopSeq = out.Stop.Sequence
	}

	content, err := jsonx.Marshal(blocks)
	if err != nil {
		return nil, err
	}

	wr := map[string]any{
		"id":            id,
		"type":          "message",
		"role":          "assistant",
		"model":         model,
		"content":       json.RawMessage(content),
		"stop_reason":   stopReason,
		"stop_sequence": seqOrNull(stopSeq),
		"usage":         buildAnthropicUsage(resp.Usage),
	}
	return jsonx.Marshal(wr)
}

func needsAnthropicSeal(origin ir.Origin, env *ir.Env) bool {
	if origin.Format != ir.FormatAnthropic {
		return true
	}
	if origin.Provider != "" {
		return true
	}
	return env != nil && env.ProviderID != ""
}

func responsePartToBlocks(part ir.Part, p ir.Path, env *ir.Env, opts EncodeOptions) ([]wireContentBlock, error) {
	if part.Type == ir.PartReasoning && part.Reasoning != nil {
		r := part.Reasoning
		var out []wireContentBlock
		if r.Text != "" {
			sig := ""
			if r.Signature != nil && r.Signature.Value != "" {
				if !needsAnthropicSeal(r.Signature.Origin, env) {
					sig = r.Signature.Value
				} else {
					sig = env.Sealer().Seal(ir.SealRecord{
						Origin:  r.Signature.Origin,
						Kind:    ir.SealSignature,
						Value:   r.Signature.Value,
						TextSum: ir.TextSum(r.Text),
					})
				}
			} else if opts.SealUnsignedReasoning {
				origin := opts.UnsignedOrigin
				if origin.Format == "" {
					origin = ir.Origin{Format: ir.FormatChat, Provider: env.ProviderID}
				}
				sig = ir.SealUnsignedReasoning(env, ir.FormatAnthropic, origin, r.Text)
			}
			out = append(out, wireContentBlock{Type: "thinking", Thinking: r.Text, Signature: &sig})
		}
		if r.Sealed != nil {
			var dataStr string
			if r.Sealed.Kind == ir.SealedAnthropicRedacted && !needsAnthropicSeal(r.Sealed.Origin, env) {
				_ = json.Unmarshal(r.Sealed.Data, &dataStr)
			} else {
				sk := ir.SealEncrypted
				if r.Sealed.Kind == ir.SealedAnthropicRedacted {
					sk = ir.SealRedacted
				}
				dataStr = env.Sealer().Seal(ir.SealRecord{
					Origin: r.Sealed.Origin,
					Kind:   sk,
					Value:  string(r.Sealed.Data),
				})
			}
			out = append(out, wireContentBlock{Type: "redacted_thinking", Data: dataStr})
		}
		return out, nil
	}
	if part.Type == ir.PartBuiltinCall || part.Type == ir.PartBuiltinResult {
		if part.Builtin != nil && (part.Builtin.Kind == ir.BuiltinWebSearch || part.Builtin.Kind == ir.BuiltinWebFetch) {
			id := part.Builtin.ID
			if id == "" {
				id = "srvtoolu_" + env.NewID("", 16)
			}
			if !part.Builtin.IsResult {
				if part.Builtin.Origin != ir.FormatAnthropic && len(part.Builtin.Raw) > 0 {
					if err := env.Lose(ir.LossBuiltinActivity, p, "cross-format builtin call metadata cannot be fully represented in Anthropic server_tool_use"); err != nil {
						return nil, err
					}
				}
				inRaw := part.Builtin.Input
				if len(inRaw) == 0 && part.Builtin.Query != "" {
					inRaw, _ = jsonx.Marshal(map[string]any{"query": part.Builtin.Query})
				} else if len(inRaw) == 0 && part.Builtin.URL != "" {
					inRaw, _ = jsonx.Marshal(map[string]any{"url": part.Builtin.URL})
				}
				if len(inRaw) == 0 {
					inRaw = json.RawMessage(`{}`)
				}
				return []wireContentBlock{{Type: "server_tool_use", ID: id, Name: string(part.Builtin.Kind), Input: inRaw}}, nil
			}
			resType := "web_search_tool_result"
			if part.Builtin.Kind == ir.BuiltinWebFetch {
				resType = "web_fetch_tool_result"
			}
			var content json.RawMessage
			if part.Builtin.Origin == ir.FormatAnthropic && len(part.Builtin.Raw) > 0 {
				content = append(json.RawMessage(nil), part.Builtin.Raw...)
			} else if len(part.Builtin.Hits) > 0 {
				hits := make([]map[string]any, 0, len(part.Builtin.Hits))
				for _, h := range part.Builtin.Hits {
					item := map[string]any{"type": "web_search_result", "url": h.URL, "title": h.Title}
					if h.PageAge != "" {
						item["page_age"] = h.PageAge
					}
					hits = append(hits, item)
				}
				content, _ = jsonx.Marshal(hits)
			} else if part.Builtin.Error != "" && part.Builtin.Content == "" {
				if err := env.Lose(ir.LossBuiltinActivity, p, "cross-format builtin error result cannot be represented natively in Anthropic"); err != nil {
					return nil, err
				}
				content, _ = jsonx.Marshal(part.Builtin.Error)
			} else {
				content, _ = jsonx.Marshal(part.Builtin.Content)
			}
			return []wireContentBlock{{Type: resType, ToolUseID: id, Content: content}}, nil
		}
		if err := env.Lose(ir.LossBuiltinActivity, p, "builtin activity not supported by Anthropic"); err != nil {
			return nil, err
		}
		return nil, nil
	}
	if part.Type == ir.PartText && len(part.Citations) > 0 {
		cits := make([]json.RawMessage, 0, len(part.Citations))
		for _, c := range part.Citations {
			if len(c.Raw) > 0 {
				cits = append(cits, c.Raw)
			} else {
				b, _ := jsonx.Marshal(map[string]any{
					"type":       "web_search_result_location",
					"url":        c.URL,
					"title":      c.Title,
					"cited_text": c.CitedText,
				})
				cits = append(cits, b)
			}
		}
		return []wireContentBlock{{Type: "text", Text: part.Text, Citations: cits, CacheControl: irCacheToWire(part.Cache)}}, nil
	}
	if part.Type == ir.PartToolCall && part.Call != nil {
		call := part.Call
		args := call.Arguments
		if call.Kind == ir.ToolCustom && len(args) == 0 {
			args, _ = jsonx.Marshal(map[string]string{"input": call.Input})
		}
		if len(args) == 0 {
			args = json.RawMessage(`{}`)
		}
		wireID := ir.AnthropicSafeID(ir.EncodeCallSignatureID(env, call.ID, call.Signature))
		return []wireContentBlock{{
			Type:  "tool_use",
			ID:    wireID,
			Name:  call.Name,
			Input: args,
		}}, nil
	}
	return partToBlocks(part, p, env)
}

func stopToAnthropic(s *ir.Stop) string {
	if s == nil {
		return "end_turn"
	}
	switch s.Reason {
	case ir.StopEndTurn:
		return "end_turn"
	case ir.StopMaxTokens, ir.StopContextWindow:
		return "max_tokens"
	case ir.StopSequence:
		return "stop_sequence"
	case ir.StopToolUse:
		return "tool_use"
	case ir.StopContentFilter, ir.StopRefusal:
		return "refusal"
	case ir.StopPauseTurn:
		return "pause_turn"
	}
	return "end_turn"
}

func buildAnthropicUsage(u ir.Usage) map[string]any {
	// Anthropic input_tokens excludes cache_read_input_tokens and cache_creation_input_tokens.
	m := map[string]any{
		"input_tokens":  u.InputTokens,
		"output_tokens": u.OutputTokens,
	}
	if u.Has(ir.UCacheRead) {
		m["cache_read_input_tokens"] = u.CacheReadTokens
	}
	if u.Has(ir.UCacheWrite) {
		m["cache_creation_input_tokens"] = u.CacheWriteTokens
	}
	if u.Has(ir.UCacheWriteSplit) && (u.CacheWrite5mTokens > 0 || u.CacheWrite1hTokens > 0) {
		cc := map[string]any{}
		if u.CacheWrite5mTokens > 0 {
			cc["ephemeral_5m_input_tokens"] = u.CacheWrite5mTokens
		}
		if u.CacheWrite1hTokens > 0 {
			cc["ephemeral_1h_input_tokens"] = u.CacheWrite1hTokens
		}
		m["cache_creation"] = cc
	}
	if u.WebSearchRequests > 0 || u.WebFetchRequests > 0 {
		stu := map[string]any{}
		if u.WebSearchRequests > 0 {
			stu["web_search_requests"] = u.WebSearchRequests
		}
		if u.WebFetchRequests > 0 {
			stu["web_fetch_requests"] = u.WebFetchRequests
		}
		m["server_tool_use"] = stu
	}
	if u.Has(ir.UReasoning) && u.ReasoningTokens > 0 {
		m["output_tokens_details"] = map[string]any{
			"thinking_tokens": u.ReasoningTokens,
		}
	}
	return m
}

func seqOrNull(seq string) any {
	if seq == "" {
		return nil
	}
	return seq
}
