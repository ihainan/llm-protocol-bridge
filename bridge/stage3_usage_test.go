package bridge_test

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/anthropic"
	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/chat"
	"github.com/ihainan/llm-protocol-bridge/gemini"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/responses"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// Item 5: Usage accounting strictly matching docs/design.md section 6.
func TestStage3_UsageAccounting(t *testing.T) {
	t.Run("anthropic_message_delta_preserves_input_and_handles_explicit_zero", func(t *testing.T) {
		d := anthropic.NewStreamDecoder(anthropic.DecodeOptions{})
		c := ir.NewCollector()
		chunks := []string{
			`{"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":100,"output_tokens":1,"cache_read_input_tokens":20,"cache_creation_input_tokens":30,"cache_creation":{"ephemeral_5m_input_tokens":10,"ephemeral_1h_input_tokens":20}}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`,
			`{"type":"content_block_stop","index":0}`,
			// message_delta repeats cache_read_input_tokens but omits input_tokens!
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":8,"cache_read_input_tokens":20,"server_tool_use":{"web_search_requests":2,"web_fetch_requests":1}}}`,
			`{"type":"message_stop"}`,
		}
		for _, ch := range chunks {
			evs, err := d.Decode(nil, sse.Event{Data: []byte(ch)})
			if err != nil {
				t.Fatalf("Decode: %v", err)
			}
			for _, e := range evs {
				c.Add(e)
			}
		}
		u := c.Response().Usage
		if u.InputTokens != 100 || u.CacheReadTokens != 20 || u.CacheWriteTokens != 30 || u.OutputTokens != 8 {
			t.Fatalf("unexpected usage: %+v", u)
		}
		if u.CacheWrite5mTokens != 10 || u.CacheWrite1hTokens != 20 || !u.Has(ir.UCacheWriteSplit) {
			t.Errorf("cache write split missing: %+v", u)
		}
		if u.WebSearchRequests != 2 || u.WebFetchRequests != 1 || !u.Has(ir.UWebSearch) || !u.Has(ir.UWebFetch) {
			t.Errorf("server_tool_use counts missing: %+v", u)
		}
	})

	t.Run("anthropic_encode_excludes_cache_from_input_tokens", func(t *testing.T) {
		resp := &ir.Response{
			ID:    "msg_1",
			Model: "claude-4",
			Outputs: []ir.Output{{
				Index: 0,
				Parts: []ir.Part{ir.TextPart("ok")},
				Stop:  ir.Stop{Reason: ir.StopEndTurn},
			}},
			Usage: ir.Usage{
				InputTokens:        100,
				CacheReadTokens:    20,
				CacheWriteTokens:   30,
				CacheWrite5mTokens: 10,
				CacheWrite1hTokens: 20,
				OutputTokens:       15,
				WebSearchRequests:  2,
				Reported:           ir.UInput | ir.UCacheRead | ir.UCacheWrite | ir.UCacheWriteSplit | ir.UOutput | ir.UWebSearch,
			},
		}
		body, err := anthropic.BuildResponse(resp, anthropic.EncodeOptions{Env: &ir.Env{}, Model: "claude-4"})
		if err != nil {
			t.Fatalf("BuildResponse: %v", err)
		}
		var parsed struct {
			Usage struct {
				InputTokens              int `json:"input_tokens"`
				OutputTokens             int `json:"output_tokens"`
				CacheReadInputTokens     int `json:"cache_read_input_tokens"`
				CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
				CacheCreation            *struct {
					Ephemeral5m int `json:"ephemeral_5m_input_tokens"`
					Ephemeral1h int `json:"ephemeral_1h_input_tokens"`
				} `json:"cache_creation"`
				ServerToolUse *struct {
					WebSearchRequests int `json:"web_search_requests"`
				} `json:"server_tool_use"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if parsed.Usage.InputTokens != 100 {
			t.Errorf("Anthropic input_tokens double-counted cache: got %d, want 100 (body=%s)", parsed.Usage.InputTokens, body)
		}
		if parsed.Usage.CacheReadInputTokens != 20 || parsed.Usage.CacheCreationInputTokens != 30 {
			t.Errorf("cache tokens mismatch: %+v", parsed.Usage)
		}
		if parsed.Usage.CacheCreation == nil || parsed.Usage.CacheCreation.Ephemeral5m != 10 || parsed.Usage.CacheCreation.Ephemeral1h != 20 {
			t.Errorf("cache_creation split missing: %s", body)
		}
		if parsed.Usage.ServerToolUse == nil || parsed.Usage.ServerToolUse.WebSearchRequests != 2 {
			t.Errorf("server_tool_use missing: %s", body)
		}
	})

	t.Run("gemini_usage_bidirectional_thoughts_and_tool_prompt", func(t *testing.T) {
		raw := []byte(`{
			"candidates": [{"index":0,"content":{"role":"model","parts":[{"text":"42"}]},"finishReason":"STOP"}],
			"usageMetadata": {
				"promptTokenCount": 120,
				"cachedContentTokenCount": 20,
				"toolUsePromptTokenCount": 15,
				"candidatesTokenCount": 30,
				"thoughtsTokenCount": 50,
				"totalTokenCount": 215
			}
		}`)
		resp, err := gemini.ParseResponse(raw, gemini.DecodeOptions{})
		if err != nil {
			t.Fatalf("ParseResponse: %v", err)
		}
		// Per design.md §6:
		// Input = promptTokenCount - cachedContentTokenCount + toolUsePromptTokenCount = 120 - 20 + 15 = 115
		// CacheRead = 20 -> PromptTokens() = 135
		// Output = candidatesTokenCount + thoughtsTokenCount = 30 + 50 = 80
		// Reasoning = 50 -> TotalTokens() = 215
		u := resp.Usage
		if u.InputTokens != 115 || u.CacheReadTokens != 20 || u.ToolUsePromptTokens != 15 {
			t.Errorf("Gemini decode input mismatch: %+v", u)
		}
		if u.OutputTokens != 80 || u.ReasoningTokens != 50 {
			t.Errorf("Gemini decode output/reasoning mismatch: %+v", u)
		}
		if u.TotalTokens() != 215 {
			t.Errorf("TotalTokens: want 215, got %d", u.TotalTokens())
		}

		// Encode back to Gemini: candidatesTokenCount must be Output - Reasoning = 30,
		// promptTokenCount must be PromptTokens() - ToolUsePrompt = 120.
		out, err := gemini.BuildResponse(resp, gemini.EncodeOptions{Env: &ir.Env{}})
		if err != nil {
			t.Fatalf("BuildResponse: %v", err)
		}
		var w struct {
			UsageMetadata struct {
				PromptTokenCount        int `json:"promptTokenCount"`
				CandidatesTokenCount    int `json:"candidatesTokenCount"`
				ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
				CachedContentTokenCount int `json:"cachedContentTokenCount"`
				ToolUsePromptTokenCount int `json:"toolUsePromptTokenCount"`
				TotalTokenCount         int `json:"totalTokenCount"`
			} `json:"usageMetadata"`
		}
		if err := json.Unmarshal(out, &w); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if w.UsageMetadata.PromptTokenCount != 120 || w.UsageMetadata.CandidatesTokenCount != 30 ||
			w.UsageMetadata.ThoughtsTokenCount != 50 || w.UsageMetadata.CachedContentTokenCount != 20 ||
			w.UsageMetadata.ToolUsePromptTokenCount != 15 || w.UsageMetadata.TotalTokenCount != 215 {
			t.Errorf("Gemini encode usage round-trip mismatch: %+v", w.UsageMetadata)
		}
	})

	t.Run("bridge_stream_usage_populated_in_streaming_mode", func(t *testing.T) {
		plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat,
			[]byte(`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"}],"stream":true}`), bridge.Options{})
		if err != nil {
			t.Fatalf("Convert: %v", err)
		}
		s := plan.NewStream()
		chatSSE := "data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n" +
			"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":25,\"total_tokens\":125,\"prompt_tokens_details\":{\"cached_tokens\":40},\"completion_tokens_details\":{\"reasoning_tokens\":10}}}\n\n" +
			"data: [DONE]\n\n"
		r := sse.NewReader(bytes.NewReader([]byte(chatSSE)))
		var out []byte
		for {
			ev, err := r.Next()
			if err != nil {
				break
			}
			out, err = s.Feed(out, ev)
			if err != nil {
				t.Fatalf("Feed: %v", err)
			}
		}
		_, err = s.End(out, ir.EndDone)
		if err != nil {
			t.Fatalf("End: %v", err)
		}
		u := s.Usage()
		if u.InputTokens != 60 || u.CacheReadTokens != 40 || u.OutputTokens != 25 || u.ReasoningTokens != 10 {
			t.Errorf("Stream.Usage() in NewStream() mode: got %+v, want input=60 cache_read=40 output=25 reasoning=10", u)
		}
	})

	t.Run("chat_and_responses_dialect_cache_fields", func(t *testing.T) {
		chatBody := []byte(`{"id":"1","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110,"prompt_cache_hit_tokens":30,"prompt_cache_miss_tokens":70,"cache_creation_input_tokens":15}}`)
		resp, err := chat.ParseResponse(chatBody, chat.DecodeOptions{
			Dialect: chat.Dialect{
				CacheReadFields:  []string{"prompt_cache_hit_tokens"},
				CacheWriteFields: []string{"cache_creation_input_tokens"},
			},
		})
		if err != nil {
			t.Fatalf("chat.ParseResponse: %v", err)
		}
		if resp.Usage.CacheReadTokens != 30 || resp.Usage.CacheWriteTokens != 15 || resp.Usage.InputTokens != 55 {
			t.Errorf("chat dialect cache fields: got %+v", resp.Usage)
		}

		respBody := []byte(`{"id":"r1","object":"response","model":"m","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":100,"output_tokens":10,"total_tokens":110,"input_tokens_details":{"cached_tokens":20,"custom_cache_write":10}}}`)
		r2, err := responses.ParseResponse(respBody, responses.DecodeOptions{
			Dialect: responses.Dialect{CacheWriteField: "custom_cache_write"},
		})
		if err != nil {
			t.Fatalf("responses.ParseResponse: %v", err)
		}
		if r2.Usage.CacheReadTokens != 20 || r2.Usage.CacheWriteTokens != 10 || r2.Usage.InputTokens != 70 {
			t.Errorf("responses dialect CacheWriteField: got %+v", r2.Usage)
		}
	})
}
