package bridge_test

import (
	"bytes"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

var (
	chatBody = []byte(`{"model":"gpt-4","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hello"},{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"Read","arguments":"{\"path\":\"/x\"}"}}]},{"role":"tool","tool_call_id":"c1","content":"done"},{"role":"user","content":"go on"}],"tools":[{"type":"function","function":{"name":"Read","description":"Reads","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}],"temperature":1,"stream":false}`)

	responsesBody = []byte(`{"model":"gpt-4","instructions":"be brief","input":[{"type":"message","role":"user","content":"hello"},{"type":"function_call","call_id":"c1","name":"Read","arguments":"{\"path\":\"/x\"}"},{"type":"function_call_output","call_id":"c1","output":"done"},{"type":"message","role":"user","content":"go on"}],"tools":[{"type":"function","name":"Read","description":"Reads","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}],"stream":false}`)

	anthropicBody = []byte(`{"model":"claude-3","max_tokens":1024,"system":"be brief","messages":[{"role":"user","content":"hello"},{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"Read","input":{"path":"/x"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":"done"},{"type":"text","text":"go on"}]}],"tools":[{"name":"Read","description":"Reads","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}]}`)

	geminiBody = []byte(`{"contents":[{"role":"user","parts":[{"text":"hello"}]},{"role":"model","parts":[{"functionCall":{"name":"Read","args":{"path":"/x"}}}]},{"role":"user","parts":[{"functionResponse":{"name":"Read","response":{"result":"done"}}},{"text":"go on"}]}],"tools":[{"functionDeclarations":[{"name":"Read","description":"Reads","parameters":{"type":"OBJECT","properties":{"path":{"type":"STRING"}}}}]}]}`)
)

func benchConvert(b *testing.B, client, upstream ir.Format, body []byte) {
	b.Helper()
	o := opts()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := bridge.Convert(client, upstream, body, o)
		if err != nil {
			b.Fatalf("Convert(%s→%s): %v", client, upstream, err)
		}
	}
}

func BenchmarkConvert_ChatToAnthropic(b *testing.B) {
	benchConvert(b, ir.FormatChat, ir.FormatAnthropic, chatBody)
}
func BenchmarkConvert_ChatToResponses(b *testing.B) {
	benchConvert(b, ir.FormatChat, ir.FormatResponses, chatBody)
}
func BenchmarkConvert_ChatToGemini(b *testing.B) {
	benchConvert(b, ir.FormatChat, ir.FormatGemini, chatBody)
}
func BenchmarkConvert_AnthropicToChat(b *testing.B) {
	benchConvert(b, ir.FormatAnthropic, ir.FormatChat, anthropicBody)
}
func BenchmarkConvert_AnthropicToResponses(b *testing.B) {
	benchConvert(b, ir.FormatAnthropic, ir.FormatResponses, anthropicBody)
}
func BenchmarkConvert_AnthropicToGemini(b *testing.B) {
	benchConvert(b, ir.FormatAnthropic, ir.FormatGemini, anthropicBody)
}
func BenchmarkConvert_ResponsesToChat(b *testing.B) {
	benchConvert(b, ir.FormatResponses, ir.FormatChat, responsesBody)
}
func BenchmarkConvert_ResponsesToAnthropic(b *testing.B) {
	benchConvert(b, ir.FormatResponses, ir.FormatAnthropic, responsesBody)
}
func BenchmarkConvert_ResponsesToGemini(b *testing.B) {
	benchConvert(b, ir.FormatResponses, ir.FormatGemini, responsesBody)
}
func BenchmarkConvert_GeminiToChat(b *testing.B) {
	benchConvert(b, ir.FormatGemini, ir.FormatChat, geminiBody)
}
func BenchmarkConvert_GeminiToAnthropic(b *testing.B) {
	benchConvert(b, ir.FormatGemini, ir.FormatAnthropic, geminiBody)
}
func BenchmarkConvert_GeminiToResponses(b *testing.B) {
	benchConvert(b, ir.FormatGemini, ir.FormatResponses, geminiBody)
}

var (
	benchChatEvent = sse.Event{
		Data: []byte(`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"content":"hello world"},"finish_reason":null}]}`),
	}
	benchAnthropicEvent = sse.Event{
		Name: "content_block_delta",
		Data: []byte(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello world"}}`),
	}
	benchChatFullStream = []byte(
		`data: {"id":"c1","model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}` + "\n\n" +
			`data: {"id":"c1","model":"gpt-4","choices":[{"index":0,"delta":{"content":"hello "}}]}` + "\n\n" +
			`data: {"id":"c1","model":"gpt-4","choices":[{"index":0,"delta":{"content":"world"}}]}` + "\n\n" +
			`data: {"id":"c1","model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}` + "\n\n" +
			"data: [DONE]\n\n",
	)
)

func BenchmarkStream_EventChatToAnthropic(b *testing.B) {
	plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, anthropicBody, opts())
	if err != nil {
		b.Fatalf("Convert: %v", err)
	}
	s := plan.NewStream()
	var dst []byte
	dst, _ = s.Feed(dst[:0], sse.Event{Data: []byte(`{"id":"c1","model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}`)})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dst, err = s.Feed(dst[:0], benchChatEvent)
		if err != nil {
			b.Fatalf("Feed: %v", err)
		}
	}
}

func BenchmarkStream_EventAnthropicToResponses(b *testing.B) {
	plan, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, responsesBody, opts())
	if err != nil {
		b.Fatalf("Convert: %v", err)
	}
	s := plan.NewStream()
	var dst []byte
	dst, _ = s.Feed(dst[:0], sse.Event{Name: "message_start", Data: []byte(`{"type":"message_start","message":{"id":"m1","model":"claude","usage":{"input_tokens":5,"output_tokens":0}}}`)})
	dst, _ = s.Feed(dst[:0], sse.Event{Name: "content_block_start", Data: []byte(`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dst, err = s.Feed(dst[:0], benchAnthropicEvent)
		if err != nil {
			b.Fatalf("Feed: %v", err)
		}
	}
}

func BenchmarkStream_FullChatToAnthropic(b *testing.B) {
	plan, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, anthropicBody, opts())
	if err != nil {
		b.Fatalf("Convert: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := plan.NewStream()
		r := sse.NewReader(bytes.NewReader(benchChatFullStream))
		var dst []byte
		for {
			ev, err := r.Next()
			if err != nil {
				break
			}
			dst, err = s.Feed(dst, ev)
			if err != nil {
				b.Fatalf("Feed: %v", err)
			}
		}
		if _, err := s.End(dst, ir.EndDone); err != nil {
			b.Fatalf("End: %v", err)
		}
	}
}
