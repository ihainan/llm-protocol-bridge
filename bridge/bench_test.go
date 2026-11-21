package bridge_test

import (
	"testing"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
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

func BenchmarkConvert_ChatToAnthropic(b *testing.B)    { benchConvert(b, ir.FormatChat, ir.FormatAnthropic, chatBody) }
func BenchmarkConvert_ChatToResponses(b *testing.B)    { benchConvert(b, ir.FormatChat, ir.FormatResponses, chatBody) }
func BenchmarkConvert_ChatToGemini(b *testing.B)       { benchConvert(b, ir.FormatChat, ir.FormatGemini, chatBody) }
func BenchmarkConvert_AnthropicToChat(b *testing.B)    { benchConvert(b, ir.FormatAnthropic, ir.FormatChat, anthropicBody) }
func BenchmarkConvert_AnthropicToResponses(b *testing.B) { benchConvert(b, ir.FormatAnthropic, ir.FormatResponses, anthropicBody) }
func BenchmarkConvert_AnthropicToGemini(b *testing.B)  { benchConvert(b, ir.FormatAnthropic, ir.FormatGemini, anthropicBody) }
func BenchmarkConvert_ResponsesToChat(b *testing.B)    { benchConvert(b, ir.FormatResponses, ir.FormatChat, responsesBody) }
func BenchmarkConvert_ResponsesToAnthropic(b *testing.B) { benchConvert(b, ir.FormatResponses, ir.FormatAnthropic, responsesBody) }
func BenchmarkConvert_ResponsesToGemini(b *testing.B)  { benchConvert(b, ir.FormatResponses, ir.FormatGemini, responsesBody) }
func BenchmarkConvert_GeminiToChat(b *testing.B)       { benchConvert(b, ir.FormatGemini, ir.FormatChat, geminiBody) }
func BenchmarkConvert_GeminiToAnthropic(b *testing.B)  { benchConvert(b, ir.FormatGemini, ir.FormatAnthropic, geminiBody) }
func BenchmarkConvert_GeminiToResponses(b *testing.B)  { benchConvert(b, ir.FormatGemini, ir.FormatResponses, geminiBody) }
