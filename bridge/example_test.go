package bridge_test

import (
	"fmt"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// ExampleConvert_chatToAnthropic shows how to convert a Chat Completions
// request body to an Anthropic Messages request body.
func ExampleConvert_chatToAnthropic() {
	chatBody := []byte(`{"model":"claude-3-5-sonnet","messages":[{"role":"user","content":"hello"}]}`)

	// client=chat (the caller's format), upstream=anthropic (where you will send it).
	plan, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic, chatBody, bridge.Options{})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	// plan.Body is the Anthropic request body. Send it to Anthropic.
	// plan.Losses lists any fields that could not be expressed in the target format.
	fmt.Println("upstream:", plan.Upstream)
	fmt.Println("losses:", len(plan.Losses))
	// Output:
	// upstream: anthropic
	// losses: 0
}
