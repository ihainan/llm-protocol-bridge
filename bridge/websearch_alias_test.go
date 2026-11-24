package bridge_test

import (
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// All four Responses names of the hosted web search tool parse as the same
// builtin and convert like web_search.
func TestResponsesWebSearchDatedAliases(t *testing.T) {
	for _, typ := range []string{"web_search", "web_search_2025_08_26", "web_search_preview", "web_search_preview_2025_03_11"} {
		t.Run(typ, func(t *testing.T) {
			body := `{"model":"m","input":"hi","tools":[{"type":"` + typ + `","search_context_size":"low"}]}`
			req, err := bridge.Parse(ir.FormatResponses, []byte(body), bridge.Options{})
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			if len(req.Builtins) != 1 || req.Builtins[0].Kind != ir.BuiltinWebSearch {
				t.Fatalf("builtins: %+v", req.Builtins)
			}
			if !strings.Contains(string(req.Builtins[0].Config), `"search_context_size":"low"`) {
				t.Errorf("config lost: %s", req.Builtins[0].Config)
			}
			// the same entry inside additional_tools
			at := `{"model":"m","input":[{"type":"additional_tools","tools":[{"type":"` + typ + `"}]},{"type":"message","role":"user","content":"hi"}]}`
			req, err = bridge.Parse(ir.FormatResponses, []byte(at), bridge.Options{})
			if err != nil || len(req.Builtins) != 1 || req.Builtins[0].Kind != ir.BuiltinWebSearch {
				t.Fatalf("additional_tools: %v %+v", err, req)
			}
			// Responses to Responses keeps the client's own tool entry
			plan, err := bridge.Convert(ir.FormatResponses, ir.FormatResponses, []byte(body), bridge.Options{Model: "m"})
			if err != nil {
				t.Fatalf("Convert to Responses: %v", err)
			}
			if !strings.Contains(string(plan.Body), `"type":"`+typ+`"`) {
				t.Errorf("tool type not kept: %s", plan.Body)
			}
			// towards Anthropic it becomes Anthropic's server web search tool
			plan, err = bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, []byte(body), bridge.Options{Model: "m"})
			if err != nil {
				t.Fatalf("Convert to Anthropic: %v", err)
			}
			if !strings.Contains(string(plan.Body), `"web_search_20250305"`) {
				t.Errorf("no Anthropic web search tool: %s", plan.Body)
			}
		})
	}
}
