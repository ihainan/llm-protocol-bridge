package bridge_test

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/anthropic"
	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// cachePaths returns the JSON paths of every cache_control member in an
// Anthropic request body, in document order, with their values.
func cachePaths(t *testing.T, body []byte) []string {
	t.Helper()
	var v any
	if err := json.Unmarshal(body, &v); err != nil {
		t.Fatalf("invalid JSON: %v\n%s", err, body)
	}
	var out []string
	var walk func(path string, v any)
	walk = func(path string, v any) {
		switch x := v.(type) {
		case map[string]any:
			// Visit keys in a fixed order so paths are deterministic.
			for _, k := range []string{"tools", "system", "messages", "content"} {
				if c, ok := x[k]; ok {
					walk(path+"."+k, c)
				}
			}
			if cc, ok := x["cache_control"]; ok {
				b, _ := json.Marshal(cc)
				out = append(out, strings.TrimPrefix(path, ".")+"="+string(b))
			}
		case []any:
			for i, e := range x {
				walk(path+"["+itoa(i)+"]", e)
			}
		}
	}
	walk("", v)
	return out
}

func itoa(i int) string {
	b, _ := json.Marshal(i)
	return string(b)
}

func convertWith(t *testing.T, from, to ir.Format, body string, o bridge.Options) *bridge.Plan {
	t.Helper()
	plan, err := bridge.Convert(from, to, []byte(body), o)
	if err != nil {
		t.Fatalf("Convert(%s->%s): %v\nbody: %s", from, to, err, body)
	}
	return plan
}

func cacheLosses(p *bridge.Plan) []string {
	var out []string
	for _, l := range p.Losses {
		if l.Kind == ir.LossCacheMarker {
			out = append(out, l.Path.String())
		}
	}
	return out
}

func wantPaths(t *testing.T, label string, got, want []string) {
	t.Helper()
	if len(got) == 0 && len(want) == 0 {
		return
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("%s:\n got: %q\nwant: %q", label, got, want)
	}
}

const eph = `{"type":"ephemeral"}`

func TestChatCacheMarkers_ToAnthropic(t *testing.T) {
	body := `{"model":"m","messages":[
		{"role":"system","content":[{"type":"text","text":"long system","cache_control":{"type":"ephemeral","ttl":"1h"}}]},
		{"role":"user","content":[
			{"type":"image_url","image_url":{"url":"https://example.com/a.png"},"cache_control":{"type":"ephemeral"}},
			{"type":"file","file":{"filename":"a.pdf","file_data":"data:application/pdf;base64,JVBERi0="},"cache_control":{"type":"ephemeral"}},
			{"type":"text","text":"question"}]}],
		"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}},"cache_control":{"type":"ephemeral","ttl":"1h"}}]}`
	plan := convertWith(t, ir.FormatChat, ir.FormatAnthropic, body, opts())
	wantPaths(t, "markers", cachePaths(t, plan.Body), []string{
		`tools[0]={"ttl":"1h","type":"ephemeral"}`,
		`system[0]={"ttl":"1h","type":"ephemeral"}`,
		"messages[0].content[0]=" + eph,
		"messages[0].content[1]=" + eph,
	})
	wantPaths(t, "losses", cacheLosses(plan), nil)
}

func TestChatCacheMarkers_MessageLevel(t *testing.T) {
	body := `{"model":"m","messages":[
		{"role":"user","content":"plain string","cache_control":{"type":"ephemeral"}},
		{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}],"cache_control":{"type":"ephemeral"}},
		{"role":"tool","tool_call_id":"call_1","content":"result","cache_control":{"type":"ephemeral"}},
		{"role":"user","content":[{"type":"text","text":"a","cache_control":{"type":"ephemeral","ttl":"1h"}},{"type":"text","text":"b"}],"cache_control":{"type":"ephemeral"}}],
		"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}`
	plan := convertWith(t, ir.FormatChat, ir.FormatAnthropic, body, opts())
	// Five markers: user text, tool_use, tool_result, "a" (own 1h marker) and
	// "b" (message-level). The limit keeps the first and last three.
	// The 1h marker follows 5m ones, which Anthropic rejects, so it is
	// shortened to the 5m default.
	wantPaths(t, "markers", cachePaths(t, plan.Body), []string{
		"messages[0].content[0]=" + eph,
		"messages[2].content[0]=" + eph,
		"messages[2].content[1]=" + eph,
		"messages[2].content[2]=" + eph,
	})
	wantPaths(t, "losses", cacheLosses(plan), []string{
		"messages[1].content[0].cache_control",
		"messages[3].content[0].cache_control.ttl",
	})
}

func TestChatCacheMarkers_ToolMessageHoisted(t *testing.T) {
	body := `{"model":"m","messages":[
		{"role":"user","content":"q"},
		{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}},{"id":"call_2","type":"function","function":{"name":"f","arguments":"{}"}}]},
		{"role":"tool","tool_call_id":"call_1","content":[{"type":"text","text":"only","cache_control":{"type":"ephemeral"}}]},
		{"role":"tool","tool_call_id":"call_2","content":[{"type":"text","text":"r1"},{"type":"text","text":"r2","cache_control":{"type":"ephemeral","ttl":"5m"}}]}],
		"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}`
	plan := convertWith(t, ir.FormatChat, ir.FormatAnthropic, body, opts())
	wantPaths(t, "markers", cachePaths(t, plan.Body), []string{
		"messages[2].content[0]=" + eph,
		`messages[2].content[1]={"ttl":"5m","type":"ephemeral"}`,
	})
	// The single-text tool result still uses the plain string form.
	if !strings.Contains(string(plan.Body), `"tool_use_id":"call_1","content":"only"`) {
		t.Fatalf("expected string content for single-text tool result: %s", plan.Body)
	}
}

func TestChatCacheMarkers_InvalidShapesBecomeLosses(t *testing.T) {
	body := `{"model":"m","messages":[{"role":"user","content":[
		{"type":"text","text":"a","cache_control":"ephemeral"},
		{"type":"text","text":"b","cache_control":{"type":"persistent"}},
		{"type":"text","text":"c","cache_control":{"type":"ephemeral","ttl":"10m"}},
		{"type":"text","text":"d","cache_control":null}]}]}`
	plan := convertWith(t, ir.FormatChat, ir.FormatAnthropic, body, opts())
	wantPaths(t, "markers", cachePaths(t, plan.Body), nil)
	wantPaths(t, "losses", cacheLosses(plan), []string{
		"messages[0].content[0].cache_control",
		"messages[0].content[1].cache_control.type",
		"messages[0].content[2].cache_control.ttl",
	})
	// A caller that denies the loss gets an error instead.
	strict := opts()
	strict.Deny = ir.LossSet{ir.LossCacheMarker: true}
	if _, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic, []byte(body), strict); err == nil {
		t.Fatal("expected an error when cache.marker losses are denied")
	}
}

func TestCacheMarkers_LimitAndEmptyText(t *testing.T) {
	body := `{"model":"m","max_tokens":100,
		"tools":[{"name":"a","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral","ttl":"1h"}}],
		"system":[{"type":"text","text":"s","cache_control":{"type":"ephemeral"}}],
		"messages":[
			{"role":"user","content":[{"type":"text","text":"u1","cache_control":{"type":"ephemeral"}}]},
			{"role":"assistant","content":[{"type":"text","text":"a1","cache_control":{"type":"ephemeral"}}]},
			{"role":"user","content":[{"type":"text","text":"u2","cache_control":{"type":"ephemeral"}}]}]}`
	plan := convertWith(t, ir.FormatAnthropic, ir.FormatAnthropic, body, opts())
	wantPaths(t, "markers", cachePaths(t, plan.Body), []string{
		`tools[0]={"ttl":"1h","type":"ephemeral"}`,
		"messages[0].content[0]=" + eph,
		"messages[1].content[0]=" + eph,
		"messages[2].content[0]=" + eph,
	})
	wantPaths(t, "losses", cacheLosses(plan), []string{"system[0].cache_control"})

	// Anthropic rejects cache_control on empty text; Chat clients can send one.
	chatBody := `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"","cache_control":{"type":"ephemeral"}},{"type":"text","text":"u","cache_control":{"type":"ephemeral"}}]}]}`
	plan = convertWith(t, ir.FormatChat, ir.FormatAnthropic, chatBody, opts())
	wantPaths(t, "empty markers", cachePaths(t, plan.Body), []string{"messages[0].content[1]=" + eph})
	wantPaths(t, "empty losses", cacheLosses(plan), []string{"messages[0].content[0].cache_control"})
}

func TestAnthropicCacheMarkers_ToolBlocksPreserved(t *testing.T) {
	body := `{"model":"m","max_tokens":100,"messages":[
		{"role":"user","content":"q"},
		{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"f","input":{},"cache_control":{"type":"ephemeral"}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"only","cache_control":{"type":"ephemeral"}}],"cache_control":{"type":"ephemeral"}}]}],
		"tools":[{"name":"f","input_schema":{"type":"object"}}]}`
	plan := convertWith(t, ir.FormatAnthropic, ir.FormatAnthropic, body, opts())
	wantPaths(t, "markers", cachePaths(t, plan.Body), []string{
		"messages[1].content[0]=" + eph,
		"messages[2].content[0].content[0]=" + eph,
		"messages[2].content[0]=" + eph,
	})
}

func TestAnthropicCacheMarkers_BuiltinToolCountsTowardLimit(t *testing.T) {
	body := `{"model":"m","max_tokens":100,
		"tools":[{"name":"a","input_schema":{"type":"object"}},{"type":"web_search_20250305","name":"web_search","cache_control":{"type":"ephemeral"}}],
		"system":[{"type":"text","text":"s1","cache_control":{"type":"ephemeral"}},{"type":"text","text":"s2","cache_control":{"type":"ephemeral"}}],
		"messages":[{"role":"user","content":[{"type":"text","text":"u1","cache_control":{"type":"ephemeral"}},{"type":"text","text":"u2","cache_control":{"type":"ephemeral"}}]}]}`
	plan := convertWith(t, ir.FormatAnthropic, ir.FormatAnthropic, body, opts())
	wantPaths(t, "markers", cachePaths(t, plan.Body), []string{
		"tools[1]=" + eph,
		"system[1]=" + eph,
		"messages[0].content[0]=" + eph,
		"messages[0].content[1]=" + eph,
	})
	wantPaths(t, "losses", cacheLosses(plan), []string{"system[0].cache_control"})

	// Dropping the builtin's marker keeps the rest of its members in order.
	body2 := strings.Replace(body, `"system":[`, `"system":[{"type":"text","text":"s0","cache_control":{"type":"ephemeral"}},`, 1)
	body2 = strings.Replace(body2, `{"name":"a","input_schema":{"type":"object"}}`, `{"name":"a","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}}`, 1)
	plan2 := convertWith(t, ir.FormatAnthropic, ir.FormatAnthropic, body2, opts())
	if !strings.Contains(string(plan2.Body), `{"type":"web_search_20250305","name":"web_search"}`) {
		t.Fatalf("expected builtin without cache_control: %s", plan2.Body)
	}
}

// Formats without prompt cache markers keep v0.1.4 behaviour: the markers are
// dropped, recorded as losses, and the output equals the unmarked request's.
func TestChatCacheMarkers_OtherUpstreamsRecordLoss(t *testing.T) {
	marked := `{"model":"m","messages":[
		{"role":"system","content":[{"type":"text","text":"sys","cache_control":{"type":"ephemeral"}}]},
		{"role":"user","content":[{"type":"text","text":"u","cache_control":{"type":"ephemeral"}}]},
		{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}],"cache_control":{"type":"ephemeral"}},
		{"role":"tool","tool_call_id":"call_1","content":"r","cache_control":{"type":"ephemeral"}}],
		"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}},"cache_control":{"type":"ephemeral"}}]}`
	plain := `{"model":"m","messages":[
		{"role":"system","content":[{"type":"text","text":"sys"}]},
		{"role":"user","content":[{"type":"text","text":"u"}]},
		{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]},
		{"role":"tool","tool_call_id":"call_1","content":"r"}],
		"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}`
	for _, to := range []ir.Format{ir.FormatChat, ir.FormatResponses, ir.FormatGemini} {
		pm := convertWith(t, ir.FormatChat, to, marked, opts())
		pp := convertWith(t, ir.FormatChat, to, plain, opts())
		if string(pm.Body) != string(pp.Body) {
			t.Errorf("%s: marked body differs from plain\nmarked: %s\nplain:  %s", to, pm.Body, pp.Body)
		}
		if n := len(cacheLosses(pm)); n != 5 {
			t.Errorf("%s: expected 5 cache.marker losses, got %d: %v", to, n, cacheLosses(pm))
		}
	}
}

func autoOpts(mode anthropic.AutoCacheMode) bridge.Options {
	o := opts()
	o.Anthropic = anthropic.Dialect{AutoCache: mode}
	return o
}

const agentLoop = `{"model":"m","messages":[
	{"role":"system","content":"You are a coding agent."},
	{"role":"user","content":"fix the bug"},
	{"role":"assistant","content":"Looking.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"read","arguments":"{\"p\":\"a\"}"}}]},
	{"role":"tool","tool_call_id":"call_1","content":"file body"},
	{"role":"assistant","content":null,"tool_calls":[{"id":"call_2","type":"function","function":{"name":"read","arguments":"{\"p\":\"b\"}"}}]},
	{"role":"tool","tool_call_id":"call_2","content":"other body"}],
	"tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object"}}},{"type":"function","function":{"name":"write","parameters":{"type":"object"}}}]}`

func TestAutoCache_Prefix(t *testing.T) {
	plan := convertWith(t, ir.FormatChat, ir.FormatAnthropic, agentLoop, autoOpts(anthropic.AutoCachePrefix))
	wantPaths(t, "markers", cachePaths(t, plan.Body), []string{
		"tools[1]=" + eph,
		"system[0]=" + eph,
	})
	if !strings.Contains(string(plan.Body), `"system":[{"type":"text","text":"You are a coding agent.","cache_control":{"type":"ephemeral"}}]`) {
		t.Fatalf("expected system array form with marker: %s", plan.Body)
	}
	wantPaths(t, "losses", cacheLosses(plan), nil)
}

func TestAutoCache_Conversation(t *testing.T) {
	plan := convertWith(t, ir.FormatChat, ir.FormatAnthropic, agentLoop, autoOpts(anthropic.AutoCacheConversation))
	wantPaths(t, "markers", cachePaths(t, plan.Body), []string{
		"tools[1]=" + eph,
		"system[0]=" + eph,
		"messages[2].content[0]=" + eph,
		"messages[4].content[0]=" + eph,
	})

	// The previous request in the same loop ended where this one's earlier
	// user-message marker sits, so its final breakpoint is read back exactly.
	prev := `{"model":"m","messages":[
	{"role":"system","content":"You are a coding agent."},
	{"role":"user","content":"fix the bug"},
	{"role":"assistant","content":"Looking.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"read","arguments":"{\"p\":\"a\"}"}}]},
	{"role":"tool","tool_call_id":"call_1","content":"file body"}],
	"tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object"}}},{"type":"function","function":{"name":"write","parameters":{"type":"object"}}}]}`
	pp := convertWith(t, ir.FormatChat, ir.FormatAnthropic, prev, autoOpts(anthropic.AutoCacheConversation))
	wantPaths(t, "prev markers", cachePaths(t, pp.Body), []string{
		"tools[1]=" + eph,
		"system[0]=" + eph,
		"messages[0].content[0]=" + eph,
		"messages[2].content[0]=" + eph,
	})
	prefixOf := func(body []byte) string {
		var v struct {
			Messages []json.RawMessage `json:"messages"`
		}
		_ = json.Unmarshal(body, &v)
		return string(v.Messages[2])
	}
	if prefixOf(pp.Body) != prefixOf(plan.Body) {
		t.Fatalf("breakpoint block differs between turns:\nprev: %s\ncur:  %s", prefixOf(pp.Body), prefixOf(plan.Body))
	}
}

func TestAutoCache_SingleTurnAndSkips(t *testing.T) {
	single := `{"model":"m","messages":[{"role":"user","content":"hi"}]}`
	p := convertWith(t, ir.FormatChat, ir.FormatAnthropic, single, autoOpts(anthropic.AutoCachePrefix))
	wantPaths(t, "prefix single", cachePaths(t, p.Body), nil)
	p = convertWith(t, ir.FormatChat, ir.FormatAnthropic, single, autoOpts(anthropic.AutoCacheConversation))
	wantPaths(t, "conversation single", cachePaths(t, p.Body), []string{"messages[0].content[0]=" + eph})

	// Thinking blocks and empty text never receive a marker.
	thinking := `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"q"},{"role":"assistant","content":[{"type":"text","text":"ok"},{"type":"thinking","thinking":"t","signature":"sig"}]}]}`
	p = convertWith(t, ir.FormatAnthropic, ir.FormatAnthropic, thinking, autoOpts(anthropic.AutoCacheConversation))
	wantPaths(t, "thinking", cachePaths(t, p.Body), []string{
		"messages[0].content[0]=" + eph,
		"messages[1].content[0]=" + eph,
	})

	// Any client marker disables automatic placement, including one the
	// builder drops (a folded mid-conversation system message).
	clientMarked := `{"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":"u1"},{"role":"assistant","content":"a1"},{"role":"system","content":[{"type":"text","text":"mid","cache_control":{"type":"ephemeral"}}]},{"role":"user","content":"u2"}]}`
	o := autoOpts(anthropic.AutoCacheConversation)
	o.Anthropic.MidSystem = "fold"
	p = convertWith(t, ir.FormatChat, ir.FormatAnthropic, clientMarked, o)
	wantPaths(t, "client marked", cachePaths(t, p.Body), nil)
	wantPaths(t, "client marked losses", cacheLosses(p), []string{"messages[2].content[0].cache_control"})

	if _, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic, []byte(single), autoOpts("always")); err == nil {
		t.Fatal("expected an error for an unknown AutoCache mode")
	}
}

// AutoCache only affects Anthropic upstreams.
func TestAutoCache_OtherUpstreamsUnchanged(t *testing.T) {
	for _, to := range []ir.Format{ir.FormatChat, ir.FormatResponses, ir.FormatGemini} {
		on := convertWith(t, ir.FormatChat, to, agentLoop, autoOpts(anthropic.AutoCacheConversation))
		off := convertWith(t, ir.FormatChat, to, agentLoop, opts())
		if string(on.Body) != string(off.Body) || len(on.Losses) != len(off.Losses) {
			t.Errorf("%s: AutoCache changed a non-Anthropic conversion", to)
		}
	}
}

func TestCacheMarkers_TTLOrder(t *testing.T) {
	ok := `{"model":"m","messages":[{"role":"system","content":[{"type":"text","text":"s","cache_control":{"type":"ephemeral","ttl":"1h"}}]},{"role":"user","content":[{"type":"text","text":"u","cache_control":{"type":"ephemeral","ttl":"5m"}}]}],
		"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}},"cache_control":{"type":"ephemeral","ttl":"1h"}}]}`
	plan := convertWith(t, ir.FormatChat, ir.FormatAnthropic, ok, opts())
	wantPaths(t, "ordered", cachePaths(t, plan.Body), []string{
		`tools[0]={"ttl":"1h","type":"ephemeral"}`,
		`system[0]={"ttl":"1h","type":"ephemeral"}`,
		`messages[0].content[0]={"ttl":"5m","type":"ephemeral"}`,
	})
	wantPaths(t, "ordered losses", cacheLosses(plan), nil)

	bad := `{"model":"m","max_tokens":10,"tools":[{"name":"f","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}},{"type":"web_search_20250305","name":"web_search","cache_control":{"type":"ephemeral","ttl":"1h"}}],
		"system":[{"type":"text","text":"s","cache_control":{"type":"ephemeral","ttl":"1h"}}],"messages":[{"role":"user","content":"u"}]}`
	plan = convertWith(t, ir.FormatAnthropic, ir.FormatAnthropic, bad, opts())
	wantPaths(t, "reordered", cachePaths(t, plan.Body), []string{
		"tools[0]=" + eph,
		"system[0]=" + eph,
	})
	wantPaths(t, "reordered losses", cacheLosses(plan), []string{
		"tools[1].cache_control.ttl",
		"system[0].cache_control.ttl",
	})
}
