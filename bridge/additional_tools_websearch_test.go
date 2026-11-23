package bridge_test

import (
	"bytes"
	"encoding/json"
	"io"
	"strings"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/anthropic"
	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/chat"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

// codexAdditionalToolsRequest mirrors a Codex-style Responses request that omits
// top-level tools and declares namespaced custom + function tools via an
// input[0] item of type "additional_tools".
const codexAdditionalToolsRequest = `{
  "model": "gpt-6.1-test",
  "stream": true,
  "store": false,
  "tool_choice": "auto",
  "parallel_tool_calls": false,
  "include": ["reasoning.encrypted_content"],
  "reasoning": {"effort": "low", "context": "all_turns"},
  "text": {"verbosity": "low"},
  "prompt_cache_key": "sess-001",
  "client_metadata": {"turn_id": "t-1"},
  "input": [
    {
      "type": "additional_tools",
      "id": "at_001",
      "role": "developer",
      "tools": [
        {
          "type": "namespace",
          "name": "functions",
          "description": "",
          "tools": [
            {
              "type": "custom",
              "name": "exec",
              "description": "Run a script",
              "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"}
            },
            {
              "type": "function",
              "name": "wait",
              "description": "Wait for background task",
              "strict": false,
              "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}
            }
          ]
        },
        {
          "type": "namespace",
          "name": "clock",
          "tools": [
            {
              "type": "function",
              "name": "sleep",
              "description": "Sleep for N seconds",
              "strict": true,
              "parameters": {"type": "object", "properties": {"seconds": {"type": "integer"}}, "required": ["seconds"], "additionalProperties": false}
            }
          ]
        },
        {
          "type": "namespace",
          "name": "collaboration",
          "tools": [
            {
              "type": "function",
              "name": "spawn_agent",
              "description": "Spawn a helper agent",
              "strict": false,
              "parameters": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]}
            }
          ]
        }
      ]
    },
    {
      "type": "message",
      "id": "msg_dev_1",
      "role": "developer",
      "content": [{"type": "input_text", "text": "You are a coding assistant."}]
    },
    {
      "type": "message",
      "id": "msg_user_1",
      "role": "user",
      "content": [{"type": "input_text", "text": "Run a quick command and sleep."}]
    }
  ]
}`

func TestResponsesAdditionalTools_AllDirectionsAndRoundTrip(t *testing.T) {
	targets := []struct {
		name      string
		upstream  ir.Format
		checkBody func(t *testing.T, body []byte) map[string]string // returns logical "ns/name" -> wireName
		nsResp    func(wireExec, wireSleep string) []byte
		sseResp   func(wireExec, wireSleep string) []byte
	}{
		{
			name:     "responses_to_chat",
			upstream: ir.FormatChat,
			checkBody: func(t *testing.T, body []byte) map[string]string {
				var req struct {
					Messages []struct {
						Role string `json:"role"`
					} `json:"messages"`
					Tools []struct {
						Type     string `json:"type"`
						Function struct {
							Name string `json:"name"`
						} `json:"function"`
					} `json:"tools"`
				}
				if err := json.Unmarshal(body, &req); err != nil {
					t.Fatalf("unmarshal chat body: %v", err)
				}
				if len(req.Tools) != 4 {
					t.Fatalf("expected 4 tools, got %d (%s)", len(req.Tools), body)
				}
				if len(req.Messages) < 2 || req.Messages[0].Role != "system" {
					t.Fatalf("leading developer message should become system instruction after additional_tools: %s", body)
				}
				m := map[string]string{}
				for _, tl := range req.Tools {
					m[tl.Function.Name] = tl.Function.Name
				}
				return m
			},
			nsResp: func(wireExec, wireSleep string) []byte {
				return []byte(`{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-6.1-test","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"` + wireExec + `","arguments":"{\"input\":\"console.log(1)\"}"}},{"id":"call_2","type":"function","function":{"name":"` + wireSleep + `","arguments":"{\"seconds\":3}"}}]}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)
			},
			sseResp: func(wireExec, wireSleep string) []byte {
				return []byte(
					"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-6.1-test\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"" + wireExec + "\",\"arguments\":\"{\\\"input\\\":\\\"console.log(1)\\\"}\"}}]}}]}\n\n" +
						"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-6.1-test\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_2\",\"type\":\"function\",\"function\":{\"name\":\"" + wireSleep + "\",\"arguments\":\"{\\\"seconds\\\":3}\"}}]}}]}\n\n" +
						"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-6.1-test\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n" +
						"data: [DONE]\n\n")
			},
		},
		{
			name:     "responses_to_anthropic",
			upstream: ir.FormatAnthropic,
			checkBody: func(t *testing.T, body []byte) map[string]string {
				var req struct {
					System json.RawMessage `json:"system"`
					Tools  []struct {
						Name string `json:"name"`
					} `json:"tools"`
				}
				if err := json.Unmarshal(body, &req); err != nil {
					t.Fatalf("unmarshal anthropic body: %v", err)
				}
				if len(req.Tools) != 4 {
					t.Fatalf("expected 4 tools, got %d (%s)", len(req.Tools), body)
				}
				if !strings.Contains(string(req.System), "coding assistant") {
					t.Fatalf("expected developer instruction in anthropic system: %s", body)
				}
				m := map[string]string{}
				for _, tl := range req.Tools {
					m[tl.Name] = tl.Name
				}
				return m
			},
			nsResp: func(wireExec, wireSleep string) []byte {
				return []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","stop_reason":"tool_use","content":[{"type":"tool_use","id":"call_1","name":"` + wireExec + `","input":{"input":"console.log(1)"}},{"type":"tool_use","id":"call_2","name":"` + wireSleep + `","input":{"seconds":3}}],"usage":{"input_tokens":10,"output_tokens":5}}`)
			},
			sseResp: func(wireExec, wireSleep string) []byte {
				return []byte(
					"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-test\",\"content\":[],\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n" +
						"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"" + wireExec + "\",\"input\":{}}}\n\n" +
						"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"input\\\":\\\"console.log(1)\\\"}\"}}\n\n" +
						"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
						"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_2\",\"name\":\"" + wireSleep + "\",\"input\":{}}}\n\n" +
						"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"seconds\\\":3}\"}}\n\n" +
						"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n" +
						"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":5}}\n\n" +
						"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
			},
		},
		{
			name:     "responses_to_gemini",
			upstream: ir.FormatGemini,
			checkBody: func(t *testing.T, body []byte) map[string]string {
				var req struct {
					Tools []struct {
						FunctionDeclarations []struct {
							Name string `json:"name"`
						} `json:"functionDeclarations"`
					} `json:"tools"`
				}
				if err := json.Unmarshal(body, &req); err != nil {
					t.Fatalf("unmarshal gemini body: %v", err)
				}
				if len(req.Tools) != 1 || len(req.Tools[0].FunctionDeclarations) != 4 {
					t.Fatalf("expected 4 functionDeclarations, got %s", body)
				}
				m := map[string]string{}
				for _, fd := range req.Tools[0].FunctionDeclarations {
					m[fd.Name] = fd.Name
				}
				return m
			},
			nsResp: func(wireExec, wireSleep string) []byte {
				return []byte(`{"candidates":[{"index":0,"finishReason":"STOP","content":{"role":"model","parts":[{"functionCall":{"name":"` + wireExec + `","args":{"input":"console.log(1)"}}},{"functionCall":{"name":"` + wireSleep + `","args":{"seconds":3}}}]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}`)
			},
			sseResp: func(wireExec, wireSleep string) []byte {
				return []byte(
					"data: {\"candidates\":[{\"index\":0,\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"name\":\"" + wireExec + "\",\"args\":{\"input\":\"console.log(1)\"}}}]}}]}\n\n" +
						"data: {\"candidates\":[{\"index\":0,\"finishReason\":\"STOP\",\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"name\":\"" + wireSleep + "\",\"args\":{\"seconds\":3}}}]}}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":5,\"totalTokenCount\":15}}\n\n")
			},
		},
	}

	assertResponsesOutput := func(t *testing.T, label string, output []map[string]any) {
		t.Helper()
		var sawExec, sawSleep bool
		for _, item := range output {
			switch item["type"] {
			case "custom_tool_call":
				if item["name"] == "exec" && item["namespace"] == "functions" && item["input"] == "console.log(1)" {
					sawExec = true
				}
			case "function_call":
				if item["name"] == "sleep" && item["namespace"] == "clock" {
					sawSleep = true
				}
			}
		}
		if !sawExec || !sawSleep {
			b, _ := json.Marshal(output)
			t.Fatalf("%s: expected custom_tool_call(functions/exec) and function_call(clock/sleep), got %s", label, b)
		}
	}

	for _, tc := range targets {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := bridge.Convert(ir.FormatResponses, tc.upstream, []byte(codexAdditionalToolsRequest), bridge.Options{})
			if err != nil {
				t.Fatalf("Convert failed: %v", err)
			}
			wireSet := tc.checkBody(t, plan.Body)
			// Find wire names for exec (in functions) and sleep (in clock)
			var wireExec, wireSleep string
			for w := range wireSet {
				if strings.HasSuffix(w, "exec") {
					wireExec = w
				}
				if strings.HasSuffix(w, "sleep") {
					wireSleep = w
				}
			}
			if wireExec == "" || wireSleep == "" {
				t.Fatalf("missing wire names in %v", wireSet)
			}

			// 1. Non-stream round-trip
			nsOut, _, err := plan.ConvertResponse(tc.nsResp(wireExec, wireSleep))
			if err != nil {
				t.Fatalf("ConvertResponse failed: %v", err)
			}
			var nsObj struct {
				Output []map[string]any `json:"output"`
			}
			if err := json.Unmarshal(nsOut, &nsObj); err != nil {
				t.Fatalf("unmarshal non-stream Responses output: %v", err)
			}
			assertResponsesOutput(t, tc.name+"/nonstream", nsObj.Output)

			// 2. Stream round-trip
			st := plan.NewStream()
			var streamOut []byte
			sr := sse.NewReader(bytes.NewReader(tc.sseResp(wireExec, wireSleep)))
			for {
				ev, err := sr.Next()
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Fatalf("sse.Next failed: %v", err)
				}
				streamOut, err = st.Feed(streamOut, ev)
				if err != nil {
					t.Fatalf("stream Feed failed: %v", err)
				}
			}
			var endErr error
			streamOut, endErr = st.End(streamOut, ir.EndEOF)
			if endErr != nil {
				t.Fatalf("stream End failed: %v", endErr)
			}
			var completedOutput []map[string]any
			for _, line := range strings.Split(string(streamOut), "\n") {
				if !strings.HasPrefix(line, "data: ") {
					continue
				}
				var ev struct {
					Type     string `json:"type"`
					Response struct {
						Output []map[string]any `json:"output"`
					} `json:"response"`
				}
				if json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &ev) == nil && ev.Type == "response.completed" {
					completedOutput = ev.Response.Output
				}
			}
			assertResponsesOutput(t, tc.name+"/stream", completedOutput)
		})
	}
}

func TestResponsesAdditionalTools_PrecedenceAndValidation(t *testing.T) {
	// 1. Later declaration wins across top-level tools and additional_tools items (replacing in place).
	t.Run("later_additional_tools_overrides_top_level_and_earlier_block", func(t *testing.T) {
		body := `{
		  "model": "m",
		  "tools": [
		    {"type": "function", "name": "ping", "description": "from-top-level", "parameters": {"type": "object"}},
		    {"type": "namespace", "name": "ns1", "tools": [
		      {"type": "function", "name": "alpha", "description": "top-ns1", "parameters": {"type": "object"}}
		    ]}
		  ],
		  "tool_choice": {"type": "function", "name": "beta", "namespace": "ns2"},
		  "input": [
		    {"type": "additional_tools", "id": "at_1", "role": "developer", "tools": [
		      {"type": "function", "name": "ping", "description": "from-at1", "parameters": {"type": "object"}},
		      {"type": "namespace", "name": "ns1", "tools": [
		        {"type": "function", "name": "alpha", "description": "at1-ns1", "parameters": {"type": "object"}}
		      ]},
		      {"type": "namespace", "name": "ns2", "tools": [
		        {"type": "function", "name": "beta", "description": "at1-ns2", "parameters": {"type": "object"}}
		      ]}
		    ]},
		    {"type": "additional_tools", "id": "at_2", "role": "developer", "tools": [
		      {"type": "function", "name": "ping", "description": "from-at2", "parameters": {"type": "object"}}
		    ]},
		    {"type": "message", "role": "user", "content": "hi"}
		  ]
		}`
		req, err := bridge.Parse(ir.FormatResponses, []byte(body), bridge.Options{})
		if err != nil {
			t.Fatalf("Parse failed: %v", err)
		}
		if len(req.Tools) != 3 {
			t.Fatalf("expected 3 merged tools, got %d: %+v", len(req.Tools), req.Tools)
		}
		if req.Tools[0].Name != "ping" || req.Tools[0].Description != "from-at2" {
			t.Fatalf("expected ping overridden in-place by at_2, got %+v", req.Tools[0])
		}
		if req.Tools[1].Namespace != "ns1" || req.Tools[1].Name != "alpha" || req.Tools[1].Description != "at1-ns1" {
			t.Fatalf("expected ns1/alpha overridden in-place by at_1, got %+v", req.Tools[1])
		}
		if req.Tools[2].Namespace != "ns2" || req.Tools[2].Name != "beta" {
			t.Fatalf("expected ns2/beta appended, got %+v", req.Tools[2])
		}
	})

	// 2. Same tool name in different namespaces are distinct and both kept.
	t.Run("same_name_different_namespaces_both_kept", func(t *testing.T) {
		body := `{
		  "model": "m",
		  "tools": [{"type": "function", "name": "run", "description": "bare", "parameters": {"type": "object"}}],
		  "input": [
		    {"type": "additional_tools", "tools": [
		      {"type": "namespace", "name": "functions", "tools": [
		        {"type": "function", "name": "run", "description": "ns-run", "parameters": {"type": "object"}}
		      ]}
		    ]},
		    {"type": "message", "role": "user", "content": "hi"}
		  ]
		}`
		req, err := bridge.Parse(ir.FormatResponses, []byte(body), bridge.Options{})
		if err != nil {
			t.Fatalf("Parse failed: %v", err)
		}
		if len(req.Tools) != 2 {
			t.Fatalf("expected 2 distinct tools across namespaces, got %d", len(req.Tools))
		}
	})

	// 3. Duplicate function/custom name within a single additional_tools block is rejected 400,
	//    while duplicate/aliased builtins (e.g. web_search + web_search_preview) are accepted
	//    to preserve exact v0.1.0 compatibility.
	t.Run("duplicate_within_single_additional_tools_rejected", func(t *testing.T) {
		body := `{
		  "model": "m",
		  "input": [
		    {"type": "additional_tools", "tools": [
		      {"type": "function", "name": "dup", "parameters": {"type": "object"}},
		      {"type": "function", "name": "dup", "parameters": {"type": "object"}}
		    ]},
		    {"type": "message", "role": "user", "content": "hi"}
		  ]
		}`
		if _, err := bridge.Parse(ir.FormatResponses, []byte(body), bridge.Options{}); err == nil {
			t.Fatalf("expected error for duplicate tool inside same additional_tools block")
		}
		aliasedBuiltin := `{
		  "model": "m",
		  "input": [
		    {"type": "additional_tools", "tools": [
		      {"type": "web_search", "external_web_access": false},
		      {"type": "web_search_preview", "external_web_access": false}
		    ]},
		    {"type": "message", "role": "user", "content": "hi"}
		  ]
		}`
		req, err := bridge.Parse(ir.FormatResponses, []byte(aliasedBuiltin), bridge.Options{})
		if err != nil {
			t.Fatalf("expected aliased web_search builtins to be accepted: %v", err)
		}
		if len(req.Builtins) != 2 {
			t.Fatalf("expected both builtin entries preserved, got %d", len(req.Builtins))
		}
	})

	// 4. Null/omitted tools in additional_tools is a no-op; non-array tools is 400.
	t.Run("null_or_invalid_tools_in_additional_tools", func(t *testing.T) {
		okBody := `{"model":"m","input":[{"type":"additional_tools","id":"at_empty","role":"developer","tools":null},{"type":"message","role":"user","content":"hi"}]}`
		if _, err := bridge.Parse(ir.FormatResponses, []byte(okBody), bridge.Options{}); err != nil {
			t.Fatalf("null additional_tools.tools should succeed: %v", err)
		}
		badBody := `{"model":"m","input":[{"type":"additional_tools","tools":"not-an-array"},{"type":"message","role":"user","content":"hi"}]}`
		if _, err := bridge.Parse(ir.FormatResponses, []byte(badBody), bridge.Options{}); err == nil {
			t.Fatalf("non-array additional_tools.tools should fail")
		}
	})
}

func TestAnthropicDisableWebSearch(t *testing.T) {
	// 1. Responses -> Anthropic with web_search + function tool
	reqBody := `{
	  "model": "claude-sonnet-4-6",
	  "tool_choice": "auto",
	  "tools": [
	    {"type": "web_search"},
	    {"type": "function", "name": "read_file", "description": "Read file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}
	  ],
	  "input": "search and read"
	}`

	// Default (DisableWebSearch=false): web_search_20250305 is emitted.
	planOn, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, []byte(reqBody), bridge.Options{})
	if err != nil {
		t.Fatalf("default Convert failed: %v", err)
	}
	if !strings.Contains(string(planOn.Body), `"web_search_20250305"`) {
		t.Fatalf("expected web_search_20250305 when DisableWebSearch=false: %s", planOn.Body)
	}

	// DisableWebSearch=true: web_search dropped, function tool kept, LossWebSearch recorded.
	var seenLosses []ir.Loss
	planOff, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, []byte(reqBody), bridge.Options{
		Anthropic: anthropic.Dialect{DisableWebSearch: true},
		OnLoss:    func(l ir.Loss) { seenLosses = append(seenLosses, l) },
	})
	if err != nil {
		t.Fatalf("DisableWebSearch Convert failed: %v", err)
	}
	if strings.Contains(string(planOff.Body), "web_search") {
		t.Fatalf("expected web_search stripped when DisableWebSearch=true: %s", planOff.Body)
	}
	if !strings.Contains(string(planOff.Body), `"read_file"`) {
		t.Fatalf("expected regular function tool preserved: %s", planOff.Body)
	}
	hasLoss := false
	for _, l := range planOff.Losses {
		if l.Kind == ir.LossWebSearch {
			if l.Path.String() != "tools" {
				t.Fatalf("expected LossWebSearch path \"tools\", got %q", l.Path.String())
			}
			hasLoss = true
		}
	}
	if !hasLoss {
		t.Fatalf("expected ir.LossWebSearch in plan.Losses, got %+v", planOff.Losses)
	}
	hasCallbackLoss := false
	for _, l := range seenLosses {
		if l.Kind == ir.LossWebSearch {
			hasCallbackLoss = true
		}
	}
	if !hasCallbackLoss {
		t.Fatalf("expected ir.LossWebSearch passed to OnLoss callback")
	}

	// Deny: LossSet{LossWebSearch: true} rejects it.
	if _, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, []byte(reqBody), bridge.Options{
		Anthropic: anthropic.Dialect{DisableWebSearch: true},
		Deny:      ir.LossSet{ir.LossWebSearch: true},
	}); err == nil {
		t.Fatalf("expected Deny[LossWebSearch] to reject dropped web_search")
	}

	// 2. Only web_search + tool_choice="auto" -> both tools and tool_choice omitted cleanly.
	onlyWS := `{"model":"claude-sonnet-4-6","tool_choice":"auto","parallel_tool_calls":false,"tools":[{"type":"web_search"}],"input":"hi"}`
	planOnlyWS, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, []byte(onlyWS), bridge.Options{
		Anthropic: anthropic.Dialect{DisableWebSearch: true},
	})
	if err != nil {
		t.Fatalf("only-web_search auto Convert failed: %v", err)
	}
	if strings.Contains(string(planOnlyWS.Body), `"tools"`) || strings.Contains(string(planOnlyWS.Body), `"tool_choice"`) {
		t.Fatalf("expected both tools and tool_choice omitted when only web_search was dropped: %s", planOnlyWS.Body)
	}

	// 3. Only web_search + tool_choice="required" or specific tool -> explicit unsupported error.
	onlyWSReq := `{"model":"claude-sonnet-4-6","tool_choice":"required","tools":[{"type":"web_search"}],"input":"hi"}`
	if _, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, []byte(onlyWSReq), bridge.Options{
		Anthropic: anthropic.Dialect{DisableWebSearch: true},
	}); err == nil {
		t.Fatalf("expected error when tool_choice=required and only web_search was dropped")
	}
	directReq := &ir.Request{
		Model:      "claude-sonnet-4-6",
		Messages:   []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("hi")}}},
		Builtins:   []ir.BuiltinTool{{Kind: ir.BuiltinWebSearch, Origin: ir.FormatResponses}},
		ToolChoice: ir.ToolChoice{Mode: ir.ToolChoiceTool, Name: "web_search"},
	}
	if _, err := bridge.Build(directReq, ir.FormatResponses, ir.FormatAnthropic, bridge.Options{
		Anthropic: anthropic.Dialect{DisableWebSearch: true},
	}); err == nil {
		t.Fatalf("expected error from bridge.Build when ToolChoiceTool targets dropped web_search and no tools remain")
	}

	// 4. Anthropic-origin web_search_20250305 is also stripped when DisableWebSearch=true.
	anthReq := `{"model":"claude-sonnet-4-6","max_tokens":256,"tools":[{"type":"web_search_20250305","name":"web_search"}],"messages":[{"role":"user","content":"hi"}]}`
	planAnth, err := bridge.Convert(ir.FormatAnthropic, ir.FormatAnthropic, []byte(anthReq), bridge.Options{
		Anthropic: anthropic.Dialect{DisableWebSearch: true},
	})
	if err != nil {
		t.Fatalf("Anthropic->Anthropic DisableWebSearch failed: %v", err)
	}
	if strings.Contains(string(planAnth.Body), "web_search") {
		t.Fatalf("expected Anthropic-origin web_search_20250305 dropped: %s", planAnth.Body)
	}

	// 5. When DisableWebSearch=true drops web_search while other function tools remain,
	//    forcing ToolChoiceTool to the dropped web_search (by Name or Builtin) fails,
	//    unless a surviving regular function tool has Name=="web_search".
	mixedName := &ir.Request{
		Model:      "claude-sonnet-4-6",
		Messages:   []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("hi")}}},
		Tools:      []ir.Tool{{Kind: ir.ToolFunction, Name: "read_file"}},
		Builtins:   []ir.BuiltinTool{{Kind: ir.BuiltinWebSearch, Origin: ir.FormatResponses}},
		ToolChoice: ir.ToolChoice{Mode: ir.ToolChoiceTool, Name: "web_search"},
	}
	if _, err := bridge.Build(mixedName, ir.FormatResponses, ir.FormatAnthropic, bridge.Options{
		Anthropic: anthropic.Dialect{DisableWebSearch: true},
	}); err == nil {
		t.Fatalf("expected error when ToolChoiceTool(Name=web_search) targets dropped web_search alongside function tools")
	}
	mixedBuiltin := &ir.Request{
		Model:      "claude-sonnet-4-6",
		Messages:   []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("hi")}}},
		Tools:      []ir.Tool{{Kind: ir.ToolFunction, Name: "web_search"}},
		Builtins:   []ir.BuiltinTool{{Kind: ir.BuiltinWebSearch, Origin: ir.FormatResponses}},
		ToolChoice: ir.ToolChoice{Mode: ir.ToolChoiceTool, Builtin: ir.BuiltinWebSearch},
	}
	if _, err := bridge.Build(mixedBuiltin, ir.FormatResponses, ir.FormatAnthropic, bridge.Options{
		Anthropic: anthropic.Dialect{DisableWebSearch: true},
	}); err == nil {
		t.Fatalf("expected error when ToolChoiceTool(Builtin=web_search) targets dropped builtin")
	}
	survivingFn := &ir.Request{
		Model:      "claude-sonnet-4-6",
		Messages:   []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("hi")}}},
		Tools:      []ir.Tool{{Kind: ir.ToolFunction, Name: "web_search"}},
		Builtins:   []ir.BuiltinTool{{Kind: ir.BuiltinWebSearch, Origin: ir.FormatResponses}},
		ToolChoice: ir.ToolChoice{Mode: ir.ToolChoiceTool, Name: "web_search"},
	}
	if planSurv, err := bridge.Build(survivingFn, ir.FormatResponses, ir.FormatAnthropic, bridge.Options{
		Anthropic: anthropic.Dialect{DisableWebSearch: true},
	}); err != nil {
		t.Fatalf("expected surviving regular function named web_search to succeed: %v", err)
	} else if !strings.Contains(string(planSurv.Body), `"tool_choice":{"type":"tool","name":"web_search"}`) {
		t.Fatalf("expected tool_choice preserved for surviving regular function: %s", planSurv.Body)
	}

	// Namespaced regular function named "web_search" selected by bare name (both unique and wire-name collision).
	nsReq := `{"model":"claude-sonnet-4-6","input":"hi","tools":[{"type":"web_search"},{"type":"namespace","name":"ns","tools":[{"type":"function","name":"web_search","parameters":{"type":"object"}}]}],"tool_choice":{"type":"function","name":"web_search"}}`
	if planNS, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, []byte(nsReq), bridge.Options{
		Anthropic: anthropic.Dialect{DisableWebSearch: true},
	}); err != nil {
		t.Fatalf("expected namespaced web_search selected by bare name to succeed: %v", err)
	} else if !strings.Contains(string(planNS.Body), `"tool_choice":{"type":"tool","name":"ns__web_search"}`) {
		t.Fatalf("expected resolved wire name ns__web_search: %s", planNS.Body)
	}
	nsCollisionReq := `{"model":"claude-sonnet-4-6","input":"hi","tools":[{"type":"web_search"},{"type":"function","name":"ns__web_search","parameters":{"type":"object"}},{"type":"namespace","name":"ns","tools":[{"type":"function","name":"web_search","parameters":{"type":"object"}}]}],"tool_choice":{"type":"function","name":"web_search"}}`
	if planCol, err := bridge.Convert(ir.FormatResponses, ir.FormatAnthropic, []byte(nsCollisionReq), bridge.Options{
		Anthropic: anthropic.Dialect{DisableWebSearch: true},
	}); err != nil {
		t.Fatalf("expected colliding namespaced web_search selected by bare name to succeed: %v", err)
	} else if !strings.Contains(string(planCol.Body), `"tool_choice":{"type":"tool","name":"ns__web_search_2"}`) {
		t.Fatalf("expected resolved colliding wire name ns__web_search_2: %s", planCol.Body)
	}
}

// TestChatBuildDisableWebSearchOff_ByteExactV010 pins default Chat build behavior
// against v0.1.0 when DisableWebSearch is false (including named tool_choice targeting
// an undeclared tool, which v0.1.0 forwarded unchanged).
func TestChatBuildDisableWebSearchOff_ByteExactV010(t *testing.T) {
	req := `{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"foo","parameters":{"type":"object"}}}],"tool_choice":{"type":"function","function":{"name":"missing"}}}`
	plan, err := bridge.Convert(ir.FormatChat, ir.FormatChat, []byte(req), bridge.Options{})
	if err != nil {
		t.Fatalf("Chat->Chat default Convert should preserve v0.1.0 behavior, got error: %v", err)
	}
	want := `{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"foo","parameters":{"type":"object"}}}],"tool_choice":{"type":"function","function":{"name":"missing"}}}`
	if string(plan.Body) != want {
		t.Fatalf("Chat->Chat output diverged from v0.1.0:\ngot:  %s\nwant: %s", plan.Body, want)
	}

	// Namespaced regular function named "web_search" selected by bare name alongside dropped web_search.
	nsReq := `{"model":"m","input":"hi","tools":[{"type":"web_search","external_web_access":true},{"type":"namespace","name":"ns","tools":[{"type":"function","name":"web_search","parameters":{"type":"object"}}]}],"tool_choice":{"type":"function","name":"web_search"}}`
	if planNS, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, []byte(nsReq), bridge.Options{
		Chat: chat.Dialect{DisableWebSearch: true},
	}); err != nil {
		t.Fatalf("expected namespaced web_search selected by bare name toward Chat to succeed: %v", err)
	} else if !strings.Contains(string(planNS.Body), `"tool_choice":{"type":"function","function":{"name":"ns__web_search"}}`) {
		t.Fatalf("expected resolved wire name ns__web_search in Chat body: %s", planNS.Body)
	}
}

// TestResponsesDuplicateAndAliasedBuiltins_ByteExactV010 pins byte-for-byte
// compatibility with v0.1.0 when a Responses request declares duplicate or
// aliased web_search builtins (web_search + web_search_preview) alongside
// regular function tools and DisableWebSearch is false.
func TestResponsesDuplicateAndAliasedBuiltins_ByteExactV010(t *testing.T) {
	cachedAliased := `{"model":"m","input":"hi","tools":[{"type":"web_search","external_web_access":false},{"type":"web_search_preview","external_web_access":false},{"type":"function","name":"ping","description":"Ping","parameters":{"type":"object","properties":{"x":{"type":"string"}}}}]}`
	liveAliased := `{"model":"m","input":"hi","tools":[{"type":"web_search"},{"type":"web_search_preview"},{"type":"function","name":"ping","description":"Ping","parameters":{"type":"object","properties":{"x":{"type":"string"}}}}]}`

	cases := []struct {
		name       string
		req        string
		upstream   ir.Format
		wantErr    bool
		wantBody   string
		wantLosses int
	}{
		{
			name:       "cached_aliased_to_chat",
			req:        cachedAliased,
			upstream:   ir.FormatChat,
			wantBody:   `{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"ping","description":"Ping","parameters":{"type":"object","properties":{"x":{"type":"string"}}}}}]}`,
			wantLosses: 2,
		},
		{
			name:       "cached_aliased_to_anthropic",
			req:        cachedAliased,
			upstream:   ir.FormatAnthropic,
			wantBody:   `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":16384,"tools":[{"name":"ping","description":"Ping","input_schema":{"type":"object","properties":{"x":{"type":"string"}}}}]}`,
			wantLosses: 2,
		},
		{
			name:       "cached_aliased_to_gemini",
			req:        cachedAliased,
			upstream:   ir.FormatGemini,
			wantBody:   `{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"tools":[{"functionDeclarations":[{"name":"ping","description":"Ping","parameters":{"properties":{"x":{"type":"string"}},"type":"object"}}]}]}`,
			wantLosses: 2,
		},
		{
			name:     "live_aliased_to_chat_rejected",
			req:      liveAliased,
			upstream: ir.FormatChat,
			wantErr:  true,
		},
		{
			name:     "live_aliased_to_anthropic",
			req:      liveAliased,
			upstream: ir.FormatAnthropic,
			wantBody: `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":16384,"tools":[{"name":"ping","description":"Ping","input_schema":{"type":"object","properties":{"x":{"type":"string"}}}},{"name":"web_search","type":"web_search_20250305"},{"name":"web_search","type":"web_search_20250305"}]}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := bridge.Convert(ir.FormatResponses, tc.upstream, []byte(tc.req), bridge.Options{})
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error matching v0.1.0, got body=%s", plan.Body)
				}
				return
			}
			if err != nil {
				t.Fatalf("Convert failed: %v", err)
			}
			if string(plan.Body) != tc.wantBody {
				t.Fatalf("body diverged from v0.1.0:\ngot:  %s\nwant: %s", plan.Body, tc.wantBody)
			}
			if len(plan.Losses) != tc.wantLosses {
				t.Fatalf("losses count = %d, want %d: %+v", len(plan.Losses), tc.wantLosses, plan.Losses)
			}
		})
	}
}

// TestChatToAnthropicDisableWebSearchOff_ByteExactV010 pins byte-for-byte
// compatibility with v0.1.0 when Anthropic.DisableWebSearch is false, including
// Chat requests carrying web_search_options without regular tools and explicit
// tool_choice ("required", "none", "auto", named function, parallel_tool_calls:false).
func TestChatToAnthropicDisableWebSearchOff_ByteExactV010(t *testing.T) {
	cases := []struct {
		name string
		req  string
		want string
	}{
		{
			name: "web_search_options_tool_choice_required",
			req:  `{"model":"claude-sonnet-4-6","web_search_options":{"search_context_size":"low"},"tool_choice":"required","messages":[{"role":"user","content":"hi"}]}`,
			want: `{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":16384,"tool_choice":{"type":"any"}}`,
		},
		{
			name: "web_search_options_tool_choice_none",
			req:  `{"model":"claude-sonnet-4-6","web_search_options":{"search_context_size":"low"},"tool_choice":"none","messages":[{"role":"user","content":"hi"}]}`,
			want: `{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":16384,"tool_choice":{"type":"none"}}`,
		},
		{
			name: "web_search_options_tool_choice_auto",
			req:  `{"model":"claude-sonnet-4-6","web_search_options":{"search_context_size":"low"},"tool_choice":"auto","messages":[{"role":"user","content":"hi"}]}`,
			want: `{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":16384,"tool_choice":{"type":"auto"}}`,
		},
		{
			name: "web_search_options_parallel_tool_calls_false",
			req:  `{"model":"claude-sonnet-4-6","web_search_options":{"search_context_size":"low"},"parallel_tool_calls":false,"messages":[{"role":"user","content":"hi"}]}`,
			want: `{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":16384,"tool_choice":{"type":"auto","disable_parallel_tool_use":true}}`,
		},
		{
			name: "web_search_options_tool_choice_named",
			req:  `{"model":"claude-sonnet-4-6","web_search_options":{"search_context_size":"low"},"tool_choice":{"type":"function","function":{"name":"foo"}},"messages":[{"role":"user","content":"hi"}]}`,
			want: `{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":16384,"tool_choice":{"type":"tool","name":"foo"}}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := bridge.Convert(ir.FormatChat, ir.FormatAnthropic, []byte(tc.req), bridge.Options{
				Unknown: ir.UnknownIgnore,
			})
			if err != nil {
				t.Fatalf("Convert failed: %v", err)
			}
			if string(plan.Body) != tc.want {
				t.Fatalf("output diverged from v0.1.0:\ngot:  %s\nwant: %s", plan.Body, tc.want)
			}
		})
	}
}

func TestChatDisableWebSearch(t *testing.T) {
	// 1. Responses -> Chat with live web_search (external_web_access: true) + function tool (Codex YOLO mode).
	liveReq := `{
	  "model": "gpt-5.4-mini",
	  "tool_choice": "auto",
	  "tools": [
	    {"type": "web_search", "external_web_access": true},
	    {"type": "function", "name": "exec_command", "description": "Run command", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}
	  ],
	  "input": "check status"
	}`

	// Default (DisableWebSearch=false): live web_search is still rejected.
	if _, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, []byte(liveReq), bridge.Options{}); err == nil {
		t.Fatalf("expected live web_search to fail toward Chat when DisableWebSearch=false")
	}

	// DisableWebSearch=true: live web_search dropped, function tool preserved, ir.LossWebSearch recorded.
	var seenLosses []ir.Loss
	planOff, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, []byte(liveReq), bridge.Options{
		Chat:   chat.Dialect{DisableWebSearch: true},
		OnLoss: func(l ir.Loss) { seenLosses = append(seenLosses, l) },
	})
	if err != nil {
		t.Fatalf("Chat DisableWebSearch Convert failed: %v", err)
	}
	if strings.Contains(string(planOff.Body), "web_search") {
		t.Fatalf("expected web_search stripped from Chat body: %s", planOff.Body)
	}
	if !strings.Contains(string(planOff.Body), `"exec_command"`) {
		t.Fatalf("expected regular function tool preserved: %s", planOff.Body)
	}
	hasLoss := false
	for _, l := range planOff.Losses {
		if l.Kind == ir.LossWebSearch {
			if l.Path.String() != "tools" {
				t.Fatalf("expected LossWebSearch path \"tools\", got %q", l.Path.String())
			}
			hasLoss = true
		}
	}
	if !hasLoss {
		t.Fatalf("expected ir.LossWebSearch in plan.Losses, got %+v", planOff.Losses)
	}
	hasCallback := false
	for _, l := range seenLosses {
		if l.Kind == ir.LossWebSearch {
			hasCallback = true
		}
	}
	if !hasCallback {
		t.Fatalf("expected ir.LossWebSearch passed to OnLoss")
	}

	// Deny[LossWebSearch] rejects it.
	if _, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, []byte(liveReq), bridge.Options{
		Chat: chat.Dialect{DisableWebSearch: true},
		Deny: ir.LossSet{ir.LossWebSearch: true},
	}); err == nil {
		t.Fatalf("expected Deny[LossWebSearch] to reject dropped web_search toward Chat")
	}

	// 2. Cached web_search (external_web_access: false) still records LossCachedWebSearch.
	cachedReq := `{"model":"gpt-5.4-mini","tools":[{"type":"web_search","external_web_access":false}],"input":"hi"}`
	planCached, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, []byte(cachedReq), bridge.Options{
		Chat: chat.Dialect{DisableWebSearch: true},
	})
	if err != nil {
		t.Fatalf("cached web_search toward Chat failed: %v", err)
	}
	hasCachedLoss := false
	for _, l := range planCached.Losses {
		if l.Kind == ir.LossCachedWebSearch {
			hasCachedLoss = true
		}
	}
	if !hasCachedLoss {
		t.Fatalf("expected LossCachedWebSearch for cached web_search, got %+v", planCached.Losses)
	}

	// 3. Only live web_search + tool_choice="auto" -> tools and tool_choice omitted.
	onlyWS := `{"model":"gpt-5.4-mini","tool_choice":"auto","tools":[{"type":"web_search","external_web_access":true}],"input":"hi"}`
	planOnlyWS, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, []byte(onlyWS), bridge.Options{
		Chat: chat.Dialect{DisableWebSearch: true},
	})
	if err != nil {
		t.Fatalf("only-web_search auto toward Chat failed: %v", err)
	}
	if strings.Contains(string(planOnlyWS.Body), `"tools"`) || strings.Contains(string(planOnlyWS.Body), `"tool_choice"`) {
		t.Fatalf("expected both tools and tool_choice omitted: %s", planOnlyWS.Body)
	}

	// 4. Only live web_search + tool_choice="required" or ToolChoiceTool -> explicit error.
	onlyWSReq := `{"model":"gpt-5.4-mini","tool_choice":"required","tools":[{"type":"web_search","external_web_access":true}],"input":"hi"}`
	if _, err := bridge.Convert(ir.FormatResponses, ir.FormatChat, []byte(onlyWSReq), bridge.Options{
		Chat: chat.Dialect{DisableWebSearch: true},
	}); err == nil {
		t.Fatalf("expected error when tool_choice=required and only web_search was dropped toward Chat")
	}
	directReq := &ir.Request{
		Model:      "gpt-5.4-mini",
		Messages:   []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("hi")}}},
		Builtins:   []ir.BuiltinTool{{Kind: ir.BuiltinWebSearch, Origin: ir.FormatResponses}},
		ToolChoice: ir.ToolChoice{Mode: ir.ToolChoiceTool, Name: "web_search"},
	}
	if _, err := bridge.Build(directReq, ir.FormatResponses, ir.FormatChat, bridge.Options{
		Chat: chat.Dialect{DisableWebSearch: true},
	}); err == nil {
		t.Fatalf("expected error from bridge.Build when ToolChoiceTool targets dropped web_search toward Chat")
	}
	// Even when regular function tools remain, forcing a dropped/builtin web_search must fail.
	mixedForcedName := &ir.Request{
		Model:      "gpt-5.4-mini",
		Messages:   []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("hi")}}},
		Tools:      []ir.Tool{{Kind: ir.ToolFunction, Name: "exec_command"}},
		Builtins:   []ir.BuiltinTool{{Kind: ir.BuiltinWebSearch, Origin: ir.FormatResponses}},
		ToolChoice: ir.ToolChoice{Mode: ir.ToolChoiceTool, Name: "web_search"},
	}
	if _, err := bridge.Build(mixedForcedName, ir.FormatResponses, ir.FormatChat, bridge.Options{
		Chat: chat.Dialect{DisableWebSearch: true},
	}); err == nil {
		t.Fatalf("expected error when ToolChoiceTool(Name=web_search) targets dropped web_search alongside function tools")
	}
	mixedForcedBuiltin := &ir.Request{
		Model:      "gpt-5.4-mini",
		Messages:   []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{ir.TextPart("hi")}}},
		Tools:      []ir.Tool{{Kind: ir.ToolFunction, Name: "exec_command"}},
		Builtins:   []ir.BuiltinTool{{Kind: ir.BuiltinWebSearch, Origin: ir.FormatResponses}},
		ToolChoice: ir.ToolChoice{Mode: ir.ToolChoiceTool, Builtin: ir.BuiltinWebSearch},
	}
	if _, err := bridge.Build(mixedForcedBuiltin, ir.FormatResponses, ir.FormatChat, bridge.Options{
		Chat: chat.Dialect{DisableWebSearch: true},
	}); err == nil {
		t.Fatalf("expected error when ToolChoiceTool(Builtin=web_search) targets dropped web_search alongside function tools")
	}

	// 5. Anthropic web_search_20250305 and Gemini googleSearch toward Chat with DisableWebSearch=true.
	anthReq := `{"model":"gpt-5.4-mini","max_tokens":128,"tools":[{"type":"web_search_20250305","name":"web_search"}],"messages":[{"role":"user","content":"hi"}]}`
	if planA, err := bridge.Convert(ir.FormatAnthropic, ir.FormatChat, []byte(anthReq), bridge.Options{
		Chat: chat.Dialect{DisableWebSearch: true},
	}); err != nil {
		t.Fatalf("Anthropic->Chat DisableWebSearch failed: %v", err)
	} else if strings.Contains(string(planA.Body), "web_search") {
		t.Fatalf("expected web_search stripped in Anthropic->Chat: %s", planA.Body)
	}
	gemReq := `{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"tools":[{"googleSearch":{}}]}`
	if planG, err := bridge.Convert(ir.FormatGemini, ir.FormatChat, []byte(gemReq), bridge.Options{
		Model: "gpt-5.4-mini",
		Chat:  chat.Dialect{DisableWebSearch: true},
	}); err != nil {
		t.Fatalf("Gemini->Chat DisableWebSearch failed: %v", err)
	} else if strings.Contains(string(planG.Body), "web_search") || strings.Contains(string(planG.Body), "googleSearch") {
		t.Fatalf("expected googleSearch stripped in Gemini->Chat: %s", planG.Body)
	}
}
