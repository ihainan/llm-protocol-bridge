package ir_test

import (
	"encoding/json"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/ir"
)

// --- Collector tests ---

func TestCollector_SimpleText(t *testing.T) {
	c := ir.NewCollector()
	c.Add(ir.Event{Type: ir.EvStart, ID: "r1", Model: "m1"})
	c.Add(ir.Event{Type: ir.EvText, Index: 0, Text: "hello "})
	c.Add(ir.Event{Type: ir.EvText, Index: 0, Text: "world"})
	c.Add(ir.Event{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopEndTurn}})

	resp := c.Response()
	if resp.ID != "r1" {
		t.Errorf("ID: want r1, got %q", resp.ID)
	}
	if len(resp.Outputs) != 1 {
		t.Fatalf("outputs: want 1, got %d", len(resp.Outputs))
	}
	out := resp.Outputs[0]
	if len(out.Parts) != 1 {
		t.Fatalf("parts: want 1, got %d", len(out.Parts))
	}
	if out.Parts[0].Text != "hello world" {
		t.Errorf("text: want %q, got %q", "hello world", out.Parts[0].Text)
	}
	if out.Stop.Reason != ir.StopEndTurn {
		t.Errorf("stop: want end_turn, got %q", out.Stop.Reason)
	}
}

func TestCollector_ToolCall(t *testing.T) {
	c := ir.NewCollector()
	c.Add(ir.Event{Type: ir.EvStart, ID: "r2", Model: "m"})
	c.Add(ir.Event{Type: ir.EvToolStart, Index: 0, ID: "tc1", Name: "Read", Kind: ir.ToolFunction})
	c.Add(ir.Event{Type: ir.EvToolArgs, Index: 0, Text: `{"path":"/x"}`})
	c.Add(ir.Event{Type: ir.EvToolEnd, Index: 0})
	c.Add(ir.Event{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopToolUse}})

	resp := c.Response()
	if len(resp.Outputs) != 1 {
		t.Fatalf("outputs: want 1, got %d", len(resp.Outputs))
	}
	out := resp.Outputs[0]
	if !ir.HasToolCalls(out.Parts) {
		t.Error("HasToolCalls: expected true")
	}
	call := out.Parts[0].Call
	if call == nil {
		t.Fatal("parts[0].Call is nil")
	}
	if call.Name != "Read" {
		t.Errorf("call.Name: want Read, got %q", call.Name)
	}
	if string(call.Arguments) != `{"path":"/x"}` {
		t.Errorf("call.Arguments: want %q, got %q", `{"path":"/x"}`, call.Arguments)
	}
}

func TestCollector_UsageMerge(t *testing.T) {
	c := ir.NewCollector()
	c.Add(ir.Event{Type: ir.EvStart, Usage: &ir.Usage{InputTokens: 10, Reported: ir.UInput}})
	c.Add(ir.Event{Type: ir.EvUsage, Usage: &ir.Usage{OutputTokens: 5, Reported: ir.UOutput}})

	resp := c.Response()
	if resp.Usage.InputTokens != 10 {
		t.Errorf("InputTokens: want 10, got %d", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 5 {
		t.Errorf("OutputTokens: want 5, got %d", resp.Usage.OutputTokens)
	}
}

func TestCollector_FinalStop_ToolUse(t *testing.T) {
	// A stop with Reason=end_turn, but parts contain a tool call → FinalStop upgrades to tool_use.
	c := ir.NewCollector()
	c.Add(ir.Event{Type: ir.EvStart})
	c.Add(ir.Event{Type: ir.EvToolStart, Index: 0, ID: "t1", Name: "F", Kind: ir.ToolFunction})
	c.Add(ir.Event{Type: ir.EvToolEnd, Index: 0})
	c.Add(ir.Event{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopEndTurn}})

	resp := c.Response()
	if resp.Outputs[0].Stop.Reason != ir.StopToolUse {
		t.Errorf("FinalStop: want tool_use, got %q", resp.Outputs[0].Stop.Reason)
	}
}

func TestCollector_MultipleOutputs(t *testing.T) {
	c := ir.NewCollector()
	c.Add(ir.Event{Type: ir.EvStart})
	c.Add(ir.Event{Type: ir.EvText, Index: 0, Text: "first"})
	c.Add(ir.Event{Type: ir.EvStop, Index: 0, Stop: &ir.Stop{Reason: ir.StopEndTurn}})
	c.Add(ir.Event{Type: ir.EvText, Index: 1, Text: "second"})
	c.Add(ir.Event{Type: ir.EvStop, Index: 1, Stop: &ir.Stop{Reason: ir.StopEndTurn}})

	resp := c.Response()
	if len(resp.Outputs) != 2 {
		t.Fatalf("outputs: want 2, got %d", len(resp.Outputs))
	}
}

func TestCollector_ErrorEvent(t *testing.T) {
	c := ir.NewCollector()
	c.Add(ir.Event{Type: ir.EvError, Err: &ir.APIError{Status: 500, Message: "oops"}})
	if err := c.Err(); err == nil {
		t.Error("expected Err() to return non-nil after EvError")
	}
}

// --- Usage tests ---

func TestUsage_PromptTokens(t *testing.T) {
	u := ir.Usage{
		InputTokens:      100,
		CacheReadTokens:  20,
		CacheWriteTokens: 10,
		Reported:         ir.UInput | ir.UCacheRead | ir.UCacheWrite,
	}
	want := 130
	if got := u.PromptTokens(); got != want {
		t.Errorf("PromptTokens: want %d, got %d", want, got)
	}
}

func TestUsage_TotalTokens(t *testing.T) {
	u := ir.Usage{
		InputTokens:  100,
		OutputTokens: 50,
		Reported:     ir.UInput | ir.UOutput,
	}
	if got := u.TotalTokens(); got != 150 {
		t.Errorf("TotalTokens: want 150, got %d", got)
	}
}

func TestUsage_MergeOverwrites(t *testing.T) {
	a := ir.Usage{InputTokens: 10, Reported: ir.UInput}
	b := ir.Usage{InputTokens: 20, OutputTokens: 5, Reported: ir.UInput | ir.UOutput}
	a.Merge(b)
	if a.InputTokens != 20 {
		t.Errorf("after Merge: InputTokens want 20, got %d", a.InputTokens)
	}
	if a.OutputTokens != 5 {
		t.Errorf("after Merge: OutputTokens want 5, got %d", a.OutputTokens)
	}
}

func TestUsage_Has(t *testing.T) {
	u := ir.Usage{Reported: ir.UInput | ir.UCacheRead}
	if !u.Has(ir.UInput) {
		t.Error("Has(UInput): want true")
	}
	if u.Has(ir.UOutput) {
		t.Error("Has(UOutput): want false")
	}
}

// --- Env tests ---

func TestEnv_Lose_Allowed(t *testing.T) {
	env := &ir.Env{Allow: ir.DefaultAllowed}
	err := env.Lose(ir.LossRequestUser, ir.Root, "test")
	if err != nil {
		t.Errorf("Lose on allowed kind should return nil, got %v", err)
	}
	losses := env.Losses()
	if len(losses) != 1 {
		t.Errorf("losses: want 1, got %d", len(losses))
	}
}

func TestEnv_Lose_Denied(t *testing.T) {
	env := &ir.Env{Deny: ir.LossSet{ir.LossRequestUser: true}}
	err := env.Lose(ir.LossRequestUser, ir.Root, "test")
	if err == nil {
		t.Error("Lose on denied kind should return error")
	}
}

func TestEnv_NewID(t *testing.T) {
	env := &ir.Env{}
	id := env.NewID("test_", 10)
	if len(id) < 5 {
		t.Errorf("NewID returned suspiciously short id: %q", id)
	}
	if id[:5] != "test_" {
		t.Errorf("NewID prefix: want test_, got %q", id[:5])
	}
}

// --- EventsOf tests ---

func TestEventsOf_RoundTrip(t *testing.T) {
	resp := &ir.Response{
		ID:    "r1",
		Model: "m",
		Outputs: []ir.Output{{
			Index: 0,
			Parts: []ir.Part{
				ir.TextPart("hello"),
				{
					Type: ir.PartToolCall,
					Call: &ir.ToolCall{ID: "c1", Name: "F", Kind: ir.ToolFunction, Arguments: json.RawMessage(`{"x":1}`)},
				},
			},
			Stop: ir.Stop{Reason: ir.StopToolUse},
		}},
		Usage: ir.Usage{InputTokens: 5, OutputTokens: 2, Reported: ir.UInput | ir.UOutput},
	}

	evs := ir.EventsOf(resp)

	c := ir.NewCollector()
	for _, ev := range evs {
		c.Add(ev)
	}
	resp2 := c.Response()

	if resp2.ID != resp.ID {
		t.Errorf("ID: want %q, got %q", resp.ID, resp2.ID)
	}
	if len(resp2.Outputs) != 1 {
		t.Fatalf("outputs: want 1, got %d", len(resp2.Outputs))
	}
	if resp2.Outputs[0].Stop.Reason != ir.StopToolUse {
		t.Errorf("stop: want tool_use, got %q", resp2.Outputs[0].Stop.Reason)
	}
	if resp2.Usage.InputTokens != 5 {
		t.Errorf("InputTokens: want 5, got %d", resp2.Usage.InputTokens)
	}
}
