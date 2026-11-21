package ir

// UsageFields is a bit set of the Usage counters an upstream reported.
type UsageFields uint16

// Usage counter bits.
const (
	UInput UsageFields = 1 << iota
	UCacheRead
	UCacheWrite
	UCacheWriteSplit
	UOutput
	UReasoning
	UToolUsePrompt
	UWebSearch
	UWebFetch
)

// Usage counts tokens with one convention for every format: InputTokens excludes
// cache reads and writes, OutputTokens includes reasoning.
type Usage struct {
	InputTokens         int `json:"input_tokens"`
	CacheReadTokens     int `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens    int `json:"cache_write_tokens,omitempty"`
	CacheWrite5mTokens  int `json:"cache_write_5m_tokens,omitempty"`
	CacheWrite1hTokens  int `json:"cache_write_1h_tokens,omitempty"`
	OutputTokens        int `json:"output_tokens"`
	ReasoningTokens     int `json:"reasoning_tokens,omitempty"`
	ToolUsePromptTokens int `json:"tool_use_prompt_tokens,omitempty"`
	WebSearchRequests   int `json:"web_search_requests,omitempty"`
	WebFetchRequests    int `json:"web_fetch_requests,omitempty"`
	// Reported marks the counters the upstream actually reported.
	Reported UsageFields `json:"reported,omitempty"`
}

// PromptTokens is every prompt token, cached or not (OpenAI and Gemini semantics).
func (u Usage) PromptTokens() int { return u.InputTokens + u.CacheReadTokens + u.CacheWriteTokens }

// TotalTokens is the prompt plus the output.
func (u Usage) TotalTokens() int { return u.PromptTokens() + u.OutputTokens }

// Has reports whether every counter in f was reported.
func (u Usage) Has(f UsageFields) bool { return u.Reported&f == f }

// Any reports whether any counter was reported.
func (u Usage) Any() bool { return u.Reported != 0 }

// Merge applies a cumulative snapshot: every counter v reports replaces the one in
// u. Streams repeat cumulative counts, so snapshots are never added.
func (u *Usage) Merge(v Usage) {
	if v.Reported&UInput != 0 {
		u.InputTokens = v.InputTokens
	}
	if v.Reported&UCacheRead != 0 {
		u.CacheReadTokens = v.CacheReadTokens
	}
	if v.Reported&UCacheWrite != 0 {
		u.CacheWriteTokens = v.CacheWriteTokens
	}
	if v.Reported&UCacheWriteSplit != 0 {
		u.CacheWrite5mTokens = v.CacheWrite5mTokens
		u.CacheWrite1hTokens = v.CacheWrite1hTokens
	}
	if v.Reported&UOutput != 0 {
		u.OutputTokens = v.OutputTokens
	}
	if v.Reported&UReasoning != 0 {
		u.ReasoningTokens = v.ReasoningTokens
	}
	if v.Reported&UToolUsePrompt != 0 {
		u.ToolUsePromptTokens = v.ToolUsePromptTokens
	}
	if v.Reported&UWebSearch != 0 {
		u.WebSearchRequests = v.WebSearchRequests
	}
	if v.Reported&UWebFetch != 0 {
		u.WebFetchRequests = v.WebFetchRequests
	}
	u.Reported |= v.Reported
}

// Add sums two independent usages (for example two rounds of a tool loop).
func (u *Usage) Add(v Usage) {
	u.InputTokens += v.InputTokens
	u.CacheReadTokens += v.CacheReadTokens
	u.CacheWriteTokens += v.CacheWriteTokens
	u.CacheWrite5mTokens += v.CacheWrite5mTokens
	u.CacheWrite1hTokens += v.CacheWrite1hTokens
	u.OutputTokens += v.OutputTokens
	u.ReasoningTokens += v.ReasoningTokens
	u.ToolUsePromptTokens += v.ToolUsePromptTokens
	u.WebSearchRequests += v.WebSearchRequests
	u.WebFetchRequests += v.WebFetchRequests
	u.Reported |= v.Reported
}

// FromInclusivePrompt fills InputTokens from a prompt count that includes cache
// reads and writes (OpenAI semantics). When the cached counts exceed the prompt
// count, the upstream evidently reported an exclusive count, which is kept.
func (u *Usage) FromInclusivePrompt(prompt int) {
	cached := u.CacheReadTokens + u.CacheWriteTokens
	if cached > prompt {
		u.InputTokens = prompt
	} else {
		u.InputTokens = prompt - cached
	}
	u.Reported |= UInput
}
