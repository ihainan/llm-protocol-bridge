package anthropic

import (
	"encoding/json"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// maxCacheBreakpoints is the number of cache_control breakpoints Anthropic
// accepts in one request; more is rejected with HTTP 400.
const maxCacheBreakpoints = 4

// toolEntry is one element of the outgoing tools array: a function tool built
// from the IR (wt) or a pre-serialised builtin tool (raw).
type toolEntry struct {
	wt   *wireTool
	raw  json.RawMessage
	path ir.Path
}

// renderTools marshals the tools array exactly as a []json.RawMessage of
// individually marshalled tools.
func renderTools(tools []toolEntry) ([]byte, error) {
	raws := make([]json.RawMessage, 0, len(tools))
	for _, t := range tools {
		if t.wt != nil {
			tb, err := jsonx.Marshal(t.wt)
			if err != nil {
				return nil, err
			}
			raws = append(raws, tb)
			continue
		}
		raws = append(raws, t.raw)
	}
	return jsonx.Marshal(raws)
}

// setCache attaches an IR cache marker to a block and remembers its IR path.
func (b *wireContentBlock) setCache(cc *ir.CacheControl, p ir.Path) {
	b.CacheControl = irCacheToWire(cc)
	if cc != nil {
		b.cachePath = p.Key("cache_control")
	}
}

// cacheSite is one cache_control breakpoint in the outgoing request.
type cacheSite struct {
	path  ir.Path
	empty bool   // the marker sits on an empty text block
	ttl   string // "" (5m default), "5m" or "1h"
	clear func()
	// shorten sets the TTL to the 5-minute default.
	shorten func()
}

// collectCacheSites lists the breakpoints present in the built request in
// render order: tools, system, then messages (inner tool_result content
// before its enclosing tool_result).
func collectCacheSites(tools []toolEntry, sys []wireContentBlock, msgs []msgBlocks) []cacheSite {
	var sites []cacheSite
	for i := range tools {
		t := &tools[i]
		if t.wt != nil {
			if t.wt.CacheControl != nil {
				wt := t.wt
				sites = append(sites, cacheSite{
					path:    t.path.Key("cache_control"),
					ttl:     wt.CacheControl.TTL,
					clear:   func() { wt.CacheControl = nil },
					shorten: func() { wt.CacheControl.TTL = "" },
				})
			}
			continue
		}
		if ok, ttl := rawCacheControl(t.raw); ok {
			sites = append(sites, cacheSite{
				path:    t.path.Key("cache_control"),
				ttl:     ttl,
				clear:   func() { t.raw = rawWithoutCacheControl(t.raw) },
				shorten: func() { t.raw = rawWithoutCacheTTL(t.raw) },
			})
		}
	}
	addBlock := func(b *wireContentBlock) {
		if b.CacheControl == nil {
			return
		}
		sites = append(sites, cacheSite{
			path:    b.cachePath,
			empty:   b.Type == "text" && b.Text == "",
			ttl:     b.CacheControl.TTL,
			clear:   func() { b.CacheControl = nil },
			shorten: func() { b.CacheControl.TTL = "" },
		})
	}
	for i := range sys {
		addBlock(&sys[i])
	}
	for m := range msgs {
		for j := range msgs[m].blocks {
			b := &msgs[m].blocks[j]
			for k := range b.inner {
				addBlock(&b.inner[k])
			}
			addBlock(b)
		}
	}
	return sites
}

// placeCacheBreakpoints validates client cache markers, enforces the
// four-breakpoint limit, and applies the automatic placement strategy.
//
// Markers on empty text blocks (rejected by Anthropic) are dropped first.
// When more than four remain, the first one in render order (normally the
// stable tools/system prefix) and the last three (the most recent turns) are
// kept and the others dropped. Anthropic also requires 1-hour breakpoints to
// precede 5-minute ones, so a 1-hour marker after a 5-minute one is shortened
// to the 5-minute default. Every dropped or shortened marker is recorded as
// ir.LossCacheMarker. The automatic strategy runs only when the request
// carried no cache marker of its own.
func placeCacheBreakpoints(req *ir.Request, mode AutoCacheMode, env *ir.Env, tools []toolEntry, sys []wireContentBlock, msgs []msgBlocks) error {
	if !mode.Valid() {
		return env.Err(ir.Invalid(ir.Root, "unknown anthropic AutoCache mode: "+string(mode)))
	}
	sites := collectCacheSites(tools, sys, msgs)
	kept := sites[:0:0]
	for _, s := range sites {
		if s.empty {
			if err := env.Lose(ir.LossCacheMarker, s.path, "cache_control is not allowed on an empty text block"); err != nil {
				return err
			}
			s.clear()
			continue
		}
		kept = append(kept, s)
	}
	if n := len(kept); n > maxCacheBreakpoints {
		last := n - (maxCacheBreakpoints - 1)
		for i := 1; i < last; i++ {
			if err := env.Lose(ir.LossCacheMarker, kept[i].path, "Anthropic allows at most 4 cache breakpoints per request; the first and the last three are kept"); err != nil {
				return err
			}
			kept[i].clear()
		}
		kept = append(kept[:1:1], kept[last:]...)
	}
	seenShort := false
	for _, s := range kept {
		if s.ttl != "1h" {
			seenShort = true
			continue
		}
		if !seenShort {
			continue
		}
		if err := env.Lose(ir.LossCacheMarker, s.path.Key("ttl"), "a 1h cache breakpoint must precede 5m breakpoints; shortened to 5m"); err != nil {
			return err
		}
		s.shorten()
	}
	if mode == AutoCacheOff || len(sites) > 0 || requestHasCacheMarker(req) {
		return nil
	}
	switch mode {
	case AutoCachePrefix:
		autoCachePrefix(tools, sys)
	case AutoCacheConversation:
		autoCachePrefix(tools, sys)
		autoCacheConversation(msgs)
	}
	return nil
}

// requestHasCacheMarker reports whether the client sent any cache marker,
// including ones the builder could not emit (for example on a folded
// mid-conversation system message or on a hosted tool the dialect drops).
func requestHasCacheMarker(req *ir.Request) bool {
	for _, t := range req.Tools {
		if t.Cache != nil {
			return true
		}
	}
	for _, bt := range req.Builtins {
		if ok, _ := rawCacheControl(bt.Raw); ok {
			return true
		}
	}
	if partsHaveCacheMarker(req.Instructions) {
		return true
	}
	for _, m := range req.Messages {
		if partsHaveCacheMarker(m.Parts) {
			return true
		}
	}
	return false
}

func partsHaveCacheMarker(parts []ir.Part) bool {
	for _, p := range parts {
		if p.Cache != nil {
			return true
		}
		if p.Result != nil && partsHaveCacheMarker(p.Result.Content) {
			return true
		}
	}
	return false
}

func autoCacheMarker() *wireCacheControl { return &wireCacheControl{Type: "ephemeral"} }

// autoCachePrefix marks the last function tool and the last cacheable system
// block. Builtin tools after the last function tool stay outside the marked
// prefix; callers may append further tools after BuildRequest without
// invalidating it.
func autoCachePrefix(tools []toolEntry, sys []wireContentBlock) {
	for i := len(tools) - 1; i >= 0; i-- {
		if tools[i].wt != nil {
			tools[i].wt.CacheControl = autoCacheMarker()
			break
		}
	}
	for i := len(sys) - 1; i >= 0; i-- {
		if cacheableBlock(&sys[i]) {
			sys[i].CacheControl = autoCacheMarker()
			break
		}
	}
}

// autoCacheConversation marks the last cacheable block of the final message
// (written for the next turn to read) and of the user message before it (an
// exact match for the previous request's final breakpoint, independent of
// Anthropic's 20-block lookback window).
func autoCacheConversation(msgs []msgBlocks) {
	if len(msgs) == 0 {
		return
	}
	last := len(msgs) - 1
	markLastCacheable(msgs[last].blocks)
	for i := last - 1; i >= 0; i-- {
		if msgs[i].role == "user" {
			markLastCacheable(msgs[i].blocks)
			return
		}
	}
}

func markLastCacheable(blocks []wireContentBlock) {
	for i := len(blocks) - 1; i >= 0; i-- {
		if cacheableBlock(&blocks[i]) {
			blocks[i].CacheControl = autoCacheMarker()
			return
		}
	}
}

// cacheableBlock reports whether Anthropic accepts cache_control on b.
// Thinking blocks and empty text blocks cannot carry a breakpoint.
func cacheableBlock(b *wireContentBlock) bool {
	switch b.Type {
	case "text":
		return b.Text != ""
	case "image", "document", "tool_use", "tool_result":
		return true
	}
	return false
}

// rawCacheControl reports whether a pre-serialised tool object carries a
// non-null cache_control member, and its ttl.
func rawCacheControl(raw json.RawMessage) (bool, string) {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return false, ""
	}
	for _, kv := range kvs {
		if kv.Key != "cache_control" || jsonx.IsNull(kv.Value) {
			continue
		}
		var ttl string
		if ckvs, err := jsonx.Object(kv.Value); err == nil {
			for _, c := range ckvs {
				if c.Key == "ttl" {
					ttl, _ = jsonx.String(c.Value)
				}
			}
		}
		return true, ttl
	}
	return false, ""
}

// rawWithoutCacheTTL re-serialises a tool object with the ttl member removed
// from its cache_control object (the 5-minute default), keeping every other
// member in its original order.
func rawWithoutCacheTTL(raw json.RawMessage) json.RawMessage {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return raw
	}
	out := jsonx.Start(nil)
	for _, kv := range kvs {
		if kv.Key != "cache_control" {
			out.Raw(kv.Key, kv.Value)
			continue
		}
		ckvs, err := jsonx.Object(kv.Value)
		if err != nil {
			out.Raw(kv.Key, kv.Value)
			continue
		}
		cc := jsonx.Start(nil)
		for _, c := range ckvs {
			if c.Key != "ttl" {
				cc.Raw(c.Key, c.Value)
			}
		}
		out.Raw(kv.Key, cc.End())
	}
	return out.End()
}

// rawWithoutCacheControl re-serialises a tool object without cache_control,
// keeping the other members in their original order.
func rawWithoutCacheControl(raw json.RawMessage) json.RawMessage {
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return raw
	}
	out := jsonx.Start(nil)
	for _, kv := range kvs {
		if kv.Key == "cache_control" {
			continue
		}
		out.Raw(kv.Key, kv.Value)
	}
	return out.End()
}
