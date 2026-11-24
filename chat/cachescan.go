package chat

import (
	"bytes"
	"encoding/json"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// Positions of a Chat request at which the parser reads cache_control itself.
type cacheScope int

const (
	scopeOther    cacheScope = iota // anywhere else: a marker here is not carried
	scopeMessage                    // messages[i]
	scopePart                       // messages[i].content[j]
	scopeToolCall                   // messages[i].tool_calls[k]
	scopeTool                       // tools[i]
	scopeFunction                   // tools[i].function (parameters skipped)
)

// reportUncarriedCacheMarkers records a cache.marker loss for every non-null
// cache_control member at a position the parser does not read (request level,
// inside image_url / file / function objects, ...), so no client marker is
// ignored silently. Each object's members are read last-wins. JSON Schemas
// (tool parameters, response_format) and metadata are user data and are not
// searched. Requests that never mention cache_control are not walked.
func reportUncarriedCacheMarkers(body []byte, env *ir.Env) error {
	if !bytes.Contains(body, []byte(`"cache_control"`)) {
		return nil
	}
	return walkCacheMarkers(body, ir.Root, scopeOther, true, env)
}

func walkCacheMarkers(raw json.RawMessage, p ir.Path, scope cacheScope, root bool, env *ir.Env) error {
	if elems, ok := jsonx.Array(raw); ok {
		for i, e := range elems {
			if err := walkCacheMarkers(e, p.At(i), scopeOther, false, env); err != nil {
				return err
			}
		}
		return nil
	}
	kvs, err := jsonx.Object(raw)
	if err != nil {
		return nil
	}
	last := make(map[string]json.RawMessage, len(kvs))
	var order []string
	for _, kv := range kvs {
		if _, seen := last[kv.Key]; !seen {
			order = append(order, kv.Key)
		}
		last[kv.Key] = kv.Value
	}
	for _, k := range order {
		v := last[k]
		if k == "cache_control" {
			if scope == scopeOther || scope == scopeFunction {
				if err := loseIgnoredCacheControl(v, p.Key(k), "cache_control is not carried at this position; mark content parts, messages or tools instead", env); err != nil {
					return err
				}
			}
			continue
		}
		if root && (k == "response_format" || k == "metadata") || scope == scopeFunction && k == "parameters" {
			continue
		}
		child := scopeOther
		switch {
		case scope == scopeTool && k == "function":
			child = scopeFunction
		case root && k == "messages":
			if err := walkScopedArray(v, p.Key(k), scopeMessage, env); err != nil {
				return err
			}
			continue
		case root && k == "tools":
			if err := walkScopedArray(v, p.Key(k), scopeTool, env); err != nil {
				return err
			}
			continue
		case scope == scopeMessage && k == "content":
			if err := walkScopedArray(v, p.Key(k), scopePart, env); err != nil {
				return err
			}
			continue
		case scope == scopeMessage && k == "tool_calls":
			if err := walkScopedArray(v, p.Key(k), scopeToolCall, env); err != nil {
				return err
			}
			continue
		}
		if err := walkCacheMarkers(v, p.Key(k), child, false, env); err != nil {
			return err
		}
	}
	return nil
}

// walkScopedArray walks the elements of an array whose elements sit at scope.
func walkScopedArray(raw json.RawMessage, p ir.Path, scope cacheScope, env *ir.Env) error {
	elems, ok := jsonx.Array(raw)
	if !ok {
		return nil
	}
	for i, e := range elems {
		if err := walkCacheMarkers(e, p.At(i), scope, false, env); err != nil {
			return err
		}
	}
	return nil
}
