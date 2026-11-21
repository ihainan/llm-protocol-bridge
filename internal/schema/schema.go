// Package schema adapts JSON Schemas between what the formats accept for tool
// parameters and structured output.
package schema

import (
	"encoding/json"
	"sort"
	"strings"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
)

// EmptyObject is the schema of a tool without parameters.
var EmptyObject = json.RawMessage(`{"type":"object","properties":{}}`)

// ObjectRoot checks that a tool parameter schema can describe a JSON object and
// returns it (EmptyObject when absent). A schema without a type, with type
// "object", or with a type list containing "object" is accepted.
func ObjectRoot(raw json.RawMessage) (json.RawMessage, bool) {
	if jsonx.Absent(raw) {
		return EmptyObject, true
	}
	if jsonx.Kind(raw) != '{' {
		return nil, false
	}
	var s struct {
		Type json.RawMessage `json:"type"`
	}
	if json.Unmarshal(raw, &s) != nil {
		return nil, false
	}
	if jsonx.IsNull(s.Type) {
		return raw, true
	}
	if t, ok := jsonx.String(s.Type); ok {
		return raw, t == "object"
	}
	var ts []string
	if json.Unmarshal(s.Type, &ts) == nil {
		for _, t := range ts {
			if t == "object" {
				return raw, true
			}
		}
	}
	return nil, false
}

// FromOpenAPI converts a Gemini OpenAPI-subset schema (upper-case types,
// "nullable") into JSON Schema.
func FromOpenAPI(raw json.RawMessage) json.RawMessage {
	var v any
	if json.Unmarshal(raw, &v) != nil {
		return raw
	}
	out, err := jsonx.Marshal(fromOpenAPI(v))
	if err != nil {
		return raw
	}
	return out
}

func fromOpenAPI(v any) any {
	switch x := v.(type) {
	case map[string]any:
		m := make(map[string]any, len(x))
		nullable := false
		for k, val := range x {
			switch k {
			case "type":
				if s, ok := val.(string); ok {
					m[k] = strings.ToLower(s)
					continue
				}
				m[k] = val
			case "nullable":
				nullable, _ = val.(bool)
			case "properties", "$defs", "definitions", "patternProperties":
				if pm, ok := val.(map[string]any); ok {
					pp := make(map[string]any, len(pm))
					for pk, pv := range pm {
						pp[pk] = fromOpenAPI(pv)
					}
					m[k] = pp
					continue
				}
				m[k] = val
			case "propertyOrdering":
				// Gemini-only ordering hint, not JSON Schema
			default:
				m[k] = fromOpenAPI(val)
			}
		}
		if nullable {
			if t, ok := m["type"].(string); ok {
				m["type"] = []any{t, "null"}
			}
		}
		return m
	case []any:
		out := make([]any, len(x))
		for i := range x {
			out[i] = fromOpenAPI(x[i])
		}
		return out
	}
	return v
}

// openAPIKeys are the schema keywords Gemini's OpenAPI subset accepts.
var openAPIKeys = map[string]bool{
	"type": true, "format": true, "title": true, "description": true, "nullable": true,
	"enum": true, "maxItems": true, "minItems": true, "properties": true, "required": true,
	"minProperties": true, "maxProperties": true, "minLength": true, "maxLength": true,
	"pattern": true, "example": true, "anyOf": true, "propertyOrdering": true,
	"default": true, "items": true, "minimum": true, "maximum": true,
}

// ToOpenAPI converts JSON Schema into Gemini's OpenAPI subset: type lists with
// "null" become nullable, keywords outside the subset are dropped, local $refs
// are inlined.
func ToOpenAPI(raw json.RawMessage) json.RawMessage {
	var v any
	if json.Unmarshal(raw, &v) != nil {
		return raw
	}
	root, _ := v.(map[string]any)
	defs := map[string]any{}
	for _, k := range []string{"$defs", "definitions"} {
		if d, ok := root[k].(map[string]any); ok {
			for name, s := range d {
				defs["#/"+k+"/"+name] = s
			}
		}
	}
	out, err := jsonx.Marshal(toOpenAPI(v, defs, 0))
	if err != nil {
		return raw
	}
	return out
}

func toOpenAPI(v any, defs map[string]any, depth int) any {
	switch x := v.(type) {
	case map[string]any:
		if ref, ok := x["$ref"].(string); ok && depth < 16 {
			if d, ok := defs[ref]; ok {
				return toOpenAPI(d, defs, depth+1)
			}
		}
		m := make(map[string]any, len(x))
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			val := x[k]
			if !openAPIKeys[k] {
				if k == "oneOf" {
					m["anyOf"] = toOpenAPI(val, defs, depth+1)
				}
				if k == "const" {
					m["enum"] = []any{val}
				}
				continue
			}
			switch k {
			case "type":
				if list, ok := val.([]any); ok {
					var types []string
					for _, t := range list {
						if s, _ := t.(string); s == "null" {
							m["nullable"] = true
						} else if s != "" {
							types = append(types, s)
						}
					}
					if len(types) == 1 {
						m["type"] = types[0]
					} else if len(types) > 1 {
						var alts []any
						for _, t := range types {
							alts = append(alts, map[string]any{"type": t})
						}
						m["anyOf"] = alts
					}
					continue
				}
				m[k] = val
			case "properties":
				if pm, ok := val.(map[string]any); ok {
					pp := make(map[string]any, len(pm))
					for pk, pv := range pm {
						pp[pk] = toOpenAPI(pv, defs, depth+1)
					}
					m[k] = pp
					continue
				}
				m[k] = val
			case "enum", "required", "propertyOrdering", "example", "default":
				m[k] = val
			default:
				m[k] = toOpenAPI(val, defs, depth+1)
			}
		}
		return m
	case []any:
		out := make([]any, len(x))
		for i := range x {
			out[i] = toOpenAPI(x[i], defs, depth+1)
		}
		return out
	}
	return v
}

// WithoutKeys returns the schema with the given top-level-and-nested keywords
// removed (for example "$schema" for providers that reject it).
func WithoutKeys(raw json.RawMessage, keys ...string) json.RawMessage {
	if len(keys) == 0 || !containsAny(raw, keys) {
		return raw
	}
	var v any
	if json.Unmarshal(raw, &v) != nil {
		return raw
	}
	drop := map[string]bool{}
	for _, k := range keys {
		drop[k] = true
	}
	out, err := jsonx.Marshal(without(v, drop))
	if err != nil {
		return raw
	}
	return out
}

func containsAny(raw []byte, keys []string) bool {
	for _, k := range keys {
		if strings.Contains(string(raw), `"`+k+`"`) {
			return true
		}
	}
	return false
}

func without(v any, drop map[string]bool) any {
	switch x := v.(type) {
	case map[string]any:
		m := make(map[string]any, len(x))
		for k, val := range x {
			if drop[k] {
				continue
			}
			if k == "properties" {
				if pm, ok := val.(map[string]any); ok {
					pp := make(map[string]any, len(pm))
					for pk, pv := range pm {
						pp[pk] = without(pv, drop)
					}
					m[k] = pp
					continue
				}
			}
			m[k] = without(val, drop)
		}
		return m
	case []any:
		out := make([]any, len(x))
		for i := range x {
			out[i] = without(x[i], drop)
		}
		return out
	}
	return v
}
