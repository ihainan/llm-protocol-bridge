// Package schema adapts JSON Schemas between what the formats accept for tool
// parameters and structured output.
package schema

import (
	"encoding/json"
	"reflect"
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
	out, _ := ToOpenAPIWithLoss(raw)
	return out
}

// ToOpenAPIWithLoss converts JSON Schema into Gemini's OpenAPI subset and
// returns the sorted list of validation constraint keywords that were dropped.
func ToOpenAPIWithLoss(raw json.RawMessage) (json.RawMessage, []string) {
	var v any
	if json.Unmarshal(raw, &v) != nil {
		return raw, nil
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
	droppedSet := map[string]bool{}
	out, err := jsonx.Marshal(toOpenAPI(v, defs, 0, droppedSet))
	if err != nil {
		return raw, nil
	}
	var dropped []string
	for k := range droppedSet {
		dropped = append(dropped, k)
	}
	sort.Strings(dropped)
	return out, dropped
}

var nonConstraintMetaKeys = map[string]bool{
	"$schema":     true,
	"$id":         true,
	"$defs":       true,
	"definitions": true,
	"$ref":        true,
	"$comment":    true,
	"title":       true,
	"examples":    true,
	"default":     true,
	"const":       true,
}

func toOpenAPI(v any, defs map[string]any, depth int, dropped map[string]bool) any {
	switch x := v.(type) {
	case map[string]any:
		if ref, ok := x["$ref"].(string); ok && depth < 16 {
			if d, ok := defs[ref]; ok {
				resolved := toOpenAPI(d, defs, depth+1, dropped)
				if len(x) > 1 {
					// Convert sibling keywords via the normal toOpenAPI pipeline and
					// intersect them with the resolved target instead of overwriting.
					sib := make(map[string]any, len(x)-1)
					for k, val := range x {
						if k != "$ref" {
							sib[k] = val
						}
					}
					sibOut := toOpenAPI(sib, defs, depth+1, dropped)
					rm, isMap := resolved.(map[string]any)
					sm, sibIsMap := sibOut.(map[string]any)
					if isMap && sibIsMap {
						mergeRefOpenAPI(rm, sm, dropped)
					} else if dropped != nil {
						dropped["$ref"] = true
					}
				}
				return resolved
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
					if dropped != nil {
						dropped["oneOf"] = true
					}
					m["anyOf"] = toOpenAPI(val, defs, depth+1, dropped)
				} else if k == "const" {
					m["enum"] = []any{val}
				} else if dropped != nil && !nonConstraintMetaKeys[k] {
					dropped[k] = true
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
						pp[pk] = toOpenAPI(pv, defs, depth+1, dropped)
					}
					m[k] = pp
					continue
				}
				m[k] = val
			case "enum", "required", "propertyOrdering", "example", "default":
				m[k] = val
			default:
				m[k] = toOpenAPI(val, defs, depth+1, dropped)
			}
		}
		return m
	case []any:
		out := make([]any, len(x))
		for i := range x {
			out[i] = toOpenAPI(x[i], defs, depth+1, dropped)
		}
		return out
	}
	return v
}

// mergeRefOpenAPI intersects converted sibling OpenAPI keywords (src) into a
// converted $ref target schema (dst). Compatible constraints are intersected
// (e.g. stricter numeric bounds, merged properties, unioned required);
// unmergeable conflicts are recorded in dropped so callers and Deny can see them.
func mergeRefOpenAPI(dst, src map[string]any, dropped map[string]bool) {
	keys := make([]string, 0, len(src))
	for k := range src {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		sv := src[k]
		switch k {
		case "description", "example", "default", "propertyOrdering":
			dst[k] = sv
		case "type":
			// A sibling with an explicit non-nullable type excludes null, so
			// clear nullable:true inherited from the $ref target.
			if src["nullable"] != true && dst["nullable"] == true {
				delete(dst, "nullable")
			}
			rv, has := dst[k]
			if !has {
				dst[k] = sv
			} else if !reflect.DeepEqual(rv, sv) && dropped != nil {
				dropped[k] = true
			}
		case "nullable":
			sNull, _ := sv.(bool)
			if !sNull {
				if dst["nullable"] == true {
					dst["nullable"] = false
				}
				continue
			}
			// Intersection of a non-nullable typed target with nullable:true stays non-nullable.
			_, hasType := dst["type"]
			if (!hasType && dst["anyOf"] == nil && dst["enum"] == nil) || dst["nullable"] == true {
				dst["nullable"] = true
			}
		case "properties":
			sp, ok := sv.(map[string]any)
			if !ok {
				continue
			}
			dp, has := dst["properties"].(map[string]any)
			if !has {
				dst["properties"] = sp
				continue
			}
			pkeys := make([]string, 0, len(sp))
			for pk := range sp {
				pkeys = append(pkeys, pk)
			}
			sort.Strings(pkeys)
			for _, pk := range pkeys {
				pv := sp[pk]
				if existing, ok := dp[pk]; ok {
					em, eOK := existing.(map[string]any)
					pm, pOK := pv.(map[string]any)
					if eOK && pOK {
						mergeRefOpenAPI(em, pm, dropped)
					} else if !reflect.DeepEqual(existing, pv) && dropped != nil {
						dropped["properties"] = true
					}
				} else {
					dp[pk] = pv
				}
			}
		case "required":
			sr, ok := sv.([]any)
			if !ok {
				continue
			}
			dr, has := dst["required"].([]any)
			if !has {
				dst["required"] = sr
				continue
			}
			seen := make(map[string]bool, len(dr))
			for _, item := range dr {
				if s, ok := item.(string); ok {
					seen[s] = true
				}
			}
			for _, item := range sr {
				if s, ok := item.(string); ok && !seen[s] {
					seen[s] = true
					dr = append(dr, s)
				}
			}
			dst["required"] = dr
		case "minimum", "minLength", "minItems", "minProperties":
			rv, has := dst[k]
			if !has {
				dst[k] = sv
				continue
			}
			rf, rOK := rv.(float64)
			sf, sOK := sv.(float64)
			if rOK && sOK {
				if sf > rf {
					dst[k] = sf
				}
			} else if !reflect.DeepEqual(rv, sv) && dropped != nil {
				dropped[k] = true
			}
		case "maximum", "maxLength", "maxItems", "maxProperties":
			rv, has := dst[k]
			if !has {
				dst[k] = sv
				continue
			}
			rf, rOK := rv.(float64)
			sf, sOK := sv.(float64)
			if rOK && sOK {
				if sf < rf {
					dst[k] = sf
				}
			} else if !reflect.DeepEqual(rv, sv) && dropped != nil {
				dropped[k] = true
			}
		case "items":
			rv, has := dst[k]
			if !has {
				dst[k] = sv
				continue
			}
			rm, rOK := rv.(map[string]any)
			sm, sOK := sv.(map[string]any)
			if rOK && sOK {
				mergeRefOpenAPI(rm, sm, dropped)
			} else if !reflect.DeepEqual(rv, sv) && dropped != nil {
				dropped[k] = true
			}
		default:
			rv, has := dst[k]
			if !has {
				dst[k] = sv
			} else if !reflect.DeepEqual(rv, sv) && dropped != nil {
				dropped[k] = true
			}
		}
	}
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
