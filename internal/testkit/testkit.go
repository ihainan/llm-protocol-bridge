// Package testkit provides shared helpers for llm-protocol-bridge tests.
package testkit

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/ihainan/llm-protocol-bridge/ir"
)

// ReadFile reads path and fails the test on error.
func ReadFile(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile %s: %v", path, err)
	}
	return b
}

// JSONEqual compares two JSON values for semantic equality (key order and
// whitespace are ignored). It reports a diff via t.Errorf and returns false on mismatch.
func JSONEqual(t *testing.T, label string, want, got []byte) bool {
	t.Helper()
	var wv, gv any
	if err := json.Unmarshal(want, &wv); err != nil {
		t.Errorf("%s: want JSON invalid: %v\n%s", label, err, want)
		return false
	}
	if err := json.Unmarshal(got, &gv); err != nil {
		t.Errorf("%s: got JSON invalid: %v\n%s", label, err, got)
		return false
	}
	wn, _ := json.Marshal(wv)
	gn, _ := json.Marshal(gv)
	if string(wn) != string(gn) {
		t.Errorf("%s:\nwant: %s\n got: %s", label, wn, gn)
		return false
	}
	return true
}

// DefaultEnv returns an Env with all default losses allowed and no special options.
func DefaultEnv(style ir.PathStyle) *ir.Env {
	return &ir.Env{Allow: ir.DefaultAllowed, Style: style}
}
