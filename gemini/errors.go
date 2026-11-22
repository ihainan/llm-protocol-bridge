package gemini

import (
	"encoding/json"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// ParseError decodes an upstream Gemini error response body.
func ParseError(status int, body []byte) *ir.APIError {
	e := &ir.APIError{Status: status, Type: ir.TypeForStatus(status), Raw: body}
	var resp wireError
	if json.Unmarshal(body, &resp) == nil && resp.Error.Message != "" {
		e.Message = resp.Error.Message
		e.Code = resp.Error.Status // Gemini uses Status as an RPC code string
		if resp.Error.Code != 0 {
			status = resp.Error.Code
			e.Status = status
			e.Type = ir.TypeForStatus(status)
		}
		e.Type = refineGeminiType(e.Type, e.Code, e.Message)
	}
	return e
}

// BuildError serialises an ir.APIError to a Gemini error response body.
func BuildError(err *ir.APIError) (int, []byte) {
	status := err.Status
	if status == 0 {
		status = ir.StatusForType(err.Type)
	}
	body, _ := jsonx.Marshal(map[string]any{
		"error": map[string]any{
			"code":    status,
			"message": err.Message,
			"status":  geminiStatusCode(err.Type),
		},
	})
	return status, body
}

// BuildStreamError writes an in-stream Gemini error event (JSON object).
func BuildStreamError(dst []byte, err *ir.APIError) []byte {
	status := err.Status
	if status == 0 {
		status = ir.StatusForType(err.Type)
	}
	body, _ := jsonx.Marshal(map[string]any{
		"error": map[string]any{
			"code":    status,
			"message": err.Message,
			"status":  geminiStatusCode(err.Type),
		},
	})
	dst = append(dst, body...)
	dst = append(dst, '\n')
	return dst
}

// BuildStreamErrorSSE writes an in-stream Gemini error framed as an SSE data event.
func BuildStreamErrorSSE(dst []byte, err *ir.APIError) []byte {
	status := err.Status
	if status == 0 {
		status = ir.StatusForType(err.Type)
	}
	body, _ := jsonx.Marshal(map[string]any{
		"error": map[string]any{
			"code":    status,
			"message": err.Message,
			"status":  geminiStatusCode(err.Type),
		},
	})
	dst = append(dst, "data: "...)
	dst = append(dst, body...)
	dst = append(dst, '\n', '\n')
	return dst
}

func refineGeminiType(t ir.ErrorType, code, msg string) ir.ErrorType {
	if ir.IsContextLength(&ir.APIError{Type: t, Code: code, Message: msg}) {
		return ir.ErrRequestTooLarge
	}
	switch code {
	case "RESOURCE_EXHAUSTED":
		return ir.ErrRateLimit
	case "INVALID_ARGUMENT":
		return ir.ErrInvalidRequest
	case "UNAUTHENTICATED":
		return ir.ErrAuthentication
	case "PERMISSION_DENIED":
		return ir.ErrPermission
	case "NOT_FOUND":
		return ir.ErrNotFound
	case "UNAVAILABLE":
		return ir.ErrOverloaded
	case "DEADLINE_EXCEEDED":
		return ir.ErrTimeout
	}
	return t
}

func geminiStatusCode(t ir.ErrorType) string {
	switch t {
	case ir.ErrInvalidRequest:
		return "INVALID_ARGUMENT"
	case ir.ErrAuthentication:
		return "UNAUTHENTICATED"
	case ir.ErrPermission:
		return "PERMISSION_DENIED"
	case ir.ErrNotFound:
		return "NOT_FOUND"
	case ir.ErrRateLimit:
		return "RESOURCE_EXHAUSTED"
	case ir.ErrOverloaded:
		return "UNAVAILABLE"
	case ir.ErrTimeout:
		return "DEADLINE_EXCEEDED"
	case ir.ErrRequestTooLarge:
		return "INVALID_ARGUMENT"
	}
	return "INTERNAL"
}
