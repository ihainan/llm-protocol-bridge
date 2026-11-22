package responses

import (
	"encoding/json"
	"strings"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// ParseError decodes an upstream Responses API error response body.
func ParseError(status int, body []byte) *ir.APIError {
	e := &ir.APIError{Status: status, Type: ir.TypeForStatus(status), Raw: body}
	var resp wireErrorResponse
	if json.Unmarshal(body, &resp) == nil && resp.Error.Message != "" {
		e.Message = resp.Error.Message
		e.Param = resp.Error.Param
		if resp.Error.Type != "" {
			e.Type = mapErrorType(resp.Error.Type)
		}
		e.Code = resp.Error.Code
		e.Type = refineType(e.Type, e.Code, e.Message)
	}
	return e
}

// BuildError serialises an ir.APIError to a Responses API error response body.
func BuildError(err *ir.APIError) (int, []byte) {
	status := err.Status
	if status == 0 {
		status = ir.StatusForType(err.Type)
	}
	body, _ := jsonx.Marshal(wireErrorResponse{
		Error: wireError{
			Message: err.Message,
			Type:    openaiErrorType(err.Type),
			Code:    err.Code,
			Param:   err.Param,
		},
	})
	return status, body
}

// BuildStreamError writes an in-stream Responses error event (response.failed).
func BuildStreamError(dst []byte, err *ir.APIError) []byte {
	body, _ := jsonx.Marshal(map[string]any{
		"type":            "response.failed",
		"sequence_number": 0,
		"response": map[string]any{
			"id":     "",
			"object": "response",
			"model":  "",
			"status": "failed",
			"output": []any{},
			"error": map[string]any{
				"code":    err.Code,
				"message": err.Message,
			},
		},
	})
	dst = append(dst, "event: response.failed\ndata: "...)
	dst = append(dst, body...)
	dst = append(dst, '\n', '\n')
	return dst
}

func openaiErrorType(t ir.ErrorType) string {
	switch t {
	case ir.ErrInvalidRequest:
		return "invalid_request_error"
	case ir.ErrAuthentication:
		return "authentication_error"
	case ir.ErrBilling:
		return "billing_error"
	case ir.ErrPermission:
		return "permission_error"
	case ir.ErrNotFound:
		return "not_found_error"
	case ir.ErrRequestTooLarge:
		return "request_too_large"
	case ir.ErrRateLimit:
		return "rate_limit_error"
	case ir.ErrOverloaded:
		return "overloaded_error"
	case ir.ErrTimeout:
		return "timeout_error"
	default:
		return "server_error"
	}
}

func mapErrorType(t string) ir.ErrorType {
	switch t {
	case "invalid_request_error":
		return ir.ErrInvalidRequest
	case "authentication_error":
		return ir.ErrAuthentication
	case "billing_error":
		return ir.ErrBilling
	case "permission_error", "permission_denied":
		return ir.ErrPermission
	case "not_found_error":
		return ir.ErrNotFound
	case "request_too_large":
		return ir.ErrRequestTooLarge
	case "rate_limit_error":
		return ir.ErrRateLimit
	case "overloaded_error":
		return ir.ErrOverloaded
	case "timeout_error":
		return ir.ErrTimeout
	case "server_error", "api_error":
		return ir.ErrServer
	}
	return ir.ErrServer
}

func refineType(t ir.ErrorType, code, msg string) ir.ErrorType {
	if ir.IsContextLength(&ir.APIError{Type: t, Code: code, Message: msg}) {
		return ir.ErrRequestTooLarge
	}
	if code == "insufficient_quota" || strings.Contains(msg, "quota") {
		return ir.ErrBilling
	}
	return t
}
