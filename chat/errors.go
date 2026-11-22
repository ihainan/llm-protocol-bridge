package chat

import (
	"encoding/json"
	"strings"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// ParseError decodes an upstream Chat Completions error response.
func ParseError(status int, body []byte) *ir.APIError {
	e := &ir.APIError{Status: status, Type: ir.TypeForStatus(status), Raw: body}
	var resp wireErrorResponse
	if json.Unmarshal(body, &resp) == nil && resp.Error.Message != "" {
		e.Message = resp.Error.Message
		e.Param = resp.Error.Param
		if resp.Error.Type != "" {
			e.Type = mapErrorType(resp.Error.Type)
		}
		// code may be a string or null
		if len(resp.Error.Code) > 0 {
			s, ok := jsonx.String(resp.Error.Code)
			if ok {
				e.Code = s
			}
		}
		e.Type = refineType(e.Type, e.Code, e.Message)
	}
	return e
}

// BuildError serialises an ir.APIError to a Chat Completions error response body.
func BuildError(err *ir.APIError) (int, []byte) {
	status := err.Status
	if status == 0 {
		status = ir.StatusForType(err.Type)
	}
	body, _ := jsonx.Marshal(wireErrorResponse{
		Error: wireError{
			Message: err.Message,
			Type:    openaiErrorType(err.Type),
			Code:    codeJSON(err.Code),
			Param:   err.Param,
		},
	})
	return status, body
}

// BuildStreamError writes an in-stream Chat Completions error event.
// The result is an SSE data frame: data: {"error":{...}}\n\n
func BuildStreamError(dst []byte, err *ir.APIError) []byte {
	body, _ := jsonx.Marshal(struct {
		Error wireError `json:"error"`
	}{
		Error: wireError{
			Message: err.Message,
			Type:    openaiErrorType(err.Type),
			Code:    codeJSON(err.Code),
			Param:   err.Param,
		},
	})
	dst = append(dst, "data: "...)
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

// mapErrorType maps an OpenAI error type string to ir.ErrorType.
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

// refineType uses the error code and message to narrow the type further.
func refineType(t ir.ErrorType, code, msg string) ir.ErrorType {
	if ir.IsContextLength(&ir.APIError{Type: t, Code: code, Message: msg}) {
		return ir.ErrRequestTooLarge
	}
	if code == "insufficient_quota" || strings.Contains(msg, "quota") {
		return ir.ErrBilling
	}
	return t
}

// codeJSON encodes a code string as JSON (null when empty).
func codeJSON(code string) json.RawMessage {
	if code == "" {
		return json.RawMessage("null")
	}
	b, _ := jsonx.Marshal(code)
	return b
}
