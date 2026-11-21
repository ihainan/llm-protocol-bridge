package anthropic

import (
	"strings"

	"github.com/ihainan/llm-protocol-bridge/internal/jsonx"
	"github.com/ihainan/llm-protocol-bridge/ir"
)

// ParseError decodes an upstream Anthropic error response.
func ParseError(status int, body []byte) *ir.APIError {
	e := &ir.APIError{Status: status, Type: ir.TypeForStatus(status), Raw: append([]byte(nil), body...)}
	var resp wireErrorResponse
	if jsonx.Kind(body) == '{' {
		kvs, err := jsonx.Object(body)
		if err == nil {
			for _, kv := range kvs {
				switch kv.Key {
				case "error":
					var ae wireAPIError
					if kv2, err2 := jsonx.Object(kv.Value); err2 == nil {
						for _, kv3 := range kv2 {
							switch kv3.Key {
							case "type":
								ae.Type, _ = jsonx.String(kv3.Value)
							case "message":
								ae.Message, _ = jsonx.String(kv3.Value)
							}
						}
					}
					resp.Error = ae
				}
			}
		}
	}
	if resp.Error.Message != "" {
		e.Message = resp.Error.Message
		e.Type = mapAnthropicErrorType(resp.Error.Type)
		e.Type = refineAnthropicType(e.Type, e.Message)
	}
	return e
}

// BuildError serialises an ir.APIError to an Anthropic error response.
func BuildError(err *ir.APIError) (int, []byte) {
	status := err.Status
	if status == 0 {
		status = ir.StatusForType(err.Type)
	}
	body, _ := jsonx.Marshal(wireErrorResponse{
		Type: "error",
		Error: wireAPIError{
			Type:    anthropicErrorType(err.Type),
			Message: err.Message,
		},
	})
	return status, body
}

// BuildStreamError writes an in-stream Anthropic error SSE event.
func BuildStreamError(dst []byte, err *ir.APIError) []byte {
	body, _ := jsonx.Marshal(struct {
		Type  string       `json:"type"`
		Error wireAPIError `json:"error"`
	}{
		Type: "error",
		Error: wireAPIError{
			Type:    anthropicErrorType(err.Type),
			Message: err.Message,
		},
	})
	dst = append(dst, "event: error\ndata: "...)
	dst = append(dst, body...)
	dst = append(dst, '\n', '\n')
	return dst
}

func mapAnthropicErrorType(t string) ir.ErrorType {
	switch t {
	case "invalid_request_error":
		return ir.ErrInvalidRequest
	case "authentication_error":
		return ir.ErrAuthentication
	case "billing_error":
		return ir.ErrBilling
	case "permission_error":
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
	case "api_error":
		return ir.ErrServer
	}
	return ir.ErrServer
}

func anthropicErrorType(t ir.ErrorType) string {
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
	}
	return "api_error"
}

func refineAnthropicType(t ir.ErrorType, msg string) ir.ErrorType {
	if ir.IsContextLength(&ir.APIError{Type: t, Message: msg}) {
		return ir.ErrRequestTooLarge
	}
	if strings.Contains(msg, "credit") || strings.Contains(msg, "billing") {
		return ir.ErrBilling
	}
	return t
}
