// Command liveproxy is a minimal HTTP proxy that converts LLM API requests
// and responses between wire formats using llm-protocol-bridge.
//
// Usage:
//
//	UPSTREAM_URL=https://api.anthropic.com \
//	UPSTREAM_FORMAT=anthropic \
//	UPSTREAM_KEY=... \
//	CLIENT_FORMAT=chat \
//	go run . -addr :8080
//
// The proxy receives requests in CLIENT_FORMAT, converts them to
// UPSTREAM_FORMAT, forwards them to UPSTREAM_URL, then converts the
// response (or SSE stream) back to CLIENT_FORMAT before returning it
// to the caller.
//
// Environment variables:
//
//	UPSTREAM_URL     Full base URL of the upstream provider (required).
//	UPSTREAM_FORMAT  Wire format for the upstream: chat, responses, anthropic, gemini.
//	UPSTREAM_KEY     API key sent as Authorization: Bearer <key> (optional if the
//	                 upstream uses a different auth scheme set via UPSTREAM_HEADER_*).
//	CLIENT_FORMAT    Wire format expected by the caller: chat, responses, anthropic, gemini.
//	UPSTREAM_PATH    URL path to append to UPSTREAM_URL (default depends on format).
//	UPSTREAM_HEADER_<NAME>  Extra request headers to forward to the upstream, e.g.
//	                 UPSTREAM_HEADER_X_API_KEY=... adds X-Api-Key: ...
//
// This is an example. It is not production-ready: it has no authentication
// for incoming requests, no rate limiting, and minimal error handling.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/ihainan/llm-protocol-bridge/bridge"
	"github.com/ihainan/llm-protocol-bridge/ir"
	"github.com/ihainan/llm-protocol-bridge/sse"
)

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	flag.Parse()

	cfg, err := configFromEnv()
	if err != nil {
		log.Fatalf("config: %v", err)
	}

	log.Printf("liveproxy: listening on %s, %s → %s → %s",
		*addr, cfg.clientFormat, cfg.upstreamURL, cfg.upstreamFormat)

	mux := http.NewServeMux()
	mux.HandleFunc("/", cfg.handleRequest)

	srv := &http.Server{
		Addr:         *addr,
		Handler:      mux,
		ReadTimeout:  120 * time.Second,
		WriteTimeout: 180 * time.Second,
	}
	if err := srv.ListenAndServe(); err != nil {
		log.Fatalf("server: %v", err)
	}
}

type config struct {
	clientFormat   ir.Format
	upstreamFormat ir.Format
	upstreamURL    string
	upstreamKey    string
	upstreamPath   string
	extraHeaders   map[string]string
	profile        *ir.ModelProfile
}

func configFromEnv() (*config, error) {
	upstreamURL := os.Getenv("UPSTREAM_URL")
	if upstreamURL == "" {
		return nil, fmt.Errorf("UPSTREAM_URL is required")
	}
	clientFmt, err := parseFormat(os.Getenv("CLIENT_FORMAT"), "chat")
	if err != nil {
		return nil, fmt.Errorf("CLIENT_FORMAT: %w", err)
	}
	upstreamFmt, err := parseFormat(os.Getenv("UPSTREAM_FORMAT"), "anthropic")
	if err != nil {
		return nil, fmt.Errorf("UPSTREAM_FORMAT: %w", err)
	}

	extra := map[string]string{}
	for _, kv := range os.Environ() {
		if !strings.HasPrefix(kv, "UPSTREAM_HEADER_") {
			continue
		}
		rest := strings.TrimPrefix(kv, "UPSTREAM_HEADER_")
		idx := strings.IndexByte(rest, '=')
		if idx < 0 {
			continue
		}
		name := strings.ReplaceAll(rest[:idx], "_", "-")
		extra[name] = rest[idx+1:]
	}

	var profile *ir.ModelProfile
	if rawProfile := strings.TrimSpace(os.Getenv("UPSTREAM_PROFILE_JSON")); rawProfile != "" {
		var p ir.ModelProfile
		if err := json.Unmarshal([]byte(rawProfile), &p); err != nil {
			return nil, fmt.Errorf("UPSTREAM_PROFILE_JSON: %w", err)
		}
		profile = &p
	}

	return &config{
		clientFormat:   clientFmt,
		upstreamFormat: upstreamFmt,
		upstreamURL:    strings.TrimRight(upstreamURL, "/"),
		upstreamKey:    os.Getenv("UPSTREAM_KEY"),
		upstreamPath:   os.Getenv("UPSTREAM_PATH"),
		extraHeaders:   extra,
		profile:        profile,
	}, nil
}

func parseFormat(s, def string) (ir.Format, error) {
	if s == "" {
		s = def
	}
	switch strings.ToLower(s) {
	case "chat":
		return ir.FormatChat, nil
	case "responses":
		return ir.FormatResponses, nil
	case "anthropic":
		return ir.FormatAnthropic, nil
	case "gemini":
		return ir.FormatGemini, nil
	default:
		return "", fmt.Errorf("unknown format %q; must be chat, responses, anthropic, or gemini", s)
	}
}

func (cfg *config) handleRequest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "only POST is supported", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, 8<<20))
	if err != nil {
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}

	convOpts := bridge.Options{
		Profile: cfg.profile,
	}
	if cfg.clientFormat == ir.FormatGemini {
		convOpts.Model, convOpts.Stream = parseGeminiRequestPath(r.URL.Path)
		convOpts.AltSSE = r.URL.Query().Get("alt") == "sse"
	}

	plan, err := bridge.Convert(cfg.clientFormat, cfg.upstreamFormat, body, convOpts)
	if err != nil {
		log.Printf("convert: %v", err)
		writeClientError(w, cfg.clientFormat, err)
		return
	}

	upstreamPath := cfg.upstreamPath
	if upstreamPath == "" {
		upstreamPath = defaultPath(cfg.upstreamFormat, plan.Model, plan.UpstreamStream)
	}
	upstreamURL := cfg.upstreamURL + upstreamPath

	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, upstreamURL, strings.NewReader(string(plan.Body)))
	if err != nil {
		writeClientError(w, cfg.clientFormat, err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	if cfg.upstreamKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.upstreamKey)
	}
	for k, v := range cfg.extraHeaders {
		req.Header.Set(k, v)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		log.Printf("upstream request: %v", err)
		writeClientError(w, cfg.clientFormat, &ir.APIError{Status: http.StatusBadGateway, Type: ir.ErrServer, Message: "upstream request failed"})
		return
	}
	defer resp.Body.Close()

	if !plan.ClientStream {
		cfg.handleNonStream(w, resp, plan)
		return
	}
	cfg.handleStream(w, resp, plan, convOpts.AltSSE)
}

func (cfg *config) handleNonStream(w http.ResponseWriter, resp *http.Response, plan *bridge.Plan) {
	upBody, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		writeClientError(w, cfg.clientFormat, &ir.APIError{Status: http.StatusBadGateway, Type: ir.ErrServer, Message: "failed to read upstream response"})
		return
	}
	if resp.StatusCode != http.StatusOK {
		st, errBody := plan.ConvertError(resp.StatusCode, upBody)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(st)
		w.Write(errBody)
		return
	}
	out, _, err := plan.ConvertResponse(upBody)
	if err != nil {
		log.Printf("response convert: %v", err)
		writeClientError(w, cfg.clientFormat, err)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(out)
}

func (cfg *config) handleStream(w http.ResponseWriter, resp *http.Response, plan *bridge.Plan, altSSE bool) {
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
		st, errBody := plan.ConvertError(resp.StatusCode, body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(st)
		w.Write(errBody)
		return
	}

	if cfg.clientFormat == ir.FormatGemini && !altSSE {
		w.Header().Set("Content-Type", "application/json")
	} else {
		w.Header().Set("Content-Type", "text/event-stream")
	}
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher, canFlush := w.(http.Flusher)

	s := plan.NewStream()
	r := sse.NewReader(resp.Body)

	var out []byte
	endKind := ir.EndDone
	for {
		ev, err := r.Next()
		if err != nil {
			if errors.Is(err, io.EOF) {
				endKind = ir.EndEOF
			} else if isTimeout(err) {
				endKind = ir.EndTimeout
			} else {
				endKind = ir.EndError
			}
			break
		}
		if len(ev.Data) == 0 && ev.Name == "" {
			continue
		}
		out, err = s.Feed(out[:0], ev)
		if err != nil {
			log.Printf("stream feed: %v", err)
			endKind = ir.EndError
			break
		}
		if len(out) > 0 {
			w.Write(out)
			if canFlush {
				flusher.Flush()
			}
		}
	}

	out, err := s.End(out[:0], endKind)
	if err != nil {
		log.Printf("stream end: %v", err)
	}
	if len(out) > 0 {
		w.Write(out)
		if canFlush {
			flusher.Flush()
		}
	}
}

func writeClientError(w http.ResponseWriter, client ir.Format, err error) {
	st, body := bridge.ErrorResponse(client, err)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(st)
	w.Write(body)
}

func isTimeout(err error) bool {
	var ne net.Error
	return errors.As(err, &ne) && ne.Timeout()
}

func parseGeminiRequestPath(path string) (model string, stream bool) {
	if idx := strings.Index(path, "/models/"); idx >= 0 {
		rest := path[idx+len("/models/"):]
		if colon := strings.IndexByte(rest, ':'); colon >= 0 {
			model = rest[:colon]
			action := rest[colon+1:]
			stream = strings.HasPrefix(action, "streamGenerateContent")
			return model, stream
		}
		model = strings.Trim(rest, "/")
	}
	return model, false
}

func defaultPath(f ir.Format, model string, stream bool) string {
	switch f {
	case ir.FormatChat:
		return "/v1/chat/completions"
	case ir.FormatResponses:
		return "/v1/responses"
	case ir.FormatAnthropic:
		return "/v1/messages"
	case ir.FormatGemini:
		m := url.PathEscape(model)
		if m == "" {
			m = "gemini-2.5-flash"
		}
		if stream {
			return "/v1beta/models/" + m + ":streamGenerateContent?alt=sse"
		}
		return "/v1beta/models/" + m + ":generateContent"
	default:
		return "/"
	}
}
