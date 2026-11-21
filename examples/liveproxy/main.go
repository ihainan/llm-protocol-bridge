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
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
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

	return &config{
		clientFormat:   clientFmt,
		upstreamFormat: upstreamFmt,
		upstreamURL:    strings.TrimRight(upstreamURL, "/"),
		upstreamKey:    os.Getenv("UPSTREAM_KEY"),
		upstreamPath:   os.Getenv("UPSTREAM_PATH"),
		extraHeaders:   extra,
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

	plan, err := bridge.Convert(cfg.clientFormat, cfg.upstreamFormat, body, bridge.Options{})
	if err != nil {
		log.Printf("convert: %v", err)
		http.Error(w, fmt.Sprintf("conversion error: %v", err), http.StatusUnprocessableEntity)
		return
	}

	upstreamPath := cfg.upstreamPath
	if upstreamPath == "" {
		upstreamPath = defaultPath(cfg.upstreamFormat, plan.UpstreamStream)
	}
	upstreamURL := cfg.upstreamURL + upstreamPath

	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, upstreamURL, strings.NewReader(string(plan.Body)))
	if err != nil {
		http.Error(w, "failed to build upstream request", http.StatusInternalServerError)
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
		http.Error(w, "upstream request failed", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if !plan.ClientStream {
		cfg.handleNonStream(w, resp, plan)
		return
	}
	cfg.handleStream(w, resp, plan)
}

func (cfg *config) handleNonStream(w http.ResponseWriter, resp *http.Response, plan *bridge.Plan) {
	upBody, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		http.Error(w, "failed to read upstream response", http.StatusBadGateway)
		return
	}
	if resp.StatusCode != http.StatusOK {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		w.Write(upBody)
		return
	}
	out, _, err := plan.ConvertResponse(upBody)
	if err != nil {
		log.Printf("response convert: %v", err)
		http.Error(w, fmt.Sprintf("response conversion error: %v", err), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(out)
}

func (cfg *config) handleStream(w http.ResponseWriter, resp *http.Response, plan *bridge.Plan) {
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		w.Write(body)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher, canFlush := w.(http.Flusher)

	s := plan.NewStream()
	r := sse.NewReader(resp.Body)

	var out []byte
	for {
		ev, err := r.Next()
		if err != nil {
			break
		}
		if len(ev.Data) == 0 && ev.Name == "" {
			continue
		}
		out, err = s.Feed(out[:0], ev)
		if err != nil {
			log.Printf("stream feed: %v", err)
			break
		}
		if len(out) > 0 {
			w.Write(out)
			if canFlush {
				flusher.Flush()
			}
		}
	}

	out, err := s.End(out[:0], ir.EndDone)
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

func defaultPath(f ir.Format, stream bool) string {
	switch f {
	case ir.FormatChat:
		return "/v1/chat/completions"
	case ir.FormatResponses:
		return "/v1/responses"
	case ir.FormatAnthropic:
		return "/v1/messages"
	case ir.FormatGemini:
		if stream {
			return "/v1beta/models:streamGenerateContent?alt=sse"
		}
		return "/v1beta/models:generateContent"
	default:
		return "/"
	}
}
