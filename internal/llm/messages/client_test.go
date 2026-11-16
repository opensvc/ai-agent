package messages

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/opensvc/ai-agent/internal/llm"
)

func testRequest() llm.Request {
	return llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Text: "hello"}}}
}

func newTestClient(t *testing.T, baseURL, authMode string, source TokenSource, httpClient *http.Client) *Client {
	t.Helper()
	client, err := New(Config{BaseURL: baseURL, Model: "test-model", AuthMode: authMode, TokenSource: source, Timeout: time.Minute, MaxOutputTokens: 512}, httpClient)
	if err != nil {
		t.Fatal(err)
	}
	return client
}

func TestClientStreamsWithProtocolAndAuthenticationHeaders(t *testing.T) {
	for _, mode := range []string{AuthModeAPIKey, AuthModeBearer, AuthModeNone} {
		t.Run(mode, func(t *testing.T) {
			var requests atomic.Int64
			server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				if r.Method != http.MethodPost || r.URL.Path != "/v1/messages" || r.Header.Get("anthropic-version") != apiVersion || r.Header.Get("Accept") != "text/event-stream" || r.Header.Get("Content-Type") != "application/json" {
					t.Errorf("unexpected request: %s %s headers=%v", r.Method, r.URL.Path, r.Header)
				}
				key, bearer := "", ""
				if mode == AuthModeAPIKey {
					key = "test-secret"
				}
				if mode == AuthModeBearer {
					bearer = "Bearer test-secret"
				}
				if r.Header.Get("x-api-key") != key || r.Header.Get("Authorization") != bearer {
					t.Error("unexpected authentication headers")
				}
				var body createRequest
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				if body.Model != "test-model" || body.MaxTokens != 512 || !body.Stream || len(body.Messages) != 1 || body.Messages[0].Content[0].Text != "hello" {
					t.Errorf("unexpected body: %#v", body)
				}
				w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
				_, _ = fmt.Fprint(w, textStream())
			}))
			t.Cleanup(server.Close)
			client := newTestClient(t, server.URL+"/v1/", mode, func() (string, error) { return "test-secret", nil }, server.Client())
			var completed bool
			if err := client.Stream(t.Context(), testRequest(), func(event llm.Event) error { completed = completed || event.Type == llm.EventCompleted; return nil }); err != nil {
				t.Fatal(err)
			}
			if !completed || requests.Load() != 1 {
				t.Fatalf("completed=%v requests=%d", completed, requests.Load())
			}
		})
	}
}

func TestClientRejectsInvalidRequestsBeforeNetwork(t *testing.T) {
	var requests atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { requests.Add(1) }))
	t.Cleanup(server.Close)
	client := newTestClient(t, server.URL, AuthModeNone, nil, server.Client())
	if err := client.Stream(t.Context(), llm.Request{}, func(llm.Event) error { return nil }); err == nil {
		t.Fatal("invalid request accepted")
	}
	if err := client.Stream(t.Context(), testRequest(), nil); err == nil {
		t.Fatal("nil consumer accepted")
	}
	oversized := testRequest()
	oversized.Messages[0].Text = strings.Repeat("x", maxRequestBytes)
	if err := client.Stream(t.Context(), oversized, func(llm.Event) error { return nil }); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("oversized request error = %v", err)
	}
	for _, token := range []string{"", "secret\r\nx-injected: yes", "secret\n", "secret\x00", "secret with spaces"} {
		client := newTestClient(t, server.URL, AuthModeAPIKey, func() (string, error) { return token, nil }, server.Client())
		err := client.Stream(t.Context(), testRequest(), func(llm.Event) error { return nil })
		if err == nil || strings.Contains(err.Error(), "secret") {
			t.Fatalf("invalid token error = %v", err)
		}
	}
	if requests.Load() != 0 {
		t.Fatalf("server received %d requests", requests.Load())
	}
}

func TestClientDoesNotFollowRedirects(t *testing.T) {
	var forwarded atomic.Int64
	target := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { forwarded.Add(1) }))
	t.Cleanup(target.Close)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	t.Cleanup(server.Close)
	client := newTestClient(t, server.URL, AuthModeAPIKey, func() (string, error) { return "test-secret", nil }, server.Client())
	err := client.Stream(t.Context(), testRequest(), func(llm.Event) error { return nil })
	if err == nil || !strings.Contains(err.Error(), "307") || forwarded.Load() != 0 {
		t.Fatalf("error=%v forwarded=%d", err, forwarded.Load())
	}
}

func TestClientErrorsAreBoundedAndDoNotExposeProviderPayload(t *testing.T) {
	// Invalid redirect headers are processed by net/http before CheckRedirect.
	t.Run("malformed redirect", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Location", "http://%test-secret")
			w.WriteHeader(http.StatusTemporaryRedirect)
		}))
		t.Cleanup(server.Close)
		client := newTestClient(t, server.URL, AuthModeAPIKey, func() (string, error) { return "test-secret", nil }, server.Client())
		err := client.Stream(t.Context(), testRequest(), func(llm.Event) error { return nil })
		if err == nil || strings.Contains(err.Error(), "test-secret") {
			t.Fatalf("unsafe error: %v", err)
		}
	})
	for name, test := range map[string]struct {
		status                  int
		contentType, body, want string
	}{
		"auth":           {401, "application/json", `{"error":{"type":"authentication_error","message":"test-secret user prompt"}}`, "401 (authentication_error)"},
		"untrusted type": {500, "application/json", `{"error":{"type":"test-secret","message":"user prompt"}}`, "provider_error"},
		"non JSON":       {502, "text/html", "test-secret user prompt", "502"},
		"oversized":      {500, "text/plain", strings.Repeat("x", maxErrorBodyBytes+1), "oversized"},
		"not SSE":        {200, "application/json", `{"content":"user prompt"}`, "expected text/event-stream"},
	} {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", test.contentType)
				w.WriteHeader(test.status)
				_, _ = io.WriteString(w, test.body)
			}))
			t.Cleanup(server.Close)
			client := newTestClient(t, server.URL, AuthModeAPIKey, func() (string, error) { return "test-secret", nil }, server.Client())
			err := client.Stream(t.Context(), testRequest(), func(llm.Event) error { return nil })
			if err == nil || !strings.Contains(err.Error(), test.want) || strings.Contains(err.Error(), "test-secret") || strings.Contains(err.Error(), "user prompt") {
				t.Fatalf("error = %v", err)
			}
		})
	}
}

func TestClientCancellationClosesStream(t *testing.T) {
	for _, duringStream := range []bool{false, true} {
		t.Run(fmt.Sprint(duringStream), func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				if duringStream {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = io.WriteString(w, sse(startEvent, textStartEvent, textDeltaEvent))
					w.(http.Flusher).Flush()
				} else {
					cancel()
				}
				<-r.Context().Done()
			}))
			t.Cleanup(server.Close)
			client := newTestClient(t, server.URL, AuthModeNone, nil, server.Client())
			err := client.Stream(ctx, testRequest(), func(llm.Event) error { cancel(); return nil })
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("error = %v, want cancellation", err)
			}
		})
	}
}

func TestClientPropagatesConsumerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, textStream())
	}))
	t.Cleanup(server.Close)
	want := errors.New("consumer stopped")
	client := newTestClient(t, server.URL, AuthModeNone, nil, server.Client())
	if err := client.Stream(t.Context(), testRequest(), func(llm.Event) error { return want }); !errors.Is(err, want) {
		t.Fatalf("error=%v", err)
	}
}

func TestNewValidatesConfiguration(t *testing.T) {
	valid := Config{BaseURL: "https://api.example.test/v1", Model: "model", AuthMode: AuthModeNone, Timeout: time.Minute, MaxOutputTokens: 1}
	for name, mutate := range map[string]func(*Config){
		"invalid scheme":        func(c *Config) { c.BaseURL = "file:///tmp/provider" },
		"missing host":          func(c *Config) { c.BaseURL = "https:///v1" },
		"remote HTTP":           func(c *Config) { c.BaseURL = "http://provider.test/v1" },
		"userinfo":              func(c *Config) { c.BaseURL = "https://user:password@provider.test/v1" },
		"query":                 func(c *Config) { c.BaseURL += "?key=value" },
		"empty query":           func(c *Config) { c.BaseURL += "?" },
		"fragment":              func(c *Config) { c.BaseURL += "#fragment" },
		"model":                 func(c *Config) { c.Model = " " },
		"timeout":               func(c *Config) { c.Timeout = 0 },
		"output limit":          func(c *Config) { c.MaxOutputTokens = 0 },
		"auth":                  func(c *Config) { c.AuthMode = "basic" },
		"missing key source":    func(c *Config) { c.AuthMode = AuthModeAPIKey },
		"missing bearer source": func(c *Config) { c.AuthMode = AuthModeBearer },
	} {
		t.Run(name, func(t *testing.T) {
			config := valid
			mutate(&config)
			if _, err := New(config, nil); err == nil {
				t.Fatal("invalid configuration accepted")
			}
		})
	}
	for _, baseURL := range []string{"https://api.example.test/v1/", "http://127.0.0.1:1234/v1", "http://[::1]:1234/v1"} {
		config := valid
		config.BaseURL = baseURL
		client, err := New(config, nil)
		if err != nil || !strings.HasSuffix(client.endpoint, "/v1/messages") {
			t.Fatalf("New(%q) = %v error=%v", baseURL, client, err)
		}
	}
}

// An overloaded or rate-limited provider may answer a later request: the
// agent sends it again after the wait the provider asks for.
func TestClientErrorsTellWhetherTheyAreTransient(t *testing.T) {
	for name, test := range map[string]struct {
		status     int
		retryAfter string
		body       string
		transient  bool
		wait       time.Duration
	}{
		"overloaded":     {529, "", `{"error":{"type":"overloaded_error"}}`, true, 0},
		"rate limited":   {429, "7", `{"error":{"type":"rate_limit_error"}}`, true, 7 * time.Second},
		"refused":        {400, "7", `{"error":{"type":"invalid_request_error"}}`, false, 7 * time.Second},
		"stream error":   {200, "", "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\"}}\n\n", true, 0},
		"stream refusal": {200, "", "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\"}}\n\n", false, 0},
	} {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				if test.status == http.StatusOK {
					w.Header().Set("Content-Type", "text/event-stream")
				}
				if test.retryAfter != "" {
					w.Header().Set("Retry-After", test.retryAfter)
				}
				w.WriteHeader(test.status)
				_, _ = io.WriteString(w, test.body)
			}))
			t.Cleanup(server.Close)
			client := newTestClient(t, server.URL, AuthModeAPIKey, func() (string, error) { return "test-secret", nil }, server.Client())
			err := client.Stream(t.Context(), testRequest(), func(llm.Event) error { return nil })
			var provider *llm.ProviderError
			if !errors.As(err, &provider) || provider.Transient != test.transient || provider.RetryAfter != test.wait {
				t.Fatalf("error = %#v", err)
			}
		})
	}
}
