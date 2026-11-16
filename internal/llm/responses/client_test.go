package responses

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/opensvc/ai-agent/internal/llm"
)

func TestClientStreamsTextAndUsage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost || request.URL.Path != "/v1/responses" {
			t.Errorf("got %s %s", request.Method, request.URL.Path)
		}
		if got := request.Header.Get("Authorization"); got != "" {
			t.Errorf("unexpected Authorization header %q", got)
		}
		if got := request.Header.Get("Accept"); got != "text/event-stream" {
			t.Errorf("got Accept %q", got)
		}

		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if body["model"] != "test-model" || body["stream"] != true || body["store"] != false {
			t.Errorf("unexpected request body %#v", body)
		}
		if body["max_output_tokens"] != float64(1024) {
			t.Errorf("got max_output_tokens %#v", body["max_output_tokens"])
		}

		response.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		_, _ = fmt.Fprint(response, "event: response.output_text.delta\r\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"cluster \"}\r\n\r\n")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"healthy\"}\r\n\r\n")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":2,\"total_tokens\":12,\"input_tokens_details\":{\"cached_tokens\":6}},\"output\":[]}}\r\n\r\n")
	}))
	t.Cleanup(server.Close)

	client := newTestClient(t, server.URL+"/v1", AuthModeNone, nil, server.Client())
	var events []llm.Event
	err := client.Stream(t.Context(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Text: "health"}}}, func(event llm.Event) error {
		events = append(events, event)
		return nil
	})
	if err != nil {
		t.Fatalf("stream response: %v", err)
	}
	if len(events) != 4 {
		t.Fatalf("got %d events: %#v", len(events), events)
	}
	if events[0].TextDelta+events[1].TextDelta != "cluster healthy" {
		t.Fatalf("unexpected text events %#v", events[:2])
	}
	if events[2].Type != llm.EventUsage || events[2].Usage.TotalTokens != 12 || events[2].Usage.CacheReadTokens != 6 {
		t.Fatalf("unexpected usage event %#v", events[2])
	}
	if events[3].Type != llm.EventCompleted || events[3].FinishReason != llm.FinishReasonCompleted {
		t.Fatalf("unexpected completion %#v", events[3])
	}
}

func TestClientStreamsCompleteToolCallWithBearer(t *testing.T) {
	const token = "redaction-marker"
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if got := request.Header.Get("Authorization"); got != "Bearer "+token {
			t.Errorf("got Authorization %q", got)
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		tools, ok := body["tools"].([]any)
		if !ok || len(tools) != 1 || body["tool_choice"] != "auto" {
			t.Errorf("unexpected tools body %#v", body)
		}

		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"call-1\",\"name\":\"cluster_health\",\"arguments\":\"\"}}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\"}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"}\"}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.function_call_arguments.done\",\"output_index\":0,\"arguments\":\"{}\"}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"call-1\",\"name\":\"cluster_health\",\"arguments\":\"{}\"}}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"type\":\"function_call\",\"call_id\":\"call-1\",\"name\":\"cluster_health\",\"arguments\":\"{}\"}],\"usage\":{\"input_tokens\":20,\"output_tokens\":5,\"total_tokens\":25}}}\n\n")
	}))
	t.Cleanup(server.Close)

	client := newTestClient(t, server.URL, AuthModeBearer, func() (string, error) { return token, nil }, server.Client())
	request := llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Text: "check health"}},
		Tools:    []llm.Tool{{Name: "cluster_health", Description: "Get health", InputSchema: json.RawMessage(`{"type":"object","properties":{}}`)}},
	}
	var events []llm.Event
	if err := client.Stream(t.Context(), request, func(event llm.Event) error {
		events = append(events, event)
		return nil
	}); err != nil {
		t.Fatalf("stream tool call: %v", err)
	}
	if len(events) != 3 {
		t.Fatalf("got %d events, want tool call, usage and completion: %#v", len(events), events)
	}
	if events[0].Type != llm.EventToolCall || events[0].ToolCall.ID != "call-1" || string(events[0].ToolCall.Arguments) != "{}" {
		t.Fatalf("unexpected tool call %#v", events[0])
	}
	if events[2].FinishReason != llm.FinishReasonToolCalls {
		t.Fatalf("unexpected finish reason %q", events[2].FinishReason)
	}
}

func TestNewCreateRequestMapsConversation(t *testing.T) {
	request := llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Text: "diagnose"},
			{Role: llm.RoleUser, Text: "health"},
			{Role: llm.RoleAssistant, Text: "checking", ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "cluster_health", Arguments: json.RawMessage(`{}`)}}},
			{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call-1", Content: json.RawMessage(`{"status":"healthy"}`)}}},
		},
		Tools: []llm.Tool{{Name: "cluster_health", InputSchema: json.RawMessage(`{"type":"object"}`)}},
	}
	wire, err := newCreateRequest("test-model", 512, request)
	if err != nil {
		t.Fatalf("map request: %v", err)
	}
	if len(wire.Input) != 5 || len(wire.Tools) != 1 || wire.Store || !wire.Stream {
		t.Fatalf("unexpected mapped request %#v", wire)
	}
	call, ok := wire.Input[3].(functionCallInput)
	if !ok || call.CallID != "call-1" || call.Name != "cluster_health" {
		t.Fatalf("unexpected function call %#v", wire.Input[3])
	}
	result, ok := wire.Input[4].(functionOutputInput)
	if !ok || result.CallID != "call-1" || !strings.Contains(result.Output, `"is_error":false`) {
		t.Fatalf("unexpected function output %#v", wire.Input[4])
	}
}

func TestClientRejectsProviderAndStreamErrorsWithoutExposingToken(t *testing.T) {
	const token = "redaction-marker"
	for _, test := range []struct {
		name        string
		status      int
		contentType string
		body        string
		want        string
	}{
		{name: "HTTP error", status: http.StatusUnauthorized, contentType: "application/json", body: `{"error":{"code":"unauthorized","message":"bad redaction-marker"}}`, want: "HTTP 401"},
		{name: "provider event", status: http.StatusOK, contentType: "text/event-stream", body: "data: {\"type\":\"error\",\"code\":\"bad_request\",\"message\":\"bad redaction-marker\"}\n\n", want: "provider error"},
		{name: "wrong content type", status: http.StatusOK, contentType: "application/json", body: `{}`, want: "Content-Type"},
		{name: "missing terminal", status: http.StatusOK, contentType: "text/event-stream", body: "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n", want: "terminal"},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
				response.Header().Set("Content-Type", test.contentType)
				response.WriteHeader(test.status)
				_, _ = fmt.Fprint(response, test.body)
			}))
			t.Cleanup(server.Close)
			client := newTestClient(t, server.URL, AuthModeBearer, func() (string, error) { return token, nil }, server.Client())
			err := client.Stream(t.Context(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Text: "health"}}}, func(llm.Event) error { return nil })
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Stream() error = %v, want containing %q", err, test.want)
			}
			if strings.Contains(err.Error(), token) {
				t.Fatalf("error exposes token: %q", err)
			}
		})
	}
}

func TestClientPropagatesConsumerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.completed\",\"response\":{}}\n\n")
	}))
	t.Cleanup(server.Close)

	want := errors.New("stop")
	client := newTestClient(t, server.URL, AuthModeNone, nil, server.Client())
	err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Text: "health"}}}, func(llm.Event) error { return want })
	if !errors.Is(err, want) {
		t.Fatalf("Stream() error = %v, want %v", err, want)
	}
}

func TestClientRejectsEmptyBearerBeforeNetwork(t *testing.T) {
	var requests atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { requests.Add(1) }))
	t.Cleanup(server.Close)
	client := newTestClient(t, server.URL, AuthModeBearer, func() (string, error) { return "", nil }, server.Client())
	err := client.Stream(t.Context(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Text: "health"}}}, func(llm.Event) error { return nil })
	if err == nil || !strings.Contains(err.Error(), "token is empty") {
		t.Fatalf("Stream() error = %v", err)
	}
	if requests.Load() != 0 {
		t.Fatalf("server received %d requests", requests.Load())
	}
}

func TestNewValidatesResponsesConfig(t *testing.T) {
	for _, test := range []struct {
		name    string
		config  Config
		wantErr bool
	}{
		{name: "remote TLS", config: Config{BaseURL: "https://llm.example.test/v1", Model: "model", AuthMode: AuthModeNone, Timeout: time.Minute, MaxOutputTokens: 1}},
		{name: "local HTTP", config: Config{BaseURL: "http://127.0.0.1:8080/v1", Model: "model", AuthMode: AuthModeNone, Timeout: time.Minute, MaxOutputTokens: 1}},
		{name: "remote HTTP", config: Config{BaseURL: "http://llm.example.test/v1", Model: "model", AuthMode: AuthModeNone, Timeout: time.Minute, MaxOutputTokens: 1}, wantErr: true},
		{name: "empty model", config: Config{BaseURL: "https://llm.example.test/v1", AuthMode: AuthModeNone, Timeout: time.Minute, MaxOutputTokens: 1}, wantErr: true},
		{name: "missing bearer source", config: Config{BaseURL: "https://llm.example.test/v1", Model: "model", AuthMode: AuthModeBearer, Timeout: time.Minute, MaxOutputTokens: 1}, wantErr: true},
		{name: "invalid auth", config: Config{BaseURL: "https://llm.example.test/v1", Model: "model", AuthMode: "basic", Timeout: time.Minute, MaxOutputTokens: 1}, wantErr: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := New(test.config, nil)
			if (err != nil) != test.wantErr {
				t.Fatalf("New() error = %v, wantErr=%v", err, test.wantErr)
			}
		})
	}
}

func newTestClient(t *testing.T, baseURL string, authMode string, tokenSource TokenSource, httpClient *http.Client) *Client {
	t.Helper()
	client, err := New(Config{BaseURL: baseURL, Model: "test-model", AuthMode: authMode, TokenSource: tokenSource, Timeout: time.Minute, MaxOutputTokens: 1024}, httpClient)
	if err != nil {
		t.Fatalf("create Responses client: %v", err)
	}
	return client
}

// The reasoning effort is sent when configured, and left out otherwise, so
// that the model applies its own.
func TestClientSendsTheConfiguredReasoningEffort(t *testing.T) {
	for _, effort := range []string{"", "low"} {
		var body map[string]any
		server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
			_ = json.NewDecoder(request.Body).Decode(&body)
			response.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
			_, _ = fmt.Fprint(response, "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2},\"output\":[]}}\r\n\r\n")
		}))
		client, err := New(Config{BaseURL: server.URL + "/v1", Model: "test-model", AuthMode: AuthModeNone, Timeout: time.Minute, MaxOutputTokens: 1024, ReasoningEffort: effort}, server.Client())
		if err != nil {
			t.Fatal(err)
		}
		if err := client.Stream(t.Context(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Text: "health"}}}, func(llm.Event) error { return nil }); err != nil {
			t.Fatalf("effort %q: %v", effort, err)
		}
		server.Close()
		reasoning, present := body["reasoning"]
		if effort == "" && present || effort != "" && fmt.Sprint(reasoning) != "map[effort:low]" {
			t.Fatalf("effort %q: reasoning = %v (present %v)", effort, reasoning, present)
		}
	}
}

// A spent quota is answered 429 too, but no wait renews it.
func TestClientErrorsTellWhetherTheyAreTransient(t *testing.T) {
	for name, test := range map[string]struct {
		status    int
		body      string
		transient bool
	}{
		"rate limited":   {http.StatusTooManyRequests, `{"error":{"code":"rate_limit_exceeded","message":"slow down"}}`, true},
		"quota spent":    {http.StatusTooManyRequests, `{"error":{"code":"insufficient_quota","type":"insufficient_quota","message":"no quota"}}`, false},
		"quota, no text": {http.StatusTooManyRequests, `{"error":{"code":"insufficient_quota"}}`, false},
		"unavailable":    {http.StatusServiceUnavailable, `not JSON`, true},
		"unauthorized":   {http.StatusUnauthorized, `{"error":{"code":"unauthorized","message":"bad key"}}`, false},
	} {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
				response.Header().Set("Content-Type", "application/json")
				response.WriteHeader(test.status)
				_, _ = fmt.Fprint(response, test.body)
			}))
			t.Cleanup(server.Close)
			client := newTestClient(t, server.URL, AuthModeBearer, func() (string, error) { return "token", nil }, server.Client())
			err := client.Stream(t.Context(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Text: "health"}}}, func(llm.Event) error { return nil })
			var provider *llm.ProviderError
			if !errors.As(err, &provider) || provider.Transient != test.transient || provider.Status != test.status {
				t.Fatalf("error = %#v", err)
			}
		})
	}
}
