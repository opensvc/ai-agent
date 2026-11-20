package chatcompletions

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

func TestClientStreamsTextUsageAndCompletion(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost || request.URL.Path != "/v1/chat/completions" {
			t.Errorf("got %s %s", request.Method, request.URL.Path)
		}
		if got := request.Header.Get("Accept"); got != "text/event-stream" {
			t.Errorf("got Accept %q", got)
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		streamOptions, _ := body["stream_options"].(map[string]any)
		if body["model"] != "test-model" || body["stream"] != true || body["store"] != false || streamOptions["include_usage"] != true {
			t.Errorf("unexpected request body %#v", body)
		}
		if body["max_completion_tokens"] != float64(1024) {
			t.Errorf("got max_completion_tokens %#v", body["max_completion_tokens"])
		}

		response.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"cluster \"},\"finish_reason\":null}]}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"healthy\"},\"finish_reason\":null}]}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12,\"prompt_tokens_details\":{\"cached_tokens\":4}}}\n\n")
		_, _ = fmt.Fprint(response, "data: [DONE]\n\n")
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
	if events[2].Type != llm.EventUsage || events[2].Usage.TotalTokens != 12 || events[2].Usage.CacheReadTokens != 4 {
		t.Fatalf("unexpected usage event %#v", events[2])
	}
	if events[3].Type != llm.EventCompleted || events[3].FinishReason != llm.FinishReasonCompleted {
		t.Fatalf("unexpected completion %#v", events[3])
	}
}

func TestClientStreamsFragmentedToolCallsWithBearer(t *testing.T) {
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
		_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"type\":\"function\",\"function\":{\"name\":\"cluster_health\",\"arguments\":\"{\"}}]},\"finish_reason\":null}]}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"}\"}}]},\"finish_reason\":null}]}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":5,\"total_tokens\":25}}\n\n")
		_, _ = fmt.Fprint(response, "data: [DONE]\n\n")
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
	if events[0].Type != llm.EventToolCall || events[0].ToolCall.ID != "call-1" || events[0].ToolCall.Name != "cluster_health" || string(events[0].ToolCall.Arguments) != "{}" {
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
			{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call-1", Content: json.RawMessage(`{"status":"healthy"}`), IsError: true}}},
		},
		Tools: []llm.Tool{{Name: "cluster_health", InputSchema: json.RawMessage(`{"type":"object"}`)}},
	}
	wire, err := newCreateRequest("test-model", 512, false, request)
	if err != nil {
		t.Fatalf("map request: %v", err)
	}
	if len(wire.Messages) != 4 || len(wire.Tools) != 1 || wire.Store == nil || *wire.Store || !wire.Stream || !wire.StreamOptions.IncludeUsage {
		t.Fatalf("unexpected mapped request %#v", wire)
	}
	assistant := wire.Messages[2]
	if assistant.Content == nil || *assistant.Content != "checking" || len(assistant.ToolCalls) != 1 || assistant.ToolCalls[0].Function.Name != "cluster_health" {
		t.Fatalf("unexpected assistant message %#v", assistant)
	}
	result := wire.Messages[3]
	if result.ToolCallID != "call-1" || result.Content == nil || !strings.Contains(*result.Content, `"is_error":true`) {
		t.Fatalf("unexpected tool message %#v", result)
	}
	if wire.Tools[0].Type != "function" || wire.Tools[0].Function.Name != "cluster_health" {
		t.Fatalf("unexpected function tool %#v", wire.Tools[0])
	}
}

// A reasoning model of Mistral streams its text as a list of typed parts:
// the text parts are the answer, its reasoning, a part of type thinking, is
// never emitted.
func TestClientReadsTextPartsAndLeavesReasoningOut(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":[{\"type\":\"thinking\",\"thinking\":[{\"type\":\"text\",\"text\":\"secret plan\"}],\"closed\":true}]},\"finish_reason\":null}]}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":[{\"type\":\"text\",\"text\":\"cluster \"}]},\"finish_reason\":null}]}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"healthy\"},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = fmt.Fprint(response, "data: [DONE]\n\n")
	}))
	t.Cleanup(server.Close)
	client := newTestClient(t, server.URL+"/v1", AuthModeNone, nil, server.Client())
	var text strings.Builder
	err := client.Stream(t.Context(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Text: "health"}}}, func(event llm.Event) error {
		text.WriteString(event.TextDelta)
		return nil
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if text.String() != "cluster healthy" {
		t.Fatalf("got text %q", text.String())
	}
}

// Only the parts of type text are read: the reasoning, wherever its text
// sits, is never part of the answer.
func TestDeltaContentReadsTextPartsOnly(t *testing.T) {
	for _, test := range []struct {
		content, text string
	}{
		{`"plain"`, "plain"},
		{`""`, ""},
		{`[]`, ""},
		{`[null]`, ""},
		{`[{"type":"reference","text":"x"}]`, ""},
		{`[{"type":"thinking","text":"secret"}]`, ""},
		{`[{"type":"thinking","thinking":[{"type":"text","text":"secret"}]}]`, ""},
		{`[{"type":"thinking","text":{"nested":1}}]`, ""},
		{`[{"type":"text","text":"a"},{"type":"thinking","text":"secret"},{"type":"text","text":"b"}]`, "ab"},
		{`[{"type":"text"}]`, ""},
	} {
		var content deltaContent
		if err := json.Unmarshal([]byte(test.content), &content); err != nil || content.text != test.text {
			t.Errorf("%s: got %q, %v, want %q", test.content, content.text, err, test.text)
		}
	}
	for _, refused := range []string{`{"type":"text"}`, `5`, `true`, `["text"]`, `[[]]`, `[{"type":"text","text":5}]`} {
		var content deltaContent
		if err := json.Unmarshal([]byte(refused), &content); err == nil {
			t.Errorf("%s accepted as content", refused)
		}
	}
	// A value decoded again holds the new text only.
	var content deltaContent
	_ = json.Unmarshal([]byte(`[{"type":"text","text":"a"}]`), &content)
	if err := json.Unmarshal([]byte(`[]`), &content); err != nil || content.text != "" {
		t.Fatalf("decoded again: %q, %v", content.text, err)
	}
}

// Reasoning parts before tool calls give the calls, no text; reasoning after
// the finish reason is refused, as any data is.
func TestClientReasoningPartsAroundToolCalls(t *testing.T) {
	stream := func(lines ...string) error {
		server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
			response.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
			for _, line := range lines {
				_, _ = fmt.Fprint(response, "data: "+line+"\n\n")
			}
			_, _ = fmt.Fprint(response, "data: [DONE]\n\n")
		}))
		defer server.Close()
		client := newTestClient(t, server.URL+"/v1", AuthModeNone, nil, server.Client())
		var kinds []llm.EventType
		err := client.Stream(t.Context(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Text: "health"}}}, func(event llm.Event) error {
			if event.Type == llm.EventTextDelta {
				t.Errorf("text emitted: %q", event.TextDelta)
			}
			kinds = append(kinds, event.Type)
			return nil
		})
		if err == nil && (len(kinds) != 2 || kinds[0] != llm.EventToolCall || kinds[1] != llm.EventCompleted) {
			t.Errorf("got events %v", kinds)
		}
		return err
	}
	thinking := `{"choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"secret"}]}]},"finish_reason":null}]}`
	call := `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"cluster_health","arguments":"{}"}}]},"finish_reason":null}]}`
	finish := `{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`
	if err := stream(thinking, call, finish); err != nil {
		t.Fatalf("thinking then tool calls: %v", err)
	}
	if err := stream(call, finish, thinking); err == nil {
		t.Fatal("thinking after the finish reason accepted")
	}
}

// A provider refusing store, such as Mistral, gets requests without it when
// the administrator asks; every other request sends store false.
func TestClientOmitsStoreWhenAsked(t *testing.T) {
	for _, omit := range []bool{false, true} {
		t.Run(fmt.Sprintf("omit %v", omit), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
				var body map[string]any
				if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
					t.Errorf("decode request: %v", err)
					return
				}
				if store, sent := body["store"]; omit && sent || !omit && store != false {
					t.Errorf("got store %#v, sent %v", store, sent)
				}
				response.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
				_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n")
				_, _ = fmt.Fprint(response, "data: [DONE]\n\n")
			}))
			t.Cleanup(server.Close)
			client, err := New(Config{BaseURL: server.URL + "/v1", Model: "test-model", AuthMode: AuthModeNone, Timeout: time.Minute, MaxOutputTokens: 1024, OmitStore: omit}, server.Client())
			if err != nil {
				t.Fatalf("create client: %v", err)
			}
			if err := client.Stream(t.Context(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Text: "health"}}}, func(llm.Event) error { return nil }); err != nil {
				t.Fatalf("stream: %v", err)
			}
		})
	}
}

// The volatile text of a prompt goes after its text: the prefix the provider
// caches by tokens still holds the text for the next requests.
func TestNewCreateRequestPutsVolatileTextAfterTheText(t *testing.T) {
	wire, err := newCreateRequest("test-model", 512, false, llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Text: "health", Volatile: "<interface_context>[]</interface_context>"}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(wire.Messages) != 1 || wire.Messages[0].Content == nil || *wire.Messages[0].Content != "health\n\n<interface_context>[]</interface_context>" {
		t.Fatalf("messages = %#v", wire.Messages)
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
		{name: "provider event", status: http.StatusOK, contentType: "text/event-stream", body: "data: {\"error\":{\"code\":\"bad_request\",\"message\":\"bad redaction-marker\"}}\n\n", want: "provider error"},
		{name: "wrong content type", status: http.StatusOK, contentType: "application/json", body: `{}`, want: "Content-Type"},
		{name: "missing done", status: http.StatusOK, contentType: "text/event-stream", body: "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n", want: "[DONE]"},
		{name: "done before finish", status: http.StatusOK, contentType: "text/event-stream", body: "data: [DONE]\n\n", want: "finish reason"},
		{name: "multiple choices", status: http.StatusOK, contentType: "text/event-stream", body: "data: {\"choices\":[{\"index\":0,\"delta\":{}},{\"index\":1,\"delta\":{}}]}\n\n", want: "choices"},
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

func TestClientRejectsInconsistentToolCall(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"type\":\"function\",\"function\":{\"name\":\"first\",\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"second\"}}]},\"finish_reason\":null}]}\n\n")
	}))
	t.Cleanup(server.Close)
	client := newTestClient(t, server.URL, AuthModeNone, nil, server.Client())
	err := client.Stream(t.Context(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Text: "health"}}}, func(llm.Event) error { return nil })
	if err == nil || !strings.Contains(err.Error(), "changed name") {
		t.Fatalf("Stream() error = %v", err)
	}
}

func TestClientPropagatesConsumerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n")
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

func TestNewValidatesChatCompletionsConfig(t *testing.T) {
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
		{name: "zero timeout", config: Config{BaseURL: "https://llm.example.test/v1", Model: "model", AuthMode: AuthModeNone, MaxOutputTokens: 1}, wantErr: true},
		{name: "zero max tokens", config: Config{BaseURL: "https://llm.example.test/v1", Model: "model", AuthMode: AuthModeNone, Timeout: time.Minute}, wantErr: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := New(test.config, nil)
			if (err != nil) != test.wantErr {
				t.Fatalf("New() error = %v, wantErr %v", err, test.wantErr)
			}
		})
	}
}

func TestMapFinishReason(t *testing.T) {
	tests := map[string]llm.FinishReason{
		"stop":           llm.FinishReasonCompleted,
		"tool_calls":     llm.FinishReasonToolCalls,
		"length":         llm.FinishReasonLength,
		"content_filter": llm.FinishReasonContentFilter,
		"unexpected":     llm.FinishReasonOther,
	}
	for input, want := range tests {
		if got := mapFinishReason(input); got != want {
			t.Errorf("mapFinishReason(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestConsumeStreamEnforcesEventBound(t *testing.T) {
	part := strings.Repeat("x", maxEventBytes/3+1)
	stream := "data: " + part + "\ndata: " + part + "\ndata: " + part + "\n\n"
	err := consumeStream(strings.NewReader(stream), func(llm.Event) error { return nil }, "")
	if err == nil || !strings.Contains(err.Error(), "SSE event exceeds") {
		t.Fatalf("consumeStream() error = %v", err)
	}
}

func newTestClient(t *testing.T, baseURL string, authMode string, tokenSource TokenSource, httpClient *http.Client) *Client {
	t.Helper()
	client, err := New(Config{
		BaseURL:         baseURL,
		Model:           "test-model",
		AuthMode:        authMode,
		TokenSource:     tokenSource,
		Timeout:         time.Minute,
		MaxOutputTokens: 1024,
	}, httpClient)
	if err != nil {
		t.Fatalf("create client: %v", err)
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
			_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n")
			_, _ = fmt.Fprint(response, "data: [DONE]\n\n")
		}))
		client, err := New(Config{BaseURL: server.URL + "/v1", Model: "test-model", AuthMode: AuthModeNone, Timeout: time.Minute, MaxOutputTokens: 1024, ReasoningEffort: effort}, server.Client())
		if err != nil {
			t.Fatal(err)
		}
		if err := client.Stream(t.Context(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Text: "health"}}}, func(llm.Event) error { return nil }); err != nil {
			t.Fatalf("effort %q: %v", effort, err)
		}
		server.Close()
		value, present := body["reasoning_effort"]
		if effort == "" && present || effort != "" && value != effort {
			t.Fatalf("effort %q: reasoning_effort = %v (present %v)", effort, value, present)
		}
	}
}

// A spent quota is answered 429, with or without a message: no wait renews it.
func TestClientSpentQuotaIsNotTransient(t *testing.T) {
	for _, body := range []string{`{"error":{"code":"insufficient_quota"}}`, `{"error":{"type":"insufficient_quota","message":"no quota"}}`} {
		server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
			response.Header().Set("Content-Type", "application/json")
			response.WriteHeader(http.StatusTooManyRequests)
			_, _ = fmt.Fprint(response, body)
		}))
		client, err := New(Config{BaseURL: server.URL + "/v1", Model: "test-model", AuthMode: AuthModeNone, Timeout: time.Minute, MaxOutputTokens: 1024}, server.Client())
		if err != nil {
			t.Fatal(err)
		}
		err = client.Stream(t.Context(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Text: "health"}}}, func(llm.Event) error { return nil })
		server.Close()
		var provider *llm.ProviderError
		if !errors.As(err, &provider) || provider.Transient || provider.Status != http.StatusTooManyRequests {
			t.Fatalf("body %s: error = %#v", body, err)
		}
	}
}
