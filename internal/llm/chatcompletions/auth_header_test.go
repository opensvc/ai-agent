package chatcompletions

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/opensvc/ai-agent/internal/llm"
)

// TestClientSendsTheKeyInItsHeader sends the key of the api_key mode in the
// header the configuration names, such as the api-key of an Azure OpenAI
// gateway, without an Authorization header, and keeps it out of the error.
func TestClientSendsTheKeyInItsHeader(t *testing.T) {
	const key = "gateway-secret"
	var got http.Header
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %s", request.URL.Path)
		}
		got = request.Header.Clone()
		response.Header().Set("Content-Type", "application/json")
		response.WriteHeader(http.StatusUnauthorized)
		_, _ = response.Write([]byte(`{"error":{"type":"unauthorized","message":"bad key gateway-secret"}}`))
	}))
	t.Cleanup(server.Close)
	client, err := New(Config{BaseURL: server.URL + "/v1", Model: "test-model", AuthMode: AuthModeAPIKey, AuthHeader: "Ocp-Apim-Subscription-Key",
		TokenSource: func() (string, error) { return key, nil }, Timeout: time.Minute, MaxOutputTokens: 512}, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	err = client.Stream(t.Context(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Text: "hello"}}}, func(llm.Event) error { return nil })
	if err == nil || strings.Contains(err.Error(), key) {
		t.Fatalf("error = %v", err)
	}
	if got.Get("Ocp-Apim-Subscription-Key") != key || got.Get("Authorization") != "" || got.Get("x-api-key") != "" || got.Get("api-key") != "" {
		t.Fatalf("headers = %v", got)
	}
}

func TestClientRequiresTheHeaderOfTheKey(t *testing.T) {
	if _, err := New(Config{BaseURL: "https://gateway.example.test/v1", Model: "test-model", AuthMode: AuthModeAPIKey,
		TokenSource: func() (string, error) { return "key", nil }, Timeout: time.Minute, MaxOutputTokens: 512}, nil); err == nil || !strings.Contains(err.Error(), "header") {
		t.Fatalf("error = %v", err)
	}
}
