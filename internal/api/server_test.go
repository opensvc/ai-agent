package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/opensvc/ai-agent/internal/agent"
)

func TestHealth(t *testing.T) {
	request := httptest.NewRequest(http.MethodGet, "/health", nil)
	response := httptest.NewRecorder()

	newTestHandler(t, nil).ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("got status %d, want %d", response.Code, http.StatusOK)
	}
	if got := response.Header().Get("Content-Type"); got != "application/json" {
		t.Errorf("got Content-Type %q, want application/json", got)
	}
	var health HealthResponse
	if err := json.NewDecoder(response.Body).Decode(&health); err != nil {
		t.Fatalf("decode health response: %v", err)
	}
	if health.Status != "ok" {
		t.Errorf("got health status %q, want ok", health.Status)
	}
}

func TestHealthRejectsOtherMethods(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/health", nil)
	response := httptest.NewRecorder()

	newTestHandler(t, nil).ServeHTTP(response, request)

	if response.Code != http.StatusMethodNotAllowed {
		t.Fatalf("got status %d, want %d", response.Code, http.StatusMethodNotAllowed)
	}
}

func TestUnknownRoute(t *testing.T) {
	request := httptest.NewRequest(http.MethodGet, "/unknown", nil)
	response := httptest.NewRecorder()

	newTestHandler(t, nil).ServeHTTP(response, request)

	if response.Code != http.StatusNotFound {
		t.Fatalf("got status %d, want %d", response.Code, http.StatusNotFound)
	}
}

func TestRoutesRequireAConfiguredBackend(t *testing.T) {
	handler := newTestHandler(t, turnFunc(func(context.Context, string, agent.EmitFunc) error {
		t.Fatal("a route outside a backend reached the agent")
		return nil
	}))
	for _, path := range []string{"/v1/ask", "/v1/conversations", "/v1/oc3/ask", "/v1/oc3/conversations"} {
		request := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"prompt":"hello"}`))
		request.Header.Set("Authorization", "Bearer token")
		request.Header.Set(testBackendHeader, "cluster-id")
		request.Header.Set("Content-Type", "application/json")
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusNotFound {
			t.Fatalf("%s status = %d, want %d", path, response.Code, http.StatusNotFound)
		}
	}
}

func TestRouteSelectsItsBackend(t *testing.T) {
	var called []string
	backend := func(name string) Backend {
		return Backend{
			Name: name,
			Conversations: turnService{run: turnFunc(func(_ context.Context, _ string, emit agent.EmitFunc) error {
				called = append(called, name)
				return emit(agent.Event{Type: agent.EventCompleted, Iteration: 1})
			})},
			Authenticator: allowTestTokenVerifier(),
		}
	}
	handler, err := NewHandler([]Backend{backend("om3"), backend("oc3")}, HandlerConfig{Models: testModels(), MaxConcurrentAsks: 4, AuditLogger: discardAuditLogger()})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"oc3", "om3"} {
		request := httptest.NewRequest(http.MethodPost, "/v1/"+name+"/conversations/"+testConversationID+"/runs", strings.NewReader(runBody("hello")))
		request.Header.Set("Authorization", "Bearer token")
		request.Header.Set(testBackendHeader, "cluster-id")
		request.Header.Set("Content-Type", "application/json")
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusOK {
			t.Fatalf("%s status = %d", name, response.Code)
		}
	}
	if strings.Join(called, ",") != "oc3,om3" {
		t.Fatalf("backends called = %v", called)
	}
}
