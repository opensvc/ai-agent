package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/auth"
)

type toolListerFunc func(context.Context) ([]agent.ToolInfo, error)

func (f toolListerFunc) Tools(ctx context.Context) ([]agent.ToolInfo, error) { return f(ctx) }

func newToolsTestHandler(t *testing.T, tools ToolLister) http.Handler {
	t.Helper()
	handler, err := NewHandler([]Backend{{Name: "om3", Conversations: turnService{}, Authenticator: allowTestTokenVerifier(), Tools: tools}}, HandlerConfig{Models: testModels(), MaxConcurrentAsks: 4, AuditLogger: discardAuditLogger()})
	if err != nil {
		t.Fatal(err)
	}
	return handler
}

func TestToolsServeWhatTheUserMayAskTheAssistant(t *testing.T) {
	var token string
	handler := newToolsTestHandler(t, toolListerFunc(func(ctx context.Context) ([]agent.ToolInfo, error) {
		token, _ = auth.BearerTokenFromContext(ctx)
		return []agent.ToolInfo{
			{Name: "get_object_status", Title: "Get object status", Summary: "Read the status of one object.", ReadOnly: true},
			{Name: "stop_instance", Title: "Stop instance", Summary: "Stop the instance.", Description: "Stop the instance.\nOn one node.", Arguments: map[string]string{"node": "the exact node"}, Confirmable: true},
		}, nil
	}))
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, authenticatedRequest(http.MethodGet, "/v1/om3/tools", ""))
	if recorder.Code != http.StatusOK || recorder.Header().Get("Cache-Control") != "no-store" || token != "token" {
		t.Fatalf("status %d, cache %q, token %q: %s", recorder.Code, recorder.Header().Get("Cache-Control"), token, recorder.Body)
	}
	var body map[string][]map[string]any
	if err := json.Unmarshal(recorder.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	stop := body["tools"][1]
	if len(body["tools"]) != 2 || stop["name"] != "stop_instance" || stop["title"] != "Stop instance" || stop["summary"] != "Stop the instance." ||
		stop["read_only"] != false || stop["needs_confirmation"] != true || body["tools"][0]["read_only"] != true ||
		stop["description"] != "Stop the instance.\nOn one node." || stop["arguments"].(map[string]any)["node"] != "the exact node" {
		t.Fatalf("tools = %s", recorder.Body)
	}
}

func TestToolsAnswer502WhenTheMCPDoesNot(t *testing.T) {
	handler := newToolsTestHandler(t, toolListerFunc(func(context.Context) ([]agent.ToolInfo, error) {
		return nil, errors.New("connection refused")
	}))
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, authenticatedRequest(http.MethodGet, "/v1/om3/tools", ""))
	if recorder.Code != http.StatusBadGateway || !json.Valid(recorder.Body.Bytes()) {
		t.Fatalf("status %d: %s", recorder.Code, recorder.Body)
	}
	var body struct{ Error struct{ Code string } }
	if err := json.Unmarshal(recorder.Body.Bytes(), &body); err != nil || body.Error.Code != "tools_unavailable" {
		t.Fatalf("body = %s", recorder.Body)
	}
	// A backend listing no tools, and a request without a token.
	recorder = httptest.NewRecorder()
	newToolsTestHandler(t, nil).ServeHTTP(recorder, authenticatedRequest(http.MethodGet, "/v1/om3/tools", ""))
	if recorder.Code != http.StatusNotFound {
		t.Fatalf("no lister: status %d", recorder.Code)
	}
	recorder = httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/v1/om3/tools", nil))
	if recorder.Code != http.StatusUnauthorized {
		t.Fatalf("no token: status %d", recorder.Code)
	}
}
