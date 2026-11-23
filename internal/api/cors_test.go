package api

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/conversation"
	"github.com/opensvc/ai-agent/internal/llm"
)

func TestCORSPreflightRunsBeforeAuthentication(t *testing.T) {
	for _, origins := range [][]string{{"https://a.example", "https://b.example"}, {"*"}} {
		checks := 0
		verifier := tokenVerifierFunc(func(context.Context, string) (auth.Identity, error) {
			checks++
			return auth.Identity{}, auth.ErrInvalidToken
		})
		handler, err := NewHandler([]Backend{{Name: "om3", Conversations: turnService{run: turnFunc(func(context.Context, string, agent.EmitFunc) error {
			t.Fatal("preflight reached model")
			return nil
		})}, Authenticator: verifier}}, HandlerConfig{Models: testModels(), MaxConcurrentAsks: 4, AuditLogger: discardAuditLogger(), CORSAllowedOrigins: origins})
		if err != nil {
			t.Fatal(err)
		}
		for _, origin := range []string{"https://a.example", "https://b.example", "https://unknown.example"} {
			for _, method := range []string{"GET", "POST", "PUT", "PATCH", "DELETE"} {
				r := httptest.NewRequest("OPTIONS", "/v1/om3/conversations", nil)
				r.Header.Set("Origin", origin)
				r.Header.Set("Access-Control-Request-Method", method)
				r.Header.Set("Access-Control-Request-Headers", "authorization, Content-Type, x-test-backend")
				w := httptest.NewRecorder()
				handler.ServeHTTP(w, r)
				wantOrigin, wantStatus := origin, 204
				if origins[0] == "*" {
					wantOrigin = "*"
				} else if origin == "https://unknown.example" {
					wantOrigin, wantStatus = "", 403
				}
				if w.Code != wantStatus || w.Header().Get("Access-Control-Allow-Origin") != wantOrigin || checks != 0 {
					t.Fatalf("origins=%v origin=%s method=%s status=%d checks=%d", origins, origin, method, w.Code, checks)
				}
				if w.Header().Get("Access-Control-Allow-Credentials") != "" {
					t.Fatal("CORS enabled automatic browser credentials")
				}
				if wantStatus == 204 && (w.Header().Get("Access-Control-Allow-Methods") != corsMethods || !strings.Contains(w.Header().Get("Access-Control-Allow-Headers"), testBackendHeader) || w.Header().Get("Access-Control-Max-Age") != "600") {
					t.Fatal("preflight lost allowed methods/headers/cache lifetime")
				}
			}
		}
	}
}

func TestCORSRejectsInvalidPreflight(t *testing.T) {
	called := false
	handler, err := withCORS(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { called = true }), []string{"https://a.example"}, []string{testBackendHeader})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ methods, headers []string }{
		{[]string{"TRACE"}, nil}, {[]string{"POST", "POST"}, nil}, {[]string{""}, nil}, {[]string{"post"}, nil},
		{[]string{"POST"}, []string{"X-Unexpected"}}, {[]string{"POST"}, []string{"Authorization,"}}, {[]string{"POST"}, []string{""}},
	} {
		r := httptest.NewRequest("OPTIONS", "/v1/om3/conversations", nil)
		r.Header.Set("Origin", "https://a.example")
		for _, value := range tc.methods {
			r.Header.Add("Access-Control-Request-Method", value)
		}
		for _, value := range tc.headers {
			r.Header.Add("Access-Control-Request-Headers", value)
		}
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, r)
		if w.Code != 403 || called || w.Header().Get("Access-Control-Allow-Methods") != "" || w.Header().Get("Access-Control-Allow-Origin") != "https://a.example" {
			t.Fatal("invalid preflight was accepted or lost CORS on its error")
		}
	}
}

func TestCORSActualRequestsKeepAuthenticationAndNonBrowserClients(t *testing.T) {
	for _, origins := range [][]string{nil, {"https://a.example"}, {"*"}} {
		checks := 0
		verifier := tokenVerifierFunc(func(context.Context, string) (auth.Identity, error) {
			checks++
			return auth.Identity{}, auth.ErrInvalidToken
		})
		handler, err := NewHandler([]Backend{{Name: "om3", Conversations: turnService{run: turnFunc(func(context.Context, string, agent.EmitFunc) error { return nil })}, Authenticator: verifier}}, HandlerConfig{Models: testModels(), MaxConcurrentAsks: 4, AuditLogger: discardAuditLogger(), CORSAllowedOrigins: origins})
		if err != nil {
			t.Fatal(err)
		}
		for _, origin := range []string{"", "https://a.example", "https://unknown.example"} {
			r := httptest.NewRequest("GET", "/v1/om3/conversations", nil)
			r.Header.Set("Authorization", "Bearer invalid")
			r.Header.Set(testBackendHeader, "cluster-id")
			if origin != "" {
				r.Header.Set("Origin", origin)
			}
			w := httptest.NewRecorder()
			before := checks
			handler.ServeHTTP(w, r)
			if len(origins) != 0 && origins[0] != "*" && origin == "https://unknown.example" {
				if w.Code != 403 || checks != before || w.Header().Get("Access-Control-Allow-Origin") != "" {
					t.Fatal("forbidden origin reached authentication")
				}
				continue
			}
			if w.Code != 401 || checks != before+1 {
				t.Fatal("CORS bypassed authentication or blocked a non-browser client")
			}
			want := ""
			if len(origins) != 0 && origin != "" {
				want = origin
				if origins[0] == "*" {
					want = "*"
				}
			}
			if w.Header().Get("Access-Control-Allow-Origin") != want {
				t.Fatal("authentication error lost its CORS headers")
			}
		}
		// CORS must never make a missing token sufficient for access.
		r := httptest.NewRequest("GET", "/v1/om3/conversations", nil)
		r.Header.Set("Origin", "https://a.example")
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, r)
		if w.Code != 401 {
			t.Fatal("missing Bearer token accepted")
		}
	}
}

func TestCORSPreservesStreamingConversationTurns(t *testing.T) {
	for _, origins := range [][]string{{"https://a.example"}, {"*"}} {
		execution := &testTurnExecution{run: func(_ context.Context, emit agent.EmitFunc) error {
			if err := emit(agent.Event{Type: agent.EventTextDelta, TextDelta: "healthy", Iteration: 1}); err != nil {
				return err
			}
			return emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, Iteration: 1})
		}}
		service := conversationServiceFuncs{prepare: func(context.Context, auth.Identity, string, string) (conversation.TurnExecution, error) {
			return execution, nil
		}}
		handler, err := NewHandler([]Backend{{Name: "om3", Conversations: service, Authenticator: allowTestTokenVerifier()}}, HandlerConfig{Models: testModels(), MaxConcurrentAsks: 4, AuditLogger: discardAuditLogger(), CORSAllowedOrigins: origins})
		if err != nil {
			t.Fatal(err)
		}
		r := authenticatedRequest("POST", "/v1/om3/conversations/id/runs", runInput("id", "hello"))
		r.Header.Set("Origin", "https://a.example")
		r.Header.Set("Content-Type", "application/json")
		w := &writeDeadlineRecorder{ResponseRecorder: httptest.NewRecorder()}
		handler.ServeHTTP(w, r)
		if w.Code != 200 || !w.Flushed || len(w.deadlines) == 0 || !strings.Contains(w.Body.String(), `"type":"RUN_FINISHED"`) || !strings.HasPrefix(w.Header().Get("Content-Type"), "text/event-stream") {
			t.Fatalf("CORS broke streamed run: status=%d body=%s", w.Code, w.Body.String())
		}
		if w.Header().Get("Access-Control-Allow-Origin") != origins[0] {
			t.Fatal("SSE response lost CORS headers")
		}
	}
}

func TestCORSPolicySnapshotAndAmbiguousOrigins(t *testing.T) {
	origins := []string{"https://a.example"}
	handler, err := withCORS(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(204) }), origins, nil)
	if err != nil {
		t.Fatal(err)
	}
	origins[0] = "https://unknown.example"
	for _, values := range [][]string{{"https://a.example"}, {"https://unknown.example"}, {""}, {"https://a.example", "https://a.example"}, {"https://a.example,https://unknown.example"}, {"null"}} {
		r := httptest.NewRequest("GET", "/health", nil)
		for _, value := range values {
			r.Header.Add("Origin", value)
		}
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, r)
		want := 403
		if len(values) == 1 && values[0] == "https://a.example" {
			want = 204
		}
		if w.Code != want || !strings.Contains(strings.Join(w.Header().Values("Vary"), ","), "Origin") {
			t.Fatal("origin was ambiguous, policy mutable, or Vary missing")
		}
	}
	for _, values := range [][]string{{""}, {"*", "https://a.example"}} {
		if _, err := withCORS(http.NotFoundHandler(), values, nil); err == nil {
			t.Fatal("ambiguous CORS configuration accepted")
		}
	}
}
