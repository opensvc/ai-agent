package api

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/backend/oc3"
	"github.com/opensvc/ai-agent/internal/backend/om3"
	"github.com/opensvc/ai-agent/internal/conversation"
	conversationsqlite "github.com/opensvc/ai-agent/internal/conversation/sqlite"
	"github.com/opensvc/ai-agent/internal/llm"
)

// TestBackendsKeepTheirConversationsApart serves om3 and oc3 together, each
// with its own MCP. The same identity holds separate conversations on each,
// and each backend asks its own MCP who the token is.
func TestBackendsKeepTheirConversationsApart(t *testing.T) {
	const issuer, subject = "https://idp.example.test/", "opaque-alice"
	expires := time.Now().Add(time.Hour).UTC()
	var om3Checks, oc3Checks atomic.Int32
	om3Socket := serveUnixMCP(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		om3Checks.Add(1)
		if r.Header.Get("Authorization") != "Bearer om3-token" || r.Header.Get(om3.ClusterIDHeader) != "cluster-a" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"cluster_id": "cluster-a", "issuer": issuer, "subject": subject, "expires_at": expires})
	}))
	oc3Socket := serveUnixMCP(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		oc3Checks.Add(1)
		if r.Header.Get("Authorization") != "Bearer oc3-token" || r.Header.Get(om3.ClusterIDHeader) != "" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"issuer": issuer, "subject": subject, "expires_at": expires})
	}))
	om3Client, err := om3.NewMCPClient(om3Socket)
	if err != nil {
		t.Fatal(err)
	}
	oc3Client, err := oc3.NewMCPClient(oc3Socket)
	if err != nil {
		t.Fatal(err)
	}
	directory := t.TempDir()
	if err := os.Chmod(directory, 0700); err != nil {
		t.Fatal(err)
	}
	store, err := conversationsqlite.Open(t.Context(), conversationsqlite.Config{Path: filepath.Join(directory, "conversations.db")})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	runner := apiTurnRunnerFunc(func(_ context.Context, _ []llm.Message, prompt string, _ agent.EmitFunc) (agent.TurnResult, error) {
		return agent.TurnResult{Messages: []llm.Message{{Role: llm.RoleUser, Text: prompt}, {Role: llm.RoleAssistant, Text: "answer"}}, FinishReason: llm.FinishReasonCompleted}, nil
	})
	var backends []Backend
	for name, authenticator := range map[string]Authenticator{"om3": om3.NewAuthenticator(om3Client), "oc3": oc3.NewAuthenticator(oc3Client)} {
		service, err := conversation.NewService(store, runner, conversation.ServiceConfig{Backend: name})
		if err != nil {
			t.Fatal(err)
		}
		backends = append(backends, Backend{Name: name, Conversations: service, Authenticator: authenticator})
	}
	handler, err := NewHandler(backends, HandlerConfig{Models: testModels(), MaxConcurrentAsks: 4, AuditLogger: slog.New(slog.NewTextHandler(io.Discard, nil))})
	if err != nil {
		t.Fatal(err)
	}
	call := func(method, path, token string, header http.Header) *httptest.ResponseRecorder {
		request := httptest.NewRequest(method, path, strings.NewReader(""))
		request.Header = header.Clone()
		request.Header.Set("Authorization", "Bearer "+token)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		return response
	}
	om3Header := http.Header{}
	om3Header.Set(om3.ClusterIDHeader, "cluster-a")

	created := call("POST", "/v1/oc3/conversations", "oc3-token", http.Header{})
	if created.Code != 201 || oc3Checks.Load() != 1 || om3Checks.Load() != 0 {
		t.Fatalf("oc3 create: status=%d oc3 checks=%d om3 checks=%d", created.Code, oc3Checks.Load(), om3Checks.Load())
	}
	var item ConversationEnvelope
	if err := json.Unmarshal(created.Body.Bytes(), &item); err != nil {
		t.Fatal(err)
	}
	if response := call("GET", "/v1/oc3/conversations/"+item.Conversation.ID, "oc3-token", http.Header{}); response.Code != 200 {
		t.Fatalf("oc3 read: %d", response.Code)
	}
	// The same identity on om3 neither lists nor reads the oc3 conversation.
	if response := call("GET", "/v1/om3/conversations", "om3-token", om3Header); response.Code != 200 || strings.Contains(response.Body.String(), item.Conversation.ID) {
		t.Fatalf("oc3 conversation listed on om3: %d %s", response.Code, response.Body.String())
	}
	if response := call("GET", "/v1/om3/conversations/"+item.Conversation.ID, "om3-token", om3Header); response.Code != 404 {
		t.Fatalf("oc3 conversation read on om3: %d", response.Code)
	}
	// Each backend verifies the token with its own MCP only.
	if response := call("GET", "/v1/om3/conversations", "oc3-token", om3Header); response.Code != 401 {
		t.Fatalf("oc3 token accepted on om3: %d", response.Code)
	}
	if response := call("GET", "/v1/oc3/conversations", "om3-token", http.Header{}); response.Code != 401 {
		t.Fatalf("om3 token accepted on oc3: %d", response.Code)
	}
	// oc3 needs no cluster header, and forwards none it receives.
	if response := call("GET", "/v1/oc3/conversations", "oc3-token", om3Header); response.Code != 200 {
		t.Fatalf("oc3 refused a request carrying an om3 header: %d", response.Code)
	}
	if response := call("GET", "/v1/om3/conversations", "om3-token", http.Header{}); response.Code != 401 {
		t.Fatalf("om3 accepted a request without its cluster header: %d", response.Code)
	}
}
