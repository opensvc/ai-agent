package api

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/backend/om3"
	"github.com/opensvc/ai-agent/internal/conversation"
	conversationsqlite "github.com/opensvc/ai-agent/internal/conversation/sqlite"
	"github.com/opensvc/ai-agent/internal/llm"
)

func TestRemoteIdentityProtectsLocalConversationsAndModelCalls(t *testing.T) {
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	targets := make(map[string]string)
	signWithExpiry := func(cluster, user string, key *rsa.PrivateKey, expiresAt time.Time) string {
		claims := jwt.MapClaims{"iss": "https://idp.example.test/", "sub": "opaque-" + user, "preferred_username": user, "aud": "client-" + cluster, "exp": expiresAt.Unix()}
		unsigned := jwt.NewWithClaims(jwt.SigningMethodRS256, claims)
		unsigned.Header["kid"] = "key"
		raw, err := unsigned.SignedString(key)
		if err != nil {
			t.Fatal(err)
		}
		targets[raw] = cluster
		return raw
	}
	sign := func(cluster, user string, key *rsa.PrivateKey) string {
		return signWithExpiry(cluster, user, key, time.Now().Add(time.Hour))
	}
	alice := sign("cluster-a", "alice", key)
	bob := sign("cluster-a", "bob", key)
	// One OpenID provider signs for every cluster and distinguishes them by
	// audience.
	clusterKeys := map[string]*rsa.PrivateKey{"cluster-a": key, "cluster-b": key}
	otherCluster := sign("cluster-b", "alice", clusterKeys["cluster-b"])
	var unavailable atomic.Bool
	var checks atomic.Int32
	socket := serveUnixMCP(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		checks.Add(1)
		if unavailable.Load() {
			w.WriteHeader(502)
			return
		}
		if r.URL.Path != "/mcp/auth/whoami" {
			t.Errorf("unexpected route %s", r.URL.Path)
			w.WriteHeader(404)
			return
		}
		raw := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		claims := jwt.MapClaims{}
		target := r.Header.Get(om3.ClusterIDHeader)
		options := []jwt.ParserOption{jwt.WithValidMethods([]string{"RS256"}), jwt.WithExpirationRequired()}
		if (target != "cluster-a" && target != "cluster-b") || r.Header.Get(om3.NodeHeader) != "node-b" {
			w.WriteHeader(401)
			return
		}
		options = append(options, jwt.WithIssuer("https://idp.example.test/"), jwt.WithAudience("client-"+target))
		// The header selects the daemon that verifies the token: a token of
		// another cluster fails there, so the requested cluster ID is returned.
		verifyKey, ok := clusterKeys[target]
		if !ok {
			w.WriteHeader(401)
			return
		}
		if _, err := jwt.ParseWithClaims(raw, claims, func(*jwt.Token) (any, error) { return &verifyKey.PublicKey, nil }, options...); err != nil {
			w.WriteHeader(401)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(auth.Identity{ClusterID: target, Issuer: claims["iss"].(string), Subject: claims["sub"].(string), ExpiresAt: time.Unix(int64(claims["exp"].(float64)), 0)})
	}))
	mcpClient, err := om3.NewMCPClient(socket)
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
	var modelCalls atomic.Int32
	runner := apiTurnRunnerFunc(func(_ context.Context, _ []llm.Message, prompt string, emit agent.EmitFunc) (agent.TurnResult, error) {
		modelCalls.Add(1)
		return agent.TurnResult{Messages: []llm.Message{{Role: llm.RoleUser, Text: prompt}, {Role: llm.RoleAssistant, Text: "answer"}}, FinishReason: llm.FinishReasonCompleted}, nil
	})
	service, err := conversation.NewService(store, runner, conversation.ServiceConfig{Backend: "om3"})
	if err != nil {
		t.Fatal(err)
	}
	handler, err := NewHandler([]Backend{{Name: "om3", Conversations: service, Authenticator: om3.NewAuthenticator(mcpClient)}}, HandlerConfig{Models: testModels(), MaxConcurrentAsks: 4, AuditLogger: slog.New(slog.NewTextHandler(io.Discard, nil))})
	if err != nil {
		t.Fatal(err)
	}
	call := func(method, path, token, body string) *httptest.ResponseRecorder {
		request := requestWithToken(method, path, token, body)
		request.Header.Set(om3.ClusterIDHeader, targets[token])
		request.Header.Set(om3.NodeHeader, "node-b")
		request.Header.Set("Content-Type", "application/json")
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		return response
	}
	created := call("POST", "/v1/om3/conversations", alice, "")
	if created.Code != 201 {
		t.Fatalf("create: %d", created.Code)
	}
	var item ConversationEnvelope
	if err := json.Unmarshal(created.Body.Bytes(), &item); err != nil {
		t.Fatal(err)
	}
	path := "/v1/om3/conversations/" + item.Conversation.ID
	for _, token := range []string{bob, otherCluster} {
		for _, method := range []string{"GET", "PATCH", "DELETE"} {
			body := ""
			if method == "PATCH" {
				body = `{"title":"foreign"}`
			}
			if response := call(method, path, token, body); response.Code != 404 {
				t.Fatalf("foreign access method=%s status=%d", method, response.Code)
			}
		}
		if response := call("POST", path+"/runs", token, runInput(item.Conversation.ID, "read history")); response.Code != 404 {
			t.Fatalf("foreign run: %d", response.Code)
		}
		response := call("GET", "/v1/om3/conversations", token, "")
		if response.Code != 200 || strings.Contains(response.Body.String(), item.Conversation.ID) {
			t.Fatal("foreign conversation leaked")
		}
	}
	foreignKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	forged := sign("cluster-a", "alice", foreignKey)
	expired := signWithExpiry("cluster-a", "alice", key, time.Now().Add(-time.Second))
	targets["malformed"] = "cluster-a"
	for _, token := range []string{forged, expired, "malformed"} {
		// The first request delegates the validation to the MCP; the refusal
		// is then remembered for five seconds, without asking the MCP again.
		before := checks.Load()
		for _, tc := range []struct{ method, path, body string }{
			{"GET", "/v1/om3/conversations", ""}, {"GET", path, ""}, {"GET", path + "/messages", ""}, {"DELETE", path, ""},
			{"PATCH", path, `{"title":"refused"}`}, {"POST", path + "/runs", runInput(item.Conversation.ID, "refused")},
			{"POST", "/v1/om3/conversations/conversation-1/runs", runBody("refused")}, {"POST", "/v1/om3/conversations", ""},
		} {
			response := call(tc.method, tc.path, token, tc.body)
			if response.Code != 401 || strings.Contains(response.Body.String(), token) {
				t.Fatalf("refused credential admitted: status=%d", response.Code)
			}
		}
		if checks.Load() != before+1 {
			t.Fatalf("refused credential checked %d times by the MCP, want once", checks.Load()-before)
		}
	}
	if modelCalls.Load() != 0 {
		t.Fatal("model invoked before authentication/ownership check")
	}
	// A token verified within the minute is not checked again; a token the
	// agent has not verified yet is refused while the MCP is unavailable.
	unverified := signWithExpiry("cluster-a", "alice", key, time.Now().Add(2*time.Hour))
	unavailable.Store(true)
	if response := call("DELETE", path, unverified, ""); response.Code != 503 {
		t.Fatalf("unavailable verifier bypassed: %d", response.Code)
	}
	if response := call("DELETE", path, unverified, ""); response.Code != 503 {
		t.Fatalf("unavailable verifier cached: %d", response.Code)
	}
	unavailable.Store(false)
	if response := call("GET", path, alice, ""); response.Code != 200 {
		t.Fatal("refused deletion changed existing conversation")
	}
	if response := call("POST", path+"/runs", alice, runInput(item.Conversation.ID, "hello")); response.Code != 200 || modelCalls.Load() != 1 {
		t.Fatalf("authorized run rejected: %d", response.Code)
	}
	// Each token was checked by the MCP once: alice, bob, the other cluster,
	// the three refused tokens, and the unverified token twice while the MCP
	// was unavailable. The cache reuses identities, it never establishes one.
	if checks.Load() != 8 {
		t.Fatalf("identities checked %d times by the MCP, want 8", checks.Load())
	}
}

// serveUnixMCP serves a fake MCP on a Unix socket in a short directory, so the
// path stays within the sun_path limit.
func serveUnixMCP(t *testing.T, handler http.Handler) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "mcp")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	path := filepath.Join(dir, "mcp.sock")
	listener, err := net.Listen("unix", path)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewUnstartedServer(handler)
	server.Listener = listener
	server.Start()
	t.Cleanup(server.Close)
	return path
}
