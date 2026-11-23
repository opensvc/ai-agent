//go:build integration

package api_test

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
	"testing"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/api"
	"github.com/opensvc/ai-agent/internal/backend/om3"
	"github.com/opensvc/ai-agent/internal/config"
	"github.com/opensvc/ai-agent/internal/conversation"
	conversationsqlite "github.com/opensvc/ai-agent/internal/conversation/sqlite"
	"github.com/opensvc/ai-agent/internal/llm"
	"github.com/opensvc/ai-agent/internal/llmfactory"
)

func TestLiveConversationTurnStreamsClusterHealth(t *testing.T) {
	mcpSocket := os.Getenv("OPENSVC_AI_TEST_MCP_SOCKET")
	mcpJWT := os.Getenv("OPENSVC_AI_TEST_MCP_JWT")
	clusterID := os.Getenv("OPENSVC_AI_TEST_CLUSTER_ID")
	if mcpSocket == "" || mcpJWT == "" || clusterID == "" || os.Getenv("OPENSVC_AI_LLM_PROTOCOL") == "" {
		t.Skip("live MCP and LLM configuration is unavailable")
	}
	llmConfig, err := config.LoadLLM()
	if err != nil {
		t.Fatalf("load live LLM configuration: %v", err)
	}
	agentConfig, err := config.LoadAgent()
	if err != nil {
		t.Fatalf("load live agent configuration: %v", err)
	}
	model, err := llmfactory.New(llmConfig, nil)
	if err != nil {
		t.Fatalf("create live LLM client: %v", err)
	}
	mcpClient, err := om3.NewMCPClient(mcpSocket)
	if err != nil {
		t.Fatalf("create live MCP client: %v", err)
	}
	orchestrator, err := agent.New(llm.SingleModel(model, llmConfig.Model), func(ctx context.Context) (agent.MCPSession, error) {
		return mcpClient.Connect(ctx)
	}, agent.Config{SystemPrompt: om3.SystemPrompt, MaxIterations: agentConfig.MaxIterations, Timeout: agentConfig.Timeout})
	if err != nil {
		t.Fatalf("create live agent: %v", err)
	}
	store, err := conversationsqlite.Open(t.Context(), conversationsqlite.Config{Path: filepath.Join(t.TempDir(), "conversations.db")})
	if err != nil {
		t.Fatalf("open conversation store: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	service, err := conversation.NewService(store, orchestrator, conversation.ServiceConfig{Backend: "om3"})
	if err != nil {
		t.Fatalf("create conversation service: %v", err)
	}
	handler, err := api.NewHandler([]api.Backend{{Name: "om3", Conversations: service, Authenticator: om3.NewAuthenticator(mcpClient)}}, api.HandlerConfig{
		Models:            llm.SingleModel(model, llmConfig.Model),
		MaxConcurrentAsks: 4,
		AuditLogger:       slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatalf("create live API: %v", err)
	}
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	send := func(path string, body string) *http.Response {
		t.Helper()
		request, err := http.NewRequestWithContext(t.Context(), http.MethodPost, server.URL+path, strings.NewReader(body))
		if err != nil {
			t.Fatalf("create live request: %v", err)
		}
		request.Header.Set("Authorization", "Bearer "+mcpJWT)
		request.Header.Set(om3.ClusterIDHeader, clusterID)
		if body != "" {
			request.Header.Set("Content-Type", "application/json")
		}
		response, err := server.Client().Do(request)
		if err != nil {
			t.Fatalf("send live request: %v", err)
		}
		return response
	}

	created := send("/v1/om3/conversations", "")
	defer created.Body.Close()
	if created.StatusCode != http.StatusCreated {
		t.Fatalf("live conversation creation returned HTTP %d", created.StatusCode)
	}
	var envelope api.ConversationEnvelope
	if err := json.NewDecoder(created.Body).Decode(&envelope); err != nil {
		t.Fatalf("decode live conversation: %v", err)
	}

	id := envelope.Conversation.ID
	response := send("/v1/om3/conversations/"+id+"/runs", `{"threadId":"`+id+`","runId":"live-run","messages":[{"id":"live-prompt","role":"user","content":"Use get_cluster_health to assess the current OpenSVC cluster health, then answer concisely."}]}`)
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("live run returned HTTP %d", response.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, 4<<20))
	if err != nil {
		t.Fatalf("read live run stream: %v", err)
	}
	body := string(data)
	for _, expected := range []string{`"type":"TOOL_CALL_START"`, `"toolCallName":"get_cluster_health"`, `"type":"RUN_FINISHED"`} {
		if !strings.Contains(body, expected) {
			t.Fatalf("live run stream does not contain %s: %s", expected, body)
		}
	}
	if strings.Contains(body, mcpJWT) {
		t.Fatal("live run stream exposes delegated JWT")
	}
}
