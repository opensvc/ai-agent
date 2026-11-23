//go:build integration

package agent_test

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/backend/om3"
	"github.com/opensvc/ai-agent/internal/config"
	"github.com/opensvc/ai-agent/internal/llm"
	"github.com/opensvc/ai-agent/internal/llmfactory"
)

func TestLiveAgentUsesClusterHealthTool(t *testing.T) {
	mcpSocket := os.Getenv("OPENSVC_AI_TEST_MCP_SOCKET")
	mcpJWT := os.Getenv("OPENSVC_AI_TEST_MCP_JWT")
	if mcpSocket == "" || mcpJWT == "" || os.Getenv("OPENSVC_AI_LLM_PROTOCOL") == "" {
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
	orchestrator, err := agent.New(llm.SingleModel(model, "test-model"), func(ctx context.Context) (agent.MCPSession, error) {
		return mcpClient.Connect(ctx)
	}, agent.Config{SystemPrompt: om3.SystemPrompt, MaxIterations: agentConfig.MaxIterations, Timeout: agentConfig.Timeout})
	if err != nil {
		t.Fatalf("create live agent: %v", err)
	}

	ctx := auth.WithBearerToken(t.Context(), mcpJWT)
	var (
		text      strings.Builder
		toolNames []string
		completed bool
	)
	_, err = orchestrator.RunTurn(ctx, nil, "Use get_cluster_health to assess the current OpenSVC cluster health, then answer concisely.", func(event agent.Event) error {
		switch event.Type {
		case agent.EventTextDelta:
			text.WriteString(event.TextDelta)
		case agent.EventToolStarted:
			toolNames = append(toolNames, event.ToolName)
		case agent.EventCompleted:
			completed = true
		}
		return nil
	})
	if err != nil {
		t.Fatalf("run live agent: %v", err)
	}
	if len(toolNames) == 0 || toolNames[0] != "get_cluster_health" {
		t.Fatalf("unexpected live tool calls: %v", toolNames)
	}
	if !completed || strings.TrimSpace(text.String()) == "" {
		t.Fatalf("incomplete live agent answer, completed=%v text=%q", completed, text.String())
	}
}
