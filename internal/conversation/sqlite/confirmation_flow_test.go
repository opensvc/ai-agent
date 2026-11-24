package sqlite

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/conversation"
	"github.com/opensvc/ai-agent/internal/llm"
)

// flowSession is an MCP session offering one destructive tool.
type flowSession struct{ calls []string }

func (s *flowSession) ServerInfo() (string, string) { return "flow-mcp", "v2" }

func (s *flowSession) ListResources(context.Context) ([]*mcp.Resource, error) { return nil, nil }

func (s *flowSession) ReadResource(context.Context, string) (*mcp.ReadResourceResult, error) {
	return nil, nil
}

func (s *flowSession) ListTools(context.Context) ([]*mcp.Tool, error) {
	destructive := true
	return []*mcp.Tool{{
		Name: "stop_object", Title: "Stop object", InputSchema: map[string]any{"type": "object"},
		Annotations: &mcp.ToolAnnotations{DestructiveHint: &destructive},
	}}, nil
}

func (s *flowSession) CallTool(_ context.Context, name string, arguments map[string]any) (*mcp.CallToolResult, error) {
	s.calls = append(s.calls, name+" "+arguments["path"].(string))
	return &mcp.CallToolResult{StructuredContent: map[string]any{"orchestration_id": "o-1"}}, nil
}

func (s *flowSession) Close() error { return nil }

// flowModel asks to stop the object, then answers from the tool result.
type flowModel struct{ lastResult string }

func (m *flowModel) Stream(_ context.Context, request llm.Request, emit llm.EmitFunc) error {
	// The answer of a confirmation follows the result of its call: the
	// last tool message holds it.
	var last *llm.Message
	for i := range request.Messages {
		if request.Messages[i].Role == llm.RoleTool {
			last = &request.Messages[i]
		}
	}
	if last == nil {
		if err := emit(llm.Event{Type: llm.EventToolCall, ToolCall: &llm.ToolCall{ID: "call-1", Name: "stop_object", Arguments: json.RawMessage(`{"path":"prod/svc/web"}`)}}); err != nil {
			return err
		}
		return emit(llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonToolCalls})
	}
	m.lastResult = string(last.ToolResults[0].Content)
	if err := emit(llm.Event{Type: llm.EventTextDelta, TextDelta: "answered"}); err != nil {
		return err
	}
	return emit(llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonCompleted})
}

func TestConfirmationFlowSuspendsStoresAndResumesATurn(t *testing.T) {
	for _, tc := range []struct {
		decision   agent.Decision
		wantCalls  int
		wantResult string
	}{
		{agent.DecisionApprove, 1, "o-1"},
		{agent.DecisionReject, 0, "rejected"},
	} {
		t.Run(string(tc.decision), func(t *testing.T) {
			store, _ := openTestStore(t, Config{})
			session := &flowSession{}
			model := &flowModel{}
			orchestrator, err := agent.New(llm.SingleModel(model, "gpt-test"), func(context.Context) (agent.MCPSession, error) { return session, nil }, agent.Config{SystemPrompt: "You are a test assistant.", MaxIterations: 4, Timeout: time.Minute})
			if err != nil {
				t.Fatal(err)
			}
			service, err := conversation.NewService(store, orchestrator, conversation.ServiceConfig{Backend: "om3", Agent: conversation.NewAgentInfo("v1.0.0")})
			if err != nil {
				t.Fatal(err)
			}
			identity := auth.Identity{ClusterID: testOwner.ClusterID, Issuer: testOwner.Issuer, Subject: testOwner.Subject}
			item, err := service.Create(t.Context(), identity)
			if err != nil {
				t.Fatal(err)
			}
			turn, err := service.PrepareTurn(t.Context(), identity, item.ID, "stop web")
			if err != nil {
				t.Fatal(err)
			}
			var request *agent.Confirmation
			if err := turn.Run(t.Context(), func(event agent.Event) error {
				if event.Type == agent.EventConfirmationRequired {
					request = event.Confirmation
				}
				return nil
			}); err != nil {
				t.Fatal(err)
			}
			if request == nil || len(session.calls) != 0 {
				t.Fatalf("turn not suspended before the call: request=%+v calls=%v", request, session.calls)
			}
			if _, err := service.PrepareTurn(t.Context(), identity, item.ID, "another"); !errors.Is(err, conversation.ErrBusy) {
				t.Fatalf("a new message was accepted while a confirmation waits: %v", err)
			}
			pending, err := service.PendingConfirmation(t.Context(), identity, item.ID)
			if err != nil || pending == nil || pending.ConfirmationID != request.ID {
				t.Fatalf("pending confirmation not readable: %+v %v", pending, err)
			}

			resumed, err := service.PrepareConfirmation(t.Context(), identity, item.ID, request.TurnID, request.ID, tc.decision, nil)
			if err != nil {
				t.Fatal(err)
			}
			completed := false
			if err := resumed.Run(t.Context(), func(event agent.Event) error {
				completed = completed || event.Type == agent.EventCompleted
				return nil
			}); err != nil {
				t.Fatal(err)
			}
			if !completed || len(session.calls) != tc.wantCalls || !strings.Contains(model.lastResult, tc.wantResult) {
				t.Fatalf("completed=%v calls=%v result=%s", completed, session.calls, model.lastResult)
			}
			if tc.wantCalls == 1 && session.calls[0] != "stop_object prod/svc/web" {
				t.Fatalf("the approved call changed: %v", session.calls)
			}
			history, err := store.LoadHistory(t.Context(), testOwner, item.ID)
			if err != nil || len(history) != 4 || history[0].Text != "stop web" || history[3].Text != "answered" {
				t.Fatalf("the resumed turn was not stored whole: %#v %v", history, err)
			}
			// The suspended and the resumed runs used the same agent and MCP:
			// the turn records one run, with the MCP it announced.
			stored, _, err := store.TurnMessages(t.Context(), testOwner, item.ID, request.TurnID)
			if err != nil || len(stored.Runs) != 1 || stored.Runs[0].AgentVersion != "v1.0.0" || stored.Runs[0].Model != "gpt-test" || stored.Runs[0].ModelID != llm.DefaultModelID ||
				stored.Runs[0].MCPServer != "flow-mcp" || stored.Runs[0].MCPVersion != "v2" || len(stored.Runs[0].ToolsSHA256) != 64 ||
				len(stored.Runs[0].SystemPromptSHA256) != 64 || stored.Runs[0].ToolSearch || len(stored.Runs[0].ShownTools) == 0 ||
				stored.Runs[0].History == nil || stored.Runs[0].History.Policy != "v3" {
				t.Fatalf("turn runs = %+v, %v", stored.Runs, err)
			}
		})
	}
}
