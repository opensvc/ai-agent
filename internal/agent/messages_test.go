package agent

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/llm/messages"
)

func TestRunTurnWithMessagesToolLoopAndNeutralHistory(t *testing.T) {
	var requests atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
			return
		}
		if strings.Contains(string(body), "jwt-marker") || strings.Contains(string(body), "test-provider-key") || r.Header.Get("Authorization") != "" || r.Header.Get("x-api-key") != "test-provider-key" {
			t.Error("provider credentials and OpenSVC credentials were not isolated")
		}
		var request struct {
			System []struct {
				Text string `json:"text"`
			} `json:"system"`
			Messages []struct {
				Role    string `json:"role"`
				Content []struct {
					Type      string `json:"type"`
					ToolUseID string `json:"tool_use_id"`
					Content   string `json:"content"`
				} `json:"content"`
			} `json:"messages"`
			Tools []struct {
				Name string `json:"name"`
			} `json:"tools"`
		}
		if err := json.Unmarshal(body, &request); err != nil {
			t.Error(err)
			return
		}
		if len(request.System) != 1 || request.System[0].Text != testSystemPrompt+"\n"+changesInstruction+"\n"+interfaceContextInstruction+"\n"+noPendingConfirmation || len(request.Tools) != 1 || request.Tools[0].Name != "get_cluster_health" {
			t.Error("missing system instructions or MCP tools")
		}
		position := requests.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message_start\",\"message\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n")
		if position == 1 {
			if len(request.Messages) != 1 {
				t.Errorf("initial messages = %#v", request.Messages)
			}
			_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call-1","name":"get_cluster_health","input":{}}}

data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"node\":\"node1\"}"}}

data: {"type":"content_block_stop","index":0}

data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":8}}

data: {"type":"message_stop"}

`)
			return
		}
		if position != 2 || len(request.Messages) != 3 || request.Messages[1].Role != "assistant" || request.Messages[1].Content[0].Type != "tool_use" || request.Messages[2].Role != "user" || request.Messages[2].Content[0].Type != "tool_result" || request.Messages[2].Content[0].ToolUseID != "call-1" || !strings.Contains(request.Messages[2].Content[0].Content, `"status":"healthy"`) {
			t.Errorf("unexpected tool transcript: %#v", request.Messages)
		}
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"The cluster is healthy."}}

data: {"type":"content_block_stop","index":0}

data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":6}}

data: {"type":"message_stop"}

`)
	}))
	t.Cleanup(server.Close)
	model, err := messages.New(messages.Config{
		BaseURL: server.URL + "/v1", Model: "test-model", AuthMode: messages.AuthModeAPIKey,
		TokenSource: func() (string, error) { return "test-provider-key", nil },
		Timeout:     time.Minute, MaxOutputTokens: 512,
	}, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	session := &fakeSession{
		tools:   []*mcp.Tool{{Name: "get_cluster_health", InputSchema: objectSchema()}},
		results: map[string]*mcp.CallToolResult{"get_cluster_health": {StructuredContent: map[string]any{"status": "healthy"}}},
	}
	agent := newTestAgent(t, model, session, 4)
	result, err := agent.RunTurn(auth.WithBearerToken(t.Context(), "jwt-marker"), nil, "check health", func(Event) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 2 || len(session.calls) != 1 || session.calls[0].arguments["node"] != "node1" || !session.closed || !session.connectedWithJWT {
		t.Fatalf("requests=%d session=%#v", requests.Load(), session)
	}
	if len(result.Messages) != 4 || result.Messages[3].Text != "The cluster is healthy." {
		t.Fatalf("neutral history=%#v", result.Messages)
	}
}
