package chatcompletions

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/opensvc/ai-agent/internal/llm"
)

// NoToolCall keeps the tools declared, for the calls the history holds, and
// asks for an answer without one.
func TestNoToolCallKeepsTheToolsAndForbidsACall(t *testing.T) {
	for _, noToolCall := range []bool{false, true} {
		request := llm.Request{
			Messages:   []llm.Message{{Role: llm.RoleSystem, Text: "system"}, {Role: llm.RoleUser, Text: "health"}},
			Tools:      []llm.Tool{{Name: "get_cluster_health", InputSchema: json.RawMessage(`{"type":"object"}`)}},
			NoToolCall: noToolCall,
		}
		wire, err := newCreateRequest("model", 100, false, request)
		if err != nil {
			t.Fatal(err)
		}
		data, _ := json.Marshal(wire)
		body := string(data)
		if !strings.Contains(body, "get_cluster_health") || strings.Contains(body, "none") != noToolCall {
			t.Fatalf("no tool call %v: request = %s", noToolCall, body)
		}
	}
}
