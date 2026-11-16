package agent

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/opensvc/ai-agent/internal/llm"
)

var thought = &llm.Reasoning{Protocol: "test", Data: json.RawMessage(`{"thinking":"private"}`)}

// thinkingCallStep calls name after a reasoning the next requests of the
// turn give back.
func thinkingCallStep(name, callID string) llmStep {
	return func(_ llm.Request, emit llm.EmitFunc) error {
		return emitEvents(emit,
			llm.Event{Type: llm.EventToolCall, ToolCall: &llm.ToolCall{ID: callID, Name: name, Arguments: json.RawMessage(`{}`)}},
			llm.Event{Type: llm.EventReasoning, Reasoning: thought},
			llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonToolCalls},
		)
	}
}

// The reasoning of an answer reaches the next requests of the run, and
// neither the history nor the events of the client.
func TestReasoningLivesAsLongAsTheRun(t *testing.T) {
	session := &fakeSession{
		tools:   []*mcp.Tool{{Name: "health", InputSchema: objectSchema(), Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true}}},
		results: map[string]*mcp.CallToolResult{"health": {Content: []mcp.Content{&mcp.TextContent{Text: "ok"}}}},
	}
	var given *llm.Reasoning
	model := &scriptedLLM{t: t, steps: []llmStep{
		thinkingCallStep("health", "call-1"),
		func(request llm.Request, emit llm.EmitFunc) error {
			for _, message := range request.Messages {
				if message.Role == llm.RoleAssistant && len(message.ToolCalls) == 1 && message.ToolCalls[0].ID == "call-1" {
					given = message.Reasoning
				}
			}
			return textStep("healthy")(request, emit)
		},
	}}
	result, err := newTestAgent(t, model, session, 4).RunTurn(t.Context(), nil, "health", func(event Event) error {
		if strings.Contains(string(event.ToolResult)+event.TextDelta, "private") {
			t.Fatalf("reasoning shown: %#v", event)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if given == nil || string(given.Data) != string(thought.Data) {
		t.Fatalf("reasoning given back = %#v", given)
	}
	for _, message := range result.Messages {
		if message.Reasoning != nil {
			t.Fatalf("reasoning kept in the history: %#v", message)
		}
	}
}

// A turn waiting on the user keeps no reasoning: it resumes without it.
func TestSuspendedTurnKeepsNoReasoning(t *testing.T) {
	session := &fakeSession{tools: []*mcp.Tool{
		{Name: "health", InputSchema: objectSchema(), Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true}},
		destructiveTool("stop"),
	}, results: map[string]*mcp.CallToolResult{"health": {Content: []mcp.Content{&mcp.TextContent{Text: "ok"}}}}}
	model := &scriptedLLM{t: t, steps: []llmStep{
		thinkingCallStep("health", "call-1"),
		thinkingCallStep("stop", "call-2"),
	}}
	result, err := newTestAgent(t, model, session, 4).RunTurn(t.Context(), nil, "stop it", func(Event) error { return nil })
	if err != nil || result.Suspended == nil {
		t.Fatalf("result = %#v, err = %v", result, err)
	}
	for _, message := range result.Suspended.Messages {
		if message.Reasoning != nil {
			t.Fatalf("suspended turn keeps reasoning: %#v", message)
		}
	}
	stored, err := json.Marshal(result.Suspended)
	if err != nil || strings.Contains(string(stored), "private") {
		t.Fatalf("stored state = %s, err = %v", stored, err)
	}
}
