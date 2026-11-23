package agent

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/llm"
)

func TestToolsListWhatTheModelMayBeOffered(t *testing.T) {
	tools := previewedTools()
	tools[0].Title, tools[0].Description = "Get object status", "Read the status of one object. Exact values are kept."
	tools[0].InputSchema = map[string]any{"type": "object", "properties": map[string]any{
		"path": map[string]any{"type": "string", "description": " the exact object path "},
		"wait": map[string]any{"type": "integer"},
	}}
	freeze := &mcp.Tool{Name: "freeze_object", InputSchema: objectSchema(), Annotations: &mcp.ToolAnnotations{DestructiveHint: boolPointer(false)}}
	session := &fakeSession{tools: append(tools, freeze)}
	got, err := newTestAgent(t, &scriptedLLM{t: t}, session, 4).Tools(auth.WithBearerToken(t.Context(), "jwt-marker"))
	if err != nil {
		t.Fatal(err)
	}
	// As the user: the MCP serves each user the tools it allows.
	if !session.connectedWithJWT || !session.closed {
		t.Fatalf("connected as the user %v, closed %v", session.connectedWithJWT, session.closed)
	}
	// The previews and the hidden tools are left out; the title falls back
	// to the name; a tool neither read-only nor destructive needs no
	// confirmation.
	want := []ToolInfo{
		{
			Name: "get_object_status", Title: "Get object status", Summary: "Read the status of one object.",
			Description: "Read the status of one object. Exact values are kept.", Arguments: map[string]string{"path": "the exact object path"}, ReadOnly: true,
		},
		{Name: "update_object_config", Title: "Stop object", Confirmable: true},
		{Name: "stop_instance", Title: "Stop object", Confirmable: true},
		{Name: "freeze_object", Title: "freeze_object"},
	}
	if len(got) != len(want) {
		t.Fatalf("tools = %+v", got)
	}
	for index := range want {
		if !reflect.DeepEqual(got[index], want[index]) {
			t.Fatalf("tool %d = %+v, want %+v", index, got[index], want[index])
		}
	}
}

func TestToolsFailWithTheMCP(t *testing.T) {
	agent, err := New(llm.SingleModel(&scriptedLLM{t: t}, "test-model"), func(context.Context) (MCPSession, error) {
		return nil, errors.New("connection refused")
	}, Config{SystemPrompt: testSystemPrompt, MaxIterations: 4, Timeout: 60e9})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Tools(t.Context()); err == nil || !strings.Contains(err.Error(), "connection refused") {
		t.Fatalf("error = %v", err)
	}
}

func TestToolSummaryIsTheFirstSentenceBounded(t *testing.T) {
	for description, want := range map[string]string{
		"":                                 "",
		"One sentence only.":               "One sentence only.",
		"Stop it: on node n1. Then more.":  "Stop it: on node n1.",
		"A path such as test/svc/web.x is": "A path such as test/svc/web.x is",
		"Ends on a line.\nThen more.":      "Ends on a line.",
		"  Trimmed.  ":                     "Trimmed.",
	} {
		if got := firstSentence(description); got != want {
			t.Fatalf("firstSentence(%q) = %q, want %q", description, got, want)
		}
	}
	long := bounded(firstSentence(strings.Repeat("é", maxToolSummaryBytes)), maxToolSummaryBytes)
	if len(long) > maxToolSummaryBytes+len("…") || !strings.HasSuffix(long, "…") || !strings.HasPrefix(long, "é") {
		t.Fatalf("long summary = %d bytes", len(long))
	}
	if title := bounded(strings.Repeat("t", 300), maxToolTitleBytes); len(title) != maxToolTitleBytes+len("…") {
		t.Fatalf("long title = %d bytes", len(title))
	}
}
