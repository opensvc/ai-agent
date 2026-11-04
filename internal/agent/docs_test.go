package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/opensvc/ai-agent/internal/docs"
	"github.com/opensvc/ai-agent/internal/llm"
)

func toolText(t *testing.T, result llm.ToolResult) string {
	t.Helper()
	var content struct {
		Content []struct{ Text string } `json:"content"`
	}
	if err := json.Unmarshal(result.Content, &content); err != nil || len(content.Content) != 1 {
		t.Fatalf("content = %s", result.Content)
	}
	return content.Content[0].Text
}

func quorumBook(t *testing.T) *docs.Library {
	t.Helper()
	page := "# Cluster Quorum\n\n## Best Practices\n\n" + strings.Repeat("Configure the number of nodes minus 1 arbitrators, all on the same third site. ", 4) + "\n"
	library, err := docs.New([]docs.Page{{Path: "agent/configure.quorum.md", Text: page}}, "the test book at commit abc")
	if err != nil {
		t.Fatal(err)
	}
	return library
}

func TestSearchDocsGivesTheSectionsAndTheirBook(t *testing.T) {
	library := quorumBook(t)
	call := func(arguments string) llm.ToolResult {
		return searchDocs(library, llm.ToolCall{ID: "1", Name: SearchDocsToolName, Arguments: json.RawMessage(arguments)})
	}
	result := call(`{"query":"how many arbitrators"}`)
	got := toolText(t, result)
	if result.IsError || !strings.HasPrefix(got, "Documentation: the test book at commit abc.\n\nSection: Cluster Quorum\nPage: agent/configure.quorum.md\n\n## Best Practices\n\nConfigure the number") ||
		!strings.HasSuffix(got, docsReading) {
		t.Errorf("result = %v %q", result.IsError, got)
	}
	for arguments, want := range map[string]string{
		`{"query":"zzz"}`:              "No section of the documentation matches",
		`{"query":"quorum","limit":9}`: "limit must be between 1 and 5",
		`{"limit":"x"}`:                "search_docs takes a query",
	} {
		result := call(arguments)
		if !result.IsError || !strings.Contains(toolText(t, result), want) {
			t.Errorf("%s: result = %v %q, want %q", arguments, result.IsError, toolText(t, result), want)
		}
	}
}

// TestSearchDocsRunsInATurn offers search_docs with a book, runs the search
// the model asks, and gives the model its sections.
func TestSearchDocsRunsInATurn(t *testing.T) {
	session := &fakeSession{}
	var offered []string
	model := &scriptedLLM{t: t, steps: []llmStep{
		func(request llm.Request, emit llm.EmitFunc) error {
			for _, tool := range request.Tools {
				offered = append(offered, tool.Name)
			}
			return emitEvents(emit,
				llm.Event{Type: llm.EventToolCall, ToolCall: &llm.ToolCall{ID: "call-1", Name: SearchDocsToolName, Arguments: json.RawMessage(`{"query":"arbitrators"}`)}},
				llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonToolCalls},
			)
		},
		func(_ llm.Request, emit llm.EmitFunc) error {
			return emitEvents(emit,
				llm.Event{Type: llm.EventTextDelta, TextDelta: "Two arbitrators."},
				llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonCompleted},
			)
		},
	}}
	agent, err := New(model, func(context.Context) (MCPSession, error) { return session, nil },
		Config{SystemPrompt: testSystemPrompt, MaxIterations: 3, Timeout: time.Minute, Docs: quorumBook(t)})
	if err != nil {
		t.Fatal(err)
	}
	var result string
	if _, err := agent.RunTurn(t.Context(), nil, "how many arbitrators", func(event Event) error {
		if event.Type == EventToolFinished {
			result = string(event.ToolResult)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(offered) != 1 || offered[0] != SearchDocsToolName || !strings.Contains(result, "Configure the number of nodes") || len(session.calls) != 0 {
		t.Errorf("offered %v, result %s, MCP calls %#v", offered, result, session.calls)
	}
}
