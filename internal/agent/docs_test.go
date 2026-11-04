package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/opensvc/ai-agent/internal/docs"
	"github.com/opensvc/ai-agent/internal/llm"
)

// fixedDocs serves one book, and keeps the version each search asked.
type fixedDocs struct {
	library *docs.Library
	asked   []string
}

func (f *fixedDocs) Versions() ([]string, string) {
	return []string{"3.1", "3.0"}, "the release line of this cluster, else the latest, 3.1"
}

func (f *fixedDocs) Docs(_ context.Context, _ MCPSession, version string) (*docs.Library, string, error) {
	f.asked = append(f.asked, version)
	return f.library, "This is the book of the 3.0 release line, the one of " + version + ".", nil
}

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

func TestSearchDocsGivesTheSectionsOfTheBookOfTheVersionAsked(t *testing.T) {
	page := "# Cluster Quorum\n\n## Best Practices\n\n" + strings.Repeat("Configure the number of nodes minus 1 arbitrators, all on the same third site. ", 4) + "\n"
	library, err := docs.New([]docs.Page{{Path: "agent/configure.quorum.md", Text: page}}, "the test book at commit abc")
	if err != nil {
		t.Fatal(err)
	}
	source := &fixedDocs{library: library}
	call := func(arguments string) llm.ToolResult {
		result, err := searchDocs(context.Background(), source, nil, llm.ToolCall{ID: "1", Name: SearchDocsToolName, Arguments: json.RawMessage(arguments)})
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	result := call(`{"query":"how many arbitrators","version":" v3.0.0-rc56 "}`)
	got := toolText(t, result)
	if result.IsError || !strings.HasPrefix(got, "Documentation: the test book at commit abc. This is the book of the 3.0 release line, the one of v3.0.0-rc56.\n\nSection: Cluster Quorum\nPage: agent/configure.quorum.md\n\n## Best Practices\n\nConfigure the number") {
		t.Errorf("result = %v %q", result.IsError, got)
	}
	call(`{"query":"how many arbitrators"}`)
	if len(source.asked) != 2 || source.asked[0] != "v3.0.0-rc56" || source.asked[1] != "" {
		t.Errorf("versions asked = %q", source.asked)
	}
	for arguments, want := range map[string]string{
		`{"query":"zzz"}`:              "No section of the documentation matches",
		`{"query":"quorum","limit":9}`: "limit must be between 1 and 5",
		`{"limit":"x"}`:                "search_docs takes a query",
		`{"query":"quorum","version":"` + strings.Repeat("3", maxVersionBytes+1) + `"}`: "an optional release line",
	} {
		result := call(arguments)
		if !result.IsError || !strings.Contains(toolText(t, result), want) {
			t.Errorf("%s: result = %v %q, want %q", arguments, result.IsError, toolText(t, result), want)
		}
	}
}

func TestDocsToolListsTheReleaseLines(t *testing.T) {
	tool := docsTool(&fixedDocs{})
	var schema struct {
		Properties map[string]struct {
			Description string `json:"description"`
			MaxLength   int    `json:"maxLength"`
		} `json:"properties"`
	}
	if err := json.Unmarshal(tool.InputSchema, &schema); err != nil {
		t.Fatal(err)
	}
	version := schema.Properties["version"]
	if !strings.Contains(version.Description, "the lines with a book: 3.1, 3.0; by default the release line of this cluster, else the latest, 3.1") || version.MaxLength != maxVersionBytes {
		t.Errorf("version = %+v", version)
	}
}

// shelfOf gives its book, the latest, 3.1, and keeps the version asked.
type shelfOf struct {
	library *docs.Library
	asked   []string
}

func (s *shelfOf) Lines() []string { return []string{"3.1", "3.0"} }
func (s *shelfOf) Book(_ context.Context, version string) (*docs.Library, string) {
	s.asked = append(s.asked, version)
	return s.library, "No release line was asked: this is the book of the latest one, 3.1."
}

func TestBooksDefaultsToTheVersionOfTheBackend(t *testing.T) {
	shelf := &shelfOf{}
	plain := Books{Shelf: shelf}
	if lines, byDefault := plain.Versions(); len(lines) != 2 || byDefault != "the latest, 3.1" {
		t.Errorf("no default: %v, %q", lines, byDefault)
	}
	plain.Docs(context.Background(), nil, "")
	backend := Books{Shelf: shelf, Default: func(context.Context, MCPSession) (string, error) { return "v3.0.0", nil }, DefaultIs: "the release line of this cluster"}
	if _, byDefault := backend.Versions(); byDefault != "the release line of this cluster, else the latest, 3.1" {
		t.Errorf("default: %q", byDefault)
	}
	// The version the model names comes first, the default of the backend
	// when it names none.
	backend.Docs(context.Background(), nil, "3.1")
	backend.Docs(context.Background(), nil, "")
	if want := []string{"", "3.1", "v3.0.0"}; strings.Join(shelf.asked, ",") != strings.Join(want, ",") {
		t.Errorf("versions asked = %q, want %q", shelf.asked, want)
	}
}

func quorumBook(t *testing.T) *docs.Library {
	t.Helper()
	page := "# Cluster Quorum\n\n## Best Practices\n\n" + strings.Repeat("Configure the number of nodes minus 1 arbitrators, all on the same third site. ", 4) + "\n"
	library, err := docs.New([]docs.Page{{Path: "agent/configure.quorum.md", Text: page}}, "the test book")
	if err != nil {
		t.Fatal(err)
	}
	return library
}

func TestBooksSaysTheVersionOfTheBackendCouldNotBeRead(t *testing.T) {
	shelf := &shelfOf{library: quorumBook(t)}
	books := Books{Shelf: shelf, Default: func(context.Context, MCPSession) (string, error) { return "", nil }, Unknown: "The version the nodes of this cluster run could not be read"}
	if _, says, _ := books.Docs(context.Background(), nil, ""); says != "The version the nodes of this cluster run could not be read: this is the book of the latest release line, 3.1." {
		t.Errorf("says %q", says)
	}
}

func TestSearchDocsWithNoBookSaysTheDocumentationCannotBeRead(t *testing.T) {
	result, err := searchDocs(context.Background(), Books{Shelf: &shelfOf{}}, nil, llm.ToolCall{ID: "1", Name: SearchDocsToolName, Arguments: json.RawMessage(`{"query":"quorum"}`)})
	if err != nil || !result.IsError || !strings.Contains(toolText(t, result), "The documentation cannot be read at the moment") {
		t.Errorf("result = %v %q", result.IsError, toolText(t, result))
	}
}

// TestSearchDocsAuditsTheCallsOfTheAgent runs a turn whose model searches the
// documentation with no version: the backend reads the version of the
// cluster through the MCP, a call the audit is told and the client not.
func TestSearchDocsAuditsTheCallsOfTheAgent(t *testing.T) {
	session := &fakeSession{tools: []*mcp.Tool{{Name: "get_cluster_status", InputSchema: objectSchema()}}}
	shelf := &shelfOf{library: quorumBook(t)}
	books := Books{Shelf: shelf, Default: func(ctx context.Context, session MCPSession) (string, error) {
		if _, err := session.CallTool(ctx, "get_cluster_status", map[string]any{"node_limit": 200}); err != nil {
			return "", err
		}
		return "v3.0.0-rc56", nil
	}, DefaultIs: "the release line of this cluster"}
	model := &scriptedLLM{t: t, steps: []llmStep{
		func(_ llm.Request, emit llm.EmitFunc) error {
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
		Config{SystemPrompt: testSystemPrompt, MaxIterations: 3, Timeout: time.Minute, Docs: books})
	if err != nil {
		t.Fatal(err)
	}
	var types []EventType
	var called *AgentCall
	if _, err := agent.RunTurn(t.Context(), nil, "how many arbitrators", func(event Event) error {
		types = append(types, event.Type)
		if event.Type == EventAgentToolCalled {
			called = event.AgentCall
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(session.calls) != 1 || session.calls[0].name != "get_cluster_status" || len(shelf.asked) != 1 || shelf.asked[0] != "v3.0.0-rc56" {
		t.Fatalf("MCP calls %#v, versions asked %q", session.calls, shelf.asked)
	}
	if called == nil || called.Tool != "get_cluster_status" || called.Purpose != "documentation_version" || called.Error {
		t.Fatalf("agent call = %+v", called)
	}
	// The event comes between the start and the end of the search.
	order := fmt.Sprint(types)
	if !strings.Contains(order, "tool_started agent_tool_called tool_finished") {
		t.Errorf("events = %s", order)
	}
}

// TestSearchDocsEndsTheTurnOnAFailedTransport runs a turn whose search reads
// the version of the cluster through an MCP whose transport fails: the turn
// ends on the error, as on a failed call of the model, rather than the
// search going on with the latest book.
func TestSearchDocsEndsTheTurnOnAFailedTransport(t *testing.T) {
	session := &fakeSession{tools: []*mcp.Tool{{Name: "get_cluster_status", InputSchema: objectSchema()}}, callErr: errors.New("connection reset by peer")}
	shelf := &shelfOf{library: quorumBook(t)}
	books := Books{Shelf: shelf, Default: func(ctx context.Context, session MCPSession) (string, error) {
		_, err := session.CallTool(ctx, "get_cluster_status", map[string]any{"node_limit": 200})
		return "", err
	}}
	model := &scriptedLLM{t: t, steps: []llmStep{
		func(_ llm.Request, emit llm.EmitFunc) error {
			return emitEvents(emit,
				llm.Event{Type: llm.EventToolCall, ToolCall: &llm.ToolCall{ID: "call-1", Name: SearchDocsToolName, Arguments: json.RawMessage(`{"query":"arbitrators"}`)}},
				llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonToolCalls},
			)
		},
	}}
	agent, err := New(model, func(context.Context) (MCPSession, error) { return session, nil },
		Config{SystemPrompt: testSystemPrompt, MaxIterations: 3, Timeout: time.Minute, Docs: books})
	if err != nil {
		t.Fatal(err)
	}
	var called *AgentCall
	finished := false
	_, err = agent.RunTurn(t.Context(), nil, "how many arbitrators", func(event Event) error {
		switch event.Type {
		case EventAgentToolCalled:
			called = event.AgentCall
		case EventToolFinished:
			finished = true
		}
		return nil
	})
	if err == nil || !strings.Contains(err.Error(), "connection reset by peer") {
		t.Fatalf("RunTurn() error = %v, want the transport error", err)
	}
	if called == nil || !called.Error || finished || len(shelf.asked) != 0 {
		t.Errorf("agent call %+v, search finished %v, books asked %q", called, finished, shelf.asked)
	}
}

// TestSearchDocsReadsNoVersionOfAToolNotListed runs a turn on an MCP that
// does not list get_cluster_status, such as an older one: the agent does not
// call it, the backend goes on with no version, and the search answers from
// the latest book rather than the turn failing.
func TestSearchDocsReadsNoVersionOfAToolNotListed(t *testing.T) {
	session := &fakeSession{tools: []*mcp.Tool{{Name: "get_node", InputSchema: objectSchema()}}}
	shelf := &shelfOf{library: quorumBook(t)}
	books := Books{Shelf: shelf, Default: func(ctx context.Context, session MCPSession) (string, error) {
		_, err := session.CallTool(ctx, "get_cluster_status", map[string]any{"node_limit": 200})
		if errors.Is(err, ErrToolNotListed) {
			return "", nil
		}
		return "v3.0.0", err
	}, Unknown: "The version of this cluster could not be read"}
	model := &scriptedLLM{t: t, steps: []llmStep{
		func(_ llm.Request, emit llm.EmitFunc) error {
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
		Config{SystemPrompt: testSystemPrompt, MaxIterations: 3, Timeout: time.Minute, Docs: books})
	if err != nil {
		t.Fatal(err)
	}
	var result string
	audited := false
	if _, err := agent.RunTurn(t.Context(), nil, "how many arbitrators", func(event Event) error {
		switch event.Type {
		case EventAgentToolCalled:
			audited = true
		case EventToolFinished:
			result = string(event.ToolResult)
		}
		return nil
	}); err != nil {
		t.Fatalf("RunTurn() error = %v", err)
	}
	if len(session.calls) != 0 || audited || len(shelf.asked) != 1 || shelf.asked[0] != "" {
		t.Fatalf("MCP calls %#v, audited %v, versions asked %q", session.calls, audited, shelf.asked)
	}
	if !strings.Contains(result, "The version of this cluster could not be read") || !strings.Contains(result, "Configure the number of nodes") {
		t.Errorf("search result = %s", result)
	}
}

// TestBooksReadsNoVersionWithoutABook calls no Default while the shelf has
// no book of a known line: the version would change nothing.
func TestBooksReadsNoVersionWithoutABook(t *testing.T) {
	called := false
	books := Books{Shelf: emptyShelf{}, Default: func(context.Context, MCPSession) (string, error) {
		called = true
		return "v3.0.0", nil
	}}
	if _, _, err := books.Docs(context.Background(), nil, ""); err != nil || called {
		t.Errorf("err %v, default called %v", err, called)
	}
}

// emptyShelf has no book.
type emptyShelf struct{}

func (emptyShelf) Lines() []string { return nil }
func (emptyShelf) Book(context.Context, string) (*docs.Library, string) {
	return nil, "The documentation could not be read."
}
