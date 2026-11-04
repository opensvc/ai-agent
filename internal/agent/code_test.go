package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/opensvc/ai-agent/internal/docs"
	"github.com/opensvc/ai-agent/internal/llm"
)

const speakerCode = `package nmon

// speakerNode return the speaker node: the first alive node from
// clusterConfig.Nodes.
func (t *Manager) speakerNode() string {
	for _, nodename := range t.clusterConfig.Nodes {
		return nodename
	}
	return ""
}
`

// codeShelfOf gives its index for any version, the latest at main, and
// keeps the versions asked.
type codeShelfOf struct {
	index *docs.CodeIndex
	asked []string
}

func (s *codeShelfOf) Latest() string { return "main" }
func (s *codeShelfOf) Code(_ context.Context, version string) (*docs.CodeIndex, string) {
	s.asked = append(s.asked, version)
	if version == "" {
		return s.index, "This is the latest code of om3, at main."
	}
	if version == "rc56" {
		return nil, `"rc56" names no version of the code.`
	}
	return s.index, "This is the code of " + version + "."
}

func speakerIndex(t *testing.T) *docs.CodeIndex {
	t.Helper()
	index, err := docs.NewCode([]docs.File{{Path: "daemon/nmon/speaker.go", Text: speakerCode}}, "the code of 451250d44")
	if err != nil {
		t.Fatal(err)
	}
	return index
}

func TestSearchCodeGivesTheDeclarationsWithTheirFileAndLines(t *testing.T) {
	shelf := &codeShelfOf{index: speakerIndex(t)}
	call := func(arguments string) llm.ToolResult {
		return searchCode(context.Background(), shelf, llm.ToolCall{ID: "1", Name: SearchCodeToolName, Arguments: json.RawMessage(arguments)})
	}
	// With no version, the latest code.
	result := call(`{"query":"speaker first alive node"}`)
	got := toolText(t, result)
	if result.IsError || !strings.HasPrefix(got, "Code: the code of 451250d44. This is the latest code of om3, at main.\n\nFile: daemon/nmon/speaker.go:3-10\n\n```go\n// speakerNode return") ||
		!strings.HasSuffix(got, codeReading) {
		t.Errorf("result = %v %q", result.IsError, got)
	}
	// A version the user named is searched.
	if result := call(`{"query":"speaker first alive node","version":"v3.0.0-rc56"}`); result.IsError || !strings.Contains(toolText(t, result), "This is the code of v3.0.0-rc56.") {
		t.Errorf("a version: %q", toolText(t, result))
	}
	// A version that names no code: why, and how to go on.
	if result := call(`{"query":"speaker","version":"rc56"}`); !result.IsError || !strings.Contains(toolText(t, result), `"rc56" names no version of the code. Answer without the code, or search again without a version`) {
		t.Errorf("no code: %q", toolText(t, result))
	}
	if result := call(`{"query":"zzz"}`); !result.IsError || !strings.Contains(toolText(t, result), "No declaration of the code matches") {
		t.Errorf("no match: %q", toolText(t, result))
	}
	if len(shelf.asked) != 4 || shelf.asked[0] != "" || shelf.asked[1] != "v3.0.0-rc56" {
		t.Errorf("versions asked = %q", shelf.asked)
	}
}

func TestCodeToolSearchesTheLatestCodeUnlessTheUserNamesAVersion(t *testing.T) {
	var schema struct {
		Properties map[string]struct {
			Description string `json:"description"`
		} `json:"properties"`
	}
	tool := codeTool(&codeShelfOf{})
	if err := json.Unmarshal(tool.InputSchema, &schema); err != nil {
		t.Fatal(err)
	}
	version := schema.Properties["version"].Description
	if !strings.HasPrefix(version, "leave it out to search the latest code of the daemon, at main; set it only when the user names a version") ||
		strings.Contains(version, "451250d44") || !strings.Contains(tool.Description, "its latest code unless the user names a version") {
		t.Errorf("version %q, tool %q", version, tool.Description)
	}
}

// TestSearchDocsTellsToSearchTheCode follows the sections of a documentation
// search, and its refusal when none matches, with where to search what the
// book does not say.
func TestSearchDocsTellsToSearchTheCode(t *testing.T) {
	source := &fixedDocs{library: quorumBook(t)}
	for arguments, want := range map[string]string{
		`{"query":"how many arbitrators"}`: docsReading + " " + docsThenCode,
		`{"query":"zzz"}`:                  "answer that the documentation does not say. " + docsThenCode,
	} {
		result, err := searchDocs(context.Background(), source, nil, llm.ToolCall{ID: "1", Name: SearchDocsToolName, Arguments: json.RawMessage(arguments)}, docsThenCode)
		if err != nil || !strings.HasSuffix(toolText(t, result), want) {
			t.Errorf("%s: %q", arguments, toolText(t, result))
		}
	}
}

// TestSearchCodeRunsInATurn runs a turn whose model searches the code with
// no version: the latest code, read with no call to the MCP.
func TestSearchCodeRunsInATurn(t *testing.T) {
	session := &fakeSession{tools: []*mcp.Tool{{Name: "get_cluster_status", InputSchema: objectSchema()}}}
	shelf := &codeShelfOf{index: speakerIndex(t)}
	var offered []string
	model := &scriptedLLM{t: t, steps: []llmStep{
		func(request llm.Request, emit llm.EmitFunc) error {
			for _, tool := range request.Tools {
				offered = append(offered, tool.Name)
			}
			return emitEvents(emit,
				llm.Event{Type: llm.EventToolCall, ToolCall: &llm.ToolCall{ID: "call-1", Name: SearchCodeToolName, Arguments: json.RawMessage(`{"query":"speaker node"}`)}},
				llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonToolCalls},
			)
		},
		func(_ llm.Request, emit llm.EmitFunc) error {
			return emitEvents(emit,
				llm.Event{Type: llm.EventTextDelta, TextDelta: "The first alive node."},
				llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonCompleted},
			)
		},
	}}
	agent, err := New(model, func(context.Context) (MCPSession, error) { return session, nil },
		Config{SystemPrompt: testSystemPrompt, MaxIterations: 3, Timeout: time.Minute, Code: shelf})
	if err != nil {
		t.Fatal(err)
	}
	var result string
	audited := false
	if _, err := agent.RunTurn(t.Context(), nil, "who is the speaker", func(event Event) error {
		switch event.Type {
		case EventAgentToolCalled:
			audited = true
		case EventToolFinished:
			result = string(event.ToolResult)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.Join(offered, " "), SearchCodeToolName) || audited || len(session.calls) != 0 || len(shelf.asked) != 1 || shelf.asked[0] != "" || !strings.Contains(result, "first alive node") {
		t.Errorf("offered %v, audited %v, MCP calls %#v, versions asked %q, result %s", offered, audited, session.calls, shelf.asked, result)
	}
}

func TestAnMCPToolNamedSearchCodeFailsTheTurn(t *testing.T) {
	session := &fakeSession{tools: []*mcp.Tool{{Name: SearchCodeToolName, InputSchema: objectSchema()}}}
	model := &scriptedLLM{t: t}
	agent, err := New(model, func(context.Context) (MCPSession, error) { return session, nil },
		Config{SystemPrompt: testSystemPrompt, MaxIterations: 3, Timeout: time.Minute, Code: &codeShelfOf{}})
	if err != nil {
		t.Fatal(err)
	}
	_, err = agent.RunTurn(t.Context(), nil, "who is the speaker", func(Event) error { return nil })
	if err == nil || !strings.Contains(err.Error(), `MCP tool "search_code" conflicts with the code search of the agent`) {
		t.Errorf("err = %v", err)
	}
}

// TestADocsSearchPointsToTheCodeOnlyWhenItCanBeSearched runs a documentation
// search in a turn: its result tells to search the code when the agent has
// the code, never otherwise.
func TestADocsSearchPointsToTheCodeOnlyWhenItCanBeSearched(t *testing.T) {
	for _, withCode := range []bool{true, false} {
		config := Config{SystemPrompt: testSystemPrompt, MaxIterations: 3, Timeout: time.Minute, Docs: &fixedDocs{library: quorumBook(t)}}
		if withCode {
			config.Code = &codeShelfOf{}
		}
		model := &scriptedLLM{t: t, steps: []llmStep{
			func(_ llm.Request, emit llm.EmitFunc) error {
				return emitEvents(emit,
					llm.Event{Type: llm.EventToolCall, ToolCall: &llm.ToolCall{ID: "call-1", Name: SearchDocsToolName, Arguments: json.RawMessage(`{"query":"arbitrators"}`)}},
					llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonToolCalls},
				)
			},
			func(_ llm.Request, emit llm.EmitFunc) error {
				return emitEvents(emit, llm.Event{Type: llm.EventTextDelta, TextDelta: "Two."}, llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonCompleted})
			},
		}}
		agent, err := New(model, func(context.Context) (MCPSession, error) { return &fakeSession{}, nil }, config)
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
		if strings.Contains(result, "search_code") != withCode {
			t.Errorf("with the code %v: result %s", withCode, result)
		}
	}
}

// TestCodeSearchesOfAQuestionAreBounded answers the searches of a question
// past maxCodeSearchesPerAsk with a tool error, the code not searched.
func TestCodeSearchesOfAQuestionAreBounded(t *testing.T) {
	session := &fakeSession{tools: []*mcp.Tool{{Name: "get_cluster_status", InputSchema: objectSchema()}}}
	shelf := &codeShelfOf{index: speakerIndex(t)}
	search := func(from, count int) llmStep {
		return func(_ llm.Request, emit llm.EmitFunc) error {
			var events []llm.Event
			for i := from; i < from+count; i++ {
				events = append(events, llm.Event{Type: llm.EventToolCall, ToolCall: &llm.ToolCall{ID: "call-" + string(rune('a'+i)), Name: SearchCodeToolName, Arguments: json.RawMessage(`{"query":"speaker node"}`)}})
			}
			return emitEvents(emit, append(events, llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonToolCalls})...)
		}
	}
	model := &scriptedLLM{t: t, steps: []llmStep{
		search(0, 4),
		search(4, 4),
		func(_ llm.Request, emit llm.EmitFunc) error {
			return emitEvents(emit,
				llm.Event{Type: llm.EventTextDelta, TextDelta: "The first alive node."},
				llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonCompleted},
			)
		},
	}}
	agent, err := New(model, func(context.Context) (MCPSession, error) { return session, nil },
		Config{SystemPrompt: testSystemPrompt, MaxIterations: 4, Timeout: time.Minute, Code: shelf})
	if err != nil {
		t.Fatal(err)
	}
	var spent int
	if _, err := agent.RunTurn(t.Context(), nil, "how does the daemon elect its speaker", func(event Event) error {
		if event.Type == EventToolFinished && event.ToolError && strings.Contains(string(event.ToolResult), "code searches of this question are spent") {
			spent++
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(shelf.asked) != maxCodeSearchesPerAsk || spent != 8-maxCodeSearchesPerAsk {
		t.Errorf("searches run %d, refused %d; want %d and %d", len(shelf.asked), spent, maxCodeSearchesPerAsk, 8-maxCodeSearchesPerAsk)
	}
}
