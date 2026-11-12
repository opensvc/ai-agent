package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/opensvc/ai-agent/internal/llm"
)

// rawToolsSession lists its tools as they are: the fake session announces a
// tool without annotations as read-only, which an unattended turn must not
// take for granted.
type rawToolsSession struct{ *fakeSession }

func (s rawToolsSession) ListTools(context.Context) ([]*mcp.Tool, error) { return s.tools, nil }

func unattendedCatalog() []*mcp.Tool {
	notDestructive, destructive := false, true
	return []*mcp.Tool{
		{Name: "list_nodes", InputSchema: objectSchema(), Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true}},
		// A write the chat runs without confirmation.
		{Name: "attach_node_tag", InputSchema: objectSchema(), Annotations: &mcp.ToolAnnotations{DestructiveHint: &notDestructive}},
		{Name: "stop_object", InputSchema: objectSchema(), Annotations: &mcp.ToolAnnotations{DestructiveHint: &destructive}},
		// Nothing says what it does.
		{Name: "mystery", InputSchema: objectSchema()},
		// Read-only, but for the client only.
		{Name: "preview_stop", InputSchema: objectSchema(), Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true}, Meta: mcp.Meta{hiddenToolMeta: true}},
	}
}

func unattendedAgent(t *testing.T, model llm.Client, session MCPSession) *Agent {
	t.Helper()
	agent, err := New(model, func(context.Context) (MCPSession, error) { return session, nil },
		Config{SystemPrompt: testSystemPrompt, MaxIterations: 4, Timeout: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	return agent
}

// An unattended turn offers the model the read-only tools only: not the
// writes the chat runs without confirmation, not the destructive ones, not a
// tool without annotations, not the hidden ones. A call of a tool left out,
// which the model may know of from a skill or the data it read, is answered
// as unoffered, without reaching the MCP.
func TestUnattendedTurnOffersTheReadOnlyToolsOnly(t *testing.T) {
	session := &fakeSession{tools: unattendedCatalog()}
	var first, second llm.Request
	model := &scriptedLLM{t: t, steps: []llmStep{
		func(request llm.Request, emit llm.EmitFunc) error {
			first = request
			return callsStep(llm.ToolCall{ID: "call-1", Name: "attach_node_tag", Arguments: json.RawMessage(`{}`)})(request, emit)
		},
		answerStep(t, func(request llm.Request) { second = request }),
	}}
	var events []Event
	result, err := unattendedAgent(t, model, rawToolsSession{session}).RunTurn(WithUnattended(t.Context()), nil, "report", func(event Event) error {
		events = append(events, event)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if names := toolNames(first.Tools); !slices.Equal(names, []string{"list_nodes"}) {
		t.Fatalf("tools offered = %v", names)
	}
	if system := first.Messages[0].Text; !strings.Contains(system, unattendedInstruction) {
		t.Fatalf("system prompt = %q", system)
	}
	if len(session.calls) != 0 {
		t.Fatalf("the MCP was called: %+v", session.calls)
	}
	answer := second.Messages[len(second.Messages)-1].ToolResults
	if len(answer) != 1 || !answer[0].IsError || !strings.Contains(string(answer[0].Content), "no tool of this name is offered") {
		t.Fatalf("answer to the call = %+v", answer)
	}
	notRun := ""
	for _, event := range events {
		if event.Type == EventToolFinished && event.ToolCallID == "call-1" {
			notRun = event.NotRun
		}
	}
	if notRun != NotRunUnoffered {
		t.Fatalf("the call of the write is told not run %q", notRun)
	}
	if result.Suspended != nil || events[len(events)-1].Type != EventCompleted {
		t.Fatalf("the turn did not complete: %+v", result.Suspended)
	}
}

// The same catalog, attended: the writes are offered, as before.
func TestAttendedTurnOffersTheWrites(t *testing.T) {
	var first llm.Request
	model := &scriptedLLM{t: t, steps: []llmStep{answerStep(t, func(request llm.Request) { first = request })}}
	if _, err := unattendedAgent(t, model, rawToolsSession{&fakeSession{tools: unattendedCatalog()}}).RunTurn(t.Context(), nil, "report", func(Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if names := toolNames(first.Tools); !slices.Equal(names, []string{"list_nodes", "attach_node_tag", "stop_object", "mystery"}) {
		t.Fatalf("tools offered = %v", names)
	}
}

// A skill naming a tool left out tells the model it is not served, as for
// an MCP started without its actions.
func TestUnattendedSkillNamesItsWritesAsNotServed(t *testing.T) {
	const text = "---\nname: morning\ndescription: Use for the morning report.\nmetadata:\n  opensvc/tools: list_nodes stop_object\n---\n\nList, then stop what is down.\n"
	sum := sha256.Sum256([]byte(text))
	session := &fakeSession{
		tools: unattendedCatalog(),
		texts: map[string]string{"skill://morning/SKILL.md": text},
		skills: []Skill{{
			URI: "skill://morning/SKILL.md",
			Frontmatter: map[string]any{"name": "morning", "description": "Use for the morning report.",
				"metadata": map[string]any{"opensvc/tools": "list_nodes stop_object"}},
			Files: []SkillFile{{URI: "skill://morning/SKILL.md", Digest: "sha256:" + hex.EncodeToString(sum[:]), Size: int64(len(text))}},
		}},
	}
	var second llm.Request
	model := &scriptedLLM{t: t, steps: []llmStep{
		callsStep(llm.ToolCall{ID: "call-1", Name: LoadSkillToolName, Arguments: json.RawMessage(`{"skill":"morning"}`)}),
		answerStep(t, func(request llm.Request) { second = request }),
	}}
	if _, err := unattendedAgent(t, model, rawToolsSession{session}).RunTurn(WithUnattended(t.Context()), nil, "report", func(Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	loaded := second.Messages[len(second.Messages)-1].ToolResults
	if len(loaded) != 1 || !strings.Contains(string(loaded[0].Content), "Not served by this backend: stop_object") {
		t.Fatalf("load_skill = %+v", loaded)
	}
	if slices.Contains(toolNames(second.Tools), "stop_object") {
		t.Fatalf("the skill offered stop_object: %v", toolNames(second.Tools))
	}
}

func TestUnattendedTurnRefusesTheToolsOfTheClient(t *testing.T) {
	ctx := WithClientTools(WithUnattended(t.Context()), []ClientTool{{Name: "open_page", Parameters: json.RawMessage(`{"type":"object"}`)}})
	model := &scriptedLLM{t: t}
	if _, err := unattendedAgent(t, model, &fakeSession{}).RunTurn(ctx, nil, "report", func(Event) error { return nil }); err == nil {
		t.Fatal("an unattended turn took the tools of the client")
	}
}
