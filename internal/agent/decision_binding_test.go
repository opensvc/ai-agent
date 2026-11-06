package agent

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/opensvc/ai-agent/internal/llm"
)

// suspendOnTheFirstOfTwoActions runs a turn whose answer calls two
// confirmed actions, switch_object then delete_object, and returns the turn
// suspended on the first, round-tripped through JSON as the store keeps it.
func suspendOnTheFirstOfTwoActions(t *testing.T, session *fakeSession) (*Agent, SuspendedTurn) {
	t.Helper()
	model := &scriptedLLM{t: t, steps: []llmStep{
		callsStep(
			llm.ToolCall{ID: "call-1", Name: "switch_object", Arguments: json.RawMessage(`{"path":"web","to":"n1"}`)},
			llm.ToolCall{ID: "call-2", Name: "delete_object", Arguments: json.RawMessage(`{"path":"db"}`)},
		),
		answerStep(t, nil),
	}}
	agent := newTestAgent(t, model, session, 4)
	first, err := agent.RunTurn(t.Context(), nil, "switch web", func(Event) error { return nil })
	if err != nil || first.Suspended == nil {
		t.Fatalf("first run: %+v, %v", first, err)
	}
	if call, _ := first.Suspended.PendingCall(); call.ID != "call-1" || len(session.calls) != 0 {
		t.Fatalf("suspended on %q after %d calls", call.ID, len(session.calls))
	}
	encoded, err := json.Marshal(first.Suspended)
	if err != nil {
		t.Fatal(err)
	}
	var stored SuspendedTurn
	if err := json.Unmarshal(encoded, &stored); err != nil {
		t.Fatal(err)
	}
	return agent, stored
}

// TestADecisionAppliesToTheCallShownOnly resumes a turn whose confirmed
// tool is no longer offered, removed, hidden or named as the preview of
// another: the decision, approval with or without edited arguments or
// rejection, must not pass to the next call of the answer, never shown. The
// turn fails, and nothing runs. While the tool is still offered, it runs
// with the edit and the next call waits for its own confirmation.
func TestADecisionAppliesToTheCallShownOnly(t *testing.T) {
	offered := func() []*mcp.Tool { return []*mcp.Tool{editableTool(), destructiveTool("delete_object")} }
	drifts := map[string]func() []*mcp.Tool{
		"removed": func() []*mcp.Tool { return []*mcp.Tool{destructiveTool("delete_object")} },
		"hidden": func() []*mcp.Tool {
			hidden := editableTool()
			hidden.Meta = mcp.Meta{hiddenToolMeta: true}
			return []*mcp.Tool{hidden, destructiveTool("delete_object")}
		},
		"named as a preview": func() []*mcp.Tool {
			other := destructiveTool("delete_object")
			other.Meta = mcp.Meta{previewToolMeta: "switch_object"}
			return []*mcp.Tool{editableTool(), other}
		},
	}
	decisions := map[string]struct {
		decision Decision
		edited   json.RawMessage
	}{
		"approved with an edit": {DecisionApprove, json.RawMessage(`{"path":"web","to":"n2"}`)},
		"approved":              {DecisionApprove, nil},
		"rejected":              {DecisionReject, nil},
	}
	for drift, tools := range drifts {
		for name, d := range decisions {
			t.Run(drift+"/"+name, func(t *testing.T) {
				session := &fakeSession{tools: offered()}
				agent, stored := suspendOnTheFirstOfTwoActions(t, session)
				stored.EditedArguments = d.edited
				session.tools = tools()
				var started []string
				result, err := agent.ResumeTurn(t.Context(), nil, stored, d.decision, func(event Event) error {
					if event.Type == EventToolStarted {
						started = append(started, event.ToolCallID)
					}
					return nil
				})
				if err == nil || !strings.Contains(err.Error(), `the confirmed tool "switch_object" is no longer offered`) || result.Suspended != nil {
					t.Fatalf("resume: %+v, %v", result, err)
				}
				if len(session.calls) != 0 || len(started) != 0 {
					t.Fatalf("calls %#v, started %v", session.calls, started)
				}
			})
		}
	}

	session := &fakeSession{tools: offered()}
	agent, stored := suspendOnTheFirstOfTwoActions(t, session)
	stored.EditedArguments = json.RawMessage(`{"path":"web","to":"n2"}`)
	result, err := agent.ResumeTurn(t.Context(), nil, stored, DecisionApprove, func(Event) error { return nil })
	if err != nil || result.Suspended == nil {
		t.Fatalf("resume: %+v, %v", result, err)
	}
	if call, _ := result.Suspended.PendingCall(); call.ID != "call-2" {
		t.Fatalf("suspended on %q", call.ID)
	}
	if len(session.calls) != 1 || session.calls[0].name != "switch_object" || session.calls[0].arguments["to"] != "n2" {
		t.Fatalf("calls %#v", session.calls)
	}
}

// TestADecisionAppliesToTheToolDefinitionShown resumes a turn whose
// confirmed tool the server changed since the confirmation: its description,
// its annotations or its input schema. The decision was taken on another
// definition: the turn fails and nothing runs. So does a turn suspended
// without the hash of the definition, by an earlier version of the agent.
func TestADecisionAppliesToTheToolDefinitionShown(t *testing.T) {
	changes := map[string]func(*mcp.Tool){
		"description": func(tool *mcp.Tool) { tool.Description = "Switch and also purge the object." },
		"annotations": func(tool *mcp.Tool) { tool.Annotations = &mcp.ToolAnnotations{ReadOnlyHint: true} },
		"schema": func(tool *mcp.Tool) {
			tool.InputSchema = map[string]any{"type": "object", "properties": map[string]any{"path": map[string]any{"type": "string"}}}
		},
	}
	for name, change := range changes {
		t.Run(name, func(t *testing.T) {
			session := &fakeSession{tools: []*mcp.Tool{editableTool(), destructiveTool("delete_object")}}
			agent, stored := suspendOnTheFirstOfTwoActions(t, session)
			if stored.ToolSHA256 == "" {
				t.Fatal("the suspended turn holds no hash of the confirmed tool")
			}
			changed := editableTool()
			change(changed)
			session.tools = []*mcp.Tool{changed, destructiveTool("delete_object")}
			_, err := agent.ResumeTurn(t.Context(), nil, stored, DecisionApprove, func(Event) error { return nil })
			if err == nil || !strings.Contains(err.Error(), `the confirmed tool "switch_object" changed since its confirmation`) || len(session.calls) != 0 {
				t.Fatalf("resume: %v, calls %#v", err, session.calls)
			}
		})
	}
	t.Run("suspended without the hash", func(t *testing.T) {
		session := &fakeSession{tools: []*mcp.Tool{editableTool(), destructiveTool("delete_object")}}
		agent, stored := suspendOnTheFirstOfTwoActions(t, session)
		stored.ToolSHA256 = ""
		if _, err := agent.ResumeTurn(t.Context(), nil, stored, DecisionApprove, func(Event) error { return nil }); err == nil || len(session.calls) != 0 {
			t.Fatalf("resume: %v, calls %#v", err, session.calls)
		}
	})
}

// TestAPreviewToolWithoutAnnotationsIsNotReadOnly names a preview tool the
// server lists without annotations: it is not called, and the turn shows the
// confirmation with the cause, rather than fail on a nil annotation.
func TestAPreviewToolWithoutAnnotationsIsNotReadOnly(t *testing.T) {
	update := destructiveTool("update_object_config")
	update.Meta = mcp.Meta{previewToolMeta: "preview_object_config_update"}
	tools := map[string]*mcp.Tool{
		"update_object_config":         update,
		"preview_object_config_update": {Name: "preview_object_config_update", InputSchema: objectSchema()},
	}
	session := &fakeSession{}
	got, cause := preview(t.Context(), session, tools, update, json.RawMessage(`{"path":"web"}`))
	if got != nil || !strings.Contains(cause, "not a read-only tool") || len(session.calls) != 0 {
		t.Fatalf("preview %s, cause %q, calls %#v", got, cause, session.calls)
	}
}
