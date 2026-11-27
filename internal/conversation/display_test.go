package conversation

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/llm"
)

func call(id, name, arguments string) llm.ToolCall {
	return llm.ToolCall{ID: id, Name: name, Arguments: json.RawMessage(arguments)}
}

func TestDisplayTurnTellsEachCall(t *testing.T) {
	long := strings.Repeat("é", 3000)
	messages := []llm.Message{
		{Role: llm.RoleUser, Text: "stop web"},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{call("c1", "get_object_status", `{"path":"web"}`), call("c2", "get_node_status", `{"node":"n9"}`)}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{
			{CallID: "c1", Content: json.RawMessage(`{"content":[],"structuredContent":{"note":"` + long + `"},"isError":false}`)},
			{CallID: "c2", Content: json.RawMessage(`{"content":[{"type":"text","text":"node n9 is not a node"}],"isError":true}`), IsError: true},
		}},
		{Role: llm.RoleAssistant, Text: "I will stop it.", ToolCalls: []llm.ToolCall{call("c3", "stop_instance", `{"path":"web","node":"n1"}`)}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "c3", Content: json.RawMessage(`{"content":[{"type":"text","text":"` + agent.RejectedToolResultText + `"},{"type":"text","text":"The confirmation of stop_object was answered: rejected."}]}`), IsError: true}}},
		{Role: llm.RoleAssistant, Text: "Not stopped."},
	}
	timing := TurnTiming{
		Runs:  []TimingRun{{StartedAt: 1000, EndedAt: 2500}, {StartedAt: 9000, EndedAt: 9500}},
		Calls: map[string]TimingCall{"c1": {StartedAt: 1100, EndedAt: 1400}, "c2": {StartedAt: 1110, EndedAt: 1500}, "c3": {StartedAt: 9000, EndedAt: 9010}},
		Usage: &TimingUsage{TotalTokens: 370},
	}
	steps, omitted, runs, usage := DisplayTurn(messages, timing)
	if len(steps) != 3 || omitted != 0 || len(runs) != 2 || usage == nil || usage.TotalTokens != 370 {
		t.Fatalf("steps %d, omitted %d, runs %+v, usage %+v", len(steps), omitted, runs, usage)
	}
	read, refused, rejected := steps[0], steps[1], steps[2]
	if read.Status != "completed" || *read.StartedAt != 1100 || *read.EndedAt != 1400 || string(read.Arguments) != `{"path":"web"}` {
		t.Fatalf("read = %+v", read)
	}
	// The excerpt is cut on a rune, under 4 KiB.
	if !read.ResultTruncated || len(read.ResultExcerpt) > 4096 || !strings.HasSuffix(read.ResultExcerpt, "é") {
		t.Fatalf("excerpt %d bytes, truncated %v", len(read.ResultExcerpt), read.ResultTruncated)
	}
	if refused.Status != "failed" || refused.Refusal != "node n9 is not a node" {
		t.Fatalf("refused = %+v", refused)
	}
	// A call the user rejected did not run: no end.
	if rejected.Status != "rejected" || rejected.EndedAt != nil || rejected.Refusal != "" || *rejected.StartedAt != 9000 {
		t.Fatalf("rejected = %+v", rejected)
	}
	if *runs[0].EndedAt != 2500 || runs[1].StartedAt != 9000 {
		t.Fatalf("runs = %+v", runs)
	}
}

func TestDisplayTurnWithoutTimesKeepsTheOrder(t *testing.T) {
	messages := []llm.Message{
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{call("b", "get_node_status", `{"node":"n1"}`), call("a", "get_object_status", `{}`)}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "a", Content: json.RawMessage(`{"content":[]}`)}}},
	}
	steps, _, runs, usage := DisplayTurn(messages, TurnTiming{})
	if len(steps) != 2 || steps[0].ID != "b" || steps[1].ID != "a" || steps[0].StartedAt != nil || runs != nil || usage != nil {
		t.Fatalf("steps %+v, runs %+v, usage %+v", steps, runs, usage)
	}
	// A call without result did not finish.
	if steps[0].Status != "interrupted" || steps[1].Status != "completed" {
		t.Fatalf("statuses = %s, %s", steps[0].Status, steps[1].Status)
	}
}

func TestDisplayTurnBoundsItsSteps(t *testing.T) {
	big := `{"config":"` + strings.Repeat("x", 15<<10) + `"}`
	var calls []llm.ToolCall
	var results []llm.ToolResult
	for index := range MaxDisplaySteps + 6 {
		id := "c" + strings.Repeat("0", 3) + string(rune('A'+index%26)) + strings.Repeat("z", index/26)
		calls = append(calls, call(id, "create_object", big))
		results = append(results, llm.ToolResult{CallID: id, Content: json.RawMessage(`{"content":[{"type":"text","text":"` + strings.Repeat("r", 20<<10) + `"}],"isError":true}`), IsError: true})
	}
	steps, omitted, _, _ := DisplayTurn([]llm.Message{{Role: llm.RoleAssistant, ToolCalls: calls}, {Role: llm.RoleTool, ToolResults: results}}, TurnTiming{})
	if len(steps) != MaxDisplaySteps || omitted != 6 {
		t.Fatalf("steps %d, omitted %d", len(steps), omitted)
	}
	if !steps[len(steps)-1].ArgumentsOmitted || string(steps[len(steps)-1].Arguments) != "null" || steps[0].ArgumentsOmitted {
		t.Fatal("the arguments past the bound of the turn are not omitted")
	}
	encoded, _ := json.Marshal(DisplayMessage{Steps: steps})
	if len(encoded) >= MaxMessagePageBytes-256 {
		t.Fatalf("a turn takes %d bytes, more than a page", len(encoded))
	}
}

// timingRunner suspends a turn after a read, then resumes it on the call
// the user approved with an edit, every event stamped by the agent.
type timingRunner struct {
	editableRunner
	// rejected: the user rejects the call, which the run does not start.
	rejected bool
}

var timingStart = time.UnixMilli(1_760_000_000_000)

func (r *timingRunner) RunTurn(_ context.Context, _ []llm.Message, _ string, emit agent.EmitFunc) (agent.TurnResult, error) {
	for _, event := range []agent.Event{
		{Type: agent.EventToolStarted, ToolName: "get_object_status", ToolCallID: "call-0", Iteration: 1, At: timingStart.Add(100 * time.Millisecond)},
		{Type: agent.EventToolFinished, ToolName: "get_object_status", ToolCallID: "call-0", Iteration: 1, At: timingStart.Add(400 * time.Millisecond)},
		{Type: agent.EventUsage, Usage: &llm.Usage{TotalTokens: 300}, Iteration: 1, At: timingStart.Add(500 * time.Millisecond)},
	} {
		if err := emit(event); err != nil {
			return agent.TurnResult{}, err
		}
	}
	return r.editableRunner.RunTurn(context.Background(), nil, "", emit)
}

func (r *timingRunner) ResumeTurn(ctx context.Context, history []llm.Message, state agent.SuspendedTurn, decision agent.Decision, emit agent.EmitFunc) (agent.TurnResult, error) {
	if r.rejected {
		return r.editableRunner.ResumeTurn(ctx, history, state, decision, emit)
	}
	for _, event := range []agent.Event{
		{Type: agent.EventToolStarted, ToolName: "stop_object", ToolCallID: "call-1", ToolArguments: state.EditedArguments, Iteration: 1, At: timingStart.Add(60 * time.Second)},
		{Type: agent.EventToolFinished, ToolName: "stop_object", ToolCallID: "call-1", Iteration: 1, At: timingStart.Add(61 * time.Second)},
		{Type: agent.EventUsage, Usage: &llm.Usage{TotalTokens: 70}, Iteration: 2, At: timingStart.Add(62 * time.Second)},
	} {
		if err := emit(event); err != nil {
			return agent.TurnResult{}, err
		}
	}
	return r.editableRunner.ResumeTurn(ctx, history, state, decision, emit)
}

func TestServiceRecordsTheTimesOfATurnAcrossItsRuns(t *testing.T) {
	store := newServiceTestStore()
	service := newTestService(t, store, &timingRunner{})
	execution, err := service.PrepareTurn(t.Context(), serviceTestIdentity(), store.item.ID, "stop web")
	if err != nil {
		t.Fatal(err)
	}
	var request *agent.Confirmation
	if err := execution.Run(t.Context(), func(event agent.Event) error { request = event.Confirmation; return nil }); err != nil {
		t.Fatal(err)
	}
	// Stored with the suspension: the run, the read, the tokens.
	suspended := store.suspended.Timing
	if len(suspended.Runs) != 1 || suspended.Runs[0].EndedAt == 0 || suspended.Calls["call-0"].StartedAt != 1_760_000_000_100 || suspended.Calls["call-0"].EndedAt != 1_760_000_000_400 ||
		suspended.Usage.TotalTokens != 300 {
		t.Fatalf("suspended timing = %+v", suspended)
	}
	// The decision comes 59 s later: its run starts then.
	service.now = func() time.Time { return timingStart.Add(59 * time.Second) }
	resumed, err := service.PrepareConfirmation(t.Context(), serviceTestIdentity(), store.item.ID, request.TurnID, request.ID, agent.DecisionApprove, json.RawMessage(`{"path":"db"}`))
	if err != nil {
		t.Fatal(err)
	}
	if err := resumed.Run(t.Context(), func(agent.Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	// Completed: the second run; the confirmed call from the start of its
	// run, as the stream shows it, not from the later start the agent emits;
	// with the arguments it ran with; and the tokens of both runs.
	completed := store.completedTiming
	approved := completed.Calls["call-1"]
	if len(completed.Runs) != 2 || completed.Runs[1].StartedAt != 1_760_000_059_000 || approved.StartedAt != 1_760_000_059_000 || approved.EndedAt != 1_760_000_061_000 ||
		string(approved.RanWith) != `{"path":"db"}` || completed.Usage.TotalTokens != 370 || completed.Calls["call-0"].EndedAt != 1_760_000_000_400 {
		t.Fatalf("completed timing = %+v", completed)
	}
}

func TestServiceStartsARejectedCallWithTheRunOfItsDecision(t *testing.T) {
	store := newServiceTestStore()
	service := newTestService(t, store, &timingRunner{rejected: true})
	execution, err := service.PrepareTurn(t.Context(), serviceTestIdentity(), store.item.ID, "stop web")
	if err != nil {
		t.Fatal(err)
	}
	var request *agent.Confirmation
	if err := execution.Run(t.Context(), func(event agent.Event) error { request = event.Confirmation; return nil }); err != nil {
		t.Fatal(err)
	}
	service.now = func() time.Time { return timingStart.Add(59 * time.Second) }
	resumed, err := service.PrepareConfirmation(t.Context(), serviceTestIdentity(), store.item.ID, request.TurnID, request.ID, agent.DecisionReject, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := resumed.Run(t.Context(), func(agent.Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	// The run emits nothing for the call it did not run: it starts with the
	// run, and has no end.
	if rejected := store.completedTiming.Calls["call-1"]; rejected.StartedAt != 1_760_000_059_000 || rejected.EndedAt != 0 {
		t.Fatalf("rejected = %+v", rejected)
	}
}

// A run keeps which skill the client asked, and the history names it on the
// prompt from the first run only: a skill the model loaded is not asked.
func TestRunsKeepTheSkillAsked(t *testing.T) {
	result := agent.TurnResult{Exposure: agent.Exposure{Skills: []agent.LoadedSkill{
		{Name: "node-maintenance", URI: "skill://node-maintenance/SKILL.md", Digest: "sha256:01", Asked: true},
		{Name: "morning-check", URI: "skill://morning-check/SKILL.md", Digest: "sha256:00"},
	}}}
	run := NewAgentInfo("v1").run(result, nil)
	if len(run.Skills) != 2 || !run.Skills[0].Asked || run.Skills[1].Asked {
		t.Fatalf("skills = %+v", run.Skills)
	}
	if got := AskedSkill([]TurnRun{run}); got != "node-maintenance" {
		t.Fatalf("asked = %q", got)
	}
	loaded := TurnRun{Skills: []RunSkill{{Name: "morning-check"}}}
	if got := AskedSkill([]TurnRun{loaded, run}); got != "" {
		t.Fatalf("asked by a later run = %q", got)
	}
	if got := AskedSkill(nil); got != "" {
		t.Fatalf("asked of no run = %q", got)
	}
}
