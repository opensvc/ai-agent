package agui

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/llm"
)

func types(events []Event) string {
	names := make([]string, 0, len(events))
	for _, event := range events {
		names = append(names, event.Type)
	}
	return strings.Join(names, ",")
}

func encodeAll(t *testing.T, encoder *Encoder, events ...agent.Event) []Event {
	t.Helper()
	out := encoder.Start()
	for _, event := range events {
		encoded, err := encoder.Encode(event)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, encoded...)
	}
	return out
}

func TestEncoderOrdersATurn(t *testing.T) {
	encoder := NewEncoder("c-1", "r-1")
	events := encodeAll(t, encoder,
		agent.Event{Type: agent.EventTextDelta, TextDelta: "Let me ", Iteration: 1},
		agent.Event{Type: agent.EventTextDelta, TextDelta: "look.", Iteration: 1},
		agent.Event{Type: agent.EventToolStarted, ToolName: "list_nodes", ToolCallID: "call-1", ToolArguments: json.RawMessage(`{"limit":5}`), Iteration: 1},
		agent.Event{Type: agent.EventToolFinished, ToolName: "list_nodes", ToolCallID: "call-1", ToolResult: json.RawMessage(`{"nodes":[]}`), Iteration: 1},
		agent.Event{Type: agent.EventUsage, Usage: &llm.Usage{InputTokens: 3, OutputTokens: 2, TotalTokens: 5}, Iteration: 1},
		// Refused calls never ran: the client sees none of them.
		agent.Event{Type: agent.EventToolCallsRefused, Refused: &agent.RefusedCalls{Bound: "answer", Asked: 9}, Iteration: 2},
		// A call the agent made of its own is for the audit only.
		agent.Event{Type: agent.EventAgentToolCalled, AgentCall: &agent.AgentCall{Tool: "get_cluster_status", Purpose: "documentation_version"}, Iteration: 2},
		agent.Event{Type: agent.EventTextDelta, TextDelta: "No node.", Iteration: 2},
		agent.Event{Type: agent.EventCompleted, TurnID: "t-1", Iteration: 2},
	)
	want := "RUN_STARTED,TEXT_MESSAGE_START,TEXT_MESSAGE_CONTENT,TEXT_MESSAGE_CONTENT,TEXT_MESSAGE_END,TOOL_CALL_START,TOOL_CALL_ARGS,TOOL_CALL_END,TOOL_CALL_RESULT,TEXT_MESSAGE_START,TEXT_MESSAGE_CONTENT,TEXT_MESSAGE_END,RUN_FINISHED"
	if got := types(events); got != want {
		t.Fatalf("events = %s", got)
	}
	start, args, result, second, finished := events[1], events[6], events[8], events[9], events[12]
	if start.MessageID != "r-1-1" || start.Role != "assistant" || events[5].ParentMessageID != "r-1-1" || events[5].ToolCallName != "list_nodes" ||
		args.Delta != `{"limit":5}` || *result.Content != `{"nodes":[]}` || result.ToolCallID != "call-1" || second.MessageID != "r-1-2" ||
		finished.Outcome.Type != "success" || finished.Result.(map[string]string)["turnId"] != "t-1" || finished.ThreadID != "c-1" || finished.RunID != "r-1" ||
		events[0].ProtocolVersion != "1.0" || len(finished.Usage) != 1 || finished.Usage[0] != (Usage{InputTokens: 3, OutputTokens: 2, TotalTokens: 5}) {
		t.Fatalf("events = %+v", events)
	}
	if _, err := encoder.Encode(agent.Event{Type: agent.EventTextDelta, TextDelta: "late", Iteration: 3}); err == nil {
		t.Fatal("an event after RUN_FINISHED was encoded")
	}
	if failure := encoder.Failure("agent_failed", "x"); len(failure) != 0 {
		t.Fatalf("failure after the end = %+v", failure)
	}
}

func TestEncoderInterruptsOnAConfirmation(t *testing.T) {
	encoder := NewEncoder("c-1", "r-1")
	expires := time.Date(2026, 10, 22, 10, 0, 0, 0, time.UTC)
	events := encodeAll(t, encoder,
		agent.Event{Type: agent.EventTextDelta, TextDelta: "Stopping.", Iteration: 1},
		agent.Event{Type: agent.EventConfirmationRequired, Iteration: 1, Confirmation: &agent.Confirmation{
			ID: "conf-1", TurnID: "t-1", ToolName: "stop_service", ToolTitle: "Stop service", Destructive: true, ToolCallID: "call-7",
			Arguments: json.RawMessage(`{"service":"web"}`), ArgumentTypes: map[string]string{"yaml": "application/yaml"}, ExpiresAt: expires,
		}},
	)
	if got := types(events); got != "RUN_STARTED,TEXT_MESSAGE_START,TEXT_MESSAGE_CONTENT,TEXT_MESSAGE_END,TOOL_CALL_START,TOOL_CALL_ARGS,TOOL_CALL_END,RUN_FINISHED" {
		t.Fatalf("events = %s", got)
	}
	finished := events[len(events)-1]
	interrupt := finished.Outcome.Interrupts[0]
	if finished.Outcome.Type != "interrupt" || interrupt.ID != "conf-1" || interrupt.Reason != "tool_call" || finished.Usage != nil || interrupt.ToolCallID != "call-7" ||
		interrupt.ExpiresAt != "2026-10-22T10:00:00Z" || interrupt.Metadata["turnId"] != "t-1" || interrupt.ResponseSchema["required"].([]string)[0] != "approved" ||
		events[5].Delta != `{"service":"web"}` || events[4].ToolCallID != "call-7" {
		t.Fatalf("finished = %+v", finished)
	}
	data, _ := json.Marshal(finished)
	if !strings.Contains(string(data), `"argumentTypes":{"yaml":"application/yaml"}`) {
		t.Fatalf("finished = %s", data)
	}
}

func TestEncoderResumesAConfirmedCall(t *testing.T) {
	encoder := NewEncoder("c-1", "r-2")
	encoder.Resume("call-7")
	events := encodeAll(t, encoder,
		agent.Event{Type: agent.EventToolStarted, ToolName: "stop_service", ToolCallID: "call-7", ToolArguments: json.RawMessage(`{}`), Iteration: 1},
		agent.Event{Type: agent.EventToolFinished, ToolName: "stop_service", ToolCallID: "call-7", ToolResult: json.RawMessage(`{"ok":true}`), Iteration: 1},
		agent.Event{Type: agent.EventToolStarted, ToolName: "get_service", ToolCallID: "call-8", ToolArguments: json.RawMessage(`{}`), Iteration: 2},
	)
	// The confirmed call reports its result only; the next call is whole.
	if got := types(events); got != "RUN_STARTED,TOOL_CALL_RESULT,TOOL_CALL_START,TOOL_CALL_ARGS,TOOL_CALL_END" {
		t.Fatalf("events = %s", got)
	}
	rejected := NewEncoder("c-1", "r-3").Rejected("call-7")
	if len(rejected) != 1 || rejected[0].Type != ToolCallResult || rejected[0].ToolCallID != "call-7" || !strings.Contains(*rejected[0].Content, "rejected") {
		t.Fatalf("rejected = %+v", rejected)
	}
}

func TestEncoderFailureClosesTheText(t *testing.T) {
	encoder := NewEncoder("c-1", "r-1")
	events := encodeAll(t, encoder, agent.Event{Type: agent.EventTextDelta, TextDelta: "partial", Iteration: 1})
	events = append(events, encoder.Failure("request_timeout", "the agent request timed out")...)
	if got := types(events); got != "RUN_STARTED,TEXT_MESSAGE_START,TEXT_MESSAGE_CONTENT,TEXT_MESSAGE_END,RUN_ERROR" {
		t.Fatalf("events = %s", got)
	}
	if last := events[len(events)-1]; last.Code != "request_timeout" || last.Message == "" {
		t.Fatalf("error = %+v", last)
	}
}

func TestToolResultIsBounded(t *testing.T) {
	large := json.RawMessage(`"` + strings.Repeat("a", maxToolResultBytes) + `"`)
	if got := toolResult(large); !strings.Contains(got, `"omitted":true`) {
		t.Fatalf("large result = %.80s", got)
	}
	if got := toolResult(nil); got != "{}" {
		t.Fatalf("empty result = %s", got)
	}
}

func TestResumeEntryPreviewRequest(t *testing.T) {
	for _, test := range []struct {
		payload string
		ok      bool
		edited  string
		err     bool
	}{
		{payload: `{"action":"preview","editedArgs":{"path":"api"}}`, ok: true, edited: `{"path":"api"}`},
		{payload: `{"approved":true}`},
		{payload: `{"action":"preview"}`, ok: true, err: true},
		{payload: `{"action":"preview","editedArgs":[1]}`, ok: true, err: true},
		{payload: `{"action":"run","editedArgs":{}}`, ok: true, err: true},
		{payload: `{"action":"preview","editedArgs":{},"approved":true}`, ok: true, err: true},
	} {
		edited, ok, err := ResumeEntry{Status: ResumeResolved, Payload: json.RawMessage(test.payload)}.PreviewRequest()
		if ok != test.ok || (err != nil) != test.err || (!test.err && string(edited) != test.edited) {
			t.Errorf("%s: %s %v %v", test.payload, edited, ok, err)
		}
	}
	if _, ok, _ := (ResumeEntry{Status: ResumeCancelled, Payload: json.RawMessage(`{"action":"preview"}`)}).PreviewRequest(); ok {
		t.Error("a cancelled resume is a preview")
	}
}

func TestRunAgentInputPromptAndApproval(t *testing.T) {
	for raw, want := range map[string]string{
		`{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","value":"https://x"}},{"type":"text","text":"what is it?"}]}]}`: "what is it?",
		`{"messages":[{"role":"assistant","content":"hi"},{"role":"user","content":"list nodes"}]}`:                                                    "list nodes",
		`{"messages":[{"role":"user","content":[{"type":"text","text":"list"},{"type":"text","text":"nodes"}]}]}`:                                      "list\nnodes",
		`{"messages":[{"role":"user","content":"a"},{"role":"tool","toolCallId":"x","content":"{}"}]}`:                                                 "",
		`{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","value":"https://x"}}]}]}`:                                      "",
		`{"messages":[]}`: "",
	} {
		var input RunAgentInput
		if err := json.Unmarshal([]byte(raw), &input); err != nil {
			t.Fatal(err)
		}
		prompt, skipped, err := input.Prompt()
		if (want == "") != (err != nil) || prompt != want || (strings.Contains(raw, "image") && (len(skipped) != 1 || skipped[0] != "image")) {
			t.Errorf("%s: prompt %q, %v", raw, prompt, err)
		}
	}
	for raw, want := range map[string]string{
		`{"interruptId":"c","status":"resolved","payload":{"approved":true}}`:                       "true",
		`{"interruptId":"c","status":"resolved","payload":{"approved":false}}`:                      "false",
		`{"interruptId":"c","status":"cancelled"}`:                                                  "false",
		`{"interruptId":"c","status":"resolved","payload":{"approved":true,"editedArgs":{"a":1}}}`:  "true edited",
		`{"interruptId":"c","status":"resolved","payload":{"approved":false,"editedArgs":{"a":1}}}`: "error",
		`{"interruptId":"c","status":"cancelled","payload":{"editedArgs":{"a":1}}}`:                 "error",
		`{"interruptId":"c","status":"resolved","payload":{"approved":true,"editedArgs":[1]}}`:      "error",
		`{"interruptId":"c","status":"resolved","payload":{"approved":true,"note":"x"}}`:            "error",
		`{"interruptId":"c","status":"resolved","payload":{}}`:                                      "error",
		`{"interruptId":"c","status":"later"}`:                                                      "error",
	} {
		var entry ResumeEntry
		if err := json.Unmarshal([]byte(raw), &entry); err != nil {
			t.Fatal(err)
		}
		approved, edited, err := entry.Approval()
		got := map[bool]string{true: "true", false: "false"}[approved]
		if edited != nil {
			got += " edited"
		}
		if err != nil {
			got = "error"
		}
		if got != want {
			t.Errorf("%s: %s", raw, got)
		}
	}
}

// A run that calls tools of the client is complete: it ends in success with
// the calls left for the client, which it never answers.
func TestEncoderLeavesTheCallsOfTheClientPending(t *testing.T) {
	encoder := NewEncoder("c-1", "r-1")
	events := encodeAll(t, encoder,
		agent.Event{Type: agent.EventToolStarted, ToolName: "open_page", ToolCallID: "ui-1", ToolArguments: json.RawMessage(`{"path":"/web"}`), Iteration: 1},
		agent.Event{Type: agent.EventToolStarted, ToolName: "highlight", ToolCallID: "ui-2", ToolArguments: json.RawMessage(`{}`), Iteration: 1},
		agent.Event{Type: agent.EventClientToolsPending, PendingCalls: []string{"ui-1", "ui-2"}, Iteration: 1},
	)
	if got := types(events); got != "RUN_STARTED,TOOL_CALL_START,TOOL_CALL_ARGS,TOOL_CALL_END,TOOL_CALL_START,TOOL_CALL_ARGS,TOOL_CALL_END,RUN_FINISHED" {
		t.Fatalf("events = %s", got)
	}
	finished := events[len(events)-1]
	if finished.Outcome.Type != "success" || strings.Join(finished.Outcome.PendingToolCallIDs, ",") != "ui-1,ui-2" || finished.Result != nil {
		t.Fatalf("finished = %+v", finished)
	}
}

func TestRunAgentInputToolAnswers(t *testing.T) {
	var input RunAgentInput
	if err := json.Unmarshal([]byte(`{"messages":[{"role":"user","content":"show"},{"role":"assistant","content":""},
		{"role":"tool","toolCallId":"mcp-1","content":"{}"},
		{"role":"tool","toolCallId":"ui-1","content":[{"type":"text","text":"opened"},{"type":"image","source":{"type":"url","value":"https://x"}}]},
		{"role":"tool","toolCallId":"ui-2","content":"","error":"the page does not exist"}]}`), &input); err != nil {
		t.Fatal(err)
	}
	answers, skipped, err := input.ToolAnswers()
	if err != nil || len(answers) != 3 || answers[1] != (ToolAnswer{CallID: "ui-1", Text: "opened"}) ||
		answers[2] != (ToolAnswer{CallID: "ui-2", Text: "the page does not exist", IsError: true}) || len(skipped) != 1 || skipped[0] != "image" {
		t.Fatalf("answers = %+v, %v, %v", answers, skipped, err)
	}
	// An input ending with a message of the user has no answers.
	input.Messages = append(input.Messages, Message{Role: "user", Content: json.RawMessage(`"next"`)})
	if answers, _, err := input.ToolAnswers(); err != nil || len(answers) != 0 {
		t.Fatalf("answers = %+v, %v", answers, err)
	}
	if err := json.Unmarshal([]byte(`{"messages":[{"role":"tool","content":"x"}]}`), &input); err != nil {
		t.Fatal(err)
	}
	if _, _, err := input.ToolAnswers(); err == nil {
		t.Fatal("a tool message without toolCallId was accepted")
	}
}

// The messages of the stream are named as the conversation stores them.
func TestEncoderNamesTheMessagesAsStored(t *testing.T) {
	encoder := NewEncoder("c-1", "r-1")
	encoder.SetTurn("t-1")
	events := encodeAll(t, encoder,
		agent.Event{Type: agent.EventTextDelta, TextDelta: "Looking.", Iteration: 1},
		agent.Event{Type: agent.EventToolStarted, ToolName: "list_nodes", ToolCallID: "call-1", ToolArguments: json.RawMessage(`{}`), Iteration: 1},
		agent.Event{Type: agent.EventTextDelta, TextDelta: "Done.", Iteration: 2},
	)
	if events[1].MessageID != "t-1:2" || events[4].ParentMessageID != "t-1:2" || events[8].MessageID != "t-1:4" {
		t.Fatalf("events = %+v", events)
	}
}

func TestEncoderSnapshotsTheThread(t *testing.T) {
	history := []llm.Message{
		{Role: llm.RoleUser, Text: "nodes?"},
		{Role: llm.RoleAssistant, Text: "Looking.", ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "list_nodes", Arguments: json.RawMessage(`{"limit":5}`)}, {ID: "call-2", Name: "count_nodes"}}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call-1", Content: json.RawMessage(`{"nodes":[]}`)}, {CallID: "call-2", Content: json.RawMessage(`{"count":0}`)}}},
		{Role: llm.RoleAssistant, Text: "No node."},
	}
	ids := []string{"t-1:1", "t-1:2", "t-1:3", "t-1:4"}
	prompt := "and services?"
	events, err := NewEncoder("c-1", "r-2").Snapshot(history, ids, &SnapshotMessage{ID: "client-7", Role: "user", Content: &prompt}, 0)
	if err != nil {
		t.Fatal(err)
	}
	thread := *events[0].Messages
	shape := make([]string, 0, len(thread))
	for _, message := range thread {
		shape = append(shape, message.ID+"/"+message.Role)
	}
	if events[0].Type != MessagesSnapshot || strings.Join(shape, ",") != "t-1:1/user,t-1:2/assistant,call-1-result/tool,call-2-result/tool,t-1:4/assistant,client-7/user" ||
		thread[1].ToolCalls[1].Function.Arguments != "{}" || thread[1].ToolCalls[0].Function.Arguments != `{"limit":5}` || *thread[2].Content != `{"nodes":[]}` ||
		thread[2].ToolCallID != "call-1" || events[0].Metadata != nil {
		t.Fatalf("snapshot = %+v", thread)
	}
	// Beyond 1 MiB the oldest turns are left out, and counted.
	large := strings.Repeat("a", maxSnapshotBytes)
	events, err = NewEncoder("c-1", "r-2").Snapshot([]llm.Message{{Role: llm.RoleUser, Text: large}, {Role: llm.RoleAssistant, Text: "ok"}, {Role: llm.RoleUser, Text: "next"}, {Role: llm.RoleAssistant, Text: "fine"}},
		[]string{"t-0:1", "t-0:2", "t-1:1", "t-1:2"}, nil, 2)
	if err != nil || len(*events[0].Messages) != 2 || (*events[0].Messages)[0].ID != "t-1:1" || events[0].Metadata["opensvc.truncated"].(map[string]int)["omittedTurns"] != 3 {
		t.Fatalf("truncated snapshot = %+v, %v", events, err)
	}
}

func TestRunAgentInputVersion(t *testing.T) {
	for version, want := range map[string]string{"": "served", "1.0": "served", "1.10": "newer", "1.1": "newer", "2.0": "unsupported", "0.9": "unsupported", "1": "invalid", "1.x": "invalid", "v1.0": "invalid", "1.0.1": "invalid"} {
		newer, err := RunAgentInput{ProtocolVersion: version}.Version()
		got := map[bool]string{false: "served", true: "newer"}[newer]
		switch {
		case errors.Is(err, ErrInvalidVersion):
			got = "invalid"
		case errors.Is(err, ErrUnsupportedVersion):
			got = "unsupported"
		}
		if got != want {
			t.Errorf("%q: %s", version, got)
		}
	}
}

func TestEncoderStampsTheTimeOfTheAgent(t *testing.T) {
	encoder := NewEncoder("c-1", "r-1")
	encoder.now = func() time.Time { return time.UnixMilli(5) }
	at := time.UnixMilli(1_760_000_000_400)
	events, err := encoder.Encode(agent.Event{Type: agent.EventToolStarted, ToolName: "list_nodes", ToolCallID: "call-1", ToolArguments: json.RawMessage(`{}`), Iteration: 1, At: at})
	if err != nil || len(events) != 3 {
		t.Fatalf("events = %+v, %v", events, err)
	}
	for _, event := range events {
		if event.Timestamp != 1_760_000_000_400 {
			t.Fatalf("%s stamped %d, not the time of the agent", event.Type, event.Timestamp)
		}
	}
	// An event the agent did not stamp takes the time of the encoder.
	unstamped, _ := encoder.Encode(agent.Event{Type: agent.EventToolFinished, ToolName: "list_nodes", ToolCallID: "call-1", Iteration: 1})
	if unstamped[0].Timestamp != 5 {
		t.Fatalf("unstamped = %d", unstamped[0].Timestamp)
	}
}
