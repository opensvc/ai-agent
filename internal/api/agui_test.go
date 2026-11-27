package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/agui"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/conversation"
	"github.com/opensvc/ai-agent/internal/llm"
)

// aguiEvents decodes the data lines of an AG-UI stream.
func aguiEvents(t *testing.T, body string) []map[string]any {
	t.Helper()
	var events []map[string]any
	for _, frame := range strings.Split(strings.TrimSpace(body), "\n\n") {
		if strings.HasPrefix(frame, ":") {
			continue
		}
		data, ok := strings.CutPrefix(frame, "data: ")
		if !ok || strings.Contains(frame, "event:") {
			t.Fatalf("frame is not an AG-UI data line: %q", frame)
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			t.Fatalf("decode %q: %v", data, err)
		}
		events = append(events, event)
	}
	return events
}

func eventTypes(events []map[string]any) string {
	names := make([]string, 0, len(events))
	for _, event := range events {
		names = append(names, event["type"].(string))
	}
	return strings.Join(names, ",")
}

func runRequest(t *testing.T, service ConversationService, body string) *httptest.ResponseRecorder {
	t.Helper()
	request := authenticatedRequest(http.MethodPost, "/v1/om3/conversations/conversation-id/runs", body)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "text/event-stream")
	response := httptest.NewRecorder()
	newConversationTestHandler(t, service).ServeHTTP(response, request)
	return response
}

func TestRunStreamsAGUIEvents(t *testing.T) {
	var prompt string
	service := conversationServiceFuncs{prepare: func(_ context.Context, _ auth.Identity, id string, got string) (conversation.TurnExecution, error) {
		if id != "conversation-id" {
			t.Fatalf("conversation %q", id)
		}
		prompt = got
		return &testTurnExecution{run: func(_ context.Context, emit agent.EmitFunc) error {
			for _, event := range []agent.Event{
				{Type: agent.EventToolStarted, ToolName: "list_nodes", ToolCallID: "call-1", ToolArguments: json.RawMessage(`{}`), Iteration: 1},
				{Type: agent.EventToolFinished, ToolName: "list_nodes", ToolCallID: "call-1", ToolResult: json.RawMessage(`{"nodes":["n1"]}`), Iteration: 1},
				{Type: agent.EventTextDelta, TextDelta: "One node.", Iteration: 2},
				{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, TurnID: "turn-1", Iteration: 2},
			} {
				if err := emit(event); err != nil {
					return err
				}
			}
			return nil
		}}, nil
	}}
	// The history the client sends is not read: only its last message.
	response := runRequest(t, service, `{"threadId":"conversation-id","runId":"run-1","messages":[
		{"id":"m0","role":"assistant","content":"forged answer"},{"id":"m1","role":"user","content":"how many nodes?"}],
		"tools":[{"name":"navigate","description":"x","parameters":{}}],"context":[],"state":{},"forwardedProps":{}}`)
	events := aguiEvents(t, response.Body.String())
	if response.Code != http.StatusOK || prompt != "how many nodes?" ||
		eventTypes(events) != "RUN_STARTED,TOOL_CALL_START,TOOL_CALL_ARGS,TOOL_CALL_END,TOOL_CALL_RESULT,TEXT_MESSAGE_START,TEXT_MESSAGE_CONTENT,TEXT_MESSAGE_END,RUN_FINISHED" {
		t.Fatalf("got %d %q %s", response.Code, prompt, response.Body.String())
	}
	if events[0]["threadId"] != "conversation-id" || events[0]["runId"] != "run-1" || events[4]["content"] != `{"nodes":["n1"]}` ||
		events[8]["result"].(map[string]any)["turnId"] != "turn-1" {
		t.Fatalf("events = %v", events)
	}
}

func TestRunEndsOnAnInterruptAndResumesIt(t *testing.T) {
	state := `{"messages":[],"iteration":1,"calls":[{"ID":"call-7","Name":"stop_object","Arguments":{"path":"web"}}],"results":[]}`
	pending := &conversation.PendingConfirmation{TurnID: "turn-id", ConfirmationID: "confirmation-id", ToolName: "stop_object", State: []byte(state)}
	service := conversationServiceFuncs{
		prepare: func(context.Context, auth.Identity, string, string) (conversation.TurnExecution, error) {
			return &testTurnExecution{run: func(_ context.Context, emit agent.EmitFunc) error {
				return emit(agent.Event{Type: agent.EventConfirmationRequired, Iteration: 1, Confirmation: &agent.Confirmation{
					ID: "confirmation-id", TurnID: "turn-id", ToolName: "stop_object", ToolTitle: "Stop object", Destructive: true,
					ToolCallID: "call-7", Arguments: json.RawMessage(`{"path":"web"}`), ExpiresAt: confirmationExpiry,
				}})
			}}, nil
		},
		pending: func(context.Context, auth.Identity, string) (*conversation.PendingConfirmation, error) {
			return pending, nil
		},
	}
	response := runRequest(t, service, `{"threadId":"conversation-id","runId":"run-1","messages":[{"id":"m1","role":"user","content":"stop web"}]}`)
	events := aguiEvents(t, response.Body.String())
	finished := events[len(events)-1]
	interrupt := finished["outcome"].(map[string]any)["interrupts"].([]any)[0].(map[string]any)
	if eventTypes(events) != "RUN_STARTED,TOOL_CALL_START,TOOL_CALL_ARGS,TOOL_CALL_END,RUN_FINISHED" || interrupt["id"] != "confirmation-id" || interrupt["toolCallId"] != "call-7" {
		t.Fatalf("got %s", response.Body.String())
	}

	for _, test := range []struct {
		resume   string
		decision agent.Decision
		want     string
	}{
		{`{"interruptId":"confirmation-id","status":"resolved","payload":{"approved":true}}`, agent.DecisionApprove, "RUN_STARTED,TOOL_CALL_RESULT,RUN_FINISHED"},
		{`{"interruptId":"confirmation-id","status":"resolved","payload":{"approved":false}}`, agent.DecisionReject, "RUN_STARTED,TOOL_CALL_RESULT,RUN_FINISHED"},
		{`{"interruptId":"confirmation-id","status":"cancelled"}`, agent.DecisionReject, "RUN_STARTED,TOOL_CALL_RESULT,RUN_FINISHED"},
	} {
		service.confirm = func(_ context.Context, _ auth.Identity, id, turnID, confirmationID string, decision agent.Decision) (conversation.TurnExecution, error) {
			if id != "conversation-id" || turnID != "turn-id" || confirmationID != "confirmation-id" || decision != test.decision {
				t.Fatalf("%s: confirm %q %q %q %q", test.resume, id, turnID, confirmationID, decision)
			}
			return &testTurnExecution{run: func(_ context.Context, emit agent.EmitFunc) error {
				if decision == agent.DecisionApprove {
					// The confirmed call runs: its start is not given again.
					_ = emit(agent.Event{Type: agent.EventToolStarted, ToolName: "stop_object", ToolCallID: "call-7", ToolArguments: json.RawMessage(`{}`), Iteration: 1})
					_ = emit(agent.Event{Type: agent.EventToolFinished, ToolName: "stop_object", ToolCallID: "call-7", ToolResult: json.RawMessage(`{}`), Iteration: 1})
				}
				return emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, TurnID: "turn-id", Iteration: 2})
			}}, nil
		}
		response := runRequest(t, service, `{"threadId":"conversation-id","runId":"run-2","messages":[],"resume":[`+test.resume+`]}`)
		events := aguiEvents(t, response.Body.String())
		if response.Code != http.StatusOK || eventTypes(events) != test.want || events[1]["toolCallId"] != "call-7" {
			t.Fatalf("%s: got %d %s", test.resume, response.Code, response.Body.String())
		}
	}
}

func TestRunRefusesInvalidInputs(t *testing.T) {
	service := conversationServiceFuncs{
		prepare: func(context.Context, auth.Identity, string, string) (conversation.TurnExecution, error) {
			t.Fatal("an invalid run reached the agent")
			return nil, nil
		},
		pending: func(context.Context, auth.Identity, string) (*conversation.PendingConfirmation, error) {
			return &conversation.PendingConfirmation{TurnID: "turn-id", ConfirmationID: "confirmation-id", State: []byte(`{"calls":[{"ID":"c"}]}`)}, nil
		},
		confirm: func(context.Context, auth.Identity, string, string, string, agent.Decision) (conversation.TurnExecution, error) {
			t.Fatal("an invalid resume reached the agent")
			return nil, nil
		},
	}
	for body, want := range map[string]string{
		`{"threadId":"other","runId":"r","messages":[{"role":"user","content":"x"}]}`:                                                                                            "thread_mismatch",
		`{"threadId":"conversation-id","runId":"","messages":[{"role":"user","content":"x"}]}`:                                                                                   "invalid_run_id",
		`{"threadId":"conversation-id","runId":"r 1","messages":[{"role":"user","content":"x"}]}`:                                                                                "invalid_run_id",
		`{"threadId":"conversation-id","runId":"r","messages":[{"role":"assistant","content":"x"}]}`:                                                                             "invalid_prompt",
		`{"threadId":"conversation-id","runId":"r","messages":[{"role":"user","content":"  "}]}`:                                                                                 "invalid_prompt",
		`{"threadId":"conversation-id","runId":"r","messages":[],"resume":[{"interruptId":"other","status":"resolved","payload":{"approved":true}}]}`:                            "confirmation_not_pending",
		`{"threadId":"conversation-id","runId":"r","messages":[],"resume":[{"interruptId":"confirmation-id","status":"resolved","payload":{"approved":false,"editedArgs":{}}}]}`: "invalid_decision",
		`{"threadId":"conversation-id","runId":"r","messages":[],"resume":[{"interruptId":"a","status":"cancelled"},{"interruptId":"b","status":"cancelled"}]}`:                  "invalid_decision",
		`not json`: "invalid_request",
	} {
		response := runRequest(t, service, body)
		if response.Code < 400 || !strings.Contains(response.Body.String(), `"code":"`+want+`"`) {
			t.Errorf("%s: got %d %s", body, response.Code, response.Body.String())
		}
	}
}

func TestRunErrorEndsTheRun(t *testing.T) {
	service := conversationServiceFuncs{prepare: func(context.Context, auth.Identity, string, string) (conversation.TurnExecution, error) {
		return &testTurnExecution{run: func(_ context.Context, emit agent.EmitFunc) error {
			_ = emit(agent.Event{Type: agent.EventTextDelta, TextDelta: "partial", Iteration: 1})
			return errors.New("model failed")
		}}, nil
	}}
	response := runRequest(t, service, `{"threadId":"conversation-id","runId":"run-1","messages":[{"role":"user","content":"x"}]}`)
	events := aguiEvents(t, response.Body.String())
	if eventTypes(events) != "RUN_STARTED,TEXT_MESSAGE_START,TEXT_MESSAGE_CONTENT,TEXT_MESSAGE_END,RUN_ERROR" || events[4]["code"] != "agent_failed" {
		t.Fatalf("got %s", response.Body.String())
	}
}

func TestRunOffersTheToolsOfTheClientAndLeavesTheirCallsPending(t *testing.T) {
	var offered []agent.ClientTool
	abandoned := false
	service := conversationServiceFuncs{
		abandon: func(context.Context, auth.Identity, string) error { abandoned = true; return nil },
		prepare: func(context.Context, auth.Identity, string, string) (conversation.TurnExecution, error) {
			return &testTurnExecution{run: func(ctx context.Context, emit agent.EmitFunc) error {
				offered = agent.ClientToolsForTest(ctx)
				_ = emit(agent.Event{Type: agent.EventToolStarted, ToolName: "open_page", ToolCallID: "ui-1", ToolArguments: json.RawMessage(`{"path":"/web"}`), Iteration: 1})
				return emit(agent.Event{Type: agent.EventClientToolsPending, PendingCalls: []string{"ui-1"}, Iteration: 1})
			}}, nil
		},
	}
	response := runRequest(t, service, `{"threadId":"conversation-id","runId":"run-1","messages":[{"role":"user","content":"show web"}],
		"tools":[{"name":"open_page","description":"Open a page","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}]}`)
	events := aguiEvents(t, response.Body.String())
	finished := events[len(events)-1]
	outcome := finished["outcome"].(map[string]any)
	if eventTypes(events) != "RUN_STARTED,TOOL_CALL_START,TOOL_CALL_ARGS,TOOL_CALL_END,RUN_FINISHED" || outcome["type"] != "success" ||
		outcome["pendingToolCallIds"].([]any)[0] != "ui-1" || len(offered) != 1 || offered[0].Name != "open_page" || !abandoned {
		t.Fatalf("got %s, offered %v, abandoned %v", response.Body.String(), offered, abandoned)
	}
}

func TestRunRefusesInvalidToolsOfTheClient(t *testing.T) {
	service := conversationServiceFuncs{prepare: func(context.Context, auth.Identity, string, string) (conversation.TurnExecution, error) {
		t.Fatal("invalid tools reached the agent")
		return nil, nil
	}}
	many := make([]string, 0, agent.MaxClientTools+1)
	for index := range agent.MaxClientTools + 1 {
		many = append(many, `{"name":"tool_`+string(rune('a'+index))+`","description":"x"}`)
	}
	for name, tools := range map[string]string{
		"too many":    "[" + strings.Join(many, ",") + "]",
		"duplicated":  `[{"name":"open_page","description":"x"},{"name":"open_page","description":"y"}]`,
		"agent tool":  `[{"name":"search_tools","description":"x"}]`,
		"bad name":    `[{"name":"open page","description":"x"}]`,
		"long schema": `[{"name":"open_page","description":"x","parameters":{"type":"object","description":"` + strings.Repeat("a", 65<<10) + `"}}]`,
	} {
		response := runRequest(t, service, `{"threadId":"conversation-id","runId":"run-1","messages":[{"role":"user","content":"x"}],"tools":`+tools+`}`)
		if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), `"code":"invalid_tools"`) {
			t.Errorf("%s: got %d %s", name, response.Code, response.Body.String())
		}
	}
}

// The context of the interface goes with a new prompt to its turn.
func TestRunGivesTheContextOfTheInterfaceToTheTurn(t *testing.T) {
	var given []agent.InterfaceEntry
	service := conversationServiceFuncs{prepare: func(context.Context, auth.Identity, string, string) (conversation.TurnExecution, error) {
		return &testTurnExecution{run: func(ctx context.Context, emit agent.EmitFunc) error {
			given = agent.InterfaceContextFrom(ctx)
			return emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, TurnID: "turn-1", Iteration: 1})
		}}, nil
	}}
	response := runRequest(t, service, `{"threadId":"conversation-id","runId":"run-1","messages":[{"id":"m1","role":"user","content":"why is it down?"}],
		"context":[{"description":"page","value":"/objects"},{"description":"selected object","value":"test/svc/mcp-store","extra":1}]}`)
	if response.Code != http.StatusOK || len(given) != 2 || given[1] != (agent.InterfaceEntry{Description: "selected object", Value: "test/svc/mcp-store"}) {
		t.Fatalf("got %d %#v %s", response.Code, given, response.Body.String())
	}
}

// The audit counts the context of the interface, never its values.
func TestRunAuditsTheSizeOfTheContextOnly(t *testing.T) {
	service := conversationServiceFuncs{prepare: func(context.Context, auth.Identity, string, string) (conversation.TurnExecution, error) {
		return &testTurnExecution{run: func(_ context.Context, emit agent.EmitFunc) error {
			return emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, TurnID: "turn-1", Iteration: 1})
		}}, nil
	}}
	var output bytes.Buffer
	handler, err := NewHandler([]Backend{{Name: "om3", Conversations: service, Authenticator: allowTestTokenVerifier()}},
		HandlerConfig{Models: testModels(), MaxConcurrentAsks: 4, AuditLogger: slog.New(slog.NewJSONHandler(&output, nil))})
	if err != nil {
		t.Fatal(err)
	}
	request := authenticatedRequest(http.MethodPost, "/v1/om3/conversations/conversation-id/runs", `{"threadId":"conversation-id","runId":"run-1",
		"messages":[{"id":"m1","role":"user","content":"why is it down?"}],
		"context":[{"description":"page","value":"/objects"},{"description":"selected object","value":"test/svc/mcp-store"}]}`)
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("got %d %s", response.Code, response.Body.String())
	}
	started := auditLine(t, output.String(), "conversation_run_started")
	if !strings.Contains(started, `"context_entries":2`) || !strings.Contains(started, `"context_bytes":45`) || strings.Contains(output.String(), "mcp-store") {
		t.Fatalf("audit = %s", output.String())
	}
}

// auditLine gives the line of the audit of event.
func auditLine(t *testing.T, audit, event string) string {
	t.Helper()
	for _, line := range strings.Split(audit, "\n") {
		if strings.Contains(line, `"event":"`+event+`"`) || strings.Contains(line, `"msg":"`+event+`"`) {
			return line
		}
	}
	t.Fatalf("no %s in %s", event, audit)
	return ""
}

func TestRunRefusesAnInvalidContext(t *testing.T) {
	service := conversationServiceFuncs{prepare: func(context.Context, auth.Identity, string, string) (conversation.TurnExecution, error) {
		t.Fatal("an invalid context reached the agent")
		return nil, nil
	}}
	for name, context := range map[string]string{
		"not an array": `{"page":"/objects"}`,
		"no value":     `[{"description":"page"}]`,
		"not a string": `[{"description":"page","value":1}]`,
		"control":      `[{"description":"page","value":"a\nb"}]`,
		"too long":     `[{"description":"page","value":"` + strings.Repeat("v", 3<<10) + `"}]`,
	} {
		response := runRequest(t, service, `{"threadId":"conversation-id","runId":"run-1","messages":[{"role":"user","content":"x"}],"context":`+context+`}`)
		if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), `"code":"invalid_context"`) {
			t.Errorf("%s: got %d %s", name, response.Code, response.Body.String())
		}
	}
}

func TestRunResumesWithTheAnswersOfTheClient(t *testing.T) {
	var answered []llm.ToolResult
	service := conversationServiceFuncs{
		clientCalls: func(context.Context, auth.Identity, string) ([]string, error) { return []string{"ui-1"}, nil },
		answer: func(_ context.Context, _ auth.Identity, _ string, results []llm.ToolResult) (conversation.TurnExecution, error) {
			answered = results
			return &testTurnExecution{run: func(_ context.Context, emit agent.EmitFunc) error {
				_ = emit(agent.Event{Type: agent.EventTextDelta, TextDelta: "Shown.", Iteration: 2})
				return emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, TurnID: "turn-1", Iteration: 2})
			}}, nil
		},
		prepare: func(context.Context, auth.Identity, string, string) (conversation.TurnExecution, error) {
			t.Fatal("answers ran as a prompt")
			return nil, nil
		},
	}
	// The result the agent gave in the stream, kept by the client, is not
	// an answer: only the call waiting is answered.
	response := runRequest(t, service, `{"threadId":"conversation-id","runId":"run-2","messages":[{"role":"user","content":"show web"},
		{"role":"assistant","toolCalls":[]},{"role":"tool","toolCallId":"mcp-1","content":"{}"},{"role":"tool","toolCallId":"ui-1","content":"opened"}]}`)
	events := aguiEvents(t, response.Body.String())
	if response.Code != http.StatusOK || eventTypes(events) != "RUN_STARTED,TEXT_MESSAGE_START,TEXT_MESSAGE_CONTENT,TEXT_MESSAGE_END,RUN_FINISHED" ||
		len(answered) != 1 || answered[0].CallID != "ui-1" || !strings.Contains(string(answered[0].Content), "opened") {
		t.Fatalf("got %d %s, answered %+v", response.Code, response.Body.String(), answered)
	}
	// Answers to a conversation that waits on none.
	service.clientCalls = func(context.Context, auth.Identity, string) ([]string, error) { return nil, nil }
	response = runRequest(t, service, `{"threadId":"conversation-id","runId":"run-3","messages":[{"role":"tool","toolCallId":"ui-1","content":"opened"}]}`)
	if response.Code != http.StatusConflict || !strings.Contains(response.Body.String(), `"code":"client_tools_not_pending"`) {
		t.Fatalf("got %d %s", response.Code, response.Body.String())
	}
	// Answers to a wait that expired.
	service.answer = func(context.Context, auth.Identity, string, []llm.ToolResult) (conversation.TurnExecution, error) {
		return nil, conversation.ErrConfirmationExpired
	}
	service.clientCalls = func(context.Context, auth.Identity, string) ([]string, error) { return []string{"ui-1"}, nil }
	response = runRequest(t, service, `{"threadId":"conversation-id","runId":"run-5","messages":[{"role":"tool","toolCallId":"ui-1","content":"opened"}]}`)
	if response.Code != http.StatusGone || !strings.Contains(response.Body.String(), `"code":"client_tools_expired"`) || strings.Contains(response.Body.String(), "action") {
		t.Fatalf("got %d %s", response.Code, response.Body.String())
	}
	// Answers that do not cover the calls waiting.
	service.clientCalls = func(context.Context, auth.Identity, string) ([]string, error) { return []string{"ui-1", "ui-2"}, nil }
	service.answer = func(context.Context, auth.Identity, string, []llm.ToolResult) (conversation.TurnExecution, error) {
		return nil, conversation.ErrInvalidToolResults
	}
	response = runRequest(t, service, `{"threadId":"conversation-id","runId":"run-4","messages":[{"role":"tool","toolCallId":"ui-1","content":"opened"}]}`)
	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), `"code":"invalid_tool_results"`) {
		t.Fatalf("got %d %s", response.Code, response.Body.String())
	}
}

func TestCapabilities(t *testing.T) {
	response := httptest.NewRecorder()
	newConversationTestHandler(t, noopConversationService{}).ServeHTTP(response, authenticatedRequest(http.MethodGet, "/v1/om3/capabilities", ""))
	var capabilities map[string]map[string]any
	if err := json.Unmarshal(response.Body.Bytes(), &capabilities); err != nil || response.Code != http.StatusOK ||
		capabilities["tools"]["clientProvided"] != true || capabilities["humanInTheLoop"]["interrupts"] != true || capabilities["humanInTheLoop"]["approveWithEdits"] != true ||
		capabilities["custom"]["feedback"] != true {
		t.Fatalf("got %d %s", response.Code, response.Body.String())
	}
}

// A client that does not hold the last stored message, such as another tab,
// receives the thread first; a client in phase does not.
func TestRunSnapshotsTheThreadToAClientOutOfPhase(t *testing.T) {
	history := []llm.Message{{Role: llm.RoleUser, Text: "nodes?"}, {Role: llm.RoleAssistant, Text: "Two nodes."}}
	service := conversationServiceFuncs{prepare: func(context.Context, auth.Identity, string, string) (conversation.TurnExecution, error) {
		return &testTurnExecution{turnID: "turn-2", history: history, historyIDs: []string{"turn-1:1", "turn-1:2"}, run: func(_ context.Context, emit agent.EmitFunc) error {
			_ = emit(agent.Event{Type: agent.EventTextDelta, TextDelta: "No service.", Iteration: 1})
			return emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, TurnID: "turn-2", Iteration: 1})
		}}, nil
	}}
	behind := runRequest(t, service, `{"threadId":"conversation-id","runId":"run-2","protocolVersion":"1.0","messages":[{"id":"client-1","role":"user","content":"services?"}]}`)
	events := aguiEvents(t, behind.Body.String())
	if eventTypes(events) != "RUN_STARTED,MESSAGES_SNAPSHOT,TEXT_MESSAGE_START,TEXT_MESSAGE_CONTENT,TEXT_MESSAGE_END,RUN_FINISHED" {
		t.Fatalf("got %s", behind.Body.String())
	}
	thread := events[1]["messages"].([]any)
	if len(thread) != 3 || thread[0].(map[string]any)["id"] != "turn-1:1" || thread[2].(map[string]any)["id"] != "turn-2:1" || events[2]["messageId"] != "turn-2:2" {
		t.Fatalf("snapshot = %v", events[1])
	}
	inPhase := runRequest(t, service, `{"threadId":"conversation-id","runId":"run-3","messages":[{"id":"turn-1:1","role":"user","content":"nodes?"},
		{"id":"turn-1:2","role":"assistant","content":"Two nodes."},{"id":"client-1","role":"user","content":"services?"}]}`)
	if strings.Contains(inPhase.Body.String(), "MESSAGES_SNAPSHOT") {
		t.Fatalf("a client in phase received a snapshot: %s", inPhase.Body.String())
	}
}

func TestRunReadsTheProtocolVersionOfTheClient(t *testing.T) {
	service := conversationServiceFuncs{prepare: func(context.Context, auth.Identity, string, string) (conversation.TurnExecution, error) {
		return &testTurnExecution{run: func(_ context.Context, emit agent.EmitFunc) error {
			return emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, TurnID: "turn-id", Iteration: 1})
		}}, nil
	}}
	body := func(version string) string {
		return `{"threadId":"conversation-id","runId":"run-1","protocolVersion":"` + version + `","messages":[{"role":"user","content":"x"}]}`
	}
	for version, code := range map[string]string{"2.0": "unsupported_protocol_version", "1.x": "invalid_protocol_version"} {
		if response := runRequest(t, service, body(version)); response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), `"code":"`+code+`"`) {
			t.Errorf("%s: got %d %s", version, response.Code, response.Body.String())
		}
	}
	newer := aguiEvents(t, runRequest(t, service, body("1.3")).Body.String())
	if eventTypes(newer) != "RUN_STARTED,CUSTOM,RUN_FINISHED" || newer[0]["protocolVersion"] != "1.0" || !strings.Contains(newer[1]["value"].(map[string]any)["message"].(string), "1.3") {
		t.Fatalf("newer = %v", newer)
	}
}

// The ID the client gives its prompt names it; a resumed turn is in phase
// only with a client holding the answer it waits in.
func TestRunKeepsThePromptIDOfTheClient(t *testing.T) {
	var execution *testTurnExecution
	service := conversationServiceFuncs{prepare: func(context.Context, auth.Identity, string, string) (conversation.TurnExecution, error) {
		execution = &testTurnExecution{turnID: "turn-2", history: []llm.Message{{Role: llm.RoleUser, Text: "nodes?"}, {Role: llm.RoleAssistant, Text: "Two."}},
			historyIDs: []string{"client-u1", "turn-1:2"}, run: func(_ context.Context, emit agent.EmitFunc) error {
				return emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, TurnID: "turn-2", Iteration: 1})
			}}
		return execution, nil
	}}
	// Out of phase: the prompt goes under its stored ID, which the client
	// does not hold, and the ID of the client is not kept.
	response := runRequest(t, service, `{"threadId":"conversation-id","runId":"run-2","messages":[{"id":"client-u3","role":"user","content":"services?"}]}`)
	events := aguiEvents(t, response.Body.String())
	thread := events[1]["messages"].([]any)
	if execution.promptID != "" || thread[0].(map[string]any)["id"] != "client-u1" || thread[2].(map[string]any)["id"] != "turn-2:1" {
		t.Fatalf("prompt ID %q, snapshot %v", execution.promptID, events[1])
	}
	// In phase: the ID of the client is kept.
	runRequest(t, service, `{"threadId":"conversation-id","runId":"run-3","messages":[{"id":"client-u1","role":"user","content":"nodes?"},{"id":"turn-1:2","role":"assistant","content":"Two."},{"id":"client-u3","role":"user","content":"services?"}]}`)
	if execution.promptID != "client-u3" {
		t.Fatalf("prompt ID in phase %q", execution.promptID)
	}
	if response := runRequest(t, service, `{"threadId":"conversation-id","runId":"run-3","messages":[{"id":"has space","role":"user","content":"x"}]}`); response.Code != http.StatusBadRequest ||
		!strings.Contains(response.Body.String(), `"code":"invalid_message_id"`) {
		t.Fatalf("got %d %s", response.Code, response.Body.String())
	}
	// A client holding the last stored message, but not the answer the
	// resumed turn waits in, is out of phase.
	resumed := &testTurnExecution{turnID: "turn-2", history: []llm.Message{{Role: llm.RoleUser, Text: "nodes?"}, {Role: llm.RoleAssistant, Text: "Two."}}, historyIDs: []string{"client-u1", "turn-1:2"},
		current:    []llm.Message{{Role: llm.RoleUser, Text: "stop web"}, {Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call-7", Name: "stop_object", Arguments: json.RawMessage(`{}`)}}}},
		currentIDs: []string{"client-u3", "turn-2:2"},
		run: func(_ context.Context, emit agent.EmitFunc) error {
			return emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, TurnID: "turn-2", Iteration: 1})
		}}
	held := agui.RunAgentInput{Messages: []agui.Message{{ID: "client-u1"}, {ID: "turn-1:2"}}}
	messages, ids, stored, _ := resumed.Snapshot()
	if inPhase(held, messages, ids, stored) {
		t.Fatal("a client lacking the answer the turn waits in is in phase")
	}
	held.Messages = append(held.Messages, agui.Message{ID: "turn-2:2"})
	if !inPhase(held, messages, ids, stored) {
		t.Fatal("a client holding the answer the turn waits in is out of phase")
	}
}

// The arguments the user edited reach the service, and the audit records
// their hash and size, never their content.
func TestRunResumesWithTheEditedArguments(t *testing.T) {
	state := `{"messages":[],"iteration":1,"calls":[{"ID":"call-7","Name":"switch_object","Arguments":{"path":"web"}}],"results":[]}`
	var edited json.RawMessage
	service := conversationServiceFuncs{
		pending: func(context.Context, auth.Identity, string) (*conversation.PendingConfirmation, error) {
			return &conversation.PendingConfirmation{TurnID: "turn-id", ConfirmationID: "confirmation-id", ToolName: "switch_object", State: []byte(state)}, nil
		},
		confirmEdited: func(_ context.Context, _ auth.Identity, _, _, _ string, decision agent.Decision, got json.RawMessage) (conversation.TurnExecution, error) {
			if decision != agent.DecisionApprove {
				t.Fatalf("decision %q", decision)
			}
			edited = got
			return &testTurnExecution{run: func(_ context.Context, emit agent.EmitFunc) error {
				return emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, TurnID: "turn-id", Iteration: 2})
			}}, nil
		},
	}
	var output bytes.Buffer
	handler, err := NewHandler([]Backend{{Name: "om3", Conversations: service, Authenticator: allowTestTokenVerifier()}},
		HandlerConfig{Models: testModels(), MaxConcurrentAsks: 4, AuditLogger: slog.New(slog.NewJSONHandler(&output, nil))})
	if err != nil {
		t.Fatal(err)
	}
	request := authenticatedRequest(http.MethodPost, "/v1/om3/conversations/conversation-id/runs",
		`{"threadId":"conversation-id","runId":"run-2","messages":[],"resume":[{"interruptId":"confirmation-id","status":"resolved","payload":{"approved":true,"editedArgs":{"path":"web","to":"secret-node-marker"}}}]}`)
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK || string(edited) != `{"path":"web","to":"secret-node-marker"}` {
		t.Fatalf("got %d %s, edited %s", response.Code, response.Body.String(), edited)
	}
	if !strings.Contains(output.String(), `"arguments_edited":true`) || !strings.Contains(output.String(), `"edited_arguments_sha256":"`+agent.ArgumentsHash(edited)+`"`) ||
		strings.Contains(output.String(), "secret-node-marker") {
		t.Fatalf("audit = %s", output.String())
	}
}

// The snapshot names the skill the prompt of the run asked, and the one of
// the turn a run resumes, as the history names them.
func TestRunSnapshotNamesTheSkillsAsked(t *testing.T) {
	history := []llm.Message{{Role: llm.RoleUser, Text: "nodes?"}, {Role: llm.RoleAssistant, Text: "Two nodes."}}
	completes := func(_ context.Context, emit agent.EmitFunc) error {
		return emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, TurnID: "turn-2", Iteration: 2})
	}
	state := `{"messages":[],"iteration":1,"calls":[{"ID":"call-7","Name":"stop_object","Arguments":{}}],"results":[]}`
	service := conversationServiceFuncs{
		prepare: func(context.Context, auth.Identity, string, string) (conversation.TurnExecution, error) {
			return &testTurnExecution{turnID: "turn-2", history: history, historyIDs: []string{"turn-1:1", "turn-1:2"}, run: completes}, nil
		},
		pending: func(context.Context, auth.Identity, string) (*conversation.PendingConfirmation, error) {
			return &conversation.PendingConfirmation{TurnID: "turn-2", ConfirmationID: "confirmation-id", ToolName: "stop_object", State: []byte(state)}, nil
		},
		confirm: func(context.Context, auth.Identity, string, string, string, agent.Decision) (conversation.TurnExecution, error) {
			return &testTurnExecution{turnID: "turn-2", history: history, historyIDs: []string{"turn-1:1", "turn-1:2"}, askedSkill: "node-maintenance",
				current:    []llm.Message{{Role: llm.RoleUser, Text: "on n1"}, {Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call-7", Name: "stop_object", Arguments: json.RawMessage(`{}`)}}}},
				currentIDs: []string{"turn-2:1", "turn-2:2"}, run: completes}, nil
		},
	}
	skillsOf := func(body string) map[string]any {
		events := aguiEvents(t, body)
		skills := map[string]any{}
		for _, event := range events {
			if event["type"] != "MESSAGES_SNAPSHOT" {
				continue
			}
			for _, message := range event["messages"].([]any) {
				item := message.(map[string]any)
				if item["role"] == "user" {
					skills[item["id"].(string)] = item["skill"]
				}
			}
		}
		return skills
	}
	asked := runRequest(t, service, `{"threadId":"conversation-id","runId":"run-2","messages":[{"id":"client-1","role":"user","content":"on n1"}],"forwardedProps":{"skill":"node-maintenance"}}`)
	if got := skillsOf(asked.Body.String()); len(got) != 2 || got["turn-1:1"] != nil || got["turn-2:1"] != "node-maintenance" {
		t.Fatalf("new prompt: skills = %v, body %s", got, asked.Body.String())
	}
	resumed := runRequest(t, service, `{"threadId":"conversation-id","runId":"run-3","messages":[],"resume":[{"interruptId":"confirmation-id","status":"resolved","payload":{"approved":true}}]}`)
	if got := skillsOf(resumed.Body.String()); len(got) != 2 || got["turn-1:1"] != nil || got["turn-2:1"] != "node-maintenance" {
		t.Fatalf("resumed: skills = %v, body %s", got, resumed.Body.String())
	}
}
