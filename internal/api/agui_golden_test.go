package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"testing"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/conversation"
	"github.com/opensvc/ai-agent/internal/llm"
)

// goldenStreamsPath holds the AG-UI streams the runs route serves in the
// scenarios of a client: the web interfaces replay them in their tests, so
// that a change of the stream breaks them. AGUI_GOLDEN_UPDATE=1 rewrites it.
var goldenStreamsPath = filepath.Join("testdata", "agui", "streams.json")

// goldenStream is one scenario: the RunAgentInput, and the answer of the
// route, an SSE stream or a JSON error.
type goldenStream struct {
	Name        string          `json:"name"`
	Request     json.RawMessage `json:"request"`
	Status      int             `json:"status"`
	ContentType string          `json:"contentType"`
	Body        string          `json:"body"`
}

// goldenTimestamp replaces the time of each event, for the file to be stable.
var goldenTimestamp = regexp.MustCompile(`"timestamp":\d+`)

func emitEvents(events ...agent.Event) func(context.Context, agent.EmitFunc) error {
	return func(_ context.Context, emit agent.EmitFunc) error {
		for _, event := range events {
			if err := emit(event); err != nil {
				return err
			}
		}
		return nil
	}
}

func usageEvent(input, output int64) agent.Event {
	return agent.Event{Type: agent.EventUsage, Usage: &llm.Usage{InputTokens: input, OutputTokens: output, TotalTokens: input + output}}
}

func TestGoldenAGUIStreams(t *testing.T) {
	confirmation := &agent.Confirmation{
		ID: "confirmation-1", TurnID: "turn-c", ToolName: "update_object_config", ToolTitle: "Update object config", Destructive: true,
		ToolCallID: "call-7", Arguments: json.RawMessage(`{"path":"web","yaml":"app#1:\n  start: /bin/true\n"}`),
		ArgumentTypes: map[string]string{"yaml": "application/yaml"}, ExpiresAt: confirmationExpiry,
		InputSchema: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string","description":"The path of the object."},"yaml":{"type":"string","contentMediaType":"application/yaml","description":"The configuration to apply."},"wait":{"type":"boolean","description":"Wait until every live node has it."}},"required":["path","yaml"],"additionalProperties":false}`),
	}
	pending := func(context.Context, auth.Identity, string) (*conversation.PendingConfirmation, error) {
		state := `{"messages":[],"iteration":1,"calls":[{"ID":"call-7","Name":"update_object_config","Arguments":{}}],"results":[]}`
		return &conversation.PendingConfirmation{TurnID: "turn-c", ConfirmationID: "confirmation-1", ToolName: "update_object_config", State: []byte(state)}, nil
	}
	prompt := func(execution *testTurnExecution) conversationServiceFuncs {
		return conversationServiceFuncs{pending: pending, prepare: func(context.Context, auth.Identity, string, string) (conversation.TurnExecution, error) {
			return execution, nil
		}}
	}
	resume := func(execution *testTurnExecution) conversationServiceFuncs {
		return conversationServiceFuncs{pending: pending, confirm: func(context.Context, auth.Identity, string, string, string, agent.Decision) (conversation.TurnExecution, error) {
			return execution, nil
		}}
	}
	scenarios := []struct {
		name    string
		service conversationServiceFuncs
		request string
	}{
		{
			// Two answers of the model in one turn, around the tools it calls:
			// search_tools is a tool of the agent itself.
			name: "text_and_tools",
			service: prompt(&testTurnExecution{turnID: "turn-1", run: emitEvents(
				agent.Event{Type: agent.EventTextDelta, TextDelta: "Let me check ", Iteration: 1},
				agent.Event{Type: agent.EventTextDelta, TextDelta: "the nodes.", Iteration: 1},
				agent.Event{Type: agent.EventToolStarted, ToolName: "search_tools", ToolCallID: "call-0", ToolArguments: json.RawMessage(`{"query":"nodes"}`), Iteration: 1},
				agent.Event{Type: agent.EventToolFinished, ToolName: "search_tools", ToolCallID: "call-0", ToolResult: json.RawMessage(`{"content":[{"type":"text","text":"list_nodes"}]}`), Iteration: 1},
				agent.Event{Type: agent.EventToolStarted, ToolName: "list_nodes", ToolCallID: "call-1", ToolArguments: json.RawMessage(`{}`), Iteration: 1},
				agent.Event{Type: agent.EventToolFinished, ToolName: "list_nodes", ToolCallID: "call-1", ToolResult: json.RawMessage(`{"content":[],"structuredContent":{"nodes":["n1","n2"]}}`), Iteration: 1},
				usageEvent(120, 30),
				agent.Event{Type: agent.EventTextDelta, TextDelta: "There are two nodes: **n1** and **n2**.", Iteration: 2},
				usageEvent(200, 20),
				agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, TurnID: "turn-1", Iteration: 2},
			)}),
			request: `{"threadId":"conversation-id","runId":"run-1","protocolVersion":"1.0","messages":[{"id":"u-1","role":"user","content":"How many nodes?"}],"tools":[]}`,
		},
		{
			// The run ends on the call it asks the user to confirm.
			name: "confirmation",
			service: prompt(&testTurnExecution{turnID: "turn-c", run: emitEvents(
				agent.Event{Type: agent.EventTextDelta, TextDelta: "I will apply this configuration.", Iteration: 1},
				usageEvent(300, 40),
				agent.Event{Type: agent.EventConfirmationRequired, Iteration: 1, Confirmation: confirmation},
			)}),
			request: `{"threadId":"conversation-id","runId":"run-1","protocolVersion":"1.0","messages":[{"id":"u-c","role":"user","content":"Start web with /bin/true"}],"tools":[]}`,
		},
		{
			// The approved call runs: its result, then the answer.
			name: "resume_approved",
			service: resume(&testTurnExecution{turnID: "turn-c", run: emitEvents(
				agent.Event{Type: agent.EventToolStarted, ToolName: "update_object_config", ToolCallID: "call-7", ToolArguments: confirmation.Arguments, Iteration: 1},
				agent.Event{Type: agent.EventToolFinished, ToolName: "update_object_config", ToolCallID: "call-7", ToolResult: json.RawMessage(`{"content":[],"structuredContent":{"outcome":"written"}}`), Iteration: 1},
				agent.Event{Type: agent.EventTextDelta, TextDelta: "Done: web starts with /bin/true.", Iteration: 2},
				usageEvent(400, 15),
				agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, TurnID: "turn-c", Iteration: 2},
			)}),
			request: `{"threadId":"conversation-id","runId":"run-1","protocolVersion":"1.0","messages":[{"id":"u-c","role":"user"},{"id":"turn-c:2","role":"assistant"}],"tools":[],"resume":[{"interruptId":"confirmation-1","status":"resolved","payload":{"approved":true}}]}`,
		},
		{
			// The rejected call does not run: its result says so.
			name: "resume_rejected",
			service: resume(&testTurnExecution{turnID: "turn-c", run: emitEvents(
				agent.Event{Type: agent.EventTextDelta, TextDelta: "I did not apply it.", Iteration: 2},
				agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, TurnID: "turn-c", Iteration: 2},
			)}),
			request: `{"threadId":"conversation-id","runId":"run-1","protocolVersion":"1.0","messages":[{"id":"u-c","role":"user"},{"id":"turn-c:2","role":"assistant"}],"tools":[],"resume":[{"interruptId":"confirmation-1","status":"resolved","payload":{"approved":false}}]}`,
		},
		{
			// A skill the user asked, forwardedProps.skill: the agent loads it
			// first, a step of the first iteration, the model answering from
			// the second.
			name: "skill_asked",
			service: prompt(&testTurnExecution{turnID: "turn-s", run: emitEvents(
				agent.Event{Type: agent.EventToolStarted, ToolName: "load_skill", ToolCallID: "call_skill_1", ToolArguments: json.RawMessage(`{"skill":"morning-check"}`), Iteration: 1},
				agent.Event{Type: agent.EventToolFinished, ToolName: "load_skill", ToolCallID: "call_skill_1", ToolResult: json.RawMessage(`{"content":[{"type":"text","text":"Skill morning-check, loaded from the om3 backend; follow its procedure."}]}`), Iteration: 1},
				agent.Event{Type: agent.EventTextDelta, TextDelta: "The cluster is healthy.", Iteration: 2},
				usageEvent(500, 25),
				agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, TurnID: "turn-s", Iteration: 2},
			)}),
			request: `{"threadId":"conversation-id","runId":"run-1","protocolVersion":"1.0","messages":[{"id":"u-s","role":"user","content":"/morning-check"}],"tools":[],"forwardedProps":{"skill":"morning-check"}}`,
		},
		{
			// The skill asked is not served to the user: its step fails, then
			// the run.
			name: "skill_unavailable",
			service: prompt(&testTurnExecution{turnID: "turn-u", run: func(_ context.Context, emit agent.EmitFunc) error {
				if err := emit(agent.Event{Type: agent.EventToolStarted, ToolName: "load_skill", ToolCallID: "call_skill_2", ToolArguments: json.RawMessage(`{"skill":"morning-check"}`), Iteration: 1}); err != nil {
					return err
				}
				if err := emit(agent.Event{Type: agent.EventToolFinished, ToolName: "load_skill", ToolCallID: "call_skill_2", ToolError: true, ToolResult: json.RawMessage(`{"content":[{"type":"text","text":"The skill morning-check is not served to you now."}]}`), Iteration: 1}); err != nil {
					return err
				}
				return fmt.Errorf("%w: The skill morning-check is not served to you now.", agent.ErrSkillUnavailable)
			}}),
			request: `{"threadId":"conversation-id","runId":"run-1","protocolVersion":"1.0","messages":[{"id":"u-u","role":"user","content":"/morning-check"}],"tools":[],"forwardedProps":{"skill":"morning-check"}}`,
		},
		{
			name: "agent_failed",
			service: prompt(&testTurnExecution{turnID: "turn-f", run: func(context.Context, agent.EmitFunc) error {
				return errors.New("the model provider failed")
			}}),
			request: `{"threadId":"conversation-id","runId":"run-1","protocolVersion":"1.0","messages":[{"id":"u-f","role":"user","content":"Hello"}],"tools":[]}`,
		},
		{
			// A client that lacks the last stored turn, such as another tab,
			// receives the thread first, its prompt under the stored ID.
			name: "snapshot",
			service: prompt(&testTurnExecution{turnID: "turn-2",
				history: []llm.Message{
					{Role: llm.RoleUser, Text: "How many nodes?"},
					{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "list_nodes", Arguments: json.RawMessage(`{}`)}}},
					{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call-1", Content: json.RawMessage(`{"content":[],"structuredContent":{"nodes":["n1","n2"]}}`)}}},
					{Role: llm.RoleAssistant, Text: "Two nodes."},
				},
				historyIDs: []string{"u-1", "turn-1:2", "turn-1:3", "turn-1:4"},
				run: emitEvents(
					agent.Event{Type: agent.EventTextDelta, TextDelta: "No service.", Iteration: 1},
					agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, TurnID: "turn-2", Iteration: 1},
				)}),
			request: `{"threadId":"conversation-id","runId":"run-1","protocolVersion":"1.0","messages":[{"id":"u-2","role":"user","content":"Any service?"}],"tools":[]}`,
		},
		{
			// A client reloaded while the turn waits holds its earlier turns
			// only: the thread comes first, with the call waiting.
			name: "resume_snapshot",
			service: resume(&testTurnExecution{turnID: "turn-c",
				history:    []llm.Message{{Role: llm.RoleUser, Text: "How many nodes?"}, {Role: llm.RoleAssistant, Text: "Two nodes."}},
				historyIDs: []string{"u-1", "turn-1:2"},
				current: []llm.Message{
					{Role: llm.RoleUser, Text: "Start web with /bin/true"},
					{Role: llm.RoleAssistant, Text: "I will apply this configuration.", ToolCalls: []llm.ToolCall{{ID: "call-7", Name: "update_object_config", Arguments: confirmation.Arguments}}},
				},
				currentIDs: []string{"u-c", "turn-c:2"},
				run: emitEvents(
					agent.Event{Type: agent.EventTextDelta, TextDelta: "I did not apply it.", Iteration: 2},
					agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, TurnID: "turn-c", Iteration: 2},
				)}),
			request: `{"threadId":"conversation-id","runId":"run-1","protocolVersion":"1.0","messages":[{"id":"u-1","role":"user"},{"id":"turn-1:2","role":"assistant"}],"tools":[],"resume":[{"interruptId":"confirmation-1","status":"resolved","payload":{"approved":false}}]}`,
		},
		{
			// The user approved the call with arguments of their own: the
			// result tells the model, ahead of what the call returned.
			name: "resume_edited",
			service: conversationServiceFuncs{pending: pending, confirmEdited: func(_ context.Context, _ auth.Identity, _, _, _ string, decision agent.Decision, edited json.RawMessage) (conversation.TurnExecution, error) {
				if decision != agent.DecisionApprove || string(edited) != `{"path":"web","yaml":"app#1:\n  start: /bin/false\n"}` {
					return nil, errors.New("unexpected edit")
				}
				return &testTurnExecution{turnID: "turn-c", run: emitEvents(
					agent.Event{Type: agent.EventToolStarted, ToolName: "update_object_config", ToolCallID: "call-7", ToolArguments: edited, Iteration: 1},
					agent.Event{Type: agent.EventToolFinished, ToolName: "update_object_config", ToolCallID: "call-7", Iteration: 1,
						ToolResult: json.RawMessage(`{"content":[{"type":"text","text":"The user edited the arguments before approving; the call ran with: {\"path\":\"web\",\"yaml\":\"app#1:\\n  start: /bin/false\\n\"}"}],"structuredContent":{"outcome":"written"}}`)},
					agent.Event{Type: agent.EventTextDelta, TextDelta: "Done: web starts with /bin/false, as you set it.", Iteration: 2},
					agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, TurnID: "turn-c", Iteration: 2},
				)}, nil
			}},
			request: `{"threadId":"conversation-id","runId":"run-1","protocolVersion":"1.0","messages":[{"id":"u-c","role":"user"},{"id":"turn-c:2","role":"assistant"}],"tools":[],"resume":[{"interruptId":"confirmation-1","status":"resolved","payload":{"approved":true,"editedArgs":{"path":"web","yaml":"app#1:\n  start: /bin/false\n"}}}]}`,
		},
		{
			// An edit the input schema of the tool refuses: the confirmation
			// still waits, for the user to correct the form.
			name: "invalid_edited_arguments",
			service: conversationServiceFuncs{pending: pending, confirmEdited: func(context.Context, auth.Identity, string, string, string, agent.Decision, json.RawMessage) (conversation.TurnExecution, error) {
				return nil, fmt.Errorf("%w: validating root: required: missing properties: [\"yaml\"]", agent.ErrInvalidEditedArguments)
			}},
			request: `{"threadId":"conversation-id","runId":"run-1","protocolVersion":"1.0","messages":[],"tools":[],"resume":[{"interruptId":"confirmation-1","status":"resolved","payload":{"approved":true,"editedArgs":{"path":"web"}}}]}`,
		},
		{
			// The sentence of an action, which the client says in its own
			// language from its key and params.
			name: "confirmation_preview_action",
			service: prompt(&testTurnExecution{turnID: "turn-p", run: emitEvents(
				agent.Event{Type: agent.EventConfirmationRequired, Iteration: 1, Confirmation: &agent.Confirmation{
					ID: "confirmation-p", TurnID: "turn-p", ToolName: "stop_instance", ToolTitle: "Stop instance", Destructive: true,
					ToolCallID: "call-p", Arguments: json.RawMessage(`{"path":"test/svc/web","node":"dev5n1","rids":["fs#1"]}`), ExpiresAt: confirmationExpiry,
					Preview: json.RawMessage(`{"provenance":{"source":"opensvc_daemon","observed_at":"2026-10-26T09:00:00Z"},"summary":{"text":"Stop test/svc/web on dev5n1 (fs#1)","key":"stop_instance","params":{"node":"dev5n1","path":"test/svc/web","rids":"fs#1"}}}`),
				}},
			)}),
			request: `{"threadId":"conversation-id","runId":"run-1","protocolVersion":"1.0","messages":[{"id":"u-p","role":"user","content":"Stop fs#1 of web on dev5n1"}],"tools":[]}`,
		},
		{
			// The keys a configuration update changes, read now, and the
			// sections it deletes.
			name: "confirmation_preview_config",
			service: prompt(&testTurnExecution{turnID: "turn-u", run: emitEvents(
				agent.Event{Type: agent.EventConfirmationRequired, Iteration: 1, Confirmation: &agent.Confirmation{
					ID: "confirmation-u", TurnID: "turn-u", ToolName: "update_object_config", ToolTitle: "Update object configuration", Destructive: true,
					ToolCallID: "call-u", Arguments: json.RawMessage(`{"path":"test/svc/web","set":["env.list+=b","cluster.secret=x"],"unset":["env.absent"],"delete":["fs#2"]}`), ExpiresAt: confirmationExpiry,
					InputSchema: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"},"set":{"type":"array","items":{"type":"string"}},"unset":{"type":"array","items":{"type":"string"}},"delete":{"type":"array","items":{"type":"string"}}},"required":["path"]}`),
					Preview: json.RawMessage(`{"provenance":{"source":"opensvc_daemon","observed_at":"2026-10-26T09:00:00Z"},"summary":{"text":"Update the configuration of test/svc/web: 2 set, 1 unset, 1 sections deleted","key":"config_update","params":{"deletes":"1","path":"test/svc/web","sets":"2","unsets":"1"}},` +
						`"changes":[{"key":"env.absent","op":"unset","before":"","after":"","after_computed_by_mcp":true,"note":"not set","note_keys":["not_set"]},{"key":"env.list","op":"+=","value":"b","before":"a","after":"a b","after_computed_by_mcp":true},{"key":"cluster.secret","op":"=","after_computed_by_mcp":false,"note":"secret: values not shown","note_keys":["secret"]}],` +
						`"deleted":[{"section":"fs#2","keys":["fs#2.dev","fs#2.type"]}],"read_at":"2026-10-26T09:00:00Z","secrets_omitted":1}`),
				}},
			)}),
			request: `{"threadId":"conversation-id","runId":"run-1","protocolVersion":"1.0","messages":[{"id":"u-u","role":"user","content":"Add b to env.list of web"}],"tools":[]}`,
		},
		{
			// The configuration file an object is created from, as given.
			name: "confirmation_preview_create",
			service: prompt(&testTurnExecution{turnID: "turn-n", run: emitEvents(
				agent.Event{Type: agent.EventConfirmationRequired, Iteration: 1, Confirmation: &agent.Confirmation{
					ID: "confirmation-n", TurnID: "turn-n", ToolName: "create_object", ToolTitle: "Create object", Destructive: true,
					ToolCallID: "call-n", Arguments: json.RawMessage(`{"path":"test/svc/new","config":"[DEFAULT]\nnodes = dev5n1\n\n[fs#1]\ntype = flag\n","set":["env.a=1"]}`), ExpiresAt: confirmationExpiry,
					Preview: json.RawMessage(`{"provenance":{"source":"opensvc_daemon","observed_at":"2026-10-26T09:00:00Z"},"summary":{"text":"Create test/svc/new, a svc of 2 sections","key":"object_create","params":{"kind":"svc","path":"test/svc/new","sections":"2"}},` +
						`"changes":[{"key":"env.a","op":"=","value":"1","after_computed_by_mcp":false,"note":"applied after the configuration file","note_keys":["applied_after_file"]}],"content":{"media_type":"text/x-ini","text":"[DEFAULT]\nnodes = dev5n1\n\n[fs#1]\ntype = flag\n"}}`),
				}},
			)}),
			request: `{"threadId":"conversation-id","runId":"run-1","protocolVersion":"1.0","messages":[{"id":"u-n","role":"user","content":"Create new with a flag"}],"tools":[]}`,
		},
		{
			// A preview that failed blocks nothing: the client shows the
			// arguments, and why.
			name: "confirmation_preview_error",
			service: prompt(&testTurnExecution{turnID: "turn-e", run: emitEvents(
				agent.Event{Type: agent.EventConfirmationRequired, Iteration: 1, Confirmation: &agent.Confirmation{
					ID: "confirmation-e", TurnID: "turn-e", ToolName: "update_object_config", ToolTitle: "Update object configuration", Destructive: true,
					ToolCallID: "call-e", Arguments: json.RawMessage(`{"path":"test/svc/web","set":["env.a=1"]}`), ExpiresAt: confirmationExpiry,
					PreviewError: "the preview was refused: get object config: OpenSVC daemon returned HTTP 404",
				}},
			)}),
			request: `{"threadId":"conversation-id","runId":"run-1","protocolVersion":"1.0","messages":[{"id":"u-e","role":"user","content":"Set env.a of web"}],"tools":[]}`,
		},
		{
			// The user edited the arguments and asks what the call would do
			// with them: nothing runs, the call waits again under a new
			// interrupt, with the preview of the arguments edited.
			name: "resume_preview",
			service: conversationServiceFuncs{pending: pending, confirmEdited: func(_ context.Context, _ auth.Identity, _, _, _ string, decision agent.Decision, edited json.RawMessage) (conversation.TurnExecution, error) {
				if decision != agent.DecisionPreview || string(edited) != `{"path":"api","yaml":"app#1:\n  start: /bin/true\n"}` {
					return nil, errors.New("unexpected preview")
				}
				again := *confirmation
				again.ID, again.PreviewArguments = "confirmation-2", edited
				again.Preview = json.RawMessage(`{"provenance":{"source":"opensvc_daemon","observed_at":"2026-10-26T09:00:00Z"},"summary":{"text":"Update api","key":"config_update","params":{"path":"api"}}}`)
				return &testTurnExecution{turnID: "turn-c", run: emitEvents(
					agent.Event{Type: agent.EventConfirmationRequired, Iteration: 1, Confirmation: &again},
				)}, nil
			}},
			request: `{"threadId":"conversation-id","runId":"run-1","protocolVersion":"1.0","messages":[{"id":"u-c","role":"user"},{"id":"turn-c:2","role":"assistant"}],"tools":[],"resume":[{"interruptId":"confirmation-1","status":"resolved","payload":{"action":"preview","editedArgs":{"path":"api","yaml":"app#1:\n  start: /bin/true\n"}}}]}`,
		},
		{
			name:    "confirmation_not_pending",
			service: resume(&testTurnExecution{run: emitEvents()}),
			request: `{"threadId":"conversation-id","runId":"run-1","protocolVersion":"1.0","messages":[],"tools":[],"resume":[{"interruptId":"confirmation-0","status":"resolved","payload":{"approved":true}}]}`,
		},
		{
			name: "conversation_busy",
			service: conversationServiceFuncs{prepare: func(context.Context, auth.Identity, string, string) (conversation.TurnExecution, error) {
				return nil, conversation.ErrBusy
			}},
			request: `{"threadId":"conversation-id","runId":"run-1","protocolVersion":"1.0","messages":[{"id":"u-b","role":"user","content":"Hello"}],"tools":[]}`,
		},
	}
	streams := make([]goldenStream, 0, len(scenarios))
	for _, scenario := range scenarios {
		response := runRequest(t, scenario.service, scenario.request)
		var request bytes.Buffer
		if err := json.Compact(&request, []byte(scenario.request)); err != nil {
			t.Fatalf("%s: %v", scenario.name, err)
		}
		streams = append(streams, goldenStream{
			Name: scenario.name, Request: request.Bytes(), Status: response.Code, ContentType: response.Header().Get("Content-Type"),
			Body: goldenTimestamp.ReplaceAllString(response.Body.String(), `"timestamp":1760000000000`),
		})
	}
	data, err := json.MarshalIndent(streams, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	data = append(data, '\n')
	if os.Getenv("AGUI_GOLDEN_UPDATE") == "1" {
		if err := os.WriteFile(goldenStreamsPath, data, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(goldenStreamsPath)
	if err != nil {
		t.Fatalf("read %s, written with AGUI_GOLDEN_UPDATE=1: %v", goldenStreamsPath, err)
	}
	if !bytes.Equal(data, want) {
		t.Fatalf("the AG-UI streams differ from %s: rewrite it with AGUI_GOLDEN_UPDATE=1, and copy it to the web interfaces that replay it\n%s", goldenStreamsPath, data)
	}
}
