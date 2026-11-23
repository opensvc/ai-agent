package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/llm"
)

// testSystemPrompt stands for the prompt of a backend: the agent passes it
// unchanged as the first message.
const testSystemPrompt = "You are a test assistant."

func TestAskReturnsDirectAnswerAndExposesAllTools(t *testing.T) {
	session := &fakeSession{tools: []*mcp.Tool{
		{Name: "get_cluster_health", Description: "Get health", InputSchema: objectSchema()},
		{Name: "refresh_instance_status", Description: "Refresh status", InputSchema: objectSchema()},
	}}
	model := &scriptedLLM{t: t, inspectContext: func(ctx context.Context) {
		if auth.MCPHeadersFromContext(ctx) != nil {
			t.Fatal("LLM context contains MCP headers")
		}
		if _, ok := auth.BearerTokenFromContext(ctx); ok {
			t.Fatal("LLM context contains delegated JWT")
		}
		if _, ok := auth.IdentityFromContext(ctx); ok {
			t.Fatal("LLM context contains verified OpenSVC identity")
		}
	}, steps: []llmStep{func(request llm.Request, emit llm.EmitFunc) error {
		if len(request.Messages) != 2 || request.Messages[0].Role != llm.RoleSystem || request.Messages[0].Text != testSystemPrompt+"\n"+changesInstruction+"\n"+interfaceContextInstruction+"\n"+noPendingConfirmation {
			t.Fatalf("unexpected initial messages: %#v", request.Messages)
		}
		if len(request.Tools) != 2 || request.Tools[1].Name != "refresh_instance_status" {
			t.Fatalf("all MCP tools were not exposed: %#v", request.Tools)
		}
		if strings.Contains(fmt.Sprintf("%#v", request), "jwt-marker") {
			t.Fatal("LLM request contains delegated JWT")
		}
		return emitEvents(emit,
			llm.Event{Type: llm.EventTextDelta, TextDelta: "cluster healthy"},
			llm.Event{Type: llm.EventUsage, Usage: &llm.Usage{InputTokens: 10, OutputTokens: 2, TotalTokens: 12}},
			llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonCompleted},
		)
	}}}
	agent := newTestAgent(t, model, session, 4)
	ctx := auth.WithBearerToken(t.Context(), "jwt-marker")
	ctx = auth.WithMCPHeaders(ctx, http.Header{"X-OpenSVC-Cluster-ID": {"cluster-id"}})
	ctx = auth.WithIdentity(ctx, auth.Identity{ClusterID: "cluster-id", Subject: "alice", Issuer: "node-a", Grants: []string{"guest"}})
	var events []Event
	result, err := agent.RunTurn(ctx, nil, "health of my cluster", func(event Event) error {
		events = append(events, event)
		return nil
	})
	if err != nil {
		t.Fatalf("Ask() error: %v", err)
	}
	listed, _ := session.ListTools(t.Context())
	wantHash, _ := toolCatalogHash(listed)
	if result.MCP.Name != "test-mcp" || result.MCP.Version != "v9.9.9" || len(result.MCP.ToolsSHA256) != 64 || result.MCP.ToolsSHA256 != wantHash {
		t.Fatalf("turn MCP = %+v", result.MCP)
	}
	if !session.connectedWithJWT || !session.closed {
		t.Fatalf("session JWT=%v closed=%v", session.connectedWithJWT, session.closed)
	}
	if len(session.calls) != 0 {
		t.Fatalf("unexpected MCP calls: %#v", session.calls)
	}
	if len(events) != 3 || events[0].TextDelta != "cluster healthy" || events[2].Type != EventCompleted {
		t.Fatalf("unexpected agent events: %#v", events)
	}
}

func TestAskExecutesToolAndReturnsResultToLLM(t *testing.T) {
	session := &fakeSession{
		tools: []*mcp.Tool{{Name: "get_cluster_health", Description: "Get health", InputSchema: objectSchema()}},
		results: map[string]*mcp.CallToolResult{
			"get_cluster_health": {StructuredContent: map[string]any{"status": "healthy"}},
		},
	}
	model := &scriptedLLM{t: t, steps: []llmStep{
		func(_ llm.Request, emit llm.EmitFunc) error {
			return emitEvents(emit,
				llm.Event{Type: llm.EventToolCall, ToolCall: &llm.ToolCall{ID: "call-1", Name: "get_cluster_health", Arguments: json.RawMessage(`{"node":"node1"}`)}},
				llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonToolCalls},
			)
		},
		func(request llm.Request, emit llm.EmitFunc) error {
			if len(request.Messages) != 4 {
				t.Fatalf("got %d messages, want system, user, assistant and tool", len(request.Messages))
			}
			assistant := request.Messages[2]
			result := request.Messages[3]
			if len(assistant.ToolCalls) != 1 || len(result.ToolResults) != 1 || result.ToolResults[0].CallID != "call-1" {
				t.Fatalf("unexpected tool history: assistant=%#v result=%#v", assistant, result)
			}
			if !strings.Contains(string(result.ToolResults[0].Content), `"status":"healthy"`) {
				t.Fatalf("tool result not preserved: %s", result.ToolResults[0].Content)
			}
			return emitEvents(emit,
				llm.Event{Type: llm.EventTextDelta, TextDelta: "The cluster is healthy."},
				llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonCompleted},
			)
		},
	}}
	agent := newTestAgent(t, model, session, 4)
	var events []Event
	if _, err := agent.RunTurn(auth.WithBearerToken(t.Context(), "jwt-marker"), nil, "check health", func(event Event) error {
		events = append(events, event)
		return nil
	}); err != nil {
		t.Fatalf("Ask() error: %v", err)
	}
	if len(session.calls) != 1 || session.calls[0].name != "get_cluster_health" || session.calls[0].arguments["node"] != "node1" {
		t.Fatalf("unexpected MCP calls: %#v", session.calls)
	}
	wantTypes := []EventType{EventToolStarted, EventToolFinished, EventTextDelta, EventCompleted}
	if len(events) != len(wantTypes) {
		t.Fatalf("got events %#v", events)
	}
	for index, want := range wantTypes {
		if events[index].Type != want {
			t.Fatalf("event %d type=%q, want %q", index, events[index].Type, want)
		}
	}
}

func TestAskReturnsFunctionalToolErrorToLLM(t *testing.T) {
	session := &fakeSession{
		tools: []*mcp.Tool{{Name: "get_cluster_health", InputSchema: objectSchema()}},
		results: map[string]*mcp.CallToolResult{
			"get_cluster_health": {IsError: true, StructuredContent: map[string]any{"error": "unavailable"}},
		},
	}
	model := &scriptedLLM{t: t, steps: []llmStep{
		toolCallStep("get_cluster_health", "call-1"),
		func(request llm.Request, emit llm.EmitFunc) error {
			result := request.Messages[len(request.Messages)-1].ToolResults[0]
			if !result.IsError {
				t.Fatal("functional MCP error was not returned to LLM")
			}
			return emitEvents(emit,
				llm.Event{Type: llm.EventTextDelta, TextDelta: "Health is unavailable."},
				llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonCompleted},
			)
		},
	}}
	agent := newTestAgent(t, model, session, 3)
	var finished Event
	if _, err := agent.RunTurn(t.Context(), nil, "health", func(event Event) error {
		if event.Type == EventToolFinished {
			finished = event
		}
		return nil
	}); err != nil {
		t.Fatalf("Ask() error: %v", err)
	}
	if !finished.ToolError {
		t.Fatalf("tool finished event did not report functional error: %#v", finished)
	}
}

// refusalsOf returns the tool results of the last tool message of a request,
// and the calls of the assistant message before it.
func refusalsOf(t *testing.T, request llm.Request) ([]llm.ToolCall, []llm.ToolResult) {
	t.Helper()
	last := len(request.Messages) - 1
	if last < 1 || request.Messages[last].Role != llm.RoleTool || request.Messages[last-1].Role != llm.RoleAssistant {
		t.Fatalf("request does not end with tool results: %#v", request.Messages)
	}
	return request.Messages[last-1].ToolCalls, request.Messages[last].ToolResults
}

func TestAskRefusesAnAnswerAskingTooManyToolsAtOnce(t *testing.T) {
	session := &fakeSession{tools: []*mcp.Tool{{Name: "get_cluster_health", InputSchema: objectSchema()}}}
	model := &scriptedLLM{t: t, steps: []llmStep{
		multipleToolCallStep("get_cluster_health", 12),
		func(request llm.Request, emit llm.EmitFunc) error {
			calls, results := refusalsOf(t, request)
			// The calls past the bound are not kept: the stored answer stays within it.
			if len(calls) != maxToolCallsPerTurn || len(results) != maxToolCallsPerTurn {
				t.Fatalf("kept %d calls and %d results, want %d", len(calls), len(results), maxToolCallsPerTurn)
			}
			for index, result := range results {
				if result.CallID != calls[index].ID || !result.IsError ||
					!strings.Contains(string(result.Content), "asked for 12 tools at once") ||
					!strings.Contains(string(result.Content), "at most 8 per answer, so none of them ran") {
					t.Fatalf("result %d = %#v", index, result)
				}
			}
			if request.NoToolCall {
				t.Fatal("the model may still ask for fewer tools")
			}
			return emitEvents(emit,
				llm.Event{Type: llm.EventTextDelta, TextDelta: "I will ask for fewer."},
				llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonCompleted},
			)
		},
	}}
	agent := newTestAgent(t, model, session, 3)
	started := 0
	var refused []RefusedCalls
	result, err := agent.RunTurn(t.Context(), nil, "health", func(event Event) error {
		switch event.Type {
		case EventToolStarted:
			started++
		case EventToolCallsRefused:
			refused = append(refused, *event.Refused)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("RunTurn() error = %v, want the refusal answered to the model", err)
	}
	if len(session.calls) != 0 || started != 0 {
		t.Fatalf("an answer past the bound ran tools: %#v, %d started", session.calls, started)
	}
	if len(refused) != 1 || refused[0] != (RefusedCalls{Bound: "answer", Asked: 12, Ran: 0}) {
		t.Fatalf("refused events = %#v", refused)
	}
	if text := result.Messages[len(result.Messages)-1].Text; text != "I will ask for fewer." {
		t.Fatalf("answer = %q", text)
	}
	if !session.closed {
		t.Fatal("MCP session was not closed")
	}
}

func TestAskEnforcesMaxIterationsBeforeCallingTool(t *testing.T) {
	session := &fakeSession{tools: []*mcp.Tool{{Name: "get_cluster_health", InputSchema: objectSchema()}}}
	model := &scriptedLLM{t: t, steps: []llmStep{toolCallStep("get_cluster_health", "call-1")}}
	agent := newTestAgent(t, model, session, 1)
	_, err := agent.RunTurn(t.Context(), nil, "health", func(Event) error { return nil })
	if err == nil || !strings.Contains(err.Error(), "maximum of 1 iterations") {
		t.Fatalf("Ask() error = %v", err)
	}
	if len(session.calls) != 0 {
		t.Fatalf("tool called without a remaining LLM iteration: %#v", session.calls)
	}
}

func TestAskRefusesTheCallsPastTheBoundOfTheQuestion(t *testing.T) {
	session := &fakeSession{tools: []*mcp.Tool{{Name: "get_cluster_health", InputSchema: objectSchema()}}}
	model := &scriptedLLM{t: t, steps: []llmStep{
		multipleToolCallStep("get_cluster_health", maxToolCallsPerTurn),
		multipleToolCallStep("get_cluster_health", 4),
		// 12 ran: 6 more pass the bound of 16, none of them runs.
		multipleToolCallStep("get_cluster_health", 6),
		func(request llm.Request, emit llm.EmitFunc) error {
			_, results := refusalsOf(t, request)
			for _, result := range results {
				if !result.IsError || !strings.Contains(string(result.Content), "already ran 12 tools") ||
					!strings.Contains(string(result.Content), "so 4 more at most can run and none of these 6 ran") {
					t.Fatalf("result = %#v", result)
				}
			}
			return multipleToolCallStep("get_cluster_health", 4)(request, emit)
		},
		// Every tool the question may run ran: the model answers.
		func(request llm.Request, emit llm.EmitFunc) error {
			if !request.NoToolCall {
				t.Fatal("the request after the last call of the question still takes calls")
			}
			return emitEvents(emit,
				llm.Event{Type: llm.EventTextDelta, TextDelta: "Here is what I read."},
				llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonCompleted},
			)
		},
	}}
	agent := newTestAgent(t, model, session, 8)
	var refused []RefusedCalls
	if _, err := agent.RunTurn(t.Context(), nil, "health", func(event Event) error {
		if event.Type == EventToolCallsRefused {
			refused = append(refused, *event.Refused)
		}
		return nil
	}); err != nil {
		t.Fatalf("RunTurn() error = %v", err)
	}
	if len(refused) != 1 || refused[0] != (RefusedCalls{Bound: "question", Asked: 6, Ran: 12}) {
		t.Fatalf("refused events = %#v", refused)
	}
	if len(session.calls) != maxToolCallsPerAsk {
		t.Fatalf("executed %d tools, want exactly %d", len(session.calls), maxToolCallsPerAsk)
	}
}

func TestCallLimitRefusal(t *testing.T) {
	for _, test := range []struct {
		asked, ran  int
		bound, want string
	}{
		{asked: 8, ran: 8},
		{asked: 1, ran: 15},
		{asked: 9, ran: 0, bound: "answer", want: "asked for 9 tools at once"},
		{asked: 2, ran: 15, bound: "question", want: "so 1 more at most can run"},
	} {
		bound, refusal := callLimitRefusal(test.asked, test.ran)
		if bound != test.bound {
			t.Errorf("callLimitRefusal(%d, %d) bound = %q, want %q", test.asked, test.ran, bound, test.bound)
		}
		if test.want == "" {
			if refusal != nil {
				t.Errorf("callLimitRefusal(%d, %d) = %s, want none", test.asked, test.ran, refusal)
			}
			continue
		}
		if !strings.Contains(string(refusal), test.want) {
			t.Errorf("callLimitRefusal(%d, %d) = %s, want containing %q", test.asked, test.ran, refusal, test.want)
		}
	}
}

func TestAskPropagatesMCPAndConsumerErrorsAndClosesSession(t *testing.T) {
	consumerError := errors.New("stop")
	for _, test := range []struct {
		name       string
		model      *scriptedLLM
		session    *fakeSession
		emit       EmitFunc
		wantTarget error
	}{
		{
			name:    "MCP transport",
			model:   &scriptedLLM{t: t, steps: []llmStep{toolCallStep("get_cluster_health", "call-1")}},
			session: &fakeSession{tools: []*mcp.Tool{{Name: "get_cluster_health", InputSchema: objectSchema()}}, callErr: errors.New("transport failed")},
			emit:    func(Event) error { return nil },
		},
		{
			name: "consumer",
			model: &scriptedLLM{t: t, steps: []llmStep{func(_ llm.Request, emit llm.EmitFunc) error {
				return emit(llm.Event{Type: llm.EventTextDelta, TextDelta: "partial"})
			}}},
			session:    &fakeSession{},
			emit:       func(Event) error { return consumerError },
			wantTarget: consumerError,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			agent := newTestAgent(t, test.model, test.session, 3)
			_, err := agent.RunTurn(t.Context(), nil, "health", test.emit)
			if err == nil {
				t.Fatal("Ask() succeeded, want error")
			}
			if test.wantTarget != nil && !errors.Is(err, test.wantTarget) {
				t.Fatalf("Ask() error = %v, want wrapping %v", err, test.wantTarget)
			}
			if !test.session.closed {
				t.Fatal("MCP session was not closed")
			}
		})
	}
}

func TestAskTimesOutAndClosesMCPSession(t *testing.T) {
	session := &fakeSession{}
	model := &deadlineLLM{}
	agent, err := New(llm.SingleModel(model, "test-model"), func(context.Context) (MCPSession, error) {
		return session, nil
	}, Config{SystemPrompt: testSystemPrompt, MaxIterations: 2, Timeout: 20 * time.Millisecond})
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	_, err = agent.RunTurn(t.Context(), nil, "health", func(Event) error { return nil })
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Ask() error = %v, want deadline exceeded", err)
	}
	if !session.closed {
		t.Fatal("MCP session was not closed after timeout")
	}
}

type llmStep func(llm.Request, llm.EmitFunc) error

type scriptedLLM struct {
	t              *testing.T
	steps          []llmStep
	inspectContext func(context.Context)
	position       int
}

func (c *scriptedLLM) Stream(ctx context.Context, request llm.Request, emit llm.EmitFunc) error {
	c.t.Helper()
	if c.inspectContext != nil {
		c.inspectContext(ctx)
	}
	if c.position >= len(c.steps) {
		c.t.Fatalf("unexpected LLM iteration %d", c.position+1)
	}
	step := c.steps[c.position]
	c.position++
	return step(request, emit)
}

type deadlineLLM struct{}

func (*deadlineLLM) Stream(ctx context.Context, _ llm.Request, _ llm.EmitFunc) error {
	<-ctx.Done()
	return ctx.Err()
}

type recordedCall struct {
	name      string
	arguments map[string]any
}

type fakeSession struct {
	tools            []*mcp.Tool
	results          map[string]*mcp.CallToolResult
	callErr          error
	closeErr         error
	calls            []recordedCall
	closed           bool
	connectedWithJWT bool
	// resources are the resources of the server, and texts their text by
	// URI.
	resources []*mcp.Resource
	texts     map[string]string
	reads     []string
	// skills are the skills the server lists.
	skills []Skill
}

func (s *fakeSession) ListSkills(context.Context) ([]Skill, error) {
	return s.skills, nil
}

func (s *fakeSession) ListResources(context.Context) ([]*mcp.Resource, error) {
	return s.resources, nil
}

func (s *fakeSession) ReadResource(_ context.Context, uri string) (*mcp.ReadResourceResult, error) {
	s.reads = append(s.reads, uri)
	text, ok := s.texts[uri]
	if !ok {
		return nil, fmt.Errorf("resource %s not found", uri)
	}
	return &mcp.ReadResourceResult{Contents: []*mcp.ResourceContents{{URI: uri, MIMEType: "text/markdown", Text: text}}}, nil
}

// ListTools announces a tool declared without annotations as read-only, as a
// diagnostic tool is; the confirmation tests declare their annotations.
func (s *fakeSession) ServerInfo() (string, string) { return "test-mcp", "v9.9.9" }

func (s *fakeSession) ListTools(context.Context) ([]*mcp.Tool, error) {
	tools := make([]*mcp.Tool, 0, len(s.tools))
	for _, tool := range s.tools {
		if tool != nil && tool.Annotations == nil {
			copy := *tool
			copy.Annotations = &mcp.ToolAnnotations{ReadOnlyHint: true}
			tool = &copy
		}
		tools = append(tools, tool)
	}
	return tools, nil
}

func (s *fakeSession) CallTool(_ context.Context, name string, arguments map[string]any) (*mcp.CallToolResult, error) {
	s.calls = append(s.calls, recordedCall{name: name, arguments: arguments})
	if s.callErr != nil {
		return nil, s.callErr
	}
	if result := s.results[name]; result != nil {
		return result, nil
	}
	return &mcp.CallToolResult{}, nil
}

func (s *fakeSession) Close() error {
	s.closed = true
	return s.closeErr
}

func newTestAgent(t *testing.T, model llm.Client, session *fakeSession, maxIterations int) *Agent {
	t.Helper()
	agent, err := New(llm.SingleModel(model, "test-model"), func(ctx context.Context) (MCPSession, error) {
		if token, ok := auth.BearerTokenFromContext(ctx); ok && token == "jwt-marker" {
			session.connectedWithJWT = true
		}
		return session, nil
	}, Config{SystemPrompt: testSystemPrompt, MaxIterations: maxIterations, Timeout: time.Minute})
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	return agent
}

func objectSchema() map[string]any {
	return map[string]any{"type": "object", "properties": map[string]any{}}
}

func toolCallStep(name string, callID string) llmStep {
	return func(_ llm.Request, emit llm.EmitFunc) error {
		return emitEvents(emit,
			llm.Event{Type: llm.EventToolCall, ToolCall: &llm.ToolCall{ID: callID, Name: name, Arguments: json.RawMessage(`{}`)}},
			llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonToolCalls},
		)
	}
}

func multipleToolCallStep(name string, count int) llmStep {
	return func(_ llm.Request, emit llm.EmitFunc) error {
		for index := 0; index < count; index++ {
			call := &llm.ToolCall{ID: fmt.Sprintf("call-%d", index), Name: name, Arguments: json.RawMessage(`{}`)}
			if err := emit(llm.Event{Type: llm.EventToolCall, ToolCall: call}); err != nil {
				return err
			}
		}
		return emit(llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonToolCalls})
	}
}

func emitEvents(emit llm.EmitFunc, events ...llm.Event) error {
	for _, event := range events {
		if err := emit(event); err != nil {
			return err
		}
	}
	return nil
}

func TestNewRequiresASystemPrompt(t *testing.T) {
	connect := func(context.Context) (MCPSession, error) { return nil, nil }
	if _, err := New(llm.SingleModel(&scriptedLLM{}, "test-model"), connect, Config{SystemPrompt: " ", MaxIterations: 2, Timeout: time.Minute}); err == nil {
		t.Fatal("New() accepted an empty system prompt")
	}
}

// The last iteration asks for an answer without tool call: a turn that
// needs more calls answers with what it has rather than fail.
func TestTheLastIterationAsksForAnAnswer(t *testing.T) {
	session := &fakeSession{tools: []*mcp.Tool{{Name: "get_cluster_health", InputSchema: objectSchema()}}}
	var last llm.Request
	model := &scriptedLLM{t: t, steps: []llmStep{
		func(request llm.Request, emit llm.EmitFunc) error {
			if request.NoToolCall {
				t.Fatal("the first iteration forbade tool calls")
			}
			return toolCallStep("get_cluster_health", "call-1")(request, emit)
		},
		answerStep(t, func(request llm.Request) { last = request }),
	}}
	agent := newTestAgent(t, model, session, 2)
	result, err := agent.RunTurn(t.Context(), nil, "health", func(Event) error { return nil })
	if err != nil || result.FinishReason != llm.FinishReasonCompleted || !last.NoToolCall || len(last.Tools) == 0 || len(session.calls) != 1 {
		t.Fatalf("result = %+v, %v, last request no tool call %v with %d tools", result, err, last.NoToolCall, len(last.Tools))
	}
}

func TestAskTimesEachModelCall(t *testing.T) {
	session := &fakeSession{tools: []*mcp.Tool{{Name: "get_cluster_health", InputSchema: objectSchema()}}}
	model := &scriptedLLM{t: t, steps: []llmStep{func(_ llm.Request, emit llm.EmitFunc) error {
		// The provider waits before its first event, then streams.
		time.Sleep(30 * time.Millisecond)
		if err := emit(llm.Event{Type: llm.EventTextDelta, TextDelta: "Healthy."}); err != nil {
			return err
		}
		time.Sleep(20 * time.Millisecond)
		return emitEvents(emit,
			llm.Event{Type: llm.EventUsage, Usage: &llm.Usage{InputTokens: 10, OutputTokens: 2, TotalTokens: 12}},
			llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonCompleted},
		)
	}}}
	var usage Event
	if _, err := newTestAgent(t, model, session, 3).RunTurn(t.Context(), nil, "health", func(event Event) error {
		if event.Type == EventUsage {
			usage = event
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if usage.ModelFirstEvent < 30*time.Millisecond || usage.ModelDuration < 50*time.Millisecond || usage.ModelDuration < usage.ModelFirstEvent {
		t.Fatalf("first event %v, duration %v", usage.ModelFirstEvent, usage.ModelDuration)
	}
}
