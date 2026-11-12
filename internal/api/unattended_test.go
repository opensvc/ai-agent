package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/llm"
)

type runnerFunc func(context.Context, []llm.Message, string, agent.EmitFunc) (agent.TurnResult, error)

func (f runnerFunc) RunTurn(ctx context.Context, history []llm.Message, prompt string, emit agent.EmitFunc) (agent.TurnResult, error) {
	return f(ctx, history, prompt, emit)
}

func newUnattendedHandler(t *testing.T, runner UnattendedRunner, maxConcurrentAsks int) http.Handler {
	t.Helper()
	handler, err := NewHandler([]Backend{{Name: "om3", Conversations: turnService{}, Authenticator: allowTestTokenVerifier(), Runner: runner}},
		HandlerConfig{MaxConcurrentAsks: maxConcurrentAsks, AuditLogger: discardAuditLogger()})
	if err != nil {
		t.Fatal(err)
	}
	return handler
}

func unattendedRequest(body string) *http.Request {
	request := httptest.NewRequest(http.MethodPost, "/v1/om3/runs", strings.NewReader(body))
	request.Header.Set("Authorization", "Bearer token")
	request.Header.Set(testBackendHeader, "cluster-id")
	request.Header.Set("Content-Type", "application/json")
	return request
}

const unattendedBody = `{"threadId":"schedule-3","runId":"run-42","messages":[{"id":"m1","role":"user","content":"Use the skill morning-check."}]}`

// sseEvents decodes the data lines of a stream.
func sseEvents(t *testing.T, body string) []map[string]any {
	t.Helper()
	var events []map[string]any
	for _, line := range strings.Split(body, "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			t.Fatalf("event %q: %v", data, err)
		}
		events = append(events, event)
	}
	return events
}

// An unattended run runs the prompt on no history, marked unattended, and
// finishes with the summary of its calls: their names, durations and
// errors, never their arguments nor their results.
func TestUnattendedRunStreamsItsSummary(t *testing.T) {
	start := time.Date(2026, 11, 12, 8, 0, 0, 0, time.UTC)
	runner := runnerFunc(func(ctx context.Context, history []llm.Message, prompt string, emit agent.EmitFunc) (agent.TurnResult, error) {
		if !agent.Unattended(ctx) || history != nil || prompt != "Use the skill morning-check." {
			t.Errorf("unattended %v, history %v, prompt %q", agent.Unattended(ctx), history, prompt)
		}
		for _, event := range []agent.Event{
			{Type: agent.EventToolStarted, ToolCallID: "c1", ToolName: "list_alerts", ToolArguments: json.RawMessage(`{"secret":"argument"}`), At: start},
			{Type: agent.EventToolFinished, ToolCallID: "c1", ToolName: "list_alerts", ToolResult: json.RawMessage(`{"content":[{"type":"text","text":"result"}]}`), At: start.Add(1500 * time.Millisecond)},
			{Type: agent.EventToolStarted, ToolCallID: "c2", ToolName: "attach_node_tag", At: start.Add(2 * time.Second)},
			{Type: agent.EventToolFinished, ToolCallID: "c2", ToolName: "attach_node_tag", ToolError: true, NotRun: agent.NotRunUnoffered, At: start.Add(2 * time.Second)},
			{Type: agent.EventToolCallsRefused, Refused: &agent.RefusedCalls{Bound: "question", Asked: 3, Ran: 14}},
			{Type: agent.EventTextDelta, TextDelta: "# Report"},
			{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted},
		} {
			if err := emit(event); err != nil {
				return agent.TurnResult{}, err
			}
		}
		return agent.TurnResult{FinishReason: llm.FinishReasonCompleted}, nil
	})
	response := httptest.NewRecorder()
	newUnattendedHandler(t, runner, 4).ServeHTTP(response, unattendedRequest(unattendedBody))
	if response.Code != http.StatusOK {
		t.Fatalf("status %d: %s", response.Code, response.Body.String())
	}
	events := sseEvents(t, response.Body.String())
	last := events[len(events)-1]
	if last["type"] != "RUN_FINISHED" || last["threadId"] != "schedule-3" || last["runId"] != "run-42" {
		t.Fatalf("last event %v", last)
	}
	result, _ := json.Marshal(last["result"])
	var summary UnattendedSummary
	if err := json.Unmarshal(result, &summary); err != nil {
		t.Fatal(err)
	}
	want := UnattendedSummary{Unattended: true, FinishReason: "completed", ToolCallCount: 2, RefusedCalls: 3, ToolCalls: []UnattendedToolCall{
		{ID: "c1", Name: "list_alerts", DurationMs: 1500}, {ID: "c2", Name: "attach_node_tag", Error: true, NotRun: "unoffered"},
	}}
	// The write asked for is told apart from a tool that failed, and from
	// the calls the bounds refused.
	if summary.Unattended != want.Unattended || summary.FinishReason != want.FinishReason || summary.ToolCallCount != 2 || summary.RefusedCalls != 3 ||
		summary.NotRunCalls != 1 ||
		len(summary.ToolCalls) != 2 || summary.ToolCalls[0] != want.ToolCalls[0] || summary.ToolCalls[1] != want.ToolCalls[1] {
		t.Fatalf("summary %+v", summary)
	}
	if strings.Contains(string(result), "argument") || strings.Contains(string(result), "result\"") || strings.Contains(string(result), "turnId") {
		t.Fatalf("the summary says more than identifiers: %s", result)
	}
}

func TestUnattendedRunRefusesWhatItCannotServe(t *testing.T) {
	runner := runnerFunc(func(context.Context, []llm.Message, string, agent.EmitFunc) (agent.TurnResult, error) {
		t.Error("the runner ran")
		return agent.TurnResult{}, nil
	})
	for name, c := range map[string]struct {
		body string
		code string
	}{
		"tools of the client": {`{"threadId":"t","runId":"r","messages":[{"role":"user","content":"x"}],"tools":[{"name":"open_page","parameters":{"type":"object"}}]}`, "invalid_tools"},
		"a resume":            {`{"threadId":"t","runId":"r","messages":[{"role":"user","content":"x"}],"resume":[{"interruptId":"i","status":"cancelled"}]}`, "invalid_decision"},
		"a history":           {`{"threadId":"t","runId":"r","messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"user","content":"c"}]}`, "invalid_prompt"},
		"no message":          {`{"threadId":"t","runId":"r","messages":[]}`, "invalid_prompt"},
		"not the user":        {`{"threadId":"t","runId":"r","messages":[{"role":"assistant","content":"x"}]}`, "invalid_prompt"},
		"an empty prompt":     {`{"threadId":"t","runId":"r","messages":[{"role":"user","content":"  "}]}`, "invalid_prompt"},
		"no run ID":           {`{"threadId":"t","messages":[{"role":"user","content":"x"}]}`, "invalid_run_id"},
		"no thread ID":        {`{"runId":"r","messages":[{"role":"user","content":"x"}]}`, "invalid_thread_id"},
		"two objects":         {`{"threadId":"t","runId":"r","messages":[{"role":"user","content":"x"}]}{}`, "invalid_request"},
	} {
		response := httptest.NewRecorder()
		newUnattendedHandler(t, runner, 4).ServeHTTP(response, unattendedRequest(c.body))
		if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), `"`+c.code+`"`) {
			t.Errorf("%s: %d %s", name, response.Code, response.Body.String())
		}
	}
	response := httptest.NewRecorder()
	newUnattendedHandler(t, nil, 4).ServeHTTP(response, unattendedRequest(unattendedBody))
	if response.Code != http.StatusNotFound || !strings.Contains(response.Body.String(), "unattended_unavailable") {
		t.Errorf("no runner: %d %s", response.Code, response.Body.String())
	}
}

// An unattended run takes a slot of the user as a conversation run does:
// it is no way around the bounds of the instance.
func TestUnattendedRunTakesASlot(t *testing.T) {
	running, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	runner := runnerFunc(func(_ context.Context, _ []llm.Message, _ string, emit agent.EmitFunc) (agent.TurnResult, error) {
		once.Do(func() { close(running) })
		<-release
		return agent.TurnResult{}, emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted})
	})
	handler := newUnattendedHandler(t, runner, 1)
	done := make(chan struct{})
	go func() {
		defer close(done)
		handler.ServeHTTP(httptest.NewRecorder(), unattendedRequest(unattendedBody))
	}()
	<-running
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, unattendedRequest(unattendedBody))
	close(release)
	<-done
	if response.Code != http.StatusTooManyRequests {
		t.Fatalf("second run: %d %s", response.Code, response.Body.String())
	}
}

// A runner that ends waiting on somebody fails the run: an unattended run
// never waits.
func TestUnattendedRunFailsASuspendedTurn(t *testing.T) {
	runner := runnerFunc(func(context.Context, []llm.Message, string, agent.EmitFunc) (agent.TurnResult, error) {
		return agent.TurnResult{Suspended: &agent.SuspendedTurn{}}, nil
	})
	response := httptest.NewRecorder()
	newUnattendedHandler(t, runner, 4).ServeHTTP(response, unattendedRequest(unattendedBody))
	events := sseEvents(t, response.Body.String())
	if last := events[len(events)-1]; last["type"] != "RUN_ERROR" || last["code"] != "agent_failed" {
		t.Fatalf("last event %v", last)
	}
}
