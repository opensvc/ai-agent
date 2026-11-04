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
	"time"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/conversation"
	"github.com/opensvc/ai-agent/internal/llm"
)

func TestAuditRecordsTurnLifecycleWithoutSensitiveData(t *testing.T) {
	const (
		tokenMarker  = "sensitive-jwt-marker"
		promptMarker = "sensitive-prompt-marker"
		textMarker   = "sensitive-model-text-marker"
		callerID     = "caller-controlled-request-id"
	)
	var output bytes.Buffer
	handler := newAuditTestHandler(t, &output, turnFunc(func(_ context.Context, prompt string, emit agent.EmitFunc) error {
		if prompt != promptMarker {
			t.Fatalf("prompt = %q", prompt)
		}
		for _, event := range []agent.Event{
			{Type: agent.EventTextDelta, TextDelta: textMarker, Iteration: 1},
			{Type: agent.EventToolStarted, ToolName: "get_cluster_health", Iteration: 1, At: time.UnixMilli(1_760_000_000_100)},
			{Type: agent.EventToolFinished, ToolName: "get_cluster_health", ToolError: true, Iteration: 1, At: time.UnixMilli(1_760_000_000_340)},
			{Type: agent.EventUsage, Usage: &llm.Usage{InputTokens: 10, OutputTokens: 2, TotalTokens: 12}, Iteration: 1, ModelDuration: 1500 * time.Millisecond, ModelFirstEvent: 900 * time.Millisecond},
			{Type: agent.EventToolCallsRefused, Refused: &agent.RefusedCalls{Bound: "answer", Asked: 12, Ran: 1}, Iteration: 2},
			{Type: agent.EventAgentToolCalled, AgentCall: &agent.AgentCall{Tool: "get_cluster_status", Purpose: "documentation_version", Error: true, Duration: 80 * time.Millisecond}, Iteration: 2},
			{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, Iteration: 2},
		} {
			if err := emit(event); err != nil {
				return err
			}
		}
		return nil
	}), tokenVerifierFunc(func(context.Context, string) (auth.Identity, error) {
		return auth.Identity{ClusterID: "cluster-id", Subject: "alice\noperator", Issuer: "node-a"}, nil
	}))

	request := httptest.NewRequest(http.MethodPost, "/v1/om3/conversations/conversation-1/runs", strings.NewReader(runBody(promptMarker)))
	request.Header.Set("Authorization", "Bearer "+tokenMarker)
	request.Header.Set(testBackendHeader, "cluster-id")
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(requestIDHeader, callerID)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)

	requestID := response.Header().Get(requestIDHeader)
	if len(requestID) != requestIDBytes*2 || requestID == callerID {
		t.Fatalf("server request ID = %q", requestID)
	}
	for _, marker := range []string{tokenMarker, promptMarker, textMarker, callerID} {
		if strings.Contains(output.String(), marker) {
			t.Fatalf("audit log exposes marker %q: %s", marker, output.String())
		}
	}

	events := decodeAuditEvents(t, output.String())
	wantEvents := []string{"conversation_run_started", "tool_started", "tool_finished", "llm_usage", "tool_calls_refused", "agent_tool_called", "conversation_run_completed"}
	if len(events) != len(wantEvents) {
		t.Fatalf("audit events = %#v", events)
	}
	for index, want := range wantEvents {
		if events[index]["event"] != want {
			t.Fatalf("audit event %d = %v, want %q", index, events[index]["event"], want)
		}
		if events[index]["request_id"] != requestID {
			t.Fatalf("audit request ID = %v, want %q", events[index]["request_id"], requestID)
		}
		if events[index]["backend"] != "om3" || events[index]["subject"] != "alice operator" || events[index]["issuer"] != "node-a" {
			t.Fatalf("audit identity = %#v", events[index])
		}
	}
	// The duration of a tool, on the times the agent stamped its events with.
	if events[2]["tool_name"] != "get_cluster_health" || events[2]["tool_error"] != true || events[2]["duration_ms"] != float64(240) {
		t.Fatalf("tool audit = %#v", events[2])
	}
	if events[3]["total_tokens"] != float64(12) || events[3]["duration_ms"] != float64(1500) || events[3]["first_event_ms"] != float64(900) {
		t.Fatalf("usage audit = %#v", events[3])
	}
	if refused := events[4]; refused["bound"] != "answer" || refused["calls_asked"] != float64(12) || refused["calls_ran"] != float64(1) || refused["iteration"] != float64(2) {
		t.Fatalf("refused calls audit = %#v", refused)
	}
	// A call the agent made of its own is audited apart, not counted as a
	// call of the model.
	if called := events[5]; called["tool_name"] != "get_cluster_status" || called["purpose"] != "documentation_version" || called["tool_error"] != true ||
		called["code"] != "tool_error" || called["duration_ms"] != float64(80) || called["iteration"] != float64(2) {
		t.Fatalf("agent tool call audit = %#v", called)
	}
	completed := events[6]
	if completed["iterations"] != float64(2) || completed["tool_calls"] != float64(1) || completed["total_tokens"] != float64(12) {
		t.Fatalf("completed audit = %#v", completed)
	}
}

func TestAuditRecordsAuthenticationRejectionWithoutTokenOrVerifierError(t *testing.T) {
	const (
		tokenMarker = "rejected-jwt-marker"
		errorMarker = "sensitive-verifier-error"
	)
	var output bytes.Buffer
	handler := newAuditTestHandler(t, &output, turnFunc(func(context.Context, string, agent.EmitFunc) error {
		t.Fatal("agent was called")
		return nil
	}), tokenVerifierFunc(func(context.Context, string) (auth.Identity, error) {
		return auth.Identity{ClusterID: "cluster-id"}, errors.New(errorMarker)
	}))
	request := httptest.NewRequest(http.MethodPost, "/v1/om3/conversations/conversation-1/runs", nil)
	request.Header.Set("Authorization", "Bearer "+tokenMarker)
	request.Header.Set(testBackendHeader, "cluster-id")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)

	if response.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", response.Code)
	}
	events := decodeAuditEvents(t, output.String())
	if len(events) != 1 || events[0]["event"] != "auth_rejected" || events[0]["code"] != "unauthorized" {
		t.Fatalf("audit events = %#v", events)
	}
	if events[0]["request_id"] != response.Header().Get(requestIDHeader) {
		t.Fatalf("audit request ID = %v", events[0]["request_id"])
	}
	for _, marker := range []string{tokenMarker, errorMarker} {
		if strings.Contains(output.String(), marker) {
			t.Fatalf("audit log exposes marker %q: %s", marker, output.String())
		}
	}
}

func TestAuditRecordsStableAgentFailureCodeWithoutRawError(t *testing.T) {
	const errorMarker = "sensitive-provider-error"
	var output bytes.Buffer
	handler := newAuditTestHandler(t, &output, turnFunc(func(context.Context, string, agent.EmitFunc) error {
		return errors.New(errorMarker)
	}), allowTestTokenVerifier())
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, newAuthenticatedRunRequest(runBody("health")))

	events := decodeAuditEvents(t, output.String())
	if len(events) != 2 || events[1]["event"] != "conversation_run_failed" || events[1]["code"] != "agent_failed" {
		t.Fatalf("audit events = %#v", events)
	}
	if strings.Contains(output.String(), errorMarker) {
		t.Fatalf("audit log exposes raw agent error: %s", output.String())
	}
}

func TestAuditRecordsValidatedRequestRejection(t *testing.T) {
	var output bytes.Buffer
	handler := newAuditTestHandler(t, &output, turnFunc(func(context.Context, string, agent.EmitFunc) error {
		t.Fatal("agent was called")
		return nil
	}), allowTestTokenVerifier())
	request := httptest.NewRequest(http.MethodPost, "/v1/om3/conversations/conversation-1/runs", strings.NewReader(runBody("health")))
	request.Header.Set("Authorization", "Bearer token")
	request.Header.Set(testBackendHeader, "cluster-id")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)

	events := decodeAuditEvents(t, output.String())
	if len(events) != 1 || events[0]["event"] != "conversation_run_rejected" || events[0]["code"] != "unsupported_media_type" {
		t.Fatalf("audit events = %#v", events)
	}
	if events[0]["subject"] != "test-user" || events[0]["status"] != float64(http.StatusUnsupportedMediaType) {
		t.Fatalf("rejection audit = %#v", events[0])
	}
}

func TestAuditDoesNotRecordConversationTitle(t *testing.T) {
	const titleMarker = "sensitive-conversation-title-marker"
	var output bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&output, nil))
	service := conversationServiceFuncs{update: func(_ context.Context, _ auth.Identity, id string, title string) (conversation.Conversation, error) {
		if id != "conversation-id" || title != titleMarker {
			t.Fatalf("unexpected title update id=%q title=%q", id, title)
		}
		return conversation.Conversation{ID: id, Title: titleMarker}, nil
	}}
	handler, err := NewHandler([]Backend{{Name: "om3", Conversations: service, Authenticator: allowTestTokenVerifier()}}, HandlerConfig{MaxConcurrentAsks: 4, AuditLogger: logger})
	if err != nil {
		t.Fatalf("create audited API handler: %v", err)
	}
	request := authenticatedRequest(http.MethodPatch, "/v1/om3/conversations/conversation-id", `{"title":"`+titleMarker+`"}`)
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	if strings.Contains(output.String(), titleMarker) {
		t.Fatalf("audit log exposes title: %s", output.String())
	}
	events := decodeAuditEvents(t, output.String())
	if len(events) != 1 || events[0]["event"] != "conversation_title_updated" || events[0]["conversation_id"] != "conversation-id" {
		t.Fatalf("audit events = %#v", events)
	}
}

func TestStreamFailureCode(t *testing.T) {
	for _, test := range []struct {
		name       string
		err        error
		completed  bool
		writeError bool
		contextErr error
		want       string
	}{
		{name: "completed", completed: true},
		{name: "stream write", writeError: true, want: "stream_write_failed"},
		{name: "deadline error", err: context.DeadlineExceeded, want: "request_timeout"},
		{name: "deadline context", contextErr: context.DeadlineExceeded, want: "request_timeout"},
		{name: "canceled error", err: context.Canceled, want: "request_canceled"},
		{name: "canceled context", contextErr: context.Canceled, want: "request_canceled"},
		{name: "cleanup", err: errors.New("close"), completed: true, want: "agent_cleanup_failed"},
		{name: "agent", err: errors.New("failed"), want: "agent_failed"},
		{name: "incomplete", want: "agent_incomplete"},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := streamFailureCode(test.err, test.completed, test.writeError, test.contextErr); got != test.want {
				t.Fatalf("streamFailureCode() = %q, want %q", got, test.want)
			}
		})
	}
}

func TestBoundedAuditIdentityNormalizesAndTruncates(t *testing.T) {
	value := " alice\n\t" + strings.Repeat("é", maxAuditIdentityRunes+1)
	got := boundedAuditIdentity(value)
	if strings.ContainsAny(got, "\n\t") || !strings.HasPrefix(got, "alice ") || !strings.HasSuffix(got, "…") {
		t.Fatalf("bounded identity = %q", got)
	}
}

func newAuditTestHandler(t *testing.T, output *bytes.Buffer, asker turnFunc, verifier Authenticator) http.Handler {
	t.Helper()
	logger := slog.New(slog.NewJSONHandler(output, nil))
	handler, err := NewHandler([]Backend{{Name: "om3", Conversations: turnService{run: asker}, Authenticator: verifier}}, HandlerConfig{MaxConcurrentAsks: 4, AuditLogger: logger})
	if err != nil {
		t.Fatalf("create audited API handler: %v", err)
	}
	return handler
}

func decodeAuditEvents(t *testing.T, output string) []map[string]any {
	t.Helper()
	lines := strings.Split(strings.TrimSpace(output), "\n")
	events := make([]map[string]any, 0, len(lines))
	for _, line := range lines {
		if line == "" {
			continue
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			t.Fatalf("decode audit log %q: %v", line, err)
		}
		events = append(events, event)
	}
	return events
}
