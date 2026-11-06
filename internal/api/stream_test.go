package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/agui"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/conversation"
	"github.com/opensvc/ai-agent/internal/llm"
)

type tokenVerifierFunc func(context.Context, string) (auth.Identity, error)

type readTrackingBody struct {
	reader io.Reader
	read   bool
}

func (b *readTrackingBody) Read(data []byte) (int, error) {
	b.read = true
	return b.reader.Read(data)
}

func (*readTrackingBody) Close() error { return nil }

// testBackendHeader is the header of the test backend: the API allows and
// removes it as a backend declares it.
const testBackendHeader = "X-Test-Backend"

func (f tokenVerifierFunc) Headers() []string {
	return []string{testBackendHeader}
}

func (f tokenVerifierFunc) Authenticate(ctx context.Context, _ http.Header, token string) (context.Context, auth.Identity, error) {
	identity, err := f(ctx, token)
	return ctx, identity, err
}

// runBody is the RunAgentInput of a run of testConversationID on prompt.
func runBody(prompt string) string { return runInput(testConversationID, prompt) }

// runInput is the RunAgentInput of a run of the conversation id on prompt.
func runInput(id, prompt string) string {
	data, _ := json.Marshal(prompt)
	return `{"threadId":"` + id + `","runId":"run-1","messages":[{"id":"u-1","role":"user","content":` + string(data) + `}]}`
}

func TestRunStreamsAgentEventsWithDelegatedJWT(t *testing.T) {
	const token = "test-jwt-marker"
	var called bool
	handler := newTestHandler(t, turnFunc(func(ctx context.Context, prompt string, emit agent.EmitFunc) error {
		called = true
		if got, ok := auth.BearerTokenFromContext(ctx); !ok || got != token {
			t.Fatalf("delegated JWT = %q, %v", got, ok)
		}
		if identity, ok := auth.IdentityFromContext(ctx); !ok || identity.Subject != "test-user" {
			t.Fatalf("verified identity = %+v, %v", identity, ok)
		}
		if prompt != "health of my cluster" {
			t.Fatalf("got prompt %q", prompt)
		}
		for _, event := range []agent.Event{
			{Type: agent.EventTextDelta, TextDelta: "checking", Iteration: 1},
			{Type: agent.EventToolStarted, ToolName: "get_cluster_health", ToolCallID: "call-1", Iteration: 1},
			{Type: agent.EventToolFinished, ToolName: "get_cluster_health", ToolCallID: "call-1", Iteration: 1},
			{Type: agent.EventUsage, Usage: &llm.Usage{InputTokens: 10, OutputTokens: 2, TotalTokens: 12}, Iteration: 1},
			{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, TurnID: "turn-1", Iteration: 2},
		} {
			if err := emit(event); err != nil {
				return err
			}
		}
		return nil
	}))

	request := httptest.NewRequest(http.MethodPost, "/v1/om3/conversations/conversation-1/runs", strings.NewReader(runBody("health of my cluster")))
	request.Header.Set("Authorization", "Bearer "+token)
	request.Header.Set(testBackendHeader, "cluster-id")
	request.Header.Set("Content-Type", "application/json; charset=utf-8")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)

	if !called {
		t.Fatal("agent was not called")
	}
	if response.Code != http.StatusOK {
		t.Fatalf("got status %d, want %d", response.Code, http.StatusOK)
	}
	if got := response.Header().Get("Content-Type"); got != "text/event-stream; charset=utf-8" {
		t.Fatalf("got Content-Type %q", got)
	}
	events := aguiEvents(t, response.Body.String())
	if eventTypes(events) != "RUN_STARTED,TEXT_MESSAGE_START,TEXT_MESSAGE_CONTENT,TEXT_MESSAGE_END,TOOL_CALL_START,TOOL_CALL_ARGS,TOOL_CALL_END,TOOL_CALL_RESULT,RUN_FINISHED" {
		t.Fatalf("got events %s", response.Body.String())
	}
	finished := events[len(events)-1]
	if events[2]["delta"] != "checking" || events[4]["toolCallName"] != "get_cluster_health" ||
		finished["usage"].([]any)[0].(map[string]any)["totalTokens"] != float64(12) || finished["result"].(map[string]any)["turnId"] != "turn-1" {
		t.Fatalf("got events %s", response.Body.String())
	}
	if strings.Contains(response.Body.String(), token) {
		t.Fatal("response body exposes delegated JWT")
	}
}

func TestRunRejectsTokenRejectedByVerifier(t *testing.T) {
	const token = "forged-jwt-marker"
	var agentCalls atomic.Int64
	verifier := tokenVerifierFunc(func(_ context.Context, got string) (auth.Identity, error) {
		if got != token {
			t.Fatalf("verifier got token %q", got)
		}
		return auth.Identity{ClusterID: "cluster-id"}, auth.ErrInvalidToken
	})
	handler := newTestHandlerWithVerifier(t, turnFunc(func(context.Context, string, agent.EmitFunc) error {
		agentCalls.Add(1)
		return nil
	}), verifier)
	request := httptest.NewRequest(http.MethodPost, "/v1/om3/conversations/conversation-1/runs", nil)
	body := &readTrackingBody{reader: strings.NewReader(runBody("health"))}
	request.Body = body
	request.Header.Set("Authorization", "Bearer "+token)
	request.Header.Set(testBackendHeader, "cluster-id")
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)

	if response.Code != http.StatusUnauthorized {
		t.Fatalf("got status %d, want 401", response.Code)
	}
	if response.Header().Get("WWW-Authenticate") != "Bearer" {
		t.Fatalf("got WWW-Authenticate %q", response.Header().Get("WWW-Authenticate"))
	}
	if agentCalls.Load() != 0 {
		t.Fatalf("agent called %d times", agentCalls.Load())
	}
	if body.read {
		t.Fatal("request body was read before JWT authentication")
	}
	if strings.Contains(response.Body.String(), token) {
		t.Fatal("unauthorized response exposes JWT")
	}
}

func TestRunRejectsInvalidRequestsBeforeCallingAgent(t *testing.T) {
	for _, test := range []struct {
		name          string
		authorization string
		contentType   string
		body          string
		wantStatus    int
		wantCode      string
	}{
		{name: "missing authorization", contentType: "application/json", body: runBody("health"), wantStatus: http.StatusUnauthorized, wantCode: "unauthorized"},
		{name: "wrong scheme", authorization: "Basic value", contentType: "application/json", body: runBody("health"), wantStatus: http.StatusUnauthorized, wantCode: "unauthorized"},
		{name: "missing content type", authorization: "Bearer token", body: runBody("health"), wantStatus: http.StatusUnsupportedMediaType, wantCode: "unsupported_media_type"},
		{name: "invalid JSON", authorization: "Bearer token", contentType: "application/json", body: `{`, wantStatus: http.StatusBadRequest, wantCode: "invalid_request"},
		{name: "multiple values", authorization: "Bearer token", contentType: "application/json", body: runBody("health") + `{}`, wantStatus: http.StatusBadRequest, wantCode: "invalid_request"},
		{name: "empty prompt", authorization: "Bearer token", contentType: "application/json", body: runBody("  "), wantStatus: http.StatusBadRequest, wantCode: "invalid_prompt"},
		{name: "large prompt", authorization: "Bearer token", contentType: "application/json", body: runBody(strings.Repeat("x", maxPromptBytes+1)), wantStatus: http.StatusRequestEntityTooLarge, wantCode: "prompt_too_large"},
		{name: "large request", authorization: "Bearer token", contentType: "application/json", body: `{"threadId":"conversation-1","runId":"run-1","state":"` + strings.Repeat("x", maxRunRequestBytes) + `"}`, wantStatus: http.StatusRequestEntityTooLarge, wantCode: "request_too_large"},
	} {
		t.Run(test.name, func(t *testing.T) {
			var calls atomic.Int64
			handler := newTestHandler(t, turnFunc(func(context.Context, string, agent.EmitFunc) error {
				calls.Add(1)
				return nil
			}))
			request := httptest.NewRequest(http.MethodPost, "/v1/om3/conversations/conversation-1/runs", strings.NewReader(test.body))
			if test.authorization != "" {
				request.Header.Set("Authorization", test.authorization)
				request.Header.Set(testBackendHeader, "cluster-id")
			}
			if test.contentType != "" {
				request.Header.Set("Content-Type", test.contentType)
			}
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)

			if response.Code != test.wantStatus {
				t.Fatalf("got status %d, want %d; body=%s", response.Code, test.wantStatus, response.Body.String())
			}
			var errorResponse ErrorResponse
			if err := json.NewDecoder(response.Body).Decode(&errorResponse); err != nil {
				t.Fatalf("decode error response: %v", err)
			}
			if errorResponse.Error.Code != test.wantCode {
				t.Fatalf("got error code %q, want %q", errorResponse.Error.Code, test.wantCode)
			}
			if calls.Load() != 0 {
				t.Fatalf("agent called %d times", calls.Load())
			}
		})
	}
}

// runErrorOf runs a turn that fails with err and gives the stream.
func runErrorOf(t *testing.T, err error) (string, []map[string]any) {
	t.Helper()
	handler := newTestHandler(t, turnFunc(func(context.Context, string, agent.EmitFunc) error { return err }))
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, newAuthenticatedRunRequest(runBody("health")))
	if response.Code != http.StatusOK {
		t.Fatalf("got status %d, want streaming status 200", response.Code)
	}
	return response.Body.String(), aguiEvents(t, response.Body.String())
}

func TestRunStreamsGenericRuntimeError(t *testing.T) {
	const sensitiveDetail = "provider failed with secret detail"
	body, events := runErrorOf(t, errors.New(sensitiveDetail))
	if eventTypes(events) != "RUN_STARTED,RUN_ERROR" || events[1]["code"] != "agent_failed" {
		t.Fatalf("unexpected error events %s", body)
	}
	if strings.Contains(body, sensitiveDetail) {
		t.Fatal("stream exposes internal agent error")
	}
}

func TestRunStreamsTimeoutError(t *testing.T) {
	body, events := runErrorOf(t, context.DeadlineExceeded)
	if eventTypes(events) != "RUN_STARTED,RUN_ERROR" || events[1]["code"] != "request_timeout" {
		t.Fatalf("unexpected timeout events %s", body)
	}
}

func TestRunSetsAndClearsWriteDeadline(t *testing.T) {
	response := &writeDeadlineRecorder{ResponseRecorder: httptest.NewRecorder()}
	handler := newTestHandler(t, turnFunc(func(_ context.Context, _ string, emit agent.EmitFunc) error {
		return emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, Iteration: 1})
	}))
	handler.ServeHTTP(response, newAuthenticatedRunRequest(runBody("health")))

	if len(response.deadlines) < 2 {
		t.Fatalf("got %d write deadlines, want active and cleared deadlines", len(response.deadlines))
	}
	if response.deadlines[0].IsZero() {
		t.Fatal("first write deadline is zero")
	}
	if !response.deadlines[len(response.deadlines)-1].IsZero() {
		t.Fatal("final write deadline was not cleared")
	}
}

func TestRunDoesNotEmitAfterFinished(t *testing.T) {
	handler := newTestHandler(t, turnFunc(func(_ context.Context, _ string, emit agent.EmitFunc) error {
		if err := emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, Iteration: 1}); err != nil {
			return err
		}
		return errors.New("session close failed")
	}))
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, newAuthenticatedRunRequest(runBody("health")))

	if events := aguiEvents(t, response.Body.String()); eventTypes(events) != "RUN_STARTED,RUN_FINISHED" {
		t.Fatalf("unexpected terminal events %s", response.Body.String())
	}
}

func TestRunDoesNotStreamErrorAfterCancellation(t *testing.T) {
	handler := newTestHandler(t, turnFunc(func(ctx context.Context, _ string, _ agent.EmitFunc) error {
		return ctx.Err()
	}))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, newAuthenticatedRunRequest(runBody("health")).WithContext(ctx))

	if events := aguiEvents(t, response.Body.String()); eventTypes(events) != "RUN_STARTED" {
		t.Fatalf("got events after request cancellation: %s", response.Body.String())
	}
}

func TestWriteFrameEnforcesStreamBound(t *testing.T) {
	response := httptest.NewRecorder()
	written := int64(maxStreamBytes)
	frame, err := aguiFrames([]agui.Event{{Type: agui.RunError, Code: "test"}})
	if err != nil {
		t.Fatal(err)
	}
	err = writeFrame(response, response, &written, frame)
	if err == nil || !strings.Contains(err.Error(), "stream exceeds") {
		t.Fatalf("writeFrame() error = %v", err)
	}
	if response.Body.Len() != 0 {
		t.Fatalf("writeFrame() wrote %d bytes beyond limit", response.Body.Len())
	}
}

func TestRunRejectsWhenConcurrencyLimitIsReached(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	firstDone := make(chan struct{})
	var calls atomic.Int64
	asker := turnFunc(func(_ context.Context, _ string, emit agent.EmitFunc) error {
		if calls.Add(1) == 1 {
			close(started)
			<-release
		}
		return emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, Iteration: 1})
	})
	handler := newTestHandlerWithLimit(t, asker, 1)

	firstRequest := newAuthenticatedRunRequest(runBody("first"))
	firstResponse := httptest.NewRecorder()
	go func() {
		defer close(firstDone)
		handler.ServeHTTP(firstResponse, firstRequest)
	}()
	<-started

	secondResponse := httptest.NewRecorder()
	handler.ServeHTTP(secondResponse, newAuthenticatedRunRequest(runBody("second")))
	if secondResponse.Code != http.StatusTooManyRequests {
		t.Fatalf("second status = %d, want 429", secondResponse.Code)
	}
	if secondResponse.Header().Get("Retry-After") != "1" {
		t.Fatalf("Retry-After = %q, want 1", secondResponse.Header().Get("Retry-After"))
	}
	var errorResponse ErrorResponse
	if err := json.NewDecoder(secondResponse.Body).Decode(&errorResponse); err != nil {
		t.Fatalf("decode 429 response: %v", err)
	}
	if errorResponse.Error.Code != "too_many_requests" {
		t.Fatalf("error code = %q, want too_many_requests", errorResponse.Error.Code)
	}

	close(release)
	<-firstDone
	if firstResponse.Code != http.StatusOK {
		t.Fatalf("first status = %d, want 200", firstResponse.Code)
	}

	thirdResponse := httptest.NewRecorder()
	handler.ServeHTTP(thirdResponse, newAuthenticatedRunRequest(runBody("third")))
	if thirdResponse.Code != http.StatusOK {
		t.Fatalf("third status = %d, want released slot", thirdResponse.Code)
	}
	if calls.Load() != 2 {
		t.Fatalf("agent calls = %d, want first and third only", calls.Load())
	}
}

func TestRunReleasesConcurrencySlotAfterAgentError(t *testing.T) {
	var calls atomic.Int64
	handler := newTestHandlerWithLimit(t, turnFunc(func(context.Context, string, agent.EmitFunc) error {
		calls.Add(1)
		return errors.New("agent failed")
	}), 1)

	for _, prompt := range []string{"first", "second"} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, newAuthenticatedRunRequest(runBody(prompt)))
		if response.Code != http.StatusOK {
			t.Fatalf("%s status = %d, want streaming 200", prompt, response.Code)
		}
	}
	if calls.Load() != 2 {
		t.Fatalf("agent calls = %d, want slot released after error", calls.Load())
	}
}

func TestRunReleasesConcurrencySlotAfterCancellation(t *testing.T) {
	started := make(chan struct{})
	done := make(chan struct{})
	var calls atomic.Int64
	handler := newTestHandlerWithLimit(t, turnFunc(func(ctx context.Context, _ string, emit agent.EmitFunc) error {
		if calls.Add(1) == 1 {
			close(started)
			<-ctx.Done()
			return ctx.Err()
		}
		return emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, Iteration: 1})
	}), 1)

	ctx, cancel := context.WithCancel(context.Background())
	firstRequest := newAuthenticatedRunRequest(runBody("first")).WithContext(ctx)
	go func() {
		defer close(done)
		handler.ServeHTTP(httptest.NewRecorder(), firstRequest)
	}()
	<-started
	cancel()
	<-done

	response := httptest.NewRecorder()
	handler.ServeHTTP(response, newAuthenticatedRunRequest(runBody("second")))
	if response.Code != http.StatusOK || calls.Load() != 2 {
		t.Fatalf("second status=%d calls=%d, want released slot", response.Code, calls.Load())
	}
}

func TestNewHandlerRejectsInvalidBackends(t *testing.T) {
	config := HandlerConfig{MaxConcurrentAsks: 4, AuditLogger: discardAuditLogger()}
	valid := Backend{Name: "om3", Conversations: turnService{run: turnFunc(func(context.Context, string, agent.EmitFunc) error { return nil })}, Authenticator: allowTestTokenVerifier()}
	if _, err := NewHandler(nil, config); err == nil {
		t.Fatal("NewHandler() accepted no backend")
	}
	for _, name := range []string{"", "OM3", "om/3", "om-3", strings.Repeat("a", 33)} {
		backend := valid
		backend.Name = name
		if _, err := NewHandler([]Backend{backend}, config); err == nil {
			t.Fatalf("NewHandler() accepted backend name %q", name)
		}
	}
	if _, err := NewHandler([]Backend{valid, valid}, config); err == nil {
		t.Fatal("NewHandler() accepted a duplicated backend")
	}
	if _, err := NewHandler([]Backend{{Name: "om3", Conversations: nil, Authenticator: allowTestTokenVerifier()}}, config); err == nil {
		t.Fatal("NewHandler() accepted nil conversation service")
	}
	if _, err := NewHandler([]Backend{{Name: "om3", Conversations: turnService{run: turnFunc(func(context.Context, string, agent.EmitFunc) error { return nil })}, Authenticator: nil}}, config); err == nil {
		t.Fatal("NewHandler() accepted nil verifier")
	}
	if _, err := NewHandler([]Backend{{Name: "om3", Conversations: turnService{run: turnFunc(func(context.Context, string, agent.EmitFunc) error { return nil })}, Authenticator: allowTestTokenVerifier()}}, HandlerConfig{AuditLogger: discardAuditLogger()}); err == nil {
		t.Fatal("NewHandler() accepted zero concurrent asks")
	}
	if _, err := NewHandler([]Backend{{Name: "om3", Conversations: turnService{run: turnFunc(func(context.Context, string, agent.EmitFunc) error { return nil })}, Authenticator: allowTestTokenVerifier()}}, HandlerConfig{MaxConcurrentAsks: 4}); err == nil {
		t.Fatal("NewHandler() accepted nil audit logger")
	}
}

type writeDeadlineRecorder struct {
	*httptest.ResponseRecorder
	deadlines []time.Time
}

func (r *writeDeadlineRecorder) SetWriteDeadline(deadline time.Time) error {
	r.deadlines = append(r.deadlines, deadline)
	return nil
}

func newTestHandler(t *testing.T, asker turnFunc) http.Handler {
	t.Helper()
	if asker == nil {
		asker = turnFunc(func(context.Context, string, agent.EmitFunc) error { return nil })
	}
	return newTestHandlerWithVerifier(t, asker, allowTestTokenVerifier())
}

func newTestHandlerWithLimit(t *testing.T, asker turnFunc, maxConcurrentAsks int) http.Handler {
	t.Helper()
	handler, err := NewHandler([]Backend{{Name: "om3", Conversations: turnService{run: asker}, Authenticator: allowTestTokenVerifier()}}, HandlerConfig{MaxConcurrentAsks: maxConcurrentAsks, AuditLogger: discardAuditLogger()})
	if err != nil {
		t.Fatalf("create API handler: %v", err)
	}
	return handler
}

func newTestHandlerWithVerifier(t *testing.T, asker turnFunc, verifier Authenticator) http.Handler {
	t.Helper()
	handler, err := NewHandler([]Backend{{Name: "om3", Conversations: turnService{run: asker}, Authenticator: verifier}}, HandlerConfig{MaxConcurrentAsks: 4, AuditLogger: discardAuditLogger()})
	if err != nil {
		t.Fatalf("create API handler: %v", err)
	}
	return handler
}

func discardAuditLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func allowTestTokenVerifier() Authenticator {
	return tokenVerifierFunc(func(context.Context, string) (auth.Identity, error) {
		return auth.Identity{ClusterID: "cluster-id", Subject: "test-user", Issuer: "test-node"}, nil
	})
}

func newAuthenticatedRunRequest(body string) *http.Request {
	request := httptest.NewRequest(http.MethodPost, "/v1/om3/conversations/conversation-1/runs", strings.NewReader(body))
	request.Header.Set("Authorization", "Bearer token")
	request.Header.Set(testBackendHeader, "cluster-id")
	request.Header.Set("Content-Type", "application/json")
	return request
}

// TestRunGivesEachUserAShareOfTheSlots lets one user run at most their share
// at once, issuer and subject: past it, they get 429 too_many_user_requests,
// while another user still runs in the slots left.
func TestRunGivesEachUserAShareOfTheSlots(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	firstDone := make(chan struct{})
	var calls atomic.Int64
	asker := turnFunc(func(_ context.Context, _ string, emit agent.EmitFunc) error {
		if calls.Add(1) == 1 {
			close(started)
			<-release
		}
		return emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, Iteration: 1})
	})
	verifier := tokenVerifierFunc(func(_ context.Context, token string) (auth.Identity, error) {
		return auth.Identity{ClusterID: "cluster-id", Subject: token, Issuer: "test-node"}, nil
	})
	handler, err := NewHandler([]Backend{{Name: "om3", Conversations: turnService{run: asker}, Authenticator: verifier}},
		HandlerConfig{MaxConcurrentAsks: 4, MaxConcurrentAsksPerUser: 1, AuditLogger: discardAuditLogger()})
	if err != nil {
		t.Fatal(err)
	}
	runAs := func(user, body string) *http.Request {
		request := newAuthenticatedRunRequest(runBody(body))
		request.Header.Set("Authorization", "Bearer "+user)
		return request
	}
	firstResponse := httptest.NewRecorder()
	go func() {
		defer close(firstDone)
		handler.ServeHTTP(firstResponse, runAs("alice", "first"))
	}()
	<-started

	second := httptest.NewRecorder()
	handler.ServeHTTP(second, runAs("alice", "second"))
	var refusal ErrorResponse
	if err := json.NewDecoder(second.Body).Decode(&refusal); err != nil || second.Code != http.StatusTooManyRequests || refusal.Error.Code != "too_many_user_requests" {
		t.Fatalf("alice's second run: %d %+v %v", second.Code, refusal, err)
	}
	other := httptest.NewRecorder()
	handler.ServeHTTP(other, runAs("bob", "bob's"))
	if other.Code != http.StatusOK {
		t.Fatalf("bob's run: %d %s", other.Code, other.Body.String())
	}
	close(release)
	<-firstDone
	// The slot of alice is free again.
	again := httptest.NewRecorder()
	handler.ServeHTTP(again, runAs("alice", "again"))
	if again.Code != http.StatusOK {
		t.Fatalf("alice after her run: %d", again.Code)
	}
}

// TestStoreLimitsAreToldApart names the run failures of a turn the store
// refused: the quota of the user, the limit of the instance.
func TestStoreLimitsAreToldApart(t *testing.T) {
	for err, want := range map[error]string{
		fmt.Errorf("complete: %w", conversation.ErrUserLimit): "conversation_quota",
		fmt.Errorf("complete: %w", conversation.ErrLimit):     "conversation_limit",
		errors.New("model failed"):                            "agent_failed",
	} {
		if got := streamFailureCode(err, false, false, nil); got != want {
			t.Errorf("%v: %s, want %s", err, got, want)
		}
	}
}
