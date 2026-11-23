package api

import (
	"context"
	"encoding/json"
	"errors"
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

func TestConversationLifecycleRoutes(t *testing.T) {
	now := time.Date(2026, 7, 24, 10, 0, 0, 0, time.UTC)
	item := conversation.Conversation{ID: "conversation-id", Title: "Cluster health", CreatedAt: now, UpdatedAt: now, ExpiresAt: now.Add(time.Hour)}
	deleted := false
	updated := false
	service := conversationServiceFuncs{
		create: func(_ context.Context, identity auth.Identity) (conversation.Conversation, error) {
			if identity.Subject != "test-user" || identity.Issuer != "test-node" {
				t.Fatalf("unexpected identity %+v", identity)
			}
			return item, nil
		},
		list: func(context.Context, auth.Identity) ([]conversation.Conversation, error) {
			return []conversation.Conversation{item}, nil
		},
		get: func(context.Context, auth.Identity, string) (conversation.Conversation, error) { return item, nil },
		update: func(_ context.Context, identity auth.Identity, id string, title string) (conversation.Conversation, error) {
			if identity.Subject != "test-user" || id != item.ID || title != "Renamed conversation" {
				t.Fatalf("unexpected title update identity=%+v id=%q title=%q", identity, id, title)
			}
			updated = true
			item.Title = title
			return item, nil
		},
		delete: func(context.Context, auth.Identity, string) error {
			deleted = true
			return nil
		},
	}
	handler := newConversationTestHandler(t, service)

	create := authenticatedRequest(http.MethodPost, "/v1/om3/conversations", "")
	createResponse := httptest.NewRecorder()
	handler.ServeHTTP(createResponse, create)
	if createResponse.Code != http.StatusCreated {
		t.Fatalf("create status=%d body=%s", createResponse.Code, createResponse.Body.String())
	}
	var envelope ConversationEnvelope
	if err := json.NewDecoder(createResponse.Body).Decode(&envelope); err != nil || envelope.Conversation.ID != item.ID || envelope.Conversation.Title != item.Title {
		t.Fatalf("create response=%+v error=%v", envelope, err)
	}

	listResponse := httptest.NewRecorder()
	handler.ServeHTTP(listResponse, authenticatedRequest(http.MethodGet, "/v1/om3/conversations", ""))
	if listResponse.Code != http.StatusOK || !strings.Contains(listResponse.Body.String(), item.ID) {
		t.Fatalf("list status=%d body=%s", listResponse.Code, listResponse.Body.String())
	}

	getResponse := httptest.NewRecorder()
	handler.ServeHTTP(getResponse, authenticatedRequest(http.MethodGet, "/v1/om3/conversations/"+item.ID, ""))
	if getResponse.Code != http.StatusOK || !strings.Contains(getResponse.Body.String(), item.ID) {
		t.Fatalf("get status=%d body=%s", getResponse.Code, getResponse.Body.String())
	}

	updateRequest := authenticatedRequest(http.MethodPatch, "/v1/om3/conversations/"+item.ID, `{"title":"Renamed conversation"}`)
	updateRequest.Header.Set("Content-Type", "application/json")
	updateResponse := httptest.NewRecorder()
	handler.ServeHTTP(updateResponse, updateRequest)
	if updateResponse.Code != http.StatusOK || !updated {
		t.Fatalf("update status=%d updated=%t body=%s", updateResponse.Code, updated, updateResponse.Body.String())
	}
	if err := json.NewDecoder(updateResponse.Body).Decode(&envelope); err != nil || envelope.Conversation.Title != "Renamed conversation" {
		t.Fatalf("update response=%+v error=%v", envelope, err)
	}

	deleteResponse := httptest.NewRecorder()
	handler.ServeHTTP(deleteResponse, authenticatedRequest(http.MethodDelete, "/v1/om3/conversations/"+item.ID, ""))
	if deleteResponse.Code != http.StatusNoContent || !deleted {
		t.Fatalf("delete status=%d deleted=%t", deleteResponse.Code, deleted)
	}
}

func TestConversationTitleUpdateRejectsInvalidRequests(t *testing.T) {
	tests := []struct {
		name        string
		contentType string
		body        string
		wantStatus  int
		wantCode    string
	}{
		{name: "content type", body: `{"title":"valid"}`, wantStatus: http.StatusUnsupportedMediaType, wantCode: "unsupported_media_type"},
		{name: "malformed", contentType: "application/json", body: `{"title":`, wantStatus: http.StatusBadRequest, wantCode: "invalid_request"},
		{name: "unknown field", contentType: "application/json", body: `{"title":"valid","extra":true}`, wantStatus: http.StatusBadRequest, wantCode: "invalid_request"},
		{name: "multiple values", contentType: "application/json", body: `{"title":"valid"}{}`, wantStatus: http.StatusBadRequest, wantCode: "invalid_request"},
		{name: "oversized", contentType: "application/json", body: `{"title":"` + strings.Repeat("a", maxConversationTitleRequestBytes) + `"}`, wantStatus: http.StatusRequestEntityTooLarge, wantCode: "request_too_large"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			handler := newConversationTestHandler(t, conversationServiceFuncs{})
			request := authenticatedRequest(http.MethodPatch, "/v1/om3/conversations/id", test.body)
			if test.contentType != "" {
				request.Header.Set("Content-Type", test.contentType)
			}
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != test.wantStatus {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
			var body ErrorResponse
			if err := json.NewDecoder(response.Body).Decode(&body); err != nil || body.Error.Code != test.wantCode {
				t.Fatalf("error response=%+v decode=%v", body, err)
			}
		})
	}
}

func TestConversationTitleUpdateReturnsStableServiceErrors(t *testing.T) {
	for _, test := range []struct {
		name       string
		err        error
		wantStatus int
		wantCode   string
	}{
		{name: "not found", err: conversation.ErrNotFound, wantStatus: http.StatusNotFound, wantCode: "conversation_not_found"},
		{name: "expired", err: conversation.ErrExpired, wantStatus: http.StatusGone, wantCode: "conversation_expired"},
		{name: "invalid title", err: conversation.ErrInvalid, wantStatus: http.StatusBadRequest, wantCode: "invalid_request"},
	} {
		t.Run(test.name, func(t *testing.T) {
			service := conversationServiceFuncs{update: func(context.Context, auth.Identity, string, string) (conversation.Conversation, error) {
				return conversation.Conversation{}, test.err
			}}
			handler := newConversationTestHandler(t, service)
			request := authenticatedRequest(http.MethodPatch, "/v1/om3/conversations/id", `{"title":"valid"}`)
			request.Header.Set("Content-Type", "application/json")
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != test.wantStatus {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
			var body ErrorResponse
			if err := json.NewDecoder(response.Body).Decode(&body); err != nil || body.Error.Code != test.wantCode {
				t.Fatalf("error response=%+v decode=%v", body, err)
			}
		})
	}
}

func TestConversationRunStreamsPreparedExecution(t *testing.T) {
	execution := &testTurnExecution{run: func(_ context.Context, emit agent.EmitFunc) error {
		if err := emit(agent.Event{Type: agent.EventTextDelta, TextDelta: "healthy", Iteration: 1}); err != nil {
			return err
		}
		return emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, Iteration: 1})
	}}
	service := conversationServiceFuncs{prepare: func(_ context.Context, identity auth.Identity, id string, prompt string) (conversation.TurnExecution, error) {
		if identity.Subject != "test-user" || id != "conversation-id" || prompt != "health" {
			t.Fatalf("unexpected prepared turn identity=%+v id=%q prompt=%q", identity, id, prompt)
		}
		return execution, nil
	}}
	handler := newConversationTestHandler(t, service)
	request := authenticatedRequest(http.MethodPost, "/v1/om3/conversations/conversation-id/runs", runInput("conversation-id", "health"))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("run status=%d body=%s", response.Code, response.Body.String())
	}
	if events := aguiEvents(t, response.Body.String()); eventTypes(events) != "RUN_STARTED,TEXT_MESSAGE_START,TEXT_MESSAGE_CONTENT,TEXT_MESSAGE_END,RUN_FINISHED" {
		t.Fatalf("unexpected run events %s", response.Body.String())
	}
}

func TestConversationRunReturnsStablePreStreamErrors(t *testing.T) {
	for _, test := range []struct {
		name       string
		err        error
		wantStatus int
		wantCode   string
	}{
		{name: "not found", err: conversation.ErrNotFound, wantStatus: http.StatusNotFound, wantCode: "conversation_not_found"},
		{name: "expired", err: conversation.ErrExpired, wantStatus: http.StatusGone, wantCode: "conversation_expired"},
		{name: "busy", err: conversation.ErrBusy, wantStatus: http.StatusConflict, wantCode: "conversation_busy"},
	} {
		t.Run(test.name, func(t *testing.T) {
			service := conversationServiceFuncs{prepare: func(context.Context, auth.Identity, string, string) (conversation.TurnExecution, error) {
				return nil, test.err
			}}
			handler := newConversationTestHandler(t, service)
			request := authenticatedRequest(http.MethodPost, "/v1/om3/conversations/id/runs", runInput("id", "health"))
			request.Header.Set("Content-Type", "application/json")
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != test.wantStatus {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
			var body ErrorResponse
			if err := json.NewDecoder(response.Body).Decode(&body); err != nil || body.Error.Code != test.wantCode {
				t.Fatalf("error response=%+v decode=%v", body, err)
			}
		})
	}
}

func TestConversationRoutesRequireAuthentication(t *testing.T) {
	handler := newConversationTestHandler(t, conversationServiceFuncs{})
	for _, route := range []struct{ method, path string }{
		{http.MethodPost, "/v1/om3/conversations"},
		{http.MethodGet, "/v1/om3/conversations"},
		{http.MethodGet, "/v1/om3/conversations/id"},
		{http.MethodGet, "/v1/om3/conversations/id/messages"},
		{http.MethodPatch, "/v1/om3/conversations/id"},
		{http.MethodDelete, "/v1/om3/conversations/id"},
		{http.MethodPost, "/v1/om3/conversations/id/runs"},
		{http.MethodGet, "/v1/om3/capabilities"},
		{http.MethodPut, "/v1/om3/conversations/id/turns/t/feedback"},
		{http.MethodDelete, "/v1/om3/conversations/id/turns/t/feedback"},
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest(route.method, route.path, nil))
		if response.Code != http.StatusUnauthorized {
			t.Fatalf("%s %s status=%d", route.method, route.path, response.Code)
		}
	}
}

type conversationServiceFuncs struct {
	create   func(context.Context, auth.Identity) (conversation.Conversation, error)
	get      func(context.Context, auth.Identity, string) (conversation.Conversation, error)
	list     func(context.Context, auth.Identity) ([]conversation.Conversation, error)
	update   func(context.Context, auth.Identity, string, string) (conversation.Conversation, error)
	delete   func(context.Context, auth.Identity, string) error
	prepare  func(context.Context, auth.Identity, string, string) (conversation.TurnExecution, error)
	messages func(context.Context, auth.Identity, string, conversation.MessageQuery) (conversation.MessagePage, error)
	confirm  func(context.Context, auth.Identity, string, string, string, agent.Decision) (conversation.TurnExecution, error)
	// confirmEdited, when set, takes the confirmations with the arguments
	// the user edited.
	confirmEdited func(context.Context, auth.Identity, string, string, string, agent.Decision, json.RawMessage) (conversation.TurnExecution, error)
	pending       func(context.Context, auth.Identity, string) (*conversation.PendingConfirmation, error)
	feedback      func(context.Context, auth.Identity, string, string, conversation.Feedback) (conversation.FeedbackRecord, error)
	unrate        func(context.Context, auth.Identity, string, string) error
	// clientCalls, abandon and answer serve the tools of the client.
	clientCalls func(context.Context, auth.Identity, string) ([]string, error)
	abandon     func(context.Context, auth.Identity, string) error
	answer      func(context.Context, auth.Identity, string, []llm.ToolResult) (conversation.TurnExecution, error)
}

func (s conversationServiceFuncs) PendingClientCalls(ctx context.Context, identity auth.Identity, id string) ([]string, error) {
	if s.clientCalls == nil {
		return nil, nil
	}
	return s.clientCalls(ctx, identity, id)
}

func (s conversationServiceFuncs) AbandonClientCalls(ctx context.Context, identity auth.Identity, id string) error {
	if s.abandon == nil {
		return nil
	}
	return s.abandon(ctx, identity, id)
}

func (s conversationServiceFuncs) PrepareClientResults(ctx context.Context, identity auth.Identity, id string, results []llm.ToolResult) (conversation.TurnExecution, error) {
	if s.answer == nil {
		return nil, conversation.ErrNotPending
	}
	return s.answer(ctx, identity, id, results)
}

func (s conversationServiceFuncs) SetFeedback(ctx context.Context, identity auth.Identity, id string, turnID string, feedback conversation.Feedback) (conversation.FeedbackRecord, error) {
	if s.feedback == nil {
		return conversation.FeedbackRecord{}, errors.New("unexpected feedback")
	}
	return s.feedback(ctx, identity, id, turnID, feedback)
}

func (s conversationServiceFuncs) DeleteFeedback(ctx context.Context, identity auth.Identity, id string, turnID string) error {
	if s.unrate == nil {
		return errors.New("unexpected feedback deletion")
	}
	return s.unrate(ctx, identity, id, turnID)
}

func (s conversationServiceFuncs) Create(ctx context.Context, identity auth.Identity) (conversation.Conversation, error) {
	if s.create == nil {
		return conversation.Conversation{}, errors.New("unexpected create")
	}
	return s.create(ctx, identity)
}
func (s conversationServiceFuncs) Get(ctx context.Context, identity auth.Identity, id string) (conversation.Conversation, error) {
	if s.get == nil {
		return conversation.Conversation{}, errors.New("unexpected get")
	}
	return s.get(ctx, identity, id)
}
func (s conversationServiceFuncs) List(ctx context.Context, identity auth.Identity) ([]conversation.Conversation, error) {
	if s.list == nil {
		return nil, errors.New("unexpected list")
	}
	return s.list(ctx, identity)
}
func (s conversationServiceFuncs) Messages(ctx context.Context, identity auth.Identity, id string, query conversation.MessageQuery) (conversation.MessagePage, error) {
	if s.messages == nil {
		return conversation.MessagePage{}, errors.New("unexpected messages")
	}
	return s.messages(ctx, identity, id, query)
}
func (s conversationServiceFuncs) Delete(ctx context.Context, identity auth.Identity, id string) error {
	if s.delete == nil {
		return errors.New("unexpected delete")
	}
	return s.delete(ctx, identity, id)
}
func (s conversationServiceFuncs) UpdateTitle(ctx context.Context, identity auth.Identity, id string, title string) (conversation.Conversation, error) {
	if s.update == nil {
		return conversation.Conversation{}, errors.New("unexpected update")
	}
	return s.update(ctx, identity, id, title)
}
func (s conversationServiceFuncs) PrepareTurn(ctx context.Context, identity auth.Identity, id string, prompt string) (conversation.TurnExecution, error) {
	if s.prepare == nil {
		return nil, errors.New("unexpected prepare")
	}
	return s.prepare(ctx, identity, id, prompt)
}

func (s conversationServiceFuncs) PrepareConfirmation(ctx context.Context, identity auth.Identity, id string, turnID string, confirmationID string, decision agent.Decision, edited json.RawMessage) (conversation.TurnExecution, error) {
	if s.confirmEdited != nil {
		return s.confirmEdited(ctx, identity, id, turnID, confirmationID, decision, edited)
	}
	if s.confirm == nil {
		return nil, errors.New("unexpected confirmation")
	}
	if edited != nil {
		return nil, errors.New("unexpected edited arguments")
	}
	return s.confirm(ctx, identity, id, turnID, confirmationID, decision)
}

// PendingConfirmation reports no pending confirmation unless a test sets one.
func (s conversationServiceFuncs) PendingConfirmation(ctx context.Context, identity auth.Identity, id string) (*conversation.PendingConfirmation, error) {
	if s.pending == nil {
		return nil, nil
	}
	return s.pending(ctx, identity, id)
}

type testTurnExecution struct {
	run      func(context.Context, agent.EmitFunc) error
	canceled bool
	// turnID and history are the turn and the messages, with their IDs,
	// the execution gives an AG-UI run.
	turnID     string
	promptID   string
	history    []llm.Message
	historyIDs []string
	// current and currentIDs are the messages of a resumed turn so far.
	current    []llm.Message
	currentIDs []string
}

func (e *testTurnExecution) TurnID() string {
	if e.turnID == "" {
		return "turn-id"
	}
	return e.turnID
}

func (e *testTurnExecution) SetPromptID(_ context.Context, id string) error {
	e.promptID = id
	return nil
}

func (e *testTurnExecution) PromptID() string {
	if e.promptID == "" {
		return e.TurnID() + ":1"
	}
	return e.promptID
}

func (e *testTurnExecution) Snapshot() ([]llm.Message, []string, int, int) {
	return append(append([]llm.Message(nil), e.history...), e.current...), append(append([]string(nil), e.historyIDs...), e.currentIDs...), len(e.history), 0
}

func (e *testTurnExecution) Run(ctx context.Context, emit agent.EmitFunc) error {
	return e.run(ctx, emit)
}
func (e *testTurnExecution) Cancel(string) error {
	e.canceled = true
	return nil
}

func newConversationTestHandler(t *testing.T, service ConversationService) http.Handler {
	t.Helper()
	handler, err := NewHandler([]Backend{{Name: "om3", Conversations: service, Authenticator: allowTestTokenVerifier()}}, HandlerConfig{Models: testModels(), MaxConcurrentAsks: 4, AuditLogger: discardAuditLogger()})
	if err != nil {
		t.Fatalf("create conversation test handler: %v", err)
	}
	return handler
}

func authenticatedRequest(method string, path string, body string) *http.Request {
	request := httptest.NewRequest(method, path, strings.NewReader(body))
	request.Header.Set("Authorization", "Bearer token")
	request.Header.Set(testBackendHeader, "cluster-id")
	return request
}
