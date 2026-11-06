package conversation

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/llm"
)

var serviceTestNow = time.Date(2026, 7, 24, 10, 0, 0, 0, time.UTC)

func TestServiceCompletesTurnBeforeEmittingCompletion(t *testing.T) {
	store := newServiceTestStore()
	runner := turnRunnerFunc(func(_ context.Context, history []llm.Message, prompt string, emit agent.EmitFunc) (agent.TurnResult, error) {
		if len(history) != 2 || prompt != "status now" {
			t.Fatalf("unexpected runner input history=%#v prompt=%q", history, prompt)
		}
		if err := emit(agent.Event{Type: agent.EventTextDelta, TextDelta: "healthy", Iteration: 1}); err != nil {
			return agent.TurnResult{}, err
		}
		if err := emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, Iteration: 1}); err != nil {
			return agent.TurnResult{}, err
		}
		return agent.TurnResult{Messages: []llm.Message{
			{Role: llm.RoleUser, Text: prompt},
			{Role: llm.RoleAssistant, Text: "healthy"},
		}, FinishReason: llm.FinishReasonCompleted}, nil
	})
	service := newTestService(t, store, runner)
	store.history = []llm.Message{{Role: llm.RoleUser, Text: "previous"}, {Role: llm.RoleAssistant, Text: "answer"}}

	execution, err := service.PrepareTurn(t.Context(), serviceTestIdentity(), store.item.ID, "status now")
	if err != nil {
		t.Fatalf("prepare turn: %v", err)
	}
	var eventTypes []agent.EventType
	if err := execution.Run(t.Context(), func(event agent.Event) error {
		if event.Type == agent.EventCompleted && !store.completed {
			t.Fatal("completion emitted before messages were committed")
		}
		eventTypes = append(eventTypes, event.Type)
		return nil
	}); err != nil {
		t.Fatalf("run turn: %v", err)
	}
	if !store.completed || store.failed {
		t.Fatalf("completed=%t failed=%t", store.completed, store.failed)
	}
	if len(eventTypes) != 2 || eventTypes[0] != agent.EventTextDelta || eventTypes[1] != agent.EventCompleted {
		t.Fatalf("unexpected events %v", eventTypes)
	}
	if got := store.item.ExpiresAt; !got.Equal(serviceTestNow.Add(DefaultLifetime)) {
		t.Fatalf("got sliding expiry %s, want %s", got, serviceTestNow.Add(DefaultLifetime))
	}
}

func TestServiceKeepsCommittedTurnWhenCompletionDeliveryFails(t *testing.T) {
	store := newServiceTestStore()
	runner := turnRunnerFunc(func(_ context.Context, _ []llm.Message, prompt string, emit agent.EmitFunc) (agent.TurnResult, error) {
		if err := emit(agent.Event{Type: agent.EventTextDelta, TextDelta: "committed answer", Iteration: 1}); err != nil {
			return agent.TurnResult{}, err
		}
		if err := emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, Iteration: 1}); err != nil {
			return agent.TurnResult{}, err
		}
		return agent.TurnResult{Messages: []llm.Message{
			{Role: llm.RoleUser, Text: prompt},
			{Role: llm.RoleAssistant, Text: "committed answer"},
		}, FinishReason: llm.FinishReasonCompleted}, nil
	})
	service := newTestService(t, store, runner)
	execution, err := service.PrepareTurn(t.Context(), serviceTestIdentity(), store.item.ID, "status now")
	if err != nil {
		t.Fatalf("prepare turn: %v", err)
	}
	deliveryErr := errors.New("client disconnected before completion")
	err = execution.Run(t.Context(), func(event agent.Event) error {
		if event.Type == agent.EventCompleted {
			return deliveryErr
		}
		return nil
	})
	if !errors.Is(err, deliveryErr) {
		t.Fatalf("run turn error = %v, want completion delivery error", err)
	}
	if !store.completed || store.failed {
		t.Fatalf("completed=%t failed=%t", store.completed, store.failed)
	}
	if len(store.completedMessages) != 2 || store.completedMessages[0].Text != "status now" || store.completedMessages[1].Text != "committed answer" {
		t.Fatalf("committed messages = %#v", store.completedMessages)
	}
}

func TestServiceFailedTurnDoesNotPersistPartialMessages(t *testing.T) {
	store := newServiceTestStore()
	runner := turnRunnerFunc(func(_ context.Context, _ []llm.Message, _ string, emit agent.EmitFunc) (agent.TurnResult, error) {
		if err := emit(agent.Event{Type: agent.EventTextDelta, TextDelta: "partial", Iteration: 1}); err != nil {
			return agent.TurnResult{}, err
		}
		return agent.TurnResult{}, errors.New("provider unavailable")
	})
	service := newTestService(t, store, runner)
	execution, err := service.PrepareTurn(t.Context(), serviceTestIdentity(), store.item.ID, "status")
	if err != nil {
		t.Fatalf("prepare turn: %v", err)
	}
	if err := execution.Run(t.Context(), func(agent.Event) error { return nil }); err == nil {
		t.Fatal("run turn succeeded")
	}
	if store.completed || !store.failed || store.failureCode != "agent_failed" || len(store.completedMessages) != 0 {
		t.Fatalf("unexpected terminal state completed=%t failed=%t code=%q messages=%#v", store.completed, store.failed, store.failureCode, store.completedMessages)
	}
}

// TestServiceFailsATurnWhoseRunnerPanics fails the turn of a runner that
// panics, rather than leave it running until a restart, and lets the panic
// go on to the server.
func TestServiceFailsATurnWhoseRunnerPanics(t *testing.T) {
	store := newServiceTestStore()
	runner := turnRunnerFunc(func(context.Context, []llm.Message, string, agent.EmitFunc) (agent.TurnResult, error) {
		panic("nil annotations")
	})
	service := newTestService(t, store, runner)
	execution, err := service.PrepareTurn(t.Context(), serviceTestIdentity(), store.item.ID, "status")
	if err != nil {
		t.Fatalf("prepare turn: %v", err)
	}
	func() {
		defer func() {
			if r := recover(); r != "nil annotations" {
				t.Errorf("the panic did not go on: %v", r)
			}
		}()
		_ = execution.Run(t.Context(), func(agent.Event) error { return nil })
	}()
	if !store.failed || store.failureCode != "agent_panicked" || store.completed {
		t.Fatalf("turn failed=%t code=%q completed=%t", store.failed, store.failureCode, store.completed)
	}
}

func TestServiceRejectsExpiredAndBusyConversation(t *testing.T) {
	store := newServiceTestStore()
	service := newTestService(t, store, turnRunnerFunc(func(context.Context, []llm.Message, string, agent.EmitFunc) (agent.TurnResult, error) {
		return agent.TurnResult{}, nil
	}))
	store.item.ExpiresAt = serviceTestNow
	if _, err := service.PrepareTurn(t.Context(), serviceTestIdentity(), store.item.ID, "status"); !errors.Is(err, ErrExpired) {
		t.Fatalf("expired turn error=%v", err)
	}
	store.item.ExpiresAt = serviceTestNow.Add(time.Hour)
	store.beginErr = ErrBusy
	if _, err := service.PrepareTurn(t.Context(), serviceTestIdentity(), store.item.ID, "status"); !errors.Is(err, ErrBusy) {
		t.Fatalf("busy turn error=%v", err)
	}
}

func TestServiceUpdatesNormalizedConversationTitle(t *testing.T) {
	store := newServiceTestStore()
	service := newTestService(t, store, turnRunnerFunc(func(context.Context, []llm.Message, string, agent.EmitFunc) (agent.TurnResult, error) {
		return agent.TurnResult{}, nil
	}))

	item, err := service.UpdateTitle(t.Context(), serviceTestIdentity(), store.item.ID, "  Cluster\n health  ")
	if err != nil {
		t.Fatalf("update title: %v", err)
	}
	if item.Title != "Cluster health" || store.item.Title != item.Title || !item.UpdatedAt.Equal(serviceTestNow) {
		t.Fatalf("updated conversation = %#v, stored = %#v", item, store.item)
	}
	if _, err := service.UpdateTitle(t.Context(), serviceTestIdentity(), store.item.ID, " \t "); !errors.Is(err, ErrInvalid) {
		t.Fatalf("empty title error = %v", err)
	}
	store.item.ExpiresAt = serviceTestNow
	if _, err := service.UpdateTitle(t.Context(), serviceTestIdentity(), store.item.ID, "Expired"); !errors.Is(err, ErrExpired) {
		t.Fatalf("expired title update error = %v", err)
	}
}

func TestBoundHistoryKeepsCompleteNewestTurns(t *testing.T) {
	history := []llm.Message{
		{Role: llm.RoleUser, Text: "old"}, {Role: llm.RoleAssistant, Text: "old answer"},
		{Role: llm.RoleUser, Text: "new"}, {Role: llm.RoleAssistant, Text: "new answer"},
	}
	bounded, err := boundHistory(history, 2, 1024)
	if err != nil {
		t.Fatalf("bound history: %v", err)
	}
	if len(bounded) != 2 || bounded[0].Text != "new" || bounded[1].Text != "new answer" {
		t.Fatalf("unexpected bounded history %#v", bounded)
	}
}

type turnRunnerFunc func(context.Context, []llm.Message, string, agent.EmitFunc) (agent.TurnResult, error)

func (f turnRunnerFunc) RunTurn(ctx context.Context, history []llm.Message, prompt string, emit agent.EmitFunc) (agent.TurnResult, error) {
	return f(ctx, history, prompt, emit)
}

func (f turnRunnerFunc) ResumeTurn(context.Context, []llm.Message, agent.SuspendedTurn, agent.Decision, agent.EmitFunc) (agent.TurnResult, error) {
	return agent.TurnResult{}, errors.New("this runner does not resume turns")
}

type serviceTestStore struct {
	item              Conversation
	completedTiming   TurnTiming
	history           []llm.Message
	beginErr          error
	completed         bool
	failed            bool
	failureCode       string
	completedMessages []llm.Message
	messageQuery      MessageQuery
	messageReads      int
	suspended         *PendingConfirmation
	promptID          string
}

func newServiceTestStore() *serviceTestStore {
	return &serviceTestStore{item: Conversation{
		ID: "conversation-id", Owner: Owner{Backend: "om3", ClusterID: "cluster-id", Issuer: "cluster", Subject: "user"},
		CreatedAt: serviceTestNow.Add(-time.Hour), UpdatedAt: serviceTestNow.Add(-time.Hour),
		ExpiresAt: serviceTestNow.Add(time.Hour),
	}}
}

func newTestService(t *testing.T, store *serviceTestStore, runner TurnRunner) *Service {
	t.Helper()
	service, err := NewService(store, runner, ServiceConfig{Backend: "om3"})
	if err != nil {
		t.Fatalf("create service: %v", err)
	}
	service.now = func() time.Time { return serviceTestNow }
	service.newID = func() (string, error) { return "turn-id", nil }
	return service
}

func serviceTestIdentity() auth.Identity {
	return auth.Identity{ClusterID: "cluster-id", Issuer: "cluster", Subject: "user"}
}

func (s *serviceTestStore) CreateConversation(context.Context, Conversation) error { return nil }
func (s *serviceTestStore) GetConversation(_ context.Context, owner Owner, id string) (Conversation, error) {
	if owner != s.item.Owner || id != s.item.ID {
		return Conversation{}, ErrNotFound
	}
	return s.item, nil
}
func (s *serviceTestStore) ListConversations(context.Context, Owner, int) ([]Conversation, error) {
	return []Conversation{s.item}, nil
}
func (s *serviceTestStore) UpdateConversationTitle(_ context.Context, owner Owner, id string, title string, updatedAt time.Time) (Conversation, error) {
	if owner != s.item.Owner || id != s.item.ID {
		return Conversation{}, ErrNotFound
	}
	s.item.Title = title
	s.item.UpdatedAt = updatedAt
	return s.item, nil
}
func (s *serviceTestStore) DeleteConversation(context.Context, Owner, string) error { return nil }
func (s *serviceTestStore) BeginTurn(context.Context, Owner, string, string, time.Time) (Turn, error) {
	if s.beginErr != nil {
		return Turn{}, s.beginErr
	}
	return Turn{ID: "turn-id", ConversationID: s.item.ID, Sequence: 1, Status: TurnRunning, StartedAt: serviceTestNow}, nil
}
func (s *serviceTestStore) CompleteTurn(_ context.Context, _ Owner, _ string, _ string, completedAt time.Time, expiresAt time.Time, messages []llm.Message, _ TurnRun, timing TurnTiming) error {
	s.completedTiming = timing
	s.completed = true
	s.completedMessages = append([]llm.Message(nil), messages...)
	s.item.UpdatedAt = completedAt
	s.item.ExpiresAt = expiresAt
	return nil
}
func (s *serviceTestStore) FailTurn(_ context.Context, _ Owner, _ string, _ string, _ TurnStatus, code string, _ time.Time) error {
	s.failed = true
	s.failureCode = code
	return nil
}
func (s *serviceTestStore) LoadHistoryIDs(context.Context, Owner, string) ([]llm.Message, []string, error) {
	ids := make([]string, 0, len(s.history))
	for index := range s.history {
		ids = append(ids, fmt.Sprintf("stored-turn:%d", index+1))
	}
	return append([]llm.Message(nil), s.history...), ids, nil
}
func (s *serviceTestStore) ListMessages(_ context.Context, owner Owner, id string, query MessageQuery) (MessagePage, error) {
	if owner != s.item.Owner || id != s.item.ID {
		return MessagePage{}, ErrNotFound
	}
	s.messageReads++
	s.messageQuery = query
	return MessagePage{Messages: []DisplayMessage{}}, nil
}
func (s *serviceTestStore) RecoverInterrupted(context.Context, time.Time) (int64, error) {
	return 0, nil
}
func (s *serviceTestStore) DeleteExpired(context.Context, time.Time, int) (int64, error) {
	return 0, nil
}
func (s *serviceTestStore) Close() error { return nil }
func (s *serviceTestStore) TurnMessages(context.Context, Owner, string, string) (Turn, []llm.Message, error) {
	return Turn{}, nil, ErrNotFound
}
func (s *serviceTestStore) PutFeedback(context.Context, Owner, string, FeedbackRecord) (FeedbackRecord, error) {
	return FeedbackRecord{}, ErrNotFound
}
func (s *serviceTestStore) DeleteFeedback(context.Context, Owner, string, string) error {
	return ErrNotFound
}
func (s *serviceTestStore) ExpireConfirmations(context.Context, time.Time, int) (int64, error) {
	return 0, nil
}

func (s *serviceTestStore) DeleteExpiredFeedback(context.Context, time.Time, int) (int64, error) {
	return 0, nil
}

func (s *serviceTestStore) SuspendTurn(_ context.Context, _ Owner, pending PendingConfirmation) error {
	s.suspended = &pending
	return nil
}

func (s *serviceTestStore) ClaimConfirmation(_ context.Context, _ Owner, _ string, turnID string, confirmationID string, at time.Time) (PendingConfirmation, error) {
	if s.suspended == nil || s.suspended.TurnID != turnID || s.suspended.ConfirmationID != confirmationID {
		return PendingConfirmation{}, ErrNotPending
	}
	pending := *s.suspended
	s.suspended = nil
	if !pending.ExpiresAt.After(at) {
		return PendingConfirmation{}, ErrConfirmationExpired
	}
	return pending, nil
}

func (s *serviceTestStore) GetPendingConfirmation(context.Context, Owner, string, time.Time) (*PendingConfirmation, error) {
	return s.suspended, nil
}

func (s *serviceTestStore) SetTurnPromptID(_ context.Context, _ Owner, _ string, _ string, promptID string) (bool, error) {
	s.promptID = promptID
	return true, nil
}

func (s *serviceTestStore) CancelSuspendedTurn(_ context.Context, _ Owner, _ string, turnID string, code string, _ time.Time) error {
	if s.suspended == nil || s.suspended.TurnID != turnID {
		return ErrNotPending
	}
	s.suspended, s.failed, s.failureCode = nil, true, code
	return nil
}

func TestServiceBindsConversationsToItsBackend(t *testing.T) {
	store := newServiceTestStore()
	for _, name := range []string{"", "OM3", "om/3"} {
		if _, err := NewService(store, turnRunnerFunc(nil), ServiceConfig{Backend: name}); err == nil {
			t.Fatalf("NewService() accepted backend name %q", name)
		}
	}
	other, err := NewService(store, turnRunnerFunc(nil), ServiceConfig{Backend: "oc3"})
	if err != nil {
		t.Fatal(err)
	}
	other.now = func() time.Time { return serviceTestNow }
	if _, err := other.Get(t.Context(), serviceTestIdentity(), store.item.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("another backend read the conversation: %v", err)
	}
}
