package conversation

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"maps"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/llm"
)

const (
	DefaultLifetime          = 7 * 24 * time.Hour
	DefaultListLimit         = 100
	DefaultFinalizeTimeout   = 5 * time.Second
	DefaultHistoryMessages   = 128
	DefaultHistoryBytes      = 1 << 20
	DefaultExpiryDeleteBatch = 100
	// DefaultConfirmationLifetime is how long a tool call waits for the
	// decision of the user before its turn fails without running it.
	DefaultConfirmationLifetime = 10 * time.Minute
	maxServicePromptBytes       = 32 << 10
)

type TurnRunner interface {
	RunTurn(context.Context, []llm.Message, string, agent.EmitFunc) (agent.TurnResult, error)
	ResumeTurn(context.Context, []llm.Message, agent.SuspendedTurn, agent.Decision, agent.EmitFunc) (agent.TurnResult, error)
}

type ServiceConfig struct {
	// Backend names the backend the service runs turns on. Its conversations
	// are bound to it and invisible to the services of other backends.
	Backend            string
	Lifetime           time.Duration
	ListLimit          int
	FinalizeTimeout    time.Duration
	MaxHistoryMessages int
	MaxHistoryBytes    int
	ExpiryDeleteBatch  int
	// ConfirmationLifetime bounds the wait for the decision of the user on
	// a tool call requiring confirmation.
	ConfirmationLifetime time.Duration
	// HistoryCompaction bounds the tool results of the history the model
	// receives; a field left at zero takes the value of
	// DefaultHistoryCompaction, and an invalid policy is refused.
	HistoryCompaction HistoryCompaction
	// FeedbackRetention is how long a feedback is kept.
	FeedbackRetention time.Duration
	// Agent describes the agent of the backend, recorded on each turn.
	Agent AgentInfo
}

type Service struct {
	store  Store
	runner TurnRunner
	config ServiceConfig
	now    func() time.Time
	newID  func() (string, error)
}

func NewService(store Store, runner TurnRunner, config ServiceConfig) (*Service, error) {
	if store == nil {
		return nil, fmt.Errorf("conversation store is nil")
	}
	if runner == nil {
		return nil, fmt.Errorf("conversation turn runner is nil")
	}
	if !ValidBackendName(config.Backend) {
		return nil, fmt.Errorf("conversation service backend name %q is invalid", config.Backend)
	}
	config = withServiceDefaults(config)
	if config.Lifetime <= 0 || config.ListLimit <= 0 || config.FinalizeTimeout <= 0 ||
		config.MaxHistoryMessages <= 0 || config.MaxHistoryBytes <= 0 || config.ExpiryDeleteBatch <= 0 || config.ConfirmationLifetime <= 0 ||
		config.FeedbackRetention <= 0 {
		return nil, fmt.Errorf("conversation service limits must be positive")
	}
	if err := config.HistoryCompaction.validate(); err != nil {
		return nil, err
	}
	return &Service{store: store, runner: runner, config: config, now: time.Now, newID: randomID}, nil
}

func withServiceDefaults(config ServiceConfig) ServiceConfig {
	if config.Lifetime == 0 {
		config.Lifetime = DefaultLifetime
	}
	if config.ListLimit == 0 {
		config.ListLimit = DefaultListLimit
	}
	if config.FinalizeTimeout == 0 {
		config.FinalizeTimeout = DefaultFinalizeTimeout
	}
	if config.MaxHistoryMessages == 0 {
		config.MaxHistoryMessages = DefaultHistoryMessages
	}
	if config.MaxHistoryBytes == 0 {
		config.MaxHistoryBytes = DefaultHistoryBytes
	}
	if config.ExpiryDeleteBatch == 0 {
		config.ExpiryDeleteBatch = DefaultExpiryDeleteBatch
	}
	if config.ConfirmationLifetime == 0 {
		config.ConfirmationLifetime = DefaultConfirmationLifetime
	}
	if config.FeedbackRetention == 0 {
		config.FeedbackRetention = DefaultFeedbackRetention
	}
	config.HistoryCompaction = config.HistoryCompaction.withDefaults()
	return config
}

func (s *Service) Create(ctx context.Context, identity auth.Identity) (Conversation, error) {
	owner, err := s.ownerFromIdentity(identity)
	if err != nil {
		return Conversation{}, err
	}
	now := s.now().UTC()
	if _, err := s.purgeExpired(ctx, now); err != nil {
		return Conversation{}, fmt.Errorf("delete expired conversations before creation: %w", err)
	}
	id, err := s.newID()
	if err != nil {
		return Conversation{}, fmt.Errorf("generate conversation ID: %w", err)
	}
	item := Conversation{
		ID: id, Owner: owner, CreatedAt: now, UpdatedAt: now,
		ExpiresAt: now.Add(s.config.Lifetime),
	}
	if err := s.store.CreateConversation(ctx, item); err != nil {
		return Conversation{}, err
	}
	return item, nil
}

func (s *Service) Get(ctx context.Context, identity auth.Identity, id string) (Conversation, error) {
	owner, err := s.ownerFromIdentity(identity)
	if err != nil {
		return Conversation{}, err
	}
	item, err := s.store.GetConversation(ctx, owner, id)
	if err != nil {
		return Conversation{}, err
	}
	if !item.ExpiresAt.After(s.now().UTC()) {
		return Conversation{}, ErrExpired
	}
	return item, nil
}

func (s *Service) List(ctx context.Context, identity auth.Identity) ([]Conversation, error) {
	owner, err := s.ownerFromIdentity(identity)
	if err != nil {
		return nil, err
	}
	now := s.now().UTC()
	if _, err := s.purgeExpired(ctx, now); err != nil {
		return nil, fmt.Errorf("delete expired conversations before listing: %w", err)
	}
	items, err := s.store.ListConversations(ctx, owner, s.config.ListLimit)
	if err != nil {
		return nil, err
	}
	active := make([]Conversation, 0, len(items))
	for _, item := range items {
		if item.ExpiresAt.After(now) {
			active = append(active, item)
		}
	}
	return active, nil
}

func (s *Service) Delete(ctx context.Context, identity auth.Identity, id string) error {
	item, err := s.Get(ctx, identity, id)
	if err != nil {
		return err
	}
	return s.store.DeleteConversation(ctx, item.Owner, item.ID)
}

func (s *Service) UpdateTitle(ctx context.Context, identity auth.Identity, id string, title string) (Conversation, error) {
	title, err := NormalizeTitle(title)
	if err != nil {
		return Conversation{}, err
	}
	item, err := s.Get(ctx, identity, id)
	if err != nil {
		return Conversation{}, err
	}
	return s.store.UpdateConversationTitle(ctx, item.Owner, item.ID, title, s.now().UTC())
}

type TurnExecution interface {
	Run(context.Context, agent.EmitFunc) error
	Cancel(string) error
	// TurnID is the turn the execution runs, which names its messages, and
	// PromptID the ID its message of the user goes by.
	TurnID() string
	PromptID() string
	// SetPromptID names the message of the user of a new turn with the ID
	// the client gave it, unless the conversation holds it already.
	SetPromptID(context.Context, string) error
	// Snapshot gives the messages the model reads as the execution starts,
	// with their IDs: the history, its first stored messages, and, for a
	// resumed turn, its messages so far; and the number of the turns the
	// bound of the history left out.
	Snapshot() (messages []llm.Message, ids []string, stored int, omitted int)
}

func (t *PreparedTurn) TurnID() string { return t.turnID }

// PromptID is the ID the message of the user of the turn goes by: the one
// the client gave it, or <turn ID>:1.
func (t *PreparedTurn) PromptID() string {
	return MessageID(t.turnID, t.promptID, 1)
}

// SetPromptID records the ID the client gave the message of the user of a
// new turn: the client stays in phase with the conversation on its own IDs.
func (t *PreparedTurn) SetPromptID(ctx context.Context, id string) error {
	if t.resume != nil || !ValidPromptID(id) {
		return fmt.Errorf("%w: invalid prompt ID", ErrInvalid)
	}
	recorded, err := t.service.store.SetTurnPromptID(ctx, t.owner, t.conversationID, t.turnID, id)
	if err != nil {
		return err
	}
	if recorded {
		t.promptID = id
	}
	return nil
}

func (t *PreparedTurn) Snapshot() ([]llm.Message, []string, int, int) {
	messages, ids := slices.Clone(t.history), slices.Clone(t.historyIDs)
	if t.resume != nil {
		for index, message := range t.resume.Messages {
			messages = append(messages, message)
			ids = append(ids, MessageID(t.turnID, t.promptID, int64(index+1)))
		}
		// The iteration the turn waits in: its answer and the results of
		// its calls already run, which the turn stores once it is done.
		if state := t.resume; len(state.Calls) > 0 {
			messages = append(messages, llm.Message{Role: llm.RoleAssistant, Text: state.Text, ToolCalls: slices.Clone(state.Calls)})
			ids = append(ids, fmt.Sprintf("%s:%d", t.turnID, 2*state.Iteration))
			if len(state.Results) > 0 {
				messages = append(messages, llm.Message{Role: llm.RoleTool, ToolResults: slices.Clone(state.Results)})
				ids = append(ids, fmt.Sprintf("%s:%d", t.turnID, 2*state.Iteration+1))
			}
		}
	}
	omitted := 0
	if t.historyStats != nil {
		omitted = t.historyStats.RemovedTurns
	}
	return messages, ids, len(t.history), omitted
}

func (s *Service) PrepareTurn(ctx context.Context, identity auth.Identity, conversationID string, prompt string) (TurnExecution, error) {
	if strings.TrimSpace(prompt) == "" || len(prompt) > maxServicePromptBytes {
		return nil, fmt.Errorf("%w: invalid conversation prompt", ErrInvalid)
	}
	item, err := s.Get(ctx, identity, conversationID)
	if err != nil {
		return nil, err
	}
	turnID, err := s.newID()
	if err != nil {
		return nil, fmt.Errorf("generate conversation turn ID: %w", err)
	}
	startedAt := s.now().UTC()
	turn, err := s.store.BeginTurn(ctx, item.Owner, item.ID, turnID, startedAt)
	if err != nil {
		return nil, err
	}

	history, historyIDs, err := s.store.LoadHistoryIDs(ctx, item.Owner, item.ID)
	if err != nil {
		return nil, errors.Join(err, s.failTurn(item.Owner, item.ID, turn.ID, TurnFailed, "history_failed"))
	}
	history, stats, err := s.prepareHistory(history)
	if err != nil {
		return nil, errors.Join(err, s.failTurn(item.Owner, item.ID, turn.ID, TurnFailed, "history_invalid"))
	}
	return &PreparedTurn{
		service: s, owner: item.Owner, conversationID: item.ID, turnID: turn.ID,
		prompt: prompt, history: history, historyIDs: historyIDs[len(historyIDs)-len(history):], historyStats: stats,
	}, nil
}

// PrepareConfirmation consumes the confirmation a suspended turn waits on and
// returns the execution that resumes the turn with the decision of the user.
// The decision applies to the call stored with the confirmation, the one
// shown to the user, bound by the hash of its arguments.
//
// edited, when not nil, are the arguments the user approved the call with,
// in place of those the model proposed: they are checked against the input
// schema of the tool before the confirmation is used, so that an invalid
// edit leaves it waiting for the user to correct it. With the decision
// preview, they are the arguments the preview is computed again for: the
// confirmation is replaced by a new one, nothing runs.
func (s *Service) PrepareConfirmation(ctx context.Context, identity auth.Identity, conversationID string, turnID string, confirmationID string, decision agent.Decision, edited json.RawMessage) (TurnExecution, error) {
	if !decision.Valid() || (edited != nil && decision == agent.DecisionReject) || (edited == nil && decision == agent.DecisionPreview) {
		return nil, fmt.Errorf("%w: invalid confirmation decision", ErrInvalid)
	}
	item, waiting, waitState, err := s.pendingWait(ctx, identity, conversationID)
	if err != nil {
		return nil, err
	}
	// A turn waiting on the client is resumed by its answers, never by a
	// decision.
	if waiting != nil && waiting.ConfirmationID == confirmationID && waitState.WaitsOnClient() {
		return nil, ErrNotPending
	}
	if edited != nil {
		if waiting == nil || waiting.ConfirmationID != confirmationID || waiting.TurnID != turnID {
			return nil, ErrNotPending
		}
		if err := waitState.ValidateEditedArguments(edited); err != nil {
			return nil, err
		}
	}
	pending, err := s.store.ClaimConfirmation(ctx, item.Owner, item.ID, turnID, confirmationID, s.now().UTC())
	if err != nil {
		return nil, err
	}
	var state agent.SuspendedTurn
	if err := json.Unmarshal(pending.State, &state); err != nil {
		return nil, errors.Join(fmt.Errorf("decode suspended turn: %w", err), s.failTurn(item.Owner, item.ID, turnID, TurnFailed, "confirmation_invalid"))
	}
	call, err := state.PendingCall()
	if err != nil || call.Name != pending.ToolName || agent.ArgumentsHash(call.Arguments) != pending.ArgumentsHash {
		return nil, errors.Join(fmt.Errorf("suspended turn does not hold the confirmed call"), s.failTurn(item.Owner, item.ID, turnID, TurnFailed, "confirmation_invalid"))
	}
	// The decision binds to the call shown, by its hash; the edit replaces
	// its arguments for the run only.
	state.EditedArguments = edited
	history, historyIDs, err := s.store.LoadHistoryIDs(ctx, item.Owner, item.ID)
	if err != nil {
		return nil, errors.Join(err, s.failTurn(item.Owner, item.ID, turnID, TurnFailed, "history_failed"))
	}
	history, stats, err := s.prepareHistory(history)
	if err != nil {
		return nil, errors.Join(err, s.failTurn(item.Owner, item.ID, turnID, TurnFailed, "history_invalid"))
	}
	return &PreparedTurn{
		service: s, owner: item.Owner, conversationID: item.ID, turnID: turnID,
		history: history, historyIDs: historyIDs[len(historyIDs)-len(history):], historyStats: stats, promptID: s.turnPromptID(ctx, item.Owner, item.ID, turnID), resume: &state, decision: decision,
		timing: pending.Timing,
	}, nil
}

// PendingConfirmation returns the unexpired confirmation the conversation
// waits on, or nil, for a client to show it again. A turn waiting on the
// answers of the client to its tools waits on no confirmation.
func (s *Service) PendingConfirmation(ctx context.Context, identity auth.Identity, conversationID string) (*PendingConfirmation, error) {
	_, pending, state, err := s.pendingWait(ctx, identity, conversationID)
	if err != nil || pending == nil || state.WaitsOnClient() {
		return nil, err
	}
	return pending, nil
}

// pendingWait reads what the conversation waits on, and the suspended turn.
func (s *Service) pendingWait(ctx context.Context, identity auth.Identity, conversationID string) (Conversation, *PendingConfirmation, agent.SuspendedTurn, error) {
	var state agent.SuspendedTurn
	item, err := s.Get(ctx, identity, conversationID)
	if err != nil {
		return item, nil, state, err
	}
	pending, err := s.store.GetPendingConfirmation(ctx, item.Owner, item.ID, s.now().UTC())
	if err != nil || pending == nil {
		return item, nil, state, err
	}
	if err := json.Unmarshal(pending.State, &state); err != nil {
		return item, nil, state, fmt.Errorf("decode suspended turn: %w", err)
	}
	return item, pending, state, nil
}

// PendingClientCalls returns the calls of tools of the client the
// conversation waits on, in the order they were made, or none.
func (s *Service) PendingClientCalls(ctx context.Context, identity auth.Identity, conversationID string) ([]string, error) {
	_, pending, state, err := s.pendingWait(ctx, identity, conversationID)
	if err != nil || pending == nil || !state.WaitsOnClient() {
		return nil, err
	}
	return slices.Clone(state.ClientCalls), nil
}

// AbandonClientCalls cancels the turn waiting on the answers of the client
// to the calls of its tools, which a new prompt of the client abandons: the
// protocol requires no answer from a client that leaves them. A turn waiting
// on a confirmation stays.
func (s *Service) AbandonClientCalls(ctx context.Context, identity auth.Identity, conversationID string) error {
	item, pending, state, err := s.pendingWait(ctx, identity, conversationID)
	if err != nil || pending == nil || !state.WaitsOnClient() {
		return err
	}
	return s.store.CancelSuspendedTurn(ctx, item.Owner, item.ID, pending.TurnID, "client_tools_abandoned", s.now().UTC())
}

// PrepareClientResults resumes the turn waiting on the client with its
// answers: exactly one per call of its tools the turn waits on.
func (s *Service) PrepareClientResults(ctx context.Context, identity auth.Identity, conversationID string, results []llm.ToolResult) (TurnExecution, error) {
	item, pending, state, err := s.pendingWait(ctx, identity, conversationID)
	if err != nil {
		return nil, err
	}
	if pending == nil || !state.WaitsOnClient() {
		return nil, ErrNotPending
	}
	answered := make(map[string]bool, len(results))
	for _, result := range results {
		if answered[result.CallID] || !slices.Contains(state.ClientCalls, result.CallID) {
			return nil, ErrInvalidToolResults
		}
		answered[result.CallID] = true
	}
	if len(answered) != len(state.ClientCalls) {
		return nil, ErrInvalidToolResults
	}
	if _, err := s.store.ClaimConfirmation(ctx, item.Owner, item.ID, pending.TurnID, pending.ConfirmationID, s.now().UTC()); err != nil {
		return nil, err
	}
	state.ClientResults = slices.Clone(results)
	history, historyIDs, err := s.store.LoadHistoryIDs(ctx, item.Owner, item.ID)
	if err != nil {
		return nil, errors.Join(err, s.failTurn(item.Owner, item.ID, pending.TurnID, TurnFailed, "history_failed"))
	}
	history, stats, err := s.prepareHistory(history)
	if err != nil {
		return nil, errors.Join(err, s.failTurn(item.Owner, item.ID, pending.TurnID, TurnFailed, "history_invalid"))
	}
	return &PreparedTurn{
		service: s, owner: item.Owner, conversationID: item.ID, turnID: pending.TurnID,
		history: history, historyIDs: historyIDs[len(historyIDs)-len(history):], historyStats: stats, promptID: s.turnPromptID(ctx, item.Owner, item.ID, pending.TurnID), resume: &state,
		timing: pending.Timing,
	}, nil
}

func (s *Service) Recover(ctx context.Context) (int64, error) {
	return s.store.RecoverInterrupted(ctx, s.now().UTC())
}

func (s *Service) DeleteExpired(ctx context.Context) (int64, error) {
	return s.purgeExpired(ctx, s.now().UTC())
}

// purgeExpired deletes the expired conversations, fails the turns whose
// confirmation expired, deleting its state, then deletes the expired
// feedback, and returns the number of conversations deleted.
func (s *Service) purgeExpired(ctx context.Context, at time.Time) (int64, error) {
	var total int64
	for {
		deleted, err := s.store.DeleteExpired(ctx, at, s.config.ExpiryDeleteBatch)
		if err != nil {
			return total, err
		}
		total += deleted
		if deleted < int64(s.config.ExpiryDeleteBatch) {
			break
		}
	}
	for {
		expired, err := s.store.ExpireConfirmations(ctx, at, s.config.ExpiryDeleteBatch)
		if err != nil {
			return total, fmt.Errorf("expire confirmations: %w", err)
		}
		if expired < int64(s.config.ExpiryDeleteBatch) {
			break
		}
	}
	for {
		deleted, err := s.store.DeleteExpiredFeedback(ctx, at, s.config.ExpiryDeleteBatch)
		if err != nil {
			return total, fmt.Errorf("delete expired feedback: %w", err)
		}
		if deleted < int64(s.config.ExpiryDeleteBatch) {
			return total, nil
		}
	}
}

type PreparedTurn struct {
	service        *Service
	owner          Owner
	conversationID string
	turnID         string
	prompt         string
	history        []llm.Message
	// historyIDs are the IDs of the messages of history, the turns the
	// bound of the history leaves out left out.
	historyIDs []string
	// promptID is the ID the client gave the message of the user of the
	// turn, empty when it gave none.
	promptID     string
	historyStats *HistoryStats
	resume       *agent.SuspendedTurn
	decision     agent.Decision
	// timing are the times of the turn: those of its earlier runs, then
	// this one's, recorded as its events go.
	timing  TurnTiming
	mu      sync.Mutex
	claimed bool
}

// msOf is a time in milliseconds since the epoch, as the timing keeps it.
func msOf(at time.Time) int64 { return at.UnixMilli() }

// startRun records the start of a run, at its claim. A turn records at most
// MaxTurnRuns runs: past it, the last one goes on.
func (t *PreparedTurn) startRun(at time.Time) {
	t.timing.Calls = maps.Clone(t.timing.Calls)
	t.timing.Runs = slices.Clone(t.timing.Runs)
	if len(t.timing.Runs) < MaxTurnRuns {
		t.timing.Runs = append(t.timing.Runs, TimingRun{StartedAt: msOf(at)})
	}
}

// endRun records the end of the run, at its suspension or its completion.
func (t *PreparedTurn) endRun(at time.Time) {
	if last := len(t.timing.Runs) - 1; last >= 0 {
		t.timing.Runs[last].EndedAt = max(msOf(at), t.timing.Runs[last].StartedAt)
	}
}

// record notes the times of an event of the run, on the clock of the agent:
// the start and the end of each call of a tool, with the arguments the user
// edited the call they approved with, and the tokens of the model calls.
func (t *PreparedTurn) record(event agent.Event, edited string, editedArguments json.RawMessage) {
	at := event.At
	if at.IsZero() {
		at = t.service.now()
	}
	call := func(id string) (TimingCall, bool) {
		if id == "" {
			return TimingCall{}, false
		}
		value, ok := t.timing.Calls[id]
		if !ok && len(t.timing.Calls) >= maxTimingCalls {
			return TimingCall{}, false
		}
		return value, true
	}
	switch event.Type {
	case agent.EventToolStarted:
		value, ok := call(event.ToolCallID)
		if !ok {
			return
		}
		// A start already given, the one of the run of a decision, stays.
		if value.StartedAt == 0 {
			value.StartedAt = msOf(at)
		}
		if event.ToolCallID == edited {
			value.RanWith = editedArguments
		}
		if t.timing.Calls == nil {
			t.timing.Calls = map[string]TimingCall{}
		}
		t.timing.Calls[event.ToolCallID] = value
	case agent.EventToolFinished:
		value, ok := call(event.ToolCallID)
		if !ok {
			return
		}
		value.EndedAt = msOf(at)
		if t.timing.Calls == nil {
			t.timing.Calls = map[string]TimingCall{}
		}
		t.timing.Calls[event.ToolCallID] = value
	case agent.EventUsage:
		if event.Usage != nil {
			if t.timing.Usage == nil {
				t.timing.Usage = &TimingUsage{}
			}
			t.timing.Usage.TotalTokens += event.Usage.TotalTokens
		}
	}
}

func (t *PreparedTurn) Run(ctx context.Context, emit agent.EmitFunc) error {
	if err := t.claim(); err != nil {
		return err
	}
	// A panic of the runner or of the consumer fails the turn, rather than
	// leave it running until a restart, and goes on to the server. A turn
	// already ended stays as it ended.
	defer func() {
		if r := recover(); r != nil {
			if err := t.service.failTurn(t.owner, t.conversationID, t.turnID, TurnFailed, "agent_panicked"); err != nil && !errors.Is(err, ErrConflict) {
				log.Printf("record the failure of a turn whose run panicked, conversation %s, turn %s: %v", t.conversationID, t.turnID, err)
			}
			panic(r)
		}
	}()
	if emit == nil {
		err := errors.New("conversation event consumer is nil")
		return errors.Join(err, t.service.failTurn(t.owner, t.conversationID, t.turnID, TurnFailed, "consumer_invalid"))
	}
	runStart := t.service.now().UTC()
	t.startRun(runStart)
	// The call a decision answers starts with the run that decides it, as
	// the stream shows it: the run emits its start later, once ready, and
	// emits none for a call the user rejected. Its arguments are those the
	// user edited, if any.
	edited, editedArguments := "", json.RawMessage(nil)
	if t.resume != nil && (t.decision == agent.DecisionApprove || t.decision == agent.DecisionReject) {
		if call, err := t.resume.PendingCall(); err == nil && (t.timing.Calls[call.ID].StartedAt == 0) && len(t.timing.Calls) < maxTimingCalls {
			if t.timing.Calls == nil {
				t.timing.Calls = map[string]TimingCall{}
			}
			t.timing.Calls[call.ID] = TimingCall{StartedAt: msOf(runStart)}
		}
		if call, err := t.resume.PendingCall(); err == nil && t.decision == agent.DecisionApprove && len(t.resume.EditedArguments) > 0 {
			edited, editedArguments = call.ID, t.resume.EditedArguments
		}
	}
	var completedEvent *agent.Event
	forward := func(event agent.Event) error {
		t.record(event, edited, editedArguments)
		if event.Type == agent.EventCompleted {
			if completedEvent != nil {
				return errors.New("conversation runner emitted multiple completion events")
			}
			copy := event
			completedEvent = &copy
			return nil
		}
		if event.Type == agent.EventConfirmationRequired || event.Type == agent.EventClientToolsPending {
			return errors.New("conversation runner emitted the end of a suspended turn")
		}
		return emit(event)
	}
	var result agent.TurnResult
	var runErr error
	if t.resume != nil {
		result, runErr = t.service.runner.ResumeTurn(ctx, t.history, *t.resume, t.decision, forward)
	} else {
		result, runErr = t.service.runner.RunTurn(ctx, t.history, t.prompt, forward)
	}
	// A preview asked again that fails leaves the call waiting, as it was:
	// it decided nothing.
	if runErr != nil && t.decision == agent.DecisionPreview && ctx.Err() == nil {
		state := *t.resume
		state.Preview, state.PreviewArguments, state.EditedArguments = nil, nil, nil
		state.PreviewError = "the preview could not be computed again"
		return t.suspend(state, t.service.config.Agent.run(result.MCP, result.Exposure, t.historyStats), emit)
	}
	if runErr != nil {
		status, code := turnFailure(runErr, ctx.Err())
		return errors.Join(runErr, t.service.failTurn(t.owner, t.conversationID, t.turnID, status, code))
	}
	if result.Suspended != nil {
		if completedEvent != nil {
			err := errors.New("conversation turn both completed and suspended")
			return errors.Join(err, t.service.failTurn(t.owner, t.conversationID, t.turnID, TurnFailed, "agent_incomplete"))
		}
		return t.suspend(*result.Suspended, t.service.config.Agent.run(result.MCP, result.Exposure, t.historyStats), emit)
	}
	if completedEvent == nil {
		err := errors.New("conversation turn completed without a completion event")
		return errors.Join(err, t.service.failTurn(t.owner, t.conversationID, t.turnID, TurnFailed, "agent_incomplete"))
	}
	if result.FinishReason != completedEvent.FinishReason {
		err := errors.New("conversation result finish reason does not match completion event")
		return errors.Join(err, t.service.failTurn(t.owner, t.conversationID, t.turnID, TurnFailed, "agent_incomplete"))
	}
	completedAt := t.service.now().UTC()
	t.endRun(completedAt)
	finalizeCtx, cancel := context.WithTimeout(context.Background(), t.service.config.FinalizeTimeout)
	completeErr := t.service.store.CompleteTurn(
		finalizeCtx, t.owner, t.conversationID, t.turnID, completedAt,
		completedAt.Add(t.service.config.Lifetime), result.Messages,
		t.service.config.Agent.run(result.MCP, result.Exposure, t.historyStats), t.timing,
	)
	cancel()
	if completeErr != nil {
		return errors.Join(completeErr, t.service.failTurn(t.owner, t.conversationID, t.turnID, TurnFailed, "persistence_failed"))
	}
	completedEvent.TurnID = t.turnID
	if err := emit(*completedEvent); err != nil {
		return fmt.Errorf("emit committed conversation completion: %w", err)
	}
	return nil
}

// suspend stores the turn waiting for a confirmation, then asks the client for
// the decision of the user. The request is emitted only once stored, so a
// client can always answer it.
func (t *PreparedTurn) suspend(state agent.SuspendedTurn, run TurnRun, emit agent.EmitFunc) error {
	call, err := state.PendingCall()
	if err != nil {
		return errors.Join(err, t.service.failTurn(t.owner, t.conversationID, t.turnID, TurnFailed, "agent_incomplete"))
	}
	// The call a turn waits on is stored with its arguments: a call of a
	// tool of the client may come without any.
	if len(call.Arguments) == 0 {
		call.Arguments = json.RawMessage(`{}`)
	}
	encoded, err := json.Marshal(state)
	if err != nil {
		return errors.Join(fmt.Errorf("encode suspended turn: %w", err), t.service.failTurn(t.owner, t.conversationID, t.turnID, TurnFailed, "agent_incomplete"))
	}
	confirmationID, err := t.service.newID()
	if err != nil {
		return errors.Join(fmt.Errorf("generate confirmation ID: %w", err), t.service.failTurn(t.owner, t.conversationID, t.turnID, TurnFailed, "agent_incomplete"))
	}
	createdAt := t.service.now().UTC()
	// The run ends where the wait for the decision starts.
	t.endRun(createdAt)
	pending := PendingConfirmation{
		ConversationID: t.conversationID, TurnID: t.turnID, ConfirmationID: confirmationID,
		ToolName: call.Name, ToolTitle: state.ToolTitle, Arguments: call.Arguments,
		ArgumentsHash: agent.ArgumentsHash(call.Arguments), State: encoded,
		CreatedAt: createdAt, ExpiresAt: createdAt.Add(t.service.config.ConfirmationLifetime), Run: run,
		Timing: t.timing,
	}
	suspendCtx, cancel := context.WithTimeout(context.Background(), t.service.config.FinalizeTimeout)
	suspendErr := t.service.store.SuspendTurn(suspendCtx, t.owner, pending)
	cancel()
	if suspendErr != nil {
		return errors.Join(suspendErr, t.service.failTurn(t.owner, t.conversationID, t.turnID, TurnFailed, "persistence_failed"))
	}
	if state.WaitsOnClient() {
		if err := emit(agent.Event{Type: agent.EventClientToolsPending, Iteration: state.Iteration, PendingCalls: slices.Clone(state.ClientCalls)}); err != nil {
			return fmt.Errorf("emit stored client tool calls: %w", err)
		}
		return nil
	}
	if err := emit(agent.Event{Type: agent.EventConfirmationRequired, Iteration: state.Iteration, Confirmation: &agent.Confirmation{
		ID: confirmationID, TurnID: t.turnID, ToolName: call.Name, ToolTitle: state.ToolTitle, Destructive: true,
		ToolCallID: call.ID, Arguments: call.Arguments, ArgumentTypes: state.ArgumentTypes, InputSchema: state.InputSchema, ExpiresAt: pending.ExpiresAt,
		Preview: state.Preview, PreviewError: state.PreviewError, PreviewArguments: state.PreviewArguments,
	}}); err != nil {
		return fmt.Errorf("emit stored confirmation request: %w", err)
	}
	return nil
}

func (t *PreparedTurn) Cancel(code string) error {
	if strings.TrimSpace(code) == "" {
		code = "turn_canceled"
	}
	if err := t.claim(); err != nil {
		return err
	}
	return t.service.failTurn(t.owner, t.conversationID, t.turnID, TurnCanceled, code)
}

func (t *PreparedTurn) claim() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.claimed {
		return fmt.Errorf("%w: conversation turn already consumed", ErrConflict)
	}
	t.claimed = true
	return nil
}

func (s *Service) failTurn(owner Owner, conversationID string, turnID string, status TurnStatus, code string) error {
	ctx, cancel := context.WithTimeout(context.Background(), s.config.FinalizeTimeout)
	defer cancel()
	if err := s.store.FailTurn(ctx, owner, conversationID, turnID, status, code, s.now().UTC()); err != nil {
		return fmt.Errorf("record conversation turn failure: %w", err)
	}
	return nil
}

func turnFailure(err error, contextErr error) (TurnStatus, string) {
	if errors.Is(contextErr, context.Canceled) || errors.Is(err, context.Canceled) {
		return TurnCanceled, "request_canceled"
	}
	if errors.Is(contextErr, context.DeadlineExceeded) || errors.Is(err, context.DeadlineExceeded) {
		return TurnFailed, "request_timeout"
	}
	return TurnFailed, "agent_failed"
}

func (s *Service) ownerFromIdentity(identity auth.Identity) (Owner, error) {
	owner := Owner{Backend: s.config.Backend, ClusterID: strings.TrimSpace(identity.ClusterID), Issuer: strings.TrimSpace(identity.Issuer), Subject: strings.TrimSpace(identity.Subject)}
	// The backend authenticator guarantees a cluster when its identities are
	// bound to one; the issuer and the subject are always required.
	if owner.Issuer == "" || owner.Subject == "" {
		return Owner{}, fmt.Errorf("%w: authenticated conversation owner is invalid", ErrInvalid)
	}
	return owner, nil
}

func randomID() (string, error) {
	data := make([]byte, 16)
	if _, err := rand.Read(data); err != nil {
		return "", err
	}
	return hex.EncodeToString(data), nil
}

// prepareHistory gives the model a compact copy of the stored history,
// within its bounds, and describes it.
func (s *Service) prepareHistory(history []llm.Message) ([]llm.Message, *HistoryStats, error) {
	loaded := historySize(history)
	compacted, stats, compactedAt := compactHistory(history, s.config.HistoryCompaction)
	bounded, err := boundHistory(compacted, s.config.MaxHistoryMessages, s.config.MaxHistoryBytes)
	if err != nil {
		return nil, nil, err
	}
	// The turns boundHistory leaves out, and their compacted results, are
	// not sent: they count as removed, not as compacted.
	removed := len(compacted) - len(bounded)
	for _, message := range compacted[:removed] {
		if message.Role == llm.RoleUser {
			stats.RemovedTurns++
		}
	}
	stats.CompactedResults = 0
	for _, index := range compactedAt {
		if index >= removed {
			stats.CompactedResults++
		}
	}
	stats.LoadedBytes, stats.CompactedBytes, stats.SentBytes = loaded, historySize(compacted), historySize(bounded)
	return bounded, &stats, nil
}

func boundHistory(history []llm.Message, maxMessages int, maxBytes int) ([]llm.Message, error) {
	if len(history) == 0 {
		return nil, nil
	}
	starts := make([]int, 0)
	for index, message := range history {
		if message.Role == llm.RoleUser {
			starts = append(starts, index)
		}
	}
	if len(starts) == 0 || starts[0] != 0 {
		return nil, fmt.Errorf("conversation history has no complete user turn")
	}
	for _, start := range starts {
		candidate := history[start:]
		if len(candidate) > maxMessages {
			continue
		}
		encoded, err := json.Marshal(candidate)
		if err != nil {
			return nil, fmt.Errorf("encode bounded conversation history: %w", err)
		}
		if len(encoded) <= maxBytes {
			return candidate, nil
		}
	}
	return nil, nil
}

// turnPromptID reads the ID the client gave the prompt of a turn, none when
// it cannot be read: the message is then named <turn ID>:1.
func (s *Service) turnPromptID(ctx context.Context, owner Owner, conversationID, turnID string) string {
	turn, _, err := s.store.TurnMessages(ctx, owner, conversationID, turnID)
	if err != nil {
		return ""
	}
	return turn.PromptID
}
