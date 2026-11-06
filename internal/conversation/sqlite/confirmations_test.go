package sqlite

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/opensvc/ai-agent/internal/conversation"
	"github.com/opensvc/ai-agent/internal/llm"
)

func suspendTestTurn(t *testing.T, store *Store, conversationID string, turnID string, confirmationID string, at time.Time) conversation.PendingConfirmation {
	t.Helper()
	if _, err := store.BeginTurn(t.Context(), testOwner, conversationID, turnID, at); err != nil {
		t.Fatalf("BeginTurn() error: %v", err)
	}
	pending := conversation.PendingConfirmation{
		ConversationID: conversationID, TurnID: turnID, ConfirmationID: confirmationID,
		ToolName: "stop_object", ToolTitle: "Stop object", Arguments: json.RawMessage(`{"path":"web"}`),
		ArgumentsHash: "hash", State: []byte(`{"messages":[]}`),
		CreatedAt: at, ExpiresAt: at.Add(10 * time.Minute),
	}
	if err := store.SuspendTurn(t.Context(), testOwner, pending); err != nil {
		t.Fatalf("SuspendTurn() error: %v", err)
	}
	return pending
}

func TestStoreSuspendsAndResumesATurnOnce(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	pending := suspendTestTurn(t, store, item.ID, "turn-1", "confirmation-1", testNow.Add(time.Second))

	got, err := store.GetPendingConfirmation(t.Context(), testOwner, item.ID, testNow.Add(2*time.Second))
	if err != nil || got == nil || got.ConfirmationID != "confirmation-1" || string(got.Arguments) != `{"path":"web"}` || got.ToolTitle != "Stop object" {
		t.Fatalf("GetPendingConfirmation() = %+v, %v", got, err)
	}
	if _, err := store.BeginTurn(t.Context(), testOwner, item.ID, "turn-2", testNow.Add(3*time.Second)); !errors.Is(err, conversation.ErrBusy) {
		t.Fatalf("a new turn began while a confirmation waits: %v", err)
	}
	if _, err := store.ClaimConfirmation(t.Context(), testOwner, item.ID, "turn-1", "confirmation-other", testNow.Add(3*time.Second)); !errors.Is(err, conversation.ErrNotPending) {
		t.Fatalf("a wrong confirmation ID was accepted: %v", err)
	}
	other := conversation.Owner{Backend: "om3", ClusterID: "cluster-id", Issuer: "node-a", Subject: "bob"}
	if _, err := store.ClaimConfirmation(t.Context(), other, item.ID, "turn-1", "confirmation-1", testNow.Add(3*time.Second)); !errors.Is(err, conversation.ErrNotFound) {
		t.Fatalf("another owner claimed the confirmation: %v", err)
	}
	claimed, err := store.ClaimConfirmation(t.Context(), testOwner, item.ID, "turn-1", "confirmation-1", testNow.Add(4*time.Second))
	if err != nil || string(claimed.State) != string(pending.State) || claimed.ArgumentsHash != "hash" {
		t.Fatalf("ClaimConfirmation() = %+v, %v", claimed, err)
	}
	if _, err := store.ClaimConfirmation(t.Context(), testOwner, item.ID, "turn-1", "confirmation-1", testNow.Add(5*time.Second)); !errors.Is(err, conversation.ErrNotPending) {
		t.Fatalf("a confirmation was used twice: %v", err)
	}
	if got, _ := store.GetPendingConfirmation(t.Context(), testOwner, item.ID, testNow.Add(5*time.Second)); got != nil {
		t.Fatalf("a claimed confirmation is still pending: %+v", got)
	}
	messages := []llm.Message{{Role: llm.RoleUser, Text: "stop web"}, {Role: llm.RoleAssistant, Text: "stopped"}}
	if err := store.CompleteTurn(t.Context(), testOwner, item.ID, "turn-1", testNow.Add(6*time.Second), testNow.Add(time.Hour), messages, conversation.TurnRun{}, conversation.TurnTiming{}); err != nil {
		t.Fatalf("the resumed turn did not complete: %v", err)
	}
}

func TestStoreExpiresAConfirmationWithoutResumingItsTurn(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	pending := suspendTestTurn(t, store, item.ID, "turn-1", "confirmation-1", testNow.Add(time.Second))
	late := pending.ExpiresAt.Add(time.Second)
	if got, _ := store.GetPendingConfirmation(t.Context(), testOwner, item.ID, late); got != nil {
		t.Fatalf("an expired confirmation is reported pending: %+v", got)
	}
	if _, err := store.ClaimConfirmation(t.Context(), testOwner, item.ID, "turn-1", "confirmation-1", late); !errors.Is(err, conversation.ErrConfirmationExpired) {
		t.Fatalf("an expired confirmation was accepted: %v", err)
	}
	if _, err := store.ClaimConfirmation(t.Context(), testOwner, item.ID, "turn-1", "confirmation-1", late); !errors.Is(err, conversation.ErrNotPending) {
		t.Fatalf("an expired confirmation stays claimable: %v", err)
	}
	if _, err := store.BeginTurn(t.Context(), testOwner, item.ID, "turn-2", late); err != nil {
		t.Fatalf("the conversation stays blocked after the expiry: %v", err)
	}
}

func TestStoreLetsANewTurnReplaceAnExpiredConfirmation(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	pending := suspendTestTurn(t, store, item.ID, "turn-1", "confirmation-1", testNow.Add(time.Second))
	if _, err := store.BeginTurn(t.Context(), testOwner, item.ID, "turn-2", pending.ExpiresAt.Add(time.Second)); err != nil {
		t.Fatalf("a new turn did not replace the expired confirmation: %v", err)
	}
	if _, err := store.ClaimConfirmation(t.Context(), testOwner, item.ID, "turn-1", "confirmation-1", pending.ExpiresAt.Add(time.Minute)); !errors.Is(err, conversation.ErrNotPending) {
		t.Fatalf("the replaced confirmation stays claimable: %v", err)
	}
}

func TestStoreKeepsAwaitingTurnsAcrossRecoveryAndDeletesThemWithTheConversation(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	suspendTestTurn(t, store, item.ID, "turn-1", "confirmation-1", testNow.Add(time.Second))
	if recovered, err := store.RecoverInterrupted(t.Context(), testNow.Add(2*time.Second)); err != nil || recovered != 0 {
		t.Fatalf("recovery interrupted an awaiting turn: %d %v", recovered, err)
	}
	if got, _ := store.GetPendingConfirmation(t.Context(), testOwner, item.ID, testNow.Add(3*time.Second)); got == nil {
		t.Fatal("a restart lost the pending confirmation")
	}
	if err := store.DeleteConversation(t.Context(), testOwner, item.ID); err != nil {
		t.Fatalf("an awaiting conversation cannot be deleted: %v", err)
	}
	var count int
	if err := store.db.QueryRowContext(t.Context(), "SELECT COUNT(*) FROM confirmations").Scan(&count); err != nil || count != 0 {
		t.Fatalf("confirmations left after deletion: %d %v", count, err)
	}
}

func TestStoreRejectsInvalidConfirmations(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	if _, err := store.BeginTurn(t.Context(), testOwner, item.ID, "turn-1", testNow); err != nil {
		t.Fatal(err)
	}
	valid := conversation.PendingConfirmation{
		ConversationID: item.ID, TurnID: "turn-1", ConfirmationID: "confirmation-1", ToolName: "stop_object",
		Arguments: json.RawMessage(`{}`), ArgumentsHash: "hash", State: []byte(`{}`), CreatedAt: testNow, ExpiresAt: testNow.Add(time.Minute),
	}
	for name, mutate := range map[string]func(*conversation.PendingConfirmation){
		"no tool":         func(p *conversation.PendingConfirmation) { p.ToolName = "" },
		"no arguments":    func(p *conversation.PendingConfirmation) { p.Arguments = nil },
		"no state":        func(p *conversation.PendingConfirmation) { p.State = nil },
		"already expired": func(p *conversation.PendingConfirmation) { p.ExpiresAt = p.CreatedAt },
		"unknown turn":    func(p *conversation.PendingConfirmation) { p.TurnID = "turn-2" },
	} {
		pending := valid
		mutate(&pending)
		if err := store.SuspendTurn(t.Context(), testOwner, pending); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
}

// A suspended turn the client abandons is canceled with its code, and the
// conversation takes a new turn.
func TestStoreCancelsASuspendedTurn(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	suspendTestTurn(t, store, item.ID, "turn-1", "confirmation-1", testNow.Add(time.Second))
	if err := store.CancelSuspendedTurn(t.Context(), testOwner, item.ID, "turn-1", "client_tools_abandoned", testNow.Add(2*time.Second)); err != nil {
		t.Fatal(err)
	}
	if got, err := store.GetPendingConfirmation(t.Context(), testOwner, item.ID, testNow.Add(3*time.Second)); err != nil || got != nil {
		t.Fatalf("GetPendingConfirmation() = %+v, %v", got, err)
	}
	turn, _, err := store.TurnMessages(t.Context(), testOwner, item.ID, "turn-1")
	if err != nil || turn.Status != conversation.TurnCanceled || turn.ErrorCode != "client_tools_abandoned" {
		t.Fatalf("turn = %+v, %v", turn, err)
	}
	if _, err := store.BeginTurn(t.Context(), testOwner, item.ID, "turn-2", testNow.Add(4*time.Second)); err != nil {
		t.Fatalf("a new turn did not begin: %v", err)
	}
	if err := store.CancelSuspendedTurn(t.Context(), testOwner, item.ID, "turn-1", "client_tools_abandoned", testNow.Add(5*time.Second)); !errors.Is(err, conversation.ErrNotPending) {
		t.Fatalf("second cancellation: %v", err)
	}
}

// A database of the version 5 is converted, through the version 6, its
// conversations kept.
func TestOpenConvertsTheVersion5(t *testing.T) {
	store, path := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	for _, statement := range []string{"ALTER TABLE turns DROP COLUMN timing_json", "ALTER TABLE turns DROP COLUMN prompt_id", "PRAGMA user_version = 5"} {
		if _, err := store.db.ExecContext(t.Context(), statement); err != nil {
			t.Fatal(err)
		}
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := Open(t.Context(), Config{Path: path})
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	var version, conversations int
	if err := reopened.db.QueryRowContext(t.Context(), "PRAGMA user_version").Scan(&version); err != nil || version != schemaVersion {
		t.Fatalf("version = %d, %v", version, err)
	}
	if err := reopened.db.QueryRowContext(t.Context(), "SELECT COUNT(*) FROM conversations WHERE id = ?", item.ID).Scan(&conversations); err != nil || conversations != 1 {
		t.Fatalf("conversations = %d, %v", conversations, err)
	}
	if _, err := reopened.db.ExecContext(t.Context(), "SELECT prompt_id, timing_json FROM turns"); err != nil {
		t.Fatalf("no prompt_id or timing_json: %v", err)
	}
}

// The ID the client gave its prompt names the message of the user of the
// turn in the history and in the display messages, once per conversation.
func TestStoreNamesThePromptWithTheIDOfTheClient(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	for index, turnID := range []string{"turn-1", "turn-2"} {
		at := testNow.Add(time.Duration(index+1) * time.Minute)
		if _, err := store.BeginTurn(t.Context(), testOwner, item.ID, turnID, at); err != nil {
			t.Fatal(err)
		}
		recorded, err := store.SetTurnPromptID(t.Context(), testOwner, item.ID, turnID, "client-u1")
		if err != nil || recorded != (index == 0) {
			t.Fatalf("%s: recorded %v, %v", turnID, recorded, err)
		}
		messages := []llm.Message{{Role: llm.RoleUser, Text: "question"}, {Role: llm.RoleAssistant, Text: "answer"}}
		if err := store.CompleteTurn(t.Context(), testOwner, item.ID, turnID, at.Add(time.Second), at.Add(time.Hour), messages, conversation.TurnRun{}, conversation.TurnTiming{}); err != nil {
			t.Fatal(err)
		}
	}
	_, ids, err := store.LoadHistoryIDs(t.Context(), testOwner, item.ID)
	if err != nil || strings.Join(ids, ",") != "client-u1,turn-1:2,turn-2:1,turn-2:2" {
		t.Fatalf("ids = %v, %v", ids, err)
	}
	page, err := store.ListMessages(t.Context(), testOwner, item.ID, conversation.MessageQuery{})
	if err != nil || len(page.Messages) != 4 || page.Messages[0].ID != "client-u1" || page.Messages[2].ID != "turn-2:1" {
		t.Fatalf("page = %+v, %v", page, err)
	}
	if _, err := store.SetTurnPromptID(t.Context(), testOwner, item.ID, "turn-1", "has space"); err == nil {
		t.Fatal("an invalid prompt ID was recorded")
	}
}

// TestStoreExpiresAbandonedConfirmations fails the turns whose confirmation
// expired and deletes their state, without waiting for a new turn of their
// conversation; a confirmation still pending stays.
func TestStoreExpiresAbandonedConfirmations(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	for _, id := range []string{"conversation-1", "conversation-2"} {
		if err := store.CreateConversation(t.Context(), testConversation(id, testOwner, testNow)); err != nil {
			t.Fatal(err)
		}
	}
	suspendTestTurn(t, store, "conversation-1", "turn-1", "confirmation-1", testNow)
	suspendTestTurn(t, store, "conversation-2", "turn-2", "confirmation-2", testNow.Add(5*time.Minute))
	if _, err := store.ExpireConfirmations(t.Context(), testNow, 0); !errors.Is(err, conversation.ErrInvalid) {
		t.Fatalf("a zero limit: %v", err)
	}
	expired, err := store.ExpireConfirmations(t.Context(), testNow.Add(11*time.Minute), 10)
	if err != nil || expired != 1 {
		t.Fatalf("ExpireConfirmations() = %d, %v", expired, err)
	}
	var status, code string
	if err := store.db.QueryRowContext(t.Context(), "SELECT status, error_code FROM turns WHERE id = 'turn-1'").Scan(&status, &code); err != nil {
		t.Fatal(err)
	}
	var left int
	if err := store.db.QueryRowContext(t.Context(), "SELECT COUNT(*) FROM confirmations WHERE turn_id = 'turn-1'").Scan(&left); err != nil {
		t.Fatal(err)
	}
	if status != string(conversation.TurnFailed) || code != confirmationExpiredCode || left != 0 {
		t.Fatalf("turn-1 %s %q, %d confirmations left", status, code, left)
	}
	if got, err := store.GetPendingConfirmation(t.Context(), testOwner, "conversation-2", testNow.Add(11*time.Minute)); err != nil || got == nil {
		t.Fatalf("the pending confirmation: %+v, %v", got, err)
	}
}
