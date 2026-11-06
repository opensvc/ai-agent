package sqlite

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/opensvc/ai-agent/internal/conversation"
	"github.com/opensvc/ai-agent/internal/llm"
)

var (
	testOwner = conversation.Owner{Backend: "om3", ClusterID: "cluster-id", Issuer: "node-a", Subject: "alice"}
	testNow   = time.Date(2026, 7, 24, 10, 0, 0, 123456789, time.UTC)
)

func TestStorePersistsConversationAndToolHistory(t *testing.T) {
	store, path := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatalf("CreateConversation() error: %v", err)
	}
	if err := store.CreateConversation(t.Context(), item); !errors.Is(err, conversation.ErrConflict) {
		t.Fatalf("duplicate CreateConversation() error = %v", err)
	}
	if _, err := store.GetConversation(t.Context(), conversation.Owner{Backend: "om3", ClusterID: "cluster-id", Issuer: "node-a", Subject: "bob"}, item.ID); !errors.Is(err, conversation.ErrNotFound) {
		t.Fatalf("cross-owner GetConversation() error = %v", err)
	}
	otherBackend := testOwner
	otherBackend.Backend = "oc3"
	if _, err := store.GetConversation(t.Context(), otherBackend, item.ID); !errors.Is(err, conversation.ErrNotFound) {
		t.Fatalf("cross-backend GetConversation() error = %v", err)
	}
	if listed, err := store.ListConversations(t.Context(), otherBackend, 10); err != nil || len(listed) != 0 {
		t.Fatalf("cross-backend ListConversations() = %#v, %v", listed, err)
	}
	got, err := store.GetConversation(t.Context(), testOwner, item.ID)
	if err != nil {
		t.Fatalf("GetConversation() error: %v", err)
	}
	if got.ID != item.ID || got.Owner != item.Owner || !got.CreatedAt.Equal(item.CreatedAt) || got.StoredBytes != 0 {
		t.Fatalf("GetConversation() = %#v", got)
	}
	listed, err := store.ListConversations(t.Context(), testOwner, 10)
	if err != nil || len(listed) != 1 || listed[0].ID != item.ID {
		t.Fatalf("ListConversations() = %#v, %v", listed, err)
	}

	turn, err := store.BeginTurn(t.Context(), testOwner, item.ID, "turn-1", testNow.Add(time.Second))
	if err != nil {
		t.Fatalf("BeginTurn() error: %v", err)
	}
	if turn.Sequence != 1 || turn.Status != conversation.TurnRunning {
		t.Fatalf("BeginTurn() = %#v", turn)
	}
	messages := toolTranscript()
	if err := store.CompleteTurn(t.Context(), testOwner, item.ID, turn.ID, testNow.Add(2*time.Second), testNow.Add(time.Hour), messages, conversation.TurnRun{}, conversation.TurnTiming{}); err != nil {
		t.Fatalf("CompleteTurn() error: %v", err)
	}
	history, err := store.LoadHistory(t.Context(), testOwner, item.ID)
	if err != nil {
		t.Fatalf("LoadHistory() error: %v", err)
	}
	if !reflect.DeepEqual(history, messages) {
		t.Fatalf("LoadHistory() = %#v, want %#v", history, messages)
	}
	got, err = store.GetConversation(t.Context(), testOwner, item.ID)
	if err != nil || got.Title != conversation.TitleFromPrompt(messages[0].Text) || got.StoredBytes <= 0 || !got.UpdatedAt.Equal(testNow.Add(2*time.Second)) || !got.ExpiresAt.Equal(testNow.Add(time.Hour)) {
		t.Fatalf("completed conversation = %#v, %v", got, err)
	}
	var journalMode string
	if err := store.db.QueryRowContext(t.Context(), "PRAGMA journal_mode").Scan(&journalMode); err != nil || journalMode != "wal" {
		t.Fatalf("journal mode = %q, %v", journalMode, err)
	}
	info, err := os.Stat(path)
	if err != nil || info.Mode().Perm() != 0o600 {
		t.Fatalf("database mode = %v, %v", info.Mode().Perm(), err)
	}

	if err := store.Close(); err != nil {
		t.Fatalf("Close() error: %v", err)
	}
	reopened, err := Open(t.Context(), Config{Path: path})
	if err != nil {
		t.Fatalf("reopen store: %v", err)
	}
	t.Cleanup(func() { _ = reopened.Close() })
	history, err = reopened.LoadHistory(t.Context(), testOwner, item.ID)
	if err != nil || !reflect.DeepEqual(history, messages) {
		t.Fatalf("reopened LoadHistory() = %#v, %v", history, err)
	}
	if err := reopened.DeleteConversation(t.Context(), conversation.Owner{Backend: "om3", ClusterID: "cluster-id", Issuer: "node-a", Subject: "bob"}, item.ID); !errors.Is(err, conversation.ErrNotFound) {
		t.Fatalf("cross-owner DeleteConversation() error = %v", err)
	}
	if err := reopened.DeleteConversation(t.Context(), testOwner, item.ID); err != nil {
		t.Fatalf("DeleteConversation() error: %v", err)
	}
	if _, err := reopened.GetConversation(t.Context(), testOwner, item.ID); !errors.Is(err, conversation.ErrNotFound) {
		t.Fatalf("deleted GetConversation() error = %v", err)
	}
	for _, table := range []string{"turns", "messages"} {
		var count int
		if err := reopened.db.QueryRowContext(t.Context(), "SELECT COUNT(*) FROM "+table).Scan(&count); err != nil || count != 0 {
			t.Fatalf("%s count = %d, %v", table, count, err)
		}
	}
}

func TestStoreUpdatesTitleWithoutOverwritingItOnCompletion(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	updated, err := store.UpdateConversationTitle(t.Context(), testOwner, item.ID, "Database incident", testNow.Add(time.Second))
	if err != nil {
		t.Fatalf("UpdateConversationTitle() error: %v", err)
	}
	if updated.Title != "Database incident" || !updated.UpdatedAt.Equal(testNow.Add(time.Second)) {
		t.Fatalf("UpdateConversationTitle() = %#v", updated)
	}
	if _, err := store.UpdateConversationTitle(t.Context(), conversation.Owner{Backend: "om3", ClusterID: "cluster-id", Issuer: "node-a", Subject: "bob"}, item.ID, "Foreign", testNow.Add(time.Second)); !errors.Is(err, conversation.ErrNotFound) {
		t.Fatalf("cross-owner UpdateConversationTitle() error = %v", err)
	}
	if _, err := store.UpdateConversationTitle(t.Context(), testOwner, item.ID, " not normalized ", testNow.Add(time.Second)); !errors.Is(err, conversation.ErrInvalid) {
		t.Fatalf("unnormalized UpdateConversationTitle() error = %v", err)
	}
	turn, err := store.BeginTurn(t.Context(), testOwner, item.ID, "turn-1", testNow.Add(2*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if err := store.CompleteTurn(t.Context(), testOwner, item.ID, turn.ID, testNow.Add(3*time.Second), testNow.Add(time.Hour), []llm.Message{
		{Role: llm.RoleUser, Text: "This prompt must not replace the title"},
		{Role: llm.RoleAssistant, Text: "done"},
	}, conversation.TurnRun{}, conversation.TurnTiming{}); err != nil {
		t.Fatal(err)
	}
	got, err := store.GetConversation(t.Context(), testOwner, item.ID)
	if err != nil || got.Title != "Database incident" {
		t.Fatalf("completed conversation = %#v, %v", got, err)
	}
}

func TestStoreSerializesTurnsAndRecoversInterrupted(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatalf("CreateConversation() error: %v", err)
	}

	const workers = 8
	start := make(chan struct{})
	errorsByWorker := make(chan error, workers)
	var wg sync.WaitGroup
	for index := 0; index < workers; index++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			<-start
			_, err := store.BeginTurn(context.Background(), testOwner, item.ID, fmt.Sprintf("turn-%d", index), testNow.Add(time.Second))
			errorsByWorker <- err
		}(index)
	}
	close(start)
	wg.Wait()
	close(errorsByWorker)
	succeeded := 0
	busy := 0
	for err := range errorsByWorker {
		switch {
		case err == nil:
			succeeded++
		case errors.Is(err, conversation.ErrBusy):
			busy++
		default:
			t.Fatalf("concurrent BeginTurn() error: %v", err)
		}
	}
	if succeeded != 1 || busy != workers-1 {
		t.Fatalf("concurrent BeginTurn() succeeded=%d busy=%d", succeeded, busy)
	}

	recovered, err := store.RecoverInterrupted(t.Context(), testNow.Add(2*time.Second))
	if err != nil || recovered != 1 {
		t.Fatalf("RecoverInterrupted() = %d, %v", recovered, err)
	}
	turn, err := store.BeginTurn(t.Context(), testOwner, item.ID, "turn-after-recovery", testNow.Add(3*time.Second))
	if err != nil || turn.Sequence != 2 {
		t.Fatalf("BeginTurn() after recovery = %#v, %v", turn, err)
	}
	if err := store.FailTurn(t.Context(), testOwner, item.ID, turn.ID, conversation.TurnCanceled, "client_canceled", testNow.Add(4*time.Second)); err != nil {
		t.Fatalf("FailTurn() error: %v", err)
	}
	if history, err := store.LoadHistory(t.Context(), testOwner, item.ID); err != nil || len(history) != 0 {
		t.Fatalf("failed turn history = %#v, %v", history, err)
	}
	if got, err := store.GetConversation(t.Context(), testOwner, item.ID); err != nil || got.Title != "" {
		t.Fatalf("failed turn conversation = %#v, %v", got, err)
	}
	var interrupted, canceled int
	if err := store.db.QueryRowContext(t.Context(), "SELECT COUNT(*) FROM turns WHERE status = 'interrupted'").Scan(&interrupted); err != nil {
		t.Fatal(err)
	}
	if err := store.db.QueryRowContext(t.Context(), "SELECT COUNT(*) FROM turns WHERE status = 'canceled'").Scan(&canceled); err != nil {
		t.Fatal(err)
	}
	if interrupted != 1 || canceled != 1 {
		t.Fatalf("turn status counts interrupted=%d canceled=%d", interrupted, canceled)
	}
}

func TestStoreDoesNotDeleteConversationWithRunningTurn(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	item.ExpiresAt = testNow.Add(time.Minute)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	turn, err := store.BeginTurn(t.Context(), testOwner, item.ID, "turn-1", testNow.Add(time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if err := store.DeleteConversation(t.Context(), testOwner, item.ID); !errors.Is(err, conversation.ErrBusy) {
		t.Fatalf("delete running conversation error=%v", err)
	}
	deleted, err := store.DeleteExpired(t.Context(), testNow.Add(2*time.Minute), 10)
	if err != nil || deleted != 0 {
		t.Fatalf("DeleteExpired()=%d, %v", deleted, err)
	}
	if err := store.FailTurn(t.Context(), testOwner, item.ID, turn.ID, conversation.TurnCanceled, "request_canceled", testNow.Add(3*time.Second)); err != nil {
		t.Fatal(err)
	}
	deleted, err = store.DeleteExpired(t.Context(), testNow.Add(2*time.Minute), 10)
	if err != nil || deleted != 1 {
		t.Fatalf("DeleteExpired() after turn=%d, %v", deleted, err)
	}
}

func TestStoreEnforcesLimitsWithoutPartialHistory(t *testing.T) {
	store, _ := openTestStore(t, Config{MaxConversations: 1, MaxTurns: 2, MaxMessages: 2})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatalf("CreateConversation() error: %v", err)
	}
	// The limit of the instance, for another user: the first is held by
	// its own quota, of one conversation as the instance's.
	other := conversation.Owner{Backend: testOwner.Backend, ClusterID: testOwner.ClusterID, Issuer: testOwner.Issuer, Subject: "bob"}
	if err := store.CreateConversation(t.Context(), testConversation("conversation-2", other, testNow)); !errors.Is(err, conversation.ErrLimit) || errors.Is(err, conversation.ErrUserLimit) {
		t.Fatalf("conversation limit error = %v", err)
	}
	turn1, err := store.BeginTurn(t.Context(), testOwner, item.ID, "turn-1", testNow.Add(time.Second))
	if err != nil {
		t.Fatal(err)
	}
	first := []llm.Message{{Role: llm.RoleUser, Text: "health"}, {Role: llm.RoleAssistant, Text: "healthy"}}
	if err := store.CompleteTurn(t.Context(), testOwner, item.ID, turn1.ID, testNow.Add(2*time.Second), testNow.Add(time.Hour), first, conversation.TurnRun{}, conversation.TurnTiming{}); err != nil {
		t.Fatal(err)
	}
	turn2, err := store.BeginTurn(t.Context(), testOwner, item.ID, "turn-2", testNow.Add(3*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	second := []llm.Message{{Role: llm.RoleUser, Text: "redis"}, {Role: llm.RoleAssistant, Text: "up"}}
	if err := store.CompleteTurn(t.Context(), testOwner, item.ID, turn2.ID, testNow.Add(4*time.Second), testNow.Add(time.Hour), second, conversation.TurnRun{}, conversation.TurnTiming{}); !errors.Is(err, conversation.ErrLimit) {
		t.Fatalf("message limit error = %v", err)
	}
	history, err := store.LoadHistory(t.Context(), testOwner, item.ID)
	if err != nil || !reflect.DeepEqual(history, first) {
		t.Fatalf("history after rejected completion = %#v, %v", history, err)
	}
	if err := store.FailTurn(t.Context(), testOwner, item.ID, turn2.ID, conversation.TurnFailed, "message_limit", testNow.Add(5*time.Second)); err != nil {
		t.Fatalf("FailTurn() after rejected completion: %v", err)
	}
	if _, err := store.BeginTurn(t.Context(), testOwner, item.ID, "turn-3", testNow.Add(6*time.Second)); !errors.Is(err, conversation.ErrLimit) {
		t.Fatalf("turn limit error = %v", err)
	}
}

func TestStoreRejectsInvalidAndOversizedCompletion(t *testing.T) {
	store, _ := openTestStore(t, Config{MaxConversationBytes: 128, MaxDatabaseBytes: 128})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	turn, err := store.BeginTurn(t.Context(), testOwner, item.ID, "turn-1", testNow.Add(time.Second))
	if err != nil {
		t.Fatal(err)
	}
	invalid := []llm.Message{{Role: llm.RoleSystem, Text: "do not store"}}
	if err := store.CompleteTurn(t.Context(), testOwner, item.ID, turn.ID, testNow.Add(2*time.Second), testNow.Add(time.Hour), invalid, conversation.TurnRun{}, conversation.TurnTiming{}); !errors.Is(err, conversation.ErrInvalid) {
		t.Fatalf("invalid completion error = %v", err)
	}
	oversized := []llm.Message{
		{Role: llm.RoleUser, Text: strings.Repeat("x", 100)},
		{Role: llm.RoleAssistant, Text: strings.Repeat("y", 100)},
	}
	if err := store.CompleteTurn(t.Context(), testOwner, item.ID, turn.ID, testNow.Add(2*time.Second), testNow.Add(time.Hour), oversized, conversation.TurnRun{}, conversation.TurnTiming{}); !errors.Is(err, conversation.ErrLimit) {
		t.Fatalf("oversized completion error = %v", err)
	}
	if history, err := store.LoadHistory(t.Context(), testOwner, item.ID); err != nil || len(history) != 0 {
		t.Fatalf("oversized completion history = %#v, %v", history, err)
	}
}

func TestStoreDeletesExpiredConversationsInBatches(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	for index, expiresAt := range []time.Time{testNow.Add(-time.Hour), testNow.Add(-time.Minute), testNow.Add(time.Hour)} {
		item := testConversation(fmt.Sprintf("conversation-%d", index), testOwner, testNow.Add(-2*time.Hour))
		item.ExpiresAt = expiresAt
		if err := store.CreateConversation(t.Context(), item); err != nil {
			t.Fatal(err)
		}
	}
	deleted, err := store.DeleteExpired(t.Context(), testNow, 1)
	if err != nil || deleted != 1 {
		t.Fatalf("first DeleteExpired() = %d, %v", deleted, err)
	}
	deleted, err = store.DeleteExpired(t.Context(), testNow, 10)
	if err != nil || deleted != 1 {
		t.Fatalf("second DeleteExpired() = %d, %v", deleted, err)
	}
	items, err := store.ListConversations(t.Context(), testOwner, 10)
	if err != nil || len(items) != 1 || items[0].ID != "conversation-2" {
		t.Fatalf("remaining conversations = %#v, %v", items, err)
	}
}

func TestOpenRejectsUnsafeFilesAndUnsupportedSchema(t *testing.T) {
	t.Run("directory permissions", func(t *testing.T) {
		directory := filepath.Join(t.TempDir(), "state")
		if err := os.Mkdir(directory, 0o755); err != nil {
			t.Fatal(err)
		}
		if _, err := Open(t.Context(), Config{Path: filepath.Join(directory, "conversations.db")}); err == nil || !strings.Contains(err.Error(), "directory permissions") {
			t.Fatalf("Open() error = %v", err)
		}
	})
	t.Run("file permissions", func(t *testing.T) {
		directory := secureTempDir(t)
		path := filepath.Join(directory, "conversations.db")
		if err := os.WriteFile(path, nil, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := Open(t.Context(), Config{Path: path}); err == nil || !strings.Contains(err.Error(), "file permissions") {
			t.Fatalf("Open() error = %v", err)
		}
	})
	t.Run("symlink", func(t *testing.T) {
		directory := secureTempDir(t)
		target := filepath.Join(directory, "target")
		if err := os.WriteFile(target, nil, 0o600); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(directory, "conversations.db")
		if err := os.Symlink(target, path); err != nil {
			t.Fatal(err)
		}
		if _, err := Open(t.Context(), Config{Path: path}); err == nil || !strings.Contains(err.Error(), "regular file") {
			t.Fatalf("Open() error = %v", err)
		}
	})
	t.Run("unsupported schema", func(t *testing.T) {
		store, path := openTestStore(t, Config{})
		// The previous schema is refused, not converted.
		if _, err := store.db.ExecContext(t.Context(), "PRAGMA user_version = 1"); err != nil {
			t.Fatal(err)
		}
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}
		if _, err := Open(t.Context(), Config{Path: path}); err == nil || !strings.Contains(err.Error(), "unsupported") {
			t.Fatalf("Open() error = %v", err)
		}
	})
	t.Run("corrupt database", func(t *testing.T) {
		directory := secureTempDir(t)
		path := filepath.Join(directory, "conversations.db")
		if err := os.WriteFile(path, []byte("not sqlite"), 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := Open(t.Context(), Config{Path: path}); err == nil {
			t.Fatal("Open() accepted corrupt database")
		}
	})
}

func TestOpenInitializesCurrentSchema(t *testing.T) {
	store, path := openTestStore(t, Config{})
	var version int
	if err := store.db.QueryRowContext(t.Context(), "PRAGMA user_version").Scan(&version); err != nil || version != schemaVersion {
		t.Fatalf("schema version = %d, %v", version, err)
	}
	item := testConversation("conversation-1", testOwner, testNow)
	item.Title = "Current schema"
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := Open(t.Context(), Config{Path: path})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reopened.Close() })
	got, err := reopened.GetConversation(t.Context(), testOwner, item.ID)
	if err != nil || got.Title != item.Title || got.Owner != item.Owner {
		t.Fatalf("current schema did not preserve owner/title on reopen: %+v, %v", got, err)
	}
}

func TestOpenRejectsUnrecognizedSchemaWithoutChangingData(t *testing.T) {
	path := filepath.Join(secureTempDir(t), "conversations.db")
	file, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if _, err := db.ExecContext(t.Context(), "CREATE TABLE unrelated (value TEXT); INSERT INTO unrelated VALUES ('keep')"); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(t.Context(), Config{Path: path}); err == nil || !strings.Contains(err.Error(), "unrecognized schema") {
		t.Fatalf("unrecognized database accepted: %v", err)
	}
	var value string
	if err := db.QueryRowContext(t.Context(), "SELECT value FROM unrelated").Scan(&value); err != nil || value != "keep" {
		t.Fatalf("refused database changed: %q, %v", value, err)
	}
	var count int
	if err := db.QueryRowContext(t.Context(), "SELECT COUNT(*) FROM sqlite_schema WHERE type = 'table' AND name = 'conversations'").Scan(&count); err != nil || count != 0 {
		t.Fatalf("refused database was initialized: %d, %v", count, err)
	}
}

func TestLoadHistoryRejectsCorruptRows(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	turn, err := store.BeginTurn(t.Context(), testOwner, item.ID, "turn-1", testNow.Add(time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if err := store.CompleteTurn(t.Context(), testOwner, item.ID, turn.ID, testNow.Add(2*time.Second), testNow.Add(time.Hour), []llm.Message{
		{Role: llm.RoleUser, Text: "health"},
		{Role: llm.RoleAssistant, Text: "healthy"},
	}, conversation.TurnRun{}, conversation.TurnTiming{}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.db.ExecContext(t.Context(), `
UPDATE messages
SET tool_calls_json = 'x', stored_bytes = stored_bytes - 1
WHERE turn_id = ? AND sequence = 1`, turn.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := store.LoadHistory(t.Context(), testOwner, item.ID); err == nil || !strings.Contains(err.Error(), "decode conversation tool calls") {
		t.Fatalf("LoadHistory() error = %v", err)
	}
}

func TestStoreHonorsCanceledContext(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.ListConversations(ctx, testOwner, 10); !errors.Is(err, context.Canceled) {
		t.Fatalf("ListConversations() error = %v", err)
	}
}

func openTestStore(t *testing.T, config Config) (*Store, string) {
	t.Helper()
	directory := secureTempDir(t)
	path := filepath.Join(directory, "conversations.db")
	config.Path = path
	store, err := Open(t.Context(), config)
	if err != nil {
		t.Fatalf("Open() error: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	return store, path
}

func secureTempDir(t *testing.T) string {
	t.Helper()
	directory := t.TempDir()
	if err := os.Chmod(directory, 0o700); err != nil {
		t.Fatalf("chmod temp directory: %v", err)
	}
	return directory
}

func testConversation(id string, owner conversation.Owner, now time.Time) conversation.Conversation {
	return conversation.Conversation{
		ID:        id,
		Owner:     owner,
		CreatedAt: now,
		UpdatedAt: now,
		ExpiresAt: now.Add(24 * time.Hour),
	}
}

func toolTranscript() []llm.Message {
	return []llm.Message{
		{Role: llm.RoleUser, Text: "health"},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "get_cluster_health", Arguments: json.RawMessage(`{}`)}}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call-1", Content: json.RawMessage(`{"status":"healthy"}`)}}},
		{Role: llm.RoleAssistant, Text: "cluster healthy"},
	}
}

// A user over quota is refused with ErrUserLimit, while the other users still
// create theirs; a conversation deleted frees a place. The quota counts the
// conversations of the user on every backend and cluster.
func TestStoreEnforcesTheQuotaOfAUser(t *testing.T) {
	store, _ := openTestStore(t, Config{MaxConversations: 10, MaxUserConversations: 2})
	oc3 := conversation.Owner{Backend: "oc3", ClusterID: "", Issuer: testOwner.Issuer, Subject: testOwner.Subject}
	bob := conversation.Owner{Backend: testOwner.Backend, ClusterID: testOwner.ClusterID, Issuer: testOwner.Issuer, Subject: "bob"}
	for _, item := range []conversation.Conversation{testConversation("alice-1", testOwner, testNow), testConversation("alice-2", oc3, testNow)} {
		if err := store.CreateConversation(t.Context(), item); err != nil {
			t.Fatal(err)
		}
	}
	err := store.CreateConversation(t.Context(), testConversation("alice-3", testOwner, testNow))
	if !errors.Is(err, conversation.ErrUserLimit) || errors.Is(err, conversation.ErrLimit) {
		t.Fatalf("quota error = %v", err)
	}
	if err := store.CreateConversation(t.Context(), testConversation("bob-1", bob, testNow)); err != nil {
		t.Fatalf("another user held back by the quota of the first: %v", err)
	}
	if err := store.DeleteConversation(t.Context(), testOwner, "alice-1"); err != nil {
		t.Fatal(err)
	}
	if err := store.CreateConversation(t.Context(), testConversation("alice-3", testOwner, testNow)); err != nil {
		t.Fatalf("a deleted conversation frees no place: %v", err)
	}
}

func TestStoreConfigBoundsTheQuotaByTheLimit(t *testing.T) {
	config, err := validateConfig(Config{Path: "x", MaxConversations: 50})
	if err != nil || config.MaxUserConversations != 50 {
		t.Fatalf("default quota under a smaller limit = %d, %v", config.MaxUserConversations, err)
	}
	config, err = validateConfig(Config{Path: "x"})
	if err != nil || config.MaxConversations != DefaultMaxConversations || config.MaxUserConversations != DefaultMaxUserConversations {
		t.Fatalf("defaults = %+v, %v", config, err)
	}
	if _, err := validateConfig(Config{Path: "x", MaxConversations: 10, MaxUserConversations: 11}); err == nil {
		t.Fatal("a quota over the limit was accepted")
	}
}

// TestStoreGivesEachUserAShareOfTheBytes refuses the turns of a user past
// their share of the stored bytes, issuer and subject on every backend: at
// completion when the turn would pass it, at its start once it is reached,
// before the model is called. Another user's turns still complete.
func TestStoreGivesEachUserAShareOfTheBytes(t *testing.T) {
	turn := []llm.Message{
		{Role: llm.RoleUser, Text: strings.Repeat("x", 50)},
		{Role: llm.RoleAssistant, Text: strings.Repeat("y", 50)},
	}
	_, size, err := encodeMessages(turn)
	if err != nil {
		t.Fatal(err)
	}
	bob := conversation.Owner{Backend: "oc3", Issuer: "other-issuer", Subject: "bob"}
	open := func(t *testing.T, userBytes int64) *Store {
		store, _ := openTestStore(t, Config{MaxConversationBytes: 4 * size, MaxDatabaseBytes: 8 * size, MaxUserBytes: userBytes})
		for _, item := range []conversation.Conversation{testConversation("alice-1", testOwner, testNow), testConversation("bob-1", bob, testNow)} {
			if err := store.CreateConversation(t.Context(), item); err != nil {
				t.Fatal(err)
			}
		}
		return store
	}
	t.Run("at completion", func(t *testing.T) {
		store := open(t, 2*size-1)
		completeOwnerTurn(t, store, testOwner, "alice-1", "a-1", testNow, turn)
		if _, err := store.BeginTurn(t.Context(), testOwner, "alice-1", "a-2", testNow.Add(time.Minute)); err != nil {
			t.Fatal(err)
		}
		err := store.CompleteTurn(t.Context(), testOwner, "alice-1", "a-2", testNow.Add(2*time.Minute), testNow.Add(time.Hour), turn, conversation.TurnRun{}, conversation.TurnTiming{})
		if !errors.Is(err, conversation.ErrUserLimit) {
			t.Fatalf("alice past her share = %v", err)
		}
		completeOwnerTurn(t, store, bob, "bob-1", "b-1", testNow, turn)
	})
	t.Run("at start", func(t *testing.T) {
		store := open(t, size)
		completeOwnerTurn(t, store, testOwner, "alice-1", "a-1", testNow, turn)
		if _, err := store.BeginTurn(t.Context(), testOwner, "alice-1", "a-2", testNow.Add(time.Minute)); !errors.Is(err, conversation.ErrUserLimit) {
			t.Fatalf("alice at her share = %v", err)
		}
		completeOwnerTurn(t, store, bob, "bob-1", "b-1", testNow, turn)
	})
	t.Run("configuration", func(t *testing.T) {
		if _, err := validateConfig(Config{Path: "x", MaxDatabaseBytes: 100, MaxConversationBytes: 10, MaxUserBytes: 101}); err == nil {
			t.Error("a user share above the database accepted")
		}
		if _, err := validateConfig(Config{Path: "x", MaxFeedback: 10, MaxUserFeedback: 11}); err == nil {
			t.Error("a user feedback share above the feedback limit accepted")
		}
		config, err := validateConfig(Config{Path: "x"})
		if err != nil || config.MaxUserBytes != DefaultMaxConversationBytes || config.MaxUserFeedback != DefaultMaxFeedback/defaultUserFeedbackShare {
			t.Errorf("defaults %+v, %v", config, err)
		}
	})
}
