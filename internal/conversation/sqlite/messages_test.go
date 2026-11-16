package sqlite

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/opensvc/ai-agent/internal/conversation"
	"github.com/opensvc/ai-agent/internal/llm"
)

func completeDisplayTurn(t *testing.T, store *Store, id, turnID string, at time.Time, messages []llm.Message) {
	t.Helper()
	if _, err := store.BeginTurn(t.Context(), testOwner, id, turnID, at); err != nil {
		t.Fatal(err)
	}
	if err := store.CompleteTurn(t.Context(), testOwner, id, turnID, at.Add(time.Second), at.Add(time.Hour), messages, conversation.TurnRun{}, conversation.TurnTiming{}); err != nil {
		t.Fatal(err)
	}
}

func TestDisplayMessagesPaginateCompletedTurnsWithTheirSteps(t *testing.T) {
	store, path := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	completeDisplayTurn(t, store, item.ID, "turn-1", testNow, toolTranscript())
	if _, err := store.BeginTurn(t.Context(), testOwner, item.ID, "turn-failed", testNow.Add(2*time.Second)); err != nil {
		t.Fatal(err)
	}
	if err := store.FailTurn(t.Context(), testOwner, item.ID, "turn-failed", conversation.TurnCanceled, "request_canceled", testNow.Add(3*time.Second)); err != nil {
		t.Fatal(err)
	}
	completeDisplayTurn(t, store, item.ID, "turn-3", testNow.Add(4*time.Second), []llm.Message{
		{Role: llm.RoleUser, Text: "second prompt"}, {Role: llm.RoleAssistant, Text: "second answer"},
	})
	// A running turn must neither block reads nor appear in the transcript.
	if _, err := store.BeginTurn(t.Context(), testOwner, item.ID, "turn-running", testNow.Add(6*time.Second)); err != nil {
		t.Fatal(err)
	}
	page, err := store.ListMessages(t.Context(), testOwner, item.ID, conversation.MessageQuery{Limit: 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(page.Messages) != 3 || page.NextCursor != "1:4" {
		t.Fatalf("latest page = %+v", page)
	}
	for index, want := range []string{"cluster healthy", "second prompt", "second answer"} {
		if page.Messages[index].Text != want {
			t.Fatalf("message %d = %+v", index, page.Messages[index])
		}
	}
	if page.Messages[0].ID != "turn-1:4" || !page.Messages[1].CreatedAt.Equal(testNow.Add(4*time.Second)) || !page.Messages[2].CreatedAt.Equal(testNow.Add(5*time.Second)) {
		t.Fatalf("message metadata = %+v", page.Messages)
	}
	older, err := store.ListMessages(t.Context(), testOwner, item.ID, conversation.MessageQuery{Limit: 3, Before: page.NextCursor})
	if err != nil || len(older.Messages) != 1 || older.Messages[0].Text != "health" || older.NextCursor != "" {
		t.Fatalf("older page = %+v, %v", older, err)
	}
	encoded, err := json.Marshal(page)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"tool_calls", "tool_results", "request_canceled"} {
		if strings.Contains(string(encoded), forbidden) {
			t.Fatalf("display exposes %s: %s", forbidden, encoded)
		}
	}
	// The last assistant message of a turn carries its steps; a turn stored
	// without times has none.
	steps := page.Messages[0].Steps
	if len(steps) != 1 || steps[0].ID != "call-1" || steps[0].Tool != "get_cluster_health" || steps[0].Status != "completed" ||
		steps[0].StartedAt != nil || steps[0].EndedAt != nil || page.Messages[0].Runs != nil || page.Messages[0].Usage != nil {
		t.Fatalf("steps = %+v", page.Messages[0])
	}
	if page.Messages[2].Steps != nil || page.Messages[1].Steps != nil {
		t.Fatalf("steps on a message without calls: %+v", page.Messages)
	}
	// Existing persisted turns remain readable after reopening, without a schema change.
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := Open(t.Context(), Config{Path: path})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reopened.Close() })
	again, err := reopened.ListMessages(t.Context(), testOwner, item.ID, conversation.MessageQuery{Limit: 3})
	if err != nil || !reflect.DeepEqual(again, page) {
		t.Fatalf("reopened page = %+v, %v", again, err)
	}
}

func TestDisplayMessagesAreBoundToOwner(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	completeDisplayTurn(t, store, item.ID, "turn-1", testNow, toolTranscript())
	for _, owner := range []conversation.Owner{
		{Backend: "oc3", ClusterID: testOwner.ClusterID, Issuer: testOwner.Issuer, Subject: testOwner.Subject},
		{Backend: testOwner.Backend, ClusterID: "other-cluster", Issuer: testOwner.Issuer, Subject: testOwner.Subject},
		{Backend: testOwner.Backend, ClusterID: testOwner.ClusterID, Issuer: "other-issuer", Subject: testOwner.Subject},
		{Backend: testOwner.Backend, ClusterID: testOwner.ClusterID, Issuer: testOwner.Issuer, Subject: "other-user"},
	} {
		page, err := store.ListMessages(t.Context(), owner, item.ID, conversation.MessageQuery{Before: "2:1"})
		if !errors.Is(err, conversation.ErrNotFound) || len(page.Messages) != 0 {
			t.Fatalf("foreign owner got %+v, %v", page, err)
		}
	}
	if _, err := store.ListMessages(t.Context(), testOwner, "missing", conversation.MessageQuery{}); !errors.Is(err, conversation.ErrNotFound) {
		t.Fatalf("missing messages error = %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := store.ListMessages(ctx, testOwner, item.ID, conversation.MessageQuery{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled messages error = %v", err)
	}
}

func TestDisplayMessagesEmptyPageAndStableCursor(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	page, err := store.ListMessages(t.Context(), testOwner, item.ID, conversation.MessageQuery{})
	if err != nil || page.Messages == nil || len(page.Messages) != 0 || page.NextCursor != "" {
		t.Fatalf("empty page = %+v, %v", page, err)
	}
	completeDisplayTurn(t, store, item.ID, "turn-1", testNow, toolTranscript())
	page, err = store.ListMessages(t.Context(), testOwner, item.ID, conversation.MessageQuery{Limit: 1})
	if err != nil || len(page.Messages) != 1 || page.NextCursor != "1:4" {
		t.Fatalf("latest message = %+v, %v", page, err)
	}
	completeDisplayTurn(t, store, item.ID, "turn-2", testNow.Add(2*time.Second), toolTranscript())
	older, err := store.ListMessages(t.Context(), testOwner, item.ID, conversation.MessageQuery{Limit: 1, Before: page.NextCursor})
	if err != nil || len(older.Messages) != 1 || older.Messages[0].ID != "turn-1:1" || older.NextCursor != "" {
		t.Fatalf("cursor shifted after new turn = %+v, %v", older, err)
	}
}

func TestDisplayMessagePageIsBoundedByEncodedBytes(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	// '<' is escaped to six bytes by encoding/json. The page fits one answer,
	// not two, despite the raw text being far below the byte budget.
	text := strings.Repeat("<", conversation.MaxMessagePageBytes/8)
	for index, id := range []string{"turn-1", "turn-2"} {
		completeDisplayTurn(t, store, item.ID, id, testNow.Add(time.Duration(index)*2*time.Second), []llm.Message{
			{Role: llm.RoleUser, Text: "hello"}, {Role: llm.RoleAssistant, Text: text},
		})
	}
	page, err := store.ListMessages(t.Context(), testOwner, item.ID, conversation.MessageQuery{})
	if err != nil || len(page.Messages) != 2 || page.NextCursor != "2:1" {
		t.Fatalf("byte-bounded page length = %d, cursor = %q, error = %v", len(page.Messages), page.NextCursor, err)
	}
	encoded, err := json.Marshal(page)
	if err != nil || len(encoded) > conversation.MaxMessagePageBytes {
		t.Fatalf("page bytes = %d, error = %v", len(encoded), err)
	}
	older, err := store.ListMessages(t.Context(), testOwner, item.ID, conversation.MessageQuery{Before: page.NextCursor})
	if err != nil || len(older.Messages) != 2 || older.NextCursor != "" {
		t.Fatalf("older byte-bounded page length = %d, error = %v", len(older.Messages), err)
	}
}

func TestDisplayMessagesRejectOversizedSingleMessageWithoutTruncation(t *testing.T) {
	for _, text := range []string{
		strings.Repeat("x", conversation.MaxMessagePageBytes+1),
		strings.Repeat("<", conversation.MaxMessagePageBytes/6+1),
	} {
		store, _ := openTestStore(t, Config{})
		item := testConversation("conversation-1", testOwner, testNow)
		if err := store.CreateConversation(t.Context(), item); err != nil {
			t.Fatal(err)
		}
		completeDisplayTurn(t, store, item.ID, "turn-1", testNow, []llm.Message{
			{Role: llm.RoleUser, Text: "hello"}, {Role: llm.RoleAssistant, Text: text},
		})
		if _, err := store.ListMessages(t.Context(), testOwner, item.ID, conversation.MessageQuery{}); !errors.Is(err, conversation.ErrMessageTooLarge) {
			t.Fatalf("oversized display error = %v", err)
		}
	}
}

// A message stores the tool calls the agent runs per model answer, 8, and
// not more.
func TestStoreHoldsTheToolCallsOfOneAnswer(t *testing.T) {
	batch := func(count int) []llm.Message {
		var calls []llm.ToolCall
		var results []llm.ToolResult
		for i := range count {
			id := fmt.Sprintf("call-%d", i)
			calls = append(calls, llm.ToolCall{ID: id, Name: "get_node_status", Arguments: json.RawMessage(`{}`)})
			results = append(results, llm.ToolResult{CallID: id, Content: json.RawMessage(`{}`)})
		}
		return []llm.Message{
			{Role: llm.RoleUser, Text: "status"},
			{Role: llm.RoleAssistant, ToolCalls: calls},
			{Role: llm.RoleTool, ToolResults: results},
			{Role: llm.RoleAssistant, Text: "done"},
		}
	}
	if _, _, err := encodeMessages(batch(maxStoredToolCallsPerMessage)); err != nil {
		t.Fatalf("%d calls refused: %v", maxStoredToolCallsPerMessage, err)
	}
	if _, _, err := encodeMessages(batch(maxStoredToolCallsPerMessage + 1)); err == nil {
		t.Fatal("too many calls accepted")
	}
	// Volatile text, such as the context of the interface, lives as long
	// as the requests that carry it.
	if _, _, err := encodeMessages([]llm.Message{{Role: llm.RoleUser, Text: "status", Volatile: "screen"}, {Role: llm.RoleAssistant, Text: "done"}}); err == nil {
		t.Fatal("volatile text stored")
	}
	if maxStoredToolCallsPerMessage != 8 {
		t.Fatalf("stored calls per message = %d, the agent runs 8 per answer", maxStoredToolCallsPerMessage)
	}
}

// The times of a turn are stored with its suspension, given back with its
// claim, completed, then served with its steps.
func TestDisplayMessagesServeTheTimesOfATurn(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	at := testNow.Add(time.Second)
	if _, err := store.BeginTurn(t.Context(), testOwner, item.ID, "turn-1", at); err != nil {
		t.Fatal(err)
	}
	first := conversation.TurnTiming{
		Runs:  []conversation.TimingRun{{StartedAt: 1000, EndedAt: 2500}},
		Calls: map[string]conversation.TimingCall{"call-1": {StartedAt: 1100, EndedAt: 1400}},
		Usage: &conversation.TimingUsage{TotalTokens: 300},
	}
	pending := conversation.PendingConfirmation{
		ConversationID: item.ID, TurnID: "turn-1", ConfirmationID: "confirmation-1",
		ToolName: "stop_object", ToolTitle: "Stop object", Arguments: json.RawMessage(`{"path":"web"}`),
		ArgumentsHash: "hash", State: []byte(`{"messages":[]}`),
		CreatedAt: at, ExpiresAt: at.Add(10 * time.Minute), Timing: first,
	}
	if err := store.SuspendTurn(t.Context(), testOwner, pending); err != nil {
		t.Fatal(err)
	}
	claimed, err := store.ClaimConfirmation(t.Context(), testOwner, item.ID, "turn-1", "confirmation-1", at.Add(time.Second))
	if err != nil || len(claimed.Timing.Runs) != 1 || claimed.Timing.Calls["call-1"].EndedAt != 1400 || claimed.Timing.Usage.TotalTokens != 300 {
		t.Fatalf("claimed timing = %+v, %v", claimed.Timing, err)
	}
	completed := claimed.Timing
	completed.Runs = append(completed.Runs, conversation.TimingRun{StartedAt: 9000, EndedAt: 9500})
	completed.Calls["call-2"] = conversation.TimingCall{StartedAt: 9000, EndedAt: 9300, RanWith: json.RawMessage(`{"path":"db"}`)}
	completed.Usage.TotalTokens = 370
	messages := []llm.Message{
		{Role: llm.RoleUser, Text: "stop web"},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "get_object_status", Arguments: json.RawMessage(`{"path":"web"}`)}}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call-1", Content: json.RawMessage(`{"content":[],"structuredContent":{"avail":"up"}}`)}}},
		{Role: llm.RoleAssistant, Text: "I will stop it.", ToolCalls: []llm.ToolCall{{ID: "call-2", Name: "stop_object", Arguments: json.RawMessage(`{"path":"web"}`)}}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call-2", Content: json.RawMessage(`{"content":[],"structuredContent":{"outcome":"stopped"}}`)}}},
		{Role: llm.RoleAssistant, Text: "Stopped."},
	}
	if err := store.CompleteTurn(t.Context(), testOwner, item.ID, "turn-1", at.Add(time.Minute), at.Add(time.Hour), messages, conversation.TurnRun{}, completed); err != nil {
		t.Fatal(err)
	}
	page, err := store.ListMessages(t.Context(), testOwner, item.ID, conversation.MessageQuery{})
	if err != nil || len(page.Messages) != 3 {
		t.Fatalf("page = %+v, %v", page, err)
	}
	// The steps go on the last assistant message of the turn only.
	if page.Messages[1].Text != "I will stop it." || page.Messages[1].Steps != nil {
		t.Fatalf("an earlier message of the turn carries steps: %+v", page.Messages[1])
	}
	last := page.Messages[2]
	if len(last.Steps) != 2 || *last.Steps[0].StartedAt != 1100 || *last.Steps[1].EndedAt != 9300 || string(last.Steps[1].EditedArguments) != `{"path":"db"}` ||
		len(last.Runs) != 2 || *last.Runs[1].EndedAt != 9500 || last.Usage.TotalTokens != 370 {
		t.Fatalf("last = %+v", last)
	}
}

// A turn whose text holds in a page and its steps not is served without its
// steps, counted as omitted.
func TestDisplayMessagesServeWithoutItsStepsAnAnswerTooLargeWithThem(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	text := strings.Repeat("x", conversation.MaxMessagePageBytes-256-1024)
	arguments := json.RawMessage(`{"path":"` + strings.Repeat("a", 4096) + `"}`)
	completeDisplayTurn(t, store, item.ID, "turn-1", testNow, []llm.Message{
		{Role: llm.RoleUser, Text: "hello"},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "get_object_status", Arguments: arguments}}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call-1", Content: json.RawMessage(`{"content":[]}`)}}},
		{Role: llm.RoleAssistant, Text: text},
	})
	page, err := store.ListMessages(t.Context(), testOwner, item.ID, conversation.MessageQuery{})
	if err != nil || len(page.Messages) != 2 || page.NextCursor != "" {
		t.Fatalf("page length = %d, cursor = %q, error = %v", len(page.Messages), page.NextCursor, err)
	}
	if answer := page.Messages[1]; answer.Text != text || answer.Steps != nil || answer.StepsOmitted != 1 {
		t.Fatalf("steps = %+v, omitted = %d", answer.Steps, answer.StepsOmitted)
	}
}

// The turn the cursor cuts gave its steps to its last answer, on the page
// before: its earlier answers on the next page carry none.
func TestDisplayMessagesServeTheStepsOfATurnCutByTheCursorOnce(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	completeDisplayTurn(t, store, item.ID, "turn-1", testNow, []llm.Message{
		{Role: llm.RoleUser, Text: "stop web"},
		{Role: llm.RoleAssistant, Text: "I check it.", ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "get_object_status", Arguments: json.RawMessage(`{"path":"web"}`)}}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call-1", Content: json.RawMessage(`{"content":[]}`)}}},
		{Role: llm.RoleAssistant, Text: "It is up."},
	})
	page, err := store.ListMessages(t.Context(), testOwner, item.ID, conversation.MessageQuery{Limit: 1})
	if err != nil || len(page.Messages) != 1 || page.Messages[0].Text != "It is up." || len(page.Messages[0].Steps) != 1 {
		t.Fatalf("page = %+v, %v", page, err)
	}
	older, err := store.ListMessages(t.Context(), testOwner, item.ID, conversation.MessageQuery{Before: page.NextCursor})
	if err != nil || len(older.Messages) != 2 || older.Messages[1].Text != "I check it." || older.Messages[1].Steps != nil || older.Messages[1].Runs != nil {
		t.Fatalf("older = %+v, %v", older, err)
	}
}
