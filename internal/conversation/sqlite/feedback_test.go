package sqlite

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/conversation"
	"github.com/opensvc/ai-agent/internal/llm"
)

func testFeedback(id string, turnID string, feedback conversation.Feedback, at time.Time) conversation.FeedbackRecord {
	return conversation.FeedbackRecord{
		ID: id, TurnID: turnID, Feedback: feedback, CreatedAt: at, UpdatedAt: at, ExpiresAt: at.Add(time.Hour),
		Snapshot: json.RawMessage(`{"question":"health"}`), Context: json.RawMessage(`{"backend":"om3"}`),
	}
}

func TestFeedbackIsSetReplacedListedAndDeleted(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	completeDisplayTurn(t, store, item.ID, "turn-1", testNow, toolTranscript())

	turn, messages, err := store.TurnMessages(t.Context(), testOwner, item.ID, "turn-1")
	if err != nil || turn.Status != conversation.TurnCompleted || len(messages) != 4 || messages[2].ToolResults[0].CallID != "call-1" {
		t.Fatalf("turn messages = %+v, %+v, %v", turn, messages, err)
	}

	down := conversation.Feedback{Rating: "down", Reasons: []string{"incorrect_or_outdated", "not_answered"}, Expected: "the node count"}
	stored, err := store.PutFeedback(t.Context(), testOwner, item.ID, testFeedback("feedback-1", "turn-1", down, testNow))
	if err != nil {
		t.Fatal(err)
	}
	if stored.Feedback.Reasons[0] != "not_answered" || stored.Feedback.Expected != "the node count" {
		t.Fatalf("stored = %+v", stored)
	}
	page, err := store.ListMessages(t.Context(), testOwner, item.ID, conversation.MessageQuery{})
	if err != nil {
		t.Fatal(err)
	}
	if page.Messages[0].Feedback != nil || page.Messages[1].Feedback == nil || page.Messages[1].Feedback.Rating != "down" {
		t.Fatalf("messages = %+v", page.Messages)
	}

	// A new feedback replaces the opinion and keeps the first snapshot.
	replacement := testFeedback("feedback-2", "turn-1", conversation.Feedback{Rating: "up"}, testNow.Add(time.Minute))
	replacement.Snapshot = json.RawMessage(`{"question":"other"}`)
	stored, err = store.PutFeedback(t.Context(), testOwner, item.ID, replacement)
	if err != nil {
		t.Fatal(err)
	}
	if stored.ID != "feedback-1" || stored.Feedback.Rating != "up" || len(stored.Feedback.Reasons) != 0 || string(stored.Snapshot) != `{"question":"health"}` ||
		!stored.CreatedAt.Equal(testNow) || !stored.UpdatedAt.Equal(testNow.Add(time.Minute)) {
		t.Fatalf("replaced = %+v", stored)
	}

	if err := store.DeleteFeedback(t.Context(), testOwner, item.ID, "turn-1"); err != nil {
		t.Fatal(err)
	}
	if err := store.DeleteFeedback(t.Context(), testOwner, item.ID, "turn-1"); !errors.Is(err, conversation.ErrNotFound) {
		t.Fatalf("second delete = %v", err)
	}
}

func TestFeedbackIsBoundToOwnerAndCompletedTurns(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	completeDisplayTurn(t, store, item.ID, "turn-1", testNow, toolTranscript())
	if _, err := store.BeginTurn(t.Context(), testOwner, item.ID, "turn-running", testNow.Add(time.Minute)); err != nil {
		t.Fatal(err)
	}
	other := testOwner
	other.Subject = "mallory"
	up := conversation.Feedback{Rating: "up"}
	if _, err := store.PutFeedback(t.Context(), other, item.ID, testFeedback("f", "turn-1", up, testNow)); !errors.Is(err, conversation.ErrNotFound) {
		t.Fatalf("other owner = %v", err)
	}
	if _, _, err := store.TurnMessages(t.Context(), other, item.ID, "turn-1"); !errors.Is(err, conversation.ErrNotFound) {
		t.Fatalf("other owner messages = %v", err)
	}
	if _, err := store.PutFeedback(t.Context(), testOwner, item.ID, testFeedback("f", "turn-running", up, testNow)); !errors.Is(err, conversation.ErrInvalid) {
		t.Fatalf("running turn = %v", err)
	}
	if _, err := store.PutFeedback(t.Context(), testOwner, item.ID, testFeedback("f", "turn-unknown", up, testNow)); !errors.Is(err, conversation.ErrNotFound) {
		t.Fatalf("unknown turn = %v", err)
	}
	if _, err := store.PutFeedback(t.Context(), testOwner, item.ID, testFeedback("f", "turn-1", conversation.Feedback{Rating: "up", Reasons: []string{"not_answered"}}, testNow)); !errors.Is(err, conversation.ErrInvalid) {
		t.Fatalf("reasons with up = %v", err)
	}
	if err := store.DeleteFeedback(t.Context(), other, item.ID, "turn-1"); !errors.Is(err, conversation.ErrNotFound) {
		t.Fatalf("other owner delete = %v", err)
	}
}

func TestFeedbackOutlivesItsConversationAnonymously(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	completeDisplayTurn(t, store, item.ID, "turn-1", testNow, toolTranscript())
	if _, err := store.PutFeedback(t.Context(), testOwner, item.ID, testFeedback("feedback-1", "turn-1", conversation.Feedback{Rating: "down", Expected: "more"}, testNow)); err != nil {
		t.Fatal(err)
	}
	if err := store.DeleteConversation(t.Context(), testOwner, item.ID); err != nil {
		t.Fatal(err)
	}
	var turnID *string
	if err := store.db.QueryRowContext(t.Context(), "SELECT turn_id FROM feedback WHERE id = 'feedback-1'").Scan(&turnID); err != nil || turnID != nil {
		t.Fatalf("turn_id = %v, %v", turnID, err)
	}
	var exported []conversation.FeedbackRecord
	if err := store.ExportFeedback(t.Context(), testNow.Add(-time.Hour), func(record conversation.FeedbackRecord) error {
		exported = append(exported, record)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(exported) != 1 || exported[0].TurnID != "" || exported[0].Feedback.Expected != "more" || string(exported[0].Snapshot) != `{"question":"health"}` {
		t.Fatalf("exported = %+v", exported)
	}
	if err := store.ExportFeedback(t.Context(), testNow.Add(time.Second), func(conversation.FeedbackRecord) error { return errors.New("unexpected") }); err != nil {
		t.Fatalf("export since later = %v", err)
	}
	if deleted, err := store.DeleteExpiredFeedback(t.Context(), testNow.Add(59*time.Minute), 10); err != nil || deleted != 0 {
		t.Fatalf("early expiry = %d, %v", deleted, err)
	}
	if deleted, err := store.DeleteExpiredFeedback(t.Context(), testNow.Add(time.Hour), 10); err != nil || deleted != 1 {
		t.Fatalf("expiry = %d, %v", deleted, err)
	}
}

func TestFeedbackLimits(t *testing.T) {
	store, _ := openTestStore(t, Config{MaxFeedback: 1})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	completeDisplayTurn(t, store, item.ID, "turn-1", testNow, toolTranscript())
	completeDisplayTurn(t, store, item.ID, "turn-2", testNow.Add(time.Minute), toolTranscript())
	up := conversation.Feedback{Rating: "up"}
	if _, err := store.PutFeedback(t.Context(), testOwner, item.ID, testFeedback("f1", "turn-1", up, testNow)); err != nil {
		t.Fatal(err)
	}
	// One user alone reaches their share first, here the whole limit.
	if _, err := store.PutFeedback(t.Context(), testOwner, item.ID, testFeedback("f2", "turn-2", up, testNow)); !errors.Is(err, conversation.ErrUserLimit) {
		t.Fatalf("over the limit = %v", err)
	}
	// Replacing a feedback is never refused by the limit.
	if _, err := store.PutFeedback(t.Context(), testOwner, item.ID, testFeedback("f3", "turn-1", conversation.Feedback{Rating: "down"}, testNow)); err != nil {
		t.Fatal(err)
	}
	large := testFeedback("f4", "turn-1", up, testNow)
	large.Snapshot = json.RawMessage(`"` + strings.Repeat("a", maxFeedbackRecordBytes) + `"`)
	if _, err := store.PutFeedback(t.Context(), testOwner, item.ID, large); !errors.Is(err, conversation.ErrLimit) {
		t.Fatalf("oversized record = %v", err)
	}
}

type unusedRunner struct{}

func (unusedRunner) RunTurn(context.Context, []llm.Message, string, agent.EmitFunc) (agent.TurnResult, error) {
	return agent.TurnResult{}, errors.New("unexpected turn")
}

func (unusedRunner) ResumeTurn(context.Context, []llm.Message, agent.SuspendedTurn, agent.Decision, agent.EmitFunc) (agent.TurnResult, error) {
	return agent.TurnResult{}, errors.New("unexpected turn")
}

func TestServiceFeedbackSnapshotsTheTurn(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	service, err := conversation.NewService(store, unusedRunner{}, conversation.ServiceConfig{
		Backend: "om3", FeedbackRetention: 48 * time.Hour,
		// The agent serving the feedback is not the one that ran the turn.
		Agent: conversation.NewAgentInfo("v9-current", "gpt-current"),
	})
	if err != nil {
		t.Fatal(err)
	}
	identity := auth.Identity{ClusterID: testOwner.ClusterID, Issuer: testOwner.Issuer, Subject: testOwner.Subject}
	item, err := service.Create(t.Context(), identity)
	if err != nil {
		t.Fatal(err)
	}
	// The turn waits on a confirmation, then resumes on an updated MCP.
	now := time.Now()
	first := conversation.TurnRun{
		AgentVersion: "v1.2.3", Model: "gpt-test", SystemPromptSHA256: strings.Repeat("cd", 32), ToolSearch: true,
		ShownTools: []string{"search_tools", "stop_object"},
		MCPServer:  "om3-mcp", MCPVersion: "v1", ToolsSHA256: strings.Repeat("ab", 32),
	}
	if _, err := store.BeginTurn(t.Context(), testOwner, item.ID, "turn-1", now); err != nil {
		t.Fatal(err)
	}
	if err := store.SuspendTurn(t.Context(), testOwner, conversation.PendingConfirmation{
		ConversationID: item.ID, TurnID: "turn-1", ConfirmationID: "confirmation-1",
		ToolName: "stop_object", Arguments: json.RawMessage(`{}`), ArgumentsHash: "hash", State: []byte(`{}`),
		CreatedAt: now, ExpiresAt: now.Add(time.Minute), Run: first,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.ClaimConfirmation(t.Context(), testOwner, item.ID, "turn-1", "confirmation-1", now.Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	second := first
	second.ShownTools = append([]string(nil), first.ShownTools...)
	second.MCPVersion = strings.Repeat("v", 300)
	if err := store.CompleteTurn(t.Context(), testOwner, item.ID, "turn-1", now.Add(2*time.Second), now.Add(time.Hour), toolTranscript(), second, conversation.TurnTiming{}); err != nil {
		t.Fatal(err)
	}

	record, err := service.SetFeedback(t.Context(), identity, item.ID, "turn-1", conversation.Feedback{Rating: "down", Reasons: []string{"missing_information"}, Expected: "  the nodes  "})
	if err != nil {
		t.Fatal(err)
	}
	if record.Feedback.Expected != "the nodes" || !record.ExpiresAt.Equal(record.CreatedAt.Add(48*time.Hour)) {
		t.Fatalf("record = %+v", record)
	}
	var snapshot conversation.TurnSnapshot
	if err := json.Unmarshal(record.Snapshot, &snapshot); err != nil {
		t.Fatal(err)
	}
	if snapshot.Question != "health" || snapshot.Answer != "cluster healthy" || len(snapshot.ToolCalls) != 1 ||
		snapshot.ToolCalls[0].Name != "get_cluster_health" || snapshot.ToolCalls[0].Result != `{"status":"healthy"}` || snapshot.ToolCalls[0].ResultBytes != 20 {
		t.Fatalf("snapshot = %+v", snapshot)
	}
	var agentContext conversation.FeedbackContext
	if err := json.Unmarshal(record.Context, &agentContext); err != nil {
		t.Fatal(err)
	}
	second.MCPVersion = second.MCPVersion[:256]
	if agentContext.Backend != "om3" || len(agentContext.Runs) != 2 || !agentContext.Runs[0].Equal(first) || !agentContext.Runs[1].Equal(second) {
		t.Fatalf("context = %+v", agentContext)
	}

	other := auth.Identity{ClusterID: testOwner.ClusterID, Issuer: testOwner.Issuer, Subject: "mallory"}
	if _, err := service.SetFeedback(t.Context(), other, item.ID, "turn-1", conversation.Feedback{Rating: "up"}); !errors.Is(err, conversation.ErrNotFound) {
		t.Fatalf("other identity = %v", err)
	}
	if err := service.DeleteFeedback(t.Context(), other, item.ID, "turn-1"); !errors.Is(err, conversation.ErrNotFound) {
		t.Fatalf("other identity delete = %v", err)
	}
	if err := service.DeleteFeedback(t.Context(), identity, item.ID, "turn-1"); err != nil {
		t.Fatal(err)
	}
}

func TestTurnRunsSkipAnUnchangedResumption(t *testing.T) {
	store, _ := openTestStore(t, Config{})
	item := testConversation("conversation-1", testOwner, testNow)
	if err := store.CreateConversation(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	run := conversation.TurnRun{AgentVersion: "v1", Model: "gpt", MCPServer: "om3-mcp", MCPVersion: "v1"}
	pending := conversation.PendingConfirmation{
		ConversationID: item.ID, TurnID: "turn-1", ConfirmationID: "confirmation-1",
		ToolName: "stop_object", Arguments: json.RawMessage(`{}`), ArgumentsHash: "hash", State: []byte(`{}`),
		CreatedAt: testNow, ExpiresAt: testNow.Add(time.Minute), Run: run,
	}
	if _, err := store.BeginTurn(t.Context(), testOwner, item.ID, "turn-1", testNow); err != nil {
		t.Fatal(err)
	}
	if err := store.SuspendTurn(t.Context(), testOwner, pending); err != nil {
		t.Fatal(err)
	}
	if _, err := store.ClaimConfirmation(t.Context(), testOwner, item.ID, "turn-1", "confirmation-1", testNow.Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	if err := store.CompleteTurn(t.Context(), testOwner, item.ID, "turn-1", testNow.Add(2*time.Second), testNow.Add(time.Hour), toolTranscript(), run, conversation.TurnTiming{}); err != nil {
		t.Fatal(err)
	}
	turn, _, err := store.TurnMessages(t.Context(), testOwner, item.ID, "turn-1")
	if err != nil || len(turn.Runs) != 1 || !turn.Runs[0].Equal(run) {
		t.Fatalf("runs = %+v, %v", turn.Runs, err)
	}
}

// A feedback always fits: its bounded context beside its bounded snapshot.
func TestFeedbackRecordHoldsTheBoundedContextAndSnapshot(t *testing.T) {
	if maxFeedbackRecordBytes < conversation.MaxFeedbackContextBytes+48<<10 {
		t.Fatalf("record limit %d leaves no room for a context of %d and a snapshot of 48 KiB", maxFeedbackRecordBytes, conversation.MaxFeedbackContextBytes)
	}
}

func TestTurnRunsCountTheShownToolsLeftOut(t *testing.T) {
	names := make([]string, maxRunShownTools+5)
	for i := range names {
		names[i] = fmt.Sprintf("tool_%d", i)
	}
	data, err := encodeRuns(nil, conversation.TurnRun{ShownTools: names})
	if err != nil {
		t.Fatal(err)
	}
	var runs []conversation.TurnRun
	if err := json.Unmarshal(data, &runs); err != nil || len(runs[0].ShownTools) != maxRunShownTools || runs[0].ShownToolsOmitted != 5 {
		t.Fatalf("runs = %d shown, %d omitted, %v", len(runs[0].ShownTools), runs[0].ShownToolsOmitted, err)
	}
}

// completeOwnerTurn stores a completed turn of owner in the conversation id.
func completeOwnerTurn(t *testing.T, store *Store, owner conversation.Owner, id, turnID string, at time.Time, messages []llm.Message) {
	t.Helper()
	if _, err := store.BeginTurn(t.Context(), owner, id, turnID, at); err != nil {
		t.Fatal(err)
	}
	if err := store.CompleteTurn(t.Context(), owner, id, turnID, at.Add(time.Second), at.Add(time.Hour), messages, conversation.TurnRun{}, conversation.TurnTiming{}); err != nil {
		t.Fatal(err)
	}
}

// TestFeedbackIsSharedBetweenUsers gives each user a share of the feedback:
// one user at their share does not block another; and the feedback of a
// conversation its owner deleted, which nobody can withdraw any more, makes
// room for another user's when the instance is full, rather than refuse it
// for the retention.
func TestFeedbackIsSharedBetweenUsers(t *testing.T) {
	bob := conversation.Owner{Backend: "oc3", Issuer: "other-issuer", Subject: "bob"}
	up := conversation.Feedback{Rating: "up"}
	t.Run("share", func(t *testing.T) {
		store, _ := openTestStore(t, Config{MaxFeedback: 4, MaxUserFeedback: 1})
		alice := testConversation("alice-1", testOwner, testNow)
		bobs := testConversation("bob-1", bob, testNow)
		for _, item := range []conversation.Conversation{alice, bobs} {
			if err := store.CreateConversation(t.Context(), item); err != nil {
				t.Fatal(err)
			}
		}
		completeOwnerTurn(t, store, testOwner, alice.ID, "a-1", testNow, toolTranscript())
		completeOwnerTurn(t, store, testOwner, alice.ID, "a-2", testNow.Add(time.Minute), toolTranscript())
		completeOwnerTurn(t, store, bob, bobs.ID, "b-1", testNow, toolTranscript())
		if _, err := store.PutFeedback(t.Context(), testOwner, alice.ID, testFeedback("f1", "a-1", up, testNow)); err != nil {
			t.Fatal(err)
		}
		if _, err := store.PutFeedback(t.Context(), testOwner, alice.ID, testFeedback("f2", "a-2", up, testNow)); !errors.Is(err, conversation.ErrUserLimit) {
			t.Fatalf("alice over her share = %v", err)
		}
		if _, err := store.PutFeedback(t.Context(), bob, bobs.ID, testFeedback("f3", "b-1", up, testNow)); err != nil {
			t.Fatalf("bob = %v", err)
		}
	})
	t.Run("full instance", func(t *testing.T) {
		store, _ := openTestStore(t, Config{MaxFeedback: 2, MaxUserFeedback: 2})
		alice := testConversation("alice-1", testOwner, testNow)
		bobs := testConversation("bob-1", bob, testNow)
		for _, item := range []conversation.Conversation{alice, bobs} {
			if err := store.CreateConversation(t.Context(), item); err != nil {
				t.Fatal(err)
			}
		}
		completeOwnerTurn(t, store, testOwner, alice.ID, "a-1", testNow, toolTranscript())
		completeOwnerTurn(t, store, testOwner, alice.ID, "a-2", testNow.Add(time.Minute), toolTranscript())
		completeOwnerTurn(t, store, bob, bobs.ID, "b-1", testNow, toolTranscript())
		for _, f := range []struct{ id, turn string }{{"f1", "a-1"}, {"f2", "a-2"}} {
			if _, err := store.PutFeedback(t.Context(), testOwner, alice.ID, testFeedback(f.id, f.turn, up, testNow)); err != nil {
				t.Fatal(err)
			}
		}
		// Full of feedback alice can still withdraw: bob is refused.
		if _, err := store.PutFeedback(t.Context(), bob, bobs.ID, testFeedback("f3", "b-1", up, testNow)); !errors.Is(err, conversation.ErrLimit) {
			t.Fatalf("bob on a full instance = %v", err)
		}
		// Alice deletes the conversation she rated: her feedback stays,
		// anonymous, and the oldest goes for bob's.
		if err := store.DeleteConversation(t.Context(), testOwner, alice.ID); err != nil {
			t.Fatal(err)
		}
		if _, err := store.PutFeedback(t.Context(), bob, bobs.ID, testFeedback("f3", "b-1", up, testNow)); err != nil {
			t.Fatalf("bob after alice deleted = %v", err)
		}
		var total, anonymous int
		if err := store.db.QueryRowContext(t.Context(), "SELECT COUNT(*), COALESCE(SUM(CASE WHEN turn_id IS NULL THEN 1 ELSE 0 END), 0) FROM feedback").Scan(&total, &anonymous); err != nil {
			t.Fatal(err)
		}
		if total != 2 || anonymous != 1 {
			t.Fatalf("feedback rows %d, anonymous %d", total, anonymous)
		}
	})
}
