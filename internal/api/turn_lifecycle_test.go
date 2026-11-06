package api

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/conversation"
	conversationsqlite "github.com/opensvc/ai-agent/internal/conversation/sqlite"
	"github.com/opensvc/ai-agent/internal/llm"
)

// failingWriter records the status and headers of a response and fails
// every write of its body, as a connection reset or an expired write
// deadline on the opening frame does.
type failingWriter struct {
	*httptest.ResponseRecorder
}

func (w failingWriter) Write([]byte) (int, error) { return 0, errors.New("connection reset") }

func (failingWriter) Flush() {}

// TestAFailedOpeningLeavesNoTurnRunning fails the write of the opening frame
// of a run, before the run starts: the turn the run did not take is
// canceled, so that the conversation takes the next run and can be deleted,
// rather than stay busy until a restart.
func TestAFailedOpeningLeavesNoTurnRunning(t *testing.T) {
	directory := t.TempDir()
	if err := os.Chmod(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, "conversations.db")
	store, err := conversationsqlite.Open(t.Context(), conversationsqlite.Config{Path: path})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	runs := 0
	runner := apiTurnRunnerFunc(func(_ context.Context, _ []llm.Message, prompt string, emit agent.EmitFunc) (agent.TurnResult, error) {
		runs++
		if err := emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, Iteration: 1}); err != nil {
			return agent.TurnResult{}, err
		}
		return agent.TurnResult{Messages: []llm.Message{{Role: llm.RoleUser, Text: prompt}, {Role: llm.RoleAssistant, Text: "ok"}}, FinishReason: llm.FinishReasonCompleted}, nil
	})
	service, err := conversation.NewService(store, runner, conversation.ServiceConfig{Backend: "om3"})
	if err != nil {
		t.Fatal(err)
	}
	verifier := tokenVerifierFunc(func(_ context.Context, token string) (auth.Identity, error) {
		return auth.Identity{ClusterID: "cluster-id", Issuer: "cluster", Subject: token}, nil
	})
	handler, err := NewHandler([]Backend{{Name: "om3", Conversations: service, Authenticator: verifier}}, HandlerConfig{MaxConcurrentAsks: 2, AuditLogger: discardAuditLogger()})
	if err != nil {
		t.Fatal(err)
	}
	created := httptest.NewRecorder()
	handler.ServeHTTP(created, requestWithToken(http.MethodPost, "/v1/om3/conversations", "alice", ""))
	var envelope ConversationEnvelope
	if err := json.NewDecoder(created.Body).Decode(&envelope); err != nil || created.Code != http.StatusCreated {
		t.Fatalf("create: %d, %v", created.Code, err)
	}
	id := envelope.Conversation.ID
	run := func(w http.ResponseWriter, prompt string) {
		request := requestWithToken(http.MethodPost, "/v1/om3/conversations/"+id+"/runs", "alice", runInput(id, prompt))
		request.Header.Set("Content-Type", "application/json")
		handler.ServeHTTP(w, request)
	}

	run(failingWriter{httptest.NewRecorder()}, "hello")
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var status, code string
	if err := db.QueryRow("SELECT status, error_code FROM turns WHERE conversation_id = ?", id).Scan(&status, &code); err != nil {
		t.Fatal(err)
	}
	if status != string(conversation.TurnCanceled) || code != "stream_not_started" || runs != 0 {
		t.Fatalf("turn %s %q after a failed opening, %d runs", status, code, runs)
	}
	next := httptest.NewRecorder()
	run(next, "again")
	if next.Code != http.StatusOK || runs != 1 {
		t.Fatalf("next run: %d %s, %d runs", next.Code, next.Body.String(), runs)
	}
	deleted := httptest.NewRecorder()
	handler.ServeHTTP(deleted, requestWithToken(http.MethodDelete, "/v1/om3/conversations/"+id, "alice", ""))
	if deleted.Code != http.StatusNoContent {
		t.Fatalf("delete: %d %s", deleted.Code, deleted.Body.String())
	}
}

// cancelFailing prepares turns whose cancellation fails with err.
type cancelFailing struct {
	noopConversationService
	err error
}

func (s cancelFailing) PrepareTurn(context.Context, auth.Identity, string, string) (conversation.TurnExecution, error) {
	return cancelFailingExecution{testTurnExecution: &testTurnExecution{run: func(context.Context, agent.EmitFunc) error { return nil }}, err: s.err}, nil
}

type cancelFailingExecution struct {
	*testTurnExecution
	err error
}

func (e cancelFailingExecution) Cancel(string) error { return e.err }

// TestAFailedCancellationIsAudited audits a turn that may stay running, its
// cancellation failing as with the database closed, and not the conflict of
// a turn the run already took.
func TestAFailedCancellationIsAudited(t *testing.T) {
	for name, test := range map[string]struct {
		err     error
		audited bool
	}{
		"database closed": {errors.New("database closed"), true},
		"already taken":   {fmt.Errorf("%w: conversation turn already consumed", conversation.ErrConflict), false},
	} {
		t.Run(name, func(t *testing.T) {
			var audit bytes.Buffer
			handler, err := NewHandler([]Backend{{Name: "om3", Conversations: cancelFailing{err: test.err}, Authenticator: allowTestTokenVerifier()}},
				HandlerConfig{MaxConcurrentAsks: 2, AuditLogger: slog.New(slog.NewTextHandler(&audit, nil))})
			if err != nil {
				t.Fatal(err)
			}
			handler.ServeHTTP(failingWriter{httptest.NewRecorder()}, newAuthenticatedRunRequest(runBody("hello")))
			logged := strings.Contains(audit.String(), "conversation_run_cancel_failed") && strings.Contains(audit.String(), "code=stream_not_started")
			if logged != test.audited || strings.Contains(audit.String(), "database closed") {
				t.Fatalf("audited %v, want %v: %s", logged, test.audited, audit.String())
			}
		})
	}
}
