package api

import (
	"context"
	"encoding/json"
	"errors"
	"io"
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

func TestConversationFlowPersistsHistoryAndIsolatesOwner(t *testing.T) {
	directory := t.TempDir()
	if err := os.Chmod(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	store, err := conversationsqlite.Open(t.Context(), conversationsqlite.Config{Path: filepath.Join(directory, "conversations.db")})
	if err != nil {
		t.Fatalf("open conversation store: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	turnNumber := 0
	runner := apiTurnRunnerFunc(func(_ context.Context, history []llm.Message, prompt string, emit agent.EmitFunc) (agent.TurnResult, error) {
		turnNumber++
		if turnNumber == 1 && len(history) != 0 {
			t.Fatalf("first turn history=%#v", history)
		}
		if turnNumber == 2 && (len(history) != 2 || history[0].Text != "first") {
			t.Fatalf("second turn history=%#v", history)
		}
		answer := "answer " + prompt
		if err := emit(agent.Event{Type: agent.EventTextDelta, TextDelta: answer, Iteration: 1}); err != nil {
			return agent.TurnResult{}, err
		}
		if err := emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, Iteration: 1}); err != nil {
			return agent.TurnResult{}, err
		}
		return agent.TurnResult{Messages: []llm.Message{
			{Role: llm.RoleUser, Text: prompt}, {Role: llm.RoleAssistant, Text: answer},
		}, FinishReason: llm.FinishReasonCompleted}, nil
	})
	service, err := conversation.NewService(store, runner, conversation.ServiceConfig{Backend: "om3"})
	if err != nil {
		t.Fatalf("create conversation service: %v", err)
	}
	verifier := tokenVerifierFunc(func(_ context.Context, token string) (auth.Identity, error) {
		return auth.Identity{ClusterID: "cluster-id", Issuer: "cluster", Subject: token}, nil
	})
	handler, err := NewHandler([]Backend{{Name: "om3", Conversations: service, Authenticator: verifier}}, HandlerConfig{Models: testModels(), MaxConcurrentAsks: 4, AuditLogger: slog.New(slog.NewTextHandler(io.Discard, nil))})
	if err != nil {
		t.Fatalf("create handler: %v", err)
	}

	createResponse := httptest.NewRecorder()
	handler.ServeHTTP(createResponse, requestWithToken(http.MethodPost, "/v1/om3/conversations", "alice", ""))
	if createResponse.Code != http.StatusCreated {
		t.Fatalf("create status=%d body=%s", createResponse.Code, createResponse.Body.String())
	}
	var created ConversationEnvelope
	if err := json.NewDecoder(createResponse.Body).Decode(&created); err != nil {
		t.Fatal(err)
	}
	id := created.Conversation.ID
	if created.Conversation.Title != "" {
		t.Fatalf("new conversation title=%q", created.Conversation.Title)
	}

	otherOwnerResponse := httptest.NewRecorder()
	handler.ServeHTTP(otherOwnerResponse, requestWithToken(http.MethodGet, "/v1/om3/conversations/"+id, "bob", ""))
	if otherOwnerResponse.Code != http.StatusNotFound {
		t.Fatalf("other owner status=%d body=%s", otherOwnerResponse.Code, otherOwnerResponse.Body.String())
	}

	for index, prompt := range []string{"first", "second"} {
		response := httptest.NewRecorder()
		request := requestWithToken(http.MethodPost, "/v1/om3/conversations/"+id+"/runs", "alice", runInput(id, prompt))
		request.Header.Set("Content-Type", "application/json")
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"type":"RUN_FINISHED"`) {
			t.Fatalf("run %q status=%d body=%s", prompt, response.Code, response.Body.String())
		}
		if index == 0 {
			messagesResponse := httptest.NewRecorder()
			handler.ServeHTTP(messagesResponse, requestWithToken(http.MethodGet, "/v1/om3/conversations/"+id+"/messages", "alice", ""))
			var page conversation.MessagePage
			if err := json.NewDecoder(messagesResponse.Body).Decode(&page); err != nil || messagesResponse.Code != http.StatusOK || len(page.Messages) != 2 || page.Messages[0].Text != "first" || page.Messages[1].Text != "answer first" {
				t.Fatalf("persisted display = %+v, status = %d, error = %v", page, messagesResponse.Code, err)
			}
			if turnNumber != 1 {
				t.Fatal("reading messages ran a model turn")
			}
			// The end of the run names the stored turn, the one its messages
			// carry, for the client to rate the answer without reloading.
			if want := `"turnId":"` + page.Messages[1].TurnID + `"`; !strings.Contains(response.Body.String(), want) {
				t.Fatalf("RUN_FINISHED without %s: %s", want, response.Body.String())
			}
			foreignMessages := httptest.NewRecorder()
			handler.ServeHTTP(foreignMessages, requestWithToken(http.MethodGet, "/v1/om3/conversations/"+id+"/messages", "bob", ""))
			if foreignMessages.Code != http.StatusNotFound || strings.Contains(foreignMessages.Body.String(), "answer first") {
				t.Fatalf("foreign display response = %d, %s", foreignMessages.Code, foreignMessages.Body)
			}
			getResponse := httptest.NewRecorder()
			handler.ServeHTTP(getResponse, requestWithToken(http.MethodGet, "/v1/om3/conversations/"+id, "alice", ""))
			var afterFirst ConversationEnvelope
			if err := json.NewDecoder(getResponse.Body).Decode(&afterFirst); err != nil || afterFirst.Conversation.Title != "first" {
				t.Fatalf("first title response=%+v error=%v", afterFirst, err)
			}

			foreignUpdate := requestWithToken(http.MethodPatch, "/v1/om3/conversations/"+id, "bob", `{"title":"Foreign"}`)
			foreignUpdate.Header.Set("Content-Type", "application/json")
			foreignUpdateResponse := httptest.NewRecorder()
			handler.ServeHTTP(foreignUpdateResponse, foreignUpdate)
			if foreignUpdateResponse.Code != http.StatusNotFound {
				t.Fatalf("foreign update status=%d body=%s", foreignUpdateResponse.Code, foreignUpdateResponse.Body.String())
			}

			update := requestWithToken(http.MethodPatch, "/v1/om3/conversations/"+id, "alice", `{"title":"Renamed incident"}`)
			update.Header.Set("Content-Type", "application/json")
			updateResponse := httptest.NewRecorder()
			handler.ServeHTTP(updateResponse, update)
			var renamed ConversationEnvelope
			if err := json.NewDecoder(updateResponse.Body).Decode(&renamed); err != nil || renamed.Conversation.Title != "Renamed incident" {
				t.Fatalf("rename response=%+v error=%v status=%d", renamed, err, updateResponse.Code)
			}
		}
	}
	getResponse := httptest.NewRecorder()
	handler.ServeHTTP(getResponse, requestWithToken(http.MethodGet, "/v1/om3/conversations/"+id, "alice", ""))
	var afterSecond ConversationEnvelope
	if err := json.NewDecoder(getResponse.Body).Decode(&afterSecond); err != nil || afterSecond.Conversation.Title != "Renamed incident" {
		t.Fatalf("final title response=%+v error=%v", afterSecond, err)
	}
	history, err := store.LoadHistory(t.Context(), conversation.Owner{Backend: "om3", ClusterID: "cluster-id", Issuer: "cluster", Subject: "alice"}, id)
	if err != nil || len(history) != 4 {
		t.Fatalf("stored history=%#v error=%v", history, err)
	}
}

type apiTurnRunnerFunc func(context.Context, []llm.Message, string, agent.EmitFunc) (agent.TurnResult, error)

func (f apiTurnRunnerFunc) RunTurn(ctx context.Context, history []llm.Message, prompt string, emit agent.EmitFunc) (agent.TurnResult, error) {
	return f(ctx, history, prompt, emit)
}

func (f apiTurnRunnerFunc) ResumeTurn(context.Context, []llm.Message, agent.SuspendedTurn, agent.Decision, agent.EmitFunc) (agent.TurnResult, error) {
	return agent.TurnResult{}, errors.New("this runner does not resume turns")
}

func requestWithToken(method string, path string, token string, body string) *http.Request {
	request := httptest.NewRequest(method, path, strings.NewReader(body))
	request.Header.Set("Authorization", "Bearer "+token)
	request.Header.Set(testBackendHeader, "cluster-id")
	return request
}
