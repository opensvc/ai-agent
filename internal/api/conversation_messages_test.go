package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/conversation"
)

func TestConversationMessagesRouteUsesVerifiedIdentityAndDoesNotLogText(t *testing.T) {
	var logs bytes.Buffer
	identity := auth.Identity{ClusterID: "cluster-id", Issuer: "issuer", Subject: "alice"}
	page := conversation.MessagePage{Messages: []conversation.DisplayMessage{
		{ID: "turn-1:1", TurnID: "turn-1", Role: "user", Text: "private prompt", CreatedAt: time.Now().UTC()},
	}, NextCursor: "1:1"}
	service := conversationServiceFuncs{messages: func(_ context.Context, got auth.Identity, id string, query conversation.MessageQuery) (conversation.MessagePage, error) {
		if got.ClusterID != identity.ClusterID || got.Issuer != identity.Issuer || got.Subject != identity.Subject || id != "conversation-1" || query.Limit != 2 || query.Before != "3:4" {
			t.Fatalf("messages input = %+v, %s, %+v", got, id, query)
		}
		return page, nil
	}}
	handler, err := NewHandler([]Backend{{Name: "om3", Conversations: service, Authenticator: tokenVerifierFunc(func(context.Context, string) (auth.Identity, error) {
		return identity, nil
	})}}, HandlerConfig{Models: testModels(), MaxConcurrentAsks: 1, AuditLogger: slog.New(slog.NewTextHandler(&logs, nil)), CORSAllowedOrigins: []string{"https://webapp.example"}})
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	request := authenticatedRequest(http.MethodGet, "/v1/om3/conversations/conversation-1/messages?limit=2&before=3%3A4", "")
	request.Header.Set("Origin", "https://webapp.example")
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK || response.Header().Get("Cache-Control") != "no-store" || response.Header().Get("Access-Control-Allow-Origin") != "https://webapp.example" {
		t.Fatalf("messages response = %d, %v, %s", response.Code, response.Header(), response.Body)
	}
	var got conversation.MessagePage
	if err := json.NewDecoder(response.Body).Decode(&got); err != nil || len(got.Messages) != 1 || got.Messages[0].Text != "private prompt" || got.NextCursor != "1:1" {
		t.Fatalf("messages body = %+v, %v", got, err)
	}
	if !strings.Contains(logs.String(), "conversation_messages_read") || strings.Contains(logs.String(), "private prompt") || strings.Contains(logs.String(), "Bearer") {
		t.Fatalf("unexpected messages audit = %s", logs.String())
	}
}

func TestConversationMessagesRejectsInvalidPaginationBeforeStorage(t *testing.T) {
	handler := newConversationTestHandler(t, conversationServiceFuncs{messages: func(context.Context, auth.Identity, string, conversation.MessageQuery) (conversation.MessagePage, error) {
		t.Fatal("invalid query accessed storage")
		return conversation.MessagePage{}, nil
	}})
	for _, query := range []string{
		"limit=0", "limit=101", "limit=-1", "limit=01", "limit=x", "limit=", "limit=1&limit=2",
		"before=", "before=0:1", "before=01:1", "before=1:0", "before=1:2:3", "before=1:1&before=2:2",
		"before=%zz", "offset=1", strings.Repeat("x", 257),
	} {
		t.Run(query, func(t *testing.T) {
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, authenticatedRequest(http.MethodGet, "/v1/om3/conversations/id/messages?"+query, ""))
			if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), `"code":"invalid_request"`) {
				t.Fatalf("query %s response = %d, %s", query, response.Code, response.Body)
			}
		})
	}
}

func TestConversationMessagesDefaultEmptyPageAndStableErrors(t *testing.T) {
	for _, test := range []struct {
		name   string
		err    error
		status int
		code   string
	}{
		{"empty", nil, http.StatusOK, ""},
		{"missing-or-foreign", conversation.ErrNotFound, http.StatusNotFound, "conversation_not_found"},
		{"expired", conversation.ErrExpired, http.StatusGone, "conversation_expired"},
		{"oversized-message", conversation.ErrMessageTooLarge, http.StatusRequestEntityTooLarge, "history_message_too_large"},
		{"internal", errors.New("private database detail"), http.StatusInternalServerError, "conversation_failed"},
	} {
		t.Run(test.name, func(t *testing.T) {
			handler := newConversationTestHandler(t, conversationServiceFuncs{messages: func(_ context.Context, _ auth.Identity, _ string, query conversation.MessageQuery) (conversation.MessagePage, error) {
				if query.Limit != conversation.DefaultMessagePageLimit || query.Before != "" {
					t.Fatalf("default query = %+v", query)
				}
				return conversation.MessagePage{}, test.err
			}})
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, authenticatedRequest(http.MethodGet, "/v1/om3/conversations/id/messages", ""))
			if response.Code != test.status || strings.Contains(response.Body.String(), "private database detail") {
				t.Fatalf("error response = %d, %s", response.Code, response.Body)
			}
			if test.err == nil && response.Body.String() != "{\"messages\":[],\"next_cursor\":\"\"}\n" {
				t.Fatalf("empty response = %s", response.Body)
			}
			if test.code != "" && !strings.Contains(response.Body.String(), `"code":"`+test.code+`"`) {
				t.Fatalf("error code response = %s", response.Body)
			}
		})
	}
}
