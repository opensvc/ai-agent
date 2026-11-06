package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/opensvc/ai-agent/internal/conversation"
)

const DefaultConversationDatabasePath = "/var/lib/opensvc-ai-agent/conversations.db"

type ConversationConfig struct {
	DatabasePath      string
	Lifetime          time.Duration
	FeedbackRetention time.Duration
	// The limits of the store, zero for its defaults: the conversations of
	// the instance, those of one user, the size of the database, and the
	// share of it one user stores.
	MaxConversations     int
	MaxUserConversations int
	MaxDatabaseBytes     int64
	MaxUserBytes         int64
}

func LoadConversation() (ConversationConfig, error) {
	return loadConversation(os.Getenv)
}

func loadConversation(getenv func(string) string) (ConversationConfig, error) {
	databasePath := strings.TrimSpace(getenv("OPENSVC_AI_CONVERSATION_DB_PATH"))
	if databasePath == "" {
		databasePath = DefaultConversationDatabasePath
	}
	lifetime := conversation.DefaultLifetime
	if value := strings.TrimSpace(getenv("OPENSVC_AI_CONVERSATION_LIFETIME")); value != "" {
		parsed, err := time.ParseDuration(value)
		if err != nil || parsed < time.Hour || parsed > 365*24*time.Hour {
			return ConversationConfig{}, fmt.Errorf("parse OPENSVC_AI_CONVERSATION_LIFETIME %q: expected a duration between 1h and 8760h", value)
		}
		lifetime = parsed
	}
	retention := conversation.DefaultFeedbackRetention
	if value := strings.TrimSpace(getenv("OPENSVC_AI_FEEDBACK_RETENTION")); value != "" {
		parsed, err := time.ParseDuration(value)
		if err != nil || parsed < 24*time.Hour || parsed > 3*365*24*time.Hour {
			return ConversationConfig{}, fmt.Errorf("parse OPENSVC_AI_FEEDBACK_RETENTION %q: expected a duration between 24h and 26280h", value)
		}
		retention = parsed
	}
	maxConversations, err := conversationLimit(getenv, "OPENSVC_AI_CONVERSATION_MAX", 1, 1_000_000)
	if err != nil {
		return ConversationConfig{}, err
	}
	maxUserConversations, err := conversationLimit(getenv, "OPENSVC_AI_CONVERSATION_MAX_PER_USER", 1, 1_000_000)
	if err != nil {
		return ConversationConfig{}, err
	}
	if maxConversations > 0 && maxUserConversations > maxConversations {
		return ConversationConfig{}, fmt.Errorf("OPENSVC_AI_CONVERSATION_MAX_PER_USER %d exceeds OPENSVC_AI_CONVERSATION_MAX %d", maxUserConversations, maxConversations)
	}
	// The database holds one conversation of the largest size at least,
	// 32 MiB.
	maxDatabaseMiB, err := conversationLimit(getenv, "OPENSVC_AI_CONVERSATION_MAX_DATABASE_MIB", 32, 1<<20)
	if err != nil {
		return ConversationConfig{}, err
	}
	// One user stores one conversation of the largest size at least; the
	// store refuses a share larger than the database when it opens.
	maxUserMiB, err := conversationLimit(getenv, "OPENSVC_AI_CONVERSATION_MAX_USER_MIB", 32, 1<<20)
	if err != nil {
		return ConversationConfig{}, err
	}
	return ConversationConfig{
		DatabasePath: databasePath, Lifetime: lifetime, FeedbackRetention: retention,
		MaxConversations: maxConversations, MaxUserConversations: maxUserConversations, MaxDatabaseBytes: int64(maxDatabaseMiB) << 20,
		MaxUserBytes: int64(maxUserMiB) << 20,
	}, nil
}

// conversationLimit reads an optional integer limit between low and high,
// zero when unset.
func conversationLimit(getenv func(string) string, name string, low, high int) (int, error) {
	value := strings.TrimSpace(getenv(name))
	if value == "" {
		return 0, nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < low || parsed > high {
		return 0, fmt.Errorf("parse %s %q: expected an integer between %d and %d", name, value, low, high)
	}
	return parsed, nil
}
