package config

import (
	"testing"
	"time"
)

func TestLoadConversation(t *testing.T) {
	config, err := loadConversation(func(key string) string {
		switch key {
		case "OPENSVC_AI_CONVERSATION_DB_PATH":
			return "/tmp/agent/conversations.db"
		case "OPENSVC_AI_CONVERSATION_LIFETIME":
			return "72h"
		default:
			return ""
		}
	})
	if err != nil {
		t.Fatalf("load conversation config: %v", err)
	}
	if config.DatabasePath != "/tmp/agent/conversations.db" || config.Lifetime != 72*time.Hour {
		t.Fatalf("unexpected conversation config %+v", config)
	}
}

func TestLoadConversationRejectsInvalidLifetime(t *testing.T) {
	for _, value := range []string{"invalid", "30m", "8761h"} {
		if _, err := loadConversation(func(key string) string {
			if key == "OPENSVC_AI_CONVERSATION_LIFETIME" {
				return value
			}
			return ""
		}); err == nil {
			t.Fatalf("lifetime %q succeeded", value)
		}
	}
}

func TestLoadConversationFeedbackRetention(t *testing.T) {
	config, err := loadConversation(func(string) string { return "" })
	if err != nil || config.FeedbackRetention != 180*24*time.Hour {
		t.Fatalf("default retention = %v, %v", config.FeedbackRetention, err)
	}
	config, err = loadConversation(func(key string) string {
		if key == "OPENSVC_AI_FEEDBACK_RETENTION" {
			return "720h"
		}
		return ""
	})
	if err != nil || config.FeedbackRetention != 720*time.Hour {
		t.Fatalf("retention = %v, %v", config.FeedbackRetention, err)
	}
	for _, value := range []string{"invalid", "23h", "26281h"} {
		if _, err := loadConversation(func(key string) string {
			if key == "OPENSVC_AI_FEEDBACK_RETENTION" {
				return value
			}
			return ""
		}); err == nil {
			t.Fatalf("retention %q succeeded", value)
		}
	}
}

func TestLoadConversationLimits(t *testing.T) {
	env := map[string]string{
		"OPENSVC_AI_CONVERSATION_MAX":              "2000",
		"OPENSVC_AI_CONVERSATION_MAX_PER_USER":     "100",
		"OPENSVC_AI_CONVERSATION_MAX_DATABASE_MIB": "1024",
		"OPENSVC_AI_CONVERSATION_MAX_USER_MIB":     "64",
	}
	config, err := loadConversation(func(key string) string { return env[key] })
	if err != nil {
		t.Fatal(err)
	}
	if config.MaxConversations != 2000 || config.MaxUserConversations != 100 || config.MaxDatabaseBytes != 1<<30 || config.MaxUserBytes != 64<<20 {
		t.Fatalf("limits = %+v", config)
	}
	config, err = loadConversation(func(string) string { return "" })
	if err != nil || config.MaxConversations != 0 || config.MaxUserConversations != 0 || config.MaxDatabaseBytes != 0 || config.MaxUserBytes != 0 {
		t.Fatalf("unset limits = %+v, %v", config, err)
	}
	for name, value := range map[string]map[string]string{
		"not a number":         {"OPENSVC_AI_CONVERSATION_MAX": "many"},
		"zero":                 {"OPENSVC_AI_CONVERSATION_MAX_PER_USER": "0"},
		"database too small":   {"OPENSVC_AI_CONVERSATION_MAX_DATABASE_MIB": "16"},
		"user share too small": {"OPENSVC_AI_CONVERSATION_MAX_USER_MIB": "16"},
		"quota over limit":     {"OPENSVC_AI_CONVERSATION_MAX": "10", "OPENSVC_AI_CONVERSATION_MAX_PER_USER": "11"},
	} {
		if _, err := loadConversation(func(key string) string { return value[key] }); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}
