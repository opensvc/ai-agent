package config

import (
	"strings"
	"testing"
	"time"
)

func TestLoadLLM(t *testing.T) {
	values := map[string]string{
		"OPENSVC_AI_LLM_PROTOCOL":          "responses",
		"OPENSVC_AI_LLM_BASE_URL":          "https://llm.example.test/v1",
		"OPENSVC_AI_LLM_MODEL":             "test-model",
		"OPENSVC_AI_LLM_AUTH_MODE":         "bearer",
		"OPENSVC_AI_LLM_API_TOKEN":         "present",
		"OPENSVC_AI_LLM_TIMEOUT":           "45s",
		"OPENSVC_AI_LLM_MAX_OUTPUT_TOKENS": "2048",
	}
	config, err := loadLLM(func(key string) string { return values[key] })
	if err != nil {
		t.Fatalf("load LLM config: %v", err)
	}
	if config.Protocol != LLMProtocolResponses || config.BaseURL != values["OPENSVC_AI_LLM_BASE_URL"] || config.Model != "test-model" {
		t.Fatalf("unexpected LLM config: %+v", config)
	}
	if config.AuthMode != LLMAuthModeBearer || config.APITokenEnv != LLMAPITokenEnv {
		t.Fatalf("unexpected LLM auth config: %+v", config)
	}
	if config.Timeout != 45*time.Second || config.MaxOutputTokens != 2048 {
		t.Fatalf("unexpected LLM limits: %+v", config)
	}
}

func TestLoadLLMDefaultsForLocalBackend(t *testing.T) {
	values := map[string]string{
		"OPENSVC_AI_LLM_PROTOCOL":  "responses",
		"OPENSVC_AI_LLM_BASE_URL":  "http://127.0.0.1:8081/v1",
		"OPENSVC_AI_LLM_MODEL":     "local-model",
		"OPENSVC_AI_LLM_AUTH_MODE": "none",
	}
	config, err := loadLLM(func(key string) string { return values[key] })
	if err != nil {
		t.Fatalf("load local LLM config: %v", err)
	}
	if config.Timeout != DefaultLLMTimeout || config.MaxOutputTokens != DefaultMaxOutputTokens {
		t.Fatalf("got defaults timeout=%s max=%d", config.Timeout, config.MaxOutputTokens)
	}
}

func TestLoadLLMAcceptsChatCompletions(t *testing.T) {
	values := map[string]string{
		"OPENSVC_AI_LLM_PROTOCOL":  LLMProtocolChatCompletions,
		"OPENSVC_AI_LLM_BASE_URL":  "http://127.0.0.1:8081/v1",
		"OPENSVC_AI_LLM_MODEL":     "local-model",
		"OPENSVC_AI_LLM_AUTH_MODE": "none",
	}
	config, err := loadLLM(func(key string) string { return values[key] })
	if err != nil {
		t.Fatalf("load Chat Completions config: %v", err)
	}
	if config.Protocol != LLMProtocolChatCompletions {
		t.Fatalf("got protocol %q", config.Protocol)
	}
}

func TestLoadLLMRejectsInvalidConfiguration(t *testing.T) {
	valid := map[string]string{
		"OPENSVC_AI_LLM_PROTOCOL":  "responses",
		"OPENSVC_AI_LLM_BASE_URL":  "https://llm.example.test/v1",
		"OPENSVC_AI_LLM_MODEL":     "test-model",
		"OPENSVC_AI_LLM_AUTH_MODE": "none",
	}
	for _, test := range []struct {
		name   string
		key    string
		value  string
		delete bool
		want   string
	}{
		{name: "missing protocol", key: "OPENSVC_AI_LLM_PROTOCOL", delete: true, want: "PROTOCOL is required"},
		{name: "unsupported protocol", key: "OPENSVC_AI_LLM_PROTOCOL", value: "unknown", want: "unsupported"},
		{name: "missing URL", key: "OPENSVC_AI_LLM_BASE_URL", delete: true, want: "BASE_URL is required"},
		{name: "missing model", key: "OPENSVC_AI_LLM_MODEL", delete: true, want: "MODEL is required"},
		{name: "invalid auth", key: "OPENSVC_AI_LLM_AUTH_MODE", value: "basic", want: "AUTH_MODE"},
		{name: "invalid timeout", key: "OPENSVC_AI_LLM_TIMEOUT", value: "never", want: "TIMEOUT"},
		{name: "zero timeout", key: "OPENSVC_AI_LLM_TIMEOUT", value: "0s", want: "positive duration"},
		{name: "invalid max tokens", key: "OPENSVC_AI_LLM_MAX_OUTPUT_TOKENS", value: "0", want: "between 1"},
	} {
		t.Run(test.name, func(t *testing.T) {
			values := make(map[string]string, len(valid)+1)
			for key, value := range valid {
				values[key] = value
			}
			if test.delete {
				delete(values, test.key)
			} else {
				values[test.key] = test.value
			}
			_, err := loadLLM(func(key string) string { return values[key] })
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("loadLLM() error = %v, want containing %q", err, test.want)
			}
		})
	}
}

func TestLoadLLMRequiresBearerTokenWithoutRetainingIt(t *testing.T) {
	values := map[string]string{
		"OPENSVC_AI_LLM_PROTOCOL":  "responses",
		"OPENSVC_AI_LLM_BASE_URL":  "https://llm.example.test/v1",
		"OPENSVC_AI_LLM_MODEL":     "test-model",
		"OPENSVC_AI_LLM_AUTH_MODE": "bearer",
	}
	if _, err := loadLLM(func(key string) string { return values[key] }); err == nil || !strings.Contains(err.Error(), LLMAPITokenEnv) {
		t.Fatalf("missing bearer token error = %v", err)
	}

	values[LLMAPITokenEnv] = "not-retained"
	config, err := loadLLM(func(key string) string { return values[key] })
	if err != nil {
		t.Fatalf("load bearer config: %v", err)
	}
	if strings.Contains(config.BaseURL+config.Model+config.Protocol+config.AuthMode+config.APITokenEnv, values[LLMAPITokenEnv]) {
		t.Fatal("LLM config retained the API token")
	}
}

func TestLoadLLMMessagesAuthentication(t *testing.T) {
	for _, mode := range []string{LLMAuthModeAPIKey, LLMAuthModeBearer, LLMAuthModeNone} {
		t.Run(mode, func(t *testing.T) {
			values := map[string]string{
				"OPENSVC_AI_LLM_PROTOCOL":  LLMProtocolMessages,
				"OPENSVC_AI_LLM_BASE_URL":  "https://api.anthropic.com/v1",
				"OPENSVC_AI_LLM_MODEL":     "test-model",
				"OPENSVC_AI_LLM_AUTH_MODE": mode,
				LLMAPITokenEnv:             "not-retained",
			}
			getenv := func(key string) string { return values[key] }
			config, err := loadLLM(getenv)
			if err != nil || config.Protocol != LLMProtocolMessages || config.AuthMode != mode || config.APITokenEnv != LLMAPITokenEnv {
				t.Fatalf("config=%#v error=%v", config, err)
			}
			delete(values, LLMAPITokenEnv)
			if _, err := loadLLM(getenv); (err != nil) != (mode != LLMAuthModeNone) {
				t.Fatalf("missing key error=%v", err)
			}
		})
	}
	for _, protocol := range []string{LLMProtocolResponses, LLMProtocolChatCompletions} {
		values := map[string]string{"OPENSVC_AI_LLM_PROTOCOL": protocol, "OPENSVC_AI_LLM_BASE_URL": "https://api.example.test/v1", "OPENSVC_AI_LLM_MODEL": "test-model", "OPENSVC_AI_LLM_AUTH_MODE": LLMAuthModeAPIKey, LLMAPITokenEnv: "present"}
		if _, err := loadLLM(func(key string) string { return values[key] }); err == nil || !strings.Contains(err.Error(), "requires the messages protocol") {
			t.Fatalf("incompatible auth error=%v", err)
		}
	}
}

func TestLoadLLMStore(t *testing.T) {
	base := map[string]string{
		"OPENSVC_AI_LLM_PROTOCOL": "chat_completions", "OPENSVC_AI_LLM_BASE_URL": "https://llm.example.test/v1",
		"OPENSVC_AI_LLM_MODEL": "test-model", "OPENSVC_AI_LLM_AUTH_MODE": "none",
	}
	load := func(changes map[string]string) (LLMConfig, error) {
		values := map[string]string{}
		for key, value := range base {
			values[key] = value
		}
		for key, value := range changes {
			values[key] = value
		}
		return loadLLM(func(key string) string { return values[key] })
	}
	// Unset or false, every request sends store false.
	for _, value := range []string{"", "false", " false "} {
		if config, err := load(map[string]string{"OPENSVC_AI_LLM_STORE": value}); err != nil || config.OmitStore {
			t.Fatalf("%q: %+v, %v", value, config, err)
		}
	}
	// An explicit false stays allowed on every protocol.
	for _, protocol := range []string{"responses", "messages"} {
		if config, err := load(map[string]string{"OPENSVC_AI_LLM_PROTOCOL": protocol, "OPENSVC_AI_LLM_STORE": "false"}); err != nil || config.OmitStore {
			t.Fatalf("%s false: %+v, %v", protocol, config, err)
		}
	}
	if config, err := load(map[string]string{"OPENSVC_AI_LLM_STORE": "omit"}); err != nil || !config.OmitStore {
		t.Fatalf("omit: %+v, %v", config, err)
	}
	// The Responses protocol stores a request unless told not to: omit is
	// refused there, as on Messages, and any other value anywhere.
	for _, changes := range []map[string]string{
		{"OPENSVC_AI_LLM_STORE": "omit", "OPENSVC_AI_LLM_PROTOCOL": "responses"},
		{"OPENSVC_AI_LLM_STORE": "omit", "OPENSVC_AI_LLM_PROTOCOL": "messages"},
		{"OPENSVC_AI_LLM_STORE": "true"},
		{"OPENSVC_AI_LLM_STORE": "Omit"},
	} {
		if _, err := load(changes); err == nil {
			t.Fatalf("%v accepted", changes)
		}
	}
}

func TestLoadLLMReasoningEffort(t *testing.T) {
	base := map[string]string{
		"OPENSVC_AI_LLM_PROTOCOL": "responses", "OPENSVC_AI_LLM_BASE_URL": "https://llm.example.test/v1",
		"OPENSVC_AI_LLM_MODEL": "test-model", "OPENSVC_AI_LLM_AUTH_MODE": "none",
	}
	load := func(changes map[string]string) (LLMConfig, error) {
		values := map[string]string{}
		for key, value := range base {
			values[key] = value
		}
		for key, value := range changes {
			values[key] = value
		}
		return loadLLM(func(key string) string { return values[key] })
	}
	// Unset, the requests ask no effort: the model applies its own.
	if config, err := load(nil); err != nil || config.ReasoningEffort != "" {
		t.Fatalf("unset: %+v, %v", config, err)
	}
	for _, protocol := range []string{"responses", "chat_completions"} {
		if config, err := load(map[string]string{"OPENSVC_AI_LLM_PROTOCOL": protocol, "OPENSVC_AI_LLM_REASONING_EFFORT": " low "}); err != nil || config.ReasoningEffort != "low" {
			t.Fatalf("%s: %+v, %v", protocol, config, err)
		}
	}
	for _, changes := range []map[string]string{
		{"OPENSVC_AI_LLM_REASONING_EFFORT": "fast"},
		{"OPENSVC_AI_LLM_REASONING_EFFORT": "Low"},
		{"OPENSVC_AI_LLM_REASONING_EFFORT": "low", "OPENSVC_AI_LLM_PROTOCOL": "messages"},
	} {
		if _, err := load(changes); err == nil {
			t.Fatalf("%v accepted", changes)
		}
	}
}
