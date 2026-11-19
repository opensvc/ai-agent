package config

import (
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"
)

const (
	LLMAPITokenEnv             = "OPENSVC_AI_LLM_API_TOKEN"
	DefaultLLMTimeout          = 2 * time.Minute
	DefaultMaxOutputTokens     = 4096
	maximumMaxOutputTokens     = 131072
	LLMProtocolResponses       = "responses"
	LLMProtocolChatCompletions = "chat_completions"
	LLMProtocolMessages        = "messages"
	LLMAuthModeNone            = "none"
	LLMAuthModeBearer          = "bearer"
	LLMAuthModeAPIKey          = "api_key"
)

// LLMReasoningEfforts are the reasoning efforts OPENSVC_AI_LLM_REASONING_EFFORT
// takes; a model accepts some of them only, and refuses the others at its
// first request.
var LLMReasoningEfforts = []string{"none", "minimal", "low", "medium", "high", "xhigh"}

// LLMConfig contains non-secret process configuration for one LLM backend.
// APITokenEnv stores only the environment variable name, never its value.
type LLMConfig struct {
	Protocol        string
	BaseURL         string
	Model           string
	AuthMode        string
	APITokenEnv     string
	Timeout         time.Duration
	MaxOutputTokens int
	// ReasoningEffort is the reasoning effort the requests ask the model
	// for; empty, the requests ask none, and the model applies its own.
	ReasoningEffort string
	// OmitStore leaves the store field out of the Chat Completions requests,
	// for a provider that refuses it, such as Mistral: its retention policy
	// then applies. By default each request sends store false, forbidding
	// the provider to keep it.
	OmitStore bool
}

// LoadLLM loads the LLM backend independently from the HTTP API configuration.
// The daemon calls it at startup to build the LLM client shared by all
// backends.
func LoadLLM() (LLMConfig, error) {
	return loadLLM(os.Getenv)
}

func loadLLM(getenv func(string) string) (LLMConfig, error) {
	config := LLMConfig{
		Protocol:        strings.TrimSpace(getenv("OPENSVC_AI_LLM_PROTOCOL")),
		BaseURL:         strings.TrimSpace(getenv("OPENSVC_AI_LLM_BASE_URL")),
		Model:           strings.TrimSpace(getenv("OPENSVC_AI_LLM_MODEL")),
		AuthMode:        strings.TrimSpace(getenv("OPENSVC_AI_LLM_AUTH_MODE")),
		APITokenEnv:     LLMAPITokenEnv,
		Timeout:         DefaultLLMTimeout,
		MaxOutputTokens: DefaultMaxOutputTokens,
	}

	if config.Protocol == "" {
		return LLMConfig{}, fmt.Errorf("OPENSVC_AI_LLM_PROTOCOL is required")
	}
	if config.Protocol != LLMProtocolResponses && config.Protocol != LLMProtocolChatCompletions && config.Protocol != LLMProtocolMessages {
		return LLMConfig{}, fmt.Errorf("OPENSVC_AI_LLM_PROTOCOL %q is unsupported", config.Protocol)
	}
	if config.BaseURL == "" {
		return LLMConfig{}, fmt.Errorf("OPENSVC_AI_LLM_BASE_URL is required")
	}
	if config.Model == "" {
		return LLMConfig{}, fmt.Errorf("OPENSVC_AI_LLM_MODEL is required")
	}
	switch config.AuthMode {
	case LLMAuthModeNone:
	case LLMAuthModeBearer, LLMAuthModeAPIKey:
		if config.AuthMode == LLMAuthModeAPIKey && config.Protocol != LLMProtocolMessages {
			return LLMConfig{}, fmt.Errorf("OPENSVC_AI_LLM_AUTH_MODE api_key requires the messages protocol")
		}
		if getenv(LLMAPITokenEnv) == "" {
			return LLMConfig{}, fmt.Errorf("%s is required for %s authentication", LLMAPITokenEnv, config.AuthMode)
		}
	default:
		return LLMConfig{}, fmt.Errorf("OPENSVC_AI_LLM_AUTH_MODE must be %q, %q or %q (messages only)", LLMAuthModeNone, LLMAuthModeBearer, LLMAuthModeAPIKey)
	}

	if value := strings.TrimSpace(getenv("OPENSVC_AI_LLM_TIMEOUT")); value != "" {
		timeout, err := time.ParseDuration(value)
		if err != nil || timeout <= 0 {
			return LLMConfig{}, fmt.Errorf("parse OPENSVC_AI_LLM_TIMEOUT %q: expected a positive duration", value)
		}
		config.Timeout = timeout
	}
	if value := strings.TrimSpace(getenv("OPENSVC_AI_LLM_MAX_OUTPUT_TOKENS")); value != "" {
		maxOutputTokens, err := strconv.Atoi(value)
		if err != nil || maxOutputTokens <= 0 || maxOutputTokens > maximumMaxOutputTokens {
			return LLMConfig{}, fmt.Errorf("parse OPENSVC_AI_LLM_MAX_OUTPUT_TOKENS %q: expected an integer between 1 and %d", value, maximumMaxOutputTokens)
		}
		config.MaxOutputTokens = maxOutputTokens
	}
	if value := strings.TrimSpace(getenv("OPENSVC_AI_LLM_REASONING_EFFORT")); value != "" {
		if !slices.Contains(LLMReasoningEfforts, value) {
			return LLMConfig{}, fmt.Errorf("OPENSVC_AI_LLM_REASONING_EFFORT %q must be one of %s", value, strings.Join(LLMReasoningEfforts, ", "))
		}
		if config.Protocol == LLMProtocolMessages {
			return LLMConfig{}, fmt.Errorf("OPENSVC_AI_LLM_REASONING_EFFORT requires the responses or chat_completions protocol")
		}
		config.ReasoningEffort = value
	}
	// store false forbids the provider to keep a request. Left out, the
	// retention policy of the provider applies, which the administrator
	// checks: OpenAI stores by default the Responses requests, and the Chat
	// Completions requests of new accounts. omit is allowed on the Chat
	// Completions protocol only, for a provider refusing the field, such as
	// Mistral, and only when the administrator asks it.
	switch value := strings.TrimSpace(getenv("OPENSVC_AI_LLM_STORE")); value {
	case "", "false":
	case "omit":
		if config.Protocol != LLMProtocolChatCompletions {
			return LLMConfig{}, fmt.Errorf("OPENSVC_AI_LLM_STORE omit requires the chat_completions protocol")
		}
		config.OmitStore = true
	default:
		return LLMConfig{}, fmt.Errorf("OPENSVC_AI_LLM_STORE %q must be false or omit", value)
	}
	return config, nil
}
