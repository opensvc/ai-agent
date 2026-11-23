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
	llmPrefix                  = "OPENSVC_AI_LLM_"
	LLMAPITokenEnv             = llmPrefix + "API_TOKEN"
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
	Protocol string
	BaseURL  string
	Model    string
	AuthMode string
	// AuthHeader is the header of the key in the api_key mode.
	AuthHeader      string
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

// LoadLLM loads the default model, of the OPENSVC_AI_LLM_* variables.
func LoadLLM() (LLMConfig, error) {
	return loadLLM(os.Getenv)
}

func loadLLM(getenv func(string) string) (LLMConfig, error) {
	return loadLLMVariables(llmPrefix, LLMAPITokenEnv, getenv)
}

// loadLLMVariables loads the model of the variables under prefix, its key
// in the variable tokenEnv.
func loadLLMVariables(prefix, tokenEnv string, getenv func(string) string) (LLMConfig, error) {
	config := LLMConfig{
		Protocol:        strings.TrimSpace(getenv(prefix + "PROTOCOL")),
		BaseURL:         strings.TrimSpace(getenv(prefix + "BASE_URL")),
		Model:           strings.TrimSpace(getenv(prefix + "MODEL")),
		AuthMode:        strings.TrimSpace(getenv(prefix + "AUTH_MODE")),
		APITokenEnv:     tokenEnv,
		Timeout:         DefaultLLMTimeout,
		MaxOutputTokens: DefaultMaxOutputTokens,
	}

	if config.Protocol == "" {
		return LLMConfig{}, fmt.Errorf("%sPROTOCOL is required", prefix)
	}
	if config.Protocol != LLMProtocolResponses && config.Protocol != LLMProtocolChatCompletions && config.Protocol != LLMProtocolMessages {
		return LLMConfig{}, fmt.Errorf("%sPROTOCOL %q is unsupported", prefix, config.Protocol)
	}
	if config.BaseURL == "" {
		return LLMConfig{}, fmt.Errorf("%sBASE_URL is required", prefix)
	}
	if config.Model == "" {
		return LLMConfig{}, fmt.Errorf("%sMODEL is required", prefix)
	}
	switch config.AuthMode {
	case LLMAuthModeNone:
	case LLMAuthModeBearer, LLMAuthModeAPIKey:
		if getenv(tokenEnv) == "" {
			return LLMConfig{}, fmt.Errorf("%s is required for %s authentication", tokenEnv, config.AuthMode)
		}
	default:
		return LLMConfig{}, fmt.Errorf("%sAUTH_MODE must be %q, %q or %q", prefix, LLMAuthModeNone, LLMAuthModeBearer, LLMAuthModeAPIKey)
	}
	// The key goes in the header the provider reads: x-api-key for the
	// Messages protocol of Anthropic, api-key for an Azure OpenAI gateway,
	// or the one the operator names, such as Ocp-Apim-Subscription-Key.
	authHeader := strings.TrimSpace(getenv(prefix + "AUTH_HEADER"))
	switch {
	case config.AuthMode != LLMAuthModeAPIKey:
		if authHeader != "" {
			return LLMConfig{}, fmt.Errorf("%sAUTH_HEADER requires the %s authentication", prefix, LLMAuthModeAPIKey)
		}
	case authHeader != "":
		if err := validateAuthHeader(authHeader); err != nil {
			return LLMConfig{}, fmt.Errorf("%sAUTH_HEADER %q: %w", prefix, authHeader, err)
		}
		config.AuthHeader = authHeader
	case config.Protocol == LLMProtocolMessages:
		config.AuthHeader = "x-api-key"
	default:
		config.AuthHeader = "api-key"
	}

	if value := strings.TrimSpace(getenv(prefix + "TIMEOUT")); value != "" {
		timeout, err := time.ParseDuration(value)
		if err != nil || timeout <= 0 {
			return LLMConfig{}, fmt.Errorf("parse %sTIMEOUT %q: expected a positive duration", prefix, value)
		}
		config.Timeout = timeout
	}
	if value := strings.TrimSpace(getenv(prefix + "MAX_OUTPUT_TOKENS")); value != "" {
		maxOutputTokens, err := strconv.Atoi(value)
		if err != nil || maxOutputTokens <= 0 || maxOutputTokens > maximumMaxOutputTokens {
			return LLMConfig{}, fmt.Errorf("parse %sMAX_OUTPUT_TOKENS %q: expected an integer between 1 and %d", prefix, value, maximumMaxOutputTokens)
		}
		config.MaxOutputTokens = maxOutputTokens
	}
	if value := strings.TrimSpace(getenv(prefix + "REASONING_EFFORT")); value != "" {
		if !slices.Contains(LLMReasoningEfforts, value) {
			return LLMConfig{}, fmt.Errorf("%sREASONING_EFFORT %q must be one of %s", prefix, value, strings.Join(LLMReasoningEfforts, ", "))
		}
		if config.Protocol == LLMProtocolMessages {
			return LLMConfig{}, fmt.Errorf("%sREASONING_EFFORT requires the responses or chat_completions protocol", prefix)
		}
		config.ReasoningEffort = value
	}
	// store false forbids the provider to keep a request. Left out, the
	// retention policy of the provider applies, which the administrator
	// checks: OpenAI stores by default the Responses requests, and the Chat
	// Completions requests of new accounts. omit is allowed on the Chat
	// Completions protocol only, for a provider refusing the field, such as
	// Mistral, and only when the administrator asks it.
	switch value := strings.TrimSpace(getenv(prefix + "STORE")); value {
	case "", "false":
	case "omit":
		if config.Protocol != LLMProtocolChatCompletions {
			return LLMConfig{}, fmt.Errorf("%sSTORE omit requires the chat_completions protocol", prefix)
		}
		config.OmitStore = true
	default:
		return LLMConfig{}, fmt.Errorf("%sSTORE %q must be false or omit", prefix, value)
	}
	return config, nil
}

// authHeaderTokenChars are the characters of a header name besides letters
// and digits (RFC 9110, token).
const authHeaderTokenChars = "!#$%&'*+-.^_`|~"

// reservedAuthHeaders are the headers the adapters set themselves, or that
// mean something to the transport: a key there would replace them.
var reservedAuthHeaders = []string{
	"accept", "accept-encoding", "anthropic-version", "authorization", "connection", "content-encoding", "content-length",
	"content-type", "cookie", "date", "expect", "host", "keep-alive", "origin", "proxy-authorization", "proxy-connection",
	"range", "te", "trailer", "transfer-encoding", "upgrade", "user-agent", "via", "x-request-id",
}

// validateAuthHeader checks the name of the header of a key: a token of at
// most 64 characters that no adapter or transport sets.
func validateAuthHeader(name string) error {
	if len(name) > 64 {
		return fmt.Errorf("must have at most 64 characters")
	}
	for _, char := range name {
		if !(char >= 'a' && char <= 'z' || char >= 'A' && char <= 'Z' || char >= '0' && char <= '9' || strings.ContainsRune(authHeaderTokenChars, char)) {
			return fmt.Errorf("is not a valid header name")
		}
	}
	if slices.Contains(reservedAuthHeaders, strings.ToLower(name)) {
		return fmt.Errorf("is a header the agent or the transport sets")
	}
	return nil
}
