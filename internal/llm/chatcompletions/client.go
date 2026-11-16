package chatcompletions

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/opensvc/ai-agent/internal/llm"
)

const (
	AuthModeNone   = "none"
	AuthModeBearer = "bearer"

	maxRequestBytes   = 4 << 20
	maxErrorBodyBytes = 64 << 10
)

type TokenSource func() (string, error)

type Config struct {
	BaseURL         string
	Model           string
	AuthMode        string
	TokenSource     TokenSource
	Timeout         time.Duration
	MaxOutputTokens int
	// ReasoningEffort is sent with each request when set.
	ReasoningEffort string
}

// Client implements llm.Client using the Chat Completions HTTP protocol.
type Client struct {
	endpoint        string
	model           string
	authMode        string
	tokenSource     TokenSource
	maxOutputTokens int
	reasoningEffort string
	httpClient      *http.Client
}

var _ llm.Client = (*Client)(nil)

func New(config Config, httpClient *http.Client) (*Client, error) {
	endpoint, err := chatCompletionsEndpoint(config.BaseURL)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(config.Model) == "" {
		return nil, fmt.Errorf("Chat Completions model is empty")
	}
	if config.Timeout <= 0 {
		return nil, fmt.Errorf("Chat Completions timeout must be positive")
	}
	if config.MaxOutputTokens <= 0 {
		return nil, fmt.Errorf("Chat Completions max output tokens must be positive")
	}
	switch config.AuthMode {
	case AuthModeNone:
	case AuthModeBearer:
		if config.TokenSource == nil {
			return nil, fmt.Errorf("Chat Completions bearer token source is missing")
		}
	default:
		return nil, fmt.Errorf("unsupported Chat Completions authentication mode %q", config.AuthMode)
	}

	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	clientCopy := *httpClient
	clientCopy.Timeout = config.Timeout
	clientCopy.CheckRedirect = func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}

	return &Client{
		endpoint:        endpoint,
		model:           config.Model,
		authMode:        config.AuthMode,
		tokenSource:     config.TokenSource,
		maxOutputTokens: config.MaxOutputTokens,
		reasoningEffort: config.ReasoningEffort,
		httpClient:      &clientCopy,
	}, nil
}

func (c *Client) Stream(ctx context.Context, request llm.Request, emit llm.EmitFunc) error {
	if err := request.Validate(); err != nil {
		return fmt.Errorf("validate LLM request: %w", err)
	}
	if emit == nil {
		return fmt.Errorf("stream Chat Completions: event consumer is nil")
	}

	wireRequest, err := newCreateRequest(c.model, c.maxOutputTokens, request)
	if err != nil {
		return err
	}
	wireRequest.ReasoningEffort = c.reasoningEffort
	body, err := json.Marshal(wireRequest)
	if err != nil {
		return fmt.Errorf("encode Chat Completions request: %w", err)
	}
	if len(body) > maxRequestBytes {
		return fmt.Errorf("encode Chat Completions request: body exceeds %d bytes", maxRequestBytes)
	}

	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, strings.NewReader(string(body)))
	if err != nil {
		return fmt.Errorf("create Chat Completions request: %w", err)
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("Accept", "text/event-stream")

	var token string
	if c.authMode == AuthModeBearer {
		token, err = c.tokenSource()
		if err != nil {
			return fmt.Errorf("load Chat Completions API token: %w", err)
		}
		if token == "" {
			return fmt.Errorf("load Chat Completions API token: token is empty")
		}
		httpRequest.Header.Set("Authorization", "Bearer "+token)
	}

	response, err := c.httpClient.Do(httpRequest)
	if err != nil {
		return fmt.Errorf("send Chat Completions request: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return readAPIError(response, token)
	}
	mediaType, _, err := mime.ParseMediaType(response.Header.Get("Content-Type"))
	if err != nil || mediaType != "text/event-stream" {
		return fmt.Errorf("read Chat Completions stream: unexpected Content-Type %q", response.Header.Get("Content-Type"))
	}
	if err := consumeStream(response.Body, emit, token); err != nil {
		return fmt.Errorf("read Chat Completions stream: %w", err)
	}
	return nil
}

func chatCompletionsEndpoint(baseURL string) (string, error) {
	parsed, err := url.ParseRequestURI(strings.TrimSpace(baseURL))
	if err != nil {
		return "", fmt.Errorf("parse Chat Completions base URL: %w", err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", fmt.Errorf("Chat Completions base URL scheme must be http or https")
	}
	if parsed.Host == "" {
		return "", fmt.Errorf("Chat Completions base URL host is empty")
	}
	if parsed.User != nil {
		return "", fmt.Errorf("Chat Completions base URL must not contain user information")
	}
	if parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", fmt.Errorf("Chat Completions base URL must not contain a query or fragment")
	}
	if parsed.Scheme == "http" {
		ip := net.ParseIP(parsed.Hostname())
		if ip == nil || !ip.IsLoopback() {
			return "", fmt.Errorf("plain HTTP Chat Completions base URL must use a loopback IP")
		}
	}
	return parsed.JoinPath("chat/completions").String(), nil
}

// readAPIError reads the error a provider answered. A transient status, such
// as 429 or 503, may be retried, unless the account has no quota left.
func readAPIError(response *http.Response, token string) error {
	noQuota, err := readAPIErrorBody(response, token)
	return &llm.ProviderError{Err: err, Status: response.StatusCode,
		Transient:  llm.TransientStatus(response.StatusCode) && !noQuota,
		RetryAfter: llm.RetryAfter(response.Header, time.Now())}
}

// readAPIErrorBody formats the error a provider answered, and tells whether
// it is the quota of the account that is spent, which a wait does not renew.
func readAPIErrorBody(response *http.Response, token string) (bool, error) {
	data, err := io.ReadAll(io.LimitReader(response.Body, maxErrorBodyBytes+1))
	if err != nil {
		return false, fmt.Errorf("Chat Completions API returned HTTP %d", response.StatusCode)
	}
	if len(data) > maxErrorBodyBytes {
		return false, fmt.Errorf("Chat Completions API returned HTTP %d with an oversized error body", response.StatusCode)
	}
	var body struct {
		Error struct {
			Code    any    `json:"code"`
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	noQuota := false
	if json.Unmarshal(data, &body) == nil {
		code := normalizeErrorText(fmt.Sprint(body.Error.Code), token)
		message := normalizeErrorText(body.Error.Message, token)
		noQuota = code == "insufficient_quota" || body.Error.Type == "insufficient_quota"
		if code != "" && code != "<nil>" && message != "" {
			return noQuota, fmt.Errorf("Chat Completions API returned HTTP %d (%s): %s", response.StatusCode, code, message)
		}
		if message != "" {
			return noQuota, fmt.Errorf("Chat Completions API returned HTTP %d: %s", response.StatusCode, message)
		}
	}
	return noQuota, fmt.Errorf("Chat Completions API returned HTTP %d", response.StatusCode)
}

func normalizeErrorText(value string, secret string) string {
	if secret != "" {
		value = strings.ReplaceAll(value, secret, "[redacted]")
	}
	value = strings.Join(strings.Fields(value), " ")
	const maximumLength = 512
	if len(value) > maximumLength {
		value = value[:maximumLength]
	}
	return value
}
