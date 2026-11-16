package messages

import (
	"bytes"
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
	AuthModeNone      = "none"
	AuthModeBearer    = "bearer"
	AuthModeAPIKey    = "api_key"
	apiVersion        = "2023-06-01"
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
}

// Client implements llm.Client using the Anthropic Messages HTTP protocol.
type Client struct {
	endpoint        string
	model           string
	authMode        string
	tokenSource     TokenSource
	maxOutputTokens int
	httpClient      *http.Client
}

var _ llm.Client = (*Client)(nil)

func New(config Config, httpClient *http.Client) (*Client, error) {
	endpoint, err := messagesEndpoint(config.BaseURL)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(config.Model) == "" {
		return nil, fmt.Errorf("Messages model is empty")
	}
	if config.Timeout <= 0 {
		return nil, fmt.Errorf("Messages timeout must be positive")
	}
	if config.MaxOutputTokens <= 0 {
		return nil, fmt.Errorf("Messages max output tokens must be positive")
	}
	switch config.AuthMode {
	case AuthModeNone:
	case AuthModeAPIKey, AuthModeBearer:
		if config.TokenSource == nil {
			return nil, fmt.Errorf("Messages API token source is missing")
		}
	default:
		return nil, fmt.Errorf("unsupported Messages authentication mode %q", config.AuthMode)
	}
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	clientCopy := *httpClient
	clientCopy.Timeout = config.Timeout
	clientCopy.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	return &Client{endpoint: endpoint, model: config.Model, authMode: config.AuthMode,
		tokenSource: config.TokenSource, maxOutputTokens: config.MaxOutputTokens, httpClient: &clientCopy}, nil
}

func (c *Client) Stream(ctx context.Context, request llm.Request, emit llm.EmitFunc) error {
	if err := request.Validate(); err != nil {
		return fmt.Errorf("validate LLM request: %w", err)
	}
	if emit == nil {
		return fmt.Errorf("stream Messages: event consumer is nil")
	}
	wireRequest, err := newCreateRequest(c.model, c.maxOutputTokens, request)
	if err != nil {
		return err
	}
	body, err := json.Marshal(wireRequest)
	if err != nil {
		return fmt.Errorf("encode Messages request: %w", err)
	}
	if len(body) > maxRequestBytes {
		return fmt.Errorf("Messages request exceeds %d bytes", maxRequestBytes)
	}
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("create Messages request: %w", err)
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("Accept", "text/event-stream")
	httpRequest.Header.Set("anthropic-version", apiVersion)
	var token string
	if c.authMode != AuthModeNone {
		token, err = c.tokenSource()
		if err != nil {
			return fmt.Errorf("load Messages API token: %w", err)
		}
		if token == "" {
			return fmt.Errorf("load Messages API token: token is empty")
		}
		for _, char := range token {
			if char < 33 || char > 126 {
				return fmt.Errorf("Messages API token contains invalid header characters")
			}
		}
		if c.authMode == AuthModeAPIKey {
			httpRequest.Header.Set("x-api-key", token)
		} else {
			httpRequest.Header.Set("Authorization", "Bearer "+token)
		}
	}
	response, err := c.httpClient.Do(httpRequest)
	if err != nil {
		// net/http can include a malformed redirect Location in its error,
		// even when following redirects is disabled. Do not echo a key.
		if token != "" && strings.Contains(err.Error(), token) {
			return fmt.Errorf("send Messages request failed (credential redacted)")
		}
		return fmt.Errorf("send Messages request: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return readAPIError(response)
	}
	mediaType, _, err := mime.ParseMediaType(response.Header.Get("Content-Type"))
	if err != nil || mediaType != "text/event-stream" {
		return fmt.Errorf("read Messages stream: expected text/event-stream")
	}
	if err := consumeStream(response.Body, emit); err != nil {
		return fmt.Errorf("read Messages stream: %w", err)
	}
	return nil
}

func messagesEndpoint(baseURL string) (string, error) {
	parsed, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil {
		return "", fmt.Errorf("Messages base URL is invalid")
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", fmt.Errorf("Messages base URL scheme must be http or https")
	}
	if parsed.Host == "" {
		return "", fmt.Errorf("Messages base URL host is empty")
	}
	if parsed.User != nil || parsed.RawQuery != "" || parsed.ForceQuery || parsed.Fragment != "" {
		return "", fmt.Errorf("Messages base URL must not contain user information, a query or fragment")
	}
	if parsed.Scheme == "http" {
		ip := net.ParseIP(parsed.Hostname())
		if ip == nil || !ip.IsLoopback() {
			return "", fmt.Errorf("plain HTTP Messages base URL must use a loopback IP")
		}
	}
	return parsed.JoinPath("messages").String(), nil
}

// Provider error bodies can echo credentials or conversation content. Only
// known error types and the HTTP status are retained, never their messages.
// A transient status, such as 429 or 529, may be retried.
func readAPIError(response *http.Response) error {
	return &llm.ProviderError{Err: readAPIErrorBody(response), Status: response.StatusCode,
		Transient: llm.TransientStatus(response.StatusCode), RetryAfter: llm.RetryAfter(response.Header, time.Now())}
}

func readAPIErrorBody(response *http.Response) error {
	data, err := io.ReadAll(io.LimitReader(response.Body, maxErrorBodyBytes+1))
	if err != nil {
		return fmt.Errorf("Messages API returned HTTP %d", response.StatusCode)
	}
	if len(data) > maxErrorBodyBytes {
		return fmt.Errorf("Messages API returned HTTP %d with an oversized error body", response.StatusCode)
	}
	var body struct {
		Error struct {
			Type string `json:"type"`
		} `json:"error"`
	}
	if json.Unmarshal(data, &body) == nil {
		return fmt.Errorf("Messages API returned HTTP %d (%s)", response.StatusCode, safeErrorType(body.Error.Type))
	}
	return fmt.Errorf("Messages API returned HTTP %d", response.StatusCode)
}

func safeErrorType(value string) string {
	switch value {
	case "invalid_request_error", "authentication_error", "permission_error", "not_found_error", "request_too_large", "rate_limit_error", "api_error", "overloaded_error":
		return value
	default:
		return "provider_error"
	}
}
