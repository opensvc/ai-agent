package llmfactory

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/opensvc/ai-agent/internal/config"
	"github.com/opensvc/ai-agent/internal/llm"
	"github.com/opensvc/ai-agent/internal/llm/chatcompletions"
	"github.com/opensvc/ai-agent/internal/llm/messages"
	"github.com/opensvc/ai-agent/internal/llm/responses"
)

func TestNewSelectsResponsesProtocol(t *testing.T) {
	client, err := New(config.LLMConfig{
		Protocol:        config.LLMProtocolResponses,
		BaseURL:         "https://llm.example.test/v1",
		Model:           "test-model",
		AuthMode:        config.LLMAuthModeNone,
		Timeout:         time.Minute,
		MaxOutputTokens: 1024,
	}, http.DefaultClient)
	if err != nil {
		t.Fatalf("create LLM client: %v", err)
	}
	if _, ok := client.(*responses.Client); !ok {
		t.Fatalf("got client type %T, want Responses client", client)
	}
}

func TestNewSelectsChatCompletionsProtocol(t *testing.T) {
	client, err := New(config.LLMConfig{
		Protocol:        config.LLMProtocolChatCompletions,
		BaseURL:         "https://llm.example.test/v1",
		Model:           "test-model",
		AuthMode:        config.LLMAuthModeNone,
		Timeout:         time.Minute,
		MaxOutputTokens: 1024,
	}, http.DefaultClient)
	if err != nil {
		t.Fatalf("create LLM client: %v", err)
	}
	if _, ok := client.(*chatcompletions.Client); !ok {
		t.Fatalf("got client type %T, want Chat Completions client", client)
	}
}

// The setting reaches the client: a request leaves store out only when
// OmitStore asks it.
func TestNewPassesOmitStoreToChatCompletions(t *testing.T) {
	for _, omit := range []bool{false, true} {
		t.Run(fmt.Sprintf("omit %v", omit), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
				var body map[string]any
				if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
					t.Errorf("decode request: %v", err)
					return
				}
				if _, sent := body["store"]; sent == omit {
					t.Errorf("omit %v: store sent %v", omit, sent)
				}
				response.Header().Set("Content-Type", "text/event-stream")
				_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
			}))
			t.Cleanup(server.Close)
			client, err := New(config.LLMConfig{
				Protocol: config.LLMProtocolChatCompletions, BaseURL: server.URL + "/v1", Model: "test-model",
				AuthMode: config.LLMAuthModeNone, Timeout: time.Minute, MaxOutputTokens: 1024, OmitStore: omit,
			}, server.Client())
			if err != nil {
				t.Fatalf("create LLM client: %v", err)
			}
			if err := client.Stream(t.Context(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Text: "health"}}}, func(llm.Event) error { return nil }); err != nil {
				t.Fatalf("stream: %v", err)
			}
		})
	}
}

func TestNewRejectsUnsupportedProtocol(t *testing.T) {
	_, err := New(config.LLMConfig{Protocol: "unknown"}, nil)
	if err == nil || !strings.Contains(err.Error(), "unsupported") {
		t.Fatalf("New() error = %v, want unsupported protocol", err)
	}
}

func TestNewSelectsMessagesProtocol(t *testing.T) {
	t.Setenv(config.LLMAPITokenEnv, "test-secret")
	for _, mode := range []string{config.LLMAuthModeAPIKey, config.LLMAuthModeBearer, config.LLMAuthModeNone} {
		client, err := New(config.LLMConfig{
			Protocol: config.LLMProtocolMessages, BaseURL: "https://api.anthropic.com/v1", Model: "test-model",
			AuthMode: mode, APITokenEnv: config.LLMAPITokenEnv, Timeout: time.Minute, MaxOutputTokens: 1024,
		}, http.DefaultClient)
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := client.(*messages.Client); !ok {
			t.Fatalf("got client type %T, want Messages client", client)
		}
	}
}

func TestEnvironmentTokenSource(t *testing.T) {
	const name = "OPENSVC_AI_TEST_TOKEN_SOURCE"
	t.Setenv(name, "placeholder")
	token, err := environmentTokenSource(name)()
	if err != nil {
		t.Fatalf("load environment token: %v", err)
	}
	if token != "placeholder" {
		t.Fatalf("got token %q", token)
	}

	t.Setenv(name, "")
	if _, err := environmentTokenSource(name)(); err == nil {
		t.Fatal("empty environment token accepted")
	}
}

func TestNewModelsCreatesTheClientOfEachModel(t *testing.T) {
	llmConfig := func(protocol string) config.LLMConfig {
		return config.LLMConfig{Protocol: protocol, BaseURL: "https://llm.example.test/v1", Model: protocol + "-model", AuthMode: config.LLMAuthModeNone, Timeout: time.Minute, MaxOutputTokens: 1024}
	}
	models, err := NewModels([]config.ModelConfig{
		{ID: llm.DefaultModelID, Label: "Default", LLM: llmConfig(config.LLMProtocolResponses)},
		{ID: "haiku", Label: "Claude Haiku", LLM: llmConfig(config.LLMProtocolMessages)},
	}, http.DefaultClient)
	if err != nil {
		t.Fatal(err)
	}
	haiku, err := models.Resolve(t.Context(), "haiku")
	if err != nil || haiku.Label != "Claude Haiku" || haiku.Name != "messages-model" {
		t.Fatalf("haiku = %+v, %v", haiku, err)
	}
	if _, ok := haiku.Client.(*messages.Client); !ok {
		t.Fatalf("haiku client = %T", haiku.Client)
	}
	if _, ok := models.Default().Client.(*responses.Client); !ok {
		t.Fatalf("default client = %T", models.Default().Client)
	}
	bad := llmConfig(config.LLMProtocolResponses)
	bad.BaseURL = "ftp://llm.example.test"
	if _, err := NewModels([]config.ModelConfig{{ID: llm.DefaultModelID, Label: "Default", LLM: llmConfig(config.LLMProtocolResponses)}, {ID: "bad", Label: "Bad", LLM: bad}}, nil); err == nil || !strings.Contains(err.Error(), `model "bad"`) {
		t.Fatalf("invalid model error = %v", err)
	}
}
