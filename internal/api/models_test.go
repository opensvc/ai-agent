package api

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/llm"
)

// unusedLLM is the client of the models of the tests of the API, whose
// turns run on fakes of the conversations, never on a model.
type unusedLLM struct{}

func (unusedLLM) Stream(context.Context, llm.Request, llm.EmitFunc) error {
	panic("the API tests run no model")
}

// testModels are the models of the tests of the API: the default and haiku.
func testModels() *llm.Models {
	models, err := llm.NewModels([]llm.Model{
		{ID: llm.DefaultModelID, Label: "GPT Luna", Name: "gpt-luna", Client: unusedLLM{}},
		{ID: "haiku", Label: "Claude Haiku", Name: "claude-haiku", Client: unusedLLM{}},
	})
	if err != nil {
		panic(err)
	}
	return models
}

// abandonRecorder records the first access of a run to its conversation.
type abandonRecorder struct {
	turnService
	accessed *bool
}

func (s abandonRecorder) AbandonClientCalls(context.Context, auth.Identity, string) error {
	*s.accessed = true
	return nil
}

func TestModelsListsTheModelsByIDAndLabel(t *testing.T) {
	handler := newTestHandler(t, nil)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, authenticatedRequest(http.MethodGet, "/v1/om3/models", ""))
	if response.Code != http.StatusOK || response.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("status=%d headers=%v body=%s", response.Code, response.Header(), response.Body.String())
	}
	// The ID and the label only: nothing of the provider leaves the agent.
	var answer struct {
		Models []map[string]any `json:"models"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &answer); err != nil {
		t.Fatal(err)
	}
	want := []map[string]any{
		{"id": "default", "label": "GPT Luna", "default": true, "ready": true},
		{"id": "haiku", "label": "Claude Haiku", "default": false, "ready": true},
	}
	if !reflect.DeepEqual(answer.Models, want) || strings.Contains(response.Body.String(), "claude-haiku") {
		t.Fatalf("models = %s", response.Body.String())
	}

	unauthenticated := httptest.NewRequest(http.MethodGet, "/v1/om3/models", nil)
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, unauthenticated)
	if response.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated status=%d", response.Code)
	}
}

func TestCapabilitiesTellTheModelsAreChosen(t *testing.T) {
	response := httptest.NewRecorder()
	newTestHandler(t, nil).ServeHTTP(response, authenticatedRequest(http.MethodGet, "/v1/om3/capabilities", ""))
	var capabilities struct {
		Custom map[string]any `json:"custom"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &capabilities); err != nil || capabilities.Custom["models"] != true {
		t.Fatalf("capabilities = %s, %v", response.Body.String(), err)
	}
}

// runWithProps runs a prompt of the conversation with the forwarded
// properties props, and gives the model the turn ran on, whether the
// conversation was accessed, the response and the audit.
func runWithProps(t *testing.T, props string) (string, bool, *httptest.ResponseRecorder, string) {
	t.Helper()
	modelID, ran, accessed := "", false, false
	service := abandonRecorder{turnService: turnService{run: func(ctx context.Context, _ string, emit agent.EmitFunc) error {
		modelID, ran = agent.ModelFrom(ctx), true
		return emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, Iteration: 1})
	}}, accessed: &accessed}
	var audit bytes.Buffer
	handler, err := NewHandler([]Backend{{Name: "om3", Conversations: service, Authenticator: allowTestTokenVerifier()}},
		HandlerConfig{Models: testModels(), MaxConcurrentAsks: 1, AuditLogger: slog.New(slog.NewJSONHandler(&audit, nil))})
	if err != nil {
		t.Fatal(err)
	}
	body := strings.TrimSuffix(runBody("check it"), "}")
	if props != "" {
		body += `,"forwardedProps":` + props
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, newAuthenticatedRunRequest(body+"}"))
	if !ran {
		modelID = "(no turn)"
	}
	return modelID, accessed, response, audit.String()
}

func TestRunChoosesItsModel(t *testing.T) {
	for _, test := range []struct {
		props string
		want  string
	}{
		{props: "", want: "default"},
		{props: `null`, want: "default"},
		{props: `"haiku"`, want: "default"},
		{props: `[]`, want: "default"},
		{props: `{}`, want: "default"},
		{props: `{"model":null,"other":1}`, want: "default"},
		{props: `{"model":"default"}`, want: "default"},
		{props: `{"model":"haiku","other":{"model":"x"}}`, want: "haiku"},
	} {
		modelID, _, response, audit := runWithProps(t, test.props)
		if response.Code != http.StatusOK || modelID != test.want || !strings.Contains(audit, `"model_id":"`+test.want+`"`) {
			t.Fatalf("props %s: status=%d model=%q audit=%s", test.props, response.Code, modelID, audit)
		}
	}
}

func TestRunRefusesAModelOfNoneOfTheAgent(t *testing.T) {
	for _, props := range []string{`{"model":"gpt-4"}`, `{"model":""}`, `{"model":7}`, `{"model":["haiku"]}`, `{"model":"` + strings.Repeat("a", 65) + `"}`} {
		modelID, accessed, response, audit := runWithProps(t, props)
		// Refused before the conversation is read, and before the stream.
		if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), `"code":"model_unknown"`) || accessed || modelID != "(no turn)" ||
			response.Header().Get("Content-Type") == "text/event-stream" {
			t.Fatalf("props %s: status=%d accessed=%v model=%q body=%s", props, response.Code, accessed, modelID, response.Body.String())
		}
		// The audit keeps the code, never the value the client sent.
		if !strings.Contains(audit, `"code":"model_unknown"`) || strings.Contains(audit, "gpt-4") {
			t.Fatalf("props %s: audit = %s", props, audit)
		}
	}
}

func TestUnattendedRunChoosesItsModel(t *testing.T) {
	var modelID string
	runner := runnerFunc(func(ctx context.Context, _ []llm.Message, _ string, emit agent.EmitFunc) (agent.TurnResult, error) {
		modelID = agent.ModelFrom(ctx)
		return agent.TurnResult{FinishReason: llm.FinishReasonCompleted}, emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, Iteration: 1})
	})
	handler := newUnattendedHandler(t, runner, 1)
	body := strings.TrimSuffix(unattendedBody, "}")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, unattendedRequest(body+`,"forwardedProps":{"model":"haiku"}}`))
	if response.Code != http.StatusOK || modelID != "haiku" {
		t.Fatalf("status=%d model=%q", response.Code, modelID)
	}
	modelID = "(no turn)"
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, unattendedRequest(body+`,"forwardedProps":{"model":"gpt-4"}}`))
	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), `"code":"model_unknown"`) || modelID != "(no turn)" {
		t.Fatalf("status=%d model=%q body=%s", response.Code, modelID, response.Body.String())
	}
}
