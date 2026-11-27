package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/llm"
)

type skillListerFunc func(context.Context) ([]agent.SkillInfo, error)

func (f skillListerFunc) Skills(ctx context.Context) ([]agent.SkillInfo, error) { return f(ctx) }

func newSkillsTestHandler(t *testing.T, skills SkillLister) http.Handler {
	t.Helper()
	handler, err := NewHandler([]Backend{{Name: "om3", Conversations: turnService{}, Authenticator: allowTestTokenVerifier(), Skills: skills}}, HandlerConfig{Models: testModels(), MaxConcurrentAsks: 4, AuditLogger: discardAuditLogger()})
	if err != nil {
		t.Fatal(err)
	}
	return handler
}

func TestSkillsServeWhatTheUserMayAsk(t *testing.T) {
	var token string
	handler := newSkillsTestHandler(t, skillListerFunc(func(ctx context.Context) ([]agent.SkillInfo, error) {
		token, _ = auth.BearerTokenFromContext(ctx)
		return []agent.SkillInfo{{Name: "morning-check", Description: "Check the cluster in the morning."}, {Name: "node-maintenance", Description: "Reboot a node."}}, nil
	}))
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, authenticatedRequest(http.MethodGet, "/v1/om3/skills", ""))
	if recorder.Code != http.StatusOK || recorder.Header().Get("Cache-Control") != "no-store" || token != "token" {
		t.Fatalf("status %d, cache %q, token %q: %s", recorder.Code, recorder.Header().Get("Cache-Control"), token, recorder.Body)
	}
	if want := `{"skills":[{"name":"morning-check","description":"Check the cluster in the morning."},{"name":"node-maintenance","description":"Reboot a node."}]}`; strings.TrimSpace(recorder.Body.String()) != want {
		t.Fatalf("skills = %s", recorder.Body)
	}
	empty := newSkillsTestHandler(t, skillListerFunc(func(context.Context) ([]agent.SkillInfo, error) { return []agent.SkillInfo{}, nil }))
	recorder = httptest.NewRecorder()
	empty.ServeHTTP(recorder, authenticatedRequest(http.MethodGet, "/v1/om3/skills", ""))
	if recorder.Code != http.StatusOK || strings.TrimSpace(recorder.Body.String()) != `{"skills":[]}` {
		t.Fatalf("no skills: %d %s", recorder.Code, recorder.Body)
	}
	unauthenticated := httptest.NewRecorder()
	handler.ServeHTTP(unauthenticated, httptest.NewRequest(http.MethodGet, "/v1/om3/skills", nil))
	if unauthenticated.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated status=%d", unauthenticated.Code)
	}
}

func TestSkillsAnswer502WhenTheMCPDoesNot(t *testing.T) {
	handler := newSkillsTestHandler(t, skillListerFunc(func(context.Context) ([]agent.SkillInfo, error) {
		return nil, errors.New("connection refused")
	}))
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, authenticatedRequest(http.MethodGet, "/v1/om3/skills", ""))
	if recorder.Code != http.StatusBadGateway || !strings.Contains(recorder.Body.String(), `"code":"skills_unavailable"`) || strings.Contains(recorder.Body.String(), "refused") {
		t.Fatalf("status %d: %s", recorder.Code, recorder.Body)
	}
	none := newSkillsTestHandler(t, nil)
	recorder = httptest.NewRecorder()
	none.ServeHTTP(recorder, authenticatedRequest(http.MethodGet, "/v1/om3/skills", ""))
	if recorder.Code != http.StatusNotFound {
		t.Fatalf("no lister status %d", recorder.Code)
	}
}

func TestCapabilitiesTellTheSkillsAreAsked(t *testing.T) {
	response := httptest.NewRecorder()
	newTestHandler(t, nil).ServeHTTP(response, authenticatedRequest(http.MethodGet, "/v1/om3/capabilities", ""))
	var capabilities struct {
		Custom map[string]any `json:"custom"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &capabilities); err != nil || capabilities.Custom["skills"] != true {
		t.Fatalf("capabilities = %s, %v", response.Body.String(), err)
	}
}

// runAskingSkill runs a prompt of the conversation with the forwarded
// properties props, its turn failing with fail, and gives the skill the turn
// was asked, whether it ran, the response and the audit.
func runAskingSkill(t *testing.T, props string, fail error) (string, bool, *httptest.ResponseRecorder, string) {
	t.Helper()
	skill, ran, accessed := "", false, false
	service := abandonRecorder{turnService: turnService{run: func(ctx context.Context, _ string, emit agent.EmitFunc) error {
		skill, ran = agent.SkillFrom(ctx), true
		if fail != nil {
			return fail
		}
		return emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, Iteration: 1})
	}}, accessed: &accessed}
	var audit bytes.Buffer
	handler, err := NewHandler([]Backend{{Name: "om3", Conversations: service, Authenticator: allowTestTokenVerifier()}},
		HandlerConfig{Models: testModels(), MaxConcurrentAsks: 1, AuditLogger: slog.New(slog.NewJSONHandler(&audit, nil))})
	if err != nil {
		t.Fatal(err)
	}
	body := strings.TrimSuffix(runBody("/morning-check"), "}") + `,"forwardedProps":` + props + "}"
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, newAuthenticatedRunRequest(body))
	return skill, ran, response, audit.String()
}

func TestRunAsksItsSkill(t *testing.T) {
	skill, ran, response, audit := runAskingSkill(t, `{"skill":"morning-check","model":"haiku"}`, nil)
	if response.Code != http.StatusOK || !ran || skill != "morning-check" || !strings.Contains(audit, `"skill":"morning-check"`) {
		t.Fatalf("status=%d ran=%v skill=%q audit=%s", response.Code, ran, skill, audit)
	}
	for _, props := range []string{`{}`, `{"skill":null}`} {
		if skill, _, response, audit := runAskingSkill(t, props, nil); response.Code != http.StatusOK || skill != "" || strings.Contains(audit, `"skill"`) {
			t.Fatalf("props %s: status=%d skill=%q", props, response.Code, skill)
		}
	}
}

func TestRunRefusesASkillNotShapedAsOne(t *testing.T) {
	for _, props := range []string{`{"skill":""}`, `{"skill":7}`, `{"skill":"Morning Check"}`, `{"skill":"/etc/hosts"}`, `{"skill":"` + strings.Repeat("a", 65) + `"}`} {
		_, ran, response, audit := runAskingSkill(t, props, nil)
		// Refused before the turn, and before the stream; the audit keeps
		// the code, never the value the client sent.
		if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), `"code":"invalid_skill"`) || ran ||
			strings.HasPrefix(response.Header().Get("Content-Type"), "text/event-stream") || strings.Contains(audit, "Morning") || strings.Contains(audit, "/etc/hosts") {
			t.Fatalf("props %s: status=%d ran=%v body=%s audit=%s", props, response.Code, ran, response.Body.String(), audit)
		}
	}
}

func TestRunTellsASkillUnavailable(t *testing.T) {
	_, _, response, audit := runAskingSkill(t, `{"skill":"morning-check"}`, fmt.Errorf("%w: The skill morning-check is not served to you now.", agent.ErrSkillUnavailable))
	if !strings.Contains(response.Body.String(), `"code":"skill_unavailable"`) || strings.Contains(response.Body.String(), "served to you now.") ||
		!strings.Contains(audit, `"code":"skill_unavailable"`) {
		t.Fatalf("body=%s audit=%s", response.Body.String(), audit)
	}
}

func TestUnattendedRunAsksItsSkill(t *testing.T) {
	skill := "(no turn)"
	runner := runnerFunc(func(ctx context.Context, _ []llm.Message, _ string, emit agent.EmitFunc) (agent.TurnResult, error) {
		skill = agent.SkillFrom(ctx)
		return agent.TurnResult{FinishReason: llm.FinishReasonCompleted}, emit(agent.Event{Type: agent.EventCompleted, FinishReason: llm.FinishReasonCompleted, Iteration: 1})
	})
	handler := newUnattendedHandler(t, runner, 1)
	body := strings.TrimSuffix(unattendedBody, "}")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, unattendedRequest(body+`,"forwardedProps":{"skill":"morning-check"}}`))
	if response.Code != http.StatusOK || skill != "morning-check" {
		t.Fatalf("status=%d skill=%q", response.Code, skill)
	}
	skill = "(no turn)"
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, unattendedRequest(body+`,"forwardedProps":{"skill":"Morning"}}`))
	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), `"code":"invalid_skill"`) || skill != "(no turn)" {
		t.Fatalf("status=%d skill=%q body=%s", response.Code, skill, response.Body.String())
	}
}
