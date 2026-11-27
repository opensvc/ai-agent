package conversation

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/llm"
)

// editableRunner suspends a turn on a call the user may edit, and records
// the arguments it is resumed with.
type editableRunner struct {
	confirmationRunner
	edited json.RawMessage
}

func (r *editableRunner) RunTurn(context.Context, []llm.Message, string, agent.EmitFunc) (agent.TurnResult, error) {
	state := suspendedStopTurn()
	state.InputSchema = json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}`)
	return agent.TurnResult{Suspended: &state}, nil
}

func (r *editableRunner) ResumeTurn(ctx context.Context, history []llm.Message, state agent.SuspendedTurn, decision agent.Decision, emit agent.EmitFunc) (agent.TurnResult, error) {
	r.edited = state.EditedArguments
	return r.confirmationRunner.ResumeTurn(ctx, history, state, decision, emit)
}

func suspendEditable(t *testing.T) (*Service, *serviceTestStore, *editableRunner, *agent.Confirmation) {
	t.Helper()
	store := newServiceTestStore()
	runner := &editableRunner{}
	service := newTestService(t, store, runner)
	execution, err := service.PrepareTurn(t.Context(), serviceTestIdentity(), store.item.ID, "stop web")
	if err != nil {
		t.Fatal(err)
	}
	var request *agent.Confirmation
	if err := execution.Run(t.Context(), func(event agent.Event) error { request = event.Confirmation; return nil }); err != nil {
		t.Fatal(err)
	}
	if request == nil || len(request.InputSchema) == 0 {
		t.Fatalf("the confirmation takes no edits: %+v", request)
	}
	return service, store, runner, request
}

func TestServiceRefusesAnInvalidEditAndKeepsTheConfirmationWaiting(t *testing.T) {
	service, store, _, request := suspendEditable(t)
	identity := serviceTestIdentity()
	for _, edited := range []string{`{"path":3}`, `{"name":"web"}`, `[]`} {
		if _, err := service.PrepareConfirmation(t.Context(), identity, store.item.ID, request.TurnID, request.ID, agent.DecisionApprove, json.RawMessage(edited)); !errors.Is(err, agent.ErrInvalidEditedArguments) {
			t.Fatalf("%s: %v", edited, err)
		}
		if store.suspended == nil {
			t.Fatalf("%s: an invalid edit used the confirmation", edited)
		}
	}
	// An edit goes with an approval only.
	if _, err := service.PrepareConfirmation(t.Context(), identity, store.item.ID, request.TurnID, request.ID, agent.DecisionReject, json.RawMessage(`{"path":"db"}`)); !errors.Is(err, ErrInvalid) || store.suspended == nil {
		t.Fatalf("an edit with a rejection: %v", err)
	}
}

func TestServiceResumesWithTheEditAndTheCallShown(t *testing.T) {
	service, store, runner, request := suspendEditable(t)
	execution, err := service.PrepareConfirmation(t.Context(), serviceTestIdentity(), store.item.ID, request.TurnID, request.ID, agent.DecisionApprove, json.RawMessage(`{"path":"db"}`))
	if err != nil {
		t.Fatal(err)
	}
	if err := execution.Run(t.Context(), func(agent.Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	// The decision binds to the call shown, whose arguments are the model's.
	if string(runner.edited) != `{"path":"db"}` || string(runner.resumedCall.Arguments) != `{"path":"web"}` || runner.resumedWith != agent.DecisionApprove {
		t.Fatalf("resumed with %s, call %s, %q", runner.edited, runner.resumedCall.Arguments, runner.resumedWith)
	}
}

func TestServiceRefusesAnEditOfACallWithoutSchema(t *testing.T) {
	store := newServiceTestStore()
	service := newTestService(t, store, &confirmationRunner{})
	execution, err := service.PrepareTurn(t.Context(), serviceTestIdentity(), store.item.ID, "stop web")
	if err != nil {
		t.Fatal(err)
	}
	if err := execution.Run(t.Context(), func(agent.Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if _, err := service.PrepareConfirmation(t.Context(), serviceTestIdentity(), store.item.ID, store.suspended.TurnID, store.suspended.ConfirmationID, agent.DecisionApprove, json.RawMessage(`{"path":"db"}`)); !errors.Is(err, agent.ErrInvalidEditedArguments) || store.suspended == nil {
		t.Fatalf("a call without schema took an edit: %v", err)
	}
}

// skillRunner suspends a turn whose prompt asked a skill, the agent loading it.
type skillRunner struct{ editableRunner }

func (r *skillRunner) RunTurn(ctx context.Context, history []llm.Message, prompt string, emit agent.EmitFunc) (agent.TurnResult, error) {
	result, err := r.editableRunner.RunTurn(ctx, history, prompt, emit)
	result.Exposure.Skills = []agent.LoadedSkill{{Name: "node-maintenance", URI: "skill://node-maintenance/SKILL.md", Digest: "sha256:01", Asked: true}}
	return result, err
}

// A turn resumed after its confirmation knows the skill its prompt asked, for
// the snapshot to name it; a new turn knows none, its request names it.
func TestServiceResumesKnowingTheSkillAsked(t *testing.T) {
	store := newServiceTestStore()
	service := newTestService(t, store, &skillRunner{})
	execution, err := service.PrepareTurn(t.Context(), serviceTestIdentity(), store.item.ID, "stop web")
	if err != nil {
		t.Fatal(err)
	}
	if execution.AskedSkill() != "" {
		t.Fatalf("a new turn knows a skill: %q", execution.AskedSkill())
	}
	var request *agent.Confirmation
	if err := execution.Run(t.Context(), func(event agent.Event) error { request = event.Confirmation; return nil }); err != nil {
		t.Fatal(err)
	}
	resumed, err := service.PrepareConfirmation(t.Context(), serviceTestIdentity(), store.item.ID, request.TurnID, request.ID, agent.DecisionReject, nil)
	if err != nil {
		t.Fatal(err)
	}
	if resumed.AskedSkill() != "node-maintenance" {
		t.Fatalf("asked = %q", resumed.AskedSkill())
	}
}
