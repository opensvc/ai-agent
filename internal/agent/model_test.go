package agent

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/opensvc/ai-agent/internal/llm"
)

// unusedModel fails the test of a turn that runs on it.
type unusedModel struct{ t *testing.T }

func (m unusedModel) Stream(context.Context, llm.Request, llm.EmitFunc) error {
	m.t.Fatal("the turn ran on a model it did not choose")
	return nil
}

// usageStep answers the text, with a usage before.
func usageStep(text string) llmStep {
	return func(_ llm.Request, emit llm.EmitFunc) error {
		return emitEvents(emit,
			llm.Event{Type: llm.EventUsage, Usage: &llm.Usage{InputTokens: 3, OutputTokens: 2, TotalTokens: 5}},
			llm.Event{Type: llm.EventTextDelta, TextDelta: text},
			llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonCompleted})
	}
}

func newModelsAgent(t *testing.T, session *fakeSession, list ...llm.Model) *Agent {
	t.Helper()
	models, err := llm.NewModels(list)
	if err != nil {
		t.Fatal(err)
	}
	agent, err := New(models, func(context.Context) (MCPSession, error) { return session, nil },
		Config{SystemPrompt: testSystemPrompt, MaxIterations: 4, Timeout: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	return agent
}

func TestRunTurnRunsOnTheModelItsContextNames(t *testing.T) {
	haiku := &scriptedLLM{t: t, steps: []llmStep{usageStep("from haiku")}}
	agent := newModelsAgent(t, &fakeSession{},
		llm.Model{ID: llm.DefaultModelID, Label: "Default", Name: "gpt-luna", Client: unusedModel{t}},
		llm.Model{ID: "haiku", Label: "Claude Haiku", Name: "claude-haiku", Client: haiku},
	)
	var usage *Event
	result, err := agent.RunTurn(WithModel(t.Context(), "haiku"), nil, "check", func(event Event) error {
		if event.Type == EventUsage {
			usage = &event
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if haiku.position != 1 || result.Model != (TurnModel{ID: "haiku", Name: "claude-haiku"}) || result.Messages[len(result.Messages)-1].Text != "from haiku" {
		t.Fatalf("result = %+v", result)
	}
	if usage == nil || usage.ModelID != "haiku" {
		t.Fatalf("usage = %+v", usage)
	}
}

func TestRunTurnRunsOnTheDefaultWhenItsContextNamesNone(t *testing.T) {
	model := &scriptedLLM{t: t, steps: []llmStep{textStep("from the default")}}
	agent := newModelsAgent(t, &fakeSession{},
		llm.Model{ID: llm.DefaultModelID, Label: "Default", Name: "gpt-luna", Client: model},
		llm.Model{ID: "haiku", Label: "Claude Haiku", Name: "claude-haiku", Client: unusedModel{t}},
	)
	result, err := agent.RunTurn(t.Context(), nil, "check", func(Event) error { return nil })
	if err != nil || result.Model != (TurnModel{ID: llm.DefaultModelID, Name: "gpt-luna"}) {
		t.Fatalf("result = %+v, %v", result, err)
	}
}

func TestRunTurnRefusesAModelOfNoneOfTheAgent(t *testing.T) {
	session := &fakeSession{}
	agent := newModelsAgent(t, session, llm.Model{ID: llm.DefaultModelID, Label: "Default", Name: "gpt-luna", Client: unusedModel{t}})
	_, err := agent.RunTurn(WithModel(t.Context(), "haiku"), nil, "check", func(Event) error { return nil })
	if !errors.Is(err, llm.ErrModelUnknown) || session.closed {
		t.Fatalf("err = %v, MCP opened = %v", err, session.closed)
	}
}

// A suspended turn resumes on the model it started on, whatever the
// context of the resume names; on the default when the operator removed
// its model, or when it was suspended before turns recorded it.
func TestResumeTurnRunsOnTheModelOfTheTurn(t *testing.T) {
	suspend := func(t *testing.T) SuspendedTurn {
		t.Helper()
		session := &fakeSession{tools: []*mcp.Tool{destructiveTool("stop_object")}}
		haiku := &scriptedLLM{t: t, steps: []llmStep{callsStep(llm.ToolCall{ID: "call-1", Name: "stop_object", Arguments: json.RawMessage(`{"path":"web"}`)})}}
		agent := newModelsAgent(t, session,
			llm.Model{ID: llm.DefaultModelID, Label: "Default", Name: "gpt-luna", Client: unusedModel{t}},
			llm.Model{ID: "haiku", Label: "Claude Haiku", Name: "claude-haiku", Client: haiku},
		)
		first, err := agent.RunTurn(WithModel(t.Context(), "haiku"), nil, "stop web", func(Event) error { return nil })
		if err != nil || first.Suspended == nil || first.Suspended.ModelID != "haiku" {
			t.Fatalf("got %+v %v", first, err)
		}
		encoded, err := json.Marshal(first.Suspended)
		if err != nil {
			t.Fatal(err)
		}
		var stored SuspendedTurn
		if err := json.Unmarshal(encoded, &stored); err != nil || stored.ModelID != "haiku" {
			t.Fatalf("stored = %+v, %v", stored, err)
		}
		return stored
	}
	resume := func(t *testing.T, stored SuspendedTurn, want TurnModel, list ...llm.Model) {
		t.Helper()
		session := &fakeSession{tools: []*mcp.Tool{destructiveTool("stop_object")}}
		agent := newModelsAgent(t, session, list...)
		result, err := agent.ResumeTurn(WithModel(t.Context(), "mini"), nil, stored, DecisionReject, func(Event) error { return nil })
		if err != nil || result.Model != want {
			t.Fatalf("result = %+v, %v", result, err)
		}
	}
	t.Run("its model", func(t *testing.T) {
		resume(t, suspend(t), TurnModel{ID: "haiku", Name: "claude-haiku"},
			llm.Model{ID: llm.DefaultModelID, Label: "Default", Name: "gpt-luna", Client: unusedModel{t}},
			llm.Model{ID: "haiku", Label: "Claude Haiku", Name: "claude-haiku", Client: &scriptedLLM{t: t, steps: []llmStep{answerStep(t, nil)}}},
			llm.Model{ID: "mini", Label: "Mini", Name: "gpt-luna-mini", Client: unusedModel{t}},
		)
	})
	t.Run("its model removed", func(t *testing.T) {
		resume(t, suspend(t), TurnModel{ID: llm.DefaultModelID, Name: "gpt-luna"},
			llm.Model{ID: llm.DefaultModelID, Label: "Default", Name: "gpt-luna", Client: &scriptedLLM{t: t, steps: []llmStep{answerStep(t, nil)}}},
			llm.Model{ID: "mini", Label: "Mini", Name: "gpt-luna-mini", Client: unusedModel{t}},
		)
	})
	t.Run("suspended before models", func(t *testing.T) {
		stored := suspend(t)
		stored.ModelID = ""
		resume(t, stored, TurnModel{ID: llm.DefaultModelID, Name: "gpt-luna"},
			llm.Model{ID: llm.DefaultModelID, Label: "Default", Name: "gpt-luna", Client: &scriptedLLM{t: t, steps: []llmStep{answerStep(t, nil)}}},
			llm.Model{ID: "haiku", Label: "Claude Haiku", Name: "claude-haiku", Client: unusedModel{t}},
		)
	})
}

// A run that fails tells its model too: the run the turn records names it.
func TestRunTurnTellsItsModelWhenItFails(t *testing.T) {
	models, err := llm.NewModels([]llm.Model{
		{ID: llm.DefaultModelID, Label: "Default", Name: "gpt-luna", Client: unusedModel{t}},
		{ID: "haiku", Label: "Claude Haiku", Name: "claude-haiku", Client: unusedModel{t}},
	})
	if err != nil {
		t.Fatal(err)
	}
	agent, err := New(models, func(context.Context) (MCPSession, error) { return nil, errors.New("MCP unreachable") },
		Config{SystemPrompt: testSystemPrompt, MaxIterations: 4, Timeout: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	result, err := agent.RunTurn(WithModel(t.Context(), "haiku"), nil, "check", func(Event) error { return nil })
	if err == nil || result.Model != (TurnModel{ID: "haiku", Name: "claude-haiku"}) {
		t.Fatalf("result = %+v, %v", result, err)
	}
}
