package agent

import (
	"context"
	"errors"
	"fmt"

	"github.com/opensvc/ai-agent/internal/llm"
)

// A turn runs on the model its client chose among those of the agent, the
// default when it chose none, from its start to its end: a turn waiting on
// the user resumes on the model it started on, unless the operator removed
// it meanwhile, when it resumes on the default.

type modelKey struct{}

// WithModel runs the new turns of ctx on the model of id; an empty id is the
// default model.
func WithModel(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, modelKey{}, id)
}

// ModelFrom is the ID of the model the new turns of ctx run on; empty for
// the default.
func ModelFrom(ctx context.Context) string {
	id, _ := ctx.Value(modelKey{}).(string)
	return id
}

// newTurnModel is the model a new turn runs on, the one its context names.
func (a *Agent) newTurnModel(ctx context.Context) (llm.Model, error) {
	model, err := a.models.Resolve(ctx, ModelFrom(ctx))
	if err != nil {
		return llm.Model{}, fmt.Errorf("agent model: %w", err)
	}
	return model, nil
}

// resumedTurnModel is the model a suspended turn resumes on: the one it
// started on, or the default when the operator removed it, or for a turn
// suspended before turns recorded their model.
func (a *Agent) resumedTurnModel(ctx context.Context, id string) (llm.Model, error) {
	model, err := a.models.Resolve(ctx, id)
	if errors.Is(err, llm.ErrModelUnknown) {
		return a.models.Default(), nil
	}
	if err != nil {
		return llm.Model{}, fmt.Errorf("agent model: %w", err)
	}
	return model, nil
}
