package llm

import (
	"context"
	"errors"
	"fmt"
	"unicode"
	"unicode/utf8"
)

const (
	// DefaultModelID is the ID of the default model: the one a turn runs on
	// when its client names none.
	DefaultModelID = "default"
	// MaxModels bounds the models the agent offers.
	MaxModels = 16
	// MaxModelIDBytes bounds the ID of a model, as a client names it.
	MaxModelIDBytes = 64
	// MaxModelLabelRunes bounds the label of a model.
	MaxModelLabelRunes = 64
)

// ErrModelUnknown is the error of a model ID the agent does not offer.
var ErrModelUnknown = errors.New("model is unknown")

// Model is a model the agent offers: its stable ID, the label the
// interfaces show, the name of the model at its provider and its client.
type Model struct {
	ID     string
	Label  string
	Name   string
	Client Client
}

// Models are the models the agent offers, the default first. Its methods
// take the context of the request: the models a caller may use come from
// the operator today, and may later depend on the caller.
type Models struct {
	list []Model
	byID map[string]int
}

// NewModels checks the models of the agent: at least one, at most
// MaxModels, the first the default, each with its own ID, a label, a name
// and a client.
func NewModels(list []Model) (*Models, error) {
	if len(list) == 0 || len(list) > MaxModels {
		return nil, fmt.Errorf("models: expected 1 to %d models, got %d", MaxModels, len(list))
	}
	if list[0].ID != DefaultModelID {
		return nil, fmt.Errorf("models: the first model must be %q, got %q", DefaultModelID, list[0].ID)
	}
	models := &Models{list: make([]Model, len(list)), byID: make(map[string]int, len(list))}
	for i, model := range list {
		if model.ID == "" || len(model.ID) > MaxModelIDBytes {
			return nil, fmt.Errorf("models: model ID %q must have 1 to %d bytes", model.ID, MaxModelIDBytes)
		}
		if _, ok := models.byID[model.ID]; ok {
			return nil, fmt.Errorf("models: model ID %q is repeated", model.ID)
		}
		if err := ValidateModelLabel(model.Label); err != nil {
			return nil, fmt.Errorf("models: model %q: %w", model.ID, err)
		}
		if model.Name == "" {
			return nil, fmt.Errorf("models: model %q has no name", model.ID)
		}
		if model.Client == nil {
			return nil, fmt.Errorf("models: model %q has no client", model.ID)
		}
		models.list[i] = model
		models.byID[model.ID] = i
	}
	return models, nil
}

// ValidateModelLabel checks the label of a model: 1 to MaxModelLabelRunes
// printable characters, without leading or trailing space.
func ValidateModelLabel(label string) error {
	if label == "" || !utf8.ValidString(label) || utf8.RuneCountInString(label) > MaxModelLabelRunes {
		return fmt.Errorf("label must have 1 to %d characters", MaxModelLabelRunes)
	}
	for _, char := range label {
		if !unicode.IsPrint(char) {
			return fmt.Errorf("label must be printable")
		}
	}
	if first, _ := utf8.DecodeRuneInString(label); unicode.IsSpace(first) {
		return fmt.Errorf("label must not start with a space")
	}
	if last, _ := utf8.DecodeLastRuneInString(label); unicode.IsSpace(last) {
		return fmt.Errorf("label must not end with a space")
	}
	return nil
}

// Default is the default model.
func (m *Models) Default() Model {
	return m.list[0]
}

// List gives the models a caller may use, the default first.
func (m *Models) List(context.Context) []Model {
	return append([]Model(nil), m.list...)
}

// Resolve gives the model of an ID for a caller: the default for an empty
// ID, ErrModelUnknown for one the agent does not offer.
func (m *Models) Resolve(_ context.Context, id string) (Model, error) {
	if id == "" {
		return m.Default(), nil
	}
	index, ok := m.byID[id]
	if !ok {
		return Model{}, ErrModelUnknown
	}
	return m.list[index], nil
}

// SingleModel gives the models of an agent offering one model, the default,
// on client, without the checks of NewModels: for a name of confidence, such
// as one of a test.
func SingleModel(client Client, name string) *Models {
	return &Models{list: []Model{{ID: DefaultModelID, Label: name, Name: name, Client: client}}, byID: map[string]int{DefaultModelID: 0}}
}
