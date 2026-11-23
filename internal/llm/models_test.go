package llm

import (
	"context"
	"errors"
	"strings"
	"testing"
)

type nopClient struct{}

func (nopClient) Stream(context.Context, Request, EmitFunc) error { return nil }

func TestModelsResolveTheirIDs(t *testing.T) {
	models, err := NewModels([]Model{
		{ID: DefaultModelID, Label: "GPT Luna", Name: "gpt-luna", Client: nopClient{}},
		{ID: "haiku", Label: "Claude Haiku", Name: "claude-haiku", Client: nopClient{}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if model, err := models.Resolve(t.Context(), ""); err != nil || model.ID != DefaultModelID {
		t.Fatalf("empty ID = %+v, %v", model, err)
	}
	if model, err := models.Resolve(t.Context(), "haiku"); err != nil || model.Name != "claude-haiku" {
		t.Fatalf("haiku = %+v, %v", model, err)
	}
	if _, err := models.Resolve(t.Context(), "HAIKU"); !errors.Is(err, ErrModelUnknown) {
		t.Fatalf("unknown = %v", err)
	}
	list := models.List(t.Context())
	list[0].ID = "changed"
	if models.Default().ID != DefaultModelID || len(models.List(t.Context())) != 2 {
		t.Fatal("the list given is the list of the models")
	}
	if single := SingleModel(nopClient{}, "gpt-luna"); single.Default().ID != DefaultModelID || single.Default().Label != "gpt-luna" {
		t.Fatalf("single = %+v", single.Default())
	}
}

func TestNewModelsRefusesInvalidModels(t *testing.T) {
	valid := Model{ID: DefaultModelID, Label: "GPT Luna", Name: "gpt-luna", Client: nopClient{}}
	with := func(change func(*Model)) []Model {
		other := Model{ID: "haiku", Label: "Claude Haiku", Name: "claude-haiku", Client: nopClient{}}
		change(&other)
		return []Model{valid, other}
	}
	many := []Model{valid}
	for i := range MaxModels {
		many = append(many, Model{ID: "m" + strings.Repeat("x", i), Label: "m", Name: "m", Client: nopClient{}})
	}
	for name, list := range map[string][]Model{
		"none":                 nil,
		"too many":             many,
		"no default first":     {with(func(*Model) {})[1], valid},
		"a repeated ID":        with(func(m *Model) { m.ID = DefaultModelID }),
		"an empty ID":          with(func(m *Model) { m.ID = "" }),
		"a long ID":            with(func(m *Model) { m.ID = strings.Repeat("a", MaxModelIDBytes+1) }),
		"no label":             with(func(m *Model) { m.Label = "" }),
		"a long label":         with(func(m *Model) { m.Label = strings.Repeat("é", MaxModelLabelRunes+1) }),
		"a control in a label": with(func(m *Model) { m.Label = "a\nb" }),
		"a spaced label":       with(func(m *Model) { m.Label = " Haiku" }),
		"no name":              with(func(m *Model) { m.Name = "" }),
		"no client":            with(func(m *Model) { m.Client = nil }),
	} {
		if _, err := NewModels(list); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if err := ValidateModelLabel(strings.Repeat("é", MaxModelLabelRunes)); err != nil {
		t.Fatalf("a label of %d characters: %v", MaxModelLabelRunes, err)
	}
}
