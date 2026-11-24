package agent

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/opensvc/ai-agent/internal/llm"
)

// searchStep asks for a search of the node tools.
func searchStep() llmStep {
	return callsStep(llm.ToolCall{ID: "search-1", Name: SearchToolName, Arguments: json.RawMessage(`{"query":"node hardware packages"}`)})
}

// A search shows its tools in its order, the best first, right after
// search_tools.
func TestASearchKeepsTheOrderOfItsResults(t *testing.T) {
	var tools []string
	var shown []string
	model := &scriptedLLM{t: t, steps: []llmStep{searchStep(), func(request llm.Request, emit llm.EmitFunc) error {
		tools, shown = toolNames(request.Tools), searchShown(lastToolResults(request)[0])
		return textStep("ok")(request, emit)
	}}}
	if _, err := searchAgent(t, model, &fakeSession{tools: searchCatalog()}, Config{}).RunTurn(t.Context(), nil, "node hardware?", func(Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if len(shown) < 2 || !slices.Equal(tools[1:1+len(shown)], shown) {
		t.Fatalf("tools %v, the search showed %v", tools, shown)
	}
}

// The next turn starts with the tools of the last request of the turn
// before, for a provider to read its prefix from the cache.
func TestTheNextTurnStartsWithTheToolsOfTheLastRequest(t *testing.T) {
	var last, next []string
	model := &scriptedLLM{t: t, steps: []llmStep{searchStep(), func(request llm.Request, emit llm.EmitFunc) error {
		last = toolNames(request.Tools)
		return textStep("listed")(request, emit)
	}, func(request llm.Request, emit llm.EmitFunc) error {
		next = toolNames(request.Tools)
		return textStep("again")(request, emit)
	}}}
	agent := searchAgent(t, model, &fakeSession{tools: searchCatalog()}, Config{})
	first, err := agent.RunTurn(t.Context(), nil, "node hardware?", func(Event) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.RunTurn(t.Context(), first.Messages, "and the packages?", func(Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if len(last) < 3 || !slices.Equal(next, last) {
		t.Fatalf("next turn tools %v, last request of the turn before %v", next, last)
	}
}

// A resumed turn sends the tools of the request it waited after, in the
// same order.
func TestAResumedTurnSendsTheToolsItWaitedWith(t *testing.T) {
	var waited, resumed []string
	model := &scriptedLLM{t: t, steps: []llmStep{searchStep(), func(request llm.Request, emit llm.EmitFunc) error {
		waited = toolNames(request.Tools)
		return callsStep(llm.ToolCall{ID: "stop-1", Name: "stop_object", Arguments: json.RawMessage(`{"path":"web"}`)})(request, emit)
	}, func(request llm.Request, emit llm.EmitFunc) error {
		resumed = toolNames(request.Tools)
		return textStep("stopped")(request, emit)
	}}}
	agent := searchAgent(t, model, &fakeSession{tools: searchCatalog()}, Config{})
	first, err := agent.RunTurn(t.Context(), nil, "stop web", func(Event) error { return nil })
	if err != nil || first.Suspended == nil {
		t.Fatalf("turn = %+v, %v", first.Suspended, err)
	}
	if _, err := agent.ResumeTurn(t.Context(), nil, *first.Suspended, DecisionApprove, func(Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if len(waited) < 3 || !slices.Equal(resumed, waited) {
		t.Fatalf("resumed with %v, waited with %v", resumed, waited)
	}
}

// The tools of a search go before those of the former searches, in its
// order, even when making room removes some of them: of the current search,
// or of a former one.
func TestShowKeepsTheOrderOfEachSearchWhenMakingRoom(t *testing.T) {
	selection := &toolSelection{search: true, catalog: map[string]llm.Tool{}, core: map[string]bool{"core": true}}
	// The budget holds the core tool and three others.
	for _, name := range []string{"core", "a", "b", "c", "d", "e", "f"} {
		selection.catalog[name] = llm.Tool{Name: name, Description: strings.Repeat("x", maxModelToolCatalogBytes/5)}
	}
	selection.add("core")
	selection.startBatch()
	for _, name := range []string{"a", "b"} {
		selection.show(name)
	}
	if !slices.Equal(selection.found, []string{"a", "b"}) {
		t.Fatalf("first search found %v", selection.found)
	}
	// The second search removes a and b, of the former search, then c, the
	// first of its own.
	selection.startBatch()
	for _, name := range []string{"c", "d", "e", "f"} {
		selection.show(name)
	}
	if !slices.Equal(selection.found, []string{"d", "e", "f"}) || selection.batch != 3 {
		t.Fatalf("found %v, batch %d", selection.found, selection.batch)
	}
}

// A turn suspended by a former version, without the order of the tools
// found, resumes with them after the core tools, as it did.
func TestAResumedTurnWithoutTheOrderOfItsToolsKeepsThem(t *testing.T) {
	var resumed []string
	model := &scriptedLLM{t: t, steps: []llmStep{searchStep(), callsStep(llm.ToolCall{ID: "stop-1", Name: "stop_object", Arguments: json.RawMessage(`{"path":"web"}`)}),
		func(request llm.Request, emit llm.EmitFunc) error {
			resumed = toolNames(request.Tools)
			return textStep("stopped")(request, emit)
		}}}
	agent := searchAgent(t, model, &fakeSession{tools: searchCatalog()}, Config{})
	first, err := agent.RunTurn(t.Context(), nil, "stop web", func(Event) error { return nil })
	if err != nil || first.Suspended == nil {
		t.Fatalf("turn = %+v, %v", first.Suspended, err)
	}
	stored := *first.Suspended
	stored.FoundTools = nil
	if _, err := agent.ResumeTurn(t.Context(), nil, stored, DecisionApprove, func(Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	for _, name := range stored.ActiveTools {
		if !slices.Contains(resumed, name) {
			t.Fatalf("resumed with %v, without %s", resumed, name)
		}
	}
}

// The common flow, a search, the call of a tool it found, the answer, then
// a question that follows: the next turn starts with the tools of the last
// request of the turn before.
func TestTheNextTurnAfterACallOfAToolFoundStartsWithTheSameTools(t *testing.T) {
	var last, next []string
	model := &scriptedLLM{t: t, steps: []llmStep{searchStep(),
		callsStep(llm.ToolCall{ID: "hw-1", Name: "list_node_hardware", Arguments: json.RawMessage(`{}`)}),
		func(request llm.Request, emit llm.EmitFunc) error {
			last = toolNames(request.Tools)
			return textStep("listed")(request, emit)
		}, func(request llm.Request, emit llm.EmitFunc) error {
			next = toolNames(request.Tools)
			return textStep("again")(request, emit)
		}}}
	agent := searchAgent(t, model, &fakeSession{tools: searchCatalog()}, Config{})
	first, err := agent.RunTurn(t.Context(), nil, "node hardware?", func(Event) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.RunTurn(t.Context(), first.Messages, "and the packages?", func(Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if len(last) < 3 || !slices.Equal(next, last) {
		t.Fatalf("next turn tools %v, last request of the turn before %v", next, last)
	}
}
