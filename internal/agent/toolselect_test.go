package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/opensvc/ai-agent/internal/llm"
	"github.com/opensvc/ai-agent/internal/toolsearch"
)

// searchCatalog is larger than the threshold of searchAgent: the model sees
// the core tools and search_tools only.
func searchCatalog() []*mcp.Tool {
	destructive := true
	tool := func(name, description string, meta mcp.Meta) *mcp.Tool {
		return &mcp.Tool{Name: name, Description: description, InputSchema: objectSchema(), Meta: meta,
			Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true}}
	}
	return []*mcp.Tool{
		tool("get_cluster_status", "The status of the cluster.", mcp.Meta{CoreToolMeta: true}),
		tool("list_node_hardware", "The PCI devices and memory banks of a node: its hardware inventory.", nil),
		tool("get_node_san_topology", "The fibre channel paths of a node.", nil),
		tool("list_node_packages", "The packages installed on a node.", nil),
		{Name: "stop_object", Description: "Stop an object.", InputSchema: objectSchema(), Annotations: &mcp.ToolAnnotations{DestructiveHint: &destructive}},
	}
}

func searchAgent(t *testing.T, model llm.Client, session *fakeSession, config Config) *Agent {
	t.Helper()
	config.SystemPrompt, config.MaxIterations, config.Timeout = testSystemPrompt, 6, time.Minute
	if config.SearchThreshold == 0 {
		config.SearchThreshold = 3
	}
	agent, err := New(llm.SingleModel(model, "test-model"), func(context.Context) (MCPSession, error) { return session, nil }, config)
	if err != nil {
		t.Fatal(err)
	}
	return agent
}

func toolNames(tools []llm.Tool) []string {
	names := make([]string, 0, len(tools))
	for _, tool := range tools {
		names = append(names, tool.Name)
	}
	return names
}

func searchCallStep(id, name, arguments string) llmStep {
	return func(_ llm.Request, emit llm.EmitFunc) error {
		return emitEvents(emit,
			llm.Event{Type: llm.EventToolCall, ToolCall: &llm.ToolCall{ID: id, Name: name, Arguments: json.RawMessage(arguments)}},
			llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonToolCalls})
	}
}

func textStep(text string) llmStep {
	return func(_ llm.Request, emit llm.EmitFunc) error {
		return emitEvents(emit, llm.Event{Type: llm.EventTextDelta, TextDelta: text}, llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonCompleted})
	}
}

// The tools a search found come right after search_tools, before the core
// tools: listed last, the model searched again for them.
func TestSearchShowsTheToolsFoundBeforeTheCore(t *testing.T) {
	session := &fakeSession{tools: searchCatalog()}
	var requests []llm.Request
	record := func(step llmStep) llmStep {
		return func(request llm.Request, emit llm.EmitFunc) error {
			requests = append(requests, request)
			return step(request, emit)
		}
	}
	model := &scriptedLLM{t: t, steps: []llmStep{
		record(searchCallStep("search-1", SearchToolName, `{"query":"node hardware inventory","limit":1}`)),
		record(searchCallStep("call-1", "list_node_hardware", `{}`)),
		record(textStep("done")),
	}}
	var events []Event
	result, err := searchAgent(t, model, session, Config{}).RunTurn(t.Context(), nil, "hardware of node1", func(event Event) error {
		events = append(events, event)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := toolNames(requests[0].Tools); !slices.Equal(got, []string{SearchToolName, "get_cluster_status"}) {
		t.Fatalf("first tools = %v", got)
	}
	if got := toolNames(requests[1].Tools); !slices.Equal(got, []string{SearchToolName, "list_node_hardware", "get_cluster_status"}) {
		t.Fatalf("tools after the search = %v", got)
	}
	if !strings.Contains(requests[0].Messages[0].Text, searchInstruction) {
		t.Fatal("the system prompt does not explain the search")
	}
	if len(session.calls) != 1 || session.calls[0].name != "list_node_hardware" {
		t.Fatalf("MCP calls = %v: the search must not reach the MCP", session.calls)
	}
	searchResult := string(result.Messages[2].ToolResults[0].Content)
	if !strings.Contains(searchResult, `list_node_hardware`) || !strings.Contains(searchResult, `hardware inventory`) || result.Messages[2].ToolResults[0].IsError {
		t.Fatalf("search result = %s", searchResult)
	}
	if events[0].Type != EventToolStarted || events[0].ToolName != SearchToolName || events[1].Type != EventToolFinished {
		t.Fatalf("events = %+v", events)
	}
}

// Without a core tool marked by the MCP, the model sees search_tools and the
// tools of the recent history only.
func TestSearchKeepsRecentToolsWithoutCoreTools(t *testing.T) {
	catalog := searchCatalog()
	catalog[0].Meta = nil
	session := &fakeSession{tools: catalog}
	history := []llm.Message{
		{Role: llm.RoleUser, Text: "packages?"},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "old", Name: "list_node_packages", Arguments: json.RawMessage(`{}`)}}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "old", Content: json.RawMessage(`{}`)}}},
		{Role: llm.RoleAssistant, Text: "listed"},
	}
	var tools []string
	model := &scriptedLLM{t: t, steps: []llmStep{func(request llm.Request, emit llm.EmitFunc) error {
		tools = toolNames(request.Tools)
		return textStep("ok")(request, emit)
	}}}
	if _, err := searchAgent(t, model, session, Config{}).RunTurn(t.Context(), history, "and now?", func(Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(tools, []string{SearchToolName, "list_node_packages"}) {
		t.Fatalf("tools = %v", tools)
	}
}

func TestSmallCatalogsAreShownWhole(t *testing.T) {
	for _, config := range []Config{{SearchThreshold: 10}, {SearchThreshold: -1}} {
		session := &fakeSession{tools: searchCatalog()}
		var tools []string
		var system string
		model := &scriptedLLM{t: t, steps: []llmStep{func(request llm.Request, emit llm.EmitFunc) error {
			tools, system = toolNames(request.Tools), request.Messages[0].Text
			return textStep("ok")(request, emit)
		}}}
		if _, err := searchAgent(t, model, session, config).RunTurn(t.Context(), nil, "hi", func(Event) error { return nil }); err != nil {
			t.Fatal(err)
		}
		if len(tools) != 5 || slices.Contains(tools, SearchToolName) || strings.Contains(system, searchInstruction) {
			t.Fatalf("threshold %d: tools = %v", config.SearchThreshold, tools)
		}
	}
}

func TestSearchRefusalsAreToolErrors(t *testing.T) {
	session := &fakeSession{tools: searchCatalog()}
	var results []llm.ToolResult
	steps := []llmStep{}
	for i, arguments := range []string{`{"query":"the of"}`, `{"query":"node","limit":11}`, `{"query":"node","other":1}`, `{"query":"kubernetes pods"}`, `{"query":"node"}`} {
		steps = append(steps, searchCallStep(string(rune('a'+i)), SearchToolName, arguments))
	}
	steps = append(steps, textStep("done"))
	model := &scriptedLLM{t: t, steps: steps}
	agent := searchAgent(t, model, session, Config{})
	agent.config.MaxIterations = 8
	result, err := agent.RunTurn(t.Context(), nil, "search", func(Event) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	for _, message := range result.Messages {
		results = append(results, message.ToolResults...)
	}
	// Invalid queries, limits and arguments are refused without counting;
	// a search without match is no error and counts.
	wantErrors := []bool{true, true, true, false, false}
	for i, want := range wantErrors {
		if results[i].IsError != want {
			t.Fatalf("result %d = %s, error %v", i, results[i].Content, results[i].IsError)
		}
	}
	if !strings.Contains(string(results[3].Content), "No tool matches") {
		t.Fatalf("no match result = %s", results[3].Content)
	}
}

// Past the searches of a turn, search_tools is no longer offered, and the
// last search says so: the model answers instead of searching in a loop. A
// search the model still calls is refused as a tool error.
func TestSearchIsLimitedPerTurn(t *testing.T) {
	session := &fakeSession{tools: searchCatalog()}
	var offered [][]string
	steps := []llmStep{}
	for i := range maxSearchesPerTurn + 1 {
		step := searchCallStep(string(rune('a'+i)), SearchToolName, `{"query":"node"}`)
		steps = append(steps, func(request llm.Request, emit llm.EmitFunc) error {
			offered = append(offered, toolNames(request.Tools))
			return step(request, emit)
		})
	}
	steps = append(steps, textStep("done"))
	agent := searchAgent(t, &scriptedLLM{t: t, steps: steps}, session, Config{})
	agent.config.MaxIterations = 8
	result, err := agent.RunTurn(t.Context(), nil, "search", func(Event) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	for i, tools := range offered {
		if slices.Contains(tools, SearchToolName) != (i < maxSearchesPerTurn) {
			t.Fatalf("request %d offered %v", i+1, tools)
		}
	}
	if final := result.Messages[2*maxSearchesPerTurn].ToolResults[0]; final.IsError || !strings.Contains(string(final.Content), "last search") {
		t.Fatalf("last search = %s", final.Content)
	}
	last := result.Messages[2*(maxSearchesPerTurn+1)].ToolResults[0]
	if !last.IsError || !strings.Contains(string(last.Content), "limited") {
		t.Fatalf("search past the limit = %s", last.Content)
	}
}

func TestACatalogToolOutsideTheShownOnesStillRuns(t *testing.T) {
	session := &fakeSession{tools: searchCatalog()}
	model := &scriptedLLM{t: t, steps: []llmStep{searchCallStep("call-1", "list_node_packages", `{}`), textStep("done")}}
	if _, err := searchAgent(t, model, session, Config{}).RunTurn(t.Context(), nil, "packages", func(Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if len(session.calls) != 1 || session.calls[0].name != "list_node_packages" {
		t.Fatalf("calls = %v", session.calls)
	}
	// A tool outside the catalog is answered with a tool error, not run.
	session = &fakeSession{tools: searchCatalog()}
	var answered llm.ToolResult
	model = &scriptedLLM{t: t, steps: []llmStep{searchCallStep("call-1", "no_such_tool", `{}`), func(request llm.Request, emit llm.EmitFunc) error {
		answered = request.Messages[len(request.Messages)-1].ToolResults[0]
		return textStep("done")(request, emit)
	}}}
	if _, err := searchAgent(t, model, session, Config{}).RunTurn(t.Context(), nil, "x", func(Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if len(session.calls) != 0 || !answered.IsError || !strings.Contains(string(answered.Content), "no tool of this name is offered") {
		t.Fatalf("calls = %v, answer = %+v", session.calls, answered)
	}
}

func TestAnMCPToolNamedLikeTheSearchIsRefused(t *testing.T) {
	catalog := append(searchCatalog(), &mcp.Tool{Name: SearchToolName, InputSchema: objectSchema()})
	model := &scriptedLLM{t: t}
	_, err := searchAgent(t, model, &fakeSession{tools: catalog}, Config{}).RunTurn(t.Context(), nil, "x", func(Event) error { return nil })
	if err == nil || !strings.Contains(err.Error(), "conflicts") {
		t.Fatalf("error = %v", err)
	}
}

func TestAResumedTurnSeesTheToolsItFound(t *testing.T) {
	session := &fakeSession{tools: searchCatalog()}
	model := &scriptedLLM{t: t, steps: []llmStep{
		searchCallStep("search-1", SearchToolName, `{"query":"stop object","limit":1}`),
		searchCallStep("call-1", "stop_object", `{}`),
	}}
	agent := searchAgent(t, model, session, Config{})
	suspended, err := agent.RunTurn(t.Context(), nil, "stop it", func(Event) error { return nil })
	if err != nil || suspended.Suspended == nil {
		t.Fatalf("turn = %+v, %v", suspended, err)
	}
	state := *suspended.Suspended
	if !slices.Contains(state.ActiveTools, "stop_object") || state.Searches != 1 {
		t.Fatalf("suspended state = %+v", state)
	}
	encoded, _ := json.Marshal(state)
	var decoded SuspendedTurn
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	var tools []string
	model.steps = append(model.steps, func(request llm.Request, emit llm.EmitFunc) error {
		tools = toolNames(request.Tools)
		return textStep("stopped")(request, emit)
	})
	if _, err := agent.ResumeTurn(t.Context(), nil, decoded, DecisionApprove, func(Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(tools, "stop_object") {
		t.Fatalf("resumed tools = %v", tools)
	}
	decoded.Searches = maxSearchesPerTurn + 1
	if _, err := agent.ResumeTurn(t.Context(), nil, decoded, DecisionApprove, func(Event) error { return nil }); err == nil || errors.Is(err, context.Canceled) {
		t.Fatalf("an out of range search count was resumed: %v", err)
	}
}

func TestSummary(t *testing.T) {
	if got := summary("First sentence. Second one."); got != "First sentence." {
		t.Fatalf("summary = %q", got)
	}
	if got := summary(strings.Repeat("é", 200)); len(got) > maxSearchSummaryBytes+len("…") || !strings.HasSuffix(got, "…") {
		t.Fatalf("long summary = %q", got)
	}
}

// A catalog of few tools too large for one model request is searched, and
// the tools shown stay within the size of a request.
func TestALargeCatalogIsSearchedWithinTheRequestSize(t *testing.T) {
	catalog := modelCatalogTools()
	for _, tool := range catalog {
		tool.Annotations = &mcp.ToolAnnotations{ReadOnlyHint: true}
	}
	if _, _, err := convertTools(catalog); err != nil {
		t.Fatalf("the discovered catalog was refused: %v", err)
	}
	var sizes []int
	measure := func(request llm.Request) {
		total := 0
		for _, tool := range request.Tools {
			total += toolBytes(tool)
		}
		sizes = append(sizes, total)
	}
	steps := []llmStep{}
	for _, name := range []string{"a", "b", "c", "d", "e"} {
		steps = append(steps, func(request llm.Request, emit llm.EmitFunc) error {
			measure(request)
			return searchCallStep("search-"+name, SearchToolName, `{"query":"`+name+` s","limit":10}`)(request, emit)
		})
	}
	steps = append(steps[:maxSearchesPerTurn], func(request llm.Request, emit llm.EmitFunc) error {
		measure(request)
		return textStep("done")(request, emit)
	})
	agent := searchAgent(t, &scriptedLLM{t: t, steps: steps}, &fakeSession{tools: catalog}, Config{SearchThreshold: 100})
	if _, err := agent.RunTurn(t.Context(), nil, "x", func(Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	for _, size := range sizes {
		if size > maxModelToolCatalogBytes {
			t.Fatalf("a request held %d bytes of tools", size)
		}
	}
}

// A search refused for its query or limit does not count against the
// searches of the turn.
func TestInvalidSearchesDoNotCount(t *testing.T) {
	session := &fakeSession{tools: searchCatalog()}
	steps := []llmStep{}
	for i := range maxSearchesPerTurn {
		steps = append(steps, searchCallStep(string(rune('a'+i)), SearchToolName, `{"query":"the of and"}`))
	}
	steps = append(steps, searchCallStep("valid", SearchToolName, `{"query":"hardware"}`), textStep("done"))
	agent := searchAgent(t, &scriptedLLM{t: t, steps: steps}, session, Config{})
	agent.config.MaxIterations = 8
	result, err := agent.RunTurn(t.Context(), nil, "x", func(Event) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	valid := result.Messages[2*(maxSearchesPerTurn+1)-1+1].ToolResults[0]
	if valid.IsError || !strings.Contains(string(valid.Content), "list_node_hardware") {
		t.Fatalf("the valid search after invalid ones = %s", valid.Content)
	}
}

// The arguments of every call, search_tools included, are bounded as a
// turn stores them.
func TestSearchArgumentsAreBounded(t *testing.T) {
	huge := `{"query":"` + strings.Repeat("a", maxToolArguments) + `"}`
	model := &scriptedLLM{t: t, steps: []llmStep{searchCallStep("big", SearchToolName, huge)}}
	_, err := searchAgent(t, model, &fakeSession{tools: searchCatalog()}, Config{}).RunTurn(t.Context(), nil, "x", func(Event) error { return nil })
	if err == nil || !strings.Contains(err.Error(), "arguments exceed") {
		t.Fatalf("error = %v", err)
	}
}

// A turn suspended while searching keeps searching when it resumes, even if
// the search would no longer be on: its planned search call runs.
func TestAResumedTurnKeepsSearching(t *testing.T) {
	session := &fakeSession{tools: searchCatalog()}
	model := &scriptedLLM{t: t, steps: []llmStep{func(_ llm.Request, emit llm.EmitFunc) error {
		return emitEvents(emit,
			llm.Event{Type: llm.EventToolCall, ToolCall: &llm.ToolCall{ID: "stop", Name: "stop_object", Arguments: json.RawMessage(`{}`)}},
			llm.Event{Type: llm.EventToolCall, ToolCall: &llm.ToolCall{ID: "search", Name: SearchToolName, Arguments: json.RawMessage(`{"query":"hardware"}`)}},
			llm.Event{Type: llm.EventCompleted, FinishReason: llm.FinishReasonToolCalls})
	}}}
	agent := searchAgent(t, model, session, Config{})
	suspended, err := agent.RunTurn(t.Context(), nil, "stop then search", func(Event) error { return nil })
	if err != nil || suspended.Suspended == nil || !suspended.Suspended.ToolSearch {
		t.Fatalf("turn = %+v, %v", suspended.Suspended, err)
	}
	if !suspended.Exposure.ToolSearch || !slices.Contains(suspended.Exposure.ShownTools, SearchToolName) || len(suspended.Exposure.SystemPromptSHA256) != 64 {
		t.Fatalf("exposure = %+v", suspended.Exposure)
	}
	// The search is now off: the threshold exceeds the catalog.
	agent.config.SearchThreshold = 100
	var searchResult string
	model.steps = append(model.steps, func(request llm.Request, emit llm.EmitFunc) error {
		searchResult = string(lastToolResults(request)[1].Content)
		return textStep("done")(request, emit)
	})
	resumed, err := agent.ResumeTurn(t.Context(), nil, *suspended.Suspended, DecisionApprove, func(Event) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(searchResult, "list_node_hardware") || !resumed.Exposure.ToolSearch {
		t.Fatalf("resumed search = %s, exposure %+v", searchResult, resumed.Exposure)
	}
}

// The hash of the system prompt is the one of the prompt sent, with the
// search instruction when the catalog is searched.
func TestExposureHashesThePromptSent(t *testing.T) {
	run := func(config Config) Exposure {
		model := &scriptedLLM{t: t, steps: []llmStep{textStep("ok")}}
		result, err := searchAgent(t, model, &fakeSession{tools: searchCatalog()}, config).RunTurn(t.Context(), nil, "x", func(Event) error { return nil })
		if err != nil {
			t.Fatal(err)
		}
		return result.Exposure
	}
	searched, whole := run(Config{}), run(Config{SearchThreshold: 100})
	if !searched.ToolSearch || whole.ToolSearch || searched.SystemPromptSHA256 == whole.SystemPromptSHA256 {
		t.Fatalf("searched = %+v, whole = %+v", searched, whole)
	}
	if len(whole.ShownTools) != len(searchCatalog()) || !slices.Equal(searched.ShownTools, []string{SearchToolName, "get_cluster_status"}) {
		t.Fatalf("shown tools: searched %v, whole %v", searched.ShownTools, whole.ShownTools)
	}
}

// When the tools shown leave no room, a tool found replaces the oldest tools
// that are not core; one that cannot fit is reported, not hidden.
func TestASearchMakesRoomForTheToolsFound(t *testing.T) {
	big := func(name string) *mcp.Tool {
		return &mcp.Tool{Name: name, Description: name + " tool.", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true},
			InputSchema: map[string]any{"type": "object", "description": strings.Repeat("s", 220<<10)}}
	}
	catalog := []*mcp.Tool{big("first"), big("second"), big("third"), big("fourth"), big("fifth")}
	catalog[0].Meta = mcp.Meta{CoreToolMeta: true}
	history := []llm.Message{
		{Role: llm.RoleUser, Text: "before"},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{
			{ID: "h2", Name: "second", Arguments: json.RawMessage(`{}`)}, {ID: "h3", Name: "third", Arguments: json.RawMessage(`{}`)},
			{ID: "h4", Name: "fourth", Arguments: json.RawMessage(`{}`)}}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "h2", Content: json.RawMessage(`{}`)}, {CallID: "h3", Content: json.RawMessage(`{}`)}, {CallID: "h4", Content: json.RawMessage(`{}`)}}},
		{Role: llm.RoleAssistant, Text: "done"},
	}
	var tools []string
	model := &scriptedLLM{t: t, steps: []llmStep{
		searchCallStep("search", SearchToolName, `{"query":"fifth","limit":1}`),
		func(request llm.Request, emit llm.EmitFunc) error {
			tools = toolNames(request.Tools)
			return textStep("ok")(request, emit)
		},
	}}
	result, err := searchAgent(t, model, &fakeSession{tools: catalog}, Config{}).RunTurn(t.Context(), history, "fifth", func(Event) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	found := string(result.Messages[2].ToolResults[0].Content)
	if !strings.Contains(found, `\"shown\":true`) || strings.Contains(found, "No tool matches") || !slices.Contains(tools, "fifth") || !slices.Contains(tools, "first") {
		t.Fatalf("found = %s, tools = %v", found, tools)
	}

	// The core tools alone leave no room: the tool found is reported as not
	// shown, with a hint, never as no match.
	for _, tool := range catalog[:4] {
		tool.Meta = mcp.Meta{CoreToolMeta: true}
	}
	model = &scriptedLLM{t: t, steps: []llmStep{searchCallStep("search", SearchToolName, `{"query":"fifth","limit":1}`), textStep("ok")}}
	result, err = searchAgent(t, model, &fakeSession{tools: catalog}, Config{}).RunTurn(t.Context(), nil, "fifth", func(Event) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	found = string(result.Messages[2].ToolResults[0].Content)
	if !strings.Contains(found, `\"name\":\"fifth\"`) || !strings.Contains(found, `\"shown\":false`) || !strings.Contains(found, "cannot be shown") ||
		!strings.Contains(found, `\"reason\":\"budget\"`) || strings.Contains(found, "low_score") {
		t.Fatalf("found without room = %s", found)
	}
}

// The model is told to write a few keywords, and the schema states the
// bounds of the index.
func TestSearchToolAsksForKeywords(t *testing.T) {
	var schema struct {
		Properties struct {
			Query struct {
				MinLength   int    `json:"minLength"`
				MaxLength   int    `json:"maxLength"`
				Description string `json:"description"`
			} `json:"query"`
			Limit struct {
				Maximum int `json:"maximum"`
			} `json:"limit"`
		} `json:"properties"`
		Required []string `json:"required"`
	}
	if err := json.Unmarshal(searchTool.InputSchema, &schema); err != nil {
		t.Fatalf("schema is not JSON: %v: %s", err, searchTool.InputSchema)
	}
	query := schema.Properties.Query
	if query.MinLength != 1 || query.MaxLength != 512 || schema.Properties.Limit.Maximum != maxSearchResults || !slices.Equal(schema.Required, []string{"query"}) {
		t.Fatalf("schema = %s", searchTool.InputSchema)
	}
	guidance := "Use 3–8 English keywords describing the required capability. Omit greetings, full sentences, and specific resource identifiers."
	for name, text := range map[string]string{"query": query.Description, "tool": searchTool.Description, "prompt": searchInstruction} {
		if !strings.Contains(text, guidance) {
			t.Errorf("%s description lacks the guidance: %q", name, text)
		}
	}
	batch := "After a search, request the independent read-only tools you need together in the next step, when all their arguments are already known. Wait for prerequisite results before making dependent calls."
	if !strings.Contains(searchInstruction, batch) {
		t.Errorf("the search instruction does not ask to batch the tools found: %q", searchInstruction)
	}
	if !strings.Contains(query.Description, "At most 512 UTF-8 bytes.") {
		t.Errorf("query description lacks the byte bound: %q", query.Description)
	}
	// A query of 512 characters of two bytes each fits maxLength, not the
	// bound of the index: the description gives the exact one.
	if _, err := mustIndex(t).Search(strings.Repeat("é", 512), 1); err == nil {
		t.Error("a query of 1024 UTF-8 bytes was searched")
	}
	if err := (llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Text: "x"}}, Tools: []llm.Tool{searchTool}}).Validate(); err != nil {
		t.Fatalf("search tool is not a valid model tool: %v", err)
	}
	refused := (&toolSelection{index: mustIndex(t)}).run(llm.ToolCall{ID: "c", Name: SearchToolName, Arguments: json.RawMessage(`{"query":"the of and"}`)}, nil)
	if !refused.IsError || !strings.Contains(string(refused.Content), "3–8 English keywords") {
		t.Fatalf("refusal = %s", refused.Content)
	}
}

func mustIndex(t *testing.T) *toolsearch.Index {
	t.Helper()
	index, err := toolsearch.New([]toolsearch.Document{{Name: "list_node_hardware", Description: "hardware"}})
	if err != nil {
		t.Fatal(err)
	}
	return index
}

// filterCatalog gives clearly ordered scores for the query "node hardware
// inventory": the hardware tool first, the disks tool far behind.
func filterCatalog() []*mcp.Tool {
	tool := func(name, description string) *mcp.Tool {
		return &mcp.Tool{Name: name, Description: description, InputSchema: objectSchema(), Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true}}
	}
	core := tool("get_cluster_status", "The status of the cluster.")
	core.Meta = mcp.Meta{CoreToolMeta: true}
	return []*mcp.Tool{
		core,
		tool("list_node_hardware", "The hardware inventory of a node: node hardware, PCI devices and memory banks of the node hardware inventory."),
		tool("list_node_disks", "The disks a node sees, with their size."),
		tool("list_node_packages", "The packages installed."),
		tool("get_node_san_topology", "The fibre channel paths."),
	}
}

func newTestSelection(t *testing.T, tools []*mcp.Tool, ratio float64) (*toolSelection, map[string]string) {
	t.Helper()
	converted, _, err := convertTools(tools)
	if err != nil {
		t.Fatal(err)
	}
	agent := &Agent{config: Config{SearchThreshold: 1, SearchScoreRatio: ratio}, indexes: &indexCache{}}
	hash, _ := toolCatalogHash(tools)
	selection, err := agent.selectTools(tools, converted, hash, nil, &SuspendedTurn{})
	if err != nil {
		t.Fatal(err)
	}
	titles := map[string]string{}
	return selection, titles
}

type searchResponse struct {
	Tools []searchMatch `json:"tools"`
	Next  string        `json:"next"`
	Hint  string        `json:"hint"`
	raw   string
}

func runSearch(t *testing.T, selection *toolSelection, titles map[string]string, arguments string) searchResponse {
	t.Helper()
	result := selection.run(llm.ToolCall{ID: "s", Name: SearchToolName, Arguments: json.RawMessage(arguments)}, titles)
	if result.IsError {
		t.Fatalf("search refused: %s", result.Content)
	}
	var envelope struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
	}
	_ = json.Unmarshal(result.Content, &envelope)
	var response searchResponse
	if err := json.Unmarshal([]byte(envelope.Content[0].Text), &response); err != nil {
		t.Fatal(err)
	}
	response.raw = envelope.Content[0].Text
	return response
}

func TestSearchShowsOnlyTheResultsNearTheBestScore(t *testing.T) {
	selection, titles := newTestSelection(t, filterCatalog(), 0)
	response := runSearch(t, selection, titles, `{"query":"node hardware inventory disks"}`)
	scores := map[string]searchMatch{}
	for _, match := range response.Tools {
		scores[match.Name] = match
	}
	best := response.Tools[0]
	if best.Name != "list_node_hardware" || !best.Shown || best.Reason != "" {
		t.Fatalf("best = %+v", best)
	}
	for _, match := range response.Tools[1:] {
		wantShown := *match.Score >= DefaultSearchScoreRatio**best.Score
		if match.Shown != wantShown || (!match.Shown && match.Reason != "low_score") {
			t.Errorf("%s: score %v of %v, shown %v, reason %q", match.Name, *match.Score, *best.Score, match.Shown, match.Reason)
		}
	}
	for _, match := range response.Tools {
		if match.Score == nil || match.Exact {
			t.Fatalf("a search result without its score: %+v", match)
		}
	}
	if !slices.ContainsFunc(response.Tools, func(match searchMatch) bool { return match.Reason == "low_score" }) {
		t.Fatalf("no result was filtered: %+v", response.Tools)
	}
	if !strings.Contains(response.Hint, "exact name") || !strings.Contains(response.Hint, "no shown tool fits") || strings.Contains(response.Hint, "exceed") {
		t.Fatalf("hint = %q", response.Hint)
	}
	// The next step names the tools shown, and comes before the hint about
	// those not shown, so that the hint is not read as the next step.
	if !strings.Contains(response.Next, "call list_node_hardware") || !strings.Contains(response.Next, "without searching for them again") ||
		strings.Contains(response.Next, "last search") {
		t.Fatalf("next = %q", response.Next)
	}
	if tools, next, hint := strings.Index(response.raw, `"tools"`), strings.Index(response.raw, `"next"`), strings.Index(response.raw, `"hint"`); tools >= next || next >= hint {
		t.Fatalf("field order of %s", response.raw)
	}
	// The results filtered stay listed with their summary, never shown.
	for _, match := range response.Tools {
		if match.Summary == "" || slices.Contains(selection.active, match.Name) != match.Shown {
			t.Fatalf("match %+v, active %v", match, selection.active)
		}
	}
}

func TestSearchingAnExactNameShowsAFilteredTool(t *testing.T) {
	selection, titles := newTestSelection(t, filterCatalog(), 0)
	first := runSearch(t, selection, titles, `{"query":"node hardware inventory disks"}`)
	var filtered string
	for _, match := range first.Tools {
		if match.Reason == "low_score" {
			filtered = match.Name
			break
		}
	}
	if filtered == "" {
		t.Fatal("nothing filtered")
	}
	second := runSearch(t, selection, titles, `{"query":"`+filtered+`"}`)
	shown := slices.IndexFunc(second.Tools, func(match searchMatch) bool { return match.Name == filtered })
	if shown < 0 || !second.Tools[shown].Shown || !slices.Contains(selection.active, filtered) {
		t.Fatalf("exact name search = %+v", second.Tools)
	}
}

func TestSearchNeverHidesAToolAlreadyShown(t *testing.T) {
	selection, titles := newTestSelection(t, filterCatalog(), 0)
	// The core tool matches "status" weakly beside a strong hardware match.
	response := runSearch(t, selection, titles, `{"query":"node hardware inventory status"}`)
	for _, match := range response.Tools {
		if match.Name == "get_cluster_status" && (!match.Shown || match.Reason != "") {
			t.Fatalf("the core tool was hidden: %+v", match)
		}
	}
	if !slices.Contains(selection.active, "get_cluster_status") {
		t.Fatal("the core tool left the active set")
	}
}

func TestANegativeRatioShowsEveryResult(t *testing.T) {
	selection, titles := newTestSelection(t, filterCatalog(), -1)
	response := runSearch(t, selection, titles, `{"query":"node hardware inventory disks"}`)
	for _, match := range response.Tools {
		if !match.Shown || match.Reason != "" {
			t.Fatalf("filtered without a ratio: %+v", match)
		}
	}
	if selection.scoreRatio != 0 {
		t.Fatalf("ratio = %v", selection.scoreRatio)
	}
}

func TestTheRatioIsRecordedInTheExposure(t *testing.T) {
	for ratio, want := range map[float64]float64{0: DefaultSearchScoreRatio, 0.2: 0.2, -1: 0} {
		model := &scriptedLLM{t: t, steps: []llmStep{textStep("ok")}}
		result, err := searchAgent(t, model, &fakeSession{tools: searchCatalog()}, Config{SearchScoreRatio: ratio}).RunTurn(t.Context(), nil, "x", func(Event) error { return nil })
		if err != nil || result.Exposure.SearchScoreRatio != want {
			t.Fatalf("ratio %v: exposure %+v, %v", ratio, result.Exposure, err)
		}
	}
}

// A tool asked by its exact name is returned alone and shown, whatever its
// rank among the tools of the catalog, even past the results of a search
// or with a name made of stop words only.
func TestAnExactNameIsFoundWhateverItsRank(t *testing.T) {
	tool := func(name, description string) *mcp.Tool {
		return &mcp.Tool{Name: name, Description: description, InputSchema: objectSchema(), Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true}}
	}
	catalog := []*mcp.Tool{tool("get_cluster_status", "The status of the cluster.")}
	// Sixty tools match the words of the name better than the tool itself.
	for i := range 60 {
		catalog = append(catalog, tool(fmt.Sprintf("list_node_item_%02d", i), "List node items: node item list, node item list, node item list."))
	}
	catalog = append(catalog, tool("list_node_item", "Rarely needed."), tool("to_the_of", "A tool named with stop words."))
	for _, name := range []string{"list_node_item", "to_the_of"} {
		selection, titles := newTestSelection(t, catalog, 0)
		response := runSearch(t, selection, titles, `{"query":" `+name+` ","limit":1}`)
		if len(response.Tools) != 1 || response.Tools[0].Name != name || !response.Tools[0].Exact || !response.Tools[0].Shown || !slices.Contains(selection.active, name) {
			t.Fatalf("%s: response = %+v", name, response.Tools)
		}
		if response.Tools[0].Score != nil || strings.Contains(response.raw, `"score"`) {
			t.Fatalf("%s: an exact name carries a score: %s", name, response.raw)
		}
		if selection.searches != 1 {
			t.Fatalf("%s: searches = %d", name, selection.searches)
		}
	}
}

// An exact name returns one tool even with a larger limit, and its own
// search never removes it to make room for others.
func TestAnExactNameIsReturnedAloneAndKept(t *testing.T) {
	big := func(name string) *mcp.Tool {
		return &mcp.Tool{Name: name, Description: name + " tool.", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true},
			InputSchema: map[string]any{"type": "object", "description": strings.Repeat("s", 220<<10)}}
	}
	catalog := []*mcp.Tool{big("first"), big("second"), big("third"), big("fourth"), big("fifth")}
	catalog[0].Meta = mcp.Meta{CoreToolMeta: true}
	selection, titles := newTestSelection(t, catalog, 0)
	// Three tools found earlier fill the budget beside the core tool.
	for _, name := range []string{"second", "third", "fourth"} {
		selection.add(name)
	}
	response := runSearch(t, selection, titles, `{"query":"fifth","limit":10}`)
	if len(response.Tools) != 1 || !response.Tools[0].Shown || !slices.Contains(selection.active, "fifth") || !slices.Contains(selection.active, "first") {
		t.Fatalf("response = %+v, active = %v", response.Tools, selection.active)
	}
}

// A model told the user no tool read a value that a tool not listed read:
// the instruction asks for a search before saying a capability is missing.
func TestTheSearchInstructionAsksToSearchBeforeDenying(t *testing.T) {
	if !strings.Contains(searchInstruction, searchBeforeDenying) {
		t.Fatalf("the search instruction does not ask to search before denying: %q", searchInstruction)
	}
	// The tools, not the documentation: a model searched the documentation
	// and the code instead, then said it found nothing.
	if !strings.Contains(searchBeforeDenying, SearchToolName+" found nothing") {
		t.Fatalf("the rule does not name %s: %q", SearchToolName, searchBeforeDenying)
	}
}

// How to show a tool found but not shown is told by the hint of a search
// result only: said in the prompt or the description of search_tools, it
// made the model search again the exact name of the tools already shown.
func TestOnlyTheSearchHintTellsHowToShowAFilteredTool(t *testing.T) {
	for name, text := range map[string]string{"instruction": searchInstruction, "description": searchTool.Description, "schema": string(searchTool.InputSchema)} {
		if strings.Contains(text, "exact name") {
			t.Fatalf("the %s tells how to show a tool: %s", name, text)
		}
	}
}

// Of the tools several searches found, the most recent come first.
func TestTheToolsFoundLastComeFirst(t *testing.T) {
	selection, titles := newTestSelection(t, filterCatalog(), 0)
	runSearch(t, selection, titles, `{"query":"list_node_disks"}`)
	runSearch(t, selection, titles, `{"query":"list_node_packages"}`)
	if got := toolNames(selection.tools()); !slices.Equal(got, []string{SearchToolName, "list_node_packages", "list_node_disks", "get_cluster_status"}) {
		t.Fatalf("tools = %v", got)
	}
}

// The tools a search of a former turn showed stay shown, as those called:
// the question after an action follows it without searching again.
func TestSearchKeepsTheToolsAFormerSearchShowed(t *testing.T) {
	catalog := searchCatalog()
	catalog[0].Meta = nil
	session := &fakeSession{tools: catalog}
	shown, _ := json.Marshal(searchReply{Tools: []searchMatch{{Name: "list_node_packages", Shown: true}, {Name: "get_node", Shown: false, Reason: notShownLowScore}}})
	history := []llm.Message{
		{Role: llm.RoleUser, Text: "packages?"},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "s", Name: SearchToolName, Arguments: json.RawMessage(`{"query":"packages"}`)}}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "s", Content: textToolResult(string(shown))}}},
		{Role: llm.RoleAssistant, Text: "which node?"},
	}
	var tools []string
	model := &scriptedLLM{t: t, steps: []llmStep{func(request llm.Request, emit llm.EmitFunc) error {
		tools = toolNames(request.Tools)
		return textStep("ok")(request, emit)
	}}}
	if _, err := searchAgent(t, model, session, Config{}).RunTurn(t.Context(), history, "n1", func(Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	// A tool found but not shown is not carried.
	if !slices.Equal(tools, []string{SearchToolName, "list_node_packages"}) {
		t.Fatalf("tools = %v", tools)
	}
}

func TestSearchShownSkipsWhatItCannotRead(t *testing.T) {
	for _, result := range []llm.ToolResult{
		{IsError: true, Content: textToolResult(`{"tools":[{"name":"a","shown":true}]}`)},
		{Content: json.RawMessage(`{"_history_compaction":"omitted"}`)},
		{Content: textToolResult("not json")},
		{},
	} {
		if names := searchShown(result); names != nil {
			t.Fatalf("searchShown(%s) = %v", result.Content, names)
		}
	}
}

// The tools a former search showed are listed before the core tools, as the
// tools found in the turn: listed after, the model searched them again.
func TestSearchListsTheToolsAFormerSearchShowedFirst(t *testing.T) {
	session := &fakeSession{tools: searchCatalog()}
	shown, _ := json.Marshal(searchReply{Tools: []searchMatch{{Name: "list_node_packages", Shown: true}}})
	history := []llm.Message{
		{Role: llm.RoleUser, Text: "packages?"},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "s", Name: SearchToolName, Arguments: json.RawMessage(`{"query":"packages"}`)}}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "s", Content: textToolResult(string(shown))}}},
		{Role: llm.RoleAssistant, Text: "which node?"},
	}
	var tools []string
	model := &scriptedLLM{t: t, steps: []llmStep{func(request llm.Request, emit llm.EmitFunc) error {
		tools = toolNames(request.Tools)
		return textStep("ok")(request, emit)
	}}}
	if _, err := searchAgent(t, model, session, Config{}).RunTurn(t.Context(), history, "n1", func(Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(tools, []string{SearchToolName, "list_node_packages", "get_cluster_status"}) {
		t.Fatalf("tools = %v", tools)
	}
}

// A search of the last turn showing more tools than one search may show
// keeps the first of them, and does not push out a tool called the turn
// before.
func TestSearchBoundsTheToolsAFormerSearchShowed(t *testing.T) {
	catalog := searchCatalog()
	var matches []searchMatch
	for i := range maxHistoryShownTools + 3 {
		name := fmt.Sprintf("tool_%d", i)
		catalog = append(catalog, &mcp.Tool{Name: name, Description: "A tool.", InputSchema: objectSchema(), Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true}})
		matches = append(matches, searchMatch{Name: name, Shown: true})
	}
	shown, _ := json.Marshal(searchReply{Tools: matches})
	history := []llm.Message{
		{Role: llm.RoleUser, Text: "hardware?"},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "h", Name: "list_node_hardware", Arguments: json.RawMessage(`{}`)}}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "h", Content: json.RawMessage(`{}`)}}},
		{Role: llm.RoleAssistant, Text: "listed"},
		{Role: llm.RoleUser, Text: "anything else?"},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "s", Name: SearchToolName, Arguments: json.RawMessage(`{"query":"tools"}`)}}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "s", Content: textToolResult(string(shown))}}},
		{Role: llm.RoleAssistant, Text: "eight tools"},
	}
	var tools []string
	model := &scriptedLLM{t: t, steps: []llmStep{func(request llm.Request, emit llm.EmitFunc) error {
		tools = toolNames(request.Tools)
		return textStep("ok")(request, emit)
	}}}
	if _, err := searchAgent(t, model, &fakeSession{tools: catalog}, Config{}).RunTurn(t.Context(), history, "and?", func(Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	want := []string{SearchToolName}
	for i := range maxHistoryShownTools {
		want = append(want, fmt.Sprintf("tool_%d", i))
	}
	want = append(want, "get_cluster_status", "list_node_hardware")
	if !slices.Equal(tools, want) {
		t.Fatalf("tools = %v, want %v", tools, want)
	}
}

// A search keeps the results the model asked, then those close to the best
// one: the tool of a family that ranks past the limit, as
// start_instance_standby did behind start_instance in real searches.
func TestCloseResultsAddsTheResultsCloseToTheBest(t *testing.T) {
	results := []toolsearch.Result{
		{Name: "a", Score: 10}, {Name: "b", Score: 9}, {Name: "c", Score: 8},
		{Name: "d", Score: 7.7}, {Name: "e", Score: 5.2}, {Name: "f", Score: 3},
	}
	names := func(results []toolsearch.Result) string {
		var out []string
		for _, result := range results {
			out = append(out, result.Name)
		}
		return strings.Join(out, ",")
	}
	if got := names(closeResults(results, 2)); got != "a,b,c,d,e" {
		t.Fatalf("limit 2: %s", got)
	}
	// A result under half the best one stops the additions.
	if got := names(closeResults(results, 5)); got != "a,b,c,d,e" {
		t.Fatalf("limit 5: %s", got)
	}
	if got := names(closeResults(results[:3], 5)); got != "a,b,c" {
		t.Fatalf("fewer than the limit: %s", got)
	}
	// A clear search adds nothing.
	clear := []toolsearch.Result{{Name: "a", Score: 10}, {Name: "b", Score: 4}, {Name: "c", Score: 3}}
	if got := names(closeResults(clear, 1)); got != "a" {
		t.Fatalf("clear: %s", got)
	}
}
