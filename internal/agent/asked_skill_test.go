package agent

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/llm"
)

// crowdedSkillSession serves maxListedSkills skills before node-maintenance:
// the model is offered the others only, the client may ask it all the same.
func crowdedSkillSession() *fakeSession {
	session := skillSession(maintenanceSkill, maintenanceFrontmatter())
	var others []Skill
	for index := range maxListedSkills {
		name := fmt.Sprintf("skill-%d", index)
		uri := "skill://" + name + "/SKILL.md"
		others = append(others, Skill{URI: uri, Frontmatter: map[string]any{"name": name, "description": "Another procedure."}, Files: []SkillFile{{URI: uri, Size: 10}}})
	}
	session.skills = append(others, session.skills...)
	return session
}

// A skill the client asks is loaded by the agent before the model is asked,
// any skill served, a call of load_skill that is the first iteration of the
// turn, kept with it: the model, asked from the second, reads its procedure
// in its first request, its tools listed, and the run records it. The
// iterations of the events are those of the messages kept, which the stream
// names by them.
func TestAskedSkillIsLoadedBeforeTheModel(t *testing.T) {
	session := crowdedSkillSession()
	var first llm.Request
	model := &scriptedLLM{t: t, steps: []llmStep{
		func(request llm.Request, emit llm.EmitFunc) error {
			first = request
			return callsStep(llm.ToolCall{ID: "call-1", Name: "list_node_packages", Arguments: json.RawMessage(`{}`)})(request, emit)
		},
		answerStep(t, nil),
	}}
	var events []Event
	ctx := WithSkill(auth.WithBearerToken(t.Context(), "jwt-marker"), "node-maintenance")
	result, err := searchAgent(t, model, session, Config{}).RunTurn(ctx, nil, "/node-maintenance", func(event Event) error {
		events = append(events, event)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, tool := range first.Tools {
		if tool.Name == LoadSkillToolName && strings.Contains(tool.Description, "node-maintenance") {
			t.Fatalf("the ninth skill was listed to the model: %s", tool.Description)
		}
	}
	results := lastToolResults(first)
	if len(results) != 1 || results[0].IsError || !strings.Contains(string(results[0].Content), "# Node maintenance") {
		t.Fatalf("first request results = %+v", results)
	}
	if names := toolNames(first.Tools); len(names) < 2 || names[1] != "list_node_packages" {
		t.Fatalf("tools of the first request = %v", names)
	}
	// The load is iteration 1, the calls of the model iteration 2, its answer
	// iteration 3: the assistant message of iteration i is kept at 2i - 1 of
	// the messages, sequence 2i of the turn.
	iterations := map[string]int{}
	for _, event := range events {
		if event.Type == EventToolStarted {
			iterations[event.ToolName] = event.Iteration
			if event.ToolName == LoadSkillToolName && !strings.HasPrefix(event.ToolCallID, "call_skill_") {
				t.Fatalf("asked call id %q", event.ToolCallID)
			}
		}
	}
	if iterations[LoadSkillToolName] != 1 || iterations["list_node_packages"] != 2 {
		t.Fatalf("iterations = %v", iterations)
	}
	for name, iteration := range iterations {
		kept := result.Messages[2*iteration-1]
		if kept.Role != llm.RoleAssistant || len(kept.ToolCalls) != 1 || kept.ToolCalls[0].Name != name {
			t.Fatalf("message of iteration %d = %+v", iteration, kept)
		}
	}
	if len(result.Messages) != 6 || result.Messages[5].Role != llm.RoleAssistant || len(result.Messages[5].ToolCalls) != 0 {
		t.Fatalf("messages = %+v", result.Messages)
	}
	if len(result.Exposure.Skills) != 1 || result.Exposure.Skills[0].Name != "node-maintenance" {
		t.Fatalf("skills = %+v", result.Exposure.Skills)
	}

	// Next turn, the skill asked is listed to the model, which loads it again
	// for its procedure, its tools still listed.
	var next llm.Request
	again := &scriptedLLM{t: t, steps: []llmStep{
		func(request llm.Request, emit llm.EmitFunc) error {
			next = request
			return callsStep(llm.ToolCall{ID: "call-2", Name: LoadSkillToolName, Arguments: json.RawMessage(`{"skill":"node-maintenance"}`)})(request, emit)
		},
		answerStep(t, nil),
	}}
	turn, err := searchAgent(t, again, crowdedSkillSession(), Config{}).RunTurn(auth.WithBearerToken(t.Context(), "jwt-marker"), result.Messages, "yes, n1", func(Event) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	listed := false
	for _, tool := range next.Tools {
		listed = listed || tool.Name == LoadSkillToolName && strings.Contains(tool.Description, "node-maintenance: Use when the user plans to reboot a node.")
	}
	if names := toolNames(next.Tools); !listed || len(names) < 2 || names[1] != "list_node_packages" {
		t.Fatalf("next turn tools = %v, load_skill listing node-maintenance %v", names, listed)
	}
	if loaded := turn.Messages[len(turn.Messages)-2].ToolResults; len(loaded) != 1 || loaded[0].IsError {
		t.Fatalf("next turn load = %+v", loaded)
	}
}

// The model loads only the skills listed to it: one beyond them, never
// loaded in the conversation, stays refused, though the agent knows it.
func TestModelLoadsTheListedSkillsOnly(t *testing.T) {
	var second llm.Request
	model := &scriptedLLM{t: t, steps: []llmStep{
		callsStep(llm.ToolCall{ID: "call-1", Name: LoadSkillToolName, Arguments: json.RawMessage(`{"skill":"node-maintenance"}`)}),
		answerStep(t, func(request llm.Request) { second = request }),
	}}
	if _, err := searchAgent(t, model, crowdedSkillSession(), Config{}).RunTurn(auth.WithBearerToken(t.Context(), "jwt-marker"), nil, "I reboot n1", func(Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if results := lastToolResults(second); len(results) != 1 || !results[0].IsError || !strings.Contains(string(results[0].Content), "one of the skills its description lists") {
		t.Fatalf("load of an unlisted skill = %+v", results)
	}
}

// A skill asked that is not served to the user, misnamed, too large, whose
// load is refused, or asked of a turn of one iteration fails the turn before
// the model is asked, its step told as failed: no answer without the
// procedure the user chose.
func TestAskedSkillUnavailableFailsTheTurn(t *testing.T) {
	tampered := skillSession(maintenanceSkill, maintenanceFrontmatter())
	tampered.texts["skill://node-maintenance/SKILL.md"] += "changed"
	large := skillSession(maintenanceSkill, maintenanceFrontmatter())
	large.skills[0].Files[0].Size = maxSkillBytes + 1
	for name, test := range map[string]struct {
		session *fakeSession
		skill   string
		config  Config
		step    bool
	}{
		"not served":    {skillSession(maintenanceSkill, maintenanceFrontmatter()), "morning-check", Config{}, true},
		"misnamed":      {skillSession(maintenanceSkill, maintenanceFrontmatter()), "Node Maintenance", Config{}, false},
		"refused":       {tampered, "node-maintenance", Config{}, true},
		"too large":     {large, "node-maintenance", Config{}, true},
		"no skills":     {&fakeSession{tools: searchCatalog()}, "node-maintenance", Config{}, true},
		"one iteration": {skillSession(maintenanceSkill, maintenanceFrontmatter()), "node-maintenance", Config{MaxIterations: 1}, false},
	} {
		t.Run(name, func(t *testing.T) {
			var events []Event
			ctx := WithSkill(auth.WithBearerToken(t.Context(), "jwt-marker"), test.skill)
			agent := searchAgent(t, &scriptedLLM{t: t}, test.session, Config{})
			if test.config.MaxIterations > 0 {
				agent.config.MaxIterations = test.config.MaxIterations
			}
			_, err := agent.RunTurn(ctx, nil, "go", func(event Event) error {
				events = append(events, event)
				return nil
			})
			if !errors.Is(err, ErrSkillUnavailable) {
				t.Fatalf("err = %v", err)
			}
			finished := len(events) == 2 && events[0].Type == EventToolStarted && events[1].Type == EventToolFinished && events[1].ToolError && events[1].Iteration == 1
			if finished != test.step || (!test.step && len(events) != 0) {
				t.Fatalf("events = %+v", events)
			}
		})
	}
}

// A read of the skill asked that fails is a failure of the MCP, not a skill
// unavailable: the client is not told to choose another.
func TestAskedSkillReadFailureIsAFailureOfTheMCP(t *testing.T) {
	session := skillSession(maintenanceSkill, maintenanceFrontmatter())
	delete(session.texts, "skill://node-maintenance/SKILL.md")
	ctx := WithSkill(auth.WithBearerToken(t.Context(), "jwt-marker"), "node-maintenance")
	_, err := searchAgent(t, &scriptedLLM{t: t}, session, Config{}).RunTurn(ctx, nil, "go", func(Event) error { return nil })
	if err == nil || errors.Is(err, ErrSkillUnavailable) || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("err = %v", err)
	}
}

// A turn that asked a skill and waits on a confirmation resumes as any:
// its iterations go on from those of the load, and a skill in the context of
// the resume is not loaded again.
func TestAskedSkillTurnSuspendsAndResumes(t *testing.T) {
	session := skillSession(maintenanceSkill, maintenanceFrontmatter())
	model := &scriptedLLM{t: t, steps: []llmStep{
		callsStep(llm.ToolCall{ID: "call-1", Name: "stop_object", Arguments: json.RawMessage(`{}`)}),
		answerStep(t, nil),
	}}
	agent := searchAgent(t, model, session, Config{})
	ctx := WithSkill(auth.WithBearerToken(t.Context(), "jwt-marker"), "node-maintenance")
	result, err := agent.RunTurn(ctx, nil, "/node-maintenance", func(Event) error { return nil })
	if err != nil || result.Suspended == nil {
		t.Fatalf("result %+v, err %v", result, err)
	}
	if result.Suspended.Iteration != 2 || len(result.Suspended.Messages) != 3 || result.Suspended.Messages[1].ToolCalls[0].Name != LoadSkillToolName {
		t.Fatalf("suspended iteration %d, messages %+v", result.Suspended.Iteration, result.Suspended.Messages)
	}
	var loads int
	resumed, err := agent.ResumeTurn(ctx, nil, *result.Suspended, DecisionApprove, func(event Event) error {
		if event.Type == EventToolStarted && event.ToolName == LoadSkillToolName {
			loads++
		}
		return nil
	})
	if err != nil || resumed.Suspended != nil || loads != 0 {
		t.Fatalf("resumed %+v, loads %d, err %v", resumed, loads, err)
	}
}

// The skills a client may ask are those served that the agent lists, each
// once, their description bounded: not only the first maxListedSkills, and
// only those a run may ask and a load reads.
func TestSkillsListsTheSkillsAClientMayAsk(t *testing.T) {
	session := crowdedSkillSession()
	session.skills = append(session.skills,
		Skill{URI: "skill://elsewhere/SKILL.md", Frontmatter: map[string]any{"name": "misplaced", "description": "Named otherwise than its path."}},
		Skill{URI: "skill://skill-0/SKILL.md", Frontmatter: map[string]any{"name": "skill-0", "description": "Again."}},
		Skill{URI: "skill://long/SKILL.md", Frontmatter: map[string]any{"name": "long", "description": strings.Repeat("x", 400)}, Files: []SkillFile{{URI: "skill://long/SKILL.md", Size: 10}}},
		// Listed to the model, but a run could not ask them, or a load read them.
		Skill{URI: "skill://Node_Check/SKILL.md", Frontmatter: map[string]any{"name": "Node_Check", "description": "Misnamed."}, Files: []SkillFile{{URI: "skill://Node_Check/SKILL.md", Size: 10}}},
		Skill{URI: "skill://big/SKILL.md", Frontmatter: map[string]any{"name": "big", "description": "Too large."}, Files: []SkillFile{{URI: "skill://big/SKILL.md", Size: maxSkillBytes + 1}}},
		Skill{URI: "skill://bare/SKILL.md", Frontmatter: map[string]any{"name": "bare", "description": "No SKILL.md of its own."}},
	)
	skills, err := searchAgent(t, &scriptedLLM{t: t}, session, Config{}).Skills(auth.WithBearerToken(t.Context(), "jwt-marker"))
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, skill := range skills {
		names = append(names, skill.Name)
		if len(skill.Description) > maxSkillSummaryBytes {
			t.Fatalf("description of %s is %d bytes", skill.Name, len(skill.Description))
		}
	}
	if len(skills) != maxListedSkills+2 || names[maxListedSkills] != "node-maintenance" || names[maxListedSkills+1] != "long" {
		t.Fatalf("skills = %v", names)
	}
	empty, err := searchAgent(t, &scriptedLLM{t: t}, &fakeSession{tools: searchCatalog()}, Config{}).Skills(auth.WithBearerToken(t.Context(), "jwt-marker"))
	if err != nil || empty == nil || len(empty) != 0 {
		t.Fatalf("no skills = %v, %v", empty, err)
	}
}

func TestValidSkillName(t *testing.T) {
	for _, name := range []string{"a", "node-maintenance", "morning-check-2", strings.Repeat("a", 64)} {
		if !ValidSkillName(name) {
			t.Errorf("%q refused", name)
		}
	}
	for _, name := range []string{"", "-a", "a-", "Node", "a_b", "a b", "/etc/hosts", strings.Repeat("a", 65)} {
		if ValidSkillName(name) {
			t.Errorf("%q taken", name)
		}
	}
}
