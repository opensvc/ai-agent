package agent

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/llm"
)

const maintenanceSkill = "---\nname: node-maintenance\ndescription: Use when the user plans to reboot a node.\nmetadata:\n  opensvc/tools: list_node_packages get_node_san_topology drain_node\n---\n\n# Node maintenance\n\nRead, then drain.\n"

// skillSession serves the search catalog, a reference text and one skill,
// listed with the manifest and the frontmatter of its SKILL.md.
func skillSession(text string, frontmatter map[string]any) *fakeSession {
	sum := sha256.Sum256([]byte(text))
	return &fakeSession{
		tools: searchCatalog(),
		resources: []*mcp.Resource{
			{URI: "om3://docs/object-config", Name: "object-config", MIMEType: "text/markdown", Description: "The configuration of an object."},
			{URI: "skill://node-maintenance/SKILL.md", Name: "node-maintenance", MIMEType: "text/markdown", Description: "Use when the user plans to reboot a node."},
		},
		texts: map[string]string{"skill://node-maintenance/SKILL.md": text},
		skills: []Skill{{
			URI:         "skill://node-maintenance/SKILL.md",
			Frontmatter: frontmatter,
			Files:       []SkillFile{{URI: "skill://node-maintenance/SKILL.md", Digest: "sha256:" + hex.EncodeToString(sum[:]), Size: int64(len(text))}},
		}},
	}
}

// loadResult is the result of a load, its read of the MCP not failing.
func loadResult(result llm.ToolResult, _ error) llm.ToolResult { return result }

func maintenanceFrontmatter() map[string]any {
	return map[string]any{
		"name": "node-maintenance", "description": "Use when the user plans to reboot a node.",
		"metadata": map[string]any{"opensvc/tools": "list_node_packages get_node_san_topology drain_node"},
	}
}

// The model loads a skill with load_skill: it gets the body, the
// frontmatter apart, with its origin, the backend of the agent; the tools the skill names are listed
// right after search_tools, as the tools a search finds; a tool the backend
// does not serve is named to the model; the run records the skill with the
// digest of its text. The files of the skill are not read_resource's.
func TestLoadSkillGivesItsProcedureAndItsTools(t *testing.T) {
	session := skillSession(maintenanceSkill, maintenanceFrontmatter())
	var first, second llm.Request
	model := &scriptedLLM{t: t, steps: []llmStep{
		func(request llm.Request, emit llm.EmitFunc) error {
			first = request
			return callsStep(llm.ToolCall{ID: "call-1", Name: LoadSkillToolName, Arguments: json.RawMessage(`{"skill":"node-maintenance"}`)})(request, emit)
		},
		answerStep(t, func(request llm.Request) { second = request }),
	}}
	result, err := searchAgent(t, model, session, Config{}).RunTurn(auth.WithBearerToken(t.Context(), "jwt-marker"), nil, "I reboot n1", func(Event) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	var load, read *llm.Tool
	for index := range first.Tools {
		switch first.Tools[index].Name {
		case LoadSkillToolName:
			load = &first.Tools[index]
		case ReadResourceToolName:
			read = &first.Tools[index]
		}
	}
	if load == nil || !strings.Contains(load.Description, "node-maintenance: Use when the user plans to reboot a node.") {
		t.Fatalf("load_skill = %+v", load)
	}
	if read == nil || strings.Contains(read.Description, "skill://") || !strings.Contains(read.Description, "om3://docs/object-config") {
		t.Fatalf("read_resource = %+v", read)
	}
	names := toolNames(second.Tools)
	if len(names) < 3 || names[0] != SearchToolName || names[1] != "list_node_packages" || names[2] != "get_node_san_topology" {
		t.Fatalf("tools after the load = %v", names)
	}
	results := lastToolResults(second)
	content := ""
	if len(results) == 1 {
		content = string(results[0].Content)
	}
	if len(results) != 1 || results[0].IsError || !strings.Contains(content, "Skill node-maintenance, loaded from the backend") ||
		!strings.Contains(content, "Not served by this backend: drain_node") || !strings.Contains(content, "# Node maintenance") ||
		strings.Contains(content, "opensvc/tools") {
		t.Fatalf("load result = %s", content)
	}
	sum := sha256.Sum256([]byte(maintenanceSkill))
	if len(result.Exposure.Skills) != 1 || result.Exposure.Skills[0] != (LoadedSkill{Name: "node-maintenance", URI: "skill://node-maintenance/SKILL.md", Digest: "sha256:" + hex.EncodeToString(sum[:])}) {
		t.Fatalf("skills = %+v", result.Exposure.Skills)
	}
	if !strings.Contains(strings.Join(result.Exposure.ShownTools, ","), LoadSkillToolName) || len(session.calls) != 0 {
		t.Fatalf("shown %v, calls %v", result.Exposure.ShownTools, session.calls)
	}
}

// A skill whose text differs from the listing, its size, its digest or its
// frontmatter, is not loaded: a tool error for the model, the turn going
// on, no tool shown and no skill recorded. So is an unknown skill.
func TestLoadSkillRefusesWhatDiffersFromTheListing(t *testing.T) {
	changed := maintenanceFrontmatter()
	changed["metadata"] = map[string]any{"opensvc/tools": "stop_object"}
	tampered := skillSession(maintenanceSkill, maintenanceFrontmatter())
	tampered.texts["skill://node-maintenance/SKILL.md"] = strings.Replace(maintenanceSkill, "then drain", "then stop", 1)
	for name, tc := range map[string]struct {
		session   *fakeSession
		arguments string
	}{
		"digest":      {tampered, `{"skill":"node-maintenance"}`},
		"frontmatter": {skillSession(maintenanceSkill, changed), `{"skill":"node-maintenance"}`},
		"unknown":     {skillSession(maintenanceSkill, maintenanceFrontmatter()), `{"skill":"node-repair"}`},
	} {
		t.Run(name, func(t *testing.T) {
			var second llm.Request
			model := &scriptedLLM{t: t, steps: []llmStep{
				callsStep(llm.ToolCall{ID: "call-1", Name: LoadSkillToolName, Arguments: json.RawMessage(tc.arguments)}),
				answerStep(t, func(request llm.Request) { second = request }),
			}}
			result, err := searchAgent(t, model, tc.session, Config{}).RunTurn(auth.WithBearerToken(t.Context(), "jwt-marker"), nil, "I reboot n1", func(Event) error { return nil })
			if err != nil {
				t.Fatal(err)
			}
			if results := lastToolResults(second); len(results) != 1 || !results[0].IsError {
				t.Fatalf("results = %+v", results)
			}
			if names := toolNames(second.Tools); strings.Contains(strings.Join(names, ","), "list_node_packages") || len(result.Exposure.Skills) != 0 {
				t.Fatalf("tools %v, skills %+v", names, result.Exposure.Skills)
			}
		})
	}
}

// A server without skills offers no load_skill; the description lists at
// most maxListedSkills skills, each line cut; a tool of the MCP named
// load_skill is refused.
func TestLoadSkillBoundsItsDescription(t *testing.T) {
	empty, err := newSkillShelf(t.Context(), &fakeSession{}, map[string]llm.Tool{}, nil)
	if err != nil || empty.tool != nil || !loadResult(empty.load(t.Context(), &fakeSession{}, "", llm.ToolCall{ID: "call-1", Name: LoadSkillToolName, Arguments: json.RawMessage(`{"skill":"a"}`)}, &toolSelection{}, false)).IsError {
		t.Fatalf("empty shelf = %+v, %v", empty, err)
	}
	session := &fakeSession{}
	for index := range maxListedSkills + 2 {
		name := "skill-" + strconv.Itoa(index)
		session.skills = append(session.skills, Skill{URI: "skill://" + name + "/SKILL.md", Frontmatter: map[string]any{"name": name, "description": strings.Repeat("d", 2*maxSkillSummaryBytes)}})
	}
	shelf, err := newSkillShelf(t.Context(), session, map[string]llm.Tool{}, nil)
	if err != nil || shelf.tool == nil || len(shelf.names) != maxListedSkills || strings.Contains(shelf.tool.Description, strings.Repeat("d", maxSkillSummaryBytes+1)) {
		t.Fatalf("shelf = %+v, %v", shelf, err)
	}
	if _, err := newSkillShelf(t.Context(), session, map[string]llm.Tool{LoadSkillToolName: {Name: LoadSkillToolName}}, nil); err == nil {
		t.Fatal("an MCP tool named load_skill was accepted")
	}
}

// The origin the model reads is the backend of the agent, never the name
// the server gives itself; a frontmatter the agent cannot read says why; a
// SKILL.md with CRLF line ends loads.
func TestLoadSkillNamesItsBackendAndItsErrors(t *testing.T) {
	load := func(session *fakeSession) llm.ToolResult {
		shelf, err := newSkillShelf(t.Context(), session, map[string]llm.Tool{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		selection := &toolSelection{catalog: map[string]llm.Tool{"list_node_packages": {Name: "list_node_packages"}}}
		return loadResult(shelf.load(t.Context(), session, "om3", llm.ToolCall{ID: "call-1", Name: LoadSkillToolName, Arguments: json.RawMessage(`{"skill":"node-maintenance"}`)}, selection, false))
	}
	if result := load(skillSession(maintenanceSkill, maintenanceFrontmatter())); result.IsError || !strings.Contains(string(result.Content), "loaded from the om3 backend") || strings.Contains(string(result.Content), "test-mcp") {
		t.Fatalf("load = %s", result.Content)
	}
	crlf := strings.ReplaceAll(maintenanceSkill, "\n", "\r\n")
	if result := load(skillSession(crlf, maintenanceFrontmatter())); result.IsError || !strings.Contains(string(result.Content), "# Node maintenance") {
		t.Fatalf("CRLF load = %s", result.Content)
	}
	listed := maintenanceFrontmatter()
	listed["tags"] = "a"
	unreadable := strings.Replace(maintenanceSkill, "metadata:", "tags:\n  - a\nmetadata:", 1)
	if result := load(skillSession(unreadable, listed)); !result.IsError || !strings.Contains(string(result.Content), "cannot be read") {
		t.Fatalf("unreadable load = %s", result.Content)
	}
}

// A SKILL.md the manifest gives beyond maxSkillBytes is refused before it
// is read.
func TestLoadSkillRefusesALargeSkillBeforeReadingIt(t *testing.T) {
	session := skillSession(maintenanceSkill, maintenanceFrontmatter())
	session.skills[0].Files[0].Size = maxSkillBytes + 1
	shelf, err := newSkillShelf(t.Context(), session, map[string]llm.Tool{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	result := loadResult(shelf.load(t.Context(), session, "om3", llm.ToolCall{ID: "call-1", Name: LoadSkillToolName, Arguments: json.RawMessage(`{"skill":"node-maintenance"}`)}, &toolSelection{}, false))
	if !result.IsError || len(session.reads) != 0 {
		t.Fatalf("load = %s, reads %v", result.Content, session.reads)
	}
}

// A skill whose name is not the last segment of its path is not listed, and
// the files of every skill received leave read_resource, those past the
// skills the description lists too.
func TestSkillShelfChecksNamesAndKeepsEveryFile(t *testing.T) {
	session := &fakeSession{}
	for index := range maxListedSkills + 1 {
		name := "skill-" + strconv.Itoa(index)
		session.skills = append(session.skills, Skill{URI: "skill://" + name + "/SKILL.md", Frontmatter: map[string]any{"name": name, "description": "d"}, Files: []SkillFile{{URI: "skill://" + name + "/SKILL.md"}}})
	}
	session.skills = append(session.skills, Skill{URI: "skill://other/SKILL.md", Frontmatter: map[string]any{"name": "liar", "description": "d"}, Files: []SkillFile{{URI: "skill://other/SKILL.md"}}})
	shelf, err := newSkillShelf(t.Context(), session, map[string]llm.Tool{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if slices.Contains(shelf.names, "liar") || len(shelf.files) != maxListedSkills+2 || !shelf.files["skill://skill-"+strconv.Itoa(maxListedSkills)+"/SKILL.md"] {
		t.Fatalf("names %v, files %v", shelf.names, shelf.files)
	}
}

// Load, question, ok: the tools of the skill the first turn loaded are
// listed in the next one, right after search_tools, from the listing of
// that turn; a skill no longer listed, or a refused load, carries nothing.
func TestLoadedSkillToolsStayListedInTheNextTurn(t *testing.T) {
	nextTools := func(t *testing.T, first, second *fakeSession) []string {
		t.Helper()
		ask := &scriptedLLM{t: t, steps: []llmStep{
			callsStep(llm.ToolCall{ID: "call-1", Name: LoadSkillToolName, Arguments: json.RawMessage(`{"skill":"node-maintenance"}`)}),
			answerStep(t, nil),
		}}
		turn, err := searchAgent(t, ask, first, Config{}).RunTurn(auth.WithBearerToken(t.Context(), "jwt-marker"), nil, "I reboot n1", func(Event) error { return nil })
		if err != nil {
			t.Fatal(err)
		}
		var names []string
		ok := &scriptedLLM{t: t, steps: []llmStep{answerStep(t, func(request llm.Request) { names = toolNames(request.Tools) })}}
		if _, err := searchAgent(t, ok, second, Config{}).RunTurn(auth.WithBearerToken(t.Context(), "jwt-marker"), turn.Messages, "Ok, go", func(Event) error { return nil }); err != nil {
			t.Fatal(err)
		}
		return names
	}
	if names := nextTools(t, skillSession(maintenanceSkill, maintenanceFrontmatter()), skillSession(maintenanceSkill, maintenanceFrontmatter())); len(names) < 3 || names[0] != SearchToolName || names[1] != "list_node_packages" || names[2] != "get_node_san_topology" {
		t.Fatalf("next turn tools = %v", names)
	}
	unlisted := skillSession(maintenanceSkill, maintenanceFrontmatter())
	unlisted.skills = nil
	if names := nextTools(t, skillSession(maintenanceSkill, maintenanceFrontmatter()), unlisted); slices.Contains(names, "list_node_packages") {
		t.Fatalf("a skill no longer listed carried %v", names)
	}
	tampered := skillSession(maintenanceSkill, maintenanceFrontmatter())
	tampered.texts["skill://node-maintenance/SKILL.md"] += "changed"
	if names := nextTools(t, tampered, skillSession(maintenanceSkill, maintenanceFrontmatter())); slices.Contains(names, "list_node_packages") {
		t.Fatalf("a refused load carried %v", names)
	}
}
