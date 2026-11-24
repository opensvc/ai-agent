package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"path"
	"slices"
	"strings"

	"github.com/opensvc/ai-agent/internal/llm"
)

const (
	// LoadSkillToolName is the tool of the agent the model loads a skill of
	// the MCP with: a procedure for a task, served as the MCP Skills
	// extension describes. Loading, not reading the file, activates a skill.
	LoadSkillToolName = "load_skill"
	// maxListedSkills bounds the skills the description of load_skill lists,
	// and maxSkillSummaryBytes the line of each: a server with many skills
	// does not grow every request.
	maxListedSkills      = 8
	maxSkillSummaryBytes = 300
	// maxSkillTools bounds the tools a skill adds to the run: one skill does
	// not fill the list of the model.
	maxSkillTools = 24
	// maxSkillBytes bounds the SKILL.md a load gives the model.
	maxSkillBytes = 64 << 10
	// skillToolsKey names, in the metadata of a skill, the tools it uses.
	skillToolsKey = "opensvc/tools"
)

// SkillSession is an MCP session whose server serves skills. A session
// without it, or a server without the extension, offers no load_skill.
type SkillSession interface {
	// ListSkills gives the skills of the server, as its skills/list lists
	// them; none when the server does not declare the extension.
	ListSkills(context.Context) ([]Skill, error)
}

// Skill is a skill as the server lists it: the URI of its SKILL.md, its
// frontmatter, and the manifest of its files.
type Skill struct {
	URI         string
	Frontmatter map[string]any
	Files       []SkillFile
}

// SkillFile is a file of a skill: its URI, the "sha256:<hex>" digest of its
// bytes, and their count.
type SkillFile struct {
	URI    string
	Digest string
	Size   int64
}

// LoadedSkill is a skill a turn loaded, for the record of what the model was
// given: its name, the URI of its SKILL.md and the digest of the text loaded.
type LoadedSkill struct {
	Name   string
	URI    string
	Digest string
}

// skillShelf holds the skills a turn may load and the tool that loads them;
// none when the server serves no skill.
type skillShelf struct {
	tool   *llm.Tool
	names  []string
	skills map[string]Skill
	// files are the URIs of every file of the skills listed, which the
	// resource reading leaves to load_skill.
	files  map[string]bool
	loaded []LoadedSkill
}

// newSkillShelf lists the skills of the session for load_skill, at most
// maxListedSkills, each named in its description by a line cut to
// maxSkillSummaryBytes: the line is all the model sees of a skill until it
// loads it. The name is the agent's: a tool of the MCP of the same name is
// refused. A listing that fails leaves the turn without the tool.
func newSkillShelf(ctx context.Context, session MCPSession, catalog map[string]llm.Tool) (*skillShelf, error) {
	shelf := &skillShelf{skills: map[string]Skill{}, files: map[string]bool{}}
	if _, taken := catalog[LoadSkillToolName]; taken {
		return nil, fmt.Errorf("MCP tool %q conflicts with the skill loading of the agent", LoadSkillToolName)
	}
	lister, ok := session.(SkillSession)
	if !ok {
		return shelf, nil
	}
	skills, err := lister.ListSkills(ctx)
	if err != nil || len(skills) == 0 {
		return shelf, nil
	}
	var lines []string
	for _, skill := range skills {
		// The files of every skill received leave read_resource, those of
		// the skills the description does not list too.
		for _, file := range skill.Files {
			shelf.files[file.URI] = true
		}
		name, _ := skill.Frontmatter["name"].(string)
		description, _ := skill.Frontmatter["description"].(string)
		// The name is the last segment of the path of its SKILL.md, as the
		// specification requires: a skill named otherwise is not listed.
		if name == "" || description == "" || path.Base(path.Dir(skill.URI)) != name || shelf.skills[name].URI != "" || len(shelf.names) == maxListedSkills {
			continue
		}
		shelf.names = append(shelf.names, name)
		shelf.skills[name] = skill
		lines = append(lines, name+": "+cutText(description, maxSkillSummaryBytes))
	}
	if len(shelf.names) == 0 {
		return shelf, nil
	}
	schema, _ := json.Marshal(map[string]any{
		"type": "object", "additionalProperties": false, "required": []string{"skill"},
		"properties": map[string]any{"skill": map[string]any{"type": "string", "enum": shelf.names, "description": "the name of the skill to load"}},
	})
	shelf.tool = &llm.Tool{
		Name: LoadSkillToolName,
		Description: "Load a skill of the backend, a procedure for a task, before the first step of a request its description matches. " +
			"Its tools are then listed among yours: call them by name, never search for them. Its tools stay listed in the next turns; load it again to read its procedure. The skills: " +
			strings.Join(lines, "; "),
		InputSchema: schema,
	}
	return shelf, nil
}

// known tells whether a call loads a skill: any call of load_skill, the name
// being the agent's.
func (s *skillShelf) known(name string) bool {
	return name == LoadSkillToolName
}

// load runs a load_skill call. It reads the SKILL.md of the skill from its
// server, unless the manifest gives it beyond maxSkillBytes, checks it
// against the manifest, its size, its digest and its frontmatter field by
// field, and gives the model its body, the frontmatter apart, with its
// origin, the backend the agent serves, never a name the server gives
// itself. The tools its metadata names are
// shown, in the place of the tools a search finds; those the backend does
// not serve, such as actions it disables, are named to the model. A refused
// load is a tool error for the model, never a failed turn.
func (s *skillShelf) load(ctx context.Context, session MCPSession, backend string, call llm.ToolCall, selection *toolSelection) llm.ToolResult {
	fail := func(message string) llm.ToolResult {
		return llm.ToolResult{CallID: call.ID, Content: textToolResult(message), IsError: true}
	}
	var arguments struct {
		Skill string `json:"skill"`
	}
	if len(s.names) == 0 {
		return fail("No skill of the backend can be loaded now: answer without it.")
	}
	if err := json.Unmarshal(call.Arguments, &arguments); err != nil || !slices.Contains(s.names, arguments.Skill) {
		return fail("load_skill takes skill, one of the skills its description lists.")
	}
	skill := s.skills[arguments.Skill]
	index := slices.IndexFunc(skill.Files, func(file SkillFile) bool { return file.URI == skill.URI })
	if index < 0 {
		return fail(fmt.Sprintf("The skill %s lists no SKILL.md of its own: it is not loaded.", arguments.Skill))
	}
	if skill.Files[index].Size > maxSkillBytes {
		return fail(fmt.Sprintf("The skill %s exceeds %d KiB: it is not loaded.", arguments.Skill, maxSkillBytes>>10))
	}
	result, err := session.ReadResource(ctx, skill.URI)
	if err != nil {
		return fail(fmt.Sprintf("The skill %s could not be read: %v", arguments.Skill, err))
	}
	var text string
	for _, contents := range result.Contents {
		if contents != nil && contents.URI == skill.URI {
			text = contents.Text
			break
		}
	}
	sum := sha256.Sum256([]byte(text))
	digest := "sha256:" + hex.EncodeToString(sum[:])
	if int64(len(text)) != skill.Files[index].Size || digest != skill.Files[index].Digest {
		return fail(fmt.Sprintf("The skill %s read differs from the one its server listed: it is not loaded.", arguments.Skill))
	}
	header, body, err := splitFrontmatter(text)
	if err != nil {
		return fail(fmt.Sprintf("The skill %s is not loaded: %v.", arguments.Skill, err))
	}
	frontmatter, err := parseFrontmatter(header)
	if err != nil {
		return fail(fmt.Sprintf("The frontmatter of the skill %s cannot be read: %v; it is not loaded.", arguments.Skill, err))
	}
	if !sameFrontmatter(frontmatter, skill.Frontmatter) {
		return fail(fmt.Sprintf("The frontmatter of the skill %s differs from the one its server listed: it is not loaded.", arguments.Skill))
	}
	var shown, missing []string
	for _, name := range declaredTools(frontmatter) {
		if _, served := selection.catalog[name]; !served {
			missing = append(missing, name)
			continue
		}
		shown = append(shown, name)
	}
	// The tools are shown in the order the skill names them, right after
	// search_tools.
	if selection.search {
		selection.startBatch()
		for _, name := range shown {
			selection.show(name)
		}
	}
	if !slices.ContainsFunc(s.loaded, func(loaded LoadedSkill) bool { return loaded.URI == skill.URI }) {
		s.loaded = append(s.loaded, LoadedSkill{Name: arguments.Skill, URI: skill.URI, Digest: digest})
	}
	origin := "the backend"
	if backend != "" {
		origin = "the " + backend + " backend"
	}
	var intro strings.Builder
	fmt.Fprintf(&intro, "Skill %s, loaded from %s; follow its procedure. ", arguments.Skill, origin)
	if len(shown) > 0 {
		intro.WriteString("Its tools are listed among yours: call them by name, never search for them. ")
	}
	if len(missing) > 0 {
		fmt.Fprintf(&intro, "Not served by this backend: %s; apply what the skill says of a missing tool. ", strings.Join(missing, ", "))
	}
	return llm.ToolResult{CallID: call.ID, Content: textToolResult(strings.TrimSpace(intro.String()) + "\n\n" + strings.TrimLeft(body, "\n"))}
}

// declaredTools gives the tools a skill names in its metadata, at most
// maxSkillTools, in its order.
func declaredTools(frontmatter map[string]any) []string {
	metadata, _ := frontmatter["metadata"].(map[string]any)
	tools, _ := metadata[skillToolsKey].(string)
	var names []string
	for _, name := range strings.Fields(tools) {
		if len(names) == maxSkillTools {
			break
		}
		if !slices.Contains(names, name) {
			names = append(names, name)
		}
	}
	return names
}

// carryTools shows the tools of the skills the recent turns loaded, as the
// tools a search of the recent turns showed are: listed before the core
// tools, at most maxHistoryTools. The model acts on the next turn on the plan
// it stated, such as after a question, with the tools at hand. The tools
// come from the listing of this turn, never from the history: a skill no
// longer served, or whose tools changed, carries nothing stale. A load the
// agent refused carries nothing.
func (s *skillShelf) carryTools(history []llm.Message, selection *toolSelection) {
	if !selection.search {
		return
	}
	results := map[string]llm.ToolResult{}
	for _, message := range history {
		for _, result := range message.ToolResults {
			results[result.CallID] = result
		}
	}
	carried := 0
	for i := len(history) - 1; i >= 0; i-- {
		for _, call := range slices.Backward(history[i].ToolCalls) {
			if call.Name != LoadSkillToolName || results[call.ID].IsError {
				continue
			}
			var arguments struct {
				Skill string `json:"skill"`
			}
			if json.Unmarshal(call.Arguments, &arguments) != nil {
				continue
			}
			skill, listed := s.skills[arguments.Skill]
			if !listed {
				continue
			}
			for _, name := range declaredTools(skill.Frontmatter) {
				if carried == maxHistoryTools {
					return
				}
				if selection.core[name] || slices.Contains(selection.carried, name) {
					continue
				}
				if selection.add(name) || slices.Contains(selection.active, name) {
					selection.carried = append(selection.carried, name)
					carried++
				}
			}
		}
	}
}
