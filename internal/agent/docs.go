package agent

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/opensvc/ai-agent/internal/docs"
	"github.com/opensvc/ai-agent/internal/llm"
	"github.com/opensvc/ai-agent/internal/toolsearch"
)

const (
	// SearchDocsToolName is the tool of the agent the model searches the
	// documentation of the backend with.
	SearchDocsToolName = "search_docs"
	// defaultDocsResults is the number of sections a search gives when the
	// model asks no limit.
	defaultDocsResults = 3
)

// docsReading closes the sections of a search: a value an example shows is
// no default, and what the sections do not say is not known from them.
const docsReading = "These sections describe OpenSVC, not this cluster. A value shown in an example is no default: never take one for the answer. " +
	"When the sections do not state the answer, look for it with the tools before saying the documentation does not."

// docsTool is the definition of search_docs.
func docsTool() llm.Tool {
	schema, _ := json.Marshal(map[string]any{
		"type": "object", "additionalProperties": false, "required": []string{"query"},
		"properties": map[string]any{
			"query": map[string]any{"type": "string", "maxLength": toolsearch.MaxQueryBytes, "description": "English keywords of the concept, procedure or setting looked for, such as: quorum arbitrator votes split"},
			"limit": map[string]any{"type": "integer", "minimum": 1, "maximum": docs.MaxResults, "description": fmt.Sprintf("the number of sections to return, %d by default", defaultDocsResults)},
		},
	})
	return llm.Tool{
		Name: SearchDocsToolName,
		Description: "Search the OpenSVC documentation, the book of the agent, and return the best matching sections in full. " +
			"Use it for what no tool returns: concepts, how a feature behaves, configuration examples, procedures, what each release added, removed or changed in the keywords, compatibility between versions, good practices. " +
			"The documentation describes OpenSVC, never the state of this cluster: the tools observe the cluster.",
		InputSchema: schema,
	}
}

// searchDocs runs a search_docs call: the sections found in the book, each
// with its page and headings. A refused search is a tool error for the
// model, never a failed turn.
func searchDocs(library *docs.Library, call llm.ToolCall) llm.ToolResult {
	fail := func(message string) llm.ToolResult {
		return llm.ToolResult{CallID: call.ID, Content: textToolResult(message), IsError: true}
	}
	var arguments struct {
		Query string `json:"query"`
		Limit int    `json:"limit"`
	}
	if err := json.Unmarshal(call.Arguments, &arguments); err != nil {
		return fail("search_docs takes a query and an optional limit.")
	}
	if arguments.Limit == 0 {
		arguments.Limit = defaultDocsResults
	}
	sections, err := library.Search(arguments.Query, arguments.Limit)
	if err != nil {
		return fail(fmt.Sprintf("The documentation search was refused: %v. Search again with a few English keywords.", err))
	}
	if len(sections) == 0 {
		return fail("No section of the documentation matches these keywords. Search again with other English keywords, or answer that the documentation does not say.")
	}
	var text strings.Builder
	fmt.Fprintf(&text, "Documentation: %s.\n\n", library.Source())
	for i, section := range sections {
		if i > 0 {
			text.WriteString("\n\n---\n\n")
		}
		fmt.Fprintf(&text, "Section: %s\nPage: %s\n\n%s", section.Title, section.Page, section.Text)
	}
	text.WriteString("\n\n---\n\n" + docsReading)
	return llm.ToolResult{CallID: call.ID, Content: textToolResult(text.String())}
}
