package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/opensvc/ai-agent/internal/docs"
	"github.com/opensvc/ai-agent/internal/llm"
	"github.com/opensvc/ai-agent/internal/toolsearch"
)

// SearchCodeToolName is the tool of the agent the model searches the source
// code of the daemon with.
const SearchCodeToolName = "search_code"

// CodeSource gives the code a turn searches: the latest code of the daemon,
// or the version the user names, as docs.CodeShelf does.
type CodeSource interface {
	// Latest is the reference of the latest code, such as main.
	Latest() string
	// Code gives the code index of version, the latest code when version is
	// empty, and what to tell of it; nil, saying why, when there is none.
	Code(ctx context.Context, version string) (*docs.CodeIndex, string)
}

// maxCodeSearchesPerAsk bounds the code searches of a question: a broad one
// searched up to 9 times, each search adding up to 16 KiB of code to the
// context, past which the model answers with the code it read.
const maxCodeSearchesPerAsk = 6

// codeSearchesSpent answers a search past maxCodeSearchesPerAsk.
var codeSearchesSpent = fmt.Sprintf("The code searches of this question are spent, %d of them: answer with the code read so far, saying what it did not show.", maxCodeSearchesPerAsk)

// docsThenCode follows the sections of a documentation search when the code
// of the daemon can be searched too: the model reads it at the moment the
// book did not answer.
const docsThenCode = "When these sections do not say how the daemon behaves, such as a timeout, a retry, an order or a decision of its own, search its code with search_code before answering, rather than answering from memory or saying it is not documented."

// codeReading closes the declarations of a code search: the code tells what
// the daemon does, not what the user should do.
const codeReading = "This is the source code of the daemon at the version named above: it tells what the daemon does and why, not what the user should do, and a constant or a comment holds at this version only, a cluster running another version possibly behaving otherwise. " +
	"A name this code uses from another package, written package.Name, is found by searching that name with its package, package.Name alone. " +
	"For how to use or configure OpenSVC, answer from the documentation."

// codeTool is the definition of search_code.
func codeTool(source CodeSource) llm.Tool {
	schema, _ := json.Marshal(map[string]any{
		"type": "object", "additionalProperties": false, "required": []string{"query"},
		"properties": map[string]any{
			"query":   map[string]any{"type": "string", "maxLength": toolsearch.MaxQueryBytes, "description": "English keywords or Go identifiers of the behavior looked for, such as: arbitrator check timeout split"},
			"limit":   map[string]any{"type": "integer", "minimum": 1, "maximum": docs.MaxResults, "description": fmt.Sprintf("the number of files, or runs of declarations of a file, to return, %d by default", defaultDocsResults)},
			"version": map[string]any{"type": "string", "maxLength": maxVersionBytes, "description": "leave it out to search the latest code of the daemon, at " + source.Latest() + "; set it only when the user names a version, as a tag in full, v<major>.<minor>.<patch>-rc<n>, or a commit"},
		},
	})
	return llm.Tool{
		Name: SearchCodeToolName,
		Description: "Search the Go source code of the OpenSVC agent daemon, om3, its latest code unless the user names a version, and return the declarations best matching, with their doc comments, file and lines. " +
			"Use it only for why the daemon behaves as it does, or how a part of it works inside, such as a timeout, a retry, an order or a decision of the orchestration, when the documentation and the tools do not say: " +
			"the code tells what the daemon does, not what the user should do. For how to use or configure OpenSVC, search the documentation.",
		InputSchema: schema,
	}
}

// searchCode runs a search_code call: the declarations found in the latest
// code, or in the code of the version asked, each with its file and lines. A
// refused search is a tool error for the model.
func searchCode(ctx context.Context, source CodeSource, call llm.ToolCall) llm.ToolResult {
	fail := func(message string) llm.ToolResult {
		return llm.ToolResult{CallID: call.ID, Content: textToolResult(message), IsError: true}
	}
	var arguments struct {
		Query   string `json:"query"`
		Limit   int    `json:"limit"`
		Version string `json:"version"`
	}
	if err := json.Unmarshal(call.Arguments, &arguments); err != nil || len(arguments.Version) > maxVersionBytes {
		return fail("search_code takes a query, an optional limit and an optional version.")
	}
	if arguments.Limit == 0 {
		arguments.Limit = defaultDocsResults
	}
	index, note := source.Code(ctx, strings.TrimSpace(arguments.Version))
	if index == nil {
		return fail(note + " Answer without the code, or search again without a version, for the latest code.")
	}
	results, err := index.Search(arguments.Query, arguments.Limit)
	if err != nil {
		return fail(fmt.Sprintf("The code search was refused: %v. Search again with a few English keywords.", err))
	}
	if len(results) == 0 {
		return fail("No declaration of the code matches these keywords. Search again with other English keywords or Go identifiers, or answer that the code was not found.")
	}
	var text strings.Builder
	fmt.Fprintf(&text, "Code: %s.", index.Source())
	if note != "" {
		text.WriteString(" " + note)
	}
	text.WriteString("\n\n")
	for i, result := range results {
		if i > 0 {
			text.WriteString("\n\n---\n\n")
		}
		fmt.Fprintf(&text, "File: %s:%d-%d\n\n```go\n%s\n```", result.Path, result.Start, result.End, strings.TrimRight(result.Text, "\n"))
	}
	text.WriteString("\n\n---\n\n" + codeReading)
	return llm.ToolResult{CallID: call.ID, Content: textToolResult(text.String())}
}
