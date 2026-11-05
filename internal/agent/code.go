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

// CodeShelf gives the code of a version of the daemon, the latest code when
// the version is empty, as docs.CodeShelf does.
type CodeShelf interface {
	// Latest is the reference of the latest code, such as main.
	Latest() string
	// Code gives the code index of version, the latest code when version is
	// empty, and what to tell of it; nil, saying why, when there is none.
	Code(ctx context.Context, version string) (*docs.CodeIndex, string)
}

// CodeSource gives the code a turn searches: the version the user names,
// else the one a search reads by default, as Codes does.
type CodeSource interface {
	// ByDefault says which code a search that names no version reads.
	ByDefault() string
	// Code gives the code index of version, else of the default one, and
	// what to tell of it; nil, saying why, when there is none. An error,
	// such as a failed MCP transport, ends the turn.
	Code(ctx context.Context, session MCPSession, version string) (*docs.CodeIndex, string, error)
}

// Codes serves the code of a shelf to a backend. Default gives the version
// of the daemon the conversation is about, such as the one the nodes of its
// cluster run, described by DefaultIs; with none, when it gives none, or
// when the code of that version cannot be read, a search reads the latest
// code. An error of Default, such as a failed MCP transport, ends the turn.
type Codes struct {
	Shelf     CodeShelf
	Default   func(ctx context.Context, session MCPSession) (string, error)
	DefaultIs string
	// Unknown says that Default gave no version, such as "The version the
	// nodes of this cluster run could not be read".
	Unknown string
}

// ByDefault says which code a search that names no version reads.
func (c Codes) ByDefault() string {
	latest := "the latest code of the daemon, at " + c.Shelf.Latest()
	if c.Default != nil && c.DefaultIs != "" {
		return c.DefaultIs + ", else " + latest
	}
	return latest
}

// Code gives the code of version, else of the default one.
func (c Codes) Code(ctx context.Context, session MCPSession, version string) (*docs.CodeIndex, string, error) {
	if version != "" || c.Default == nil {
		index, note := c.Shelf.Code(ctx, version)
		return index, note, nil
	}
	version, err := c.Default(ctx, session)
	if err != nil {
		return nil, "", err
	}
	if version == "" {
		index, note := c.Shelf.Code(ctx, "")
		if c.Unknown != "" {
			note = strings.TrimSpace(c.Unknown + ". " + note)
		}
		return index, note, nil
	}
	if index, note := c.Shelf.Code(ctx, version); index != nil {
		if c.DefaultIs != "" {
			note = strings.TrimSpace(note + " It is " + c.DefaultIs + ".")
		}
		return index, note, nil
	}
	index, note := c.Shelf.Code(ctx, "")
	return index, strings.TrimSpace(fmt.Sprintf("The code of %s, the version by default, could not be read. %s", version, note)), nil
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
			"version": map[string]any{"type": "string", "maxLength": maxVersionBytes, "description": "leave it out to search " + source.ByDefault() + "; set it only when the user names a version, as a tag in full, v<major>.<minor>.<patch>-rc<n>, or a commit"},
		},
	})
	return llm.Tool{
		Name: SearchCodeToolName,
		Description: "Search the Go source code of the OpenSVC agent daemon, om3, by default " + source.ByDefault() + ", unless the user names a version, and return the declarations best matching, with their doc comments, file and lines. " +
			"Use it only for why the daemon behaves as it does, or how a part of it works inside, such as a timeout, a retry, an order or a decision of the orchestration, when the documentation and the tools do not say: " +
			"the code tells what the daemon does, not what the user should do. For how to use or configure OpenSVC, search the documentation.",
		InputSchema: schema,
	}
}

// searchCode runs a search_code call: the declarations found in the code of
// the version asked, else of the one by default, each with its file and lines.
// A refused search is a tool error for the model; an error of the source,
// such as a failed MCP transport, ends the turn.
func searchCode(ctx context.Context, source CodeSource, session MCPSession, call llm.ToolCall) (llm.ToolResult, error) {
	fail := func(message string) (llm.ToolResult, error) {
		return llm.ToolResult{CallID: call.ID, Content: textToolResult(message), IsError: true}, nil
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
	index, note, err := source.Code(ctx, session, strings.TrimSpace(arguments.Version))
	if err != nil {
		return llm.ToolResult{}, err
	}
	if index == nil {
		return fail(note + " Answer without the code, or search again without a version, for the code read by default.")
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
	return llm.ToolResult{CallID: call.ID, Content: textToolResult(text.String())}, nil
}
