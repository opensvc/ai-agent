package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

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

// DocsSource gives the documentation a turn searches, by release line.
type DocsSource interface {
	// Versions are the release lines with a book, the latest first, and
	// what a search gives when the model names none.
	Versions() (lines []string, byDefault string)
	// Docs gives the book for version, empty when the model names none, and
	// what to tell of the choice, such as the release line it is the book of.
	// An error, such as a failed MCP transport while reading the version of
	// the target, ends the turn.
	Docs(ctx context.Context, session MCPSession, version string) (*docs.Library, string, error)
}

// BookShelf holds the books of a documentation by release line, as
// docs.Shelf does.
type BookShelf interface {
	Lines() []string
	Book(ctx context.Context, version string) (*docs.Library, string)
}

// Books serves the books of a shelf to a backend. Default gives the version
// a search is about when the model names none, such as the one the cluster
// of the conversation runs, described by DefaultIs; with none, or when it
// gives none, the latest book. An error of Default, such as a failed MCP
// transport, ends the turn, as a failed call of the model does.
type Books struct {
	Shelf     BookShelf
	Default   func(ctx context.Context, session MCPSession) (string, error)
	DefaultIs string
	// Unknown says that Default gave no version, such as "The version the
	// nodes of this cluster run could not be read".
	Unknown string
}

// Versions gives the release lines of the shelf.
func (b Books) Versions() ([]string, string) {
	lines := b.Shelf.Lines()
	byDefault := "the latest"
	if len(lines) > 0 {
		byDefault += ", " + lines[0]
	}
	if b.Default != nil && b.DefaultIs != "" {
		byDefault = b.DefaultIs + ", else " + byDefault
	}
	return lines, byDefault
}

// Docs gives the book of version, else of the default one.
func (b Books) Docs(ctx context.Context, session MCPSession, version string) (*docs.Library, string, error) {
	// With no book, or none of a known line, the version changes nothing:
	// it is not read.
	if version == "" && b.Default != nil && len(b.Shelf.Lines()) > 0 {
		var err error
		if version, err = b.Default(ctx, session); err != nil {
			return nil, "", err
		}
		if version == "" && b.Unknown != "" {
			library, choice := b.Shelf.Book(ctx, "")
			if lines := b.Shelf.Lines(); library != nil && len(lines) > 0 {
				choice = fmt.Sprintf("%s: this is the book of the latest release line, %s.", b.Unknown, lines[0])
			}
			return library, choice, nil
		}
	}
	library, choice := b.Shelf.Book(ctx, version)
	return library, choice, nil
}

// auditedSession is the session of a turn for the MCP calls the agent
// makes of its own, such as reading the version of the cluster a search is
// about: each is told the audit, never the client, as an
// agent_tool_called event.
type auditedSession struct {
	MCPSession
	emit      EmitFunc
	iteration int
	purpose   string
	// listed are the tools the MCP lists this turn: another is not called.
	listed map[string]llm.Tool
}

// ErrToolNotListed is the call of a tool the MCP does not list, such as an
// MCP of an older version, or one whose catalog leaves it out: the agent
// does not call it, and the caller goes on without its answer, rather than
// ending the turn on the error the MCP would give.
var ErrToolNotListed = errors.New("the MCP lists no such tool")

func (s *auditedSession) CallTool(ctx context.Context, name string, arguments map[string]any) (*mcp.CallToolResult, error) {
	if _, ok := s.listed[name]; !ok {
		return nil, fmt.Errorf("%w: %s", ErrToolNotListed, name)
	}
	started := time.Now()
	result, err := s.MCPSession.CallTool(ctx, name, arguments)
	call := &AgentCall{Tool: name, Purpose: s.purpose, Error: err != nil || result == nil || result.IsError, Duration: time.Since(started)}
	if emitErr := emitAgentEvent(s.emit, Event{Type: EventAgentToolCalled, AgentCall: call, Iteration: s.iteration}); emitErr != nil {
		return nil, emitErr
	}
	return result, err
}

// maxVersionBytes bounds the version a model names.
const maxVersionBytes = 64

// docsReading closes the sections of a search: a value an example shows is
// no default, and what the sections do not say is not known from them.
const docsReading = "These sections describe OpenSVC, not this cluster. A value shown in an example is no default: never take one for the answer. " +
	"When the sections do not state the answer, look for it with the tools before saying the documentation does not."

// docsTool is the definition of search_docs, the release lines of source
// listed for the model to pick.
func docsTool(source DocsSource) llm.Tool {
	lines, byDefault := source.Versions()
	version := "the release line of the OpenSVC agent the question is about, such as 3.0, as the agent_version a tool reported for the nodes concerned gives it"
	if len(lines) > 0 {
		version += "; the lines with a book: " + strings.Join(lines, ", ")
	}
	version += "; by default " + byDefault
	schema, _ := json.Marshal(map[string]any{
		"type": "object", "additionalProperties": false, "required": []string{"query"},
		"properties": map[string]any{
			"query":   map[string]any{"type": "string", "maxLength": toolsearch.MaxQueryBytes, "description": "English keywords of the concept, procedure or setting looked for, such as: quorum arbitrator votes split"},
			"limit":   map[string]any{"type": "integer", "minimum": 1, "maximum": docs.MaxResults, "description": fmt.Sprintf("the number of sections to return, %d by default", defaultDocsResults)},
			"version": map[string]any{"type": "string", "maxLength": maxVersionBytes, "description": version},
		},
	})
	return llm.Tool{
		Name: SearchDocsToolName,
		Description: "Search the OpenSVC documentation, the book of a release line of the agent, and return the best matching sections in full. " +
			"Use it for what no tool returns: concepts, how a feature behaves, configuration examples, procedures, what each release added, removed or changed in the keywords, compatibility between versions, good practices. " +
			"The documentation describes OpenSVC, never the state of this cluster: the tools observe the cluster.",
		InputSchema: schema,
	}
}

// searchDocs runs a search_docs call: the sections found in the book of
// the version asked, each with its page and headings. A refused search is a
// tool error for the model; an error of the source, such as a failed MCP
// transport, fails the turn.
func searchDocs(ctx context.Context, source DocsSource, session MCPSession, call llm.ToolCall) (llm.ToolResult, error) {
	fail := func(message string) (llm.ToolResult, error) {
		return llm.ToolResult{CallID: call.ID, Content: textToolResult(message), IsError: true}, nil
	}
	var arguments struct {
		Query   string `json:"query"`
		Limit   int    `json:"limit"`
		Version string `json:"version"`
	}
	if err := json.Unmarshal(call.Arguments, &arguments); err != nil || len(arguments.Version) > maxVersionBytes {
		return fail("search_docs takes a query, an optional limit and an optional release line.")
	}
	if arguments.Limit == 0 {
		arguments.Limit = defaultDocsResults
	}
	library, choice, err := source.Docs(ctx, session, strings.TrimSpace(arguments.Version))
	if err != nil {
		return llm.ToolResult{}, fmt.Errorf("give the documentation: %w", err)
	}
	if library == nil {
		return fail("The documentation cannot be read at the moment. Answer with the tools, and say that the documentation could not be searched.")
	}
	sections, err := library.Search(arguments.Query, arguments.Limit)
	if err != nil {
		return fail(fmt.Sprintf("The documentation search was refused: %v. Search again with a few English keywords.", err))
	}
	if len(sections) == 0 {
		return fail("No section of the documentation matches these keywords. Search again with other English keywords, or answer that the documentation does not say.")
	}
	var text strings.Builder
	fmt.Fprintf(&text, "Documentation: %s.", library.Source())
	if choice != "" {
		text.WriteString(" " + choice)
	}
	text.WriteString("\n\n")
	for i, section := range sections {
		if i > 0 {
			text.WriteString("\n\n---\n\n")
		}
		fmt.Fprintf(&text, "Section: %s\nPage: %s\n\n%s", section.Title, section.Page, section.Text)
	}
	text.WriteString("\n\n---\n\n" + docsReading)
	return llm.ToolResult{CallID: call.ID, Content: textToolResult(text.String())}, nil
}
