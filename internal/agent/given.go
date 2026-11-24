package agent

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"

	"github.com/opensvc/ai-agent/internal/llm"
)

// givenSections are the sections of the documentation and the declarations
// of the code a turn already returned in full, by the source of their result,
// their heading and their text: a long page gives several sections under one
// heading. A search of the turn returning one again gives a note in its
// place: the model searched again with other words for the same sections in
// a fifth of the documentation the turns of dev5 received, which every next
// request of the turn reads again.
type givenSections map[string]bool

// alreadyGiven is the text of a section a turn already returned. The history
// compaction of the next turns may reduce the result holding it: the model
// then searches again.
const alreadyGiven = "Already returned in full earlier in this turn: read it there, or search again if it is no longer above."

// sectionSeparator separates the sections of a result of search_docs or
// search_code, and the text closing it.
const sectionSeparator = "\n\n---\n\n"

func (g givenSections) key(source, heading, text string) string {
	sum := sha256.Sum256([]byte(text))
	return source + "\n" + heading + "\n" + hex.EncodeToString(sum[:8])
}

// give tells whether a section is given in full, the first time, or named
// with a note, the next ones.
func (g givenSections) give(source, heading, text string) bool {
	key := g.key(source, heading, text)
	if g[key] {
		return false
	}
	g[key] = true
	return true
}

// read records the sections a result of search_docs or search_code gave in
// full: its first line names its source; a section is a "Section:" line and
// its "Page:" line, or a "File:" line, then its text, a fenced block for
// the code. A section read wrong is not recorded, and is given again.
func (g givenSections) read(result llm.ToolResult) {
	text, ok := result.PlainText()
	if !ok {
		return
	}
	source, body, ok := strings.Cut(text, "\n\n")
	if !ok {
		return
	}
	for _, part := range strings.Split(body, sectionSeparator) {
		var heading, rest string
		switch {
		case strings.HasPrefix(part, "Section: "):
			first, second, _ := strings.Cut(part, "\n")
			page, sectionText, found := strings.Cut(second, "\n\n")
			if !found || !strings.HasPrefix(page, "Page: ") {
				continue
			}
			heading, rest = first+"\n"+page, sectionText
		case strings.HasPrefix(part, "File: "):
			first, block, found := strings.Cut(part, "\n\n")
			if !found || !strings.HasPrefix(block, "```go\n") || !strings.HasSuffix(block, "\n```") {
				continue
			}
			heading, rest = first, strings.TrimSuffix(strings.TrimPrefix(block, "```go\n"), "\n```")
		default:
			continue
		}
		if rest != alreadyGiven {
			g[g.key(source, heading, rest)] = true
		}
	}
}

// turnGivenSections reads the sections the turn returned so far, the turn
// resumed included: its messages and the results of its current answer.
func turnGivenSections(state *SuspendedTurn) givenSections {
	given := givenSections{}
	names := map[string]string{}
	readResults := func(results []llm.ToolResult) {
		for _, result := range results {
			if name := names[result.CallID]; (name == SearchDocsToolName || name == SearchCodeToolName) && !result.IsError {
				given.read(result)
			}
		}
	}
	for _, message := range state.Messages {
		for _, call := range message.ToolCalls {
			names[call.ID] = call.Name
		}
		readResults(message.ToolResults)
	}
	for _, call := range state.Calls {
		names[call.ID] = call.Name
	}
	readResults(state.Results)
	return given
}
