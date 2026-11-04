package docs

import (
	"context"
	"net/http"
	"net/url"
	"strings"
)

// keywordHistory are the pages of the keyword reference read with the prose:
// the releases that added, removed and changed each keyword, which no tool
// serves.
var keywordHistory = map[string]bool{
	"agent.reference.keywords/changes.md": true,
	"agent.reference.keywords/removed.md": true,
}

// Loader reads the book of an llms.txt index, and the release its keyword
// reference documents.
type Loader func(ctx context.Context, index string) (*Library, string, error)

// BookLoader reads a book of the OpenSVC documentation with client: the
// pages of the agent, under agent/, and the history of the keywords. The
// reference of the keywords is left out, the daemon serving it to the tools
// at the version it runs, and its 6 MB outweighing the prose in a search; so
// are the howtos, written for v2.1, their install and commands not those of
// v3.
func BookLoader(client *http.Client) Loader {
	return func(ctx context.Context, index string) (*Library, string, error) {
		book, err := Fetch(ctx, client, index, func(_, path string) bool {
			return strings.HasPrefix(path, "agent/") || keywordHistory[path]
		})
		if err != nil {
			return nil, "", err
		}
		source := "the OpenSVC book at " + index
		if link, err := url.Parse(index); err == nil {
			source = "the OpenSVC book at " + link.Host + strings.TrimSuffix(link.Path, "llms.txt")
		}
		if book.Build != "" {
			source += ", build " + book.Build
		}
		if book.KeywordReference != "" {
			source += ", keyword reference " + book.KeywordReference
		}
		library, err := New(book.Pages, source)
		return library, book.KeywordReference, err
	}
}
