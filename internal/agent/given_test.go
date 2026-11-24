package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/opensvc/ai-agent/internal/docs"
	"github.com/opensvc/ai-agent/internal/llm"
)

// searchArbitrators searches the book for its arbitrators, with the sections
// the turn already gave.
func searchArbitrators(t *testing.T, library *docs.Library, given givenSections) string {
	t.Helper()
	result, err := searchDocs(context.Background(), &fixedDocs{library: library}, nil, llm.ToolCall{ID: "1", Name: SearchDocsToolName, Arguments: json.RawMessage(`{"query":"arbitrators","limit":5}`)}, "", given)
	if err != nil || result.IsError {
		t.Fatalf("result %s, %v", result.Content, err)
	}
	return toolText(t, result)
}

// A section a turn already returned is named again, not given twice, by a
// search of the turn or of the turn resumed from its stored messages.
func TestASectionReturnedInTheTurnIsNotGivenTwice(t *testing.T) {
	page := "# Cluster Quorum\n\n## Best Practices\n\n" + strings.Repeat("Configure the number of nodes minus 1 arbitrators, all on the same third site. ", 4) + "\n"
	library, err := docs.New([]docs.Page{{Path: "agent/configure.quorum.md", Text: page}}, "the test book at commit abc")
	if err != nil {
		t.Fatal(err)
	}
	given := givenSections{}
	first, second := searchArbitrators(t, library, given), searchArbitrators(t, library, given)
	if !strings.Contains(first, "Configure the number of nodes") || strings.Contains(first, alreadyGiven) {
		t.Fatalf("first search = %q", first)
	}
	if strings.Contains(second, "Configure the number of nodes") || !strings.Contains(second, "Section: Cluster Quorum\nPage: agent/configure.quorum.md\n\n"+alreadyGiven) {
		t.Fatalf("second search = %q", second)
	}
	// A turn resumed reads them from its messages as the store gives them
	// back, the results of its current answer included.
	state := SuspendedTurn{
		Messages: []llm.Message{{Role: llm.RoleUser, Text: "q"}},
		Calls:    []llm.ToolCall{{ID: "1", Name: SearchDocsToolName}, {ID: "2", Name: "stop_object"}},
		Results:  []llm.ToolResult{{CallID: "1", Content: textToolResult(first)}},
	}
	data, err := json.Marshal(state)
	if err != nil {
		t.Fatal(err)
	}
	var stored SuspendedTurn
	if err := json.Unmarshal(data, &stored); err != nil {
		t.Fatal(err)
	}
	if resumed := searchArbitrators(t, library, turnGivenSections(&stored)); !strings.Contains(resumed, alreadyGiven) {
		t.Fatalf("resumed search = %q", resumed)
	}
	// The sections of another book are not the same.
	other := givenSections{}
	other.read(llm.ToolResult{Content: textToolResult(strings.Replace(first, "commit abc", "commit def", 1))})
	if strings.Contains(searchArbitrators(t, library, other), alreadyGiven) {
		t.Fatal("a section of another book was taken as given")
	}
}

// A long page gives several sections under one heading: each is given in
// full, the first time.
func TestTheSectionsOfALongPageAreAllGiven(t *testing.T) {
	var page strings.Builder
	page.WriteString("# Cluster Quorum\n\n")
	// Six sections of level 2 packed three by three: two parents titled by
	// the page alone.
	for _, part := range []string{"Votes", "Placement", "Timeouts", "Split", "Relay", "Freeze"} {
		page.WriteString("## " + part + "\n\n" + strings.Repeat("The arbitrators of the cluster "+strings.ToLower(part)+" decide a split. ", 33) + "\n\n")
	}
	library, err := docs.New([]docs.Page{{Path: "agent/configure.quorum.md", Text: page.String()}}, "the test book at commit abc")
	if err != nil {
		t.Fatal(err)
	}
	first := searchArbitrators(t, library, givenSections{})
	if strings.Count(first, "Section: Cluster Quorum\nPage: ") < 2 {
		t.Fatalf("the page did not give several sections under one heading: %q", first)
	}
	if strings.Contains(first, alreadyGiven) {
		t.Fatalf("a section of the page was taken as given: %q", first)
	}
}

// A declaration of the code a turn returned is named again, not given
// twice; another declaration of the same file is given.
func TestADeclarationReturnedInTheTurnIsNotGivenTwice(t *testing.T) {
	result := "Code: om3 at v3.0.0.\n\nFile: daemon/hb.go:10-12\n\n```go\nfunc a() {\n\treturn\n}\n```" + sectionSeparator + codeReading
	given := givenSections{}
	given.read(llm.ToolResult{Content: textToolResult(result)})
	if given.give("Code: om3 at v3.0.0.", "File: daemon/hb.go:10-12", "func a() {\n\treturn\n}") {
		t.Fatal("a declaration returned was given again")
	}
	if !given.give("Code: om3 at v3.0.0.", "File: daemon/hb.go:20-22", "func b() {}") {
		t.Fatal("another declaration was taken as given")
	}
}

// The model reads the characters of a text result as they are, the reply of
// search_tools included, while the stored result keeps the same bytes before
// and after the store.
func TestTheModelReadsTheCharactersOfATextResult(t *testing.T) {
	result := llm.ToolResult{CallID: "1", Content: textToolResult("if a < b && c > d {")}
	if text, ok := result.PlainText(); !ok || text != "if a < b && c > d {" {
		t.Fatalf("plain text = %q, %v", text, ok)
	}
	data, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	var stored llm.ToolResult
	if err := json.Unmarshal(data, &stored); err != nil || string(stored.Content) != string(result.Content) {
		t.Fatalf("stored %s, was %s", stored.Content, result.Content)
	}
	reply, _ := marshalText(searchReply{Hint: "a < b && c"})
	if !strings.Contains(string(reply), "a < b && c") {
		t.Fatalf("reply = %s", reply)
	}
}
