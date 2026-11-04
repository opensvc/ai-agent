package docs

import (
	"fmt"
	"strings"
	"testing"
	"unicode/utf8"
)

const quorumPage = `# Cluster Quorum

When a peer is flagged as stale by all heartbeats, the daemon assumes the cluster is in a split-brain situation. The daemon reports it in the cluster status, and the operators read it there before deciding anything about the services of the cluster.

## Enabling Quorum Enforcement

<div class="tabs">
<div class="tab" data-title="CLI">

` + "```bash\n# not a heading\nom cluster config update --set cluster.quorum=true\n```" + `

The keyword is shared by every node of the cluster: set it once, on any node, and it applies to the whole cluster at once.

</div>
</div>

## Quorum Behavior

Frozen nodes do not evaluate quorum. They will not shut down on split-brain. {{#include ../inc/node}}Frozen nodes still vote for peer nodes quorum evaluation. The daemon reports it in the cluster status, and the operators read it there before deciding anything about the services of the cluster.

### Use a tcp server as an arbitrator

    [arbitrator#a2]
    uri = dev2n2:22

## Empty

## Best Practices

* Configure the number of nodes minus 1 arbitrators, all on the same third site. The daemon reports it in the cluster status, and the operators read it there before deciding anything about the services of the cluster.
`

func TestLoadCutsPagesAtTheirHeadings(t *testing.T) {
	library, err := New([]Page{{Path: "agent/configure.quorum.md", Text: quorumPage}, {Path: "agent/empty.md", Text: "# Empty\n\n## Nothing\n"}}, "test book")
	if err != nil {
		t.Fatal(err)
	}
	if library.Len() != 4 {
		t.Fatalf("sections = %d, want 4: %+v", library.Len(), library.sections)
	}
	enabling := library.sections["agent/configure.quorum#enabling-quorum-enforcement"]
	if enabling.Title != "Cluster Quorum > Enabling Quorum Enforcement" || enabling.Page != "agent/configure.quorum.md" {
		t.Errorf("section = %+v", enabling)
	}
	if strings.Contains(enabling.Text, "<div") || !strings.Contains(enabling.Text, "# not a heading") {
		t.Errorf("the tabs are kept or the code is cut: %q", enabling.Text)
	}
	behavior := library.sections["agent/configure.quorum#quorum-behavior"]
	if strings.Contains(behavior.Text, "{{#include") || !strings.Contains(behavior.Text, "### Use a tcp server as an arbitrator\n\n    [arbitrator#a2]") {
		t.Errorf("the include is kept or the short section not joined: %q", behavior.Text)
	}
	if _, ok := library.sections["agent/configure.quorum#empty"]; ok {
		t.Error("a heading with no text is a section")
	}
}

func TestSearchGivesAShortPageWhole(t *testing.T) {
	library, err := New([]Page{{Path: "agent/configure.quorum.md", Text: quorumPage}}, "test book")
	if err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{"frozen node split brain shut down", "how many arbitrators third site", "enable quorum"} {
		sections, err := library.Search(query, 3)
		if err != nil || len(sections) != 1 {
			t.Fatalf("Search(%q) = %+v, %v; want the page once", query, sections, err)
		}
		page := sections[0]
		if page.ID != "agent/configure.quorum" || page.Title != "Cluster Quorum" ||
			!strings.HasPrefix(page.Text, "When a peer is flagged") ||
			!strings.Contains(page.Text, "## Enabling Quorum Enforcement\n\n```bash") ||
			!strings.Contains(page.Text, "## Best Practices\n\n* Configure the number of nodes minus 1") {
			t.Errorf("Search(%q) = %+v", query, page)
		}
	}
	if _, err := library.Search("quorum", MaxResults+1); err == nil {
		t.Error("a limit over MaxResults is accepted")
	}
}

func TestSearchGivesTheRunOfSectionsOfALongPage(t *testing.T) {
	// The page exceeds maxParentBytes, its section Base Actions does not.
	filler := strings.Repeat("The daemon keeps this state in memory and reports it in the cluster status. ", 20)
	page := "# Action\n\n## Base Actions\n\n" + filler + "\n\n### Start\n\nA frozen instance is started, and stays frozen.\n" + filler +
		"\n\n### Stop\n\nThe stop leaves the freeze as it is.\n\n## Logging\n\n" + strings.Repeat(filler, 3) + "\n"
	library, err := New([]Page{{Path: "agent/apps.operate.action.md", Text: page}}, "test book")
	if err != nil {
		t.Fatal(err)
	}
	// Stop, short, joins Start.
	if library.Len() != 3 || library.Parents() != 2 {
		t.Fatalf("sections %d, parents %d; want 3 and 2", library.Len(), library.Parents())
	}
	sections, err := library.Search("frozen instance started stays frozen stop freeze", 3)
	if err != nil || len(sections) == 0 {
		t.Fatalf("Search = %+v, %v", sections, err)
	}
	base := sections[0]
	if base.Title != "Action > Base Actions" || !strings.HasPrefix(base.Text, "The daemon keeps") ||
		!strings.Contains(base.Text, "### Start\n\nA frozen instance is started") || !strings.Contains(base.Text, "### Stop\n\nThe stop leaves") {
		t.Errorf("parent = %+v", base)
	}
	for _, section := range sections[1:] {
		if section.ID == base.ID {
			t.Errorf("parent %s given twice", base.ID)
		}
	}
}

func TestPartsCutsALongSectionAtItsParagraphs(t *testing.T) {
	paragraph := strings.Repeat("word ", maxSectionBytes/10)
	cut := parts(strings.Join([]string{paragraph, paragraph, paragraph}, "\n\n"))
	if len(cut) != 2 {
		t.Fatalf("parts = %d, want 2", len(cut))
	}
	for _, part := range cut {
		if len(part) > maxSectionBytes {
			t.Errorf("part of %d bytes", len(part))
		}
	}
}

func TestMergeKeepsTheHeadingWithNoParent(t *testing.T) {
	page := "# Ab\n\n" + strings.Repeat("The first page section is long enough to stand alone. ", 5) + "\n\n# B\n\nShort.\n\n# Longer title\n\nShort too.\n"
	library, err := New([]Page{{Path: "x.md", Text: page}}, "test book")
	if err != nil {
		t.Fatal(err)
	}
	section := library.sections["x#ab"]
	if library.Len() != 1 || !strings.Contains(section.Text, "# B\n\nShort.") || !strings.Contains(section.Text, "# Longer title\n\nShort too.") {
		t.Errorf("sections = %+v", library.sections)
	}
}

func TestLongTableIsIndexedWhole(t *testing.T) {
	var rows []string
	for i := range 200 {
		rows = append(rows, fmt.Sprintf("| `{var%d}` | the reference number %d of the table |", i, i))
	}
	rows = append(rows, "| `{exposed_devs}` | the devices the object exposes |")
	page := "# References\n\n| Reference | Description |\n|---|---|\n" + strings.Join(rows, "\n") + "\n"
	library, err := New([]Page{{Path: "refs.md", Text: page}}, "test book")
	if err != nil {
		t.Fatal(err)
	}
	if library.Len() < 2 {
		t.Fatalf("a table of %d bytes is one section", len(page))
	}
	for _, section := range library.sections {
		if len(section.Text) > maxSectionBytes || !strings.HasPrefix(section.Text, "| Reference | Description |\n|---|---|\n") {
			t.Errorf("part of %d bytes, header %q", len(section.Text), section.Text[:min(40, len(section.Text))])
		}
	}
	sections, err := library.Search("exposed_devs", 1)
	if err != nil || len(sections) != 1 || !strings.Contains(sections[0].Text, "the devices the object exposes") {
		t.Errorf("Search(exposed_devs) = %+v, %v", sections, err)
	}
}

func TestOversizedTableHeaderIsIndexed(t *testing.T) {
	page := "# Table\n\n|" + strings.Repeat("column", 1100) + "|\n|---|\n|searchabletail|"
	library, err := New([]Page{{Path: "agent/table.md", Text: page}}, "test book")
	if err != nil {
		t.Fatal(err)
	}
	for _, section := range library.sections {
		if len(section.Text) > maxSectionBytes || !utf8.ValidString(section.Text) {
			t.Fatalf("section of %d bytes, valid UTF-8 %v", len(section.Text), utf8.ValidString(section.Text))
		}
	}
	sections, err := library.Search("searchabletail", 1)
	if err != nil || len(sections) != 1 || !strings.Contains(sections[0].Text, "|searchabletail|") {
		t.Fatalf("Search(searchabletail) = %+v, %v", sections, err)
	}
}

func TestTableHeaderNearTheBoundPreservesItsText(t *testing.T) {
	for _, tc := range []struct {
		headerBytes int
		repeated    bool
	}{
		{maxSectionBytes + 464, false},
		{maxSectionBytes, false},
		{maxSectionBytes - 1, false},
		{maxSectionBytes - 2, false},
		{maxSectionBytes - 3, false},
		{maxSectionBytes - 4, false},
		{maxSectionBytes - 5, true},
	} {
		t.Run(fmt.Sprintf("header_%d_bytes", tc.headerBytes), func(t *testing.T) {
			header := "|" + strings.Repeat("x", tc.headerBytes-8) + "|\n|---|"
			body := "|" + strings.Repeat("🚀", 8) + "|\n|searchabletail|"
			var rebuilt strings.Builder
			for _, group := range lines(header + "\n" + body) {
				if len(group) > maxSectionBytes || !utf8.ValidString(group) {
					t.Fatalf("group of %d bytes, valid UTF-8 %v", len(group), utf8.ValidString(group))
				}
				if tc.repeated {
					if !strings.HasPrefix(group, header+"\n") {
						t.Fatal("a table group lost its repeated header")
					}
					group = strings.TrimPrefix(group, header+"\n")
				}
				rebuilt.WriteString(strings.ReplaceAll(group, "\n", ""))
			}
			want := body
			if !tc.repeated {
				want = header + "\n" + body
			}
			if rebuilt.String() != strings.ReplaceAll(want, "\n", "") {
				t.Fatal("splitting the table lost or duplicated text")
			}
		})
	}
}

func TestLinesCutsALineLongerThanTheBound(t *testing.T) {
	line := strings.Repeat("é", maxSectionBytes)
	for _, group := range lines(line) {
		if len(group) > maxSectionBytes || !utf8.ValidString(group) {
			t.Errorf("group of %d bytes, valid %v", len(group), utf8.ValidString(group))
		}
	}
}

func TestInflectionsMakeOneKeyOfAVerb(t *testing.T) {
	for text, want := range map[string]string{
		"Enabling Quorum Enforcement": "enabl quorum enforcement",
		"enabled":                     "enabl",
		"Running setting installed":   "run set install",
		"filling passed status":       "fill pass status",
		"red bed sing":                "red bed sing",
		"settings warnings":           "set warn",
		"exceed exceeds speed":        "exceed exceed speed",
		"string strings thing":        "string string thing",
		"what is a daemon":            " is a daemon",
		"How do I freeze my node":     "   freeze  node",
	} {
		if got := inflections(text); got != want {
			t.Errorf("inflections(%q) = %q, want %q", text, got, want)
		}
	}
}

func TestSearchGivesTheBestMatchesOnly(t *testing.T) {
	filler := strings.Repeat("The daemon keeps this state in memory and reports it in the cluster status. ", 4)
	library, err := New([]Page{
		{Path: "agent/internals.daemon.md", Text: "# Daemon\n\nThe daemon runs on every node, and its subsystems orchestrate the services. " + filler + "\n"},
		{Path: "agent/apps.operate.sessions.md", Text: "# Follow an Action\n\nWhat runs now, what ran, what is left: how do you ask after a session? Do it with the ids you are given. " + filler + "\n"},
		{Path: "agent/configure.quorum.md", Text: "# Cluster Quorum\n\nArbitrators vote when the cluster splits, and a node without quorum stops. " + filler + "\n"},
	}, "test book")
	if err != nil {
		t.Fatal(err)
	}
	// The words of a question weigh nothing: what, is and a do not pull the
	// page of the sessions ahead of the daemon.
	sections, err := library.Search("what is a daemon", 3)
	if err != nil || len(sections) == 0 || sections[0].ID != "agent/internals.daemon" {
		t.Fatalf("Search(what is a daemon) = %+v, %v", sections, err)
	}
	// A section far under the best score is left out.
	sections, err = library.Search("arbitrators vote quorum daemon", 3)
	if err != nil || len(sections) != 1 || sections[0].ID != "agent/configure.quorum" {
		t.Errorf("Search(arbitrators vote quorum daemon) = %d sections, first %+v, %v", len(sections), sections, err)
	}
}

func TestTheIntroductionOfALongPageIsPackedWithWhatFollows(t *testing.T) {
	filler := strings.Repeat("The copy of the data is sent to the nodes standing by, on its schedule. ", 30)
	page := "# Data Replication\n\nA failover service whose data lives on local disks needs a copy of that data.\n\n" +
		"## Reading the status\n\nA sync resource reports whether the copy is fresh. " + filler + "\n\n## Holding the updates\n\n" + filler + filler + "\n"
	library, err := New([]Page{{Path: "agent/apps.design.replication.md", Text: page}}, "test book")
	if err != nil {
		t.Fatal(err)
	}
	sections, err := library.Search("failover service local disks copy", 1)
	if err != nil || len(sections) != 1 {
		t.Fatalf("Search = %+v, %v", sections, err)
	}
	got := sections[0]
	if got.Title != "Data Replication" || !strings.HasPrefix(got.Text, "A failover service") || !strings.Contains(got.Text, "\n\n## Reading the status\n\nA sync resource") ||
		strings.Contains(got.Text, "## Holding the updates") {
		t.Errorf("parent = %q: %.200q", got.Title, got.Text)
	}
}

func TestPartsNeverCutsACodeBlock(t *testing.T) {
	// The prose fills a part but for the first piece of the code block: a
	// cut at the blank line after that piece would leave the block open.
	pieces := []string{"```ini\n[DEFAULT]\nnodes = n1 n2", "[fs#1]\ntype = flag", "[app#1]\nscript = /bin/true\n```"}
	prose := strings.Repeat("w", maxSectionBytes-len(pieces[0])-4)
	text := strings.Join(append([]string{prose}, append(pieces, "The end.")...), "\n\n")
	cut := parts(text)
	if len(cut) < 2 {
		t.Fatalf("parts = %d, want the text cut", len(cut))
	}
	for _, part := range cut {
		if fences(part)%2 != 0 {
			t.Errorf("a part opens a code block it does not close: %q", part[max(0, len(part)-60):])
		}
	}
}

func TestALongPageIsPackedInParentsWithTheirNeighbors(t *testing.T) {
	section := func(title string, n int) string {
		return "## " + title + "\n\n" + strings.Repeat("The "+strings.ToLower(title)+" keeps the copy of the data. ", n) + "\n\n"
	}
	// Eight sections of level 2 of about 1.3 KiB: twelve thousand bytes,
	// one parent each before, a few now, each cut between two of them.
	page := "# Data Replication\n\nA failover service whose data lives on local disks needs a copy.\n\n"
	for _, title := range []string{"Drbd", "Zfs", "Rsync", "Btrfs", "Symsnap", "Hp3par", "Dds", "Radosnap"} {
		page += section(title, 36)
	}
	library, err := New([]Page{{Path: "agent/apps.design.replication.md", Text: page}}, "test book")
	if err != nil {
		t.Fatal(err)
	}
	if library.Parents() < 2 || library.Parents() > 3 {
		t.Fatalf("parents = %d, want 2 or 3", library.Parents())
	}
	for id, parent := range library.parents {
		if len(parent.Text) > maxParentBytes || parent.Title != "Data Replication" {
			t.Errorf("parent %s: %d bytes, title %q", id, len(parent.Text), parent.Title)
		}
	}
	sections, err := library.Search("failover service local disks copy", 1)
	if err != nil || len(sections) != 1 || !strings.HasPrefix(sections[0].Text, "A failover service") || !strings.Contains(sections[0].Text, "\n\n## Drbd\n\nThe drbd") {
		t.Errorf("the introduction is not packed with the first sections: %+v, %v", sections, err)
	}
}

func TestTheBoundOfAParentCountsItsHeadings(t *testing.T) {
	// The texts of the sections fit in the bound, the page they make with
	// their rewritten headings does not.
	var page strings.Builder
	page.WriteString("# Pools\n\n")
	for i := range 60 {
		fmt.Fprintf(&page, "### A pool driver with a rather long name, number %d\n\n%s\n\n", i, strings.Repeat("x", maxParentBytes/60-12))
	}
	library, err := New([]Page{{Path: "agent/configure.pool.md", Text: page.String()}}, "test book")
	if err != nil {
		t.Fatal(err)
	}
	if library.Parents() < 2 {
		t.Fatalf("parents = %d: the page is given whole past the bound", library.Parents())
	}
	for id, parent := range library.parents {
		if len(parent.Text) > maxParentBytes {
			t.Errorf("parent %s: %d bytes, past %d", id, len(parent.Text), maxParentBytes)
		}
	}
}

// changesPage is a page of keyword changes: a heading of a release has no
// text of its own, the lists under it saying what it changed.
func changesPage(added int) string {
	var page strings.Builder
	page.WriteString("# Keyword changes\n\nWhat each release added, removed and changed.\n\n## v3.0.0-rc56\n\n### Added\n\n")
	for i := 0; i < added; i++ {
		fmt.Fprintf(&page, "- `cluster/rc56.keyword%03d`\n", i)
	}
	page.WriteString("\n### Removed\n\n- `node/rc56.gone`\n\n## v3.0.0-rc53\n\n### Added\n\n- `cluster/rc53.keyword`\n\n## v3.0.0-rc52\n\n### Changed\n\n- `svc/rc52.keyword`\n")
	return page.String()
}

func TestAPassageNamesTheReleaseOfEachKeywordChange(t *testing.T) {
	release := func(line string) string {
		return line[strings.Index(line, "/rc")+1 : strings.Index(line, "/rc")+5]
	}
	// 300 keywords added make a page packed in parents, its sections cut in
	// parts; 100 a page read whole, its introduction then the changes of a
	// release; 3 a page whose short sections merge.
	for _, added := range []int{300, 100, 3} {
		library, err := New([]Page{{Path: "agent.reference.keywords/changes.md", Text: changesPage(added)}}, "test")
		if err != nil {
			t.Fatal(err)
		}
		changes := 0
		for _, parent := range library.parents {
			// The model reads the title of a passage, then its text.
			heading := ""
			if _, release, ok := strings.Cut(parent.Title, " > v3.0.0-"); ok {
				heading = release
			}
			for _, line := range strings.Split(parent.Text, "\n") {
				if strings.HasPrefix(line, "## ") {
					heading = strings.TrimPrefix(line, "## v3.0.0-")
				}
				if strings.HasPrefix(line, "- `") {
					changes++
					if heading != release(line) {
						t.Errorf("%d added: %s is under the heading %q in %s:\n%s", added, line, heading, parent.ID, parent.Text)
					}
				}
			}
		}
		if changes != added+3 {
			t.Errorf("%d added: the parents hold %d changes, want %d", added, changes, added+3)
		}
	}
}

// TestLinesCutsALineThatIsNoUTF8 cuts at the bound a line where no rune
// starts, such as one of continuation bytes only, rather than going before
// its start.
func TestLinesCutsALineThatIsNoUTF8(t *testing.T) {
	line := strings.Repeat("\x80", 2*maxSectionBytes+100)
	groups := lines(line)
	for _, group := range groups {
		if len(group) == 0 || len(group) > maxSectionBytes {
			t.Fatalf("group of %d bytes", len(group))
		}
	}
	if joined := strings.Join(groups, ""); joined != line {
		t.Errorf("the groups hold %d bytes of the %d of the line", len(joined), len(line))
	}
}
