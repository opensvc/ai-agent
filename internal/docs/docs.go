// Package docs searches the reference documentation of a backend, the pages
// of the OpenSVC book, with the BM25 index of toolsearch: the model reads
// there the concepts, procedures and practices no tool returns.
//
// The search is precise and its answer wide: the index holds the sections
// of a page, cut at its headings, and a search returns their parent, the
// whole page when it is short, else the run of its sections, up to 6 KiB,
// holding them. A question matching one section of a page so reads its
// neighbors too, such as the practices next to an example.
package docs

import (
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/opensvc/ai-agent/internal/toolsearch"
)

const (
	// maxSectionBytes bounds the text of a section: a longer one is cut in
	// parts at its paragraphs, each indexed and returned on its own.
	maxSectionBytes = 6 << 10
	// minSectionBytes is the text under which a section joins the one
	// before it in its page, its heading kept in the text: a code example
	// under its own heading says little alone.
	minSectionBytes = 200
	// maxParentBytes bounds the text of a parent, its rewritten headings
	// included: a page longer than it is packed in runs of its sections, each
	// up to it, and a section longer than it is a parent alone.
	maxParentBytes = 6 << 10
	// maxResultBytes bounds the text of a search: the parents past it are
	// left out, the first always given.
	maxResultBytes = 16 << 10
	// MaxResults bounds the sections of a search.
	MaxResults = 5
	// minScoreRatio is the share of the best score under which a section
	// matches too little to be given: a search gives fewer parents rather
	// than noise. Measured on 243 queries models sent: 0.5 and 0.6 keep every
	// answer 0.3 finds with a fifth less text, 0.7 loses some.
	minScoreRatio = 0.5
)

// Section is one section of a page, under its headings.
type Section struct {
	// ID is the page and the anchor of the section, unique in the library.
	ID string
	// Page is the path of the page in the book, such as agent/configure.quorum.md.
	Page string
	// Title is the path of the headings, the page first.
	Title string
	Text  string
	// titles are the headings of levels 1 to 3 the section is under, empty
	// where the page has none.
	titles [3]string
}

// Library is an immutable searchable set of sections, safe for concurrent
// searches.
type Library struct {
	// source names the snapshot of the documentation, as the model reads it.
	source string
	// sections are the units of the index, parents what a search returns,
	// and parentOf the parent of each section.
	sections map[string]Section
	parents  map[string]Section
	parentOf map[string]string
	// names are the ids of the sections the index names s0, s1...: a name
	// weighs three times in toolsearch, and the words of a path, such as
	// internals or daemon, would outweigh those of the text, as a bare
	// number would that of a query holding it.
	names map[string]string
	index *toolsearch.Index
}

var (
	heading = regexp.MustCompile(`^(#{1,3})\s+(.+?)\s*#*\s*$`)
	// The tabs of the book are html blocks around markdown, and its
	// includes insert icons: the text keeps the markdown only.
	htmlLine = regexp.MustCompile(`^\s*</?div\b[^>]*>\s*$`)
	include  = regexp.MustCompile(`\{\{#include [^}]*\}\}`)
	anchor   = regexp.MustCompile(`[^a-z0-9]+`)
	word     = regexp.MustCompile(`[\p{L}\p{N}]+`)
)

// questionWords open the questions a model sends rather than keywords, "what
// is a daemon", "how do I freeze a service": they would weigh as much as the
// words asked about, and the documentation leaves them out. The stop words of
// toolsearch are measured on the tools, and left as they are.
var questionWords = map[string]bool{
	"what": true, "how": true, "why": true, "where": true, "who": true, "do": true, "does": true, "did": true,
	"can": true, "could": true, "should": true, "would": true, "will": true, "i": true, "me": true, "my": true,
	"we": true, "our": true, "you": true, "your": true, "if": true, "not": true, "all": true,
}

// inflections makes one key of the forms of a verb the index of toolsearch
// keeps apart, it making one only of the singular and the plural: enable,
// enabled and enabling, as the prose of the book and the queries of the
// model use them all. The index of the documentation only reads these keys:
// that of the tools is left as it is measured.
func inflections(text string) string {
	return word.ReplaceAllStringFunc(strings.ToLower(text), func(w string) string {
		if questionWords[w] {
			return ""
		}
		// The plural of a form, settings or embeds, is the form: its s goes
		// first, as the index would drop it after.
		if len(w) > 6 && (strings.HasSuffix(w, "ings") || strings.HasSuffix(w, "eds")) {
			w = w[:len(w)-1]
		}
		switch {
		// string and thing keep their -ing: what is left has no vowel.
		case len(w) > 5 && strings.HasSuffix(w, "ing") && strings.ContainsAny(w[:len(w)-3], "aeiouy"):
			w = w[:len(w)-3]
		// speed, exceed and need are no past.
		case len(w) > 4 && strings.HasSuffix(w, "ed") && !strings.HasSuffix(w, "eed"):
			w = w[:len(w)-2]
		default:
			return w
		}
		// running and setting give run and set; install and fill keep
		// their double l, as pass keeps its s.
		if n := len(w); n > 2 && w[n-1] == w[n-2] && !strings.ContainsRune("lsz", rune(w[n-1])) {
			w = w[:n-1]
		}
		return w
	})
}

// pageWords are the words of the path of a page, such as internals daemon
// heartbeats relay: the page is named by its file, and its words weigh as
// keywords, under those of the headings.
func pageWords(path string) string {
	return strings.NewReplacer("/", " ", ".", " ", "_", " ", "-", " ").Replace(strings.TrimSuffix(path, ".md"))
}

// Page is one page of the documentation: its path in the book, such as
// agent/configure.quorum.md, and its markdown.
type Page struct {
	Path string
	Text string
}

// New indexes the sections of pages; source names the snapshot they are.
func New(pages []Page, source string) (*Library, error) {
	library := &Library{source: source, sections: map[string]Section{}, parents: map[string]Section{}, parentOf: map[string]string{}, names: map[string]string{}}
	var documents []toolsearch.Document
	for _, page := range pages {
		name := page.Path
		sections := split(name, page.Text)
		for i, section := range sections {
			id, n := section.ID, 1
			for _, taken := library.sections[id]; taken; _, taken = library.sections[id] {
				n++
				id = fmt.Sprintf("%s-%d", section.ID, n)
			}
			sections[i].ID = id
			library.sections[id] = sections[i]
			name := "s" + strconv.Itoa(len(documents))
			library.names[name] = id
			documents = append(documents, toolsearch.Document{Name: name, Title: inflections(section.Title), Keywords: []string{inflections(pageWords(page.Path))}, Description: inflections(section.Text)})
		}
		for _, parent := range parents(name, sections) {
			library.parents[parent.ID] = parent.Section
			for _, id := range parent.children {
				library.parentOf[id] = parent.ID
			}
		}
	}
	var err error
	if library.index, err = toolsearch.New(documents); err != nil {
		return nil, fmt.Errorf("index documentation: %w", err)
	}
	return library, nil
}

// split cuts a page at its headings of level 1 to 3, those of code blocks
// apart. A section with no text, a heading followed by another, is dropped.
func split(page, text string) []Section {
	var (
		sections []Section
		titles   [3]string
		body     []string
		fenced   bool
		id       = strings.TrimSuffix(page, ".md")
	)
	flush := func() {
		// The blank lines around the text go, the indent of a code block
		// opening it stays.
		content := strings.TrimRight(strings.TrimLeft(strings.Join(body, "\n"), "\n"), " \t\n")
		body = nil
		if strings.TrimSpace(content) == "" {
			return
		}
		var path []string
		for _, title := range titles {
			if title != "" {
				path = append(path, title)
			}
		}
		for i, part := range parts(content) {
			section := Section{ID: id, Page: page, Title: strings.Join(path, " > "), Text: part, titles: titles}
			if i > 0 {
				section.ID = fmt.Sprintf("%s-part%d", id, i+1)
			}
			sections = append(sections, section)
		}
	}
	for _, line := range strings.Split(text, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "```") {
			fenced = !fenced
		}
		if !fenced {
			if match := heading.FindStringSubmatch(line); match != nil {
				flush()
				level := len(match[1])
				titles[level-1] = strings.Trim(include.ReplaceAllString(match[2], ""), " `")
				for i := level; i < len(titles); i++ {
					titles[i] = ""
				}
				id = strings.TrimSuffix(page, ".md") + "#" + strings.Trim(anchor.ReplaceAllString(strings.ToLower(titles[level-1]), "-"), "-")
				continue
			}
			if htmlLine.MatchString(line) {
				continue
			}
		}
		body = append(body, include.ReplaceAllString(line, ""))
	}
	flush()
	return merge(sections)
}

// merge joins each section shorter than minSectionBytes to the one before
// it, when the two fit in maxSectionBytes.
func merge(sections []Section) []Section {
	var (
		merged []Section
		// tails are the headings of the last section each merged one holds.
		tails [][3]string
	)
	for _, section := range sections {
		if n := len(merged); n > 0 && len(section.Text) < minSectionBytes {
			last := &merged[n-1]
			// The headings join the text at their own level.
			text := last.Text + "\n\n"
			for _, line := range headings(tails[n-1], section.titles, 0) {
				text += line + "\n\n"
			}
			if text += section.Text; len(text) <= maxSectionBytes {
				last.Text = text
				tails[n-1] = section.titles
				continue
			}
		}
		merged = append(merged, section)
		tails = append(tails, section.titles)
	}
	return merged
}

// headings are the heading lines opening the text of a section read after
// one under the headings prev, below level depth: its own heading, and those
// above it that are not those of prev. A heading with no text of its own,
// such as the release a list of keyword changes is under, is in no section:
// the sections under it say it.
func headings(prev, titles [3]string, depth int) []string {
	level := 0
	for i, title := range titles {
		if title != "" {
			level = i + 1
		}
	}
	var (
		lines   []string
		changed bool
	)
	for i := depth; i < level; i++ {
		changed = changed || titles[i] != prev[i]
		if titles[i] != "" && (changed || i == level-1) {
			lines = append(lines, strings.Repeat("#", i+1)+" "+titles[i])
		}
	}
	return lines
}

// parts cuts a text longer than maxSectionBytes at its blank lines, and a
// paragraph longer than it at its lines: every word is indexed, the index
// reading only the first bytes of a text.
func parts(text string) []string {
	if len(text) <= maxSectionBytes {
		return []string{text}
	}
	var (
		result  []string
		current strings.Builder
	)
	// A code block with blank lines is one paragraph: a cut between two of
	// its pieces would open it in one part and close it in the next.
	var blocks []string
	for _, paragraph := range strings.Split(text, "\n\n") {
		if n := len(blocks); n > 0 && fences(blocks[n-1])%2 == 1 {
			blocks[n-1] += "\n\n" + paragraph
			continue
		}
		blocks = append(blocks, paragraph)
	}
	var paragraphs []string
	for _, block := range blocks {
		paragraphs = append(paragraphs, lines(block)...)
	}
	for _, paragraph := range paragraphs {
		if current.Len() > 0 && current.Len()+len(paragraph)+2 > maxSectionBytes {
			result = append(result, current.String())
			current.Reset()
		}
		if current.Len() > 0 {
			current.WriteString("\n\n")
		}
		current.WriteString(paragraph)
	}
	if current.Len() > 0 {
		result = append(result, current.String())
	}
	return result
}

// fences counts the lines of a text that open or close a code block.
func fences(text string) int {
	n := 0
	for _, line := range strings.Split(text, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "```") {
			n++
		}
	}
	return n
}

// lines cuts a paragraph longer than maxSectionBytes in groups of lines. A
// table keeps its header atop each group when it leaves room for a whole
// UTF-8 rune; otherwise all its lines are cut normally. A line longer than
// the bound is cut on a rune boundary, or at the bound when it is no UTF-8.
func lines(paragraph string) []string {
	if len(paragraph) <= maxSectionBytes {
		return []string{paragraph}
	}
	rows := strings.Split(paragraph, "\n")
	header := ""
	if len(rows) > 2 && strings.HasPrefix(rows[0], "|") && strings.HasPrefix(rows[1], "|") && strings.Trim(rows[1], "|-: ") == "" {
		candidate := rows[0] + "\n" + rows[1]
		// Repeating a larger header could leave a negative cut, or no
		// room to consume the next rune. Keep it in the ordinary lines.
		if len(candidate)+1+utf8.UTFMax <= maxSectionBytes {
			header, rows = candidate, rows[2:]
		}
	}
	var (
		groups  []string
		current strings.Builder
	)
	start := func() {
		current.Reset()
		current.WriteString(header)
	}
	start()
	for _, row := range rows {
		for len(row) > maxSectionBytes-len(header)-1 {
			bound := maxSectionBytes - len(header) - 1
			cut := bound
			for cut > 0 && !utf8.RuneStart(row[cut]) {
				cut--
			}
			// No rune starts before the bound: the line is no UTF-8, cut
			// at the bound.
			if cut == 0 {
				cut = bound
			}
			if current.Len() > len(header) {
				groups = append(groups, current.String())
				start()
			}
			groups = append(groups, strings.TrimPrefix(header+"\n"+row[:cut], "\n"))
			row = row[cut:]
		}
		if current.Len() > len(header) && current.Len()+len(row)+1 > maxSectionBytes {
			groups = append(groups, current.String())
			start()
		}
		if current.Len() > 0 {
			current.WriteString("\n")
		}
		current.WriteString(row)
	}
	if current.Len() > len(header) {
		groups = append(groups, current.String())
	}
	return groups
}

// parent is a parent and the sections it holds.
type parent struct {
	Section
	children []string
}

// parents gives the parents of the sections of a page: the page when its
// text fits in maxParentBytes; else its sections packed in order, each
// parent up to maxParentBytes, cut between sections of level 2 when one of
// them fits whole, so that a section is read with its neighbors.
func parents(page string, sections []Section) []parent {
	// A page of headings only has no section, and no parent.
	if len(sections) == 0 {
		return nil
	}
	if whole := join(page, sections, true); len(whole.Text) <= maxParentBytes {
		return []parent{whole}
	}
	// The units to pack are the sections of level 2 with what is under them,
	// and the sections of one too long to fit, one by one; the introduction
	// of the page is a unit of its own, packed with what follows.
	var groups, units [][]Section
	for _, section := range sections {
		if n := len(groups); n > 0 && section.titles[1] != "" && groups[n-1][0].titles[0] == section.titles[0] && groups[n-1][0].titles[1] == section.titles[1] {
			groups[n-1] = append(groups[n-1], section)
			continue
		}
		groups = append(groups, []Section{section})
	}
	for _, group := range groups {
		if len(join(page, group, false).Text) <= maxParentBytes {
			units = append(units, group)
			continue
		}
		for _, section := range group {
			units = append(units, []Section{section})
		}
	}
	var (
		result []parent
		pack   []Section
	)
	for _, unit := range units {
		// The bound holds for the text the parent gives, the headings it
		// rewrites included.
		if len(pack) > 0 && len(join(page, append(slices.Clip(pack), unit...), false).Text) > maxParentBytes {
			result = append(result, join(page, pack, false))
			pack = nil
		}
		pack = append(pack, unit...)
	}
	return append(result, join(page, pack, false))
}

// join makes one parent of sections: the page, or a run of its sections,
// titled by the heading they all are under, the headings below it kept in
// the text.
func join(page string, group []Section, wholePage bool) parent {
	first := group[0]
	joined := parent{Section: Section{ID: first.ID, Page: page}}
	// The parent is under the page title, and under the heading of level 2
	// all its sections share, if they share one.
	depth := 1
	if !wholePage && first.titles[1] != "" {
		depth = 2
		for _, section := range group[1:] {
			if section.titles[0] != first.titles[0] || section.titles[1] != first.titles[1] {
				depth = 1
				break
			}
		}
	}
	if wholePage {
		joined.ID = strings.TrimSuffix(page, ".md")
	}
	joined.titles[0] = first.titles[0]
	if depth == 2 {
		joined.titles[1] = first.titles[1]
	}
	joined.Title = strings.Trim(joined.titles[0]+" > "+joined.titles[1], " >")
	var texts []string
	prev := joined.titles
	for _, section := range group {
		joined.children = append(joined.children, section.ID)
		// The headings below the parent's own open their text.
		texts = append(texts, headings(prev, section.titles, depth)...)
		texts = append(texts, section.Text)
		prev = section.titles
	}
	joined.Text = strings.Join(texts, "\n\n")
	return joined
}

// Search gives the parents of the sections best matching query, at most
// limit, the best first: a parent is given once, and those past
// maxResultBytes are left out.
func (l *Library) Search(query string, limit int) ([]Section, error) {
	if limit < 1 || limit > MaxResults {
		return nil, fmt.Errorf("documentation search limit must be between 1 and %d", MaxResults)
	}
	results, err := l.index.Search(inflections(query), toolsearch.MaxResults)
	if err != nil {
		return nil, err
	}
	var (
		found []Section
		seen  = map[string]bool{}
		size  int
	)
	for _, result := range results {
		if result.Score < minScoreRatio*results[0].Score {
			break
		}
		id := l.parentOf[l.names[result.Name]]
		if seen[id] {
			continue
		}
		seen[id] = true
		section := l.parents[id]
		if len(found) > 0 && size+len(section.Text) > maxResultBytes {
			continue
		}
		found, size = append(found, section), size+len(section.Text)
		if len(found) == limit {
			break
		}
	}
	return found, nil
}

// Source names the snapshot of the documentation.
func (l *Library) Source() string { return l.source }

// Len is the number of sections the index holds.
func (l *Library) Len() int { return len(l.sections) }

// Parents is the number of the texts a search returns.
func (l *Library) Parents() int { return len(l.parents) }
