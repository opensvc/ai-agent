package docs

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"regexp"
	"strconv"
	"strings"

	"github.com/opensvc/ai-agent/internal/toolsearch"
)

const (
	// maxCodeChunkBytes bounds the text of a declaration: a longer one is cut
	// in parts at its lines, each indexed on its own.
	maxCodeChunkBytes = 6 << 10
	// maxCodeParentBytes bounds the text of a parent: a file longer than it
	// is packed in runs of its declarations, each up to it.
	maxCodeParentBytes = 6 << 10
	// maxCodeChunks bounds the declarations of a code index: om3 has some
	// 17 500 outside its tests and its generated code.
	maxCodeChunks = 64 << 10
)

// generated is the header of a generated Go file: such code repeats the
// words of the program on thousands of declarations, and teaches nothing.
var generated = regexp.MustCompile(`(?m)^// Code generated .* DO NOT EDIT\.$`)

// identifierWords cuts an identifier at its words: bootStartWaitMax gives
// boot Start Wait Max, the search making one word of an identifier.
var identifierWords = regexp.MustCompile(`([a-z0-9])([A-Z])|([A-Z]+)([A-Z][a-z])`)

// File is a source file of a code index: its path in the tree, and its text.
type File struct {
	Path string
	Text string
}

// CodeResult is the text a code search gives: a file, or a run of its
// declarations, and the lines it spans.
type CodeResult struct {
	Path       string
	Start, End int
	Text       string
}

// codeChunk is a declaration, or a part of a long one, as the index holds it:
// its text is a slice of the file, not a copy.
type codeChunk struct {
	file       int
	start, end int
	from, to   int
	name       string
}

// CodeIndex is an immutable searchable set of the declarations of a Go
// source tree, safe for concurrent searches. A search matches declarations,
// with their doc comments, and returns their parent: the file when it is
// short, else the run of its declarations, up to 6 KiB, holding them. A short
// function is so read with its neighbors, such as the one whose comment holds
// the words a question uses.
type CodeIndex struct {
	source   string
	files    []File
	chunks   []codeChunk
	parentOf []int
	parents  []CodeResult
	index    *toolsearch.Index
	// Skipped is the number of files left out: generated, or that do not
	// parse.
	Skipped int
}

// NewCode indexes the declarations of the Go files, their tests and their
// generated code left out; source names the code, such as its version.
func NewCode(files []File, source string) (*CodeIndex, error) {
	x := &CodeIndex{source: source}
	fset := token.NewFileSet()
	var documents []toolsearch.Document
	for _, file := range files {
		if !strings.HasSuffix(file.Path, ".go") || strings.HasSuffix(file.Path, "_test.go") {
			continue
		}
		if generated.MatchString(file.Text) {
			x.Skipped++
			continue
		}
		parsed, err := parser.ParseFile(fset, file.Path, file.Text, parser.ParseComments|parser.SkipObjectResolution)
		if err != nil {
			x.Skipped++
			continue
		}
		fileIndex := len(x.files)
		x.files = append(x.files, file)
		tokenFile := fset.File(parsed.Pos())
		first := len(x.chunks)
		for _, decl := range parsed.Decls {
			from := tokenFile.Offset(decl.Pos())
			var doc *ast.CommentGroup
			name := ""
			switch d := decl.(type) {
			case *ast.FuncDecl:
				doc, name = d.Doc, d.Name.Name
				if d.Recv != nil && len(d.Recv.List) > 0 {
					name = receiver(d.Recv.List[0].Type) + "." + name
				}
			case *ast.GenDecl:
				doc = d.Doc
				name = specNames(d)
			}
			if doc != nil {
				from = tokenFile.Offset(doc.Pos())
			}
			to := tokenFile.Offset(decl.End())
			for _, part := range cutLines(file.Text, from, to) {
				x.chunks = append(x.chunks, codeChunk{file: fileIndex, from: part[0], to: part[1], start: tokenFile.Line(tokenFile.Pos(part[0])), end: tokenFile.Line(tokenFile.Pos(max(part[1]-1, part[0]))), name: name})
			}
		}
		x.pack(first)
	}
	for i, c := range x.chunks {
		text := x.files[c.file].Text[c.from:c.to]
		documents = append(documents, toolsearch.Document{
			Name:        "c" + strconv.Itoa(i),
			Title:       inflections(withWords(c.name)),
			Keywords:    []string{inflections(pageWords(strings.TrimSuffix(x.files[c.file].Path, ".go")))},
			Description: inflections(withWords(text)),
		})
	}
	var err error
	if x.index, err = toolsearch.NewBounded(documents, maxCodeChunks); err != nil {
		return nil, fmt.Errorf("index code: %w", err)
	}
	return x, nil
}

// Source names the code of the index.
func (x *CodeIndex) Source() string { return x.source }

// Len is the number of declarations indexed, Parents the number of texts a
// search may give.
func (x *CodeIndex) Len() int     { return len(x.chunks) }
func (x *CodeIndex) Parents() int { return len(x.parents) }

// pack makes the parents of the chunks of the last file, from first on: the
// file when they fit in maxCodeParentBytes, else runs of consecutive chunks
// up to it, a chunk longer than it a parent alone.
func (x *CodeIndex) pack(first int) {
	chunks := x.chunks[first:]
	if len(chunks) == 0 {
		return
	}
	file := x.files[chunks[0].file]
	whole := chunks[len(chunks)-1].to-chunks[0].from <= maxCodeParentBytes
	start := 0
	for i := range chunks {
		if !whole && i > start && chunks[i].to-chunks[start].from > maxCodeParentBytes {
			x.addParent(file, chunks[start:i])
			start = i
		}
	}
	x.addParent(file, chunks[start:])
}

func (x *CodeIndex) addParent(file File, run []codeChunk) {
	id := len(x.parents)
	x.parents = append(x.parents, CodeResult{Path: file.Path, Start: run[0].start, End: run[len(run)-1].end, Text: file.Text[run[0].from:run[len(run)-1].to]})
	for range run {
		x.parentOf = append(x.parentOf, id)
	}
}

// Search gives the parents of the declarations best matching query, at most
// limit, the best first: a parent is given once, those far under the best
// score left out, and those past maxResultBytes too, the first always given.
func (x *CodeIndex) Search(query string, limit int) ([]CodeResult, error) {
	if limit < 1 || limit > MaxResults {
		return nil, fmt.Errorf("code search limit must be between 1 and %d", MaxResults)
	}
	results, err := x.index.Search(inflections(query), toolsearch.MaxResults)
	if err != nil {
		return nil, err
	}
	var (
		found []CodeResult
		seen  = map[int]bool{}
		size  int
	)
	for _, result := range results {
		if result.Score < minScoreRatio*results[0].Score {
			break
		}
		n, _ := strconv.Atoi(strings.TrimPrefix(result.Name, "c"))
		id := x.parentOf[n]
		if seen[id] {
			continue
		}
		seen[id] = true
		parent := x.parents[id]
		if len(found) > 0 && size+len(parent.Text) > maxResultBytes {
			continue
		}
		found, size = append(found, parent), size+len(parent.Text)
		if len(found) == limit {
			break
		}
	}
	return found, nil
}

// cutLines cuts the text from..to of a file in parts of whole lines of at
// most maxCodeChunkBytes, a line longer than it a part alone.
func cutLines(text string, from, to int) [][2]int {
	if to-from <= maxCodeChunkBytes {
		return [][2]int{{from, to}}
	}
	var parts [][2]int
	start := from
	for i := from; i < to; {
		end := strings.IndexByte(text[i:to], '\n')
		next := to
		if end >= 0 {
			next = i + end + 1
		}
		if next-start > maxCodeChunkBytes && i > start {
			parts = append(parts, [2]int{start, i})
			start = i
		}
		i = next
	}
	return append(parts, [2]int{start, to})
}

// withWords gives a text and the words of its identifiers, the search making
// one word of an identifier such as bootStartWaitMax.
func withWords(text string) string {
	return text + "\n" + identifierWords.ReplaceAllString(text, "$1$3 $2$4")
}

// receiver is the type a method is declared on.
func receiver(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.StarExpr:
		return receiver(t.X)
	case *ast.IndexExpr:
		return receiver(t.X)
	case *ast.IndexListExpr:
		return receiver(t.X)
	case *ast.Ident:
		return t.Name
	}
	return ""
}

// specNames are the names a declaration of types, constants or variables
// declares.
func specNames(d *ast.GenDecl) string {
	var names []string
	for _, spec := range d.Specs {
		switch s := spec.(type) {
		case *ast.TypeSpec:
			names = append(names, s.Name.Name)
		case *ast.ValueSpec:
			for _, n := range s.Names {
				names = append(names, n.Name)
			}
		}
	}
	return strings.Join(names, " ")
}
