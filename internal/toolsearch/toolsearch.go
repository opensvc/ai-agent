// Package toolsearch ranks the tools of an MCP catalog for a text query with
// BM25, so that the agent shows the model the tools a question needs rather
// than the whole catalog.
package toolsearch

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

const (
	// MaxDocuments bounds the tools of an index.
	MaxDocuments = 1024
	// MaxFieldBytes bounds each indexed text of a tool; the rest is ignored.
	MaxFieldBytes = 8 << 10
	// MaxDocumentTerms bounds the terms indexed for one tool.
	MaxDocumentTerms = 4096
	// MaxQueryBytes bounds a query.
	MaxQueryBytes = 512
	// MaxQueryTerms bounds the distinct terms of a query; the rest is ignored.
	MaxQueryTerms = 32
	// MaxResults bounds the results of a search.
	MaxResults = 50

	// KeywordsMeta is the _meta key of a tool listing extra search keywords.
	KeywordsMeta = "opensvc/keywords"

	k1 = 1.2
	b  = 0.75
)

// Field weights: a term of the name counts as three terms of the
// description, as BM25F does by repeating the fields.
const (
	nameWeight        = 3
	titleWeight       = 2
	keywordsWeight    = 2
	descriptionWeight = 1
	parameterWeight   = 1
)

// Document is the searchable text of one tool.
type Document struct {
	Name        string
	Title       string
	Description string
	Keywords    []string
	Parameters  []Parameter
}

// Parameter is one input of a tool.
type Parameter struct {
	Name        string
	Description string
}

// Result is a tool matching a query, with its BM25 score.
type Result struct {
	Name  string
	Score float64
}

// Index is an immutable BM25 index of tools, safe for concurrent searches.
type Index struct {
	names       []string
	termCounts  []map[string]int
	lengths     []float64
	averageLen  float64
	documentsOf map[string]int
}

// FromMCP describes MCP tools for the index: name, title, description, the
// names and descriptions of the input schema properties, and the keywords of
// the KeywordsMeta metadata, a list of strings.
func FromMCP(tools []*mcp.Tool) []Document {
	documents := make([]Document, 0, len(tools))
	for _, tool := range tools {
		if tool == nil {
			continue
		}
		document := Document{Name: tool.Name, Title: tool.Title, Description: tool.Description}
		if tool.Title == "" && tool.Annotations != nil {
			document.Title = tool.Annotations.Title
		}
		switch values := tool.Meta[KeywordsMeta].(type) {
		case []string:
			document.Keywords = append(document.Keywords, values...)
		case []any:
			// As decoded from JSON.
			for _, value := range values {
				if keyword, ok := value.(string); ok {
					document.Keywords = append(document.Keywords, keyword)
				}
			}
		}
		document.Parameters = schemaParameters(tool.InputSchema)
		documents = append(documents, document)
	}
	return documents
}

// schemaParameters reads the top-level properties of a JSON Schema object,
// sorted by name. Each property is read on its own: one without a
// description, such as the boolean schema false, keeps its name and leaves
// the others intact.
func schemaParameters(schema any) []Parameter {
	data, err := json.Marshal(schema)
	if err != nil {
		return nil
	}
	var object struct {
		Properties map[string]json.RawMessage `json:"properties"`
	}
	if json.Unmarshal(data, &object) != nil {
		return nil
	}
	parameters := make([]Parameter, 0, len(object.Properties))
	for name, raw := range object.Properties {
		var property struct {
			Description string `json:"description"`
		}
		_ = json.Unmarshal(raw, &property)
		parameters = append(parameters, Parameter{Name: name, Description: property.Description})
	}
	sort.Slice(parameters, func(i, j int) bool { return parameters[i].Name < parameters[j].Name })
	return parameters
}

// New indexes documents. Names must be unique and non-empty.
func New(documents []Document) (*Index, error) {
	return NewBounded(documents, MaxDocuments)
}

// NewBounded indexes documents as New does, at most limit of them: an
// index of the declarations of a source tree holds far more than a catalog
// of tools.
func NewBounded(documents []Document, limit int) (*Index, error) {
	if len(documents) > limit {
		return nil, fmt.Errorf("search index holds at most %d documents", limit)
	}
	index := &Index{documentsOf: map[string]int{}}
	var total float64
	for _, document := range documents {
		if document.Name == "" {
			return nil, fmt.Errorf("tool search document has no name")
		}
		if _, exists := index.documentsOf[document.Name]; exists {
			return nil, fmt.Errorf("tool search document %q is duplicated", document.Name)
		}
		counts := map[string]int{}
		length := 0
		add := func(text string, weight int) {
			for _, term := range terms(bounded(text, MaxFieldBytes)) {
				if length+weight > MaxDocumentTerms {
					return
				}
				counts[term] += weight
				length += weight
			}
		}
		add(document.Name, nameWeight)
		add(document.Title, titleWeight)
		add(strings.Join(document.Keywords, " "), keywordsWeight)
		add(document.Description, descriptionWeight)
		for _, parameter := range document.Parameters {
			add(parameter.Name, parameterWeight)
			add(parameter.Description, parameterWeight)
		}
		index.documentsOf[document.Name] = len(index.names)
		index.names = append(index.names, document.Name)
		index.termCounts = append(index.termCounts, counts)
		index.lengths = append(index.lengths, float64(length))
		total += float64(length)
	}
	if len(index.names) > 0 {
		index.averageLen = total / float64(len(index.names))
	}
	return index, nil
}

// Search ranks the tools for query, the best first, and returns at most limit
// of them, those matching at least one term. Equal scores are ordered by
// name, so that a search always returns the same results.
func (x *Index) Search(query string, limit int) ([]Result, error) {
	if limit < 1 || limit > MaxResults {
		return nil, fmt.Errorf("tool search limit must be between 1 and %d", MaxResults)
	}
	if len(query) > MaxQueryBytes || !utf8.ValidString(query) {
		return nil, fmt.Errorf("tool search query must be at most %d bytes of text", MaxQueryBytes)
	}
	var queryTerms []string
	for _, term := range terms(query) {
		if !slices.Contains(queryTerms, term) {
			queryTerms = append(queryTerms, term)
			if len(queryTerms) == MaxQueryTerms {
				break
			}
		}
	}
	if len(queryTerms) == 0 {
		return nil, fmt.Errorf("tool search query has no searchable word")
	}
	count := float64(len(x.names))
	scores := make([]float64, len(x.names))
	for _, term := range queryTerms {
		frequency := 0
		for _, counts := range x.termCounts {
			if counts[term] > 0 {
				frequency++
			}
		}
		if frequency == 0 {
			continue
		}
		idf := math.Log(1 + (count-float64(frequency)+0.5)/(float64(frequency)+0.5))
		for document, counts := range x.termCounts {
			tf := float64(counts[term])
			if tf == 0 {
				continue
			}
			norm := k1 * (1 - b + b*x.lengths[document]/x.averageLen)
			scores[document] += idf * tf * (k1 + 1) / (tf + norm)
		}
	}
	results := make([]Result, 0, len(x.names))
	for document, score := range scores {
		if score > 0 {
			results = append(results, Result{Name: x.names[document], Score: score})
		}
	}
	sort.Slice(results, func(i, j int) bool {
		if results[i].Score != results[j].Score {
			return results[i].Score > results[j].Score
		}
		return results[i].Name < results[j].Name
	})
	return results[:min(limit, len(results))], nil
}

// Len is the number of indexed tools.
func (x *Index) Len() int { return len(x.names) }

// terms splits text into the keys of its words: lowercase letters and
// digits, the accents of Latin letters folded, singular and plural made one
// by stem, stop words dropped.
// Underscores, dashes and dots separate words, so that list_node_hardware
// gives list, node and hardware.
func terms(text string) []string {
	var words []string
	var word strings.Builder
	flush := func() {
		if word.Len() == 0 {
			return
		}
		term := stem(word.String())
		word.Reset()
		if !stopWords[term] {
			words = append(words, term)
		}
	}
	for _, r := range text {
		// A decomposed accent, the letter then its combining mark, is folded
		// as a precomposed one: the mark is dropped inside the word.
		if unicode.Is(unicode.Mn, r) {
			continue
		}
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			word.WriteString(fold(unicode.ToLower(r)))
			continue
		}
		flush()
	}
	flush()
	return words
}

// stem gives the singular and the plural of a word the same key, without
// guessing which singular a plural has: it removes -ies (to y), else -es,
// else -s, then a final e. cache and caches, patch and patches, status and
// statuses, node and nodes each share a key. The key is only compared, never
// shown. Words ending in ss, is, as, or us after four letters (status, alias,
// address) keep their s.
func stem(term string) string {
	switch {
	case len(term) > 4 && strings.HasSuffix(term, "ies"):
		term = term[:len(term)-3] + "y"
	case len(term) > 3 && strings.HasSuffix(term, "es"):
		term = term[:len(term)-2]
	case len(term) > 2 && strings.HasSuffix(term, "s") && !strings.HasSuffix(term, "ss") && !strings.HasSuffix(term, "is") &&
		!strings.HasSuffix(term, "as") && !(len(term) > 4 && strings.HasSuffix(term, "us")):
		term = term[:len(term)-1]
	}
	if len(term) > 3 && strings.HasSuffix(term, "e") {
		term = term[:len(term)-1]
	}
	return term
}

var accents = map[rune]string{
	'à': "a", 'â': "a", 'ä': "a", 'á': "a", 'ã': "a", 'å': "a",
	'ç': "c",
	'é': "e", 'è': "e", 'ê': "e", 'ë': "e",
	'î': "i", 'ï': "i", 'í': "i", 'ì': "i",
	'ô': "o", 'ö': "o", 'ó': "o", 'ò': "o", 'õ': "o",
	'ù': "u", 'û': "u", 'ü': "u", 'ú': "u",
	'ÿ': "y", 'ñ': "n", 'œ': "oe", 'æ': "ae",
}

func fold(r rune) string {
	if folded, ok := accents[r]; ok {
		return folded
	}
	return string(r)
}

// stopWords are frequent English words that would match every tool.
var stopWords = map[string]bool{
	"a": true, "an": true, "and": true, "any": true, "are": true, "as": true, "at": true, "be": true, "by": true,
	"for": true, "from": true, "in": true, "into": true, "is": true, "it": true, "its": true, "of": true,
	"on": true, "one": true, "or": true, "such": true, "that": true, "the": true, "their": true, "this": true,
	"to": true, "use": true, "when": true, "which": true, "with": true,
}

// bounded cuts text to at most limit bytes, on a character boundary.
func bounded(text string, limit int) string {
	if len(text) <= limit {
		return text
	}
	for limit > 0 && !utf8.RuneStart(text[limit]) {
		limit--
	}
	return text[:limit]
}
