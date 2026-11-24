package agent

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/opensvc/ai-agent/internal/llm"
	"github.com/opensvc/ai-agent/internal/toolsearch"
)

const (
	// SearchToolName is the tool of the agent the model searches the MCP
	// catalog with, when the catalog is too large to be shown whole.
	SearchToolName = "search_tools"
	// DefaultSearchThreshold is the largest catalog shown whole: beyond, the
	// model sees the core tools and search_tools.
	DefaultSearchThreshold = 20
	// CoreToolMeta marks, in the _meta of an MCP tool, a tool always shown.
	CoreToolMeta = "opensvc/core"
	// DefaultSearchScoreRatio is the share of the best score of a search
	// under which a result is listed but not shown, an experimental value.
	DefaultSearchScoreRatio = 0.30

	maxSearchesPerTurn   = 3
	defaultSearchResults = 5
	maxSearchResults     = 10
	// closeSearchResults bounds the results a search adds beyond the limit
	// the model asked, those scoring at least closeScoreRatio of the best
	// one: a family of tools sharing their words, such as start_instance and
	// start_instance_standby, ranks close, and the one asked fell past the
	// fifth result in real searches of om3 and oc3. Measured on 34 searches
	// of the model on the catalogs of both backends, on 2026-10-23: every
	// tool asked shown, against 31 with 5 results, for about two more tools
	// shown per search.
	closeSearchResults = 5
	closeScoreRatio    = 0.5
	maxHistoryTools    = 8
	// maxHistoryShownTools bounds the tools a search of the recent turns
	// showed without the model calling them, apart from maxHistoryTools: a
	// search of the last turn does not push out the tools called before. It
	// carries all the tools one search may show, in the order the turn
	// listed them: the next turn starts with the tools of the last request
	// of that turn, which a provider then reads from its cache. Carrying 4
	// of them, the next turn shared only search_tools with it, in 5 tries
	// of 5 on the catalog of dev5, and wrote the whole prefix again.
	maxHistoryShownTools  = maxSearchResults + closeSearchResults
	maxSearchSummaryBytes = 240
	// maxModelTools bounds the tools of a suspended turn state.
	maxModelTools = 1024
)

// searchKeywordsGuidance tells the model how to write a search query: a
// whole question brings words that favor the wrong tools.
const searchKeywordsGuidance = "Use 3–8 English keywords describing the required capability. Omit greetings, full sentences, and specific resource identifiers."

// searchInstruction is added to the system prompt when the model sees a
// part of the catalog only. It does not tell how to show a tool found but
// not shown: the hint of a search result does, beside those tools. Said in
// the prompt and in the description of search_tools, it made the model
// search the exact name of the tools already shown, in 11 of 12 replays of
// a first turn, against none without it. A model that found no listed tool
// for a read or an action told the user that none existed, such as no tool
// reading the value of a key or the configuration of an object, when a
// search would have found it: searchBeforeDenying asks for that search.
const searchInstruction = "Only some tools are listed. When none fits, call " + SearchToolName +
	"; the tools it finds can be called from the next step on. " + searchBeforeDenying + " " + searchKeywordsGuidance + " " + searchBatchGuidance +
	" Call the tools a search shows directly; never search again for a tool already shown."

// searchBeforeDenying asks the model to search the tools before it says a
// capability is missing: the tools not listed exist all the same. It names
// search_tools: a model searched the documentation and the code instead,
// six to eight times, never the tools, then said it found nothing.
const searchBeforeDenying = "A capability no listed tool covers may still exist: never tell the user that something cannot be read or done, or that no tool gives it, before " + SearchToolName + " found nothing for it; a search of the documentation or of the code finds no tool."

// searchBatchGuidance asks the model to request the tools a search found
// together, as the backend prompts do for independent calls.
const searchBatchGuidance = "After a search, request the independent read-only tools you need together in the next step, when all their arguments are already known. Wait for prerequisite results before making dependent calls."

// searchToolSchema describes the bounds of the query. toolsearch counts
// toolsearch.MaxQueryBytes in UTF-8 bytes, while maxLength counts characters:
// maxLength is only the loosest bound a schema can state, and the
// description gives the exact one. A query without a searchable word is
// refused, and only its first toolsearch.MaxQueryTerms distinct terms are
// searched.
var searchToolSchema = json.RawMessage(fmt.Sprintf(`{"type":"object","properties":{`+
	`"query":{"type":"string","minLength":1,"maxLength":%d,"description":%q},`+
	`"limit":{"type":"integer","minimum":1,"maximum":%d,"description":"optional number of tools to return; defaults to %d"}},`+
	`"required":["query"],"additionalProperties":false}`,
	toolsearch.MaxQueryBytes, searchKeywordsGuidance+" At most "+strconv.Itoa(toolsearch.MaxQueryBytes)+" UTF-8 bytes. Example: node hardware inventory.",
	maxSearchResults, defaultSearchResults))

var searchTool = llm.Tool{
	Name: SearchToolName,
	Description: "Search the tools of the backend when none of the listed tools fits the question. " + searchKeywordsGuidance +
		" Returns the best matching tools with a summary; those shown can be called from the next step on.",
	InputSchema: searchToolSchema,
}

// toolSelection holds the tools the model sees during a turn: the whole
// catalog when it is small, else search_tools and an active set made of the
// core tools, the tools of the recent history and the tools found.
type toolSelection struct {
	search   bool
	catalog  map[string]llm.Tool
	all      []llm.Tool
	index    *toolsearch.Index
	active   []string
	searches int
	// bytes is the size of the tools shown, bounded by
	// maxModelToolCatalogBytes.
	bytes int
	// shown are the tools sent to the model during the turn, in order.
	shown []string
	// found are the tools the searches of the turn activated, the most
	// recent search first, the tools of one search in its order.
	found []string
	// batch is the place in found of the next tool the current search or
	// skill shows.
	batch int
	// carried are the tools a search of the recent turns showed, the most
	// recent first, shown before the core tools as found.
	carried []string
	// core are the tools never removed to make room for the tools found.
	core map[string]bool
	// scoreRatio is the share of the best score under which a result is
	// not shown; 0 shows them all.
	scoreRatio float64
}

// indexCache keeps the index of the last catalog, by its hash.
type indexCache struct {
	mu    sync.Mutex
	hash  string
	index *toolsearch.Index
}

func (c *indexCache) get(hash string, tools []*mcp.Tool) (*toolsearch.Index, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.index != nil && c.hash == hash {
		return c.index, nil
	}
	index, err := toolsearch.New(toolsearch.FromMCP(tools))
	if err != nil {
		return nil, fmt.Errorf("index MCP tools: %w", err)
	}
	c.hash, c.index = hash, index
	return index, nil
}

func (a *Agent) selectTools(mcpTools []*mcp.Tool, converted []llm.Tool, catalogHash string, history []llm.Message, state *SuspendedTurn) (*toolSelection, error) {
	selection := &toolSelection{all: converted, catalog: make(map[string]llm.Tool, len(converted))}
	catalogBytes := 0
	for _, tool := range converted {
		selection.catalog[tool.Name] = tool
		catalogBytes += toolBytes(tool)
	}
	threshold := a.config.SearchThreshold
	if threshold == 0 {
		threshold = DefaultSearchThreshold
	}
	// A turn suspended while searching keeps searching when it resumes, so
	// that the search calls it planned still run in the agent. A catalog
	// too large for one model request is searched whatever the threshold.
	search := state.ToolSearch || catalogBytes > maxModelToolCatalogBytes || (threshold > 0 && len(converted) > threshold)
	if !search {
		return selection, nil
	}
	if _, exists := selection.catalog[SearchToolName]; exists {
		return nil, fmt.Errorf("MCP tool %q conflicts with the tool search of the agent", SearchToolName)
	}
	index, err := a.indexes.get(catalogHash, mcpTools)
	if err != nil {
		return nil, err
	}
	selection.search, selection.index, selection.searches = true, index, state.Searches
	selection.scoreRatio = a.config.SearchScoreRatio
	if selection.scoreRatio == 0 {
		selection.scoreRatio = DefaultSearchScoreRatio
	}
	if selection.scoreRatio < 0 {
		selection.scoreRatio = 0
	}
	selection.core = map[string]bool{}
	selection.bytes = toolBytes(searchTool)
	state.ToolSearch = true
	for _, tool := range mcpTools {
		if core, _ := tool.Meta[CoreToolMeta].(bool); core && selection.add(tool.Name) {
			selection.core[tool.Name] = true
		}
	}
	// The tools of the last turns, the most recent first, for the questions
	// that follow them: those called, and apart those a search showed. A
	// question such as "give me the result" after an action follows it with
	// a tool the search of the action showed but the model did not call:
	// listed before the core tools, as the tools found in the turn, the
	// model called it in 12 replays of 12; listed after them, it searched
	// again in 4, and without it in 9.
	results := map[string]llm.ToolResult{}
	for _, message := range history {
		for _, result := range message.ToolResults {
			results[result.CallID] = result
		}
	}
	recent := 0
	for i := len(history) - 1; i >= 0; i-- {
		for _, call := range slices.Backward(history[i].ToolCalls) {
			if call.Name != SearchToolName {
				if recent < maxHistoryTools && selection.add(call.Name) {
					recent++
				}
				continue
			}
			for _, name := range searchShown(results[call.ID]) {
				if len(selection.carried) >= maxHistoryShownTools || selection.core[name] || slices.Contains(selection.carried, name) {
					continue
				}
				if selection.add(name) || slices.Contains(selection.active, name) {
					selection.carried = append(selection.carried, name)
				}
			}
		}
	}
	for _, name := range state.ActiveTools {
		selection.add(name)
	}
	// The tools found keep their order: the resumed turn sends the tools of
	// the request it waited after, which a provider reads from its cache.
	for _, name := range state.FoundTools {
		if slices.Contains(selection.active, name) && !slices.Contains(selection.found, name) {
			selection.found = append(selection.found, name)
		}
	}
	return selection, nil
}

// searchShown gives the tools a search_tools result showed, in its order: none
// for a refused search, or a result the history compaction reduced. It relies
// on the compaction keeping whole the searches of the last turn and of the
// last turn that searched, which it reads by the name of their calls: the
// compaction reduces the former searches as any other text.
// TestCompactHistoryKeepsASearchResultWhole and
// TestCompactHistoryReadsTheToolOfEachTurn hold it.
func searchShown(result llm.ToolResult) []string {
	if result.IsError {
		return nil
	}
	var content struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
	}
	if json.Unmarshal(result.Content, &content) != nil || len(content.Content) != 1 {
		return nil
	}
	var reply searchReply
	if json.Unmarshal([]byte(content.Content[0].Text), &reply) != nil {
		return nil
	}
	var names []string
	for _, match := range reply.Tools {
		if match.Shown {
			names = append(names, match.Name)
		}
	}
	return names
}

// add activates a catalog tool, and reports whether it was not active yet. A
// tool beyond the size of a model request is not shown.
func (s *toolSelection) add(name string) bool {
	tool, ok := s.catalog[name]
	if !ok || slices.Contains(s.active, name) || s.bytes+toolBytes(tool) > maxModelToolCatalogBytes {
		return false
	}
	s.active = append(s.active, name)
	s.bytes += toolBytes(tool)
	return true
}

// show activates a tool a search found. When the tools shown leave no room,
// the oldest tools that are not core tools are removed, the tools of the
// history first, to make room for it. It reports whether the tool is shown.
func (s *toolSelection) show(name string) bool {
	if slices.Contains(s.active, name) {
		return true
	}
	tool, ok := s.catalog[name]
	if !ok {
		return false
	}
	for s.bytes+toolBytes(tool) > maxModelToolCatalogBytes {
		oldest := slices.IndexFunc(s.active, func(active string) bool { return !s.core[active] })
		if oldest < 0 {
			return false
		}
		s.bytes -= toolBytes(s.catalog[s.active[oldest]])
		if index := slices.Index(s.found, s.active[oldest]); index >= 0 && index < s.batch {
			s.batch--
		}
		s.found = slices.DeleteFunc(s.found, func(found string) bool { return found == s.active[oldest] })
		s.carried = slices.DeleteFunc(s.carried, func(carried string) bool { return carried == s.active[oldest] })
		s.active = slices.Delete(s.active, oldest, oldest+1)
	}
	if !s.add(name) {
		return false
	}
	s.found = slices.Insert(s.found, s.batch, name)
	s.batch++
	return true
}

// startBatch starts the tools one search or skill shows: they go before
// those of the former ones, in the order shown.
func (s *toolSelection) startBatch() {
	s.batch = 0
}

// tools are the tools of the next model request: search_tools, the tools
// the searches of the turn found, the most recent search first, the tools
// of one search in its order, then those of the recent searches, the core
// tools and those of the history. A tool found right after search_tools is called:
// listed after the core tools, the model searched again for a tool it had
// been shown in 12 of 24 replays, against none.
func (s *toolSelection) tools() []llm.Tool {
	tools := s.all
	if s.search {
		tools = make([]llm.Tool, 0, len(s.active)+1)
		// Past the searches of a turn, search_tools is no longer offered: the
		// model answers with the tools shown instead of searching in a loop.
		if s.searches < maxSearchesPerTurn {
			tools = append(tools, searchTool)
		}
		for _, name := range s.found {
			tools = append(tools, s.catalog[name])
		}
		for _, name := range s.carried {
			if !slices.Contains(s.found, name) && slices.Contains(s.active, name) {
				tools = append(tools, s.catalog[name])
			}
		}
		for _, name := range s.active {
			if !slices.Contains(s.found, name) && !slices.Contains(s.carried, name) {
				tools = append(tools, s.catalog[name])
			}
		}
	}
	for _, tool := range tools {
		if !slices.Contains(s.shown, tool.Name) {
			s.shown = append(s.shown, tool.Name)
		}
	}
	return tools
}

// known tells whether the model may call a tool: any tool of the catalog,
// shown or not, and search_tools when the search is on.
func (s *toolSelection) known(name string) bool {
	_, ok := s.catalog[name]
	return ok || (s.search && name == SearchToolName)
}

type searchArguments struct {
	Query string `json:"query"`
	Limit int    `json:"limit"`
}

type searchMatch struct {
	Name    string `json:"name"`
	Title   string `json:"title,omitempty"`
	Summary string `json:"summary"`
	// Score is the BM25 score of a search, absent for a tool asked by its
	// exact name.
	Score *float64 `json:"score,omitempty"`
	// Exact is set for a tool asked by its exact name, without a score.
	Exact bool `json:"exact,omitempty"`
	// Shown is false for a tool found but not shown, for Reason.
	Shown  bool   `json:"shown"`
	Reason string `json:"reason,omitempty"`
}

// The reasons a tool found is not shown.
const (
	// notShownLowScore: it scored under the ratio of the best result.
	notShownLowScore = "low_score"
	// notShownBudget: its definition does not fit in a model request.
	notShownBudget = "budget"
)

// run executes a search_tools call: the tools found become active. A
// refused search is a tool error for the model, never a failed turn.
func (s *toolSelection) run(call llm.ToolCall, titles map[string]string) llm.ToolResult {
	fail := func(message string) llm.ToolResult {
		return llm.ToolResult{CallID: call.ID, Content: textToolResult(message), IsError: true}
	}
	if s.searches >= maxSearchesPerTurn {
		return fail(fmt.Sprintf("The tool search is limited to %d searches per question. Answer with the tools available.", maxSearchesPerTurn))
	}
	s.startBatch()
	var arguments searchArguments
	decoder := json.NewDecoder(strings.NewReader(string(call.Arguments)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&arguments); err != nil {
		return fail("search_tools takes a query and an optional limit. " + searchKeywordsGuidance)
	}
	if arguments.Limit == 0 {
		arguments.Limit = defaultSearchResults
	}
	if arguments.Limit < 1 || arguments.Limit > maxSearchResults {
		return fail(fmt.Sprintf("The limit of search_tools must be between 1 and %d.", maxSearchResults))
	}
	// A query naming a tool of the catalog exactly asks for that tool alone,
	// whatever its rank: the way to show a tool a former search found but
	// did not show. It is subject to the budget, never to the score ratio.
	if exact := strings.TrimSpace(arguments.Query); len(arguments.Query) <= toolsearch.MaxQueryBytes {
		if tool, ok := s.catalog[exact]; ok {
			s.searches++
			s.show(exact)
			match := searchMatch{Name: exact, Title: titles[exact], Summary: summary(tool.Description), Exact: true, Shown: slices.Contains(s.active, exact)}
			var hint string
			if !match.Shown {
				match.Reason = notShownBudget
				hint = "The tool cannot be shown: its definition exceeds the size of a model request beside the core tools. Answer with the tools shown."
			}
			matches := []searchMatch{match}
			response := searchReply{Tools: matches, Next: s.next(matches), Hint: hint}
			text, _ := json.Marshal(response)
			return llm.ToolResult{CallID: call.ID, Content: textToolResult(string(text))}
		}
	}
	results, err := s.index.Search(arguments.Query, arguments.Limit+closeSearchResults)
	if err != nil {
		return fail("The query of search_tools has no searchable keyword, or exceeds " + strconv.Itoa(toolsearch.MaxQueryBytes) + " bytes. " + searchKeywordsGuidance)
	}
	results = closeResults(results, arguments.Limit)
	// Only a valid search counts, whether it finds tools or not.
	s.searches++
	// A result scoring under scoreRatio of the best one is not shown: BM25
	// scores compare within one search only. The raw scores are compared,
	// and a tool already shown stays shown.
	best := 0.0
	for _, result := range results {
		best = max(best, result.Score)
	}
	lowScore := map[string]bool{}
	for _, result := range results {
		switch {
		case slices.Contains(s.active, result.Name):
		case s.scoreRatio > 0 && result.Score < s.scoreRatio*best:
			lowScore[result.Name] = true
		default:
			s.show(result.Name)
		}
	}
	// shown is read from the final active set: making room for a tool may
	// have removed another one.
	matches := make([]searchMatch, 0, len(results))
	reasons := map[string]bool{}
	for _, result := range results {
		score := math.Round(result.Score*100) / 100
		match := searchMatch{
			Name: result.Name, Title: titles[result.Name], Summary: summary(s.catalog[result.Name].Description),
			Score: &score, Shown: slices.Contains(s.active, result.Name),
		}
		if !match.Shown {
			match.Reason = notShownBudget
			if lowScore[result.Name] {
				match.Reason = notShownLowScore
			}
			reasons[match.Reason] = true
		}
		matches = append(matches, match)
	}
	response := searchReply{Tools: matches, Next: s.next(matches)}
	var hints []string
	if len(matches) == 0 {
		hints = append(hints, "No tool matches these keywords. Try other English words, or answer with the tools available.")
	}
	if reasons[notShownLowScore] {
		hints = append(hints, "The tools with reason "+notShownLowScore+" matched these keywords weakly and are not shown. Only when no shown tool fits, search the exact name of one to show it.")
	}
	if reasons[notShownBudget] {
		hints = append(hints, "The tools with reason "+notShownBudget+" cannot be shown: their definitions exceed the size of a model request. Answer with the tools shown.")
	}
	response.Hint = strings.Join(hints, " ")
	text, _ := json.Marshal(response)
	return llm.ToolResult{CallID: call.ID, Content: textToolResult(string(text))}
}

// closeResults keeps the first limit results, then those after them that
// score at least closeScoreRatio of the best one, the results being sorted
// by score.
func closeResults(results []toolsearch.Result, limit int) []toolsearch.Result {
	if len(results) <= limit {
		return results
	}
	best := 0.0
	for _, result := range results {
		best = max(best, result.Score)
	}
	kept := results[:limit]
	for _, result := range results[limit:] {
		if result.Score < closeScoreRatio*best {
			break
		}
		kept = append(kept, result)
	}
	return kept
}

// searchReply is the result of search_tools for the model. Its fields
// keep this order: the tools found, then what to do next, then the hints
// about the tools not shown, so that a hint is not read as the next step.
type searchReply struct {
	Tools []searchMatch `json:"tools"`
	Next  string        `json:"next,omitempty"`
	Hint  string        `json:"hint,omitempty"`
}

// next tells the model to call the tools a search shows, rather than to
// search for them again, and when the searches of the turn are spent.
func (s *toolSelection) next(matches []searchMatch) string {
	var shown []string
	for _, match := range matches {
		if match.Shown {
			shown = append(shown, match.Name)
		}
	}
	var next string
	if len(shown) > 0 {
		next = "The tools shown are callable now: call " + strings.Join(shown, ", ") + " directly in your next step if they fit, without searching for them again."
	}
	if s.searches >= maxSearchesPerTurn {
		next = strings.TrimSpace(next + " This was the last search of this question: answer with the tools shown.")
	}
	return next
}

// summary is the first sentence of a description, bounded.
func summary(description string) string {
	if end := strings.Index(description, ". "); end >= 0 {
		description = description[:end+1]
	}
	if len(description) > maxSearchSummaryBytes {
		cut := maxSearchSummaryBytes
		for cut > 0 && !utf8.RuneStart(description[cut]) {
			cut--
		}
		description = description[:cut] + "…"
	}
	return description
}

// textToolResult is a tool result holding one text, as MCP returns it.
func textToolResult(text string) json.RawMessage {
	data, _ := json.Marshal(map[string]any{"content": []map[string]string{{"type": "text", "text": text}}})
	return data
}
