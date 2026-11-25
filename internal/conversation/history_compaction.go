package conversation

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"sort"
	"strings"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/llm"
)

// HistoryCompactionPolicy names the compaction of the history below; it is
// recorded with each run, so that a change of policy is visible in the
// feedback.
const HistoryCompactionPolicy = "v3"

// HistoryCompaction bounds the tool results of the earlier turns a model
// request carries. The stored messages are never changed: the model
// receives a compact copy, whose omissions are listed in a
// _history_compaction field of each result.
type HistoryCompaction struct {
	// MaxResultBytes bounds one compacted result, its marker included.
	MaxResultBytes int
	// MaxHistoryResultBytes bounds all the tool results of the history.
	MaxHistoryResultBytes int
	// ArrayItems is the number of items first kept of a collection, then
	// reduced until a result fits.
	ArrayItems int
	// MaxStringBytes is the longest string kept; a longer one is omitted,
	// never cut: a generic compaction cannot tell a text from an identifier.
	MaxStringBytes int
	// MaxOmissions bounds the omissions listed in a marker.
	MaxOmissions int
	// MaxDepth bounds the nesting kept.
	MaxDepth int
}

// DefaultHistoryCompaction is the policy of the agent.
var DefaultHistoryCompaction = HistoryCompaction{
	MaxResultBytes:        2 << 10,
	MaxHistoryResultBytes: 16 << 10,
	ArrayItems:            8,
	MaxStringBytes:        256,
	MaxOmissions:          16,
	MaxDepth:              6,
}

// HistoryStats describes the history a turn sent to the model.
type HistoryStats struct {
	Policy                string `json:"policy"`
	MaxResultBytes        int    `json:"max_result_bytes"`
	MaxHistoryResultBytes int    `json:"max_history_result_bytes"`
	// LoadedBytes, CompactedBytes and SentBytes are the sizes of the
	// history loaded, compacted, and sent once bounded.
	LoadedBytes    int `json:"loaded_bytes"`
	CompactedBytes int `json:"compacted_bytes"`
	SentBytes      int `json:"sent_bytes"`
	// CompactedResults counts the tool results compacted, RemovedTurns the
	// earlier turns left out.
	CompactedResults int `json:"compacted_results"`
	RemovedTurns     int `json:"removed_turns"`
}

// historyMarker is the _history_compaction field of a compacted result.
type historyMarker struct {
	Policy           string     `json:"policy"`
	Partial          bool       `json:"partial"`
	OriginalBytes    int        `json:"original_bytes"`
	Omissions        []omission `json:"omissions"`
	OmissionsOmitted int        `json:"omissions_omitted,omitempty"`
}

// omission is a part of a result left out: at path, a JSON pointer into the
// result, the kind of omission, the items kept and the items omitted, or
// the bytes omitted for a value.
type omission struct {
	Path    string `json:"path"`
	Kind    string `json:"kind"`
	Kept    int    `json:"kept"`
	Omitted int    `json:"omitted"`
}

// Kinds of omissions.
const (
	omittedArrayTail     = "array_tail"     // the items after the first ones
	omittedObjectEntries = "object_entries" // the entries holding collections, after the first ones
	omittedLongString    = "long_string"    // a string longer than MaxStringBytes, in bytes
	omittedDepth         = "depth"          // a value nested deeper than MaxDepth, in bytes
	omittedContent       = "content"        // the text contents beside the structured content
	omittedValue         = "value"          // the whole structured content, in bytes
)

// withDefaults completes a policy: a field left at zero takes its default
// value.
func (c HistoryCompaction) withDefaults() HistoryCompaction {
	defaults := DefaultHistoryCompaction
	for _, field := range []struct{ value, fallback *int }{
		{&c.MaxResultBytes, &defaults.MaxResultBytes}, {&c.MaxHistoryResultBytes, &defaults.MaxHistoryResultBytes},
		{&c.ArrayItems, &defaults.ArrayItems}, {&c.MaxStringBytes, &defaults.MaxStringBytes},
		{&c.MaxOmissions, &defaults.MaxOmissions}, {&c.MaxDepth, &defaults.MaxDepth},
	} {
		if *field.value == 0 {
			*field.value = *field.fallback
		}
	}
	return c
}

// minResultBytes is the smallest budget of a result: a marker alone fits.
const minResultBytes = 512

// validate checks a completed policy.
func (c HistoryCompaction) validate() error {
	if c.MaxResultBytes < minResultBytes || c.MaxHistoryResultBytes < c.MaxResultBytes || c.ArrayItems < 1 ||
		c.MaxStringBytes < 1 || c.MaxOmissions < 1 || c.MaxDepth < 1 {
		return fmt.Errorf("%w: history compaction needs a result budget of at least %d bytes, a history budget not below it, and positive limits", ErrInvalid, minResultBytes)
	}
	return nil
}

// searchTool is the tool whose results the agent reads back: searchShown
// gives the tools a search of the former turns showed. Its results are kept
// whole in the last turn and in the last turn that searched, a refused
// search not counted, those the agent reads them from; elsewhere they are
// whole or a marker, never compacted on their own, so that they stay stable
// from a turn to the next.
const searchTool = agent.SearchToolName

// compactHistory returns a copy of history whose tool results fit the
// policy, what it did, and the indexes of the messages holding a compacted
// result, one per result. Small results, errors and the searches the agent
// reads back are kept as they are; the others are compacted turn by turn, so
// that the history a turn sends begins as the one the turn before sent, see
// compactTurns. When the results still exceed MaxHistoryResultBytes, the
// oldest turn is left out whole and the turns kept are compacted again from
// their original results, so that a turn left out frees its room for the
// others. The last turn is never left out, and over the budget alone, the
// turns before it are left out only while their own results exceed the
// budget: the model never loses the turn the user answers, nor the earlier
// ones for nothing. Calls, arguments and call IDs are kept.
func compactHistory(history []llm.Message, policy HistoryCompaction) ([]llm.Message, HistoryStats, []int) {
	policy = policy.withDefaults()
	stats := HistoryStats{Policy: HistoryCompactionPolicy, MaxResultBytes: policy.MaxResultBytes, MaxHistoryResultBytes: policy.MaxHistoryResultBytes}
	last := 0
	for i, message := range history {
		if message.Role == llm.RoleUser {
			last = i
		}
	}
	floor := -1
	start := 0
	forms := newResultForms(policy, history)
	for {
		messages, total, compacted := policy.compactTurns(history[start:], start, forms)
		if total <= policy.MaxHistoryResultBytes || start >= last {
			stats.CompactedResults = len(compacted)
			return messages, stats, compacted
		}
		// The last turn over the budget alone: leaving the turns before it
		// out cannot bring the results under it.
		if floor < 0 {
			_, floor, _ = policy.compactTurns(history[last:], last, forms)
		}
		if floor > policy.MaxHistoryResultBytes && resultBytes(messages[:last-start]) <= policy.MaxHistoryResultBytes {
			stats.CompactedResults = len(compacted)
			return messages, stats, compacted
		}
		// The oldest turn is left out, and the others compacted again.
		next := start + 1
		for next < len(history) && history[next].Role != llm.RoleUser {
			next++
		}
		start = next
		stats.RemovedTurns++
	}
}

// resultBytes is the size of the tool results of messages.
func resultBytes(messages []llm.Message) int {
	size := 0
	for _, message := range messages {
		for _, result := range message.ToolResults {
			size += len(result.Content)
		}
	}
	return size
}

// resultForms are the forms of the results of a history, each computed once
// for a call of compactHistory, at its first need, whatever turns are left
// out: the result compacted on its own, and its marker.
type resultForms struct {
	policy  HistoryCompaction
	history []llm.Message
	alone   map[[2]int]json.RawMessage
	markers map[[2]int]json.RawMessage
	none    map[[2]int]bool
}

func newResultForms(policy HistoryCompaction, history []llm.Message) *resultForms {
	return &resultForms{policy: policy, history: history,
		alone: map[[2]int]json.RawMessage{}, markers: map[[2]int]json.RawMessage{}, none: map[[2]int]bool{}}
}

// result gives a result of the history.
func (f *resultForms) result(at [2]int) llm.ToolResult {
	return f.history[at[0]].ToolResults[at[1]]
}

// compacted gives a result compacted on its own to MaxResultBytes, or as it
// is when it fits, is an error or is a search. A history compacted begins at
// a message of the user, so that the calls of a result are always read
// before it: a result is a search, or not, whatever turns are left out.
func (f *resultForms) compacted(at [2]int, search bool) json.RawMessage {
	if form, ok := f.alone[at]; ok {
		return form
	}
	result := f.result(at)
	form := result.Content
	if !result.IsError && !search && len(form) > f.policy.MaxResultBytes {
		if compacted, ok := f.policy.compactResult(form, f.policy.MaxResultBytes); ok && len(compacted) < len(form) {
			form = compacted
		}
	}
	f.alone[at] = form
	return form
}

// marker gives the marker of a result, and reports false for a result the
// compaction does not handle, such as one of an image.
func (f *resultForms) marker(at [2]int) (json.RawMessage, bool) {
	if marker, ok := f.markers[at]; ok {
		return marker, true
	}
	if f.none[at] {
		return nil, false
	}
	content := f.result(at).Content
	marker, ok := f.policy.compactResult(content, 0)
	if !ok {
		marker, ok = f.policy.textMarker(content)
	}
	if !ok {
		f.none[at] = true
		return nil, false
	}
	f.markers[at] = marker
	return marker, true
}

// compactTurns compacts the results of history, which begins at offset in
// the history of forms, turn by turn, each decision taken on that turn and
// the turns before it, never on the next ones, so that the compaction of
// the history of a turn begins as the compaction of the history of the
// turn before, which a provider then reads from its cache. Each result is
// compacted on its own to MaxResultBytes. When the results of the turns so
// far exceed MaxHistoryResultBytes:
//   - the results of the turn itself are reduced to a marker, the largest
//     first, the first of equal sizes, while they exceed it alone;
//   - then, while all exceed it, those of the turns before, those over the
//     target alone first, then the oldest first, down to half of it, or to
//     what the turn leaves of it when less, which leaves room for the next
//     turns;
//   - then, when the turns before hold more than the turn leaves them, such
//     as errors or a search kept, the turn itself again, down to what they
//     leave it.
//
// A result reduced so stays a marker, and a marker is never put in place of
// a smaller result. Errors, and the searches of the turn and of the last
// turn that searched, at each step, are kept. It returns the copy, the size
// of its results and the indexes of the messages holding a compacted result,
// one per result.
func (c HistoryCompaction) compactTurns(history []llm.Message, offset int, forms *resultForms) ([]llm.Message, int, []int) {
	type ref struct{ message, result int }
	var refs []ref
	turnOf := map[ref]int{}
	search := map[ref]bool{}
	turn := -1
	// The tool of a result is read from the calls before it: call IDs
	// repeat from a turn to the next with some providers.
	var calls map[string]string
	for i, message := range history {
		if message.Role == llm.RoleUser {
			turn++
		}
		if len(message.ToolCalls) > 0 {
			calls = map[string]string{}
			for _, call := range message.ToolCalls {
				calls[call.ID] = call.Name
			}
		}
		for j, result := range message.ToolResults {
			r := ref{i, j}
			refs = append(refs, r)
			turnOf[r] = turn
			search[r] = calls[result.CallID] == searchTool
		}
	}
	at := func(r ref) [2]int { return [2]int{offset + r.message, r.result} }
	isError := func(r ref) bool { return history[r.message].ToolResults[r.result].IsError }
	marked := map[ref]bool{}
	size := func(r ref) int {
		if marked[r] {
			marker, _ := forms.marker(at(r))
			return len(marker)
		}
		return len(forms.compacted(at(r), search[r]))
	}
	low := c.MaxHistoryResultBytes / 2
	for step := 0; step <= turn; step++ {
		// A refused search shows no tool: the agent reads back the last
		// one that did.
		searched, before, own := -1, 0, 0
		for _, r := range refs {
			if turnOf[r] > step {
				break
			}
			if turnOf[r] < step {
				before += size(r)
			} else {
				own += size(r)
			}
			if search[r] && !isError(r) {
				searched = turnOf[r]
			}
		}
		if before+own <= c.MaxHistoryResultBytes {
			continue
		}
		reducible := func(r ref) (json.RawMessage, bool) {
			if marked[r] || isError(r) || (search[r] && turnOf[r] == searched) {
				return nil, false
			}
			marker, ok := forms.marker(at(r))
			return marker, ok && len(marker) < size(r)
		}
		// reduceTurn reduces the results of the turn itself, the largest
		// first, until they hold limit: a small result stays whole when a
		// large one makes room.
		reduceTurn := func(limit int) {
			for own > limit {
				var largest ref
				var marker json.RawMessage
				found := false
				for _, r := range refs {
					if turnOf[r] != step {
						continue
					}
					if m, ok := reducible(r); ok && (!found || size(r) > size(largest)) {
						largest, marker, found = r, m, true
					}
				}
				if !found {
					return
				}
				own -= size(largest) - len(marker)
				marked[largest] = true
			}
		}
		// The turn itself down to the budget; then, when the turns before
		// and it still exceed it, the turns before, the oldest first, down to
		// half the budget, or to what the turn leaves of it, which leaves room
		// for the next turns; then the turn itself again, when the turns
		// before cannot make room enough.
		reduceTurn(c.MaxHistoryResultBytes)
		if before+own > c.MaxHistoryResultBytes {
			target := max(min(low, c.MaxHistoryResultBytes-own), 0)
			// A result over the target alone first, whatever its age: the
			// older ones would be reduced for nothing before it.
			for _, oversized := range []bool{true, false} {
				for _, r := range refs {
					if turnOf[r] >= step || before <= target {
						break
					}
					if oversized && size(r) <= target {
						continue
					}
					if marker, ok := reducible(r); ok {
						before -= size(r) - len(marker)
						marked[r] = true
					}
				}
			}
			reduceTurn(c.MaxHistoryResultBytes - before)
		}
	}
	messages := make([]llm.Message, len(history))
	for i, message := range history {
		message.ToolCalls = slices.Clone(message.ToolCalls)
		message.ToolResults = slices.Clone(message.ToolResults)
		messages[i] = message
	}
	var indexes []int
	total := 0
	for _, r := range refs {
		content := forms.compacted(at(r), search[r])
		if marked[r] {
			content, _ = forms.marker(at(r))
		}
		if !bytes.Equal(content, history[r.message].ToolResults[r.result].Content) {
			messages[r.message].ToolResults[r.result].Content = content
			indexes = append(indexes, r.message)
		}
		total += len(content)
	}
	return messages, total, indexes
}

// compactResult compacts a structured result to at most budget bytes, its
// marker included, reducing its collections step by step, or to a marker
// alone. It reports false for a result it does not handle: one without
// structured content, or with contents other than text.
func (c HistoryCompaction) compactResult(raw json.RawMessage, budget int) (json.RawMessage, bool) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var envelope map[string]any
	if err := decoder.Decode(&envelope); err != nil {
		return nil, false
	}
	structured, ok := envelope["structuredContent"]
	if !ok || structured == nil {
		return nil, false
	}
	// The contents beside the structured content: the text copy the MCP SDK
	// adds is dropped, as the agent drops it since C0. Distinct texts, such
	// as a warning, keep the prefix of those small enough to be kept whole,
	// so that their indexes stay true, as the items of an array.
	contents, ok := textContentsOf(envelope)
	if !ok {
		return nil, false
	}
	var base []omission
	var kept []any
	if len(contents) > 0 && duplicatedText(contents, structured) {
		base = append(base, omission{Path: "/content", Kind: omittedContent, Omitted: len(contents)})
	} else if len(contents) > 0 {
		for i, content := range contents {
			if data, _ := json.Marshal(content); len(data) > c.MaxStringBytes {
				base = append(base, omission{Path: fmt.Sprintf("/content/%d", i), Kind: omittedValue, Omitted: len(data)})
				break
			}
			kept = append(kept, content)
		}
		if len(kept) < len(contents) {
			base = append(base, omission{Path: "/content", Kind: omittedArrayTail, Kept: len(kept), Omitted: len(contents) - len(kept)})
		}
	}
	encode := func(contents []any, value any, omissions []omission, withValue bool) json.RawMessage {
		marker := historyMarker{Policy: HistoryCompactionPolicy, Partial: true, OriginalBytes: len(raw), Omissions: omissions}
		if len(marker.Omissions) > c.MaxOmissions {
			marker.OmissionsOmitted = len(marker.Omissions) - c.MaxOmissions
			marker.Omissions = marker.Omissions[:c.MaxOmissions]
		}
		if contents == nil {
			contents = []any{}
		}
		out := map[string]any{"content": contents, "_history_compaction": marker}
		if withValue {
			out["structuredContent"] = value
		}
		data, _ := json.Marshal(out)
		return data
	}
	// The whole result fits the budget: contents, structured content and
	// marker. The texts kept are dropped before the structured content.
	attempts := []struct {
		contents []any
		base     []omission
	}{{kept, base}}
	if len(kept) > 0 {
		attempts = append(attempts, struct {
			contents []any
			base     []omission
		}{nil, []omission{{Path: "/content", Kind: omittedArrayTail, Omitted: len(contents)}}})
	}
	// Each width of the collections is tried at every depth, from the
	// deepest: a wide result, such as the status of a cluster of a hundred
	// objects, keeps its first items with their scalars, their states, when
	// their nested details do not fit, rather than nothing. Without items,
	// the last width, the depth no longer matters.
	for _, attempt := range attempts {
		for _, items := range c.itemSteps() {
			depths := []int{c.MaxDepth}
			if items > 0 {
				depths = depths[:0]
				for depth := c.MaxDepth; depth >= 1; depth-- {
					depths = append(depths, depth)
				}
			}
			for _, depth := range depths {
				shallow := c
				shallow.MaxDepth = depth
				omissions := slices.Clone(attempt.base)
				value := shallow.compactValue(structured, "/structuredContent", items, 0, &omissions)
				if data := encode(attempt.contents, value, omissions, true); len(data) <= budget {
					return data, true
				}
			}
		}
	}
	// The last resort is a marker alone, of two omissions: it fits the
	// smallest budget of a result.
	whole, _ := json.Marshal(structured)
	final := []omission{{Path: "/structuredContent", Kind: omittedValue, Omitted: len(whole)}}
	if len(contents) > 0 {
		final = append([]omission{{Path: "/content", Kind: omittedArrayTail, Omitted: len(contents)}}, final...)
	}
	return encode(nil, nil, final, false), true
}

// textMarker replaces a result made of text contents alone, without
// structured content, by a marker of their omission. It reports false for
// another result: a structured one, or one with contents other than text.
func (c HistoryCompaction) textMarker(raw json.RawMessage) (json.RawMessage, bool) {
	var envelope map[string]any
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return nil, false
	}
	if structured, ok := envelope["structuredContent"]; ok && structured != nil {
		return nil, false
	}
	contents, ok := textContentsOf(envelope)
	if !ok || len(contents) == 0 {
		return nil, false
	}
	marker := historyMarker{Policy: HistoryCompactionPolicy, Partial: true, OriginalBytes: len(raw),
		Omissions: []omission{{Path: "/content", Kind: omittedArrayTail, Omitted: len(contents)}}}
	data, _ := json.Marshal(map[string]any{"content": []any{}, "_history_compaction": marker})
	return data, true
}

// textContentsOf gives the contents of a result, and reports false when
// one is not a text: the compaction keeps such a result as it is.
func textContentsOf(envelope map[string]any) ([]any, bool) {
	contents, _ := envelope["content"].([]any)
	for _, content := range contents {
		if item, ok := content.(map[string]any); !ok || item["type"] != "text" {
			return nil, false
		}
	}
	return contents, true
}

// itemSteps are the numbers of items kept per collection, tried in turn.
func (c HistoryCompaction) itemSteps() []int {
	steps := []int{}
	for items := c.ArrayItems; items > 0; items /= 2 {
		steps = append(steps, items)
	}
	return append(steps, 0)
}

// duplicatedText tells contents made of one text holding the structured
// content, without annotations or metadata, as the MCP SDK adds it: the
// conditions under which the agent sends a structured result once.
func duplicatedText(contents []any, structured any) bool {
	if len(contents) != 1 {
		return false
	}
	item, ok := contents[0].(map[string]any)
	if !ok || item["type"] != "text" || len(item) != 2 {
		return false
	}
	text, ok := item["text"].(string)
	if !ok {
		return false
	}
	decoder := json.NewDecoder(strings.NewReader(text))
	decoder.UseNumber()
	var value any
	if decoder.Decode(&value) != nil {
		return false
	}
	// Only the end of the text may follow the value: a text with anything
	// after it, such as a warning, is distinct.
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return false
	}
	fromText, errText := json.Marshal(value)
	fromStructured, errStructured := json.Marshal(structured)
	return errText == nil && errStructured == nil && bytes.Equal(fromText, fromStructured)
}

// largeObjectEntries is the size from which an object is a dictionary,
// whatever its values: only its first entries are kept.
func (c HistoryCompaction) largeObjectEntries() int {
	return 4 * c.ArrayItems
}

// compactValue keeps the scalars and the small objects of scalars, such as
// a provenance, and the first items of the collections, recording every
// omission. Numbers are kept as written; kept strings are never changed: a
// longer one is omitted whole. A collection nested deeper than MaxDepth is
// omitted, but an object's value as small as a string kept. An array keeps a prefix of its items, so that their indexes stay
// true: it stops before an item it cannot keep whole.
func (c HistoryCompaction) compactValue(value any, path string, items int, depth int, omissions *[]omission) any {
	switch typed := value.(type) {
	case map[string]any:
		keys := make([]string, 0, len(typed))
		for key := range typed {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		// A large object is a dictionary, such as a status per node: only
		// its first entries are kept, whatever their values.
		if len(keys) > c.largeObjectEntries() {
			*omissions = append(*omissions, omission{Path: path, Kind: omittedObjectEntries, Kept: items, Omitted: len(keys) - items})
			keys = keys[:items]
		}
		out := make(map[string]any, len(keys))
		collections := 0
		var omitted []string
		for _, key := range keys {
			child, childPath := typed[key], path+"/"+pointerToken(key)
			switch {
			case isScalar(child):
				if text, ok := child.(string); ok && len(text) > c.MaxStringBytes {
					*omissions = append(*omissions, omission{Path: childPath, Kind: omittedLongString, Omitted: len(text)})
					continue
				}
				out[key] = child
			case c.isSmallScalarObject(child):
				out[key] = child
			case depth+1 > c.MaxDepth && c.isSmallValue(child):
				// Past the depth, a value as small as a string kept, such
				// as counts by state, is kept whole: its omission would
				// cost as much as the value, and lose what it tells.
				out[key] = child
			case depth+1 > c.MaxDepth:
				data, _ := json.Marshal(child)
				*omissions = append(*omissions, omission{Path: childPath, Kind: omittedDepth, Omitted: len(data)})
			case collections >= max(items, 1) && len(keys) > c.ArrayItems || items == 0:
				omitted = append(omitted, key)
			default:
				collections++
				out[key] = c.compactValue(child, childPath, items, depth+1, omissions)
			}
		}
		if len(omitted) > 0 {
			*omissions = append(*omissions, omission{Path: path, Kind: omittedObjectEntries, Kept: collections, Omitted: len(omitted)})
		}
		return out
	case []any:
		out := make([]any, 0, min(items, len(typed)))
		for i, child := range typed {
			if i == items {
				break
			}
			childPath := fmt.Sprintf("%s/%d", path, i)
			if text, ok := child.(string); ok && len(text) > c.MaxStringBytes {
				*omissions = append(*omissions, omission{Path: childPath, Kind: omittedLongString, Omitted: len(text)})
				break
			}
			if isCollection(child) && depth+1 > c.MaxDepth && !c.isSmallScalarObject(child) {
				data, _ := json.Marshal(child)
				*omissions = append(*omissions, omission{Path: childPath, Kind: omittedDepth, Omitted: len(data)})
				break
			}
			out = append(out, c.compactValue(child, childPath, items, depth+1, omissions))
		}
		if len(typed) > len(out) {
			*omissions = append(*omissions, omission{Path: path, Kind: omittedArrayTail, Kept: len(out), Omitted: len(typed) - len(out)})
		}
		return out
	default:
		return value
	}
}

func isScalar(value any) bool {
	switch value.(type) {
	case nil, bool, string, json.Number, float64:
		return true
	}
	return false
}

func isCollection(value any) bool {
	switch value.(type) {
	case map[string]any, []any:
		return true
	}
	return false
}

// isSmallScalarObject tells an object of a few short scalars, such as a
// provenance, kept whole.
func (c HistoryCompaction) isSmallScalarObject(value any) bool {
	object, ok := value.(map[string]any)
	if !ok || len(object) > c.ArrayItems {
		return false
	}
	for _, child := range object {
		if !isScalar(child) {
			return false
		}
		if text, ok := child.(string); ok && len(text) > c.MaxStringBytes {
			return false
		}
	}
	return true
}

// isSmallValue tells a value whose encoding is no longer than a string kept.
func (c HistoryCompaction) isSmallValue(value any) bool {
	data, err := json.Marshal(value)
	return err == nil && len(data) <= c.MaxStringBytes
}

// pointerToken escapes a key as a JSON pointer token.
func pointerToken(key string) string {
	return strings.ReplaceAll(strings.ReplaceAll(key, "~", "~0"), "/", "~1")
}

// historySize is the encoded size of messages, 0 for none.
func historySize(messages []llm.Message) int {
	if len(messages) == 0 {
		return 0
	}
	data, _ := json.Marshal(messages)
	return len(data)
}
