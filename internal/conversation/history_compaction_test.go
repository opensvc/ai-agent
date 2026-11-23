package conversation

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/llm"
)

// structuredResult is a tool result as the agent stores it.
func structuredResult(t *testing.T, structured string, withText bool) json.RawMessage {
	t.Helper()
	content := "[]"
	if withText {
		quoted, _ := json.Marshal(structured)
		content = `[{"type":"text","text":` + string(quoted) + `}]`
	}
	raw := json.RawMessage(`{"content":` + content + `,"structuredContent":` + structured + `}`)
	if !json.Valid(raw) {
		t.Fatalf("invalid result %s", raw)
	}
	return raw
}

func disks(count int) string {
	items := make([]string, count)
	for i := range items {
		items[i] = fmt.Sprintf(`{"id":"vd%c","devpath":"/dev/vd%c","size_bytes":12345678901234567%03d,"type":"disk"}`, 'a'+i%26, 'a'+i%26, i)
	}
	return `{"node":"dev5n1","count":` + fmt.Sprint(count) + `,"disks":[` + strings.Join(items, ",") + `],` +
		`"provenance":{"source":"opensvc_daemon","observed_at":"2026-10-15T13:00:00Z"}}`
}

func turnWith(prompt string, results ...llm.ToolResult) []llm.Message {
	calls := make([]llm.ToolCall, len(results))
	for i, result := range results {
		calls[i] = llm.ToolCall{ID: result.CallID, Name: "list_node_disks", Arguments: json.RawMessage(`{"node":"dev5n1"}`)}
	}
	return []llm.Message{
		{Role: llm.RoleUser, Text: prompt},
		{Role: llm.RoleAssistant, ToolCalls: calls},
		{Role: llm.RoleTool, ToolResults: results},
		{Role: llm.RoleAssistant, Text: "answer"},
	}
}

type decodedMarker struct {
	Policy           string     `json:"policy"`
	Partial          bool       `json:"partial"`
	OriginalBytes    int        `json:"original_bytes"`
	Omissions        []omission `json:"omissions"`
	OmissionsOmitted int        `json:"omissions_omitted"`
}

func decodeCompacted(t *testing.T, raw json.RawMessage) (map[string]any, decodedMarker) {
	t.Helper()
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var envelope map[string]any
	if err := decoder.Decode(&envelope); err != nil {
		t.Fatalf("compacted result is not JSON: %v: %s", err, raw)
	}
	var marker decodedMarker
	data, _ := json.Marshal(envelope["_history_compaction"])
	_ = json.Unmarshal(data, &marker)
	structured, _ := envelope["structuredContent"].(map[string]any)
	return structured, marker
}

func TestCompactHistoryKeepsSmallErrorAndUnsupportedResults(t *testing.T) {
	small := structuredResult(t, `{"node":"dev5n1","status":"up"}`, true)
	failed := json.RawMessage(`{"content":[{"type":"text","text":"` + strings.Repeat("e", 4000) + `"}]}`)
	text := json.RawMessage(`{"content":[{"type":"text","text":"` + strings.Repeat("t", 4000) + `"}]}`)
	image := json.RawMessage(`{"content":[{"type":"image","data":"` + strings.Repeat("A", 4000) + `","mimeType":"image/png"}],"structuredContent":{"a":1}}`)
	history := turnWith("q", llm.ToolResult{CallID: "a", Content: small}, llm.ToolResult{CallID: "b", Content: failed, IsError: true},
		llm.ToolResult{CallID: "c", Content: text}, llm.ToolResult{CallID: "d", Content: image})
	compacted, stats, _ := compactHistory(history, HistoryCompaction{})
	if !reflect.DeepEqual(compacted, history) || stats.CompactedResults != 0 || stats.RemovedTurns != 0 {
		t.Fatalf("results changed: %+v", stats)
	}
}

func TestCompactHistoryCompactsAStructuredResult(t *testing.T) {
	original := structuredResult(t, disks(24), true)
	history := turnWith("q", llm.ToolResult{CallID: "call-1", Content: original})
	before, _ := json.Marshal(history)
	compacted, stats, _ := compactHistory(history, HistoryCompaction{})
	if after, _ := json.Marshal(history); !bytes.Equal(before, after) {
		t.Fatal("the stored history was changed")
	}
	result := compacted[2].ToolResults[0]
	if result.CallID != "call-1" || compacted[1].ToolCalls[0].ID != "call-1" || stats.CompactedResults != 1 {
		t.Fatalf("calls and results are not paired: %+v", compacted)
	}
	if len(result.Content) > DefaultHistoryCompaction.MaxResultBytes {
		t.Fatalf("compacted result holds %d bytes", len(result.Content))
	}
	structured, marker := decodeCompacted(t, result.Content)
	if marker.Policy != "v2" || !marker.Partial || marker.OriginalBytes != len(original) {
		t.Fatalf("marker = %+v", marker)
	}
	// The scalars, the count as the tool returned it, and the nested
	// provenance are kept; the disks array holds real disks only.
	if structured["node"] != "dev5n1" || structured["count"] != json.Number("24") {
		t.Fatalf("scalars = %v", structured)
	}
	if provenance, _ := structured["provenance"].(map[string]any); provenance["observed_at"] != "2026-10-15T13:00:00Z" {
		t.Fatalf("provenance = %v", structured["provenance"])
	}
	kept, _ := structured["disks"].([]any)
	for _, disk := range kept {
		if _, ok := disk.(map[string]any)["id"]; !ok {
			t.Fatalf("an artificial item was added: %v", disk)
		}
	}
	// Integers are kept as written, never through float64.
	if first := kept[0].(map[string]any)["size_bytes"]; first != json.Number("12345678901234567000") {
		t.Fatalf("size_bytes = %v", first)
	}
	var tail *omission
	for i, o := range marker.Omissions {
		if o.Path == "/structuredContent/disks" && o.Kind == omittedArrayTail {
			tail = &marker.Omissions[i]
		}
	}
	if tail == nil || tail.Kept != len(kept) || tail.Kept+tail.Omitted != 24 {
		t.Fatalf("omissions = %+v, kept %d", marker.Omissions, len(kept))
	}
	if marker.Omissions[0] != (omission{Path: "/content", Kind: omittedContent, Omitted: 1}) {
		t.Fatalf("the text copy is not reported: %+v", marker.Omissions)
	}
}

func TestCompactHistoryOmitsLongStringsWhole(t *testing.T) {
	long := strings.Repeat("x", 300)
	structured := `{"id":"node-1","description":"` + long + `","items":[` + strings.TrimSuffix(strings.Repeat(`{"n":"`+strings.Repeat("y", 100)+`"},`, 40), ",") + `]}`
	compacted, _, _ := compactHistory(turnWith("q", llm.ToolResult{CallID: "c", Content: structuredResult(t, structured, false)}), HistoryCompaction{})
	value, marker := decodeCompacted(t, compacted[2].ToolResults[0].Content)
	if _, ok := value["description"]; ok || value["id"] != "node-1" {
		t.Fatalf("value = %v", value)
	}
	found := false
	for _, o := range marker.Omissions {
		found = found || (o.Path == "/structuredContent/description" && o.Kind == omittedLongString && o.Omitted == 300)
	}
	if !found {
		t.Fatalf("omissions = %+v", marker.Omissions)
	}
}

func TestCompactHistoryBoundsTheOmissions(t *testing.T) {
	// 8 groups of 3 short lists, each longer than kept: 24 omissions in a
	// small result.
	list := "[" + strings.TrimSuffix(strings.Repeat("1,", 60), ",") + "]"
	var fields []string
	for i := range 8 {
		fields = append(fields, fmt.Sprintf(`"g%d":{"a":%s,"b":%s,"c":%s}`, i, list, list, list))
	}
	compacted, _, _ := compactHistory(turnWith("q", llm.ToolResult{CallID: "c", Content: structuredResult(t, "{"+strings.Join(fields, ",")+"}", false)}), HistoryCompaction{})
	result := compacted[2].ToolResults[0].Content
	_, marker := decodeCompacted(t, result)
	if len(result) > DefaultHistoryCompaction.MaxResultBytes || len(marker.Omissions) > DefaultHistoryCompaction.MaxOmissions || marker.OmissionsOmitted == 0 {
		t.Fatalf("%d bytes, %d omissions, %d omitted", len(result), len(marker.Omissions), marker.OmissionsOmitted)
	}
}

func TestCompactHistoryKeepsTheGlobalBudget(t *testing.T) {
	var history []llm.Message
	for turn := range 6 {
		var results []llm.ToolResult
		for i := range 4 {
			results = append(results, llm.ToolResult{CallID: fmt.Sprintf("t%d-%d", turn, i), Content: structuredResult(t, disks(24), true)})
		}
		history = append(history, turnWith(fmt.Sprintf("question %d", turn), results...)...)
	}
	compacted, stats, _ := compactHistory(history, HistoryCompaction{})
	size := func(messages []llm.Message) (total int, perTurn []int) {
		for _, message := range messages {
			if message.Role == llm.RoleUser {
				perTurn = append(perTurn, 0)
			}
			for _, result := range message.ToolResults {
				total += len(result.Content)
				perTurn[len(perTurn)-1] += len(result.Content)
			}
		}
		return total, perTurn
	}
	total, perTurn := size(compacted)
	if total > DefaultHistoryCompaction.MaxHistoryResultBytes || compacted[0].Role != llm.RoleUser {
		t.Fatalf("history results hold %d bytes, first role %s", total, compacted[0].Role)
	}
	// The oldest turns are reduced first: the last turn keeps the most.
	if perTurn[len(perTurn)-1] < perTurn[0] {
		t.Fatalf("per turn = %v", perTurn)
	}
	if stats.CompactedResults == 0 {
		t.Fatalf("stats = %+v", stats)
	}
}

func TestCompactHistoryRemovesOldTurnsAsALastResort(t *testing.T) {
	failed := func(id string) llm.ToolResult {
		return llm.ToolResult{CallID: id, IsError: true, Content: json.RawMessage(`{"content":[{"type":"text","text":"` + strings.Repeat("e", 6000) + `"}]}`)}
	}
	history := append(append(turnWith("first", failed("a"), failed("b")), turnWith("second", failed("c"))...), turnWith("third", failed("d"))...)
	compacted, stats, _ := compactHistory(history, HistoryCompaction{})
	if stats.RemovedTurns != 1 || compacted[0].Text != "second" || len(compacted) != 8 {
		t.Fatalf("removed %d turns, first %q, %d messages", stats.RemovedTurns, compacted[0].Text, len(compacted))
	}
	// Errors are never changed.
	if !reflect.DeepEqual(compacted[2].ToolResults, history[6].ToolResults) {
		t.Fatal("an error result was changed")
	}
}

func TestPrepareHistoryDescribesTheHistorySent(t *testing.T) {
	service := &Service{config: withServiceDefaults(ServiceConfig{Backend: "om3"})}
	history := append(turnWith("first", llm.ToolResult{CallID: "a", Content: structuredResult(t, disks(24), true)}),
		turnWith("second", llm.ToolResult{CallID: "b", Content: structuredResult(t, disks(24), true)})...)
	sent, stats, err := service.prepareHistory(history)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Policy != "v2" || stats.MaxResultBytes != 2<<10 || stats.MaxHistoryResultBytes != 16<<10 || stats.CompactedResults != 2 ||
		stats.LoadedBytes != historySize(history) || stats.CompactedBytes >= stats.LoadedBytes || stats.SentBytes != historySize(sent) || stats.RemovedTurns != 0 {
		t.Fatalf("stats = %+v", stats)
	}
	other := *stats
	other.SentBytes++
	run := TurnRun{History: stats}
	if run.Equal(TurnRun{History: &other}) || !run.Equal(TurnRun{History: stats}) || run.Equal(TurnRun{}) {
		t.Fatal("runs compare their history")
	}
}

func TestPrepareHistoryOfAFirstTurn(t *testing.T) {
	service := &Service{config: withServiceDefaults(ServiceConfig{Backend: "om3"})}
	for _, history := range [][]llm.Message{nil, {}} {
		_, stats, err := service.prepareHistory(history)
		if err != nil || stats.LoadedBytes != 0 || stats.CompactedBytes != 0 || stats.SentBytes != 0 {
			t.Fatalf("empty history stats = %+v, %v", stats, err)
		}
	}
}

func valueDepth(value any) int {
	switch typed := value.(type) {
	case map[string]any:
		deepest := 0
		for _, child := range typed {
			deepest = max(deepest, valueDepth(child))
		}
		return deepest + 1
	case []any:
		deepest := 0
		for _, child := range typed {
			deepest = max(deepest, valueDepth(child))
		}
		return deepest + 1
	}
	return 0
}

func compactOne(t *testing.T, structured string, content string) (map[string]any, decodedMarker, json.RawMessage) {
	t.Helper()
	raw := json.RawMessage(`{"content":` + content + `,"structuredContent":` + structured + `}`)
	if !json.Valid(raw) {
		t.Fatalf("invalid result %s", raw)
	}
	compacted, _, _ := compactHistory(turnWith("q", llm.ToolResult{CallID: "c", Content: raw}), HistoryCompaction{})
	result := compacted[2].ToolResults[0].Content
	value, marker := decodeCompacted(t, result)
	return value, marker, result
}

func hasOmission(marker decodedMarker, want omission) bool {
	for _, o := range marker.Omissions {
		if o == want {
			return true
		}
	}
	return false
}

// An array keeps the prefix before its first string too long to keep, so
// that the indexes of the items kept stay true.
func TestCompactHistoryStopsAnArrayBeforeALongString(t *testing.T) {
	long := strings.Repeat("l", 300)
	value, marker, _ := compactOne(t, `{"lines":["first","`+long+`","third"],"pad":"`+strings.Repeat("p", 200)+`","more":[`+strings.TrimSuffix(strings.Repeat(`"`+strings.Repeat("m", 200)+`",`, 12), ",")+`]}`, "[]")
	if lines, _ := value["lines"].([]any); len(lines) != 1 || lines[0] != "first" {
		t.Fatalf("lines = %v", value["lines"])
	}
	if !hasOmission(marker, omission{Path: "/structuredContent/lines/1", Kind: omittedLongString, Omitted: 300}) ||
		!hasOmission(marker, omission{Path: "/structuredContent/lines", Kind: omittedArrayTail, Kept: 1, Omitted: 2}) {
		t.Fatalf("omissions = %+v", marker.Omissions)
	}
}

// A large dictionary of scalars keeps its first entries, and the small
// objects beside it, such as the provenance, are kept whole.
func TestCompactHistoryReducesALargeDictionaryOfScalars(t *testing.T) {
	entries := make([]string, 300)
	for i := range entries {
		entries[i] = fmt.Sprintf(`"node%03d":"up"`, i)
	}
	value, marker, result := compactOne(t, `{"status":{`+strings.Join(entries, ",")+`},"provenance":{"source":"daemon","observed_at":"2026-10-15T13:00:00Z"}}`, "[]")
	status, _ := value["status"].(map[string]any)
	if len(status) == 0 || status["node000"] != "up" || len(result) > DefaultHistoryCompaction.MaxResultBytes {
		t.Fatalf("status = %v, %d bytes", status, len(result))
	}
	if provenance, _ := value["provenance"].(map[string]any); provenance["observed_at"] != "2026-10-15T13:00:00Z" {
		t.Fatalf("provenance = %v", value["provenance"])
	}
	if !hasOmission(marker, omission{Path: "/structuredContent/status", Kind: omittedObjectEntries, Kept: len(status), Omitted: 300 - len(status)}) {
		t.Fatalf("omissions = %+v", marker.Omissions)
	}
}

// Only the text copy the MCP SDK adds is dropped: a distinct text is kept
// when short and omitted whole when long.
func TestCompactHistoryKeepsADistinctText(t *testing.T) {
	structured := disks(24)
	quoted, _ := json.Marshal(structured)
	_, marker, result := compactOne(t, structured, `[{"type":"text","text":`+string(quoted)+`}]`)
	if !hasOmission(marker, omission{Path: "/content", Kind: omittedContent, Omitted: 1}) || strings.Contains(string(result), `"type":"text"`) {
		t.Fatalf("the duplicate was not dropped: %s", result)
	}
	for name, content := range map[string]string{
		"warning":     `[{"type":"text","text":"Warning: the cache is 2 hours old."}]`,
		"annotations": `[{"type":"text","text":` + string(quoted) + `,"annotations":{"priority":1}}]`,
	} {
		_, marker, result := compactOne(t, structured, content)
		if name == "warning" && !strings.Contains(string(result), "Warning: the cache is 2 hours old.") {
			t.Fatalf("%s: the distinct text was dropped: %s", name, result)
		}
		if hasOmission(marker, omission{Path: "/content", Kind: omittedContent, Omitted: 1}) {
			t.Fatalf("%s: a distinct text was taken for the copy", name)
		}
	}
	long := strings.Repeat("w", 400)
	_, marker, result = compactOne(t, structured, `[{"type":"text","text":"`+long+`"}]`)
	if strings.Contains(string(result), long) || !hasOmission(marker, omission{Path: "/content", Kind: omittedArrayTail, Kept: 0, Omitted: 1}) {
		t.Fatalf("a long distinct text = %s", result)
	}
}

// MaxDepth bounds the nested arrays as the nested objects.
func TestCompactHistoryBoundsNestedArrays(t *testing.T) {
	nested := strings.Repeat("[", 21) + "1" + strings.Repeat("]", 21)
	value, marker, _ := compactOne(t, `{"deep":`+nested+`,"pad":"`+strings.Repeat("p", 200)+`","more":[`+strings.TrimSuffix(strings.Repeat(`"`+strings.Repeat("m", 200)+`",`, 12), ",")+`]}`, "[]")
	if depth := valueDepth(value["deep"]); depth > DefaultHistoryCompaction.MaxDepth {
		t.Fatalf("a value kept at depth %d", depth)
	}
	found := false
	for _, o := range marker.Omissions {
		found = found || o.Kind == omittedDepth
	}
	if !found {
		t.Fatalf("omissions = %+v", marker.Omissions)
	}
}

// A turn left out frees its room: the turns kept are compacted again from
// their original results.
func TestCompactHistoryRecompactsTheTurnsKept(t *testing.T) {
	failed := llm.ToolResult{CallID: "old", IsError: true, Content: json.RawMessage(`{"content":[{"type":"text","text":"` + strings.Repeat("e", 20000) + `"}]}`)}
	recent := structuredResult(t, disks(6), false)
	history := append(turnWith("old", failed), turnWith("recent", llm.ToolResult{CallID: "new", Content: recent})...)
	compacted, stats, _ := compactHistory(history, HistoryCompaction{})
	if stats.RemovedTurns != 1 || compacted[0].Text != "recent" {
		t.Fatalf("removed %d turns, first %q", stats.RemovedTurns, compacted[0].Text)
	}
	if got := compacted[2].ToolResults[0].Content; len(recent) <= DefaultHistoryCompaction.MaxResultBytes && !bytes.Equal(got, recent) {
		t.Fatalf("the recent result was not restored: %s", got)
	} else if len(recent) > DefaultHistoryCompaction.MaxResultBytes {
		value, _ := decodeCompacted(t, got)
		if _, ok := value["disks"]; !ok {
			t.Fatalf("the recent result stayed a marker: %s", got)
		}
	}
}

func TestHistoryCompactionPolicyIsCompletedOrRefused(t *testing.T) {
	completed := HistoryCompaction{MaxResultBytes: 4096}.withDefaults()
	if completed.MaxHistoryResultBytes != DefaultHistoryCompaction.MaxHistoryResultBytes || completed.MaxOmissions != DefaultHistoryCompaction.MaxOmissions || completed.validate() != nil {
		t.Fatalf("completed = %+v", completed)
	}
	for _, policy := range []HistoryCompaction{
		{MaxOmissions: -1}, {ArrayItems: -2}, {MaxResultBytes: 100}, {MaxResultBytes: 32 << 10}, {MaxDepth: -1}, {MaxStringBytes: -5},
	} {
		if _, err := NewService(&serviceTestStore{}, turnRunnerFunc(nil), ServiceConfig{Backend: "om3", HistoryCompaction: policy}); err == nil {
			t.Fatalf("policy %+v accepted", policy)
		}
	}
}

// The results of the turns boundHistory leaves out count as removed, not
// as compacted.
func TestPrepareHistoryCountsTheResultsSent(t *testing.T) {
	history := append(turnWith("first", llm.ToolResult{CallID: "a", Content: structuredResult(t, disks(24), true)}),
		turnWith("second", llm.ToolResult{CallID: "b", Content: structuredResult(t, disks(24), true)})...)
	service := &Service{config: withServiceDefaults(ServiceConfig{Backend: "om3", MaxHistoryMessages: 4})}
	sent, stats, err := service.prepareHistory(history)
	if err != nil {
		t.Fatal(err)
	}
	if len(sent) != 4 || stats.RemovedTurns != 1 || stats.CompactedResults != 1 {
		t.Fatalf("%d messages sent, stats = %+v", len(sent), stats)
	}
}

func textContents(texts ...string) string {
	items := make([]string, len(texts))
	for i, text := range texts {
		quoted, _ := json.Marshal(text)
		items[i] = `{"type":"text","text":` + string(quoted) + `}`
	}
	return "[" + strings.Join(items, ",") + "]"
}

// The budget holds for the whole result, its texts and marker included,
// down to the smallest budget of a result.
func TestCompactedResultsFitTheirBudget(t *testing.T) {
	many := make([]string, 20)
	for i := range many {
		many[i] = strings.Repeat(string(rune('a'+i)), 200)
	}
	bigMeta := `[{"type":"text","text":"short","_meta":{"k":"` + strings.Repeat("m", 4000) + `"}}]`
	longKeys := make([]string, 40)
	for i := range longKeys {
		longKeys[i] = fmt.Sprintf(`"%s%02d":[1,2,3,4,5,6,7,8,9]`, strings.Repeat("k", 100), i)
	}
	for name, raw := range map[string]string{
		"twenty texts":  `{"content":` + textContents(many...) + `,"structuredContent":` + disks(24) + `}`,
		"big metadata":  `{"content":` + bigMeta + `,"structuredContent":` + disks(24) + `}`,
		"long keys":     `{"content":[],"structuredContent":{` + strings.Join(longKeys, ",") + `}}`,
		"scalar values": `{"content":[],"structuredContent":{"a":"` + strings.Repeat("v", 250) + `","b":"` + strings.Repeat("w", 250) + `","c":"` + strings.Repeat("x", 250) + `","d":"` + strings.Repeat("y", 250) + `","e":"` + strings.Repeat("z", 250) + `","f":"` + strings.Repeat("u", 250) + `","g":"` + strings.Repeat("t", 250) + `","h":"` + strings.Repeat("s", 250) + `","i":"` + strings.Repeat("r", 250) + `"}}`,
	} {
		for _, policy := range []HistoryCompaction{{}, {MaxResultBytes: minResultBytes}} {
			policy = policy.withDefaults()
			if !json.Valid([]byte(raw)) {
				t.Fatalf("%s: invalid input", name)
			}
			compacted, ok := policy.compactResult(json.RawMessage(raw), policy.MaxResultBytes)
			if !ok || len(compacted) > policy.MaxResultBytes || !json.Valid(compacted) {
				t.Errorf("%s, budget %d: %d bytes, handled %v", name, policy.MaxResultBytes, len(compacted), ok)
			}
		}
	}
}

// A text with anything after the JSON value is distinct from the
// structured content, never taken for its copy.
func TestADuplicateIsTheValueAlone(t *testing.T) {
	structured := map[string]any{"ok": true}
	for _, text := range []string{`{"ok":true} Warning: inventory is incomplete.`, `{"ok":true} 2`, `{"ok":true} true`, `{"ok":true}{}`} {
		if duplicatedText([]any{map[string]any{"type": "text", "text": text}}, structured) {
			t.Errorf("%q taken for the copy", text)
		}
	}
	if !duplicatedText([]any{map[string]any{"type": "text", "text": ` {"ok": true} `}}, structured) {
		t.Error("the copy was not recognized")
	}
}

// The texts kept are a prefix of the contents: an index of the marker
// designates the content it had in the original result.
func TestCompactedContentsKeepTheirIndexes(t *testing.T) {
	_, marker, result := compactOne(t, disks(24), textContents(strings.Repeat("l", 400), "Warning: partial inventory."))
	var envelope struct {
		Content []map[string]any `json:"content"`
	}
	_ = json.Unmarshal(result, &envelope)
	if len(envelope.Content) != 0 || !hasOmission(marker, omission{Path: "/content", Kind: omittedArrayTail, Kept: 0, Omitted: 2}) {
		t.Fatalf("contents = %v, omissions = %+v", envelope.Content, marker.Omissions)
	}
	found := false
	for _, o := range marker.Omissions {
		found = found || (o.Path == "/content/0" && o.Kind == omittedValue)
	}
	if !found {
		t.Fatalf("the long first text is not designated: %+v", marker.Omissions)
	}
	_, marker, result = compactOne(t, disks(24), textContents("Warning: partial inventory.", strings.Repeat("l", 400)))
	_ = json.Unmarshal(result, &envelope)
	if len(envelope.Content) != 1 || envelope.Content[0]["text"] != "Warning: partial inventory." ||
		!hasOmission(marker, omission{Path: "/content", Kind: omittedArrayTail, Kept: 1, Omitted: 1}) {
		t.Fatalf("prefix = %v, omissions = %+v", envelope.Content, marker.Omissions)
	}
}

// The agent reads the tools the searches of the former turns showed in their
// results: a search result, a text content alone, stays whole under the
// budget of a result and under the global budget.
func TestCompactHistoryKeepsASearchResultWhole(t *testing.T) {
	var tools []string
	for i := range 40 {
		tools = append(tools, fmt.Sprintf(`{"name":"tool_%d","summary":"%s","shown":true}`, i, strings.Repeat("a summary ", 20)))
	}
	reply, _ := json.Marshal(`{"tools":[` + strings.Join(tools, ",") + `],"next":"The tools shown are callable now."}`)
	search := json.RawMessage(`{"content":[{"type":"text","text":` + string(reply) + `}]}`)
	if len(search) <= DefaultHistoryCompaction.MaxResultBytes {
		t.Fatalf("the search result holds %d bytes, not more than the budget of a result", len(search))
	}
	history := turnWith("search", llm.ToolResult{CallID: "s", Content: search})
	history[1].ToolCalls[0].Name = "search_tools"
	for turn := range 6 {
		var results []llm.ToolResult
		for i := range 4 {
			results = append(results, llm.ToolResult{CallID: fmt.Sprintf("t%d-%d", turn, i), Content: structuredResult(t, disks(24), true)})
		}
		history = append(history, turnWith(fmt.Sprintf("question %d", turn), results...)...)
	}
	compacted, stats, _ := compactHistory(history, HistoryCompaction{})
	if stats.CompactedResults == 0 {
		t.Fatalf("the history was not compacted: %+v", stats)
	}
	for _, message := range compacted {
		for _, result := range message.ToolResults {
			if result.CallID == "s" {
				if string(result.Content) != string(search) {
					t.Fatalf("the search result was compacted: %s", result.Content)
				}
				return
			}
		}
	}
	t.Fatalf("the search turn was left out: %+v", stats)
}

// textResult is a result of a tool of text alone, as search_docs gives.
func textResult(size int) json.RawMessage {
	return json.RawMessage(`{"content":[{"type":"text","text":"` + strings.Repeat("d", size) + `"}]}`)
}

// named gives the calls of a turn the name of a tool.
func named(turn []llm.Message, name string) []llm.Message {
	for i := range turn[1].ToolCalls {
		turn[1].ToolCalls[i].Name = name
	}
	return turn
}

func TestCompactHistoryReducesTextResultsBeforeLeavingATurnOut(t *testing.T) {
	// The turn of 2026-11-23 on dev1: three searches of the documentation
	// of 10 to 17 KiB, over the budget of the whole history alone.
	docs := named(turnWith("how to declare an md disk", llm.ToolResult{CallID: "a", Content: textResult(10000)},
		llm.ToolResult{CallID: "b", Content: textResult(17000)}, llm.ToolResult{CallID: "c", Content: textResult(13000)}), "search_docs")
	history := append(turnWith("is there a problem", llm.ToolResult{CallID: "s", Content: structuredResult(t, disks(24), true)}), docs...)
	compacted, stats, indexes := compactHistory(history, HistoryCompaction{})
	if stats.RemovedTurns != 0 || len(compacted) != len(history) || compacted[0].Text != "is there a problem" || compacted[4].Text != "how to declare an md disk" {
		t.Fatalf("turns left out: %+v, %d messages", stats, len(compacted))
	}
	if total := resultBytes(compacted); total > DefaultHistoryCompaction.MaxHistoryResultBytes {
		t.Fatalf("results hold %d bytes", total)
	}
	// The oldest texts are reduced first, to a marker of their contents.
	_, marker := decodeCompacted(t, compacted[6].ToolResults[0].Content)
	if marker.Policy != HistoryCompactionPolicy || marker.OriginalBytes != len(textResult(10000)) || len(marker.Omissions) != 1 ||
		marker.Omissions[0] != (omission{Path: "/content", Kind: omittedArrayTail, Omitted: 1}) {
		t.Fatalf("text marker = %+v", marker)
	}
	if !json.Valid(compacted[6].ToolResults[0].Content) || len(indexes) != stats.CompactedResults {
		t.Fatalf("indexes %v, stats %+v", indexes, stats)
	}
	// The room the texts free goes back to the structured result.
	if value, _ := decodeCompacted(t, compacted[2].ToolResults[0].Content); value["disks"] == nil {
		t.Fatalf("the structured result stayed a marker: %s", compacted[2].ToolResults[0].Content)
	}
	// Calls and arguments stay: the model knows what it searched.
	if !reflect.DeepEqual(compacted[5].ToolCalls, history[5].ToolCalls) {
		t.Fatal("the calls of the searches changed")
	}
}

func TestCompactHistoryKeepsATextThatFits(t *testing.T) {
	// The structured results are reduced first: a text that then fits
	// stays whole.
	var results []llm.ToolResult
	for i := range 6 {
		results = append(results, llm.ToolResult{CallID: fmt.Sprint("s", i), Content: structuredResult(t, disks(24), true)})
	}
	docs := named(turnWith("search", llm.ToolResult{CallID: "d", Content: textResult(9000)}), "search_docs")
	history := append(docs, turnWith("disks", results...)...)
	compacted, stats, _ := compactHistory(history, HistoryCompaction{})
	if stats.RemovedTurns != 0 || string(compacted[2].ToolResults[0].Content) != string(history[2].ToolResults[0].Content) {
		t.Fatalf("the text was reduced: %+v", stats)
	}
}

func TestCompactHistoryNeverLeavesTheLastTurnOut(t *testing.T) {
	// A last turn over the budget alone, by results kept whole: it is sent,
	// over the budget, rather than no history, and the turn before it,
	// reduced to markers, is kept: leaving it out would not bring the
	// results under the budget.
	search := named(turnWith("find a tool", llm.ToolResult{CallID: "t", Content: textResult(20000)}), agent.SearchToolName)
	history := append(turnWith("first", llm.ToolResult{CallID: "a", Content: textResult(8000)}), search...)
	compacted, stats, _ := compactHistory(history, HistoryCompaction{})
	if stats.RemovedTurns != 0 || len(compacted) != 8 || compacted[0].Text != "first" {
		t.Fatalf("removed %d turns, %d messages", stats.RemovedTurns, len(compacted))
	}
	if string(compacted[6].ToolResults[0].Content) != string(search[2].ToolResults[0].Content) {
		t.Fatal("the search_tools result was compacted")
	}
	if _, marker := decodeCompacted(t, compacted[2].ToolResults[0].Content); marker.OriginalBytes != len(textResult(8000)) {
		t.Fatalf("the former text was not reduced: %s", compacted[2].ToolResults[0].Content)
	}
}

func TestCompactHistoryKeepsTheTurnsBeforeALastTurnOverTheBudget(t *testing.T) {
	small := func(id string) llm.ToolResult {
		return llm.ToolResult{CallID: id, Content: structuredResult(t, `{"node":"dev5n1","status":"up"}`, false)}
	}
	failed := func(id string) llm.ToolResult {
		return llm.ToolResult{CallID: id, IsError: true, Content: json.RawMessage(`{"content":[{"type":"text","text":"` + strings.Repeat("e", 17000) + `"}]}`)}
	}
	history := append(append(turnWith("first", small("a")), turnWith("second", small("b"))...), turnWith("third", failed("c"))...)
	compacted, stats, _ := compactHistory(history, HistoryCompaction{})
	if stats.RemovedTurns != 0 || len(compacted) != len(history) {
		t.Fatalf("removed %d turns, %d messages", stats.RemovedTurns, len(compacted))
	}
	// Turns whose own results exceed the budget are still left out.
	history = append(turnWith("large", failed("d")), history...)
	compacted, stats, _ = compactHistory(history, HistoryCompaction{})
	if stats.RemovedTurns != 1 || compacted[0].Text != "first" {
		t.Fatalf("removed %d turns, first %q", stats.RemovedTurns, compacted[0].Text)
	}
}

func TestCompactHistoryRestoresTheStructuredResultsAMarkerFrees(t *testing.T) {
	// A text over the budget alone is reduced to a marker; the structured
	// results of the last turn are then compacted to their budget, not to a
	// marker.
	docs := named(turnWith("docs", llm.ToolResult{CallID: "d", Content: textResult(17000)}), "search_docs")
	var results []llm.ToolResult
	for i := range 3 {
		results = append(results, llm.ToolResult{CallID: fmt.Sprint("s", i), Content: structuredResult(t, disks(24), true)})
	}
	history := append(docs, turnWith("disks", results...)...)
	compacted, stats, _ := compactHistory(history, HistoryCompaction{})
	if stats.RemovedTurns != 0 || resultBytes(compacted) > DefaultHistoryCompaction.MaxHistoryResultBytes {
		t.Fatalf("stats %+v, results hold %d bytes", stats, resultBytes(compacted))
	}
	if _, marker := decodeCompacted(t, compacted[2].ToolResults[0].Content); marker.OriginalBytes != len(textResult(17000)) {
		t.Fatalf("the text was not reduced: %d bytes", len(compacted[2].ToolResults[0].Content))
	}
	for _, result := range compacted[6].ToolResults {
		if value, _ := decodeCompacted(t, result.Content); value["disks"] == nil || len(result.Content) > DefaultHistoryCompaction.MaxResultBytes {
			t.Fatalf("the structured result was not compacted to its budget: %s", result.Content)
		}
	}
}

func TestCompactHistoryReadsTheToolOfEachTurn(t *testing.T) {
	// Some providers number their calls from zero in each answer: the name
	// of a result's tool is the one of the call before it.
	search := named(turnWith("find", llm.ToolResult{CallID: "call_0", Content: textResult(3000)}), agent.SearchToolName)
	docs := named(turnWith("docs", llm.ToolResult{CallID: "call_0", Content: textResult(17000)}), "search_docs")
	for _, order := range [][2][]llm.Message{{search, docs}, {docs, search}} {
		history := append(slices.Clone(order[0]), order[1]...)
		compacted, stats, _ := compactHistory(history, HistoryCompaction{})
		if stats.RemovedTurns != 0 || len(compacted) != 8 {
			t.Fatalf("removed %d turns", stats.RemovedTurns)
		}
		for i, message := range compacted {
			if len(message.ToolResults) == 0 {
				continue
			}
			whole := string(message.ToolResults[0].Content) == string(history[i].ToolResults[0].Content)
			if isSearch := history[i-1].ToolCalls[0].Name == agent.SearchToolName; whole != isSearch {
				t.Fatalf("%s result kept whole: %v", history[i-1].ToolCalls[0].Name, whole)
			}
		}
	}
}

func TestCompactHistoryReducesTheFormerSearches(t *testing.T) {
	// The agent reads back the searches of the last turn that searched:
	// those of the turns before are reduced rather than turns left out.
	var history []llm.Message
	for turn := range 5 {
		history = append(history, named(turnWith(fmt.Sprint("q", turn),
			llm.ToolResult{CallID: fmt.Sprint(turn, "a"), Content: textResult(3500)},
			llm.ToolResult{CallID: fmt.Sprint(turn, "b"), Content: textResult(3500)}), agent.SearchToolName)...)
	}
	compacted, stats, _ := compactHistory(history, HistoryCompaction{})
	if stats.RemovedTurns != 0 || compacted[0].Text != "q0" || resultBytes(compacted) > DefaultHistoryCompaction.MaxHistoryResultBytes {
		t.Fatalf("stats %+v, results hold %d bytes", stats, resultBytes(compacted))
	}
	if !reflect.DeepEqual(compacted[len(compacted)-2].ToolResults, history[len(history)-2].ToolResults) {
		t.Fatal("the searches of the last turn were reduced")
	}
}

func TestTextMarkerTakesTextsAlone(t *testing.T) {
	policy := DefaultHistoryCompaction
	mixed := json.RawMessage(`{"content":[{"type":"text","text":"a"},{"type":"image","data":"AAAA","mimeType":"image/png"}]}`)
	if _, ok := policy.textMarker(mixed); ok {
		t.Fatal("a result with an image was reduced")
	}
	// The envelope of an MCP result, isError included, is reduced.
	mcp := json.RawMessage(`{"content":[{"type":"text","text":"` + strings.Repeat("d", 4000) + `"}],"isError":false}`)
	if content, ok := policy.textMarker(mcp); !ok || !json.Valid(content) || len(content) >= len(mcp) {
		t.Fatalf("the MCP text result was not reduced: %s", content)
	}
}
