package responses

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/opensvc/ai-agent/internal/llm"
)

// A result of text alone is sent as it is, its line breaks and quotes not
// escaped, a text holding JSON too; a structured result and an error keep
// their JSON, the error its flag.
func TestAResultOfTextAloneIsSentAsItIs(t *testing.T) {
	request := llm.Request{Messages: []llm.Message{
		{Role: llm.RoleUser, Text: "q"},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "a", Name: "search_code", Arguments: json.RawMessage(`{}`)}, {ID: "b", Name: "get_status", Arguments: json.RawMessage(`{}`)}, {ID: "c", Name: "stop", Arguments: json.RawMessage(`{}`)}, {ID: "d", Name: "search_tools", Arguments: json.RawMessage(`{}`)}}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{
			{CallID: "a", Content: json.RawMessage(`{"content":[{"type":"text","text":"if a < b {\n\treturn \"x\"\n}"}]}`)},
			{CallID: "b", Content: json.RawMessage(`{"content":[],"structuredContent":{"up":true},"isError":false}`)},
			{CallID: "c", Content: json.RawMessage(`{"content":[{"type":"text","text":"refused"}]}`), IsError: true},
			{CallID: "d", Content: json.RawMessage(`{"content":[{"type":"text","text":"{\"tools\":[]}"}]}`)},
		}},
	}}
	wire, err := newCreateRequest("m", 1024, request)
	if err != nil {
		t.Fatal(err)
	}
	data, _ := json.Marshal(wire)
	var decoded any
	_ = json.Unmarshal(data, &decoded)
	var texts []string
	var walk func(any)
	walk = func(value any) {
		switch typed := value.(type) {
		case map[string]any:
			for _, child := range typed {
				walk(child)
			}
		case []any:
			for _, child := range typed {
				walk(child)
			}
		case string:
			texts = append(texts, typed)
		}
	}
	walk(decoded)
	raw, structured, errorKept, jsonText, errorFlag := false, false, false, false, false
	for _, text := range texts {
		raw = raw || text == "if a < b {\n\treturn \"x\"\n}"
		structured = structured || strings.Contains(text, `"structuredContent":{"up":true}`)
		errorKept = errorKept || strings.Contains(text, `{"content":[{"type":"text","text":"refused"}]}`)
		jsonText = jsonText || text == `{"tools":[]}`
		errorFlag = errorFlag || strings.Contains(text, `"is_error":true`)
	}
	// Messages flags the error on its block, the others in their envelope.
	errorFlag = errorFlag || strings.Contains(string(data), `"is_error":true`)
	if !raw || !structured || !errorKept || !jsonText || !errorFlag {
		t.Fatalf("raw text %v, structured result kept %v, error kept %v, JSON text as is %v, in %s", raw, structured, errorKept, jsonText, data)
	}
}
