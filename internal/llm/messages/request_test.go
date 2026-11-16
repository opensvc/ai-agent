package messages

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/opensvc/ai-agent/internal/llm"
)

func TestCreateRequestPreservesTextToolsAndResults(t *testing.T) {
	request := llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Text: "instructions"},
			{Role: llm.RoleSystem, Text: "more instructions"},
			{Role: llm.RoleUser, Text: "health"},
			{Role: llm.RoleAssistant, Text: "checking", ToolCalls: []llm.ToolCall{
				{ID: "call-1", Name: "health", Arguments: json.RawMessage(`{"node":"node1"}`)},
				{ID: "call-2", Name: "health", Arguments: json.RawMessage(`{}`)},
			}},
			{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call-1", Content: json.RawMessage(`{"status":"ok"}`)}}},
			{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call-2", Content: json.RawMessage(`{"error":"unavailable"}`), IsError: true}}},
		},
		Tools: []llm.Tool{{Name: "health", Description: "Get health", InputSchema: json.RawMessage(`{"type":"object"}`)}},
	}
	if err := request.Validate(); err != nil {
		t.Fatal(err)
	}
	wire, err := newCreateRequest("model", 512, request)
	if err != nil {
		t.Fatal(err)
	}
	if wire.Model != "model" || wire.MaxTokens != 512 || !wire.Stream || len(wire.System) != 1 || wire.System[0].Text != "instructions\n\nmore instructions" {
		t.Fatalf("unexpected request: %#v", wire)
	}
	if len(wire.Tools) != 1 || string(wire.Tools[0].InputSchema) != `{"type":"object"}` {
		t.Fatalf("unexpected tools: %#v", wire.Tools)
	}
	if len(wire.Messages) != 3 || wire.Messages[1].Role != "assistant" || len(wire.Messages[1].Content) != 3 || wire.Messages[2].Role != "user" || len(wire.Messages[2].Content) != 2 {
		t.Fatalf("unexpected messages: %#v", wire.Messages)
	}
	result := wire.Messages[2].Content[1]
	if result.Type != "tool_result" || result.ToolUseID != "call-2" || !result.IsError || result.Content != `{"error":"unavailable"}` {
		t.Fatalf("unexpected result: %#v", result)
	}
	body, err := json.Marshal(wire)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"store", "thinking", "authorization", "api_key"} {
		if _, exists := fields[field]; exists {
			t.Fatalf("unexpected field %s", field)
		}
	}
}

// The tools and the system prompt are cached, then the conversation up to
// its last block: two cache points, of the four a request may hold.
func TestCreateRequestMarksTheCachePoints(t *testing.T) {
	user := llm.Message{Role: llm.RoleUser, Text: "health"}
	call := llm.Message{Role: llm.RoleAssistant, Text: "checking", ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "health", Arguments: json.RawMessage(`{}`)}}}
	result := llm.Message{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call-1", Content: json.RawMessage(`{}`)}}}
	for name, messages := range map[string][]llm.Message{
		"prompt":       {{Role: llm.RoleSystem, Text: "instructions"}, user},
		"tool results": {{Role: llm.RoleSystem, Text: "instructions"}, user, call, result},
		"no system":    {user, call, result},
	} {
		t.Run(name, func(t *testing.T) {
			wire, err := newCreateRequest("model", 100, llm.Request{Messages: messages})
			if err != nil {
				t.Fatal(err)
			}
			body, err := json.Marshal(wire)
			if err != nil {
				t.Fatal(err)
			}
			marks := strings.Count(string(body), `"cache_control":{"type":"ephemeral"}`)
			want := 1
			if len(wire.System) != 0 {
				want = 2
				if wire.System[0].CacheControl == nil {
					t.Fatalf("system not cached: %s", body)
				}
			}
			last := wire.Messages[len(wire.Messages)-1]
			if marks != want || last.Content[len(last.Content)-1].CacheControl == nil {
				t.Fatalf("cache points: %s", body)
			}
		})
	}
}

// The volatile text of a prompt goes after a cache point, in a block of its
// own: the next turns, the prompt given without it, read the conversation up
// to the prompt from the cache. Three cache points then, of four.
func TestCreateRequestCachesThePromptBeforeItsVolatileText(t *testing.T) {
	system := llm.Message{Role: llm.RoleSystem, Text: "instructions"}
	prompt := llm.Message{Role: llm.RoleUser, Text: "why is it down?", Volatile: "<interface_context>[]</interface_context>"}
	call := llm.Message{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "health", Arguments: json.RawMessage(`{}`)}}}
	result := llm.Message{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call-1", Content: json.RawMessage(`{}`)}}}
	for name, messages := range map[string][]llm.Message{
		"first request": {system, prompt},
		"tool results":  {system, prompt, call, result},
	} {
		t.Run(name, func(t *testing.T) {
			wire, err := newCreateRequest("model", 100, llm.Request{Messages: messages})
			if err != nil {
				t.Fatal(err)
			}
			body, err := json.Marshal(wire)
			if err != nil {
				t.Fatal(err)
			}
			blocks := wire.Messages[0].Content
			if len(blocks) != 2 || blocks[0].Text != prompt.Text || blocks[0].CacheControl == nil || blocks[1].Text != prompt.Volatile {
				t.Fatalf("prompt blocks: %s", body)
			}
			last := wire.Messages[len(wire.Messages)-1]
			if strings.Count(string(body), `"cache_control"`) != 3 || last.Content[len(last.Content)-1].CacheControl == nil {
				t.Fatalf("cache points: %s", body)
			}
		})
	}
	wire, err := newCreateRequest("model", 100, llm.Request{Messages: []llm.Message{system, prompt}, NoToolCall: true})
	if err != nil {
		t.Fatal(err)
	}
	if body, _ := json.Marshal(wire); strings.Count(string(body), `"cache_control"`) != 1 || len(wire.Messages[0].Content) != 2 {
		t.Fatalf("the last request caches the prompt: %s", body)
	}
}

// The last request of a turn takes no call: its tool_choice invalidates the
// cache of the conversation, whose tail is not written.
func TestCreateRequestCachesNoTailOnTheLastRequest(t *testing.T) {
	wire, err := newCreateRequest("model", 100, llm.Request{Messages: []llm.Message{{Role: llm.RoleSystem, Text: "instructions"}, {Role: llm.RoleUser, Text: "health"}}, NoToolCall: true})
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(wire)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(string(body), `"cache_control"`) != 1 || wire.System[0].CacheControl == nil {
		t.Fatalf("cache points: %s", body)
	}
}

func TestCreateRequestMergesAdjacentRoles(t *testing.T) {
	request := llm.Request{Messages: []llm.Message{
		{Role: llm.RoleUser, Text: "first"}, {Role: llm.RoleUser, Text: "second"},
		{Role: llm.RoleAssistant, Text: "one"}, {Role: llm.RoleAssistant, Text: "two"},
		{Role: llm.RoleUser, Text: "next"},
	}}
	wire, err := newCreateRequest("model", 100, request)
	if err != nil {
		t.Fatal(err)
	}
	if len(wire.Messages) != 3 || len(wire.Messages[0].Content) != 2 || len(wire.Messages[1].Content) != 2 {
		t.Fatalf("messages not merged: %#v", wire.Messages)
	}
}

func TestCreateRequestRejectsUnrepresentableHistory(t *testing.T) {
	user := llm.Message{Role: llm.RoleUser, Text: "hello"}
	call := llm.Message{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call", Name: "health", Arguments: json.RawMessage(`{}`)}}}
	result := llm.Message{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call", Content: json.RawMessage(`{}`)}}}
	for name, history := range map[string][]llm.Message{
		"late system":      {user, {Role: llm.RoleSystem, Text: "late"}},
		"unmatched result": {user, result},
		"missing result":   {user, call},
		"intervening user": {user, call, user, result},
		"duplicate result": {user, call, result, result},
		"duplicate call":   {user, call, result, call, result},
		"system only":      {{Role: llm.RoleSystem, Text: "system"}},
		"assistant first":  {{Role: llm.RoleAssistant, Text: "hello"}, user},
		"prefill":          {user, {Role: llm.RoleAssistant, Text: "hello"}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := newCreateRequest("model", 100, llm.Request{Messages: history}); err == nil || !strings.Contains(err.Error(), "Messages") {
				t.Fatalf("error = %v", err)
			}
		})
	}
}
