package responses

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/opensvc/ai-agent/internal/llm"
)

const completedWithReasoning = `{"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2},"output":[` +
	`{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"secret-1"},` +
	`{"type":"function_call","id":"fc_1","call_id":"call-1","name":"health","arguments":"{}"},` +
	`{"type":"reasoning","id":"rs_2","summary":[],"encrypted_content":"secret-2"},` +
	`{"type":"function_call","id":"fc_2","call_id":"call-2","name":"health","arguments":"{\"node\":\"n1\"}"}]}}`

// The encrypted reasoning of an answer comes back in the order of its
// output, and the next request of the turn gives it back in that order.
func TestReasoningGivenBackInTheOrderOfTheOutput(t *testing.T) {
	var events []llm.Event
	err := consumeStream(strings.NewReader("data: "+completedWithReasoning+"\n\n"), func(event llm.Event) error {
		if err := event.Validate(); err != nil {
			t.Fatal(err)
		}
		events = append(events, event)
		return nil
	}, "")
	if err != nil {
		t.Fatal(err)
	}
	var reasoning *llm.Reasoning
	var calls []llm.ToolCall
	for _, event := range events {
		switch event.Type {
		case llm.EventReasoning:
			reasoning = event.Reasoning
		case llm.EventToolCall:
			calls = append(calls, *event.ToolCall)
		}
	}
	if reasoning == nil || len(calls) != 2 {
		t.Fatalf("events = %#v", events)
	}
	request := llm.Request{Messages: []llm.Message{
		{Role: llm.RoleUser, Text: "health"},
		{Role: llm.RoleAssistant, ToolCalls: calls, Reasoning: reasoning},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call-1", Content: json.RawMessage(`{}`)}, {CallID: "call-2", Content: json.RawMessage(`{}`)}}},
	}}
	wire, err := newCreateRequest("model", 100, request)
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(wire.Input)
	if err != nil {
		t.Fatal(err)
	}
	order := []string{`"encrypted_content":"secret-1"`, `"call_id":"call-1"`, `"encrypted_content":"secret-2"`, `"call_id":"call-2"`, `"type":"function_call_output"`}
	position := 0
	for _, marker := range order {
		next := strings.Index(string(body[position:]), marker)
		if next < 0 {
			t.Fatalf("%s not in order in %s", marker, body)
		}
		position += next + len(marker)
	}
	without, err := buildCreateRequest("model", 100, request, false)
	if err != nil {
		t.Fatal(err)
	}
	if body, _ := json.Marshal(without.Input); strings.Contains(string(body), "secret") {
		t.Fatalf("reasoning given back without replay: %s", body)
	}
}

// Reasoning that does not describe the message, such as one of another
// answer, is not given back.
func TestReasoningOfAnotherAnswerIsNotGivenBack(t *testing.T) {
	reasoning := answerReasoning([]json.RawMessage{
		json.RawMessage(`{"type":"reasoning","encrypted_content":"secret"}`),
		json.RawMessage(`{"type":"function_call","call_id":"call-1"}`),
	})
	if reasoning == nil {
		t.Fatal("no reasoning")
	}
	for name, message := range map[string]llm.Message{
		"other call": {Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call-9", Name: "health", Arguments: json.RawMessage(`{}`)}}, Reasoning: reasoning},
		"added text": {Role: llm.RoleAssistant, Text: "hello", ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "health", Arguments: json.RawMessage(`{}`)}}, Reasoning: reasoning},
		"other protocol": {Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "health", Arguments: json.RawMessage(`{}`)}},
			Reasoning: &llm.Reasoning{Protocol: "messages", Data: reasoning.Data}},
	} {
		if replayOrder(message) != nil {
			t.Fatalf("%s: reasoning given back", name)
		}
	}
	if answerReasoning([]json.RawMessage{json.RawMessage(`{"type":"reasoning","encrypted_content":null}`)}) != nil {
		t.Fatal("reasoning without encrypted content kept")
	}
}
