package agent

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/opensvc/ai-agent/internal/llm"
)

func callTurn(prompt string, ids ...string) []llm.Message {
	calls := make([]llm.ToolCall, 0, len(ids))
	results := make([]llm.ToolResult, 0, len(ids))
	for _, id := range ids {
		calls = append(calls, llm.ToolCall{ID: id, Name: "get_node_status", Arguments: json.RawMessage(`{}`)})
		results = append(results, llm.ToolResult{CallID: id, Content: json.RawMessage(`{"content":[]}`)})
	}
	return []llm.Message{
		{Role: llm.RoleUser, Text: prompt},
		{Role: llm.RoleAssistant, ToolCalls: calls},
		{Role: llm.RoleTool, ToolResults: results},
		{Role: llm.RoleAssistant, Text: "done"},
	}
}

func callIDs(messages []llm.Message) (calls []string, results []string) {
	for _, message := range messages {
		for _, call := range message.ToolCalls {
			calls = append(calls, call.ID)
		}
		for _, result := range message.ToolResults {
			results = append(results, result.CallID)
		}
	}
	return calls, results
}

// The IDs of the providers of the agent go as they are, whoever gave them:
// the prefix the providers cache stays the same.
func TestSendableCallIDsKeepTheIDsOfTheProviders(t *testing.T) {
	messages := append(callTurn("first", "toolu_01AbCdEfGhIjKlMnOpQrStUv", "call_Xb3kQ9pL2mN7vR4tY8wZ1aC5"), callTurn("second", "Ab3dE5gH1", "chatcmpl-tool-1f2e3d4c5b6a")...)
	sent := withSendableCallIDs(messages)
	if &sent[0] != &messages[0] {
		t.Fatal("messages without an ID to rename were copied")
	}
}

// A server numbering its calls from zero at each answer gives the IDs of
// calls before: renamed, with their results, in the request only.
func TestSendableCallIDsRenameAnIDOfACallBefore(t *testing.T) {
	messages := append(callTurn("first", "call_0", "call_1"), callTurn("second", "call_0")...)
	original, _ := json.Marshal(messages)
	sent := withSendableCallIDs(messages)
	calls, results := callIDs(sent)
	if calls[0] != "call_0" || calls[1] != "call_1" || calls[2] == "call_0" || !sendableCallID(calls[2]) || !reflect.DeepEqual(calls, results) {
		t.Fatalf("calls = %v, results = %v", calls, results)
	}
	// The same at each request, and the history untouched.
	if again, _ := callIDs(withSendableCallIDs(messages)); !reflect.DeepEqual(again, calls) {
		t.Fatalf("renamed %v, then %v", calls, again)
	}
	if after, _ := json.Marshal(messages); string(after) != string(original) {
		t.Fatal("the history was changed")
	}
	if err := (llm.Request{Messages: sent}).Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestSendableCallIDsRenameAnIDOutOfTheirForm(t *testing.T) {
	for _, id := range []string{"call:1", "call.1", "call 1", "é1", strings.Repeat("a", maxSendableCallIDBytes+1)} {
		calls, results := callIDs(withSendableCallIDs(callTurn("check", id, "call_ok")))
		if calls[0] == id || !sendableCallID(calls[0]) || calls[1] != "call_ok" || !reflect.DeepEqual(calls, results) {
			t.Fatalf("%q: calls = %v, results = %v", id, calls, results)
		}
	}
	// A derived ID never takes an ID of the request.
	used := map[string]bool{}
	first := derivedCallID("call:1", 1, 0, used)
	used[first] = true
	if second := derivedCallID("call:1", 1, 0, used); second == first || !sendableCallID(second) {
		t.Fatalf("derived %q then %q", first, second)
	}
}

// A conversation of a server reusing its IDs goes on with a model of the
// Messages protocol, which refuses two calls of one ID: the request it
// gets holds one call per ID.
func TestRunTurnSendsTheModelSendableCallIDs(t *testing.T) {
	history := append(callTurn("first", "call_0"), callTurn("second", "call_0")...)
	model := &scriptedLLM{t: t, steps: []llmStep{answerStep(t, func(request llm.Request) {
		calls, results := callIDs(request.Messages)
		if len(calls) != 2 || calls[0] == calls[1] || !reflect.DeepEqual(calls, results) {
			t.Fatalf("calls = %v, results = %v", calls, results)
		}
	})}}
	agent := newTestAgent(t, model, &fakeSession{}, 4)
	if _, err := agent.RunTurn(t.Context(), history, "third", func(Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
}
