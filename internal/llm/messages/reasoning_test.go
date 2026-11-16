package messages

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/opensvc/ai-agent/internal/llm"
)

var replayTools = []llm.Tool{{Name: "health", Description: "Get health", InputSchema: json.RawMessage(`{"type":"object"}`)}}

// answer is an assistant message of the turn calling health, with the
// thinking the model gave before the call, bound to fingerprint.
func answer(callID, fingerprint string) llm.Message {
	data := fmt.Sprintf(`{"fingerprint":%q,"content":[{"type":"thinking","thinking":"","signature":"sig-%s"},{"type":"text","text":"checking"},{"type":"tool_use","id":%q,"name":"health","input":{}}]}`, fingerprint, callID, callID)
	return llm.Message{Role: llm.RoleAssistant, Text: "checking", ToolCalls: []llm.ToolCall{{ID: callID, Name: "health", Arguments: json.RawMessage(`{}`)}},
		Reasoning: &llm.Reasoning{Protocol: Protocol, Data: json.RawMessage(data)}}
}

func result(callID string) llm.Message {
	return llm.Message{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: callID, Content: json.RawMessage(`{}`)}}}
}

func turn(answers ...llm.Message) llm.Request {
	messages := []llm.Message{{Role: llm.RoleSystem, Text: "instructions"}, {Role: llm.RoleUser, Text: "health"}}
	for _, message := range answers {
		messages = append(messages, message, result(message.ToolCalls[0].ID))
	}
	return llm.Request{Messages: messages, Tools: replayTools}
}

// thinkingOf lists, per assistant message of the wire request, whether it
// gives its thinking back.
func thinkingOf(t *testing.T, wire createRequest) []bool {
	t.Helper()
	var given []bool
	for _, message := range wire.Messages {
		if message.Role != "assistant" {
			continue
		}
		body, err := json.Marshal(message)
		if err != nil {
			t.Fatal(err)
		}
		given = append(given, strings.Contains(string(body), `"type":"thinking"`))
	}
	return given
}

func TestCreateRequestGivesTheThinkingOfTheTurnBack(t *testing.T) {
	current, err := newCreateRequest("model", 100, turn())
	if err != nil {
		t.Fatal(err)
	}
	fingerprint := current.Fingerprint
	tampered := answer("call-2", fingerprint)
	tampered.Text = "edited"
	for name, test := range map[string]struct {
		request llm.Request
		want    []bool
	}{
		"bound to the request": {turn(answer("call-1", fingerprint), answer("call-2", fingerprint)), []bool{true, true}},
		// The tools changed after the first answer, such as after a search:
		// its thinking cannot be given back, the later one can.
		"tools changed since": {turn(answer("call-1", "before-search"), answer("call-2", fingerprint)), []bool{false, true}},
		// Thinking is left out from the start only, never from the middle.
		"later one unbound": {turn(answer("call-1", fingerprint), answer("call-2", "other")), []bool{false, false}},
		"content changed":   {turn(answer("call-1", fingerprint), tampered), []bool{false, false}},
		"incomplete": {turn(answer("call-1", fingerprint), llm.Message{Role: llm.RoleAssistant, Text: "checking",
			ToolCalls: []llm.ToolCall{{ID: "call-2", Name: "health", Arguments: json.RawMessage(`{}`)}},
			Reasoning: &llm.Reasoning{Protocol: Protocol, Data: json.RawMessage(`{"fingerprint":"` + fingerprint + `","incomplete":true}`)}}), []bool{false, false}},
		"another protocol": {turn(llm.Message{Role: llm.RoleAssistant, Text: "checking", ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "health", Arguments: json.RawMessage(`{}`)}},
			Reasoning: &llm.Reasoning{Protocol: "responses", Data: json.RawMessage(`{}`)}}, answer("call-2", fingerprint)), []bool{false, true}},
	} {
		t.Run(name, func(t *testing.T) {
			if err := test.request.Validate(); err != nil {
				t.Fatal(err)
			}
			wire, err := newCreateRequest("model", 100, test.request)
			if err != nil {
				t.Fatal(err)
			}
			if got := thinkingOf(t, wire); fmt.Sprint(got) != fmt.Sprint(test.want) {
				t.Fatalf("thinking given back = %v, want %v", got, test.want)
			}
		})
	}
}

// The answer given back is the content the model gave, in its order.
func TestCreateRequestGivesTheAnswerBackAsGiven(t *testing.T) {
	current, err := newCreateRequest("model", 100, turn())
	if err != nil {
		t.Fatal(err)
	}
	wire, err := newCreateRequest("model", 100, turn(answer("call-1", current.Fingerprint)))
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(wire.Messages[1])
	if err != nil {
		t.Fatal(err)
	}
	want := `{"role":"assistant","content":[{"type":"thinking","thinking":"","signature":"sig-call-1"},{"type":"text","text":"checking"},{"type":"tool_use","id":"call-1","name":"health","input":{}}]}`
	if string(body) != want {
		t.Fatalf("answer = %s", body)
	}
	without, err := buildCreateRequest("model", 100, turn(answer("call-1", current.Fingerprint)), false)
	if err != nil {
		t.Fatal(err)
	}
	if got := thinkingOf(t, without); fmt.Sprint(got) != "[false]" {
		t.Fatalf("thinking given back without replay: %v", got)
	}
}

// The fingerprint binds the system prompt and the set of the tools, not
// their order.
func TestRequestFingerprint(t *testing.T) {
	a := wireTool{Name: "a", InputSchema: json.RawMessage(`{}`)}
	b := wireTool{Name: "b", InputSchema: json.RawMessage(`{}`)}
	base := requestFingerprint([]string{"s"}, []wireTool{a, b})
	if requestFingerprint([]string{"s"}, []wireTool{b, a}) != base {
		t.Fatal("the order of the tools changed the fingerprint")
	}
	for name, other := range map[string]string{
		"system": requestFingerprint([]string{"t"}, []wireTool{a, b}),
		"tools":  requestFingerprint([]string{"s"}, []wireTool{a}),
	} {
		if other == base {
			t.Fatalf("a change of the %s kept the fingerprint", name)
		}
	}
}

// An answer whose call takes & < or > is given back: the encoding of its
// arguments may differ from that of the message, which are the same call.
func TestAnswerWithEscapedArgumentsGivenBack(t *testing.T) {
	arguments := json.RawMessage(`{"filter":"a&b<c>"}`)
	blocks := []pendingBlock{
		{block: wireBlock{Type: "thinking"}, signature: []byte("sig")},
		{block: wireBlock{Type: "text"}},
		{block: wireBlock{Type: "tool_use", ID: "call-1", Name: "health", Input: json.RawMessage(`{}`)}, arguments: arguments},
	}
	current, err := newCreateRequest("model", 100, turn())
	if err != nil {
		t.Fatal(err)
	}
	reasoning := answerReasoning(blocks, current.Fingerprint)
	// The empty text block is left out: the provider refuses one.
	if reasoning == nil || strings.Contains(string(reasoning.Data), `"type":"text"`) {
		t.Fatalf("reasoning = %#v", reasoning)
	}
	message := llm.Message{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "health", Arguments: arguments}}, Reasoning: reasoning}
	request := turn()
	request.Messages = append(request.Messages, message, result("call-1"))
	wire, err := newCreateRequest("model", 100, request)
	if err != nil {
		t.Fatal(err)
	}
	if got := thinkingOf(t, wire); fmt.Sprint(got) != "[true]" {
		t.Fatalf("thinking given back = %v", got)
	}
}
