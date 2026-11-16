package messages

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/opensvc/ai-agent/internal/llm"
)

const startEvent = `{"type":"message_start","message":{"type":"message","role":"assistant","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":1}}}`
const textStartEvent = `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`
const textDeltaEvent = `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"OK"}}`
const blockStopEvent = `{"type":"content_block_stop","index":0}`
const messageDeltaEvent = `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`
const stopEvent = `{"type":"message_stop"}`
const toolStartEvent = `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call-1","name":"health","input":{}}}`
const toolDeltaEvent = `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"node\":\"node1\"}"}}`
const toolFinishEvent = `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":8}}`

func sse(events ...string) string {
	var stream strings.Builder
	for _, event := range events {
		var payload struct {
			Type string `json:"type"`
		}
		_ = json.Unmarshal([]byte(event), &payload)
		fmt.Fprintf(&stream, "event: %s\ndata: %s\n\n", payload.Type, event)
	}
	return stream.String()
}

func textStream() string {
	return sse(startEvent, textStartEvent, textDeltaEvent, blockStopEvent, messageDeltaEvent, stopEvent)
}

func collectStream(t *testing.T, stream string) ([]llm.Event, error) {
	t.Helper()
	var events []llm.Event
	err := consumeStream(strings.NewReader(stream), "", func(event llm.Event) error {
		if err := event.Validate(); err != nil {
			t.Fatalf("invalid neutral event: %v", err)
		}
		events = append(events, event)
		return nil
	})
	return events, err
}

func TestStreamTextAndCumulativeUsage(t *testing.T) {
	start := strings.Replace(startEvent, `"output_tokens":1`, `"output_tokens":1,"cache_creation_input_tokens":3,"cache_read_input_tokens":4`, 1)
	stream := sse(start, `{"type":"ping"}`, `{"type":"future_informational_event"}`, textStartEvent, textDeltaEvent, blockStopEvent,
		`{"type":"message_delta","delta":{},"usage":{"output_tokens":2}}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`, stopEvent)
	events, err := collectStream(t, stream)
	if err != nil {
		t.Fatal(err)
	}
	want := []llm.Event{
		{Type: llm.EventTextDelta, TextDelta: "OK"},
		{Type: llm.EventUsage, Usage: &llm.Usage{InputTokens: 17, OutputTokens: 5, TotalTokens: 22, CacheReadTokens: 4, CacheWriteTokens: 3}},
		{Type: llm.EventCompleted, FinishReason: llm.FinishReasonCompleted},
	}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("events = %#v, want %#v", events, want)
	}
}

func TestStreamToolArgumentsAndMultipleCalls(t *testing.T) {
	stream := sse(startEvent, toolStartEvent,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"node\":"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"node1\"}"}}`, blockStopEvent,
		`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call-2","name":"health","input":{}}}`,
		`{"type":"content_block_stop","index":1}`, toolFinishEvent, stopEvent)
	events, err := collectStream(t, stream)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 4 || events[0].Type != llm.EventToolCall || events[0].ToolCall.ID != "call-1" || string(events[0].ToolCall.Arguments) != `{"node":"node1"}` || events[1].ToolCall.ID != "call-2" || string(events[1].ToolCall.Arguments) != `{}` || events[3].FinishReason != llm.FinishReasonToolCalls {
		t.Fatalf("events = %#v", events)
	}
}

func TestStreamSupportsInitialBlockContent(t *testing.T) {
	for name, block := range map[string]string{
		"text": strings.Replace(textStartEvent, `"text":""`, `"text":"initial"`, 1),
		"tool": strings.Replace(toolStartEvent, `"input":{}`, `"input":{"node":"node1"}`, 1),
	} {
		t.Run(name, func(t *testing.T) {
			finish := messageDeltaEvent
			if name == "tool" {
				finish = toolFinishEvent
			}
			events, err := collectStream(t, sse(startEvent, block, blockStopEvent, finish, stopEvent))
			if err != nil {
				t.Fatal(err)
			}
			if name == "text" && events[0].TextDelta != "initial" {
				t.Fatalf("events = %#v", events)
			}
			if name == "tool" && string(events[0].ToolCall.Arguments) != `{"node":"node1"}` {
				t.Fatalf("events = %#v", events)
			}
		})
	}
}

const thinkingStartEvent = `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`

// The thinking of the recent models, adaptive by default, is never shown:
// it comes back as the reasoning of the answer, its blocks in their order,
// bound to the fingerprint of the request.
func TestStreamKeepsThinkingForTheTurn(t *testing.T) {
	stream := sse(startEvent,
		thinkingStartEvent,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"private reasoning"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"c2lnbmF0dXJl"}}`,
		blockStopEvent,
		`{"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"opaque"}}`,
		`{"type":"content_block_stop","index":1}`,
		strings.Replace(textStartEvent, `"index":0`, `"index":2`, 1),
		strings.Replace(textDeltaEvent, `"index":0`, `"index":2`, 1),
		`{"type":"content_block_stop","index":2}`,
		strings.Replace(toolStartEvent, `"index":0`, `"index":3`, 1),
		strings.Replace(toolDeltaEvent, `"index":0`, `"index":3`, 1),
		`{"type":"content_block_stop","index":3}`,
		toolFinishEvent, stopEvent)
	var events []llm.Event
	err := consumeStream(strings.NewReader(stream), "request-fingerprint", func(event llm.Event) error {
		if err := event.Validate(); err != nil {
			t.Fatalf("invalid neutral event: %v", err)
		}
		events = append(events, event)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	var types []llm.EventType
	for _, event := range events {
		types = append(types, event.Type)
		if strings.Contains(event.TextDelta, "private") || strings.Contains(event.TextDelta, "opaque") {
			t.Fatalf("thinking shown: %#v", event)
		}
	}
	want := []llm.EventType{llm.EventTextDelta, llm.EventToolCall, llm.EventReasoning, llm.EventUsage, llm.EventCompleted}
	if !reflect.DeepEqual(types, want) || events[0].TextDelta != "OK" || string(events[1].ToolCall.Arguments) != `{"node":"node1"}` {
		t.Fatalf("events = %#v", events)
	}
	reasoning := events[2].Reasoning
	wantData := `{"fingerprint":"request-fingerprint","content":[` +
		`{"type":"thinking","thinking":"private reasoning","signature":"c2lnbmF0dXJl"},` +
		`{"type":"redacted_thinking","data":"opaque"},` +
		`{"type":"text","text":"OK"},` +
		`{"type":"tool_use","id":"call-1","name":"health","input":{"node":"node1"}}]}`
	if reasoning.Protocol != Protocol || string(reasoning.Data) != wantData {
		t.Fatalf("reasoning = %s %s", reasoning.Protocol, reasoning.Data)
	}
}

// An answer without thinking has no reasoning to give back.
func TestStreamWithoutThinkingHasNoReasoning(t *testing.T) {
	events, err := collectStream(t, sse(startEvent, toolStartEvent, toolDeltaEvent, blockStopEvent, toolFinishEvent, stopEvent))
	if err != nil {
		t.Fatal(err)
	}
	for _, event := range events {
		if event.Type == llm.EventReasoning {
			t.Fatalf("reasoning without thinking: %#v", event)
		}
	}
}

func TestStreamSSEFraming(t *testing.T) {
	stream := strings.Replace(textStream(), `data: `+textDeltaEvent, "data: {\"type\":\"content_block_delta\",\ndata: \"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"OK\"}}", 1)
	stream = ": comment\nretry: 1000\nid: ignored\n" + strings.ReplaceAll(stream, "\n", "\r\n")
	stream = strings.TrimRight(stream, "\r\n")
	events, err := collectStream(t, stream)
	if err != nil || len(events) != 3 {
		t.Fatalf("events=%#v error=%v", events, err)
	}
}

func TestStreamRejectsInvalidResponsesWithoutToolCallsOrCompletion(t *testing.T) {
	for name, stream := range map[string]string{
		"empty":                    "",
		"missing start":            sse(textStartEvent, blockStopEvent, messageDeltaEvent, stopEvent),
		"duplicate start":          sse(startEvent, startEvent),
		"missing stop":             sse(startEvent, toolStartEvent, toolDeltaEvent, blockStopEvent, toolFinishEvent),
		"incomplete block":         sse(startEvent, toolStartEvent, toolDeltaEvent, toolFinishEvent, stopEvent),
		"missing reason":           sse(startEvent, stopEvent),
		"invalid index":            sse(startEvent, strings.Replace(toolStartEvent, `"index":0`, `"index":-1`, 1)),
		"missing index":            sse(startEvent, strings.Replace(toolStartEvent, `"index":0,`, ``, 1)),
		"missing block":            sse(startEvent, `{"type":"content_block_start","index":0}`),
		"unknown delta index":      sse(startEvent, textDeltaEvent),
		"delta after block stop":   sse(startEvent, textStartEvent, blockStopEvent, textDeltaEvent),
		"content after finish":     sse(startEvent, messageDeltaEvent, textStartEvent),
		"mismatched delta":         sse(startEvent, toolStartEvent, textDeltaEvent),
		"missing delta":            sse(startEvent, textStartEvent, `{"type":"content_block_delta","index":0}`),
		"unknown block":            sse(startEvent, strings.Replace(toolStartEvent, `"tool_use"`, `"server_tool_use"`, 1)),
		"text delta in thinking":   sse(startEvent, thinkingStartEvent, `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"private"}}`),
		"delta in redacted":        sse(startEvent, `{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"opaque"}}`, `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"private"}}`),
		"invalid args":             sse(startEvent, toolStartEvent, strings.Replace(toolDeltaEvent, `{\"node\":\"node1\"}`, `[1]`, 1), blockStopEvent, toolFinishEvent, stopEvent),
		"truncated args":           sse(startEvent, toolStartEvent, strings.Replace(toolDeltaEvent, `{\"node\":\"node1\"}`, `{\"node\":`, 1), blockStopEvent, toolFinishEvent, stopEvent),
		"conflicting input":        sse(startEvent, strings.Replace(toolStartEvent, `"input":{}`, `"input":{"node":"node2"}`, 1), toolDeltaEvent),
		"tool reason without tool": sse(startEvent, toolFinishEvent, stopEvent),
		"tool with text reason":    sse(startEvent, toolStartEvent, toolDeltaEvent, blockStopEvent, messageDeltaEvent, stopEvent),
		"truncated tool":           sse(startEvent, toolStartEvent, toolDeltaEvent, blockStopEvent, strings.Replace(toolFinishEvent, `"tool_use"`, `"max_tokens"`, 1), stopEvent),
		"duplicate tool ID":        sse(startEvent, toolStartEvent, blockStopEvent, strings.Replace(toolStartEvent, `"index":0`, `"index":1`, 1)),
		"duplicate stop":           textStream() + sse(stopEvent),
		"malformed trailing JSON":  textStream() + "data: {broken}\n\n",
		"mismatched event":         "event: ping\ndata: " + startEvent + "\n\n",
		"negative usage":           sse(strings.Replace(startEvent, `"input_tokens":10`, `"input_tokens":-1`, 1)),
		"decreasing usage":         sse(startEvent, `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}`, stopEvent),
		"overflow":                 sse(strings.Replace(startEvent, `"input_tokens":10`, `"input_tokens":9223372036854775807`, 1), messageDeltaEvent, stopEvent),
		"pause turn":               sse(startEvent, strings.Replace(messageDeltaEvent, `"end_turn"`, `"pause_turn"`, 1), stopEvent),
		"changed reason":           sse(startEvent, messageDeltaEvent, strings.Replace(messageDeltaEvent, `"end_turn"`, `"max_tokens"`, 1), stopEvent),
	} {
		t.Run(name, func(t *testing.T) {
			events, err := collectStream(t, stream)
			if err == nil {
				t.Fatal("invalid stream accepted")
			}
			for _, event := range events {
				if event.Type == llm.EventToolCall || event.Type == llm.EventCompleted {
					t.Fatalf("unsafe event after invalid stream: %#v", event)
				}
			}
		})
	}
}

func TestStreamProviderErrorsDoNotExposeRawPayload(t *testing.T) {
	for _, providerType := range []string{"overloaded_error", "secret-marker"} {
		_, err := collectStream(t, sse(fmt.Sprintf(`{"type":"error","error":{"type":%q,"message":"secret-marker user prompt"}}`, providerType)))
		if err == nil || strings.Contains(err.Error(), "secret-marker") || strings.Contains(err.Error(), "user prompt") {
			t.Fatalf("unsafe error: %v", err)
		}
	}
}

func TestStreamPropagatesConsumerErrors(t *testing.T) {
	want := errors.New("stop")
	err := consumeStream(strings.NewReader(textStream()), "", func(llm.Event) error { return want })
	if !errors.Is(err, want) {
		t.Fatalf("error = %v, want %v", err, want)
	}
}

func TestStreamBounds(t *testing.T) {
	part := strings.Repeat("x", maxEventBytes/3+1)
	for name, stream := range map[string]string{
		"line":   "data: " + strings.Repeat("x", maxLineBytes),
		"event":  "data: " + part + "\ndata: " + part + "\ndata: " + part + "\n\n",
		"stream": strings.Repeat(": ignored\n", maxStreamBytes/10+1),
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := collectStream(t, stream); err == nil {
				t.Fatal("oversized stream accepted")
			}
		})
	}
	var events []string
	events = append(events, startEvent, toolStartEvent)
	for i := 0; i < 3; i++ {
		events = append(events, fmt.Sprintf(`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":%q}}`, part))
	}
	if _, err := collectStream(t, sse(events...)); err == nil || !strings.Contains(err.Error(), "arguments exceed") {
		t.Fatalf("argument bound error = %v", err)
	}
	events = []string{startEvent}
	for i := 0; i <= maxToolCalls; i++ {
		events = append(events, fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"tool_use","id":"call-%d","name":"health","input":{}}}`, i, i), fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, i))
	}
	if _, err := collectStream(t, sse(events...)); err == nil || !strings.Contains(err.Error(), "count exceeds") {
		t.Fatalf("call count bound error = %v", err)
	}
}

func TestMapFinishReason(t *testing.T) {
	for input, want := range map[string]llm.FinishReason{
		"end_turn": llm.FinishReasonCompleted, "stop_sequence": llm.FinishReasonCompleted,
		"tool_use": llm.FinishReasonToolCalls, "max_tokens": llm.FinishReasonLength,
		"model_context_window_exceeded": llm.FinishReasonLength, "refusal": llm.FinishReasonContentFilter,
		"unknown": llm.FinishReasonOther,
	} {
		if got := mapFinishReason(input); got != want {
			t.Fatalf("mapFinishReason(%q)=%q, want %q", input, got, want)
		}
	}
}
