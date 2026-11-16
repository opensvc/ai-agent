package llm

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestRequestValidate(t *testing.T) {
	valid := Request{
		Messages: []Message{
			{Role: RoleSystem, Text: "Diagnose the OpenSVC cluster."},
			{Role: RoleUser, Text: "Is the cluster healthy?"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call-1", Name: "cluster_health", Arguments: json.RawMessage(`{}`)}}},
			{Role: RoleTool, ToolResults: []ToolResult{{CallID: "call-1", Content: json.RawMessage(`{"status":"healthy"}`)}}},
		},
		Tools: []Tool{{
			Name:        "cluster_health",
			Description: "Get cluster health",
			InputSchema: json.RawMessage(`{"type":"object","properties":{}}`),
		}},
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid request rejected: %v", err)
	}

	for _, test := range []struct {
		name   string
		mutate func(*Request)
		want   string
	}{
		{name: "no messages", mutate: func(r *Request) { r.Messages = nil }, want: "no messages"},
		{name: "duplicate tool", mutate: func(r *Request) { r.Tools = append(r.Tools, r.Tools[0]) }, want: "duplicated"},
		{name: "invalid schema", mutate: func(r *Request) { r.Tools[0].InputSchema = json.RawMessage(`[]`) }, want: "JSON object"},
		{name: "unsupported role", mutate: func(r *Request) { r.Messages[0].Role = Role("developer") }, want: "unsupported role"},
		{name: "user tool call", mutate: func(r *Request) {
			r.Messages[1].ToolCalls = []ToolCall{{ID: "call-2", Name: "tool", Arguments: json.RawMessage(`{}`)}}
		}, want: "contains tool content"},
		{name: "invalid arguments", mutate: func(r *Request) { r.Messages[2].ToolCalls[0].Arguments = json.RawMessage(`[]`) }, want: "arguments"},
		{name: "invalid result", mutate: func(r *Request) { r.Messages[3].ToolResults[0].Content = json.RawMessage(`invalid`) }, want: "valid JSON"},
		{name: "volatile system text", mutate: func(r *Request) { r.Messages[0].Volatile = "screen" }, want: "volatile"},
		{name: "invalid volatile text", mutate: func(r *Request) { r.Messages[1].Volatile = "\xff" }, want: "volatile"},
		{name: "two volatile texts", mutate: func(r *Request) {
			r.Messages[1].Volatile = "screen"
			r.Messages = append(r.Messages, Message{Role: RoleAssistant, Text: "done"}, Message{Role: RoleUser, Text: "next", Volatile: "screen"})
		}, want: "after another one"},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := cloneRequest(valid)
			test.mutate(&request)
			err := request.Validate()
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Validate() error = %v, want containing %q", err, test.want)
			}
		})
	}
}

func TestEventValidate(t *testing.T) {
	validEvents := []Event{
		{Type: EventTextDelta, TextDelta: "healthy"},
		{Type: EventToolCall, ToolCall: &ToolCall{ID: "call-1", Name: "cluster_health", Arguments: json.RawMessage(`{}`)}},
		{Type: EventUsage, Usage: &Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15}},
		{Type: EventCompleted, FinishReason: FinishReasonCompleted},
	}
	for _, event := range validEvents {
		if err := event.Validate(); err != nil {
			t.Errorf("event %+v rejected: %v", event, err)
		}
	}

	invalidEvents := []Event{
		{},
		{Type: EventTextDelta},
		{Type: EventToolCall},
		{Type: EventUsage, Usage: &Usage{InputTokens: -1}},
		{Type: EventCompleted},
		{Type: EventCompleted, FinishReason: FinishReasonCompleted, TextDelta: "unexpected"},
	}
	for _, event := range invalidEvents {
		if err := event.Validate(); err == nil {
			t.Errorf("invalid event %+v accepted", event)
		}
	}
}

func TestClientContractStreamsOrderedEvents(t *testing.T) {
	client := testClient{}
	var got []EventType
	err := client.Stream(t.Context(), Request{Messages: []Message{{Role: RoleUser, Text: "health"}}}, func(event Event) error {
		got = append(got, event.Type)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream() error: %v", err)
	}
	want := []EventType{EventTextDelta, EventCompleted}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("got events %v, want %v", got, want)
	}
}

func TestClientContractPropagatesConsumerError(t *testing.T) {
	want := errors.New("stop consuming")
	err := (testClient{}).Stream(context.Background(), Request{Messages: []Message{{Role: RoleUser, Text: "health"}}}, func(Event) error {
		return want
	})
	if !errors.Is(err, want) {
		t.Fatalf("Stream() error = %v, want %v", err, want)
	}
}

type testClient struct{}

var _ Client = testClient{}

func (testClient) Stream(_ context.Context, request Request, emit EmitFunc) error {
	if err := request.Validate(); err != nil {
		return err
	}
	for _, event := range []Event{
		{Type: EventTextDelta, TextDelta: "healthy"},
		{Type: EventCompleted, FinishReason: FinishReasonCompleted},
	} {
		if err := emit(event); err != nil {
			return err
		}
	}
	return nil
}

func cloneRequest(request Request) Request {
	clone := request
	clone.Messages = append([]Message(nil), request.Messages...)
	for i := range clone.Messages {
		clone.Messages[i].ToolCalls = append([]ToolCall(nil), request.Messages[i].ToolCalls...)
		clone.Messages[i].ToolResults = append([]ToolResult(nil), request.Messages[i].ToolResults...)
	}
	clone.Tools = append([]Tool(nil), request.Tools...)
	return clone
}
