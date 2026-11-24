package llm

import "encoding/json"

// Request contains the conversation state and tools available for one model
// turn. Model selection and provider configuration belong to the Client.
type Request struct {
	Messages []Message
	Tools    []Tool
	// NoToolCall asks for an answer in text: the tools stay declared, as
	// the history may hold calls of them, and none may be called.
	NoToolCall bool
}

type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Message represents text, assistant tool calls, or tool results in the
// conversation. Validate enforces the content allowed for each role.
type Message struct {
	Role        Role
	Text        string
	ToolCalls   []ToolCall
	ToolResults []ToolResult
	// Reasoning is the reasoning of the model behind an assistant message,
	// which the adapter that produced it gives back to the model in the next
	// requests of the same turn. It lives as long as the turn runs: never
	// stored, nor kept when the turn waits on the user.
	Reasoning *Reasoning `json:"-"`
	// Volatile is text of a user message the model reads after Text, which
	// later requests do not repeat, such as what the screen of the user
	// showed with a prompt. An adapter keeps it out of the prefix it caches,
	// so that the next requests, the message given without it, still find
	// the prefix up to Text. One message a request at most. Never stored.
	Volatile string `json:"-"`
}

// TextWithVolatile gives the text of a message and its volatile text after
// it, for a protocol that takes one text per message.
func (m Message) TextWithVolatile() string {
	if m.Volatile == "" {
		return m.Text
	}
	return m.Text + "\n\n" + m.Volatile
}

// MaxReasoningBytes bounds the reasoning of one assistant message.
const MaxReasoningBytes = 1 << 20

// Reasoning is opaque to all but the adapter of its Protocol, which another
// adapter ignores: the reasoning of the model in the wire form of its
// protocol, such as the thinking blocks of the Messages protocol.
type Reasoning struct {
	Protocol string
	Data     json.RawMessage
}

// Tool describes a callable function using a JSON Schema input object.
type Tool struct {
	Name        string
	Description string
	InputSchema json.RawMessage
}

// ToolCall is a complete model-requested call. Protocol adapters accumulate
// streamed argument fragments before emitting it.
type ToolCall struct {
	ID        string
	Name      string
	Arguments json.RawMessage
}

// ToolResult associates a JSON-encoded MCP result with a model tool call.
type ToolResult struct {
	CallID  string
	Content json.RawMessage
	IsError bool
}

// PlainText gives the text of a result made of one text content alone,
// without error, structured content or any other field, but an isError
// false: an adapter sends it as it is, rather than in its JSON envelope,
// whose escapes of line breaks and quotes cost tokens and bury code and
// documentation. It reports false for any other result.
func (r ToolResult) PlainText() (string, bool) {
	if r.IsError {
		return "", false
	}
	var envelope map[string]json.RawMessage
	if json.Unmarshal(r.Content, &envelope) != nil {
		return "", false
	}
	for key, value := range envelope {
		if key != "content" && (key != "isError" || string(value) != "false") {
			return "", false
		}
	}
	var contents []map[string]json.RawMessage
	if json.Unmarshal(envelope["content"], &contents) != nil || len(contents) != 1 || len(contents[0]) != 2 || string(contents[0]["type"]) != `"text"` {
		return "", false
	}
	// An empty text keeps its envelope: no result was ever sent empty.
	var text string
	if json.Unmarshal(contents[0]["text"], &text) != nil || text == "" {
		return "", false
	}
	return text, true
}

type EventType string

const (
	EventTextDelta EventType = "text_delta"
	EventToolCall  EventType = "tool_call"
	EventUsage     EventType = "usage"
	EventCompleted EventType = "completed"
	// EventReasoning gives, before the completion, the reasoning of the
	// answer for the next requests of the turn.
	EventReasoning EventType = "reasoning"
)

// Event is one provider-neutral item emitted by Client.Stream. Provider errors
// are returned as Go errors rather than encoded as events.
type Event struct {
	Type         EventType
	TextDelta    string
	ToolCall     *ToolCall
	Usage        *Usage
	FinishReason FinishReason
	Reasoning    *Reasoning
}

// Usage reports the common token counters exposed by LLM protocols.
// InputTokens counts every input token, those read from or written to the
// cache of the provider included; CacheReadTokens and CacheWriteTokens tell
// how many of them were, 0 when the protocol does not say.
type Usage struct {
	InputTokens      int64
	OutputTokens     int64
	TotalTokens      int64
	CacheReadTokens  int64
	CacheWriteTokens int64
}

type FinishReason string

const (
	FinishReasonCompleted     FinishReason = "completed"
	FinishReasonToolCalls     FinishReason = "tool_calls"
	FinishReasonLength        FinishReason = "length"
	FinishReasonContentFilter FinishReason = "content_filter"
	FinishReasonOther         FinishReason = "other"
)
