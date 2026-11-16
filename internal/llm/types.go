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

type EventType string

const (
	EventTextDelta EventType = "text_delta"
	EventToolCall  EventType = "tool_call"
	EventUsage     EventType = "usage"
	EventCompleted EventType = "completed"
)

// Event is one provider-neutral item emitted by Client.Stream. Provider errors
// are returned as Go errors rather than encoded as events.
type Event struct {
	Type         EventType
	TextDelta    string
	ToolCall     *ToolCall
	Usage        *Usage
	FinishReason FinishReason
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
