package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// Validate rejects conversation states that cannot be represented safely by
// the supported text and tool-calling contracts.
func (r Request) Validate() error {
	if len(r.Messages) == 0 {
		return fmt.Errorf("LLM request has no messages")
	}

	toolNames := make(map[string]struct{}, len(r.Tools))
	for i, tool := range r.Tools {
		if strings.TrimSpace(tool.Name) == "" {
			return fmt.Errorf("LLM tool %d has an empty name", i)
		}
		if _, exists := toolNames[tool.Name]; exists {
			return fmt.Errorf("LLM tool name %q is duplicated", tool.Name)
		}
		toolNames[tool.Name] = struct{}{}
		if err := validateJSONObject(tool.InputSchema); err != nil {
			return fmt.Errorf("LLM tool %q input schema: %w", tool.Name, err)
		}
	}

	for i, message := range r.Messages {
		if err := message.validate(); err != nil {
			return fmt.Errorf("LLM message %d: %w", i, err)
		}
	}
	return nil
}

func (m Message) validate() error {
	switch m.Role {
	case RoleSystem, RoleUser:
		if strings.TrimSpace(m.Text) == "" {
			return fmt.Errorf("%s message has empty text", m.Role)
		}
		if len(m.ToolCalls) != 0 || len(m.ToolResults) != 0 {
			return fmt.Errorf("%s message contains tool content", m.Role)
		}
	case RoleAssistant:
		if strings.TrimSpace(m.Text) == "" && len(m.ToolCalls) == 0 {
			return fmt.Errorf("assistant message has no content")
		}
		if len(m.ToolResults) != 0 {
			return fmt.Errorf("assistant message contains tool results")
		}
		for i, call := range m.ToolCalls {
			if err := call.validate(); err != nil {
				return fmt.Errorf("tool call %d: %w", i, err)
			}
		}
	case RoleTool:
		if m.Text != "" || len(m.ToolCalls) != 0 {
			return fmt.Errorf("tool message contains text or tool calls")
		}
		if len(m.ToolResults) == 0 {
			return fmt.Errorf("tool message has no results")
		}
		for i, result := range m.ToolResults {
			if err := result.validate(); err != nil {
				return fmt.Errorf("tool result %d: %w", i, err)
			}
		}
	default:
		return fmt.Errorf("unsupported role %q", m.Role)
	}
	return nil
}

func (c ToolCall) validate() error {
	if strings.TrimSpace(c.ID) == "" {
		return fmt.Errorf("call ID is empty")
	}
	if strings.TrimSpace(c.Name) == "" {
		return fmt.Errorf("tool name is empty")
	}
	if err := validateJSONObject(c.Arguments); err != nil {
		return fmt.Errorf("arguments: %w", err)
	}
	return nil
}

func (r ToolResult) validate() error {
	if strings.TrimSpace(r.CallID) == "" {
		return fmt.Errorf("call ID is empty")
	}
	if len(r.Content) == 0 || !json.Valid(r.Content) {
		return fmt.Errorf("content is not valid JSON")
	}
	return nil
}

// Validate rejects malformed or internally inconsistent stream events.
func (e Event) Validate() error {
	switch e.Type {
	case EventTextDelta:
		if e.TextDelta == "" {
			return fmt.Errorf("text delta is empty")
		}
		if e.ToolCall != nil || e.Usage != nil || e.FinishReason != "" {
			return fmt.Errorf("text delta event contains unrelated fields")
		}
	case EventToolCall:
		if e.ToolCall == nil {
			return fmt.Errorf("tool call event has no call")
		}
		if err := e.ToolCall.validate(); err != nil {
			return fmt.Errorf("tool call event: %w", err)
		}
		if e.TextDelta != "" || e.Usage != nil || e.FinishReason != "" {
			return fmt.Errorf("tool call event contains unrelated fields")
		}
	case EventUsage:
		if e.Usage == nil {
			return fmt.Errorf("usage event has no counters")
		}
		if e.Usage.InputTokens < 0 || e.Usage.OutputTokens < 0 || e.Usage.TotalTokens < 0 || e.Usage.CacheReadTokens < 0 || e.Usage.CacheWriteTokens < 0 {
			return fmt.Errorf("usage event contains negative counters")
		}
		if e.Usage.CacheReadTokens > e.Usage.InputTokens || e.Usage.CacheWriteTokens > e.Usage.InputTokens-e.Usage.CacheReadTokens {
			return fmt.Errorf("usage event counts more cached tokens than input tokens")
		}
		if e.TextDelta != "" || e.ToolCall != nil || e.FinishReason != "" {
			return fmt.Errorf("usage event contains unrelated fields")
		}
	case EventCompleted:
		if !e.FinishReason.valid() {
			return fmt.Errorf("completed event has invalid finish reason %q", e.FinishReason)
		}
		if e.TextDelta != "" || e.ToolCall != nil || e.Usage != nil {
			return fmt.Errorf("completed event contains unrelated fields")
		}
	default:
		return fmt.Errorf("unsupported event type %q", e.Type)
	}
	return nil
}

func (r FinishReason) valid() bool {
	switch r {
	case FinishReasonCompleted, FinishReasonToolCalls, FinishReasonLength, FinishReasonContentFilter, FinishReasonOther:
		return true
	default:
		return false
	}
}

func validateJSONObject(value json.RawMessage) error {
	trimmed := bytes.TrimSpace(value)
	if len(trimmed) == 0 || trimmed[0] != '{' || !json.Valid(trimmed) {
		return fmt.Errorf("value is not a valid JSON object")
	}
	return nil
}
