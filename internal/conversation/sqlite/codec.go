package sqlite

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"

	"github.com/opensvc/ai-agent/internal/llm"
)

const (
	// maxStoredToolCallsPerMessage is the maximum of tool calls the agent
	// runs per model answer.
	maxStoredToolCallsPerMessage = 8
	maxStoredToolArguments       = 256 << 10
	maxStoredToolResult          = 1 << 20
)

type storedToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type storedToolResult struct {
	CallID  string          `json:"call_id"`
	Content json.RawMessage `json:"content"`
	IsError bool            `json:"is_error"`
}

type encodedMessage struct {
	sequence    int
	role        string
	text        string
	toolCalls   []byte
	toolResults []byte
	storedBytes int64
}

func encodeMessages(messages []llm.Message) ([]encodedMessage, int64, error) {
	if len(messages) == 0 {
		return nil, 0, fmt.Errorf("conversation turn has no messages")
	}
	if err := (llm.Request{Messages: messages}).Validate(); err != nil {
		return nil, 0, fmt.Errorf("validate conversation messages: %w", err)
	}
	if err := validateMessageSequence(messages); err != nil {
		return nil, 0, err
	}
	encoded := make([]encodedMessage, 0, len(messages))
	var total int64
	for index, message := range messages {
		if message.Role == llm.RoleSystem {
			return nil, 0, fmt.Errorf("conversation message %d contains a system prompt", index)
		}
		// The reasoning of the model lives as long as its run.
		if message.Reasoning != nil {
			return nil, 0, fmt.Errorf("conversation message %d contains the reasoning of the model", index)
		}
		if len(message.ToolCalls) > maxStoredToolCallsPerMessage || len(message.ToolResults) > maxStoredToolCallsPerMessage {
			return nil, 0, fmt.Errorf("conversation message %d contains too many tool items", index)
		}
		calls := make([]storedToolCall, 0, len(message.ToolCalls))
		for callIndex, call := range message.ToolCalls {
			if len(call.Arguments) > maxStoredToolArguments {
				return nil, 0, fmt.Errorf("conversation message %d tool call %d arguments exceed %d bytes", index, callIndex, maxStoredToolArguments)
			}
			calls = append(calls, storedToolCall{ID: call.ID, Name: call.Name, Arguments: call.Arguments})
		}
		results := make([]storedToolResult, 0, len(message.ToolResults))
		for resultIndex, result := range message.ToolResults {
			if len(result.Content) > maxStoredToolResult {
				return nil, 0, fmt.Errorf("conversation message %d tool result %d exceeds %d bytes", index, resultIndex, maxStoredToolResult)
			}
			results = append(results, storedToolResult{CallID: result.CallID, Content: result.Content, IsError: result.IsError})
		}
		callData, err := json.Marshal(calls)
		if err != nil {
			return nil, 0, fmt.Errorf("encode conversation message %d tool calls: %w", index, err)
		}
		resultData, err := json.Marshal(results)
		if err != nil {
			return nil, 0, fmt.Errorf("encode conversation message %d tool results: %w", index, err)
		}
		size := int64(len(message.Role) + len(message.Text) + len(callData) + len(resultData))
		if size < 0 || total > int64(^uint64(0)>>1)-size {
			return nil, 0, fmt.Errorf("conversation message size overflow")
		}
		total += size
		encoded = append(encoded, encodedMessage{
			sequence:    index + 1,
			role:        string(message.Role),
			text:        message.Text,
			toolCalls:   callData,
			toolResults: resultData,
			storedBytes: size,
		})
	}
	return encoded, total, nil
}

func decodeMessage(role string, text string, callData []byte, resultData []byte) (llm.Message, error) {
	var calls []storedToolCall
	if err := decodeStrictJSON(callData, &calls); err != nil {
		return llm.Message{}, fmt.Errorf("decode conversation tool calls: %w", err)
	}
	if len(calls) > maxStoredToolCallsPerMessage {
		return llm.Message{}, fmt.Errorf("decode conversation tool calls: count exceeds %d", maxStoredToolCallsPerMessage)
	}
	for index, call := range calls {
		if len(call.Arguments) > maxStoredToolArguments {
			return llm.Message{}, fmt.Errorf("decode conversation tool call %d: arguments exceed %d bytes", index, maxStoredToolArguments)
		}
	}
	var results []storedToolResult
	if err := decodeStrictJSON(resultData, &results); err != nil {
		return llm.Message{}, fmt.Errorf("decode conversation tool results: %w", err)
	}
	if len(results) > maxStoredToolCallsPerMessage {
		return llm.Message{}, fmt.Errorf("decode conversation tool results: count exceeds %d", maxStoredToolCallsPerMessage)
	}
	for index, result := range results {
		if len(result.Content) > maxStoredToolResult {
			return llm.Message{}, fmt.Errorf("decode conversation tool result %d: content exceeds %d bytes", index, maxStoredToolResult)
		}
	}
	message := llm.Message{Role: llm.Role(role), Text: text}
	if len(calls) > 0 {
		message.ToolCalls = make([]llm.ToolCall, len(calls))
		for index, call := range calls {
			message.ToolCalls[index] = llm.ToolCall{
				ID:        call.ID,
				Name:      call.Name,
				Arguments: append(json.RawMessage(nil), call.Arguments...),
			}
		}
	}
	if len(results) > 0 {
		message.ToolResults = make([]llm.ToolResult, len(results))
		for index, result := range results {
			message.ToolResults[index] = llm.ToolResult{
				CallID:  result.CallID,
				Content: append(json.RawMessage(nil), result.Content...),
				IsError: result.IsError,
			}
		}
	}
	return message, nil
}

func decodeStrictJSON(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return fmt.Errorf("trailing JSON data")
	}
	return nil
}

func validateMessageSequence(messages []llm.Message) error {
	expectedRole := llm.RoleUser
	var pendingCalls []llm.ToolCall
	for index, message := range messages {
		if message.Role != expectedRole {
			return fmt.Errorf("conversation message %d has role %q, expected %q", index, message.Role, expectedRole)
		}
		switch message.Role {
		case llm.RoleUser:
			expectedRole = llm.RoleAssistant
		case llm.RoleAssistant:
			if len(message.ToolCalls) == 0 {
				expectedRole = llm.RoleUser
				pendingCalls = nil
			} else {
				expectedRole = llm.RoleTool
				pendingCalls = message.ToolCalls
			}
		case llm.RoleTool:
			if err := matchToolResults(pendingCalls, message.ToolResults); err != nil {
				return fmt.Errorf("conversation message %d: %w", index, err)
			}
			expectedRole = llm.RoleAssistant
			pendingCalls = nil
		}
	}
	if expectedRole != llm.RoleUser {
		return fmt.Errorf("conversation messages end before a final assistant message")
	}
	return nil
}

func matchToolResults(calls []llm.ToolCall, results []llm.ToolResult) error {
	if len(calls) != len(results) {
		return fmt.Errorf("tool result count %d does not match tool call count %d", len(results), len(calls))
	}
	pending := make(map[string]struct{}, len(calls))
	for _, call := range calls {
		if _, exists := pending[call.ID]; exists {
			return fmt.Errorf("tool call ID %q is duplicated", call.ID)
		}
		pending[call.ID] = struct{}{}
	}
	for _, result := range results {
		if _, exists := pending[result.CallID]; !exists {
			return fmt.Errorf("tool result references unexpected call ID %q", result.CallID)
		}
		delete(pending, result.CallID)
	}
	return nil
}
