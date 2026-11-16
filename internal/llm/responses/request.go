package responses

import (
	"encoding/json"
	"fmt"

	"github.com/opensvc/ai-agent/internal/llm"
)

type createRequest struct {
	Model           string         `json:"model"`
	Input           []any          `json:"input"`
	Tools           []functionTool `json:"tools,omitempty"`
	ToolChoice      string         `json:"tool_choice,omitempty"`
	Stream          bool           `json:"stream"`
	Store           bool           `json:"store"`
	MaxOutputTokens int            `json:"max_output_tokens"`
	// Reasoning is left out unless configured: the model applies its own.
	Reasoning *reasoning `json:"reasoning,omitempty"`
	// Include asks for the encrypted reasoning of the model, which the next
	// requests of the turn give back, nothing being stored.
	Include []string `json:"include,omitempty"`
}

// reasoning is the reasoning effort a request asks for.
type reasoning struct {
	Effort string `json:"effort"`
}

type inputMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type functionCallInput struct {
	Type      string `json:"type"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type functionOutputInput struct {
	Type   string `json:"type"`
	CallID string `json:"call_id"`
	Output string `json:"output"`
}

type functionTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

type toolOutput struct {
	Content json.RawMessage `json:"content"`
	IsError bool            `json:"is_error"`
}

func newCreateRequest(model string, maxOutputTokens int, request llm.Request) (createRequest, error) {
	return buildCreateRequest(model, maxOutputTokens, request, true)
}

// buildCreateRequest gives back the reasoning of the answers of the turn,
// in the order of their output, when replay is set.
func buildCreateRequest(model string, maxOutputTokens int, request llm.Request, replay bool) (createRequest, error) {
	wire := createRequest{Model: model, Stream: true, Store: false, MaxOutputTokens: maxOutputTokens}
	for _, message := range request.Messages {
		switch message.Role {
		case llm.RoleSystem, llm.RoleUser:
			wire.Input = append(wire.Input, inputMessage{Role: string(message.Role), Content: message.Text})
		case llm.RoleAssistant:
			if order := replayOrder(message); replay && order != nil {
				calls := make(map[string]llm.ToolCall, len(message.ToolCalls))
				for _, call := range message.ToolCalls {
					calls[call.ID] = call
				}
				for _, item := range order {
					switch {
					case item.Reasoning != nil:
						wire.Input = append(wire.Input, item.Reasoning)
					case item.Message:
						wire.Input = append(wire.Input, inputMessage{Role: string(message.Role), Content: message.Text})
					default:
						call := calls[item.CallID]
						wire.Input = append(wire.Input, functionCallInput{Type: "function_call", CallID: call.ID, Name: call.Name, Arguments: string(call.Arguments)})
					}
				}
				continue
			}
			if message.Text != "" {
				wire.Input = append(wire.Input, inputMessage{Role: string(message.Role), Content: message.Text})
			}
			for _, call := range message.ToolCalls {
				wire.Input = append(wire.Input, functionCallInput{Type: "function_call", CallID: call.ID, Name: call.Name, Arguments: string(call.Arguments)})
			}
		case llm.RoleTool:
			for _, result := range message.ToolResults {
				output, err := json.Marshal(toolOutput{Content: result.Content, IsError: result.IsError})
				if err != nil {
					return createRequest{}, fmt.Errorf("encode tool result %q: %w", result.CallID, err)
				}
				wire.Input = append(wire.Input, functionOutputInput{Type: "function_call_output", CallID: result.CallID, Output: string(output)})
			}
		}
	}
	for _, tool := range request.Tools {
		wire.Tools = append(wire.Tools, functionTool{Type: "function", Name: tool.Name, Description: tool.Description, Parameters: tool.InputSchema})
	}
	if len(wire.Tools) != 0 {
		wire.ToolChoice = "auto"
		if request.NoToolCall {
			wire.ToolChoice = "none"
		}
	}
	return wire, nil
}
