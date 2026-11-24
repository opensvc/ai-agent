package chatcompletions

import (
	"encoding/json"
	"fmt"

	"github.com/opensvc/ai-agent/internal/llm"
)

type createRequest struct {
	Model               string         `json:"model"`
	Messages            []message      `json:"messages"`
	Tools               []functionTool `json:"tools,omitempty"`
	ToolChoice          string         `json:"tool_choice,omitempty"`
	Stream              bool           `json:"stream"`
	StreamOptions       streamOptions  `json:"stream_options"`
	Store               *bool          `json:"store,omitempty"`
	MaxCompletionTokens int            `json:"max_completion_tokens"`
	// ReasoningEffort is left out unless configured: the model applies its own.
	ReasoningEffort string `json:"reasoning_effort,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type message struct {
	Role       string         `json:"role"`
	Content    *string        `json:"content,omitempty"`
	ToolCalls  []wireToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

type wireToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function wireFunction `json:"function"`
}

type wireFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type functionTool struct {
	Type     string             `json:"type"`
	Function functionDefinition `json:"function"`
}

type functionDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

type toolOutput struct {
	Content json.RawMessage `json:"content"`
	IsError bool            `json:"is_error"`
}

// newCreateRequest maps a request; store is false, forbidding the provider
// to keep it, unless omitStore leaves it out for a provider refusing it.
func newCreateRequest(model string, maxOutputTokens int, omitStore bool, request llm.Request) (createRequest, error) {
	wire := createRequest{
		Model:               model,
		Stream:              true,
		StreamOptions:       streamOptions{IncludeUsage: true},
		MaxCompletionTokens: maxOutputTokens,
	}
	if !omitStore {
		wire.Store = new(bool)
	}
	for _, neutralMessage := range request.Messages {
		switch neutralMessage.Role {
		case llm.RoleSystem, llm.RoleUser:
			// The prefix of the provider's cache is matched by tokens: the
			// volatile text after the text leaves the text in it.
			content := neutralMessage.TextWithVolatile()
			wire.Messages = append(wire.Messages, message{Role: string(neutralMessage.Role), Content: &content})
		case llm.RoleAssistant:
			assistant := message{Role: string(neutralMessage.Role)}
			if neutralMessage.Text != "" {
				content := neutralMessage.Text
				assistant.Content = &content
			}
			for _, call := range neutralMessage.ToolCalls {
				assistant.ToolCalls = append(assistant.ToolCalls, wireToolCall{
					ID:       call.ID,
					Type:     "function",
					Function: wireFunction{Name: call.Name, Arguments: string(call.Arguments)},
				})
			}
			wire.Messages = append(wire.Messages, assistant)
		case llm.RoleTool:
			for _, result := range neutralMessage.ToolResults {
				// A result of text alone goes as it is.
				content, plain := result.PlainText()
				if !plain {
					output, err := json.Marshal(toolOutput{Content: result.Content, IsError: result.IsError})
					if err != nil {
						return createRequest{}, fmt.Errorf("encode tool result %q: %w", result.CallID, err)
					}
					content = string(output)
				}
				wire.Messages = append(wire.Messages, message{Role: string(neutralMessage.Role), Content: &content, ToolCallID: result.CallID})
			}
		}
	}
	for _, tool := range request.Tools {
		wire.Tools = append(wire.Tools, functionTool{
			Type: "function",
			Function: functionDefinition{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.InputSchema,
			},
		})
	}
	if len(wire.Tools) != 0 {
		wire.ToolChoice = "auto"
		if request.NoToolCall {
			wire.ToolChoice = "none"
		}
	}
	return wire, nil
}
