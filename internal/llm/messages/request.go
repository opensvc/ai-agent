package messages

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/opensvc/ai-agent/internal/llm"
)

type createRequest struct {
	Model     string        `json:"model"`
	MaxTokens int           `json:"max_tokens"`
	Stream    bool          `json:"stream"`
	System    []wireBlock   `json:"system,omitempty"`
	Messages  []wireMessage `json:"messages"`
	Tools     []wireTool    `json:"tools,omitempty"`
	// ToolChoice is set to none only: auto is the default.
	ToolChoice *wireToolChoice `json:"tool_choice,omitempty"`
	// Fingerprint hashes the system prompt and the tools, which the
	// thinking of the answer is bound to.
	Fingerprint string `json:"-"`
}

type wireToolChoice struct {
	Type string `json:"type"`
}

type wireMessage struct {
	Role    string      `json:"role"`
	Content []wireBlock `json:"content"`
}

type wireBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   string          `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
	// Thinking, Signature and Data are read from the thinking blocks of an
	// answer, which a request gives back as raw.
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
	Data      string `json:"data,omitempty"`
	// CacheControl marks the end of a prefix the provider caches.
	CacheControl *cacheControl `json:"cache_control,omitempty"`
	// raw is a block of an answer given back as the model gave it.
	raw json.RawMessage
}

func (b wireBlock) MarshalJSON() ([]byte, error) {
	if b.raw != nil {
		return b.raw, nil
	}
	type plain wireBlock
	return json.Marshal(plain(b))
}

// cacheControl asks the provider to cache the prompt up to its block, for
// five minutes: the tools and the system prompt, then the conversation.
type cacheControl struct {
	Type string `json:"type"`
}

type wireTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

func newCreateRequest(model string, maxTokens int, request llm.Request) (createRequest, error) {
	return buildCreateRequest(model, maxTokens, request, true)
}

// buildCreateRequest gives back the thinking of the answers of the turn when
// replay is set, as far as their content binds them.
func buildCreateRequest(model string, maxTokens int, request llm.Request, replay bool) (createRequest, error) {
	result := createRequest{Model: model, MaxTokens: maxTokens, Stream: true}
	var system []string
	for _, message := range request.Messages {
		if message.Role == llm.RoleSystem {
			system = append(system, message.Text)
		}
	}
	for _, tool := range request.Tools {
		result.Tools = append(result.Tools, wireTool{Name: tool.Name, Description: tool.Description, InputSchema: tool.InputSchema})
	}
	result.Fingerprint = requestFingerprint(system, result.Tools)
	answers := map[int][]json.RawMessage{}
	if replay {
		answers = replayable(request.Messages, result.Fingerprint)
	}
	pending := make(map[string]struct{})
	seen := make(map[string]struct{})
	appendMessage := func(role string, blocks []wireBlock) {
		if n := len(result.Messages); n > 0 && result.Messages[n-1].Role == role {
			result.Messages[n-1].Content = append(result.Messages[n-1].Content, blocks...)
		} else {
			result.Messages = append(result.Messages, wireMessage{Role: role, Content: blocks})
		}
	}
	for index, message := range request.Messages {
		if len(pending) != 0 && message.Role != llm.RoleTool {
			return createRequest{}, fmt.Errorf("Messages history has unresolved tool calls before message %d", index)
		}
		switch message.Role {
		case llm.RoleSystem:
			if len(result.Messages) != 0 {
				return createRequest{}, fmt.Errorf("Messages system instructions must precede conversation messages")
			}
		case llm.RoleUser:
			appendMessage("user", []wireBlock{{Type: "text", Text: message.Text}})
		case llm.RoleAssistant:
			var blocks []wireBlock
			for _, call := range message.ToolCalls {
				if _, exists := seen[call.ID]; exists {
					return createRequest{}, fmt.Errorf("Messages history has duplicate tool call IDs")
				}
				seen[call.ID] = struct{}{}
				pending[call.ID] = struct{}{}
			}
			if content, ok := answers[index]; ok {
				// The answer as the model gave it, its thinking included.
				for _, raw := range content {
					blocks = append(blocks, wireBlock{raw: raw})
				}
				appendMessage("assistant", blocks)
				continue
			}
			if message.Text != "" {
				blocks = append(blocks, wireBlock{Type: "text", Text: message.Text})
			}
			for _, call := range message.ToolCalls {
				blocks = append(blocks, wireBlock{Type: "tool_use", ID: call.ID, Name: call.Name, Input: call.Arguments})
			}
			appendMessage("assistant", blocks)
		case llm.RoleTool:
			var blocks []wireBlock
			for _, toolResult := range message.ToolResults {
				if _, exists := pending[toolResult.CallID]; !exists {
					return createRequest{}, fmt.Errorf("Messages history has an unmatched or duplicate tool result")
				}
				delete(pending, toolResult.CallID)
				blocks = append(blocks, wireBlock{Type: "tool_result", ToolUseID: toolResult.CallID, Content: string(toolResult.Content), IsError: toolResult.IsError})
			}
			appendMessage("user", blocks)
		default:
			return createRequest{}, fmt.Errorf("Messages history contains an unsupported role")
		}
	}
	if len(pending) != 0 {
		return createRequest{}, fmt.Errorf("Messages history has unresolved tool calls")
	}
	if len(result.Messages) == 0 || result.Messages[0].Role != "user" || result.Messages[len(result.Messages)-1].Role != "user" {
		return createRequest{}, fmt.Errorf("Messages history must begin and end with a user message or tool results")
	}
	// Two cache points: the tools and the system prompt, the same for every
	// request of the backend, and the conversation up to its last message,
	// which the next request of the turn extends.
	if len(system) != 0 {
		result.System = []wireBlock{{Type: "text", Text: strings.Join(system, "\n\n"), CacheControl: &cacheControl{Type: "ephemeral"}}}
	}
	// The last request of a turn, without tool calls, invalidates the cache
	// of the conversation: its tail would be written and never read.
	if last := &result.Messages[len(result.Messages)-1]; len(last.Content) != 0 && !request.NoToolCall {
		last.Content[len(last.Content)-1].CacheControl = &cacheControl{Type: "ephemeral"}
	}
	if request.NoToolCall && len(result.Tools) != 0 {
		result.ToolChoice = &wireToolChoice{Type: "none"}
	}
	return result, nil
}
