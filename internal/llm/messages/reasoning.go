package messages

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"
	"strings"

	"github.com/opensvc/ai-agent/internal/llm"
)

// Protocol names the reasoning this adapter produces and gives back.
const Protocol = "messages"

// replay is the reasoning of one answer: its content blocks in the order the
// model gave them, its thinking with the text and the calls it led to, and
// the fingerprint of the system prompt and the tools of its request.
//
// A thinking block is bound to what preceded it: the system prompt, the set
// of the tools and the messages before it. A request whose system prompt or
// tools changed, such as after search_tools showed more tools, cannot give
// it back: the provider refuses the request, or drops the block. Thinking
// blocks may only be left out from the start of the conversation, never
// from its middle: the request gives back the thinking of the answers after
// the last one it cannot, and of none before it.
type replay struct {
	Fingerprint string `json:"fingerprint"`
	// Incomplete tells the thinking of the answer could not be kept, such as
	// past its bound: no earlier thinking can be given back after it.
	Incomplete bool              `json:"incomplete,omitempty"`
	Content    []json.RawMessage `json:"content,omitempty"`
}

// requestFingerprint hashes what a thinking block is bound to beside the
// messages: the system prompt and the tools, as a set sorted by name.
func requestFingerprint(system []string, tools []wireTool) string {
	sorted := slices.Clone(tools)
	slices.SortFunc(sorted, func(a, b wireTool) int { return strings.Compare(a.Name, b.Name) })
	data, _ := json.Marshal(struct {
		System []string   `json:"system"`
		Tools  []wireTool `json:"tools"`
	}{system, sorted})
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// replayable returns, by the index of their message, the answers of the
// request whose content, thinking included, is given back as the model gave
// it, those after the last answer whose thinking cannot be.
func replayable(messages []llm.Message, fingerprint string) map[int][]json.RawMessage {
	content := make(map[int][]json.RawMessage)
	for index, message := range messages {
		if message.Role != llm.RoleAssistant || message.Reasoning == nil || message.Reasoning.Protocol != Protocol {
			continue
		}
		var answer replay
		if json.Unmarshal(message.Reasoning.Data, &answer) != nil || answer.Incomplete || answer.Fingerprint != fingerprint || !answer.matches(message) {
			clear(content)
			continue
		}
		content[index] = answer.Content
	}
	return content
}

// matches tells whether the content of an answer is that of the message:
// its text and its calls, by their ID and name, in their order, beside the
// thinking. The arguments are not compared: their encoding may differ, such
// as escaping & < and >, for the same call.
func (r replay) matches(message llm.Message) bool {
	var text strings.Builder
	var calls []llm.ToolCall
	thinking := false
	for _, raw := range r.Content {
		var block struct {
			Type  string          `json:"type"`
			Text  string          `json:"text"`
			ID    string          `json:"id"`
			Name  string          `json:"name"`
			Input json.RawMessage `json:"input"`
		}
		if json.Unmarshal(raw, &block) != nil {
			return false
		}
		switch block.Type {
		case "thinking", "redacted_thinking":
			thinking = true
		case "text":
			text.WriteString(block.Text)
		case "tool_use":
			calls = append(calls, llm.ToolCall{ID: block.ID, Name: block.Name, Arguments: block.Input})
		default:
			return false
		}
	}
	if !thinking || text.String() != message.Text || len(calls) != len(message.ToolCalls) {
		return false
	}
	for index, call := range calls {
		want := message.ToolCalls[index]
		if call.ID != want.ID || call.Name != want.Name {
			return false
		}
	}
	return true
}

// answerReasoning is the reasoning of an answer whose content held thinking,
// nil without thinking.
func answerReasoning(blocks []pendingBlock, fingerprint string) *llm.Reasoning {
	thinking := false
	content := make([]json.RawMessage, 0, len(blocks))
	for _, pending := range blocks {
		var block any
		switch pending.block.Type {
		case "thinking":
			thinking = true
			block = struct {
				Type      string `json:"type"`
				Thinking  string `json:"thinking"`
				Signature string `json:"signature"`
			}{"thinking", string(pending.thinking), string(pending.signature)}
		case "redacted_thinking":
			thinking = true
			block = struct {
				Type string `json:"type"`
				Data string `json:"data"`
			}{"redacted_thinking", pending.block.Data}
		case "text":
			// The provider refuses an empty text block, which the request
			// without thinking leaves out too.
			if len(pending.text) == 0 {
				continue
			}
			block = struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}{"text", string(pending.text)}
		case "tool_use":
			input := pending.block.Input
			if len(pending.arguments) > 0 {
				input = pending.arguments
			}
			block = struct {
				Type  string          `json:"type"`
				ID    string          `json:"id"`
				Name  string          `json:"name"`
				Input json.RawMessage `json:"input"`
			}{"tool_use", pending.block.ID, pending.block.Name, input}
		}
		data, err := json.Marshal(block)
		if err != nil {
			return incompleteReasoning(fingerprint)
		}
		content = append(content, data)
	}
	if !thinking {
		return nil
	}
	data, err := json.Marshal(replay{Fingerprint: fingerprint, Content: content})
	if err != nil || len(data) > llm.MaxReasoningBytes {
		return incompleteReasoning(fingerprint)
	}
	return &llm.Reasoning{Protocol: Protocol, Data: data}
}

func incompleteReasoning(fingerprint string) *llm.Reasoning {
	data, _ := json.Marshal(replay{Fingerprint: fingerprint, Incomplete: true})
	return &llm.Reasoning{Protocol: Protocol, Data: data}
}
