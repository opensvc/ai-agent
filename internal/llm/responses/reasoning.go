package responses

import (
	"encoding/json"
	"slices"

	"github.com/opensvc/ai-agent/internal/llm"
)

// Protocol names the reasoning this adapter produces and gives back.
const Protocol = "responses"

// includeReasoning asks the provider for the reasoning of the model,
// encrypted, which a request that stores nothing gives back.
const includeReasoning = "reasoning.encrypted_content"

// replayItem is an output item of an answer, in its order: a reasoning item
// as the provider gave it, or the place of the call or of the text of the
// message, which the request renders from the message.
type replayItem struct {
	Reasoning json.RawMessage `json:"reasoning,omitempty"`
	CallID    string          `json:"call_id,omitempty"`
	Message   bool            `json:"message,omitempty"`
}

type replay struct {
	Items []replayItem `json:"items"`
}

// answerReasoning is the reasoning of an answer whose output held encrypted
// reasoning items, in the order of its output, nil without one.
func answerReasoning(output []json.RawMessage) *llm.Reasoning {
	var answer replay
	reasoning, message := false, false
	for _, raw := range output {
		var item struct {
			Type             string  `json:"type"`
			CallID           string  `json:"call_id"`
			EncryptedContent *string `json:"encrypted_content"`
		}
		if json.Unmarshal(raw, &item) != nil {
			return nil
		}
		switch item.Type {
		case "reasoning":
			if item.EncryptedContent == nil || *item.EncryptedContent == "" {
				continue
			}
			reasoning = true
			answer.Items = append(answer.Items, replayItem{Reasoning: raw})
		case "function_call":
			answer.Items = append(answer.Items, replayItem{CallID: item.CallID})
		case "message":
			// The text of the messages of an answer is one text: it is
			// rendered once, in the place of the first.
			if !message {
				answer.Items = append(answer.Items, replayItem{Message: true})
			}
			message = true
		}
	}
	if !reasoning {
		return nil
	}
	data, err := json.Marshal(answer)
	if err != nil || len(data) > llm.MaxReasoningBytes {
		return nil
	}
	return &llm.Reasoning{Protocol: Protocol, Data: data}
}

// replayOrder returns the output items of an answer in their order, when its
// reasoning describes the text and the calls of the message, nil otherwise.
func replayOrder(message llm.Message) []replayItem {
	if message.Reasoning == nil || message.Reasoning.Protocol != Protocol {
		return nil
	}
	var answer replay
	if json.Unmarshal(message.Reasoning.Data, &answer) != nil {
		return nil
	}
	var calls []string
	text := false
	for _, item := range answer.Items {
		switch {
		case item.Reasoning != nil:
		case item.CallID != "":
			calls = append(calls, item.CallID)
		case item.Message:
			text = true
		default:
			return nil
		}
	}
	want := make([]string, 0, len(message.ToolCalls))
	for _, call := range message.ToolCalls {
		want = append(want, call.ID)
	}
	if !slices.Equal(calls, want) || text != (message.Text != "") {
		return nil
	}
	return answer.Items
}
