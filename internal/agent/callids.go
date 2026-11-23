package agent

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"

	"github.com/opensvc/ai-agent/internal/llm"
)

// A conversation may run its turns on models of several providers, whose
// history then holds the IDs of the tool calls of each. The providers of
// the agent accept the IDs of one another, but not two calls of one ID,
// which a server that numbers its calls from zero at each answer gives, nor
// an ID out of letters, digits, '-' and '_', the form the Messages protocol
// takes. Such an ID is renamed in the requests only, after the calls before
// it: the history keeps it, the stream and the audit show it, and an ID
// already sendable is sent as it is, so that the prefix the providers cache
// stays the same from a request to the next.

// maxSendableCallIDBytes bounds a sendable tool call ID.
const maxSendableCallIDBytes = 64

// sendableCallID tells whether a provider takes the tool call ID id.
func sendableCallID(id string) bool {
	if id == "" || len(id) > maxSendableCallIDBytes {
		return false
	}
	for _, char := range id {
		if !(char >= 'a' && char <= 'z' || char >= 'A' && char <= 'Z' || char >= '0' && char <= '9' || char == '-' || char == '_') {
			return false
		}
	}
	return true
}

// withSendableCallIDs gives the messages of a request with each tool call ID
// a provider would refuse renamed, and the results of its call with it: an
// ID out of the sendable form, or an ID of a call before it. The new ID is
// derived from the old and its rank, the same at each request. The messages
// are returned as they are when no ID is renamed.
func withSendableCallIDs(messages []llm.Message) []llm.Message {
	used := map[string]bool{}
	// current is the ID each ID stands for, in the results after its call.
	current := map[string]string{}
	var renamed []llm.Message
	// own gives the message of index in renamed, the messages copied at
	// the first renaming, its calls and results at their first.
	own := func(index int) *llm.Message {
		if renamed == nil {
			renamed = append([]llm.Message(nil), messages...)
		}
		message := &renamed[index]
		if sameCalls(message.ToolCalls, messages[index].ToolCalls) {
			message.ToolCalls = append([]llm.ToolCall(nil), messages[index].ToolCalls...)
		}
		if sameResults(message.ToolResults, messages[index].ToolResults) {
			message.ToolResults = append([]llm.ToolResult(nil), messages[index].ToolResults...)
		}
		return message
	}
	for index, message := range messages {
		for callIndex, call := range message.ToolCalls {
			id := call.ID
			if !sendableCallID(id) || used[id] {
				id = derivedCallID(call.ID, index, callIndex, used)
				own(index).ToolCalls[callIndex].ID = id
			}
			used[id] = true
			current[call.ID] = id
		}
		for resultIndex, result := range message.ToolResults {
			if id, ok := current[result.CallID]; ok && id != result.CallID {
				own(index).ToolResults[resultIndex].CallID = id
			}
		}
	}
	if renamed == nil {
		return messages
	}
	return renamed
}

// derivedCallID is the sendable ID of the call of id at its rank, unused.
func derivedCallID(id string, message, call int, used map[string]bool) string {
	for attempt := 0; ; attempt++ {
		sum := sha256.Sum256([]byte(id + "\x00" + strconv.Itoa(message) + "\x00" + strconv.Itoa(call) + "\x00" + strconv.Itoa(attempt)))
		derived := "call_" + hex.EncodeToString(sum[:12])
		if !used[derived] {
			return derived
		}
	}
}

// sameCalls tells whether two slices share their backing array: a renamed
// message still holding the calls of the original.
func sameCalls(a, b []llm.ToolCall) bool {
	return len(a) > 0 && len(b) > 0 && &a[0] == &b[0]
}

// sameResults tells whether two slices share their backing array.
func sameResults(a, b []llm.ToolResult) bool {
	return len(a) > 0 && len(b) > 0 && &a[0] == &b[0]
}
