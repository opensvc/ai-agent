package responses

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"

	"github.com/opensvc/ai-agent/internal/llm"
)

const (
	maxStreamBytes          = 16 << 20
	maxLineBytes            = 1 << 20
	maxEventBytes           = 2 << 20
	maxPendingToolCallCount = 128
)

type pendingCall struct {
	call    llm.ToolCall
	emitted bool
}

type streamState struct {
	emit         llm.EmitFunc
	secret       string
	calls        map[int]*pendingCall
	hadToolCalls bool
	terminal     bool
}

func consumeStream(reader io.Reader, emit llm.EmitFunc, secret string) error {
	limited := &io.LimitedReader{R: reader, N: maxStreamBytes + 1}
	scanner := bufio.NewScanner(limited)
	scanner.Buffer(make([]byte, 64<<10), maxLineBytes)
	state := &streamState{emit: emit, secret: secret, calls: make(map[int]*pendingCall)}

	var data bytes.Buffer
	dispatch := func() error {
		if data.Len() == 0 {
			return nil
		}
		payload := append([]byte(nil), data.Bytes()...)
		data.Reset()
		if bytes.Equal(bytes.TrimSpace(payload), []byte("[DONE]")) {
			return nil
		}
		return state.consume(payload)
	}

	for scanner.Scan() {
		line := bytes.TrimSuffix(scanner.Bytes(), []byte{'\r'})
		if len(line) == 0 {
			if err := dispatch(); err != nil {
				return err
			}
			continue
		}
		if line[0] == ':' {
			continue
		}
		field, value, found := bytes.Cut(line, []byte{':'})
		if !found || !bytes.Equal(field, []byte("data")) {
			continue
		}
		value = bytes.TrimPrefix(value, []byte{' '})
		if data.Len() != 0 {
			data.WriteByte('\n')
		}
		if data.Len()+len(value) > maxEventBytes {
			return fmt.Errorf("SSE event exceeds %d bytes", maxEventBytes)
		}
		data.Write(value)
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("scan SSE stream: %w", err)
	}
	if limited.N <= 0 {
		return fmt.Errorf("SSE stream exceeds %d bytes", maxStreamBytes)
	}
	if err := dispatch(); err != nil {
		return err
	}
	if !state.terminal {
		return fmt.Errorf("SSE stream ended without a terminal response event")
	}
	return nil
}

func (s *streamState) consume(data []byte) error {
	var envelope struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return fmt.Errorf("decode SSE event: %w", err)
	}
	switch envelope.Type {
	case "response.output_text.delta":
		var event struct {
			Delta string `json:"delta"`
		}
		if err := json.Unmarshal(data, &event); err != nil {
			return fmt.Errorf("decode text delta: %w", err)
		}
		if event.Delta != "" {
			return s.emitEvent(llm.Event{Type: llm.EventTextDelta, TextDelta: event.Delta})
		}
	case "response.output_item.added":
		var event struct {
			OutputIndex int      `json:"output_index"`
			Item        wireItem `json:"item"`
		}
		if err := json.Unmarshal(data, &event); err != nil {
			return fmt.Errorf("decode output item: %w", err)
		}
		if event.Item.Type == "function_call" {
			call, err := s.getOrCreateCall(event.OutputIndex)
			if err != nil {
				return err
			}
			call.call = llm.ToolCall{ID: event.Item.CallID, Name: event.Item.Name, Arguments: json.RawMessage(event.Item.Arguments)}
		}
	case "response.function_call_arguments.delta":
		var event struct {
			OutputIndex int    `json:"output_index"`
			Delta       string `json:"delta"`
		}
		if err := json.Unmarshal(data, &event); err != nil {
			return fmt.Errorf("decode function arguments delta: %w", err)
		}
		call, ok := s.calls[event.OutputIndex]
		if !ok {
			return fmt.Errorf("function arguments delta references unknown output index %d", event.OutputIndex)
		}
		call.call.Arguments = append(call.call.Arguments, event.Delta...)
		if len(call.call.Arguments) > maxEventBytes {
			return fmt.Errorf("function arguments exceed %d bytes", maxEventBytes)
		}
	case "response.function_call_arguments.done":
		var event struct {
			OutputIndex int    `json:"output_index"`
			Arguments   string `json:"arguments"`
		}
		if err := json.Unmarshal(data, &event); err != nil {
			return fmt.Errorf("decode completed function arguments: %w", err)
		}
		call, ok := s.calls[event.OutputIndex]
		if !ok {
			return fmt.Errorf("completed function arguments reference unknown output index %d", event.OutputIndex)
		}
		if event.Arguments != "" {
			call.call.Arguments = json.RawMessage(event.Arguments)
		}
		return s.emitCall(call)
	case "response.output_item.done":
		var event struct {
			OutputIndex int      `json:"output_index"`
			Item        wireItem `json:"item"`
		}
		if err := json.Unmarshal(data, &event); err != nil {
			return fmt.Errorf("decode completed output item: %w", err)
		}
		if event.Item.Type == "function_call" {
			call, err := s.getOrCreateCall(event.OutputIndex)
			if err != nil {
				return err
			}
			if call.emitted {
				return nil
			}
			call.call = llm.ToolCall{ID: event.Item.CallID, Name: event.Item.Name, Arguments: json.RawMessage(event.Item.Arguments)}
			return s.emitCall(call)
		}
	case "response.completed":
		var event responseEvent
		if err := json.Unmarshal(data, &event); err != nil {
			return fmt.Errorf("decode completed response: %w", err)
		}
		if err := s.emitCallsFromResponse(event.Response.Output); err != nil {
			return err
		}
		if err := s.emitUsage(event.Response.Usage); err != nil {
			return err
		}
		reason := llm.FinishReasonCompleted
		if s.hadToolCalls {
			reason = llm.FinishReasonToolCalls
		}
		if err := s.emitEvent(llm.Event{Type: llm.EventCompleted, FinishReason: reason}); err != nil {
			return err
		}
		s.terminal = true
	case "response.incomplete":
		var event responseEvent
		if err := json.Unmarshal(data, &event); err != nil {
			return fmt.Errorf("decode incomplete response: %w", err)
		}
		if err := s.emitUsage(event.Response.Usage); err != nil {
			return err
		}
		reason := llm.FinishReasonOther
		if event.Response.IncompleteDetails.Reason == "max_output_tokens" {
			reason = llm.FinishReasonLength
		}
		if err := s.emitEvent(llm.Event{Type: llm.EventCompleted, FinishReason: reason}); err != nil {
			return err
		}
		s.terminal = true
	case "response.failed":
		var event responseEvent
		if err := json.Unmarshal(data, &event); err != nil {
			return fmt.Errorf("decode failed response: %w", err)
		}
		return s.providerError(event.Response.Error.Code, event.Response.Error.Message)
	case "error":
		var event struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		}
		if err := json.Unmarshal(data, &event); err != nil {
			return fmt.Errorf("decode provider error: %w", err)
		}
		return s.providerError(event.Code, event.Message)
	}
	return nil
}

type wireItem struct {
	Type      string `json:"type"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type wireUsage struct {
	InputTokens        int64 `json:"input_tokens"`
	OutputTokens       int64 `json:"output_tokens"`
	TotalTokens        int64 `json:"total_tokens"`
	InputTokensDetails struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"input_tokens_details"`
}

type responseEvent struct {
	Response struct {
		Output            []wireItem `json:"output"`
		Usage             *wireUsage `json:"usage"`
		IncompleteDetails struct {
			Reason string `json:"reason"`
		} `json:"incomplete_details"`
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	} `json:"response"`
}

func (s *streamState) emitCallsFromResponse(items []wireItem) error {
	for index, item := range items {
		if item.Type != "function_call" {
			continue
		}
		call, ok := s.calls[index]
		if !ok {
			var err error
			call, err = s.getOrCreateCall(index)
			if err != nil {
				return err
			}
		}
		if !call.emitted {
			call.call = llm.ToolCall{ID: item.CallID, Name: item.Name, Arguments: json.RawMessage(item.Arguments)}
			if err := s.emitCall(call); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *streamState) getOrCreateCall(index int) (*pendingCall, error) {
	if index < 0 {
		return nil, fmt.Errorf("function call has invalid output index %d", index)
	}
	if call, ok := s.calls[index]; ok {
		return call, nil
	}
	if len(s.calls) >= maxPendingToolCallCount {
		return nil, fmt.Errorf("pending tool call count exceeds %d", maxPendingToolCallCount)
	}
	call := &pendingCall{}
	s.calls[index] = call
	return call, nil
}

func (s *streamState) emitCall(call *pendingCall) error {
	if call.emitted {
		return nil
	}
	if err := s.emitEvent(llm.Event{Type: llm.EventToolCall, ToolCall: &call.call}); err != nil {
		return err
	}
	call.emitted = true
	s.hadToolCalls = true
	return nil
}

func (s *streamState) emitUsage(usage *wireUsage) error {
	if usage == nil {
		return nil
	}
	// The tokens read from the cache are counted in the input; a compatible
	// endpoint counting more is not believed beyond it.
	cached := min(max(usage.InputTokensDetails.CachedTokens, 0), max(usage.InputTokens, 0))
	return s.emitEvent(llm.Event{Type: llm.EventUsage, Usage: &llm.Usage{InputTokens: usage.InputTokens, OutputTokens: usage.OutputTokens, TotalTokens: usage.TotalTokens,
		CacheReadTokens: cached}})
}

func (s *streamState) emitEvent(event llm.Event) error {
	if err := event.Validate(); err != nil {
		return fmt.Errorf("invalid neutral LLM event: %w", err)
	}
	if err := s.emit(event); err != nil {
		return fmt.Errorf("consume neutral LLM event: %w", err)
	}
	return nil
}

func (s *streamState) providerError(code string, message string) error {
	code = normalizeErrorText(code, s.secret)
	message = normalizeErrorText(message, s.secret)
	if code != "" && message != "" {
		return fmt.Errorf("Responses provider error (%s): %s", code, message)
	}
	if message != "" {
		return fmt.Errorf("Responses provider error: %s", message)
	}
	return fmt.Errorf("Responses provider error")
}
