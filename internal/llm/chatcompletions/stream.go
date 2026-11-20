package chatcompletions

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"sort"

	"github.com/opensvc/ai-agent/internal/llm"
)

const (
	maxStreamBytes          = 16 << 20
	maxLineBytes            = 1 << 20
	maxEventBytes           = 2 << 20
	maxToolArgumentsBytes   = 2 << 20
	maxPendingToolCallCount = 128
)

type pendingCall struct {
	id        string
	typeName  string
	name      string
	arguments []byte
}

type streamState struct {
	emit         llm.EmitFunc
	secret       string
	calls        map[int]*pendingCall
	finishReason llm.FinishReason
	finished     bool
	done         bool
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
			return state.complete()
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
	if !state.done {
		return fmt.Errorf("SSE stream ended without [DONE]")
	}
	return nil
}

// deltaContent is the text of a delta: a string, or a list of typed parts,
// as Mistral streams it when it reasons, its reasoning a part of type
// thinking among parts of type text. The text parts are the answer; the
// others, the reasoning of the model among them, are left out: never
// emitted, never stored.
type deltaContent struct {
	text string
}

func (c *deltaContent) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		c.text = text
		return nil
	}
	var parts []struct {
		Type string          `json:"type"`
		Text json.RawMessage `json:"text"`
	}
	if err := json.Unmarshal(data, &parts); err != nil {
		return fmt.Errorf("decode delta content: %w", err)
	}
	var joined string
	for _, part := range parts {
		// Only a part of type text, exactly, is the answer: any other, the
		// reasoning among them, is left out unread, even one holding a text
		// field, and none can fail the turn.
		if part.Type != "text" || len(part.Text) == 0 {
			continue
		}
		var piece string
		if err := json.Unmarshal(part.Text, &piece); err != nil {
			return fmt.Errorf("decode delta content text part: %w", err)
		}
		joined += piece
	}
	c.text = joined
	return nil
}

type streamChunk struct {
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Content   *deltaContent   `json:"content"`
			ToolCalls []wireToolDelta `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *wireUsage `json:"usage"`
	Error *wireError `json:"error"`
}

type wireToolDelta struct {
	Index    *int   `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type wireUsage struct {
	PromptTokens        int64 `json:"prompt_tokens"`
	CompletionTokens    int64 `json:"completion_tokens"`
	TotalTokens         int64 `json:"total_tokens"`
	PromptTokensDetails *struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

type wireError struct {
	Code    any    `json:"code"`
	Message string `json:"message"`
	Type    string `json:"type"`
}

func (s *streamState) consume(data []byte) error {
	if s.done {
		return fmt.Errorf("received an SSE event after [DONE]")
	}
	var chunk streamChunk
	if err := json.Unmarshal(data, &chunk); err != nil {
		return fmt.Errorf("decode SSE event: %w", err)
	}
	if chunk.Error != nil {
		return s.providerError(fmt.Sprint(chunk.Error.Code), chunk.Error.Message)
	}
	if len(chunk.Choices) > 1 {
		return fmt.Errorf("SSE event contains %d choices, expected at most one", len(chunk.Choices))
	}
	if len(chunk.Choices) == 1 {
		choice := chunk.Choices[0]
		if choice.Index != 0 {
			return fmt.Errorf("SSE event contains unexpected choice index %d", choice.Index)
		}
		if s.finished && (choice.Delta.Content != nil || len(choice.Delta.ToolCalls) != 0 || choice.FinishReason != nil) {
			return fmt.Errorf("SSE event contains choice data after finish reason")
		}
		if choice.Delta.Content != nil && choice.Delta.Content.text != "" {
			if err := s.emitEvent(llm.Event{Type: llm.EventTextDelta, TextDelta: choice.Delta.Content.text}); err != nil {
				return err
			}
		}
		for _, delta := range choice.Delta.ToolCalls {
			if err := s.consumeToolDelta(delta); err != nil {
				return err
			}
		}
		if choice.FinishReason != nil {
			if s.finished {
				return fmt.Errorf("SSE stream contains multiple finish reasons")
			}
			reason := mapFinishReason(*choice.FinishReason)
			if reason == llm.FinishReasonToolCalls {
				if len(s.calls) == 0 {
					return fmt.Errorf("tool_calls finish reason has no tool calls")
				}
				if err := s.emitToolCalls(); err != nil {
					return err
				}
			} else if len(s.calls) != 0 {
				return fmt.Errorf("finish reason %q has pending tool calls", *choice.FinishReason)
			}
			s.finishReason = reason
			s.finished = true
		}
	}
	if chunk.Usage != nil {
		// The tokens read from the cache are counted in the prompt; a
		// compatible endpoint counting more is not believed beyond it.
		var cached int64
		if details := chunk.Usage.PromptTokensDetails; details != nil {
			cached = min(max(details.CachedTokens, 0), max(chunk.Usage.PromptTokens, 0))
		}
		if err := s.emitEvent(llm.Event{Type: llm.EventUsage, Usage: &llm.Usage{
			InputTokens:     chunk.Usage.PromptTokens,
			OutputTokens:    chunk.Usage.CompletionTokens,
			TotalTokens:     chunk.Usage.TotalTokens,
			CacheReadTokens: cached,
		}}); err != nil {
			return err
		}
	}
	return nil
}

func (s *streamState) consumeToolDelta(delta wireToolDelta) error {
	if s.finished {
		return fmt.Errorf("tool call delta received after finish reason")
	}
	if delta.Index == nil || *delta.Index < 0 {
		return fmt.Errorf("tool call delta has an invalid index")
	}
	call, ok := s.calls[*delta.Index]
	if !ok {
		if len(s.calls) >= maxPendingToolCallCount {
			return fmt.Errorf("tool call count exceeds %d", maxPendingToolCallCount)
		}
		call = &pendingCall{}
		s.calls[*delta.Index] = call
	}
	if err := setStableField(&call.id, delta.ID, "ID", *delta.Index); err != nil {
		return err
	}
	if err := setStableField(&call.typeName, delta.Type, "type", *delta.Index); err != nil {
		return err
	}
	if err := setStableField(&call.name, delta.Function.Name, "name", *delta.Index); err != nil {
		return err
	}
	if len(call.arguments)+len(delta.Function.Arguments) > maxToolArgumentsBytes {
		return fmt.Errorf("tool call %d arguments exceed %d bytes", *delta.Index, maxToolArgumentsBytes)
	}
	call.arguments = append(call.arguments, delta.Function.Arguments...)
	return nil
}

func setStableField(target *string, value string, field string, index int) error {
	if value == "" {
		return nil
	}
	if *target != "" && *target != value {
		return fmt.Errorf("tool call %d changed %s from %q to %q", index, field, *target, value)
	}
	*target = value
	return nil
}

func (s *streamState) emitToolCalls() error {
	indices := make([]int, 0, len(s.calls))
	for index := range s.calls {
		indices = append(indices, index)
	}
	sort.Ints(indices)
	for _, index := range indices {
		pending := s.calls[index]
		if pending.typeName != "function" {
			return fmt.Errorf("tool call %d has unsupported type %q", index, pending.typeName)
		}
		call := llm.ToolCall{ID: pending.id, Name: pending.name, Arguments: json.RawMessage(pending.arguments)}
		if err := s.emitEvent(llm.Event{Type: llm.EventToolCall, ToolCall: &call}); err != nil {
			return err
		}
	}
	return nil
}

func (s *streamState) complete() error {
	if s.done {
		return fmt.Errorf("SSE stream contains multiple [DONE] markers")
	}
	if !s.finished {
		return fmt.Errorf("[DONE] received before a finish reason")
	}
	if err := s.emitEvent(llm.Event{Type: llm.EventCompleted, FinishReason: s.finishReason}); err != nil {
		return err
	}
	s.done = true
	return nil
}

func mapFinishReason(reason string) llm.FinishReason {
	switch reason {
	case "stop":
		return llm.FinishReasonCompleted
	case "tool_calls":
		return llm.FinishReasonToolCalls
	case "length":
		return llm.FinishReasonLength
	case "content_filter":
		return llm.FinishReasonContentFilter
	default:
		return llm.FinishReasonOther
	}
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
	if code != "" && code != "<nil>" && message != "" {
		return fmt.Errorf("Chat Completions provider error (%s): %s", code, message)
	}
	if message != "" {
		return fmt.Errorf("Chat Completions provider error: %s", message)
	}
	return fmt.Errorf("Chat Completions provider error")
}
