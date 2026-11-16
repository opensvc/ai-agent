package messages

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"

	"github.com/opensvc/ai-agent/internal/llm"
)

const (
	maxStreamBytes        = 16 << 20
	maxLineBytes          = 1 << 20
	maxEventBytes         = 2 << 20
	maxToolArgumentsBytes = 2 << 20
	maxContentBlocks      = 256
	maxToolCalls          = 128
)

type streamEvent struct {
	Type    string `json:"type"`
	Index   *int   `json:"index"`
	Message *struct {
		Type       string            `json:"type"`
		Role       string            `json:"role"`
		Content    []json.RawMessage `json:"content"`
		StopReason *string           `json:"stop_reason"`
		Usage      *wireUsage        `json:"usage"`
	} `json:"message"`
	ContentBlock *wireBlock `json:"content_block"`
	Delta        *struct {
		Type        string  `json:"type"`
		Text        string  `json:"text"`
		Thinking    string  `json:"thinking"`
		Signature   string  `json:"signature"`
		PartialJSON string  `json:"partial_json"`
		StopReason  *string `json:"stop_reason"`
	} `json:"delta"`
	Usage *wireUsage `json:"usage"`
	Error *struct {
		Type string `json:"type"`
	} `json:"error"`
}

type wireUsage struct {
	InputTokens              *int64 `json:"input_tokens"`
	OutputTokens             *int64 `json:"output_tokens"`
	CacheCreationInputTokens *int64 `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     *int64 `json:"cache_read_input_tokens"`
}

type pendingBlock struct {
	block     wireBlock
	arguments []byte
	closed    bool
	// text, thinking and signature accumulate the deltas of the block, for
	// the reasoning of the answer to give it back as the model gave it.
	text, thinking, signature []byte
}

type streamState struct {
	emit llm.EmitFunc
	// fingerprint is that of the request, which its thinking is bound to.
	fingerprint                             string
	started                                 bool
	finishing                               bool
	stopped                                 bool
	blocks                                  []pendingBlock
	calls                                   int
	callIDs                                 map[string]struct{}
	stopReason                              string
	input, output, cacheCreation, cacheRead int64
}

func consumeStream(reader io.Reader, fingerprint string, emit llm.EmitFunc) error {
	limited := &io.LimitedReader{R: reader, N: maxStreamBytes + 1}
	scanner := bufio.NewScanner(limited)
	scanner.Buffer(make([]byte, 64<<10), maxLineBytes)
	state := streamState{emit: emit, fingerprint: fingerprint, callIDs: make(map[string]struct{})}
	var data bytes.Buffer
	var eventName string
	dispatch := func() error {
		defer func() { data.Reset(); eventName = "" }()
		if data.Len() == 0 {
			return nil
		}
		return state.consume(data.Bytes(), eventName)
	}
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			if err := dispatch(); err != nil {
				return err
			}
			continue
		}
		if line[0] == ':' {
			continue
		}
		field, value, _ := bytes.Cut(line, []byte{':'})
		value = bytes.TrimPrefix(value, []byte{' '})
		switch string(field) {
		case "event":
			eventName = string(value)
		case "data":
			if data.Len()+len(value)+1 > maxEventBytes {
				return fmt.Errorf("SSE event exceeds %d bytes", maxEventBytes)
			}
			if data.Len() != 0 {
				data.WriteByte('\n')
			}
			data.Write(value)
		}
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
	if !state.stopped {
		return fmt.Errorf("SSE stream ended without message_stop")
	}
	// Defer executable tool calls and completion until the entire bounded stream
	// has been validated; malformed/truncated responses must not run tools.
	return state.complete()
}

func (s *streamState) consume(data []byte, eventName string) error {
	if s.stopped {
		return fmt.Errorf("SSE event received after message_stop")
	}
	var event streamEvent
	if json.Unmarshal(data, &event) != nil {
		return fmt.Errorf("invalid Messages SSE JSON")
	}
	if event.Type == "" || (eventName != "" && eventName != event.Type) {
		return fmt.Errorf("Messages SSE event type is missing or inconsistent")
	}
	if event.Type == "error" {
		if event.Error == nil {
			return fmt.Errorf("Messages stream returned a provider error")
		}
		// An overloaded, rate-limited or failing provider may answer a later
		// request, which the agent sends when this one gave no event.
		errorType := safeErrorType(event.Error.Type)
		transient := errorType == "overloaded_error" || errorType == "api_error" || errorType == "rate_limit_error"
		return &llm.ProviderError{Err: fmt.Errorf("Messages stream returned %s", errorType), Transient: transient}
	}
	if event.Type == "ping" {
		return nil
	}
	switch event.Type {
	case "message_start":
		if s.started || event.Message == nil || event.Message.Type != "message" || event.Message.Role != "assistant" || len(event.Message.Content) != 0 || event.Message.StopReason != nil || event.Message.Usage == nil || event.Message.Usage.InputTokens == nil || event.Message.Usage.OutputTokens == nil {
			return fmt.Errorf("invalid Messages message_start")
		}
		s.started = true
		return s.updateUsage(event.Message.Usage)
	case "content_block_start":
		if !s.started || s.finishing || event.Index == nil || *event.Index != len(s.blocks) || len(s.blocks) >= maxContentBlocks || event.ContentBlock == nil || (len(s.blocks) > 0 && !s.blocks[len(s.blocks)-1].closed) {
			return fmt.Errorf("invalid Messages content_block_start")
		}
		block := *event.ContentBlock
		switch block.Type {
		case "text":
		case "thinking", "redacted_thinking":
			// The reasoning of the model, adaptive by default on the recent
			// models: never shown to the client, given back to the model in
			// the next requests of the turn.
		case "tool_use":
			if s.calls >= maxToolCalls {
				return fmt.Errorf("Messages tool call count exceeds %d", maxToolCalls)
			}
			if err := (llm.Event{Type: llm.EventToolCall, ToolCall: &llm.ToolCall{ID: block.ID, Name: block.Name, Arguments: block.Input}}).Validate(); err != nil {
				return fmt.Errorf("invalid Messages tool_use block")
			}
			if _, exists := s.callIDs[block.ID]; exists {
				return fmt.Errorf("duplicate Messages tool call ID")
			}
			s.callIDs[block.ID] = struct{}{}
			s.calls++
		default:
			return fmt.Errorf("unsupported Messages content block (only text, thinking and client tool_use are supported)")
		}
		s.blocks = append(s.blocks, pendingBlock{block: block})
		pending := &s.blocks[len(s.blocks)-1]
		pending.thinking = append(pending.thinking, block.Thinking...)
		pending.signature = append(pending.signature, block.Signature...)
		if block.Type == "text" {
			pending.text = append(pending.text, block.Text...)
			return s.emitText(block.Text)
		}
	case "content_block_delta", "content_block_stop":
		if !s.started || s.finishing || event.Index == nil || *event.Index < 0 || *event.Index >= len(s.blocks) {
			return fmt.Errorf("invalid Messages content block index or sequence")
		}
		block := &s.blocks[*event.Index]
		if block.closed {
			return fmt.Errorf("Messages content received after block stop")
		}
		if event.Type == "content_block_stop" {
			block.closed = true
			return nil
		}
		if event.Delta == nil {
			return fmt.Errorf("Messages content delta is missing")
		}
		if block.block.Type == "text" && event.Delta.Type == "text_delta" {
			block.text = append(block.text, event.Delta.Text...)
			return s.emitText(event.Delta.Text)
		}
		if block.block.Type == "thinking" && event.Delta.Type == "thinking_delta" {
			block.thinking = append(block.thinking, event.Delta.Thinking...)
			return nil
		}
		if block.block.Type == "thinking" && event.Delta.Type == "signature_delta" {
			block.signature = append(block.signature, event.Delta.Signature...)
			return nil
		}
		if block.block.Type != "tool_use" || event.Delta.Type != "input_json_delta" {
			return fmt.Errorf("Messages content delta does not match its block")
		}
		if len(block.arguments)+len(event.Delta.PartialJSON) > maxToolArgumentsBytes {
			return fmt.Errorf("Messages tool arguments exceed %d bytes", maxToolArgumentsBytes)
		}
		if len(event.Delta.PartialJSON) > 0 {
			var input map[string]json.RawMessage
			if json.Unmarshal(block.block.Input, &input) != nil || len(input) != 0 {
				return fmt.Errorf("Messages tool input conflicts with JSON deltas")
			}
			block.arguments = append(block.arguments, event.Delta.PartialJSON...)
		}
	case "message_delta":
		if !s.started || event.Delta == nil || (len(s.blocks) > 0 && !s.blocks[len(s.blocks)-1].closed) {
			return fmt.Errorf("invalid Messages message_delta")
		}
		s.finishing = true
		if event.Delta.StopReason != nil {
			if s.stopReason != "" && s.stopReason != *event.Delta.StopReason {
				return fmt.Errorf("Messages stop reason changed")
			}
			s.stopReason = *event.Delta.StopReason
		}
		return s.updateUsage(event.Usage)
	case "message_stop":
		if !s.started || !s.finishing || s.stopReason == "" {
			return fmt.Errorf("Messages message_stop received before a stop reason")
		}
		s.stopped = true
	default:
		// New informational event types may be introduced by the protocol.
		// Content block types/deltas still fail closed to avoid losing history.
	}
	return nil
}

func (s *streamState) updateUsage(usage *wireUsage) error {
	if usage == nil {
		return nil
	}
	for _, field := range []struct {
		source *int64
		target *int64
	}{
		{usage.InputTokens, &s.input}, {usage.OutputTokens, &s.output},
		{usage.CacheCreationInputTokens, &s.cacheCreation}, {usage.CacheReadInputTokens, &s.cacheRead},
	} {
		if field.source == nil {
			continue
		}
		if *field.source < 0 || *field.source < *field.target {
			return fmt.Errorf("invalid Messages cumulative token usage")
		}
		*field.target = *field.source
	}
	return nil
}

func (s *streamState) emitText(text string) error {
	if text == "" {
		return nil
	}
	return s.emitEvent(llm.Event{Type: llm.EventTextDelta, TextDelta: text})
}

func (s *streamState) emitEvent(event llm.Event) error {
	if err := event.Validate(); err != nil {
		return fmt.Errorf("invalid Messages stream event")
	}
	return s.emit(event)
}

func (s *streamState) complete() error {
	finish := mapFinishReason(s.stopReason)
	if s.stopReason == "pause_turn" {
		return fmt.Errorf("Messages server-side tool continuation is unsupported")
	}
	if (finish == llm.FinishReasonToolCalls) != (s.calls != 0) {
		return fmt.Errorf("Messages stop reason is inconsistent with tool calls")
	}
	var calls []llm.ToolCall
	for _, block := range s.blocks {
		if !block.closed {
			return fmt.Errorf("Messages stream contains an unfinished content block")
		}
		if block.block.Type != "tool_use" {
			continue
		}
		arguments := block.block.Input
		if len(block.arguments) > 0 {
			arguments = block.arguments
		}
		call := llm.ToolCall{ID: block.block.ID, Name: block.block.Name, Arguments: arguments}
		if err := (llm.Event{Type: llm.EventToolCall, ToolCall: &call}).Validate(); err != nil {
			return fmt.Errorf("Messages tool arguments are not a valid JSON object")
		}
		calls = append(calls, call)
	}
	var total int64
	for _, count := range []int64{s.input, s.cacheCreation, s.cacheRead, s.output} {
		if count > math.MaxInt64-total {
			return fmt.Errorf("Messages token usage overflows")
		}
		total += count
	}
	for _, call := range calls {
		if err := s.emitEvent(llm.Event{Type: llm.EventToolCall, ToolCall: &call}); err != nil {
			return err
		}
	}
	if reasoning := answerReasoning(s.blocks, s.fingerprint); reasoning != nil {
		if err := s.emitEvent(llm.Event{Type: llm.EventReasoning, Reasoning: reasoning}); err != nil {
			return err
		}
	}
	if err := s.emitEvent(llm.Event{Type: llm.EventUsage, Usage: &llm.Usage{InputTokens: s.input + s.cacheCreation + s.cacheRead, OutputTokens: s.output, TotalTokens: total,
		CacheReadTokens: s.cacheRead, CacheWriteTokens: s.cacheCreation}}); err != nil {
		return err
	}
	return s.emitEvent(llm.Event{Type: llm.EventCompleted, FinishReason: finish})
}

func mapFinishReason(reason string) llm.FinishReason {
	switch reason {
	case "end_turn", "stop_sequence":
		return llm.FinishReasonCompleted
	case "tool_use":
		return llm.FinishReasonToolCalls
	case "max_tokens", "model_context_window_exceeded":
		return llm.FinishReasonLength
	case "refusal":
		return llm.FinishReasonContentFilter
	default:
		return llm.FinishReasonOther
	}
}
