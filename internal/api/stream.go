package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"sync"
	"time"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/llm"
)

const (
	maxPromptBytes = 32 << 10
	maxStreamBytes = 16 << 20
)

var (
	// streamWriteTimeout bounds each write of the stream, so that a client
	// reading it slowly is cut off; it is cleared between writes, a tool or
	// a model request may run longer without any.
	streamWriteTimeout = 15 * time.Second
	// streamKeepAliveInterval is how often the stream sends a comment while
	// nothing else is written, which keeps idle proxies from closing it.
	streamKeepAliveInterval = 10 * time.Second
)

// Confirmation is a tool call waiting for the decision of the user, as the
// conversation read returns it: the exact call that runs if the user
// approves it.
type Confirmation struct {
	ID        string           `json:"id"`
	TurnID    string           `json:"turn_id"`
	Tool      ConfirmationTool `json:"tool"`
	Arguments json.RawMessage  `json:"arguments"`
	// ArgumentTypes are the media types of text arguments, such as
	// {"yaml": "application/yaml"}, as the input schema of the tool
	// declares them: the client shows them by it, not by guessing.
	ArgumentTypes map[string]string `json:"argument_types,omitempty"`
	// Editable tells whether the user may edit the arguments before
	// approving them, against InputSchema, the input schema of the tool.
	Editable    bool            `json:"editable"`
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
	ExpiresAt   time.Time       `json:"expires_at"`
	// Preview tells what the call would do, as the preview tool of the
	// server gives it, or PreviewError why it could not; PreviewArguments
	// are the edited arguments it was computed for, absent when it is of
	// the arguments of the call.
	Preview          json.RawMessage `json:"preview,omitempty"`
	PreviewError     string          `json:"preview_error,omitempty"`
	PreviewArguments json.RawMessage `json:"preview_arguments,omitempty"`
}

type ConfirmationTool struct {
	Name        string `json:"name"`
	Title       string `json:"title"`
	Destructive bool   `json:"destructive"`
}

type ErrorResponse struct {
	Error APIError `json:"error"`
}

type APIError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func acquireAgentSlot(response http.ResponseWriter, request *http.Request, limiter *askLimiter, audit auditLogger, prefix string, attributes []slog.Attr) bool {
	if limiter.tryAcquire() {
		return true
	}
	attributes = append(append([]slog.Attr(nil), attributes...),
		slog.Int("status", http.StatusTooManyRequests),
		slog.String("code", "too_many_requests"),
	)
	audit.event(request.Context(), prefix+"_rejected", attributes...)
	response.Header().Set("Retry-After", "1")
	writeJSONError(response, http.StatusTooManyRequests, "too_many_requests", "too many agent requests are already running")
	return false
}

// streamRun streams the events of a run of the agent in AG-UI, within the
// bounds of the stream, and audits it.
func streamRun(response http.ResponseWriter, request *http.Request, audit auditLogger, prefix string, baseAttributes []slog.Attr, codec *aguiCodec, run func(agent.EmitFunc) error) bool {
	flusher, ok := response.(http.Flusher)
	if !ok {
		audit.event(request.Context(), prefix+"_rejected", appendAttributes(baseAttributes,
			slog.Int("status", http.StatusInternalServerError),
			slog.String("code", "streaming_unavailable"),
		)...)
		writeJSONError(response, http.StatusInternalServerError, "streaming_unavailable", "response streaming is unavailable")
		return false
	}
	if err := setStreamWriteDeadline(response); err != nil {
		audit.event(request.Context(), prefix+"_rejected", appendAttributes(baseAttributes,
			slog.Int("status", http.StatusInternalServerError),
			slog.String("code", "streaming_unavailable"),
		)...)
		writeJSONError(response, http.StatusInternalServerError, "streaming_unavailable", "response streaming is unavailable")
		return false
	}
	defer clearStreamWriteDeadline(response)

	response.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	response.Header().Set("Cache-Control", "no-cache")
	response.Header().Set("X-Content-Type-Options", "nosniff")
	response.WriteHeader(http.StatusOK)
	flusher.Flush()
	clearStreamWriteDeadline(response)
	startedAt := time.Now()
	audit.event(request.Context(), prefix+"_started", appendAttributes(baseAttributes, slog.Int("status", http.StatusOK))...)
	opening, openErr := codec.begin()

	writeFailed := false
	completed := false
	finishReason := ""
	lastIteration := 0
	toolCalls := 0
	var usage llm.Usage
	var activeToolName string
	var activeToolStartedAt time.Time
	var streamBytes int64
	// The writes of the events and of the keep-alive comments are
	// serialized, each under its own deadline: an HTTP/2 stream whose write
	// deadline passes is reset, however long ago it was written.
	var writeLock sync.Mutex
	write := func(frame func() error) error {
		writeLock.Lock()
		defer writeLock.Unlock()
		if err := setStreamWriteDeadline(response); err != nil {
			return err
		}
		defer clearStreamWriteDeadline(response)
		return frame()
	}
	stopKeepAlive := make(chan struct{})
	var keepAlive sync.WaitGroup
	keepAlive.Add(1)
	go func() {
		defer keepAlive.Done()
		ticker := time.NewTicker(streamKeepAliveInterval)
		defer ticker.Stop()
		for {
			select {
			case <-stopKeepAlive:
				return
			case <-request.Context().Done():
				return
			case <-ticker.C:
				if err := write(func() error { return writeSSEComment(response, flusher, &streamBytes) }); err != nil {
					return
				}
			}
		}
	}()
	recordEvent := func(event agent.Event) {
		switch event.Type {
		case agent.EventToolStarted:
			toolCalls++
			activeToolName = event.ToolName
			activeToolStartedAt = eventTime(event)
			audit.event(request.Context(), "tool_started",
				slog.String("tool_name", event.ToolName),
				slog.Int("iteration", event.Iteration),
			)
		case agent.EventToolFinished:
			duration := int64(0)
			if activeToolName == event.ToolName && !activeToolStartedAt.IsZero() {
				duration = eventTime(event).Sub(activeToolStartedAt).Milliseconds()
			}
			attributes := []slog.Attr{
				slog.String("tool_name", event.ToolName),
				slog.Int("iteration", event.Iteration),
				slog.Bool("tool_error", event.ToolError),
				slog.Int64("duration_ms", duration),
			}
			if event.ToolError {
				attributes = append(attributes, slog.String("code", "tool_error"))
			}
			audit.event(request.Context(), "tool_finished", attributes...)
			activeToolName = ""
			activeToolStartedAt = time.Time{}
		case agent.EventAgentToolCalled:
			attributes := []slog.Attr{
				slog.String("tool_name", event.AgentCall.Tool),
				slog.String("purpose", event.AgentCall.Purpose),
				slog.Int("iteration", event.Iteration),
				slog.Bool("tool_error", event.AgentCall.Error),
				slog.Int64("duration_ms", event.AgentCall.Duration.Milliseconds()),
			}
			if event.AgentCall.Error {
				attributes = append(attributes, slog.String("code", "tool_error"))
			}
			audit.event(request.Context(), "agent_tool_called", attributes...)
		case agent.EventToolCallsRefused:
			audit.event(request.Context(), "tool_calls_refused",
				slog.Int("iteration", event.Iteration),
				slog.String("bound", event.Refused.Bound),
				slog.Int("calls_asked", event.Refused.Asked),
				slog.Int("calls_ran", event.Refused.Ran),
			)
		case agent.EventUsage:
			usage.InputTokens = saturatingAdd(usage.InputTokens, event.Usage.InputTokens)
			usage.OutputTokens = saturatingAdd(usage.OutputTokens, event.Usage.OutputTokens)
			usage.TotalTokens = saturatingAdd(usage.TotalTokens, event.Usage.TotalTokens)
			audit.event(request.Context(), "llm_usage",
				slog.Int("iteration", event.Iteration),
				slog.Int64("input_tokens", event.Usage.InputTokens),
				slog.Int64("output_tokens", event.Usage.OutputTokens),
				slog.Int64("total_tokens", event.Usage.TotalTokens),
				slog.Int64("duration_ms", event.ModelDuration.Milliseconds()),
				slog.Int64("first_event_ms", event.ModelFirstEvent.Milliseconds()),
			)
		case agent.EventCompleted:
			completed = true
			finishReason = string(event.FinishReason)
		case agent.EventConfirmationRequired:
			// The turn ends here, stored, until the user decides.
			completed = true
			finishReason = string(agent.EventConfirmationRequired)
			audit.event(request.Context(), "confirmation_requested",
				slog.String("tool_name", event.Confirmation.ToolName),
				slog.String("turn_id", boundedAuditID(event.Confirmation.TurnID)),
				slog.Int("iteration", event.Iteration),
			)
		}
	}
	err := openErr
	if err == nil && len(opening) > 0 {
		if err = write(func() error { return writeFrame(response, flusher, &streamBytes, opening) }); err != nil {
			writeFailed = true
		}
	}
	if err == nil {
		err = run(func(event agent.Event) error {
			lastIteration = max(lastIteration, event.Iteration)
			if event.Type != agent.EventToolStarted {
				recordEvent(event)
			}
			frame, err := codec.event(event)
			if err != nil {
				return err
			}
			if len(frame) > 0 {
				if err := write(func() error { return writeFrame(response, flusher, &streamBytes, frame) }); err != nil {
					writeFailed = true
					return err
				}
			}
			if event.Type == agent.EventToolStarted {
				recordEvent(event)
			}
			return nil
		})
	}
	close(stopKeepAlive)
	keepAlive.Wait()
	failureCode := streamFailureCode(err, completed, writeFailed, request.Context().Err())
	if activeToolName != "" && failureCode == "" {
		failureCode = "agent_incomplete"
	}
	if activeToolName != "" {
		audit.event(request.Context(), "tool_finished",
			slog.String("tool_name", activeToolName),
			slog.Int("iteration", lastIteration),
			slog.Bool("tool_error", true),
			slog.Int64("duration_ms", time.Since(activeToolStartedAt).Milliseconds()),
			slog.String("code", failureCode),
		)
	}
	terminalAttributes := []slog.Attr{
		slog.Int64("duration_ms", time.Since(startedAt).Milliseconds()),
		slog.Int("iterations", lastIteration),
		slog.Int("tool_calls", toolCalls),
		slog.Int64("input_tokens", usage.InputTokens),
		slog.Int64("output_tokens", usage.OutputTokens),
		slog.Int64("total_tokens", usage.TotalTokens),
	}
	if failureCode == "" {
		terminalAttributes = append(terminalAttributes, slog.String("finish_reason", finishReason))
		audit.event(request.Context(), prefix+"_completed", appendAttributes(baseAttributes, terminalAttributes...)...)
	} else {
		terminalAttributes = append(terminalAttributes, slog.String("code", failureCode))
		audit.event(request.Context(), prefix+"_failed", appendAttributes(baseAttributes, terminalAttributes...)...)
	}
	if err != nil && !completed && !writeFailed && request.Context().Err() == nil {
		code := "agent_failed"
		message := "the agent could not complete the request"
		if errors.Is(err, context.DeadlineExceeded) {
			code = "request_timeout"
			message = "the agent request timed out"
		} else if errors.Is(err, agent.ErrClientToolConflict) {
			code = "tool_name_conflict"
			message = "a tool of the client is named as a tool of the agent; rename it"
		}
		if frame, err := codec.failure(code, message); err == nil && len(frame) > 0 {
			_ = write(func() error { return writeFrame(response, flusher, &streamBytes, frame) })
		}
	}
	return true
}

func appendAttributes(base []slog.Attr, extra ...slog.Attr) []slog.Attr {
	attributes := make([]slog.Attr, 0, len(base)+len(extra))
	attributes = append(attributes, base...)
	return append(attributes, extra...)
}

func streamFailureCode(err error, completed bool, writeFailed bool, contextErr error) string {
	if writeFailed {
		return "stream_write_failed"
	}
	if errors.Is(contextErr, context.DeadlineExceeded) || errors.Is(err, context.DeadlineExceeded) {
		return "request_timeout"
	}
	if contextErr != nil || errors.Is(err, context.Canceled) {
		return "request_canceled"
	}
	if err != nil {
		if completed {
			return "agent_cleanup_failed"
		}
		return "agent_failed"
	}
	if !completed {
		return "agent_incomplete"
	}
	return ""
}

func saturatingAdd(current int64, value int64) int64 {
	if value < 0 {
		return current
	}
	if value > math.MaxInt64-current {
		return math.MaxInt64
	}
	return current + value
}

func setStreamWriteDeadline(response http.ResponseWriter) error {
	err := http.NewResponseController(response).SetWriteDeadline(time.Now().Add(streamWriteTimeout))
	if errors.Is(err, http.ErrNotSupported) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("set API stream write deadline: %w", err)
	}
	return nil
}

func clearStreamWriteDeadline(response http.ResponseWriter) {
	_ = http.NewResponseController(response).SetWriteDeadline(time.Time{})
}

// writeSSEComment writes a comment line, which SSE clients ignore.
func writeSSEComment(response io.Writer, flusher http.Flusher, streamBytes *int64) error {
	frame := []byte(": keepalive\n\n")
	if *streamBytes+int64(len(frame)) > maxStreamBytes {
		return fmt.Errorf("API stream exceeds %d bytes", maxStreamBytes)
	}
	written, err := response.Write(frame)
	*streamBytes += int64(written)
	if err != nil {
		return fmt.Errorf("write API stream comment: %w", err)
	}
	flusher.Flush()
	return nil
}

// writeFrame writes frames of the stream, within its size.
func writeFrame(response io.Writer, flusher http.Flusher, streamBytes *int64, frame []byte) error {
	if *streamBytes+int64(len(frame)) > maxStreamBytes {
		return fmt.Errorf("API stream exceeds %d bytes", maxStreamBytes)
	}
	written, err := response.Write(frame)
	*streamBytes += int64(written)
	if err != nil {
		return fmt.Errorf("write API stream event: %w", err)
	}
	if written != len(frame) {
		return fmt.Errorf("write API stream event: %w", io.ErrShortWrite)
	}
	flusher.Flush()
	return nil
}

func writeJSONError(response http.ResponseWriter, status int, code string, message string) {
	response.Header().Set("Content-Type", "application/json")
	response.Header().Set("X-Content-Type-Options", "nosniff")
	response.WriteHeader(status)
	_ = json.NewEncoder(response).Encode(ErrorResponse{Error: APIError{Code: code, Message: message}})
}

// eventTime is when the agent emitted an event, the time the stream and the
// history carry; now for an event it did not stamp.
func eventTime(event agent.Event) time.Time {
	if event.At.IsZero() {
		return time.Now()
	}
	return event.At
}
