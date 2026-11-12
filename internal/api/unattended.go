package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"net/http"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/agui"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/llm"
)

// UnattendedRunner runs a turn on no history: the agent of a backend.
type UnattendedRunner interface {
	RunTurn(context.Context, []llm.Message, string, agent.EmitFunc) (agent.TurnResult, error)
}

// maxUnattendedToolCalls bounds the calls a summary lists; the count goes on.
const maxUnattendedToolCalls = 256

// UnattendedSummary is the result of the RUN_FINISHED of an unattended run:
// what the run did, by identifiers only, never the arguments of a call nor
// its result, as the audit.
type UnattendedSummary struct {
	Unattended   bool                 `json:"unattended"`
	FinishReason string               `json:"finishReason"`
	ToolCalls    []UnattendedToolCall `json:"toolCalls"`
	// ToolCallCount counts the calls, beyond those listed.
	ToolCallCount int `json:"toolCallCount"`
	// NotRunCalls counts the calls that did not run, the tool not offered
	// or the call needing a confirmation: the writes the model asked for.
	NotRunCalls int `json:"notRunCalls"`
	// RefusedCalls counts the calls the bounds of the question refused
	// before they ran, as too many at once or in all; none is listed.
	RefusedCalls int `json:"refusedCalls"`
}

// UnattendedToolCall is a call of a tool an unattended run made.
type UnattendedToolCall struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	DurationMs int64  `json:"durationMs"`
	// Error is set on a call that failed, or did not run.
	Error bool `json:"error"`
	// NotRun tells why a call did not run: "unoffered", the tool not offered,
	// such as a write, or "needs_confirmation"; absent for a call that ran.
	NotRun string `json:"notRun,omitempty"`
}

// errUnattendedSuspended fails a run that ended waiting on somebody, which
// an unattended turn never does: it offers no tool needing a confirmation,
// and refuses such a call rather than wait. Should one suspend all the
// same, the stream ends with RUN_ERROR alone: the agent emits no event for
// a suspension, the conversation service does, once it stored the turn.
var errUnattendedSuspended = errors.New("the unattended turn ended waiting on a decision")

// serveUnattendedRun runs the agent once for a client no user watches, such
// as the scheduler of a collector: the message of the user is the prompt, on
// no history, nothing is stored, and the model is offered the read-only tools
// of the MCP only. The run never waits: it completes, or fails. The stream
// speaks AG-UI, and RUN_FINISHED carries the summary of the calls.
func serveUnattendedRun(backend string, runner UnattendedRunner, limiter *askLimiter, audit auditLogger) http.HandlerFunc {
	return func(response http.ResponseWriter, request *http.Request) {
		attributes := []slog.Attr{slog.Bool("unattended", true)}
		reject := func(status int, code, message string) {
			audit.event(request.Context(), "unattended_run_rejected", appendAttributes(attributes,
				slog.Int("status", status), slog.String("code", code),
			)...)
			writeJSONError(response, status, code, message)
		}
		if runner == nil {
			reject(http.StatusNotFound, "unattended_unavailable", "this backend serves no unattended run")
			return
		}
		input, status, apiError := decodeUnattendedRequest(response, request)
		if apiError != nil {
			reject(status, apiError.Code, apiError.Message)
			return
		}
		attributes = append(attributes, slog.String("run_id", boundedAuditID(input.RunID)), slog.String("protocol_version", boundedAuditID(input.ProtocolVersion)))
		prompt, skipped, err := input.Prompt()
		switch {
		case err != nil:
			reject(http.StatusBadRequest, "invalid_prompt", err.Error())
			return
		case strings.TrimSpace(prompt) == "":
			reject(http.StatusBadRequest, "invalid_prompt", "the message of the user must not be empty")
			return
		case len(prompt) > maxPromptBytes:
			reject(http.StatusRequestEntityTooLarge, "prompt_too_large", "the message of the user is too large")
			return
		}
		identity, ok := auth.IdentityFromContext(request.Context())
		if !ok {
			writeUnauthorized(response)
			return
		}
		// The slots of the conversations: an unattended run is a run of the
		// user, whoever it is, and takes their share.
		key := userKey(backend, identity.ClusterID, identity.Issuer, identity.Subject)
		if !acquireAgentSlot(response, request, limiter, key, audit, "unattended_run", attributes) {
			return
		}
		defer limiter.release(key)

		codec := &aguiCodec{encoder: agui.NewEncoder(input.ThreadID, input.RunID)}
		if newer, _ := input.Version(); newer {
			codec.opening = append(codec.opening, codec.encoder.Warning("the client declares AG-UI "+input.ProtocolVersion+", newer than "+agui.ProtocolVersion+": the run is served in "+agui.ProtocolVersion, nil)...)
		}
		if len(skipped) > 0 {
			codec.opening = append(codec.opening, codec.encoder.Warning("the agent reads the text parts of a message only: the other parts were skipped", skipped)...)
		}
		type start struct {
			index int
			at    time.Time
		}
		var (
			lock    sync.Mutex
			summary = UnattendedSummary{Unattended: true, ToolCalls: []UnattendedToolCall{}}
			started = map[string]start{}
		)
		codec.encoder.SetResult(func() any {
			lock.Lock()
			defer lock.Unlock()
			return summary
		})
		record := func(event agent.Event) {
			lock.Lock()
			defer lock.Unlock()
			switch event.Type {
			case agent.EventToolStarted:
				summary.ToolCallCount++
				if len(summary.ToolCalls) < maxUnattendedToolCalls {
					started[event.ToolCallID] = start{index: len(summary.ToolCalls), at: eventTime(event)}
					summary.ToolCalls = append(summary.ToolCalls, UnattendedToolCall{ID: boundedAuditID(event.ToolCallID), Name: boundedAuditID(event.ToolName)})
				}
			case agent.EventToolFinished:
				if event.NotRun != "" {
					summary.NotRunCalls++
				}
				if begun, ok := started[event.ToolCallID]; ok {
					call := &summary.ToolCalls[begun.index]
					call.DurationMs = max(eventTime(event).Sub(begun.at).Milliseconds(), 0)
					call.Error = event.ToolError
					call.NotRun = event.NotRun
					delete(started, event.ToolCallID)
				}
			case agent.EventToolCallsRefused:
				if event.Refused != nil {
					summary.RefusedCalls += event.Refused.Asked
				}
			case agent.EventCompleted:
				summary.FinishReason = string(event.FinishReason)
			}
		}
		streamRun(response, request, audit, "unattended_run", attributes, codec, func(emit agent.EmitFunc) error {
			result, err := runner.RunTurn(agent.WithUnattended(request.Context()), nil, prompt, func(event agent.Event) error {
				record(event)
				return emit(event)
			})
			if err == nil && result.Suspended != nil {
				return errUnattendedSuspended
			}
			return err
		})
	}
}

// decodeUnattendedRequest reads the RunAgentInput of an unattended run: one
// message of the user, no history, no tools of the client, no resume. The
// thread names the run for the client only; nothing is stored under it.
func decodeUnattendedRequest(response http.ResponseWriter, request *http.Request) (agui.RunAgentInput, int, *APIError) {
	mediaType, _, err := mime.ParseMediaType(request.Header.Get("Content-Type"))
	if err != nil || mediaType != "application/json" {
		return agui.RunAgentInput{}, http.StatusUnsupportedMediaType, &APIError{Code: "unsupported_media_type", Message: "Content-Type must be application/json"}
	}
	request.Body = http.MaxBytesReader(response, request.Body, maxRunRequestBytes)
	decoder := json.NewDecoder(request.Body)
	var input agui.RunAgentInput
	if err := decoder.Decode(&input); err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			return input, http.StatusRequestEntityTooLarge, &APIError{Code: "request_too_large", Message: "request body is too large"}
		}
		return input, http.StatusBadRequest, &APIError{Code: "invalid_request", Message: "request body must be an AG-UI RunAgentInput"}
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return input, http.StatusBadRequest, &APIError{Code: "invalid_request", Message: "request body must contain one JSON object"}
	}
	validID := func(id string) bool {
		return id != "" && len(id) <= maxRunIDBytes && !strings.ContainsFunc(id, func(r rune) bool { return !unicode.IsPrint(r) || unicode.IsSpace(r) })
	}
	if !validID(input.RunID) {
		return input, http.StatusBadRequest, &APIError{Code: "invalid_run_id", Message: fmt.Sprintf("runId must be at most %d printable bytes without spaces", maxRunIDBytes)}
	}
	if !validID(input.ThreadID) {
		return input, http.StatusBadRequest, &APIError{Code: "invalid_thread_id", Message: fmt.Sprintf("threadId must be at most %d printable bytes without spaces", maxRunIDBytes)}
	}
	if _, err := input.Version(); errors.Is(err, agui.ErrInvalidVersion) {
		return input, http.StatusBadRequest, &APIError{Code: "invalid_protocol_version", Message: err.Error()}
	} else if err != nil {
		return input, http.StatusBadRequest, &APIError{Code: "unsupported_protocol_version", Message: err.Error()}
	}
	if len(input.Tools) > 0 {
		return input, http.StatusBadRequest, &APIError{Code: "invalid_tools", Message: "an unattended run takes no tools of the client"}
	}
	if len(input.Resume) > 0 {
		return input, http.StatusBadRequest, &APIError{Code: "invalid_decision", Message: "an unattended run resumes nothing: it never waits on a decision"}
	}
	if len(input.Messages) != 1 || input.Messages[0].Role != "user" {
		return input, http.StatusBadRequest, &APIError{Code: "invalid_prompt", Message: "an unattended run takes one message of the user, without history"}
	}
	return input, 0, nil
}
