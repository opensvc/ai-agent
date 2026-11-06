package api

import (
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"mime"
	"net/http"
	"time"

	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/conversation"
)

const maxFeedbackRequestBytes = 8 << 10

// FeedbackRequest is the feedback of the user on one turn.
type FeedbackRequest struct {
	Rating   string   `json:"rating"`
	Reasons  []string `json:"reasons"`
	Expected string   `json:"expected"`
}

type FeedbackResponse struct {
	Feedback  conversation.Feedback `json:"feedback"`
	CreatedAt time.Time             `json:"created_at"`
	UpdatedAt time.Time             `json:"updated_at"`
}

func servePutFeedback(service ConversationService, audit auditLogger) http.HandlerFunc {
	return func(response http.ResponseWriter, request *http.Request) {
		id, turnID := request.PathValue("id"), request.PathValue("turn_id")
		attributes := []slog.Attr{slog.String("conversation_id", boundedAuditID(id)), slog.String("turn_id", boundedAuditID(turnID))}
		feedbackRequest, status, apiError := decodeFeedbackRequest(response, request)
		if apiError != nil {
			audit.event(request.Context(), "turn_feedback_rejected", appendAttributes(attributes,
				slog.Int("status", status), slog.String("code", apiError.Code))...)
			writeJSONError(response, status, apiError.Code, apiError.Message)
			return
		}
		identity, ok := auth.IdentityFromContext(request.Context())
		if !ok {
			writeUnauthorized(response)
			return
		}
		record, err := service.SetFeedback(request.Context(), identity, id, turnID, conversation.Feedback{
			Rating: feedbackRequest.Rating, Reasons: feedbackRequest.Reasons, Expected: feedbackRequest.Expected,
		})
		if err != nil {
			writeFeedbackError(response, request, audit, err, attributes)
			return
		}
		// The expected text is left out of the audit: it is free text.
		audit.event(request.Context(), "turn_feedback_set", appendAttributes(attributes,
			slog.String("rating", record.Feedback.Rating), slog.Any("reasons", record.Feedback.Reasons),
			slog.Bool("expected", record.Feedback.Expected != ""))...)
		writeJSON(response, http.StatusOK, FeedbackResponse{Feedback: record.Feedback, CreatedAt: record.CreatedAt, UpdatedAt: record.UpdatedAt})
	}
}

func serveDeleteFeedback(service ConversationService, audit auditLogger) http.HandlerFunc {
	return func(response http.ResponseWriter, request *http.Request) {
		id, turnID := request.PathValue("id"), request.PathValue("turn_id")
		attributes := []slog.Attr{slog.String("conversation_id", boundedAuditID(id)), slog.String("turn_id", boundedAuditID(turnID))}
		identity, ok := auth.IdentityFromContext(request.Context())
		if !ok {
			writeUnauthorized(response)
			return
		}
		if err := service.DeleteFeedback(request.Context(), identity, id, turnID); err != nil {
			writeFeedbackError(response, request, audit, err, attributes)
			return
		}
		audit.event(request.Context(), "turn_feedback_deleted", attributes...)
		response.WriteHeader(http.StatusNoContent)
	}
}

func writeFeedbackError(response http.ResponseWriter, request *http.Request, audit auditLogger, err error, attributes []slog.Attr) {
	status, code, message := conversationError(err)
	switch {
	case errors.Is(err, conversation.ErrNotFound):
		status, code, message = http.StatusNotFound, "feedback_target_not_found", "the conversation, the turn or its feedback was not found"
	case errors.Is(err, conversation.ErrUserLimit):
		code, message = "feedback_quota", "your feedback quota is reached: withdraw feedback, or wait for it to expire"
	case errors.Is(err, conversation.ErrLimit):
		code, message = "feedback_limit", "the feedback limit has been reached"
	case errors.Is(err, conversation.ErrInvalid):
		code, message = "invalid_feedback", "the feedback is invalid: a rating up or down, reasons only with down, an expected text of at most 1000 characters, on a completed turn"
	}
	audit.event(request.Context(), "turn_feedback_rejected", appendAttributes(attributes, slog.Int("status", status), slog.String("code", code))...)
	writeJSONError(response, status, code, message)
}

func decodeFeedbackRequest(response http.ResponseWriter, request *http.Request) (FeedbackRequest, int, *APIError) {
	mediaType, _, err := mime.ParseMediaType(request.Header.Get("Content-Type"))
	if err != nil || mediaType != "application/json" {
		return FeedbackRequest{}, http.StatusUnsupportedMediaType, &APIError{Code: "unsupported_media_type", Message: "Content-Type must be application/json"}
	}
	request.Body = http.MaxBytesReader(response, request.Body, maxFeedbackRequestBytes)
	decoder := json.NewDecoder(request.Body)
	decoder.DisallowUnknownFields()
	var feedback FeedbackRequest
	if err := decoder.Decode(&feedback); err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			return FeedbackRequest{}, http.StatusRequestEntityTooLarge, &APIError{Code: "request_too_large", Message: "request body is too large"}
		}
		return FeedbackRequest{}, http.StatusBadRequest, &APIError{Code: "invalid_request", Message: "request body must be a JSON object containing rating, and optionally reasons and expected"}
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return FeedbackRequest{}, http.StatusBadRequest, &APIError{Code: "invalid_request", Message: "request body must contain one JSON object"}
	}
	return feedback, 0, nil
}
