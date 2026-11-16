package api

import (
	"context"
	"log/slog"
	"net/http"
	"strings"
	"unicode"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/requestid"
)

const (
	requestIDHeader       = requestid.Header
	requestIDBytes        = requestid.Bytes
	maxAuditIdentityRunes = 256
	maxAuditIDRunes       = 128
)

type auditLogger struct {
	logger *slog.Logger
}

func withRequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		requestID, err := requestid.New()
		if err != nil {
			writeJSONError(response, http.StatusInternalServerError, "request_id_failed", "the request could not be processed")
			return
		}
		response.Header().Set(requestIDHeader, requestID)
		ctx := requestid.With(request.Context(), requestID)
		next.ServeHTTP(response, request.WithContext(ctx))
	})
}

func (a auditLogger) event(ctx context.Context, event string, attributes ...slog.Attr) {
	base := []slog.Attr{
		slog.String("event", event),
		slog.String("request_id", requestid.FromContext(ctx)),
	}
	if identity, ok := auth.IdentityFromContext(ctx); ok {
		base = append(base,
			slog.String("cluster_id", boundedAuditIdentity(identity.ClusterID)),
			slog.String("subject", boundedAuditIdentity(identity.Subject)),
			slog.String("issuer", boundedAuditIdentity(identity.Issuer)),
		)
	}
	base = append(base, attributes...)
	a.logger.LogAttrs(ctx, slog.LevelInfo, "agent audit", base...)
}

func boundedAuditIdentity(value string) string {
	return boundedAuditText(value, maxAuditIdentityRunes)
}

func boundedAuditID(value string) string {
	return boundedAuditText(value, maxAuditIDRunes)
}

func boundedAuditText(value string, maximum int) string {
	value = strings.Map(func(r rune) rune {
		if unicode.IsControl(r) || unicode.In(r, unicode.Cf) {
			return ' '
		}
		return r
	}, value)
	value = strings.Join(strings.Fields(value), " ")
	runes := []rune(value)
	if len(runes) <= maximum {
		return value
	}
	return string(runes[:maximum]) + "…"
}

// interfaceContextAttributes tell the size of the context of the interface
// of a run, never its content; none without a context.
func interfaceContextAttributes(entries []agent.InterfaceEntry) []slog.Attr {
	if len(entries) == 0 {
		return nil
	}
	size := 0
	for _, entry := range entries {
		size += len(entry.Description) + len(entry.Value)
	}
	return []slog.Attr{slog.Int("context_entries", len(entries)), slog.Int("context_bytes", size)}
}
