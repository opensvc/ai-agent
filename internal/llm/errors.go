package llm

import (
	"errors"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// ProviderError is a failure the provider reported: the HTTP status of its
// answer, or 0 for an error event of its stream. Err is the error the
// adapter formatted, which tells no more than it did without this type.
type ProviderError struct {
	Err    error
	Status int
	// Transient tells that the same request may succeed later: a provider
	// overloaded, rate-limited or failing on its own side, not a request it
	// refused.
	Transient bool
	// RetryAfter is the wait the provider asked for, 0 when it gave none.
	RetryAfter time.Duration
}

func (e *ProviderError) Error() string { return e.Err.Error() }

func (e *ProviderError) Unwrap() error { return e.Err }

// TransientStatus tells whether an HTTP status of a provider is one a later
// request may not meet again: a timeout, a rate limit, or an error or an
// overload of the provider (529 for Anthropic).
func TransientStatus(status int) bool {
	switch status {
	case http.StatusRequestTimeout, http.StatusTooManyRequests, http.StatusInternalServerError,
		http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout, 529:
		return true
	}
	return false
}

// maxRetryAfter bounds the wait RetryAfter tells: a longer wait asked is
// told as this one, longer than any wait the agent makes.
const maxRetryAfter = time.Hour

// RetryAfter reads the Retry-After header of a provider answer, a number of
// seconds or an HTTP date, 0 when it is absent, invalid or past, and at most
// maxRetryAfter.
func RetryAfter(header http.Header, now time.Time) time.Duration {
	value := header.Get("Retry-After")
	if value == "" {
		return 0
	}
	if seconds, err := strconv.ParseInt(value, 10, 64); err == nil {
		if seconds <= 0 {
			return 0
		}
		return time.Duration(min(seconds, int64(maxRetryAfter/time.Second))) * time.Second
	} else if errors.Is(err, strconv.ErrRange) && !strings.HasPrefix(value, "-") {
		return maxRetryAfter
	}
	at, err := http.ParseTime(value)
	if err != nil {
		return 0
	}
	if wait := at.Sub(now); wait > 0 {
		return min(wait, maxRetryAfter)
	}
	return 0
}
