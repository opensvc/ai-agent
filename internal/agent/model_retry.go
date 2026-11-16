package agent

import (
	"context"
	"errors"
	"time"

	"github.com/opensvc/ai-agent/internal/llm"
)

const (
	// maxModelRetries bounds the requests a model call sends again after a
	// transient failure of the provider, such as an overload.
	maxModelRetries = 2
	// maxModelRetryWait bounds the wait before a request sent again: a
	// provider asking for longer fails the turn, which holds a run slot of
	// the instance meanwhile.
	maxModelRetryWait = 20 * time.Second
	// minModelRequestTime is the time a request sent again must have before
	// the deadline of the turn, past its wait.
	minModelRequestTime = 10 * time.Second
)

// modelRetryBackoff is the wait before each request sent again, when the
// provider asks for none.
var modelRetryBackoff = [maxModelRetries]time.Duration{time.Second, 4 * time.Second}

// ModelRetry is a model request sent again after a transient failure of the
// provider: the attempt it was, from 1, the HTTP status of the failure, 0
// for an error of the stream, and the wait before the next one.
type ModelRetry struct {
	Attempt int
	Status  int
	Wait    time.Duration
}

// modelRetry tells whether a model request that failed with err, before any
// event, is sent again, and after which wait. A request that gave an event,
// such as text the client was shown, is never sent again.
func modelRetry(ctx context.Context, err error, attempt int) (ModelRetry, bool) {
	var provider *llm.ProviderError
	if attempt >= maxModelRetries || !errors.As(err, &provider) || !provider.Transient {
		return ModelRetry{}, false
	}
	wait := modelRetryBackoff[attempt]
	if provider.RetryAfter > 0 {
		wait = provider.RetryAfter
	}
	if wait > maxModelRetryWait {
		return ModelRetry{}, false
	}
	if deadline, ok := ctx.Deadline(); ok && time.Until(deadline) < wait+minModelRequestTime {
		return ModelRetry{}, false
	}
	return ModelRetry{Attempt: attempt + 1, Status: provider.Status, Wait: wait}, true
}

// waitModelRetry waits before a request sent again, or until ctx ends.
func waitModelRetry(ctx context.Context, wait time.Duration) error {
	timer := time.NewTimer(wait)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
