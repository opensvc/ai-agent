package agent

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/opensvc/ai-agent/internal/llm"
)

func failStep(err error) llmStep {
	return func(llm.Request, llm.EmitFunc) error { return err }
}

func overloaded(retryAfter time.Duration) error {
	return &llm.ProviderError{Err: fmt.Errorf("Messages API returned HTTP 529 (overloaded_error)"), Status: 529, Transient: true, RetryAfter: retryAfter}
}

func quickModelRetries(t *testing.T) {
	t.Helper()
	saved := modelRetryBackoff
	modelRetryBackoff = [maxModelRetries]time.Duration{time.Millisecond, time.Millisecond}
	t.Cleanup(func() { modelRetryBackoff = saved })
}

// A request the provider failed for an overload, before any event, is sent
// again: the turn keeps the call it ran, and the retry is audited.
func TestModelRequestRetriedAfterTransientFailure(t *testing.T) {
	quickModelRetries(t)
	session := &fakeSession{
		tools:   []*mcp.Tool{{Name: "health", InputSchema: objectSchema(), Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true}}},
		results: map[string]*mcp.CallToolResult{"health": {Content: []mcp.Content{&mcp.TextContent{Text: "ok"}}}},
	}
	model := &scriptedLLM{t: t, steps: []llmStep{
		toolCallStep("health", "call-1"),
		failStep(overloaded(0)),
		failStep(fmt.Errorf("read Messages stream: %w", &llm.ProviderError{Err: errors.New("Messages stream returned overloaded_error"), Transient: true})),
		textStep("healthy"),
	}}
	var retries []ModelRetry
	result, err := newTestAgent(t, model, session, 4).RunTurn(t.Context(), nil, "health", func(event Event) error {
		if event.Type == EventModelRetried {
			retries = append(retries, *event.Retry)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(session.calls) != 1 || result.Messages[len(result.Messages)-1].Text != "healthy" {
		t.Fatalf("calls = %#v, messages = %#v", session.calls, result.Messages)
	}
	if len(retries) != 2 || retries[0] != (ModelRetry{Attempt: 1, Status: 529, Wait: time.Millisecond}) || retries[1].Attempt != 2 || retries[1].Status != 0 {
		t.Fatalf("retries = %#v", retries)
	}
}

func TestModelRequestNotRetried(t *testing.T) {
	quickModelRetries(t)
	textThenFail := func(_ llm.Request, emit llm.EmitFunc) error {
		if err := emit(llm.Event{Type: llm.EventTextDelta, TextDelta: "partial"}); err != nil {
			return err
		}
		return overloaded(0)
	}
	for name, steps := range map[string][]llmStep{
		// The client was shown its text: a request sent again would repeat it.
		"after an event":  {textThenFail},
		"refused request": {failStep(&llm.ProviderError{Err: errors.New("Messages API returned HTTP 400 (invalid_request_error)"), Status: http.StatusBadRequest})},
		"untyped failure": {failStep(errors.New("read Messages stream: invalid Messages SSE JSON"))},
		"long wait":       {failStep(overloaded(maxModelRetryWait + time.Second))},
		"retries spent":   {failStep(overloaded(0)), failStep(overloaded(0)), failStep(overloaded(0))},
	} {
		t.Run(name, func(t *testing.T) {
			model := &scriptedLLM{t: t, steps: steps}
			retried := 0
			_, err := newTestAgent(t, model, &fakeSession{}, 4).RunTurn(t.Context(), nil, "health", func(event Event) error {
				if event.Type == EventModelRetried {
					retried++
				}
				return nil
			})
			if err == nil || model.position != len(steps) || retried != len(steps)-1 {
				t.Fatalf("err = %v, requests = %d, retried = %d", err, model.position, retried)
			}
		})
	}
}

// A retry never waits past the deadline of the turn.
func TestModelRetryRespectsTheDeadline(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), minModelRequestTime)
	defer cancel()
	if _, ok := modelRetry(ctx, overloaded(time.Second), 0); ok {
		t.Fatal("retry past the deadline of the turn")
	}
	if retry, ok := modelRetry(t.Context(), overloaded(3*time.Second), 0); !ok || retry.Wait != 3*time.Second {
		t.Fatalf("retry = %#v, %v", retry, ok)
	}
}
