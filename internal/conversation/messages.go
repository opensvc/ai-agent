package conversation

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/llm"
)

const (
	DefaultMessagePageLimit = 50
	MaxMessagePageLimit     = 100
	MaxMessagePageBytes     = 1 << 20
)

// DisplayMessage is a public transcript projection, not model context. The
// last assistant message of a turn carries the steps of the turn, the calls
// of its tools with bounded arguments and an excerpt of their results, which
// the user saw stream; system prompts and provider state never are.
type DisplayMessage struct {
	ID        string    `json:"id"`
	TurnID    string    `json:"turn_id"`
	Role      string    `json:"role"`
	Text      string    `json:"text"`
	CreatedAt time.Time `json:"created_at"`
	// Feedback is the feedback of the user on the turn of an assistant
	// message, if any.
	Feedback *Feedback `json:"feedback,omitempty"`
	// Steps are the calls of the tools of the turn, in their order, and
	// StepsOmitted those past MaxDisplaySteps; Runs the runs of the turn and
	// Usage its tokens, absent for a turn stored before they were.
	Steps        []DisplayStep `json:"steps,omitempty"`
	StepsOmitted int           `json:"steps_omitted,omitempty"`
	Runs         []DisplayRun  `json:"runs,omitempty"`
	Usage        *DisplayUsage `json:"usage,omitempty"`
	// ModelID is the ID of the model the turn last ran on, for the client to
	// offer it again; absent for a turn recorded before runs had it.
	ModelID string `json:"model_id,omitempty"`
	// Skill is, on the message of the user that starts a turn, the name of
	// the skill the client asked with it, which the agent loaded first;
	// absent when it asked none, or for a turn recorded before runs had it.
	Skill string `json:"skill,omitempty"`
}

// DisplayStep is one call of a tool of a turn, as the stream showed it:
// times in milliseconds of the clock of the agent, absent when the turn
// stored none.
type DisplayStep struct {
	ID        string          `json:"id"`
	Tool      string          `json:"tool"`
	Arguments json.RawMessage `json:"arguments"`
	// ArgumentsOmitted tells arguments past MaxDisplayStepArgumentsBytes,
	// given as null.
	ArgumentsOmitted bool   `json:"arguments_omitted,omitempty"`
	StartedAt        *int64 `json:"started_at,omitempty"`
	EndedAt          *int64 `json:"ended_at,omitempty"`
	// Status is completed, failed (an error of the tool), rejected (by the
	// user) or interrupted (no result).
	Status          string          `json:"status"`
	ResultExcerpt   string          `json:"result_excerpt"`
	ResultTruncated bool            `json:"result_truncated"`
	Refusal         string          `json:"refusal"`
	EditedArguments json.RawMessage `json:"edited_arguments,omitempty"`
}

type DisplayRun struct {
	StartedAt int64  `json:"started_at"`
	EndedAt   *int64 `json:"ended_at,omitempty"`
}

type DisplayUsage struct {
	TotalTokens int64 `json:"total_tokens"`
}

const (
	// MaxDisplaySteps bounds the steps of a turn.
	MaxDisplaySteps = 64
	// MaxDisplayStepArgumentsBytes bounds the arguments of a step: larger
	// ones, such as a configuration file, are left out.
	MaxDisplayStepArgumentsBytes = 16 << 10
	// The excerpt of a result and the text of a refusal, cut on a rune.
	maxDisplayExcerptBytes = 4 << 10
	maxDisplayRefusalBytes = 16 << 10
	// The arguments and the refusals of the steps of a turn, in all: with
	// the excerpts, a turn holds in a page of MaxMessagePageBytes.
	maxDisplayTurnArgumentsBytes = 192 << 10
	maxDisplayTurnRefusalBytes   = 128 << 10
)

// cutUTF8 cuts a text to at most limit bytes on a rune boundary, and tells
// whether it did.
func cutUTF8(text string, limit int) (string, bool) {
	if len(text) <= limit {
		return text, false
	}
	for limit > 0 && !utf8.RuneStart(text[limit]) {
		limit--
	}
	return text[:limit], true
}

// resultText is the text of the parts of a tool result, joined.
func resultText(content json.RawMessage) string {
	var payload struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if json.Unmarshal(content, &payload) != nil {
		return ""
	}
	var texts []string
	for _, part := range payload.Content {
		if part.Type == "text" {
			texts = append(texts, part.Text)
		}
	}
	return strings.Join(texts, "\n")
}

// DisplayTurn gives the steps, the runs and the tokens of a turn, from its
// messages and its times.
func DisplayTurn(messages []llm.Message, timing TurnTiming) (steps []DisplayStep, omitted int, runs []DisplayRun, usage *DisplayUsage) {
	index := map[string]int{}
	argumentsBytes, refusalBytes := 0, 0
	for _, message := range messages {
		for _, call := range message.ToolCalls {
			if len(steps) == MaxDisplaySteps {
				omitted++
				continue
			}
			step := DisplayStep{ID: call.ID, Tool: call.Name, Arguments: call.Arguments, Status: "interrupted"}
			if len(step.Arguments) == 0 {
				step.Arguments = json.RawMessage("{}")
			}
			if len(step.Arguments) > MaxDisplayStepArgumentsBytes || argumentsBytes+len(step.Arguments) > maxDisplayTurnArgumentsBytes {
				step.Arguments, step.ArgumentsOmitted = json.RawMessage("null"), true
			}
			argumentsBytes += len(step.Arguments)
			if times, ok := timing.Calls[call.ID]; ok {
				if times.StartedAt > 0 {
					step.StartedAt = &times.StartedAt
				}
				if times.EndedAt > 0 {
					step.EndedAt = &times.EndedAt
				}
				if len(times.RanWith) > 0 && len(times.RanWith) <= MaxDisplayStepArgumentsBytes && argumentsBytes+len(times.RanWith) <= maxDisplayTurnArgumentsBytes {
					step.EditedArguments = times.RanWith
					argumentsBytes += len(times.RanWith)
				}
			}
			index[call.ID] = len(steps)
			steps = append(steps, step)
		}
		for _, result := range message.ToolResults {
			position, ok := index[result.CallID]
			if !ok {
				continue
			}
			step := &steps[position]
			text := resultText(result.Content)
			switch {
			// The answer of the confirmation may follow the text.
			case result.IsError && strings.HasPrefix(text, agent.RejectedToolResultText):
				// A call the user rejected did not run: it has no end.
				step.Status, step.EndedAt = "rejected", nil
			case result.IsError:
				step.Status = "failed"
				step.Refusal, _ = cutUTF8(text, min(maxDisplayRefusalBytes, max(maxDisplayTurnRefusalBytes-refusalBytes, 0)))
				refusalBytes += len(step.Refusal)
			default:
				step.Status = "completed"
			}
			step.ResultExcerpt, step.ResultTruncated = cutUTF8(string(result.Content), maxDisplayExcerptBytes)
		}
	}
	for _, run := range timing.Runs {
		item := DisplayRun{StartedAt: run.StartedAt}
		if run.EndedAt > 0 {
			ended := run.EndedAt
			item.EndedAt = &ended
		}
		runs = append(runs, item)
	}
	if timing.Usage != nil {
		usage = &DisplayUsage{TotalTokens: timing.Usage.TotalTokens}
	}
	return steps, omitted, runs, usage
}

type MessagePage struct {
	Messages   []DisplayMessage `json:"messages"`
	NextCursor string           `json:"next_cursor"`
}

type MessageQuery struct {
	Limit  int
	Before string
}

// Bounds supplies defaults and validates a strict, exclusive sequence cursor.
// A cursor is only a position within an owned conversation, never authority.
func (q MessageQuery) Bounds() (int, int64, int64, error) {
	limit := q.Limit
	if limit == 0 {
		limit = DefaultMessagePageLimit
	}
	if limit < 1 || limit > MaxMessagePageLimit {
		return 0, 0, 0, fmt.Errorf("%w: invalid message page limit", ErrInvalid)
	}
	if q.Before == "" {
		return limit, math.MaxInt64, math.MaxInt64, nil
	}
	if len(q.Before) > 39 {
		return 0, 0, 0, fmt.Errorf("%w: invalid message cursor", ErrInvalid)
	}
	turnText, messageText, ok := strings.Cut(q.Before, ":")
	turn, turnErr := strconv.ParseInt(turnText, 10, 64)
	message, messageErr := strconv.ParseInt(messageText, 10, 64)
	if !ok || turnErr != nil || messageErr != nil || turn < 1 || message < 1 ||
		q.Before != fmt.Sprintf("%d:%d", turn, message) {
		return 0, 0, 0, fmt.Errorf("%w: invalid message cursor", ErrInvalid)
	}
	return limit, turn, message, nil
}

func (s *Service) Messages(ctx context.Context, identity auth.Identity, id string, query MessageQuery) (MessagePage, error) {
	limit, _, _, err := query.Bounds()
	if err != nil {
		return MessagePage{}, err
	}
	item, err := s.Get(ctx, identity, id)
	if err != nil {
		return MessagePage{}, err
	}
	query.Limit = limit
	return s.store.ListMessages(ctx, item.Owner, item.ID, query)
}
