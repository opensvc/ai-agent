package sqlite

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"slices"

	"github.com/opensvc/ai-agent/internal/conversation"
)

// ListMessages reads the display messages of the completed turns, the last
// assistant message of a turn carrying its steps. The model's bounded-history
// window is not applied to display.
func (s *Store) ListMessages(ctx context.Context, owner conversation.Owner, id string, query conversation.MessageQuery) (conversation.MessagePage, error) {
	if err := validateOwnerAndID(owner, id); err != nil {
		return conversation.MessagePage{}, err
	}
	limit, beforeTurn, beforeMessage, err := query.Bounds()
	if err != nil {
		return conversation.MessagePage{}, err
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return conversation.MessagePage{}, fmt.Errorf("begin list messages transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	if err := requireConversation(ctx, tx, owner, id); err != nil {
		return conversation.MessagePage{}, err
	}
	// The messages of the page first, the most recent first; the steps of
	// their turns once the rows are closed.
	type candidate struct {
		item                          conversation.DisplayMessage
		turnSequence, messageSequence int64
		textValid                     bool
	}
	rows, err := tx.QueryContext(ctx, `
SELECT t.sequence, t.id, t.prompt_id, m.sequence, m.role,
       CASE WHEN length(CAST(m.text AS BLOB)) <= ? THEN m.text ELSE NULL END,
       t.started_at, t.completed_at, f.rating, f.reasons_json, f.expected
FROM messages AS m
JOIN turns AS t ON t.id = m.turn_id
LEFT JOIN feedback AS f ON f.turn_id = t.id AND m.role = 'assistant'
WHERE t.conversation_id = ? AND t.status = 'completed'
  AND (m.role = 'user' OR (m.role = 'assistant' AND m.text <> ''))
  AND (t.sequence < ? OR (t.sequence = ? AND m.sequence < ?))
ORDER BY t.sequence DESC, m.sequence DESC
LIMIT ?`, conversation.MaxMessagePageBytes, id, beforeTurn, beforeTurn, beforeMessage, limit+1)
	if err != nil {
		return conversation.MessagePage{}, fmt.Errorf("query display messages: %w", err)
	}
	defer rows.Close()
	var candidates []candidate
	for rows.Next() {
		var turnSequence, messageSequence, startedAt int64
		var completedAt sql.NullInt64
		var text, rating, expected sql.NullString
		var reasons []byte
		var item conversation.DisplayMessage
		var promptID string
		if err := rows.Scan(&turnSequence, &item.TurnID, &promptID, &messageSequence, &item.Role, &text, &startedAt, &completedAt, &rating, &reasons, &expected); err != nil {
			return conversation.MessagePage{}, fmt.Errorf("scan display message: %w", err)
		}
		if rating.Valid {
			feedback := conversation.Feedback{Rating: rating.String, Expected: expected.String}
			if err := decodeStrictJSON(reasons, &feedback.Reasons); err != nil {
				return conversation.MessagePage{}, fmt.Errorf("decode feedback reasons: %w", err)
			}
			if feedback.Reasons == nil {
				feedback.Reasons = []string{}
			}
			item.Feedback = &feedback
		}
		if turnSequence < 1 || messageSequence < 1 || !completedAt.Valid {
			return conversation.MessagePage{}, fmt.Errorf("invalid stored display message")
		}
		item.ID = conversation.MessageID(item.TurnID, promptID, messageSequence)
		item.Text = text.String
		item.CreatedAt = fromUnixNano(startedAt)
		if item.Role == "assistant" {
			item.CreatedAt = fromUnixNano(completedAt.Int64)
		}
		candidates = append(candidates, candidate{item: item, turnSequence: turnSequence, messageSequence: messageSequence, textValid: text.Valid})
	}
	if err := rows.Err(); err != nil {
		return conversation.MessagePage{}, fmt.Errorf("iterate display messages: %w", err)
	}
	if err := rows.Close(); err != nil {
		return conversation.MessagePage{}, fmt.Errorf("close display messages: %w", err)
	}
	page := conversation.MessagePage{Messages: make([]conversation.DisplayMessage, 0, limit)}
	// Reserve envelope/cursor space; JSON escaping is measured, not estimated.
	remaining := conversation.MaxMessagePageBytes - 256
	var oldest string
	more := false
	stepped := map[string]bool{}
	// The turn last read, for its steps and its skill asked: the messages of
	// a turn come together, so that each turn is read once.
	var lastTurn conversation.Turn
	turnOf := func(turnID string) (conversation.Turn, error) {
		if lastTurn.ID == turnID {
			return lastTurn, nil
		}
		turn, err := getTurn(ctx, tx, id, turnID)
		if err != nil {
			return conversation.Turn{}, err
		}
		lastTurn = turn
		return turn, nil
	}
	for _, candidate := range candidates {
		if len(page.Messages) == limit {
			more = true
			break
		}
		item := candidate.item
		// The most recent first: the first assistant message of a turn met
		// is its last one, which carries its steps; unless the turn is the
		// one the cursor cuts, its later messages on the page before.
		cut := query.Before != "" && candidate.turnSequence == beforeTurn
		if item.Role == "assistant" && !stepped[item.TurnID] && !cut {
			stepped[item.TurnID] = true
			turn, err := turnOf(item.TurnID)
			if err != nil {
				return conversation.MessagePage{}, err
			}
			messages, err := turnMessages(ctx, tx, item.TurnID)
			if err != nil {
				return conversation.MessagePage{}, err
			}
			item.Steps, item.StepsOmitted, item.Runs, item.Usage = conversation.DisplayTurn(messages, turn.Timing)
			if last := len(turn.Runs) - 1; last >= 0 {
				item.ModelID = turn.Runs[last].ModelID
			}
		}
		// The prompt of a turn names the skill the client asked with it.
		if item.Role == "user" && candidate.messageSequence == 1 {
			turn, err := turnOf(item.TurnID)
			if err != nil {
				return conversation.MessagePage{}, err
			}
			item.Skill = conversation.AskedSkill(turn.Runs)
		}
		encoded, err := json.Marshal(item)
		if err != nil {
			return conversation.MessagePage{}, fmt.Errorf("encode display message: %w", err)
		}
		// A message that holds in no page with its steps is served without
		// them, counted as omitted, rather than leaving the history unread.
		if len(page.Messages) == 0 && candidate.textValid && len(encoded)+1 > remaining && len(item.Steps) > 0 {
			item.StepsOmitted += len(item.Steps)
			item.Steps = nil
			if encoded, err = json.Marshal(item); err != nil {
				return conversation.MessagePage{}, fmt.Errorf("encode display message: %w", err)
			}
		}
		if !candidate.textValid || len(encoded)+1 > remaining {
			if len(page.Messages) == 0 {
				return conversation.MessagePage{}, conversation.ErrMessageTooLarge
			}
			more = true
			break
		}
		page.Messages = append(page.Messages, item)
		remaining -= len(encoded) + 1
		oldest = fmt.Sprintf("%d:%d", candidate.turnSequence, candidate.messageSequence)
	}
	if err := tx.Commit(); err != nil {
		return conversation.MessagePage{}, fmt.Errorf("commit list messages transaction: %w", err)
	}
	if more {
		page.NextCursor = oldest
	}
	slices.Reverse(page.Messages)
	return page, nil
}
