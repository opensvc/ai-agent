package sqlite

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/opensvc/ai-agent/internal/conversation"
	"github.com/opensvc/ai-agent/internal/llm"
)

// DefaultMaxFeedback bounds the stored feedback, whatever their conversations
// become.
const DefaultMaxFeedback = 4096

// maxFeedbackRecordBytes bounds the snapshot and context of one feedback.
const maxFeedbackRecordBytes = 64 << 10

func (s *Store) TurnMessages(ctx context.Context, owner conversation.Owner, conversationID string, turnID string) (conversation.Turn, []llm.Message, error) {
	if err := validateOwnerAndID(owner, conversationID); err != nil {
		return conversation.Turn{}, nil, err
	}
	if err := validateIdentifier("turn", turnID); err != nil {
		return conversation.Turn{}, nil, err
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return conversation.Turn{}, nil, fmt.Errorf("begin turn messages transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	if err := requireConversation(ctx, tx, owner, conversationID); err != nil {
		return conversation.Turn{}, nil, err
	}
	turn, err := getTurn(ctx, tx, conversationID, turnID)
	if err != nil {
		return conversation.Turn{}, nil, err
	}
	messages, err := turnMessages(ctx, tx, turnID)
	if err != nil {
		return conversation.Turn{}, nil, err
	}
	if err := tx.Commit(); err != nil {
		return conversation.Turn{}, nil, fmt.Errorf("commit turn messages transaction: %w", err)
	}
	return turn, messages, nil
}

func (s *Store) PutFeedback(ctx context.Context, owner conversation.Owner, conversationID string, record conversation.FeedbackRecord) (conversation.FeedbackRecord, error) {
	if err := validateOwnerAndID(owner, conversationID); err != nil {
		return conversation.FeedbackRecord{}, err
	}
	if err := validateIdentifier("turn", record.TurnID); err != nil {
		return conversation.FeedbackRecord{}, err
	}
	if err := validateIdentifier("feedback", record.ID); err != nil {
		return conversation.FeedbackRecord{}, err
	}
	feedback, err := record.Feedback.Validate()
	if err != nil {
		return conversation.FeedbackRecord{}, err
	}
	if record.CreatedAt.IsZero() || !record.ExpiresAt.After(record.CreatedAt) || !json.Valid(record.Snapshot) || !json.Valid(record.Context) {
		return conversation.FeedbackRecord{}, fmt.Errorf("%w: feedback record is incomplete", conversation.ErrInvalid)
	}
	if len(record.Snapshot)+len(record.Context) > maxFeedbackRecordBytes {
		return conversation.FeedbackRecord{}, fmt.Errorf("%w: feedback record exceeds %d bytes", conversation.ErrLimit, maxFeedbackRecordBytes)
	}
	reasons, err := json.Marshal(feedback.Reasons)
	if err != nil {
		return conversation.FeedbackRecord{}, fmt.Errorf("encode feedback reasons: %w", err)
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return conversation.FeedbackRecord{}, fmt.Errorf("begin feedback transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	if err := requireConversation(ctx, tx, owner, conversationID); err != nil {
		return conversation.FeedbackRecord{}, err
	}
	turn, err := getTurn(ctx, tx, conversationID, record.TurnID)
	if err != nil {
		return conversation.FeedbackRecord{}, err
	}
	if turn.Status != conversation.TurnCompleted {
		return conversation.FeedbackRecord{}, fmt.Errorf("%w: only a completed turn takes a feedback", conversation.ErrInvalid)
	}
	stored, err := getFeedback(ctx, tx, record.TurnID)
	switch {
	case errors.Is(err, conversation.ErrNotFound):
		if err := s.makeRoomForFeedback(ctx, tx, owner); err != nil {
			return conversation.FeedbackRecord{}, err
		}
		if _, err := tx.ExecContext(ctx, `
INSERT INTO feedback (id, turn_id, rating, reasons_json, expected, snapshot_json, context_json, created_at, updated_at, expires_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			record.ID, record.TurnID, feedback.Rating, reasons, feedback.Expected, []byte(record.Snapshot), []byte(record.Context),
			toUnixNano(record.CreatedAt), toUnixNano(record.CreatedAt), toUnixNano(record.ExpiresAt)); err != nil {
			return conversation.FeedbackRecord{}, fmt.Errorf("insert feedback: %w", err)
		}
		stored = record
		stored.Feedback = feedback
		stored.UpdatedAt = record.CreatedAt
	case err != nil:
		return conversation.FeedbackRecord{}, err
	default:
		if _, err := tx.ExecContext(ctx, `
UPDATE feedback SET rating = ?, reasons_json = ?, expected = ?, updated_at = ?
WHERE id = ?`, feedback.Rating, reasons, feedback.Expected, toUnixNano(record.CreatedAt), stored.ID); err != nil {
			return conversation.FeedbackRecord{}, fmt.Errorf("update feedback: %w", err)
		}
		stored.Feedback = feedback
		stored.UpdatedAt = record.CreatedAt
	}
	if err := tx.Commit(); err != nil {
		return conversation.FeedbackRecord{}, fmt.Errorf("commit feedback: %w", err)
	}
	return stored, nil
}

// makeRoomForFeedback checks that owner may store one more feedback: at most
// MaxUserFeedback on the turns of their conversations, its issuer and
// subject on every backend and cluster. With the instance full, the oldest
// feedback of a deleted conversation goes, which nobody can withdraw any
// more: one user deleting the conversations they rated does not block the
// feedback of the others for the retention.
func (s *Store) makeRoomForFeedback(ctx context.Context, tx *sql.Tx, owner conversation.Owner) error {
	var owned, count int
	if err := tx.QueryRowContext(ctx, `
SELECT
  (SELECT COUNT(*) FROM feedback AS f JOIN turns AS t ON t.id = f.turn_id JOIN conversations AS c ON c.id = t.conversation_id
   WHERE c.issuer = ? AND c.subject = ?),
  (SELECT COUNT(*) FROM feedback)`, owner.Issuer, owner.Subject).Scan(&owned, &count); err != nil {
		return fmt.Errorf("count feedback: %w", err)
	}
	if owned >= s.config.MaxUserFeedback {
		return fmt.Errorf("%w: a user stores at most %d feedback", conversation.ErrUserLimit, s.config.MaxUserFeedback)
	}
	if count < s.config.MaxFeedback {
		return nil
	}
	result, err := tx.ExecContext(ctx, `
DELETE FROM feedback WHERE id = (SELECT id FROM feedback WHERE turn_id IS NULL ORDER BY created_at, id LIMIT 1)`)
	if err != nil {
		return fmt.Errorf("evict anonymous feedback: %w", err)
	}
	if evicted, err := result.RowsAffected(); err != nil || evicted == 0 {
		return fmt.Errorf("%w: feedback limit reached", conversation.ErrLimit)
	}
	return nil
}

func (s *Store) DeleteFeedback(ctx context.Context, owner conversation.Owner, conversationID string, turnID string) error {
	if err := validateOwnerAndID(owner, conversationID); err != nil {
		return err
	}
	if err := validateIdentifier("turn", turnID); err != nil {
		return err
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin delete feedback transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	if err := requireConversation(ctx, tx, owner, conversationID); err != nil {
		return err
	}
	if _, err := getTurn(ctx, tx, conversationID, turnID); err != nil {
		return err
	}
	result, err := tx.ExecContext(ctx, "DELETE FROM feedback WHERE turn_id = ?", turnID)
	if err != nil {
		return fmt.Errorf("delete feedback: %w", err)
	}
	if affected, err := result.RowsAffected(); err != nil {
		return fmt.Errorf("count deleted feedback: %w", err)
	} else if affected == 0 {
		return conversation.ErrNotFound
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit delete feedback: %w", err)
	}
	return nil
}

func (s *Store) DeleteExpiredFeedback(ctx context.Context, at time.Time, limit int) (int64, error) {
	if at.IsZero() {
		return 0, fmt.Errorf("%w: expiry time is zero", conversation.ErrInvalid)
	}
	if limit < 1 || limit > maxListLimit {
		return 0, fmt.Errorf("%w: expiry delete limit must be between 1 and %d", conversation.ErrInvalid, maxListLimit)
	}
	result, err := s.db.ExecContext(ctx, `
DELETE FROM feedback
WHERE id IN (SELECT id FROM feedback WHERE expires_at <= ? ORDER BY expires_at, id LIMIT ?)`, toUnixNano(at), limit)
	if err != nil {
		return 0, fmt.Errorf("delete expired feedback: %w", err)
	}
	return result.RowsAffected()
}

// ExportFeedback calls emit with each feedback created since the time given,
// the oldest first. The records it emits carry no turn ID.
func (s *Store) ExportFeedback(ctx context.Context, since time.Time, emit func(conversation.FeedbackRecord) error) error {
	rows, err := s.db.QueryContext(ctx, `
SELECT id, rating, reasons_json, expected, snapshot_json, context_json, created_at, updated_at, expires_at
FROM feedback WHERE created_at >= ? ORDER BY created_at, id`, toUnixNano(since))
	if err != nil {
		return fmt.Errorf("read feedback: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		record, err := scanFeedback(rows, false)
		if err != nil {
			return err
		}
		if err := emit(record); err != nil {
			return err
		}
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("iterate feedback: %w", err)
	}
	return nil
}

func getFeedback(ctx context.Context, tx *sql.Tx, turnID string) (conversation.FeedbackRecord, error) {
	record, err := scanFeedback(tx.QueryRowContext(ctx, `
SELECT id, turn_id, rating, reasons_json, expected, snapshot_json, context_json, created_at, updated_at, expires_at
FROM feedback WHERE turn_id = ?`, turnID), true)
	if errors.Is(err, sql.ErrNoRows) {
		return conversation.FeedbackRecord{}, conversation.ErrNotFound
	}
	return record, err
}

func scanFeedback(scanner rowScanner, withTurn bool) (conversation.FeedbackRecord, error) {
	var record conversation.FeedbackRecord
	var turnID sql.NullString
	var reasons, snapshot, agentContext []byte
	var createdAt, updatedAt, expiresAt int64
	targets := []any{&record.ID}
	if withTurn {
		targets = append(targets, &turnID)
	}
	targets = append(targets, &record.Feedback.Rating, &reasons, &record.Feedback.Expected, &snapshot, &agentContext, &createdAt, &updatedAt, &expiresAt)
	if err := scanner.Scan(targets...); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return conversation.FeedbackRecord{}, err
		}
		return conversation.FeedbackRecord{}, fmt.Errorf("scan feedback: %w", err)
	}
	if err := decodeStrictJSON(reasons, &record.Feedback.Reasons); err != nil {
		return conversation.FeedbackRecord{}, fmt.Errorf("decode feedback reasons: %w", err)
	}
	if record.Feedback.Reasons == nil {
		record.Feedback.Reasons = []string{}
	}
	record.TurnID = turnID.String
	record.Snapshot, record.Context = snapshot, agentContext
	record.CreatedAt, record.UpdatedAt, record.ExpiresAt = fromUnixNano(createdAt), fromUnixNano(updatedAt), fromUnixNano(expiresAt)
	return record, nil
}

// turnMessages reads the messages of a turn, in their order.
func turnMessages(ctx context.Context, tx *sql.Tx, turnID string) ([]llm.Message, error) {
	rows, err := tx.QueryContext(ctx, `
SELECT role, text, tool_calls_json, tool_results_json
FROM messages WHERE turn_id = ? ORDER BY sequence`, turnID)
	if err != nil {
		return nil, fmt.Errorf("read turn messages: %w", err)
	}
	defer rows.Close()
	var messages []llm.Message
	for rows.Next() {
		var role, text string
		var callData, resultData []byte
		if err := rows.Scan(&role, &text, &callData, &resultData); err != nil {
			return nil, fmt.Errorf("scan turn message: %w", err)
		}
		message, err := decodeMessage(role, text, callData, resultData)
		if err != nil {
			return nil, err
		}
		messages = append(messages, message)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate turn messages: %w", err)
	}
	return messages, nil
}
