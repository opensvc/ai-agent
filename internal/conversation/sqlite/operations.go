package sqlite

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/opensvc/ai-agent/internal/conversation"
	"github.com/opensvc/ai-agent/internal/llm"
	sqliteDriver "modernc.org/sqlite"
	sqliteLib "modernc.org/sqlite/lib"
)

const (
	maxIdentifierBytes = 128
	maxIdentityBytes   = 256
	maxErrorCodeBytes  = 128
	maxListLimit       = 1000
)

// storedBytes sums the bytes the conversations of owner store, on its
// backend and cluster, and those of the database. The cluster is the
// tenant: a local user of om3 names its node as its issuer, which another
// cluster may name too.
func (s *Store) storedBytes(ctx context.Context, tx *sql.Tx, owner conversation.Owner) (user int64, database int64, err error) {
	if err := tx.QueryRowContext(ctx, `
SELECT COALESCE(SUM(CASE WHEN backend = ? AND cluster_id = ? AND issuer = ? AND subject = ? THEN stored_bytes ELSE 0 END), 0), COALESCE(SUM(stored_bytes), 0)
FROM conversations`, owner.Backend, owner.ClusterID, owner.Issuer, owner.Subject).Scan(&user, &database); err != nil {
		return 0, 0, fmt.Errorf("sum conversation bytes: %w", err)
	}
	return user, database, nil
}

func (s *Store) CreateConversation(ctx context.Context, item conversation.Conversation) error {
	if err := validateConversation(item); err != nil {
		return err
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin create conversation transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	var owned int
	if err := tx.QueryRowContext(ctx, "SELECT COUNT(*) FROM conversations WHERE issuer = ? AND subject = ?", item.Owner.Issuer, item.Owner.Subject).Scan(&owned); err != nil {
		return fmt.Errorf("count the conversations of the user: %w", err)
	}
	if owned >= s.config.MaxUserConversations {
		return fmt.Errorf("%w: a user holds at most %d conversations", conversation.ErrUserLimit, s.config.MaxUserConversations)
	}
	var count int
	if err := tx.QueryRowContext(ctx, "SELECT COUNT(*) FROM conversations").Scan(&count); err != nil {
		return fmt.Errorf("count conversations: %w", err)
	}
	if count >= s.config.MaxConversations {
		return fmt.Errorf("%w: maximum conversation count is %d", conversation.ErrLimit, s.config.MaxConversations)
	}
	_, err = tx.ExecContext(ctx, `
INSERT INTO conversations(id, title, backend, cluster_id, issuer, subject, created_at, updated_at, expires_at, stored_bytes)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)`,
		item.ID,
		item.Title,
		item.Owner.Backend,
		item.Owner.ClusterID,
		item.Owner.Issuer,
		item.Owner.Subject,
		toUnixNano(item.CreatedAt),
		toUnixNano(item.UpdatedAt),
		toUnixNano(item.ExpiresAt),
	)
	if err != nil {
		if isConstraintError(err) {
			return fmt.Errorf("%w: conversation ID %q already exists", conversation.ErrConflict, item.ID)
		}
		return fmt.Errorf("insert conversation: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit create conversation: %w", err)
	}
	return nil
}

func (s *Store) GetConversation(ctx context.Context, owner conversation.Owner, id string) (conversation.Conversation, error) {
	if err := validateOwnerAndID(owner, id); err != nil {
		return conversation.Conversation{}, err
	}
	return scanConversation(s.db.QueryRowContext(ctx, `
SELECT id, title, backend, cluster_id, issuer, subject, created_at, updated_at, expires_at, stored_bytes
FROM conversations
WHERE id = ? AND backend = ? AND cluster_id = ? AND issuer = ? AND subject = ?`, id, owner.Backend, owner.ClusterID, owner.Issuer, owner.Subject))
}

func (s *Store) ListConversations(ctx context.Context, owner conversation.Owner, limit int) ([]conversation.Conversation, error) {
	if err := validateOwner(owner); err != nil {
		return nil, err
	}
	if limit < 1 || limit > maxListLimit {
		return nil, fmt.Errorf("%w: conversation list limit must be between 1 and %d", conversation.ErrInvalid, maxListLimit)
	}
	rows, err := s.db.QueryContext(ctx, `
SELECT id, title, backend, cluster_id, issuer, subject, created_at, updated_at, expires_at, stored_bytes
FROM conversations
WHERE backend = ? AND cluster_id = ? AND issuer = ? AND subject = ?
ORDER BY updated_at DESC, id
LIMIT ?`, owner.Backend, owner.ClusterID, owner.Issuer, owner.Subject, limit)
	if err != nil {
		return nil, fmt.Errorf("list conversations: %w", err)
	}
	defer rows.Close()
	items := make([]conversation.Conversation, 0)
	for rows.Next() {
		item, err := scanConversation(rows)
		if err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate conversations: %w", err)
	}
	return items, nil
}

func (s *Store) DeleteConversation(ctx context.Context, owner conversation.Owner, id string) error {
	if err := validateOwnerAndID(owner, id); err != nil {
		return err
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin delete conversation transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	if err := requireConversation(ctx, tx, owner, id); err != nil {
		return err
	}
	var running int
	if err := tx.QueryRowContext(ctx, "SELECT COUNT(*) FROM turns WHERE conversation_id = ? AND status = 'running'", id).Scan(&running); err != nil {
		return fmt.Errorf("inspect conversation before deletion: %w", err)
	}
	if running != 0 {
		return conversation.ErrBusy
	}
	result, err := tx.ExecContext(ctx, "DELETE FROM conversations WHERE id = ? AND backend = ? AND cluster_id = ? AND issuer = ? AND subject = ?", id, owner.Backend, owner.ClusterID, owner.Issuer, owner.Subject)
	if err != nil {
		return fmt.Errorf("delete conversation: %w", err)
	}
	if affected, err := result.RowsAffected(); err != nil {
		return fmt.Errorf("count deleted conversations: %w", err)
	} else if affected == 0 {
		return conversation.ErrNotFound
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit delete conversation: %w", err)
	}
	return nil
}

func (s *Store) UpdateConversationTitle(ctx context.Context, owner conversation.Owner, id string, title string, updatedAt time.Time) (conversation.Conversation, error) {
	if err := validateOwnerAndID(owner, id); err != nil {
		return conversation.Conversation{}, err
	}
	if err := validateStoredTitle(title, false); err != nil {
		return conversation.Conversation{}, err
	}
	if updatedAt.IsZero() {
		return conversation.Conversation{}, fmt.Errorf("%w: conversation update time is zero", conversation.ErrInvalid)
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return conversation.Conversation{}, fmt.Errorf("begin update conversation title transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	item, err := scanConversation(tx.QueryRowContext(ctx, `
SELECT id, title, backend, cluster_id, issuer, subject, created_at, updated_at, expires_at, stored_bytes
FROM conversations
WHERE id = ? AND backend = ? AND cluster_id = ? AND issuer = ? AND subject = ?`, id, owner.Backend, owner.ClusterID, owner.Issuer, owner.Subject))
	if err != nil {
		return conversation.Conversation{}, err
	}
	updatedAt = updatedAt.UTC()
	if updatedAt.Before(item.UpdatedAt) {
		return conversation.Conversation{}, fmt.Errorf("%w: conversation update time precedes current update", conversation.ErrInvalid)
	}
	if _, err := tx.ExecContext(ctx, `
UPDATE conversations
SET title = ?, updated_at = ?
WHERE id = ? AND backend = ? AND cluster_id = ? AND issuer = ? AND subject = ?`, title, toUnixNano(updatedAt), id, owner.Backend, owner.ClusterID, owner.Issuer, owner.Subject); err != nil {
		return conversation.Conversation{}, fmt.Errorf("update conversation title: %w", err)
	}
	item.Title = title
	item.UpdatedAt = updatedAt
	if err := tx.Commit(); err != nil {
		return conversation.Conversation{}, fmt.Errorf("commit update conversation title: %w", err)
	}
	return item, nil
}

func (s *Store) BeginTurn(ctx context.Context, owner conversation.Owner, conversationID string, turnID string, startedAt time.Time) (conversation.Turn, error) {
	if err := validateOwnerAndID(owner, conversationID); err != nil {
		return conversation.Turn{}, err
	}
	if err := validateIdentifier("turn", turnID); err != nil {
		return conversation.Turn{}, err
	}
	if startedAt.IsZero() {
		return conversation.Turn{}, fmt.Errorf("%w: turn start time is zero", conversation.ErrInvalid)
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return conversation.Turn{}, fmt.Errorf("begin turn transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	if err := requireConversation(ctx, tx, owner, conversationID); err != nil {
		return conversation.Turn{}, err
	}
	if err := failExpiredConfirmation(ctx, tx, conversationID, startedAt); err != nil {
		return conversation.Turn{}, err
	}
	var count, running int
	var maximumSequence int64
	if err := tx.QueryRowContext(ctx, `
SELECT COUNT(*), COALESCE(MAX(sequence), 0), COALESCE(SUM(CASE WHEN status IN ('running', 'awaiting_confirmation') THEN 1 ELSE 0 END), 0)
FROM turns
WHERE conversation_id = ?`, conversationID).Scan(&count, &maximumSequence, &running); err != nil {
		return conversation.Turn{}, fmt.Errorf("inspect conversation turns: %w", err)
	}
	if running != 0 {
		return conversation.Turn{}, conversation.ErrBusy
	}
	if count >= s.config.MaxTurns {
		return conversation.Turn{}, fmt.Errorf("%w: maximum turn count is %d", conversation.ErrLimit, s.config.MaxTurns)
	}
	// A turn without room left to store would call the model and the tools,
	// then fail: it is refused before.
	userBytes, databaseBytes, err := s.storedBytes(ctx, tx, owner)
	if err != nil {
		return conversation.Turn{}, err
	}
	if userBytes >= s.config.MaxUserBytes {
		return conversation.Turn{}, fmt.Errorf("%w: a user stores at most %d bytes", conversation.ErrUserLimit, s.config.MaxUserBytes)
	}
	if databaseBytes >= s.config.MaxDatabaseBytes {
		return conversation.Turn{}, fmt.Errorf("%w: maximum database size is %d bytes", conversation.ErrLimit, s.config.MaxDatabaseBytes)
	}
	turn := conversation.Turn{
		ID:             turnID,
		ConversationID: conversationID,
		Sequence:       maximumSequence + 1,
		Status:         conversation.TurnRunning,
		StartedAt:      startedAt.UTC(),
	}
	_, err = tx.ExecContext(ctx, `
INSERT INTO turns(id, conversation_id, sequence, status, error_code, started_at, completed_at)
VALUES (?, ?, ?, 'running', '', ?, NULL)`, turn.ID, turn.ConversationID, turn.Sequence, toUnixNano(turn.StartedAt))
	if err != nil {
		if isConstraintError(err) {
			return conversation.Turn{}, fmt.Errorf("%w: turn ID %q or running turn already exists", conversation.ErrConflict, turnID)
		}
		return conversation.Turn{}, fmt.Errorf("insert conversation turn: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return conversation.Turn{}, fmt.Errorf("commit begin turn: %w", err)
	}
	return turn, nil
}

func (s *Store) CompleteTurn(ctx context.Context, owner conversation.Owner, conversationID string, turnID string, completedAt time.Time, expiresAt time.Time, messages []llm.Message, run conversation.TurnRun, timing conversation.TurnTiming) error {
	if err := validateOwnerAndID(owner, conversationID); err != nil {
		return err
	}
	if err := validateIdentifier("turn", turnID); err != nil {
		return err
	}
	if completedAt.IsZero() {
		return fmt.Errorf("%w: turn completion time is zero", conversation.ErrInvalid)
	}
	if !expiresAt.After(completedAt) {
		return fmt.Errorf("%w: conversation expiry must follow turn completion", conversation.ErrInvalid)
	}
	encoded, addedBytes, err := encodeMessages(messages)
	if err != nil {
		return fmt.Errorf("%w: %v", conversation.ErrInvalid, err)
	}
	initialTitle := conversation.TitleFromPrompt(messages[0].Text)
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin complete turn transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	storedBytes, err := conversationStoredBytes(ctx, tx, owner, conversationID)
	if err != nil {
		return err
	}
	turn, err := getTurn(ctx, tx, conversationID, turnID)
	if err != nil {
		return err
	}
	if turn.Status != conversation.TurnRunning {
		return fmt.Errorf("%w: turn %q is %s", conversation.ErrConflict, turnID, turn.Status)
	}
	if completedAt.Before(turn.StartedAt) {
		return fmt.Errorf("%w: turn completion precedes start", conversation.ErrInvalid)
	}
	runs, err := encodeRuns(turn.Runs, run)
	if err != nil {
		return err
	}
	encodedTiming, err := encodeTiming(timing)
	if err != nil {
		return err
	}
	var messageCount int
	if err := tx.QueryRowContext(ctx, `
SELECT COUNT(*)
FROM messages AS m
JOIN turns AS t ON t.id = m.turn_id
WHERE t.conversation_id = ?`, conversationID).Scan(&messageCount); err != nil {
		return fmt.Errorf("count conversation messages: %w", err)
	}
	if messageCount+len(encoded) > s.config.MaxMessages {
		return fmt.Errorf("%w: maximum message count is %d", conversation.ErrLimit, s.config.MaxMessages)
	}
	if exceedsLimit(storedBytes, addedBytes, s.config.MaxConversationBytes) {
		return fmt.Errorf("%w: maximum conversation size is %d bytes", conversation.ErrLimit, s.config.MaxConversationBytes)
	}
	userBytes, databaseBytes, err := s.storedBytes(ctx, tx, owner)
	if err != nil {
		return err
	}
	// The share of the user first: one user filling the database would
	// make the turns of every other fail.
	if exceedsLimit(userBytes, addedBytes, s.config.MaxUserBytes) {
		return fmt.Errorf("%w: a user stores at most %d bytes", conversation.ErrUserLimit, s.config.MaxUserBytes)
	}
	if exceedsLimit(databaseBytes, addedBytes, s.config.MaxDatabaseBytes) {
		return fmt.Errorf("%w: maximum database size is %d bytes", conversation.ErrLimit, s.config.MaxDatabaseBytes)
	}
	for _, message := range encoded {
		if _, err := tx.ExecContext(ctx, `
INSERT INTO messages(turn_id, sequence, role, text, tool_calls_json, tool_results_json, stored_bytes)
VALUES (?, ?, ?, ?, ?, ?, ?)`,
			turnID, message.sequence, message.role, message.text, message.toolCalls, message.toolResults, message.storedBytes,
		); err != nil {
			return fmt.Errorf("insert conversation message %d: %w", message.sequence, err)
		}
	}
	if _, err := tx.ExecContext(ctx, `
UPDATE turns
SET status = 'completed', error_code = '', completed_at = ?, runs_json = ?, timing_json = ?
WHERE id = ? AND conversation_id = ? AND status = 'running'`, toUnixNano(completedAt), runs, encodedTiming, turnID, conversationID); err != nil {
		return fmt.Errorf("complete conversation turn: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `
UPDATE conversations
SET title = CASE WHEN title = '' THEN ? ELSE title END,
    stored_bytes = stored_bytes + ?, updated_at = ?, expires_at = ?
WHERE id = ?`, initialTitle, addedBytes, toUnixNano(completedAt), toUnixNano(expiresAt), conversationID); err != nil {
		return fmt.Errorf("update completed conversation: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit completed turn: %w", err)
	}
	return nil
}

func (s *Store) FailTurn(ctx context.Context, owner conversation.Owner, conversationID string, turnID string, status conversation.TurnStatus, errorCode string, completedAt time.Time) error {
	if err := validateOwnerAndID(owner, conversationID); err != nil {
		return err
	}
	if err := validateIdentifier("turn", turnID); err != nil {
		return err
	}
	if status != conversation.TurnFailed && status != conversation.TurnCanceled && status != conversation.TurnInterrupted {
		return fmt.Errorf("%w: invalid terminal turn status %q", conversation.ErrInvalid, status)
	}
	errorCode = strings.TrimSpace(errorCode)
	if errorCode == "" || len(errorCode) > maxErrorCodeBytes || strings.IndexFunc(errorCode, unicode.IsControl) >= 0 {
		return fmt.Errorf("%w: invalid turn error code", conversation.ErrInvalid)
	}
	if completedAt.IsZero() {
		return fmt.Errorf("%w: turn completion time is zero", conversation.ErrInvalid)
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin fail turn transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	if _, err := conversationStoredBytes(ctx, tx, owner, conversationID); err != nil {
		return err
	}
	turn, err := getTurn(ctx, tx, conversationID, turnID)
	if err != nil {
		return err
	}
	if turn.Status != conversation.TurnRunning {
		return fmt.Errorf("%w: turn %q is %s", conversation.ErrConflict, turnID, turn.Status)
	}
	if completedAt.Before(turn.StartedAt) {
		return fmt.Errorf("%w: turn completion precedes start", conversation.ErrInvalid)
	}
	if _, err := tx.ExecContext(ctx, `
UPDATE turns
SET status = ?, error_code = ?, completed_at = ?
WHERE id = ? AND conversation_id = ? AND status = 'running'`, status, errorCode, toUnixNano(completedAt), turnID, conversationID); err != nil {
		return fmt.Errorf("fail conversation turn: %w", err)
	}
	if _, err := tx.ExecContext(ctx, "UPDATE conversations SET updated_at = ? WHERE id = ?", toUnixNano(completedAt), conversationID); err != nil {
		return fmt.Errorf("update failed conversation: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit failed turn: %w", err)
	}
	return nil
}

func (s *Store) LoadHistory(ctx context.Context, owner conversation.Owner, conversationID string) ([]llm.Message, error) {
	messages, _, err := s.LoadHistoryIDs(ctx, owner, conversationID)
	return messages, err
}

// LoadHistoryIDs loads the history of a conversation, as LoadHistory does,
// with the ID of each message, <turn ID>:<sequence>, as the display
// messages and the AG-UI stream name them.
func (s *Store) LoadHistoryIDs(ctx context.Context, owner conversation.Owner, conversationID string) ([]llm.Message, []string, error) {
	if err := validateOwnerAndID(owner, conversationID); err != nil {
		return nil, nil, err
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("begin load history transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	if err := requireConversation(ctx, tx, owner, conversationID); err != nil {
		return nil, nil, err
	}
	rows, err := tx.QueryContext(ctx, `
SELECT t.id, t.prompt_id, m.sequence, m.role, m.text, m.tool_calls_json, m.tool_results_json, m.stored_bytes
FROM messages AS m
JOIN turns AS t ON t.id = m.turn_id
WHERE t.conversation_id = ? AND t.status = 'completed'
ORDER BY t.sequence, m.sequence`, conversationID)
	if err != nil {
		return nil, nil, fmt.Errorf("load conversation history: %w", err)
	}
	defer rows.Close()
	messages := make([]llm.Message, 0)
	ids := make([]string, 0)
	var total int64
	for rows.Next() {
		var turnID, promptID, role, text string
		var sequence int64
		var callData, resultData []byte
		var storedBytes int64
		if err := rows.Scan(&turnID, &promptID, &sequence, &role, &text, &callData, &resultData, &storedBytes); err != nil {
			return nil, nil, fmt.Errorf("scan conversation history: %w", err)
		}
		actualBytes := int64(len(role) + len(text) + len(callData) + len(resultData))
		if storedBytes != actualBytes || storedBytes < 0 {
			return nil, nil, fmt.Errorf("conversation history stored byte count is invalid")
		}
		if exceedsLimit(total, storedBytes, s.config.MaxConversationBytes) {
			return nil, nil, fmt.Errorf("conversation history exceeds %d bytes", s.config.MaxConversationBytes)
		}
		message, err := decodeMessage(role, text, callData, resultData)
		if err != nil {
			return nil, nil, err
		}
		messages = append(messages, message)
		ids = append(ids, conversation.MessageID(turnID, promptID, sequence))
		total += storedBytes
	}
	if err := rows.Err(); err != nil {
		return nil, nil, fmt.Errorf("iterate conversation history: %w", err)
	}
	if err := rows.Close(); err != nil {
		return nil, nil, fmt.Errorf("close conversation history rows: %w", err)
	}
	if len(messages) > s.config.MaxMessages {
		return nil, nil, fmt.Errorf("conversation history exceeds %d messages", s.config.MaxMessages)
	}
	if len(messages) > 0 {
		if err := (llm.Request{Messages: messages}).Validate(); err != nil {
			return nil, nil, fmt.Errorf("validate loaded conversation history: %w", err)
		}
		if err := validateMessageSequence(messages); err != nil {
			return nil, nil, err
		}
	}
	if err := tx.Commit(); err != nil {
		return nil, nil, fmt.Errorf("commit load conversation history: %w", err)
	}
	return messages, ids, nil
}

func (s *Store) RecoverInterrupted(ctx context.Context, at time.Time) (int64, error) {
	if at.IsZero() {
		return 0, fmt.Errorf("%w: recovery time is zero", conversation.ErrInvalid)
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return 0, fmt.Errorf("begin conversation recovery transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	if _, err := tx.ExecContext(ctx, `
UPDATE conversations
SET updated_at = ?
WHERE id IN (SELECT conversation_id FROM turns WHERE status = 'running')`, toUnixNano(at)); err != nil {
		return 0, fmt.Errorf("update recovered conversations: %w", err)
	}
	result, err := tx.ExecContext(ctx, `
UPDATE turns
SET status = 'interrupted', error_code = 'agent_restarted', completed_at = ?
WHERE status = 'running'`, toUnixNano(at))
	if err != nil {
		return 0, fmt.Errorf("recover interrupted turns: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("count recovered turns: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return 0, fmt.Errorf("commit conversation recovery: %w", err)
	}
	return affected, nil
}

func (s *Store) DeleteExpired(ctx context.Context, at time.Time, limit int) (int64, error) {
	if at.IsZero() {
		return 0, fmt.Errorf("%w: expiry time is zero", conversation.ErrInvalid)
	}
	if limit < 1 || limit > maxListLimit {
		return 0, fmt.Errorf("%w: expiry delete limit must be between 1 and %d", conversation.ErrInvalid, maxListLimit)
	}
	result, err := s.db.ExecContext(ctx, `
DELETE FROM conversations
WHERE id IN (
    SELECT id FROM conversations
    WHERE expires_at <= ?
	  AND NOT EXISTS (
	      SELECT 1 FROM turns
	      WHERE turns.conversation_id = conversations.id AND turns.status = 'running'
	  )
    ORDER BY expires_at, id
    LIMIT ?
)`, toUnixNano(at), limit)
	if err != nil {
		return 0, fmt.Errorf("delete expired conversations: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("count expired conversations: %w", err)
	}
	return affected, nil
}

type rowScanner interface {
	Scan(...any) error
}

func scanConversation(scanner rowScanner) (conversation.Conversation, error) {
	var item conversation.Conversation
	var createdAt, updatedAt, expiresAt int64
	if err := scanner.Scan(
		&item.ID,
		&item.Title,
		&item.Owner.Backend,
		&item.Owner.ClusterID,
		&item.Owner.Issuer,
		&item.Owner.Subject,
		&createdAt,
		&updatedAt,
		&expiresAt,
		&item.StoredBytes,
	); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return conversation.Conversation{}, conversation.ErrNotFound
		}
		return conversation.Conversation{}, fmt.Errorf("scan conversation: %w", err)
	}
	item.CreatedAt = fromUnixNano(createdAt)
	item.UpdatedAt = fromUnixNano(updatedAt)
	item.ExpiresAt = fromUnixNano(expiresAt)
	return item, nil
}

func requireConversation(ctx context.Context, tx *sql.Tx, owner conversation.Owner, id string) error {
	var exists int
	if err := tx.QueryRowContext(ctx, `
SELECT 1 FROM conversations
WHERE id = ? AND backend = ? AND cluster_id = ? AND issuer = ? AND subject = ?`, id, owner.Backend, owner.ClusterID, owner.Issuer, owner.Subject).Scan(&exists); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return conversation.ErrNotFound
		}
		return fmt.Errorf("find conversation: %w", err)
	}
	return nil
}

func conversationStoredBytes(ctx context.Context, tx *sql.Tx, owner conversation.Owner, id string) (int64, error) {
	var storedBytes int64
	if err := tx.QueryRowContext(ctx, `
SELECT stored_bytes FROM conversations
WHERE id = ? AND backend = ? AND cluster_id = ? AND issuer = ? AND subject = ?`, id, owner.Backend, owner.ClusterID, owner.Issuer, owner.Subject).Scan(&storedBytes); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return 0, conversation.ErrNotFound
		}
		return 0, fmt.Errorf("read conversation bytes: %w", err)
	}
	return storedBytes, nil
}

func getTurn(ctx context.Context, tx *sql.Tx, conversationID string, turnID string) (conversation.Turn, error) {
	var item conversation.Turn
	var status string
	var runs, timing []byte
	var startedAt int64
	var completedAt sql.NullInt64
	if err := tx.QueryRowContext(ctx, `
SELECT id, conversation_id, sequence, status, error_code, started_at, completed_at, runs_json, prompt_id, timing_json
FROM turns
WHERE id = ? AND conversation_id = ?`, turnID, conversationID).Scan(
		&item.ID,
		&item.ConversationID,
		&item.Sequence,
		&status,
		&item.ErrorCode,
		&startedAt,
		&completedAt,
		&runs,
		&item.PromptID,
		&timing,
	); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return conversation.Turn{}, conversation.ErrNotFound
		}
		return conversation.Turn{}, fmt.Errorf("read conversation turn: %w", err)
	}
	if err := decodeStrictJSON(runs, &item.Runs); err != nil {
		return conversation.Turn{}, fmt.Errorf("decode turn runs: %w", err)
	}
	timingValue, err := decodeTiming(timing)
	if err != nil {
		return conversation.Turn{}, err
	}
	item.Timing = timingValue
	item.Status = conversation.TurnStatus(status)
	item.StartedAt = fromUnixNano(startedAt)
	if completedAt.Valid {
		value := fromUnixNano(completedAt.Int64)
		item.CompletedAt = &value
	}
	return item, nil
}

func validateConversation(item conversation.Conversation) error {
	if err := validateOwnerAndID(item.Owner, item.ID); err != nil {
		return err
	}
	if item.CreatedAt.IsZero() || item.UpdatedAt.IsZero() || item.ExpiresAt.IsZero() {
		return fmt.Errorf("%w: conversation timestamps must be set", conversation.ErrInvalid)
	}
	if item.UpdatedAt.Before(item.CreatedAt) || !item.ExpiresAt.After(item.CreatedAt) {
		return fmt.Errorf("%w: conversation timestamps are inconsistent", conversation.ErrInvalid)
	}
	if item.StoredBytes != 0 {
		return fmt.Errorf("%w: new conversation stored bytes must be zero", conversation.ErrInvalid)
	}
	if err := validateStoredTitle(item.Title, true); err != nil {
		return err
	}
	return nil
}

func validateStoredTitle(title string, allowEmpty bool) error {
	if title == "" && allowEmpty {
		return nil
	}
	normalized, err := conversation.NormalizeTitle(title)
	if err != nil {
		return err
	}
	if normalized != title {
		return fmt.Errorf("%w: conversation title is not normalized", conversation.ErrInvalid)
	}
	return nil
}

func validateOwnerAndID(owner conversation.Owner, id string) error {
	if err := validateOwner(owner); err != nil {
		return err
	}
	return validateIdentifier("conversation", id)
}

func validateOwner(owner conversation.Owner) error {
	if !conversation.ValidBackendName(owner.Backend) || owner.ClusterID != "" && !validIdentity(owner.ClusterID) || !validIdentity(owner.Issuer) || !validIdentity(owner.Subject) {
		return fmt.Errorf("%w: conversation owner is invalid", conversation.ErrInvalid)
	}
	return nil
}

func validIdentity(value string) bool {
	return value != "" && value == strings.TrimSpace(value) && len(value) <= maxIdentityBytes && strings.IndexFunc(value, unsafeIdentityRune) < 0
}

func unsafeIdentityRune(value rune) bool {
	return unicode.IsControl(value) || unicode.In(value, unicode.Cf)
}

func validateIdentifier(kind string, value string) error {
	if value == "" || value != strings.TrimSpace(value) || len(value) > maxIdentifierBytes || strings.IndexFunc(value, unicode.IsControl) >= 0 {
		return fmt.Errorf("%w: %s ID is invalid", conversation.ErrInvalid, kind)
	}
	return nil
}

// maxRunShownTools bounds the tools a run records as shown.
const maxRunShownTools = 256

// encodeRuns adds run to the runs of a turn, unless it equals the last one,
// with its texts announced by servers bounded.
// encodeTiming encodes the times of a turn, empty when it has none.
func encodeTiming(timing conversation.TurnTiming) ([]byte, error) {
	if timing.Empty() {
		return []byte{}, nil
	}
	if err := timing.Validate(); err != nil {
		return nil, err
	}
	encoded, err := json.Marshal(timing)
	if err != nil {
		return nil, fmt.Errorf("encode turn timing: %w", err)
	}
	return encoded, nil
}

// decodeTiming decodes the times of a turn, none for a turn stored before
// they were.
func decodeTiming(data []byte) (conversation.TurnTiming, error) {
	var timing conversation.TurnTiming
	if len(data) == 0 {
		return timing, nil
	}
	if err := decodeStrictJSON(data, &timing); err != nil {
		return conversation.TurnTiming{}, fmt.Errorf("decode turn timing: %w", err)
	}
	if err := timing.Validate(); err != nil {
		return conversation.TurnTiming{}, err
	}
	return timing, nil
}

func encodeRuns(runs []conversation.TurnRun, run conversation.TurnRun) ([]byte, error) {
	for _, field := range []*string{&run.AgentVersion, &run.Model, &run.SystemPromptSHA256, &run.MCPServer, &run.MCPVersion, &run.ToolsSHA256} {
		*field = boundedText(*field, 256)
	}
	if len(run.ShownTools) > maxRunShownTools {
		run.ShownToolsOmitted += len(run.ShownTools) - maxRunShownTools
		run.ShownTools = run.ShownTools[:maxRunShownTools]
	}
	shown := make([]string, 0, len(run.ShownTools))
	for _, name := range run.ShownTools {
		shown = append(shown, boundedText(name, 128))
	}
	run.ShownTools = shown
	if len(runs) == 0 || !runs[len(runs)-1].Equal(run) {
		if len(runs) >= conversation.MaxTurnRuns {
			return nil, fmt.Errorf("%w: a turn records at most %d runs", conversation.ErrLimit, conversation.MaxTurnRuns)
		}
		runs = append(runs, run)
	}
	data, err := json.Marshal(runs)
	if err != nil {
		return nil, fmt.Errorf("encode turn runs: %w", err)
	}
	return data, nil
}

// boundedText cuts text announced by a server to at most limit bytes, on a
// character boundary.
func boundedText(text string, limit int) string {
	if len(text) <= limit {
		return text
	}
	for limit > 0 && !utf8.RuneStart(text[limit]) {
		limit--
	}
	return text[:limit]
}

func toUnixNano(value time.Time) int64 {
	return value.UTC().UnixNano()
}

func fromUnixNano(value int64) time.Time {
	return time.Unix(0, value).UTC()
}

func exceedsLimit(current int64, added int64, limit int64) bool {
	return current < 0 || added < 0 || current > limit || added > limit-current
}

func isConstraintError(err error) bool {
	var sqliteErr *sqliteDriver.Error
	if !errors.As(err, &sqliteErr) {
		return false
	}
	switch sqliteErr.Code() {
	case sqliteLib.SQLITE_CONSTRAINT, sqliteLib.SQLITE_CONSTRAINT_PRIMARYKEY, sqliteLib.SQLITE_CONSTRAINT_UNIQUE:
		return true
	default:
		return false
	}
}
