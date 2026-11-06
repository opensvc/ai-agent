package conversation

import (
	"context"
	"errors"
	"time"

	"github.com/opensvc/ai-agent/internal/llm"
)

var (
	ErrNotFound = errors.New("conversation not found")
	ErrBusy     = errors.New("conversation has a running turn or a turn awaiting confirmation")
	ErrExpired  = errors.New("conversation expired")
	ErrConflict = errors.New("conversation store conflict")
	ErrLimit    = errors.New("conversation store limit exceeded")
	// ErrUserLimit is the quota of conversations of one user reached: the
	// user deletes some, or waits for them to expire; the others are not
	// held back.
	ErrUserLimit       = errors.New("conversation quota of the user reached")
	ErrInvalid         = errors.New("invalid conversation store input")
	ErrMessageTooLarge = errors.New("conversation display message exceeds page size")
	// ErrNotPending refuses a decision on a turn that waits on no such
	// confirmation: unknown, already decided or for another turn.
	ErrNotPending = errors.New("no such pending confirmation")
	// ErrConfirmationExpired refuses a decision arriving after the expiry of
	// its confirmation; the turn has failed without running the call.
	ErrConfirmationExpired = errors.New("confirmation expired")
	// ErrInvalidToolResults refuses answers of the client that are not
	// exactly one per call of its tools the turn waits on.
	ErrInvalidToolResults = errors.New("the answers do not match the calls of the tools of the client the turn waits on")
)

type Store interface {
	CreateConversation(context.Context, Conversation) error
	GetConversation(context.Context, Owner, string) (Conversation, error)
	ListConversations(context.Context, Owner, int) ([]Conversation, error)
	UpdateConversationTitle(context.Context, Owner, string, string, time.Time) (Conversation, error)
	DeleteConversation(context.Context, Owner, string) error

	BeginTurn(context.Context, Owner, string, string, time.Time) (Turn, error)
	// CompleteTurn stores the messages of a turn and adds its last run to
	// the runs of the turn, unless it equals the previous one.
	CompleteTurn(context.Context, Owner, string, string, time.Time, time.Time, []llm.Message, TurnRun, TurnTiming) error
	FailTurn(context.Context, Owner, string, string, TurnStatus, string, time.Time) error
	SuspendTurn(context.Context, Owner, PendingConfirmation) error
	ClaimConfirmation(context.Context, Owner, string, string, string, time.Time) (PendingConfirmation, error)
	GetPendingConfirmation(context.Context, Owner, string, time.Time) (*PendingConfirmation, error)
	// SetTurnPromptID records the ID the client gave the message of the
	// user of a running turn, and tells whether it did: an ID a turn of the
	// conversation holds already is not recorded twice.
	SetTurnPromptID(context.Context, Owner, string, string, string) (bool, error)
	// CancelSuspendedTurn cancels a suspended turn with a code, and deletes
	// what it waits on.
	CancelSuspendedTurn(context.Context, Owner, string, string, string, time.Time) error
	// LoadHistoryIDs loads the messages of the completed turns, with the ID
	// of each, <turn ID>:<sequence>.
	LoadHistoryIDs(context.Context, Owner, string) ([]llm.Message, []string, error)
	ListMessages(context.Context, Owner, string, MessageQuery) (MessagePage, error)

	// TurnMessages reads one turn of an owned conversation and its messages.
	TurnMessages(context.Context, Owner, string, string) (Turn, []llm.Message, error)
	// PutFeedback stores the feedback on a completed turn of an owned
	// conversation, or replaces its rating, reasons and expected text,
	// keeping the snapshot and the creation time of the first one.
	PutFeedback(context.Context, Owner, string, FeedbackRecord) (FeedbackRecord, error)
	DeleteFeedback(context.Context, Owner, string, string) error
	DeleteExpiredFeedback(context.Context, time.Time, int) (int64, error)
	// ExpireConfirmations fails at most the given count of turns whose
	// confirmation expired, deleting the confirmation and its state.
	ExpireConfirmations(context.Context, time.Time, int) (int64, error)

	RecoverInterrupted(context.Context, time.Time) (int64, error)
	DeleteExpired(context.Context, time.Time, int) (int64, error)
	Close() error
}
