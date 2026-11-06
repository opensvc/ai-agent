package sqlite

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/opensvc/ai-agent/internal/conversation"
	_ "modernc.org/sqlite"
)

const (
	DefaultMaxConversations = 4096
	// DefaultMaxUserConversations bounds the conversations of one user,
	// its OpenID issuer and subject, every backend and cluster together:
	// one user does not fill the store of the others.
	DefaultMaxUserConversations = 128
	DefaultMaxTurns             = 128
	DefaultMaxMessages          = 256
	DefaultMaxConversationBytes = 32 << 20
	DefaultMaxDatabaseBytes     = 256 << 20
	// defaultUserBytesShare and defaultUserFeedbackShare are the parts of
	// the database bytes and of the feedback one user may hold by default:
	// a full instance needs that many users.
	defaultUserBytesShare    = 8
	defaultUserFeedbackShare = 8
)

type Config struct {
	Path                 string
	MaxConversations     int
	MaxUserConversations int
	MaxTurns             int
	MaxMessages          int
	MaxConversationBytes int64
	MaxDatabaseBytes     int64
	// MaxUserBytes bounds the bytes the conversations of one user store on
	// one backend and cluster, so that
	// one user does not fill the database for the others; 0 for a share of
	// MaxDatabaseBytes, defaultUserBytesShare, at least MaxConversationBytes.
	MaxUserBytes int64
	MaxFeedback  int
	// MaxUserFeedback bounds the feedback one user stores on the turns of
	// their conversations of one backend and cluster; 0 for a share of
	// MaxFeedback,
	// defaultUserFeedbackShare.
	MaxUserFeedback int
}

type Store struct {
	db     *sql.DB
	config Config
}

var _ conversation.Store = (*Store)(nil)

func Open(ctx context.Context, config Config) (_ *Store, err error) {
	config, err = validateConfig(config)
	if err != nil {
		return nil, err
	}
	path, created, err := prepareDatabaseFile(config.Path)
	if err != nil {
		return nil, err
	}
	config.Path = path
	if created {
		defer func() {
			if err != nil {
				_ = os.Remove(path)
			}
		}()
	}

	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("open conversation SQLite database: %w", err)
	}
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)
	defer func() {
		if err != nil {
			_ = db.Close()
		}
	}()
	if err := db.PingContext(ctx); err != nil {
		return nil, fmt.Errorf("ping conversation SQLite database: %w", err)
	}
	if err := configureDatabase(ctx, db); err != nil {
		return nil, err
	}
	if err := initializeSchema(ctx, db); err != nil {
		return nil, err
	}
	var integrity string
	if err := db.QueryRowContext(ctx, "PRAGMA quick_check").Scan(&integrity); err != nil {
		return nil, fmt.Errorf("check conversation SQLite integrity: %w", err)
	}
	if integrity != "ok" {
		return nil, fmt.Errorf("check conversation SQLite integrity: %s", integrity)
	}
	return &Store{db: db, config: config}, nil
}

func (s *Store) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	if err := s.db.Close(); err != nil {
		return fmt.Errorf("close conversation SQLite database: %w", err)
	}
	return nil
}

func validateConfig(config Config) (Config, error) {
	config.Path = strings.TrimSpace(config.Path)
	if config.Path == "" {
		return Config{}, fmt.Errorf("conversation SQLite path is empty")
	}
	if config.MaxConversations == 0 {
		config.MaxConversations = DefaultMaxConversations
	}
	if config.MaxUserConversations == 0 {
		config.MaxUserConversations = min(DefaultMaxUserConversations, config.MaxConversations)
	}
	if config.MaxTurns == 0 {
		config.MaxTurns = DefaultMaxTurns
	}
	if config.MaxMessages == 0 {
		config.MaxMessages = DefaultMaxMessages
	}
	if config.MaxConversationBytes == 0 {
		config.MaxConversationBytes = DefaultMaxConversationBytes
	}
	if config.MaxDatabaseBytes == 0 {
		config.MaxDatabaseBytes = DefaultMaxDatabaseBytes
	}
	if config.MaxFeedback == 0 {
		config.MaxFeedback = DefaultMaxFeedback
	}
	if config.MaxUserBytes == 0 {
		config.MaxUserBytes = min(max(config.MaxDatabaseBytes/defaultUserBytesShare, config.MaxConversationBytes), config.MaxDatabaseBytes)
	}
	if config.MaxUserFeedback == 0 {
		config.MaxUserFeedback = max(config.MaxFeedback/defaultUserFeedbackShare, 1)
	}
	if config.MaxConversations < 1 || config.MaxUserConversations < 1 || config.MaxTurns < 1 || config.MaxMessages < 1 || config.MaxConversationBytes < 1 || config.MaxDatabaseBytes < 1 || config.MaxFeedback < 1 || config.MaxUserBytes < 1 || config.MaxUserFeedback < 1 {
		return Config{}, fmt.Errorf("conversation SQLite limits must be positive")
	}
	if config.MaxUserConversations > config.MaxConversations {
		return Config{}, fmt.Errorf("conversation SQLite per-user conversation limit exceeds the conversation limit")
	}
	if config.MaxConversationBytes > config.MaxDatabaseBytes {
		return Config{}, fmt.Errorf("conversation SQLite per-conversation byte limit exceeds database byte limit")
	}
	if config.MaxUserBytes > config.MaxDatabaseBytes {
		return Config{}, fmt.Errorf("conversation SQLite per-user byte limit exceeds database byte limit")
	}
	if config.MaxUserFeedback > config.MaxFeedback {
		return Config{}, fmt.Errorf("conversation SQLite per-user feedback limit exceeds the feedback limit")
	}
	return config, nil
}

func prepareDatabaseFile(path string) (string, bool, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return "", false, fmt.Errorf("resolve conversation SQLite path: %w", err)
	}
	directory := filepath.Dir(absolute)
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return "", false, fmt.Errorf("create conversation SQLite directory: %w", err)
	}
	info, err := os.Lstat(directory)
	if err != nil {
		return "", false, fmt.Errorf("inspect conversation SQLite directory: %w", err)
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return "", false, fmt.Errorf("conversation SQLite directory is not a regular directory")
	}
	if info.Mode().Perm()&0o077 != 0 {
		return "", false, fmt.Errorf("conversation SQLite directory permissions must not grant group or other access")
	}

	file, err := os.OpenFile(absolute, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0o600)
	if err == nil {
		if closeErr := file.Close(); closeErr != nil {
			_ = os.Remove(absolute)
			return "", false, fmt.Errorf("close new conversation SQLite file: %w", closeErr)
		}
		return absolute, true, nil
	}
	if !os.IsExist(err) {
		return "", false, fmt.Errorf("create conversation SQLite file: %w", err)
	}
	info, err = os.Lstat(absolute)
	if err != nil {
		return "", false, fmt.Errorf("inspect conversation SQLite file: %w", err)
	}
	if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		return "", false, fmt.Errorf("conversation SQLite path is not a regular file")
	}
	if info.Mode().Perm()&0o077 != 0 {
		return "", false, fmt.Errorf("conversation SQLite file permissions must not grant group or other access")
	}
	return absolute, false, nil
}

func configureDatabase(ctx context.Context, db *sql.DB) error {
	for _, statement := range []string{
		"PRAGMA foreign_keys = ON",
		"PRAGMA synchronous = FULL",
		"PRAGMA busy_timeout = 5000",
		"PRAGMA secure_delete = ON",
	} {
		if _, err := db.ExecContext(ctx, statement); err != nil {
			return fmt.Errorf("configure conversation SQLite database: %w", err)
		}
	}
	var journalMode string
	if err := db.QueryRowContext(ctx, "PRAGMA journal_mode = WAL").Scan(&journalMode); err != nil {
		return fmt.Errorf("enable conversation SQLite WAL: %w", err)
	}
	if !strings.EqualFold(journalMode, "wal") {
		return fmt.Errorf("enable conversation SQLite WAL: got journal mode %q", journalMode)
	}
	return nil
}
