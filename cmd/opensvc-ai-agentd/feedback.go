package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"time"

	"github.com/opensvc/ai-agent/internal/config"
	"github.com/opensvc/ai-agent/internal/conversation"
	conversationsqlite "github.com/opensvc/ai-agent/internal/conversation/sqlite"
)

// feedbackExport is one line of the feedback export. It carries no identity
// and no conversation or turn ID.
type feedbackExport struct {
	ID        string                    `json:"id"`
	CreatedAt time.Time                 `json:"created_at"`
	UpdatedAt time.Time                 `json:"updated_at"`
	Rating    string                    `json:"rating"`
	Reasons   []string                  `json:"reasons"`
	Expected  string                    `json:"expected"`
	Context   json.RawMessage           `json:"context"`
	Snapshot  conversation.TurnSnapshot `json:"snapshot"`
}

// runFeedback runs the feedback command: export writes the stored feedback
// as JSON lines, for the operator to review or to send to OpenSVC.
func runFeedback(args []string, output io.Writer) error {
	if len(args) == 0 || args[0] != "export" {
		return fmt.Errorf("usage: opensvc-ai-agentd feedback export [-since 720h] [-with-tool-results]")
	}
	flags := flag.NewFlagSet("feedback export", flag.ContinueOnError)
	since := flags.Duration("since", 30*24*time.Hour, "export the feedback created within this duration")
	withResults := flags.Bool("with-tool-results", false, "keep the truncated tool results, which may hold inventory data")
	if err := flags.Parse(args[1:]); err != nil {
		return err
	}
	if *since <= 0 || flags.NArg() != 0 {
		return fmt.Errorf("usage: opensvc-ai-agentd feedback export [-since 720h] [-with-tool-results]")
	}
	conversationConfig, err := config.LoadConversation()
	if err != nil {
		return fmt.Errorf("load conversation configuration: %w", err)
	}
	ctx := context.Background()
	store, err := conversationsqlite.Open(ctx, conversationsqlite.Config{
		Path:                 conversationConfig.DatabasePath,
		MaxConversations:     conversationConfig.MaxConversations,
		MaxUserConversations: conversationConfig.MaxUserConversations,
		MaxDatabaseBytes:     conversationConfig.MaxDatabaseBytes,
		MaxUserBytes:         conversationConfig.MaxUserBytes,
	})
	if err != nil {
		return fmt.Errorf("open conversation store: %w", err)
	}
	defer store.Close()
	encoder := json.NewEncoder(output)
	return store.ExportFeedback(ctx, time.Now().Add(-*since), func(record conversation.FeedbackRecord) error {
		line := feedbackExport{
			ID: record.ID, CreatedAt: record.CreatedAt, UpdatedAt: record.UpdatedAt, Rating: record.Feedback.Rating,
			Reasons: record.Feedback.Reasons, Expected: record.Feedback.Expected, Context: record.Context,
		}
		if err := json.Unmarshal(record.Snapshot, &line.Snapshot); err != nil {
			return fmt.Errorf("decode feedback %s snapshot: %w", record.ID, err)
		}
		if !*withResults {
			for i := range line.Snapshot.ToolCalls {
				line.Snapshot.ToolCalls[i].Result = ""
			}
		}
		return encoder.Encode(line)
	})
}
