package agent

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/opensvc/ai-agent/internal/llm"
)

type EventType string

const (
	EventTextDelta    EventType = "text_delta"
	EventToolStarted  EventType = "tool_started"
	EventToolFinished EventType = "tool_finished"
	EventUsage        EventType = "usage"
	EventCompleted    EventType = "completed"
	// EventConfirmationRequired ends a turn that waits for the user to
	// confirm a tool call. The conversation service emits it once the
	// suspended turn is stored.
	EventConfirmationRequired EventType = "confirmation_required"
	// EventClientToolsPending ends a turn that waits for the client to
	// answer the calls of its tools, once the suspended turn is stored.
	EventClientToolsPending EventType = "client_tools_pending"
	// EventToolCallsRefused reports a model answer past a bound of the tool
	// calls, none of which ran: for the audit, not for the client.
	EventToolCallsRefused EventType = "tool_calls_refused"
	// EventAgentToolCalled reports a call of an MCP tool the agent made of
	// its own, not the model, such as the read of the version of the cluster
	// a documentation search is about: for the audit, not for the client.
	EventAgentToolCalled EventType = "agent_tool_called"
)

type Event struct {
	Type      EventType
	TextDelta string
	ToolName  string
	ToolError bool
	// ToolCallID, ToolArguments and ToolResult describe a tool event: the
	// call of the model, its JSON arguments, and with tool_finished its
	// JSON result, for a client to show them.
	ToolCallID    string
	ToolArguments json.RawMessage
	ToolResult    json.RawMessage
	Usage         *llm.Usage
	FinishReason  llm.FinishReason
	Confirmation  *Confirmation
	Iteration     int
	// TurnID is set on the completion of a stored turn, for the client to
	// refer to it, such as with a feedback.
	TurnID string
	// PendingCalls are the calls of tools of the client the turn waits on,
	// in the order they were made.
	PendingCalls []string
	// NotRun tells, on a tool_finished event, why the call did not run:
	// NotRunUnoffered or NotRunNeedsConfirmation; empty for a call that ran,
	// successfully or not.
	NotRun string
	// Refused describes the answer of a tool_calls_refused event.
	Refused *RefusedCalls
	// AgentCall describes the call of an agent_tool_called event.
	AgentCall *AgentCall
	// ModelDuration and ModelFirstEvent time the model call of a usage
	// event, on the clock of the agent: from the request to the usage, and
	// to the first event of its stream.
	ModelDuration   time.Duration
	ModelFirstEvent time.Duration
	// At is when the agent emitted the event, on its clock.
	At time.Time
}

// RefusedCalls is a model answer refused for the bound it passed: "answer",
// more than maxToolCallsPerTurn calls at once, or "question", more than
// maxToolCallsPerAsk in all.
type RefusedCalls struct {
	Bound string
	// Asked is the number of calls of the answer, Ran the number the
	// question had run before it.
	Asked, Ran int
}

// AgentCall is a call of an MCP tool the agent made of its own: the tool,
// what for, whether it failed, and how long it took.
type AgentCall struct {
	Tool     string
	Purpose  string
	Error    bool
	Duration time.Duration
}

type EmitFunc func(Event) error

func (e Event) Validate() error {
	if e.Iteration <= 0 {
		return fmt.Errorf("agent event iteration must be positive")
	}
	switch e.Type {
	case EventTextDelta:
		if e.TextDelta == "" || e.ToolName != "" || e.Usage != nil || e.FinishReason != "" {
			return fmt.Errorf("invalid text delta event")
		}
	case EventToolStarted:
		if e.ToolName == "" || e.ToolCallID == "" || e.TextDelta != "" || e.Usage != nil || e.FinishReason != "" || e.ToolError {
			return fmt.Errorf("invalid tool started event")
		}
	case EventToolFinished:
		if e.ToolName == "" || e.ToolCallID == "" || e.TextDelta != "" || e.Usage != nil || e.FinishReason != "" {
			return fmt.Errorf("invalid tool finished event")
		}
	case EventUsage:
		if e.Usage == nil || e.TextDelta != "" || e.ToolName != "" || e.FinishReason != "" {
			return fmt.Errorf("invalid usage event")
		}
	case EventCompleted:
		if e.FinishReason == "" || e.TextDelta != "" || e.ToolName != "" || e.Usage != nil || e.ToolError {
			return fmt.Errorf("invalid completed event")
		}
	case EventConfirmationRequired:
		c := e.Confirmation
		if c == nil || c.ID == "" || c.TurnID == "" || c.ToolName == "" || c.ToolCallID == "" || c.ExpiresAt.IsZero() || len(c.Arguments) == 0 ||
			e.TextDelta != "" || e.ToolName != "" || e.Usage != nil || e.FinishReason != "" || e.ToolError {
			return fmt.Errorf("invalid confirmation required event")
		}
	case EventClientToolsPending:
		if len(e.PendingCalls) == 0 || e.TextDelta != "" || e.ToolName != "" || e.Usage != nil || e.FinishReason != "" || e.ToolError {
			return fmt.Errorf("invalid client tools pending event")
		}
	case EventToolCallsRefused:
		r := e.Refused
		if r == nil || (r.Bound != "answer" && r.Bound != "question") || r.Asked <= 0 || r.Ran < 0 ||
			e.TextDelta != "" || e.ToolName != "" || e.Usage != nil || e.FinishReason != "" || e.ToolError {
			return fmt.Errorf("invalid tool calls refused event")
		}
	case EventAgentToolCalled:
		c := e.AgentCall
		if c == nil || c.Tool == "" || c.Purpose == "" || c.Duration < 0 ||
			e.TextDelta != "" || e.ToolName != "" || e.Usage != nil || e.FinishReason != "" || e.ToolError {
			return fmt.Errorf("invalid agent tool called event")
		}
	default:
		return fmt.Errorf("unsupported agent event type %q", e.Type)
	}
	return nil
}

// The reasons a call of the model did not run, on its tool_finished event.
const (
	// NotRunUnoffered is a call of a tool the turn does not offer, such as
	// a write in an unattended turn.
	NotRunUnoffered = "unoffered"
	// NotRunNeedsConfirmation is a call that needs the confirmation of a
	// user, in an unattended turn.
	NotRunNeedsConfirmation = "needs_confirmation"
)
