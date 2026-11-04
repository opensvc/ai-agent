package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"

	"github.com/opensvc/ai-agent/internal/llm"
)

// ClientTool is a tool of the client of the agent, such as a frontend tool
// of AG-UI: the agent offers it to the model, and never runs it. A call of
// it suspends the turn until the client answers it.
type ClientTool struct {
	Name        string
	Description string
	Parameters  json.RawMessage
}

// ErrClientToolConflict refuses a tool of the client named as a tool of the
// MCP or of the agent: a client tool must not take the place of one, such
// as a destructive MCP tool the agent confirms.
var ErrClientToolConflict = errors.New("a tool of the client is named as a tool of the agent")

// unofferedToolResult answers a call of a tool the turn does not offer, such
// as a tool of the client the history called and the client no longer
// declares: a tool error the model reads, rather than a failed turn.
const unofferedToolResult = "no tool of this name is offered in this run, such as a tool of the client it no longer declares: answer without it"

type clientToolsKey struct{}

// WithClientTools gives the turns run with ctx the tools of the client.
func WithClientTools(ctx context.Context, tools []ClientTool) context.Context {
	return context.WithValue(ctx, clientToolsKey{}, slices.Clone(tools))
}

func clientToolsFrom(ctx context.Context) []ClientTool {
	tools, _ := ctx.Value(clientToolsKey{}).([]ClientTool)
	return tools
}

// ClientToolResult is the answer of the client to a call of one of its
// tools, as the model reads the result of a tool.
func ClientToolResult(callID, text string, isError bool) llm.ToolResult {
	return llm.ToolResult{CallID: callID, Content: textToolResult(text), IsError: isError}
}

// WaitsOnClient tells whether the turn waits on the answers of the client
// to calls of its tools, rather than on a confirmation.
func (s SuspendedTurn) WaitsOnClient() bool {
	call, err := s.PendingCall()
	return err == nil && slices.Contains(s.ClientCalls, call.ID)
}

// clientResult gives the answer of the client to a call.
func (s SuspendedTurn) clientResult(callID string) (llm.ToolResult, bool) {
	for _, result := range s.ClientResults {
		if result.CallID == callID {
			return result, true
		}
	}
	return llm.ToolResult{}, false
}

// clientTools gives the tools of the client as the model takes them, and
// refuses one named as a tool of the MCP or of the agent.
func clientTools(offered []ClientTool, taken func(string) bool) ([]llm.Tool, map[string]bool, error) {
	tools := make([]llm.Tool, 0, len(offered))
	names := make(map[string]bool, len(offered))
	for _, tool := range offered {
		if taken(tool.Name) || tool.Name == SearchToolName || tool.Name == ReadResourceToolName || tool.Name == LoadSkillToolName || tool.Name == SearchDocsToolName || names[tool.Name] {
			return nil, nil, errors.Join(ErrClientToolConflict, errors.New("tool "+tool.Name))
		}
		names[tool.Name] = true
		parameters := tool.Parameters
		if len(parameters) == 0 {
			parameters = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		tools = append(tools, llm.Tool{Name: tool.Name, Description: tool.Description, InputSchema: parameters})
	}
	return tools, names, nil
}

const (
	// MaxClientTools bounds the tools of the client of one run, and
	// maxClientToolsBytes their names, descriptions and schemas together.
	MaxClientTools           = 16
	maxClientToolsBytes      = 256 << 10
	maxClientToolSchemaBytes = 64 << 10
)

// clientToolName is the name of a tool the providers of models take, at
// most 64 characters, as OpenAI bounds it: a name the model provider
// refuses would fail the run.
var clientToolName = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

// ValidateClientTools checks the tools of a client as the model takes them,
// within the bounds of the tools of the MCP: a name, unique, that is not a
// tool of the agent, a description and a schema of bounded size. A name of
// a tool of the MCP is refused when the turn reads its catalog.
func ValidateClientTools(tools []ClientTool) error {
	if len(tools) > MaxClientTools {
		return fmt.Errorf("a run offers at most %d tools of the client", MaxClientTools)
	}
	total := 0
	for _, tool := range tools {
		switch {
		case !clientToolName.MatchString(tool.Name):
			return fmt.Errorf("the name of tool %q must be 1 to 64 letters, digits, underscores or hyphens", tool.Name)
		case len(tool.Description) > maxToolDescriptionBytes:
			return fmt.Errorf("the description of tool %q exceeds %d bytes", tool.Name, maxToolDescriptionBytes)
		case len(tool.Parameters) > maxClientToolSchemaBytes:
			return fmt.Errorf("the parameters of tool %q exceed %d bytes", tool.Name, maxClientToolSchemaBytes)
		}
		total += len(tool.Name) + len(tool.Description) + len(tool.Parameters)
	}
	if total > maxClientToolsBytes {
		return fmt.Errorf("the tools of the client exceed %d bytes", maxClientToolsBytes)
	}
	converted, _, err := clientTools(tools, func(string) bool { return false })
	if err != nil {
		return err
	}
	if len(converted) == 0 {
		return nil
	}
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Text: "validate client tools"}}, Tools: converted}
	if err := request.Validate(); err != nil {
		return fmt.Errorf("the tools of the client are invalid: %w", err)
	}
	return nil
}

// ClientToolsForTest gives the tools of the client ctx carries, for the
// tests of the packages that set them.
func ClientToolsForTest(ctx context.Context) []ClientTool { return clientToolsFrom(ctx) }
