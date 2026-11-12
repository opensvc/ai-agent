package agent

import (
	"context"
	"slices"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// An unattended turn runs for a client that no user watches, such as the
// scheduler of a collector: nobody answers a question or confirms a call.
// The model is offered only the tools of the MCP that declare themselves
// read-only, so that no call can change anything whatever the prompt, the
// skills or the data the tools return say; the tools of the client are
// refused. The turn never waits: it completes, or fails.

type unattendedKey struct{}

// WithUnattended marks the turns run with ctx unattended.
func WithUnattended(ctx context.Context) context.Context {
	return context.WithValue(ctx, unattendedKey{}, true)
}

// Unattended tells whether the turns run with ctx are unattended.
func Unattended(ctx context.Context) bool {
	unattended, _ := ctx.Value(unattendedKey{}).(bool)
	return unattended
}

// unattendedInstruction is added to the system prompt of an unattended turn.
const unattendedInstruction = "This turn is unattended: no user follows it, nobody can answer a question or confirm an action. " +
	"Only read-only tools are offered; a tool a skill names that is not offered cannot be used: " +
	"say in your answer what it would have been used for. Do not ask questions: answer once, completely."

// unattendedToolResult answers a call that would need a confirmation in an
// unattended turn. The read-only tools need none, and only they are offered:
// the answer guards against a change of that rule rather than a case met.
const unattendedToolResult = "this action needs the confirmation of a user, and this turn is unattended: it was not executed"

// readOnlyTools keeps the tools that declare themselves read-only. A tool
// without annotations, or whose readOnlyHint is false, is left out: an
// unattended turn offers what is known to change nothing, not what is not
// known to change something.
func readOnlyTools(tools []*mcp.Tool) []*mcp.Tool {
	return slices.DeleteFunc(slices.Clone(tools), func(tool *mcp.Tool) bool {
		return tool == nil || tool.Annotations == nil || !tool.Annotations.ReadOnlyHint
	})
}
