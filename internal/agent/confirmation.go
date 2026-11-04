package agent

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/opensvc/ai-agent/internal/llm"
)

// Decision is the answer of the user to a confirmation request.
type Decision string

const (
	DecisionApprove Decision = "approve"
	DecisionReject  Decision = "reject"
	// DecisionPreview asks the preview of the pending call again, with the
	// arguments the user edited: nothing runs, and the turn waits again.
	DecisionPreview Decision = "preview"
)

func (d Decision) Valid() bool {
	return d == DecisionApprove || d == DecisionReject || d == DecisionPreview
}

// rejectedToolResult is the tool result the model receives for an action the
// user rejected: the call was not executed.
var rejectedToolResult = json.RawMessage(`{"content":[{"type":"text","text":"` + RejectedToolResultText + `"}]}`)

// RejectedToolResultText is the text of the result of a call the user
// rejected, which the history keeps.
const RejectedToolResultText = "The user rejected this action. It was not executed."

// noPendingConfirmation ends the system message of a new turn: the service
// refuses a new turn while a confirmation is pending, so a prompt never
// answers one, whatever it says.
const noPendingConfirmation = "No confirmation is pending: the previous turn is complete."

// confirmationTarget is the form of a path or node value the confirmation
// state names; another value is left out.
var confirmationTarget = regexp.MustCompile(`^[A-Za-z0-9._/#@:-]{1,128}$`)

// confirmationState is the line the system message of a run ends with: the
// state of the confirmation, which the agent knows and the model would have
// to reconstitute. A resumed turn names the call the user answered, its
// tool and the path and node it targets.
func confirmationState(state *SuspendedTurn, decision Decision) string {
	if decision == "" {
		return noPendingConfirmation
	}
	call, err := state.PendingCall()
	if err != nil {
		return noPendingConfirmation
	}
	arguments := call.Arguments
	if len(state.EditedArguments) > 0 {
		arguments = state.EditedArguments
	}
	var fields map[string]any
	_ = json.Unmarshal(arguments, &fields)
	target := ""
	for _, key := range []string{"path", "node"} {
		if value, ok := fields[key].(string); ok && confirmationTarget.MatchString(value) {
			target += fmt.Sprintf(" %s %s", key, value)
		}
	}
	if target != "" {
		target = " on" + target
	}
	answer := "approved"
	if decision == DecisionReject {
		answer = "rejected"
	}
	return fmt.Sprintf("The confirmation of %s%s was answered: %s. It covers that call only; no other confirmation is pending.", call.Name, target, answer)
}

// SuspendedTurn is the state of a turn waiting for the user to confirm a tool
// call. It holds everything RunTurn needs to resume: the messages of the turn
// so far, and the calls of the current model iteration with the results
// already obtained. The pending call is Calls[len(Results)].
type SuspendedTurn struct {
	Messages       []llm.Message    `json:"messages"`
	Iteration      int              `json:"iteration"`
	TotalToolCalls int              `json:"total_tool_calls"`
	Text           string           `json:"text"`
	Calls          []llm.ToolCall   `json:"calls"`
	Results        []llm.ToolResult `json:"results"`
	// ActiveTools are the tools the searches of the turn found, and
	// Searches their count, for the resumed turn to see the same tools.
	ActiveTools []string `json:"active_tools,omitempty"`
	Searches    int      `json:"searches,omitempty"`
	// ToolSearch tells that the turn searched the catalog: it keeps
	// searching when it resumes, so that its planned search calls run.
	ToolSearch bool `json:"tool_search,omitempty"`
	// CodeSearches is the count of the code searches of the question, bound
	// by maxCodeSearchesPerAsk across its resumes.
	CodeSearches int    `json:"code_searches,omitempty"`
	ToolTitle    string `json:"tool_title"`
	// ArgumentTypes are the media types the input schema of the pending
	// tool gives its text arguments, for the client to show them.
	ArgumentTypes map[string]string `json:"argument_types,omitempty"`
	// InputSchema is the input schema of the pending tool, which the user
	// may edit its arguments against, absent when the call takes no edits.
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
	// EditedArguments are the arguments the user approved the pending call
	// with, in place of those the model proposed, which its message keeps.
	EditedArguments json.RawMessage `json:"edited_arguments,omitempty"`
	// ClientCalls are the calls of tools of the client the turn waits on,
	// by their ID, and ClientResults their answers, given to resume it.
	ClientCalls   []string         `json:"client_calls,omitempty"`
	ClientResults []llm.ToolResult `json:"client_results,omitempty"`
	// Preview tells what the pending call would do, as the preview tool of
	// the server gives it, or PreviewError why it could not; PreviewArguments
	// are the arguments the user edited it was computed for, absent when it
	// is of the arguments of the call.
	Preview          json.RawMessage `json:"preview,omitempty"`
	PreviewError     string          `json:"preview_error,omitempty"`
	PreviewArguments json.RawMessage `json:"preview_arguments,omitempty"`
}

// PendingCall returns the call waiting for confirmation.
func (s SuspendedTurn) PendingCall() (llm.ToolCall, error) {
	if len(s.Results) >= len(s.Calls) {
		return llm.ToolCall{}, fmt.Errorf("suspended turn has no pending tool call")
	}
	return s.Calls[len(s.Results)], nil
}

func (s SuspendedTurn) validate(maxIterations int) error {
	if len(s.Messages) == 0 || s.Messages[0].Role != llm.RoleUser {
		return fmt.Errorf("suspended turn does not start with a user message")
	}
	if s.Iteration <= 0 || s.Iteration >= maxIterations {
		return fmt.Errorf("suspended turn iteration %d is out of range", s.Iteration)
	}
	if len(s.Calls) == 0 || len(s.Calls) > maxToolCallsPerTurn || len(s.Results) >= len(s.Calls) {
		return fmt.Errorf("suspended turn has no pending tool call")
	}
	if s.Searches < 0 || s.Searches > maxSearchesPerTurn || len(s.ActiveTools) > maxModelTools {
		return fmt.Errorf("suspended turn tool search state is out of range")
	}
	if s.CodeSearches < 0 || s.CodeSearches > maxCodeSearchesPerAsk {
		return fmt.Errorf("suspended turn code search count %d is out of range", s.CodeSearches)
	}
	if len(s.ArgumentTypes) > maxArgumentTypes {
		return fmt.Errorf("suspended turn has more than %d argument types", maxArgumentTypes)
	}
	if len(s.InputSchema) > maxEditableSchemaBytes || len(s.EditedArguments) > maxToolArguments {
		return fmt.Errorf("suspended turn input schema or edited arguments are too large")
	}
	if len(s.Preview) > maxPreviewBytes || len(s.PreviewError) > maxPreviewErrorBytes+len("…") || len(s.PreviewArguments) > maxToolArguments {
		return fmt.Errorf("suspended turn preview is too large")
	}
	if len(s.EditedArguments) > 0 && len(s.InputSchema) == 0 {
		return fmt.Errorf("suspended turn holds edited arguments for a call that takes no edits")
	}
	if s.TotalToolCalls < len(s.Calls) || s.TotalToolCalls > maxToolCallsPerAsk {
		return fmt.Errorf("suspended turn tool call count %d is out of range", s.TotalToolCalls)
	}
	for index, result := range s.Results {
		if result.CallID != s.Calls[index].ID {
			return fmt.Errorf("suspended turn result %d does not match its call", index)
		}
	}
	return nil
}

// ArgumentsHash binds a confirmation to the exact arguments shown to the
// user: to their values, whatever their JSON form. The suspended turn is
// stored through json.Marshal, which compacts the arguments of its calls and
// escapes <, > and & in them: the hash of the bytes would change on the way,
// and a call holding them could never be approved.
func ArgumentsHash(arguments json.RawMessage) string {
	sum := sha256.Sum256(canonicalArguments(arguments))
	return hex.EncodeToString(sum[:])
}

// canonicalArguments is one JSON form of the values of arguments: object keys
// sorted, no space, no HTML escape, numbers as written. Arguments that do not
// decode are kept as they are.
func canonicalArguments(arguments json.RawMessage) []byte {
	decoder := json.NewDecoder(bytes.NewReader(arguments))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil || decoder.More() {
		return arguments
	}
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return arguments
	}
	return bytes.TrimSuffix(buffer.Bytes(), []byte("\n"))
}

// Confirmation describes a tool call waiting for the decision of the user.
type Confirmation struct {
	ID          string
	TurnID      string
	ToolName    string
	ToolTitle   string
	Destructive bool
	// ToolCallID is the call of the model waiting for the decision.
	ToolCallID string
	Arguments  json.RawMessage
	// ArgumentTypes are the media types of the text arguments, such as
	// application/yaml, as the input schema of the tool declares them.
	ArgumentTypes map[string]string
	// InputSchema is the input schema of the tool, when the user may edit
	// the arguments before approving them.
	InputSchema json.RawMessage
	ExpiresAt   time.Time
	// Preview, PreviewError and PreviewArguments are those of the
	// suspended turn.
	Preview          json.RawMessage
	PreviewError     string
	PreviewArguments json.RawMessage
}

// requiresConfirmation tells whether a tool call must be confirmed by the
// user before it runs. MCP annotations are hints: a missing destructiveHint
// on a tool that is not read-only means destructive, so the absence of
// annotations requires a confirmation.
func requiresConfirmation(tool *mcp.Tool) bool {
	annotations := tool.Annotations
	if annotations != nil && annotations.ReadOnlyHint {
		return false
	}
	if annotations == nil || annotations.DestructiveHint == nil {
		return true
	}
	return *annotations.DestructiveHint
}

// toolTitle is the human title of a tool, its name when it declares none.
// maxArgumentTypes bounds the argument types a confirmation gives, and
// maxArgumentType the length of an argument name and of a media type.
const (
	maxArgumentTypes = 32
	maxArgumentType  = 128
)

// argumentTypes reads the media types the input schema of a tool gives its
// properties, in contentMediaType as JSON Schema writes it: the format of a
// text argument the client cannot tell from the text, such as a YAML
// definition. A schema the agent cannot read gives none.
func argumentTypes(tool *mcp.Tool) map[string]string {
	data, err := json.Marshal(tool.InputSchema)
	if err != nil {
		return nil
	}
	var schema struct {
		Properties map[string]struct {
			ContentMediaType string `json:"contentMediaType"`
		} `json:"properties"`
	}
	if json.Unmarshal(data, &schema) != nil {
		return nil
	}
	var types map[string]string
	for name, property := range schema.Properties {
		mediaType := property.ContentMediaType
		if mediaType == "" || len(name) > maxArgumentType || len(mediaType) > maxArgumentType || len(types) == maxArgumentTypes {
			continue
		}
		if types == nil {
			types = map[string]string{}
		}
		types[name] = mediaType
	}
	return types
}

func toolTitle(tool *mcp.Tool) string {
	if tool.Title != "" {
		return tool.Title
	}
	if tool.Annotations != nil && tool.Annotations.Title != "" {
		return tool.Annotations.Title
	}
	return tool.Name
}

// maxEditableSchemaBytes bounds the input schema of a tool whose arguments
// the user may edit before approving a call: a larger schema, or one
// without properties to fill, takes no edits.
const maxEditableSchemaBytes = 64 << 10

// editableSchema gives the input schema of a tool the user may edit the
// arguments of a call against: an object schema, within its bound, that
// resolves without loading another schema. Any other takes no edits.
func editableSchema(tool *mcp.Tool) json.RawMessage {
	if tool.InputSchema == nil {
		return nil
	}
	data, err := json.Marshal(tool.InputSchema)
	if err != nil || len(data) > maxEditableSchemaBytes {
		return nil
	}
	if _, err := resolveSchema(data); err != nil {
		return nil
	}
	return data
}

func resolveSchema(data json.RawMessage) (*jsonschema.Resolved, error) {
	var schema jsonschema.Schema
	if err := json.Unmarshal(data, &schema); err != nil {
		return nil, err
	}
	if schema.Type != "object" {
		return nil, fmt.Errorf("the input schema is not an object schema")
	}
	return schema.Resolve(nil)
}

// ErrInvalidEditedArguments refuses arguments the user edited that are not
// a JSON object valid against the input schema of the tool.
var ErrInvalidEditedArguments = errors.New("the edited arguments are invalid")

// ValidateEditedArguments checks the arguments the user edited against the
// input schema of the pending call, a full replacement of the arguments the
// model proposed. The error names the violation, its path and constraint.
func (s SuspendedTurn) ValidateEditedArguments(edited json.RawMessage) error {
	if len(s.InputSchema) == 0 {
		return fmt.Errorf("%w: this action takes no edits", ErrInvalidEditedArguments)
	}
	if len(edited) > maxToolArguments {
		return fmt.Errorf("%w: they exceed %d bytes", ErrInvalidEditedArguments, maxToolArguments)
	}
	if _, err := decodeToolArguments(llm.ToolCall{Name: "edited", Arguments: edited}); err != nil {
		return fmt.Errorf("%w: they must be one JSON object", ErrInvalidEditedArguments)
	}
	resolved, err := resolveSchema(s.InputSchema)
	if err != nil {
		return fmt.Errorf("%w: the input schema of the tool cannot be read", ErrInvalidEditedArguments)
	}
	var instance any
	if err := json.Unmarshal(edited, &instance); err != nil {
		return fmt.Errorf("%w: they must be one JSON object", ErrInvalidEditedArguments)
	}
	if err := resolved.Validate(instance); err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidEditedArguments, err)
	}
	return nil
}

// editedToolResult tells the model, ahead of the result of the call, that
// the user edited its arguments and which it ran with: the message of the
// model keeps the call it proposed.
func editedToolResult(result json.RawMessage, edited json.RawMessage) json.RawMessage {
	note := map[string]string{"type": "text", "text": "The user edited the arguments before approving; the call ran with: " + string(edited)}
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(result, &payload); err != nil {
		return result
	}
	var content []json.RawMessage
	if raw, ok := payload["content"]; ok {
		if err := json.Unmarshal(raw, &content); err != nil {
			return result
		}
	}
	noteData, _ := json.Marshal(note)
	content = append([]json.RawMessage{noteData}, content...)
	payload["content"], _ = json.Marshal(content)
	data, err := json.Marshal(payload)
	if err != nil {
		return result
	}
	return data
}
