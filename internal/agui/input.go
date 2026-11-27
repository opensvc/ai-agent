package agui

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/llm"
)

// RunAgentInput is the request of an AG-UI client for one run. The agent
// reads the thread, the run, the last message of the user, the context of
// the interface, the tools of the client and the resume of an interrupt; the
// history it keeps is its own, so the other messages and the state are not
// read.
type RunAgentInput struct {
	ThreadID    string `json:"threadId"`
	RunID       string `json:"runId"`
	ParentRunID string `json:"parentRunId,omitempty"`
	// ProtocolVersion is the version of AG-UI the client speaks, absent
	// for a client from before the protocol carried one.
	ProtocolVersion string          `json:"protocolVersion,omitempty"`
	State           json.RawMessage `json:"state,omitempty"`
	Messages        []Message       `json:"messages"`
	Tools           []Tool          `json:"tools"`
	Context         json.RawMessage `json:"context,omitempty"`
	ForwardedProps  json.RawMessage `json:"forwardedProps,omitempty"`
	Resume          []ResumeEntry   `json:"resume,omitempty"`
}

type Message struct {
	ID         string          `json:"id"`
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content,omitempty"`
	ToolCallID string          `json:"toolCallId,omitempty"`
	// Error is set on the answer of a tool of the client that failed.
	Error string `json:"error,omitempty"`
}

// Tool is a tool of the client, which the agent offers the model and the
// client runs.
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// maxToolAnswersBytes bounds the answers of the client to the calls of its
// tools in one run.
const maxToolAnswersBytes = 256 << 10

// ToolAnswer is the answer of the client to a call of one of its tools.
type ToolAnswer struct {
	CallID  string
	Text    string
	IsError bool
}

// Model gives the ID of the model the client chose, in the model member of
// the forwarded properties: empty for the default, when the properties are
// absent, not an object, or without a model. A model that is not a string
// of 1 to llm.MaxModelIDBytes bytes is an error.
func (input RunAgentInput) Model() (string, error) {
	var props map[string]json.RawMessage
	if json.Unmarshal(input.ForwardedProps, &props) != nil {
		return "", nil
	}
	raw, ok := props["model"]
	if !ok || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return "", nil
	}
	var id string
	if err := json.Unmarshal(raw, &id); err != nil || id == "" || len(id) > llm.MaxModelIDBytes {
		return "", fmt.Errorf("forwardedProps.model must be the ID of a model of the agent, a string of 1 to %d bytes", llm.MaxModelIDBytes)
	}
	return id, nil
}

// Skill gives the name of the skill the client asked the new turn to load
// first, in the skill member of the forwarded properties: empty for none,
// when the properties are absent, not an object, or without a skill. A skill
// that is not a string shaped as the name of a skill is an error.
func (input RunAgentInput) Skill() (string, error) {
	var props map[string]json.RawMessage
	if json.Unmarshal(input.ForwardedProps, &props) != nil {
		return "", nil
	}
	raw, ok := props["skill"]
	if !ok || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return "", nil
	}
	var name string
	if err := json.Unmarshal(raw, &name); err != nil || !agent.ValidSkillName(name) {
		return "", fmt.Errorf("forwardedProps.skill must be the name of a skill, lowercase letters, digits and hyphens, at most %d bytes; GET /skills lists them", agent.MaxSkillNameBytes)
	}
	return name, nil
}

// InterfaceContext gives the context of the input, AG-UI Context entries
// {description, value}: what the interface of the user shows, such as the
// page and the record open on it. Absent, null or empty, there is none.
// Other members of an entry are ignored, as the protocol allows them.
func (input RunAgentInput) InterfaceContext() ([]agent.InterfaceEntry, error) {
	raw := bytes.TrimSpace(input.Context)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return nil, nil
	}
	var entries []struct {
		Description *string `json:"description"`
		Value       *string `json:"value"`
	}
	if err := json.Unmarshal(raw, &entries); err != nil {
		return nil, errors.New("the context must be an array of {description, value} strings")
	}
	result := make([]agent.InterfaceEntry, 0, len(entries))
	for index, entry := range entries {
		if entry.Description == nil || entry.Value == nil {
			return nil, fmt.Errorf("context entry %d must have a description and a value", index)
		}
		result = append(result, agent.InterfaceEntry{Description: *entry.Description, Value: *entry.Value})
	}
	if err := agent.ValidateInterfaceContext(result); err != nil {
		return nil, err
	}
	return result, nil
}

// ToolAnswers gives the answers of the client the input ends with: the
// tool messages after its last message of another role, in their order,
// and the content parts the agent cannot read, skipped. An input that does
// not end with tool messages has none.
func (input RunAgentInput) ToolAnswers() ([]ToolAnswer, []string, error) {
	first := len(input.Messages)
	for first > 0 && input.Messages[first-1].Role == "tool" {
		first--
	}
	var answers []ToolAnswer
	var skipped []string
	total := 0
	for _, message := range input.Messages[first:] {
		text, parts, err := contentText(message.Content)
		if err != nil {
			return nil, nil, fmt.Errorf("tool message %q: %w", message.ToolCallID, err)
		}
		skipped = append(skipped, parts...)
		if message.ToolCallID == "" {
			return nil, nil, errors.New("a tool message must name the call it answers with toolCallId")
		}
		answer := ToolAnswer{CallID: message.ToolCallID, Text: text, IsError: message.Error != ""}
		if answer.IsError {
			answer.Text = strings.TrimSpace(message.Error + "\n" + text)
		}
		total += len(answer.Text)
		if total > maxToolAnswersBytes {
			return nil, nil, fmt.Errorf("the answers of the client exceed %d bytes", maxToolAnswersBytes)
		}
		answers = append(answers, answer)
	}
	return answers, skipped, nil
}

// contentText reads a content, a string or a list of parts: the text of its
// text parts joined by line breaks, and the types of the parts skipped.
func contentText(content json.RawMessage) (string, []string, error) {
	if len(content) == 0 {
		return "", nil, nil
	}
	var text string
	if json.Unmarshal(content, &text) == nil {
		return text, nil, nil
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(content, &parts); err != nil {
		return "", nil, errors.New("a content must be a text or a list of content parts")
	}
	var texts, skipped []string
	for _, part := range parts {
		if part.Type != "text" {
			skipped = append(skipped, part.Type)
			continue
		}
		texts = append(texts, part.Text)
	}
	return strings.Join(texts, "\n"), skipped, nil
}

// ResumeEntry answers an interrupt of the previous run.
type ResumeEntry struct {
	InterruptID string          `json:"interruptId"`
	Status      string          `json:"status"`
	Payload     json.RawMessage `json:"payload,omitempty"`
}

// The statuses of a resume entry.
const (
	ResumeResolved  = "resolved"
	ResumeCancelled = "cancelled"
)

// Prompt gives the text of the last message of the input, which must be a
// message of the user: a string, or text parts joined by line breaks. The
// parts the agent cannot read, such as images, are skipped and named, as
// the protocol has a producer do rather than fail the run; a message left
// without text is refused.
func (input RunAgentInput) Prompt() (string, []string, error) {
	if len(input.Messages) == 0 {
		return "", nil, errors.New("messages must end with a message of the user")
	}
	last := input.Messages[len(input.Messages)-1]
	if last.Role != "user" {
		return "", nil, fmt.Errorf("the last message must be a message of the user, not %q", last.Role)
	}
	text, skipped, err := contentText(last.Content)
	if err != nil {
		return "", nil, fmt.Errorf("the message of the user: %w", err)
	}
	if text == "" && len(skipped) > 0 {
		return "", skipped, errors.New("the message of the user holds no text part, the only parts the agent reads")
	}
	return text, skipped, nil
}

// Approval reads the answer to a confirmation: approved, true or false, of
// a resolved entry, and the arguments the user edited, a full replacement of
// those of the call, with an approval only; a cancelled entry rejects the
// call. Any other member of the payload is refused.
// PreviewRequest reads a resolved resume that asks the preview of the call
// again with edited arguments, {"action": "preview", "editedArgs": {…}}:
// the arguments, and whether the resume is one.
func (entry ResumeEntry) PreviewRequest() (editedArgs json.RawMessage, ok bool, err error) {
	if entry.Status != ResumeResolved {
		return nil, false, nil
	}
	var probe struct {
		Action *string `json:"action"`
	}
	if json.Unmarshal(entry.Payload, &probe) != nil || probe.Action == nil {
		return nil, false, nil
	}
	var payload struct {
		Action     string          `json:"action"`
		EditedArgs json.RawMessage `json:"editedArgs"`
	}
	decoder := json.NewDecoder(bytes.NewReader(entry.Payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&payload); err != nil || payload.Action != "preview" ||
		!bytes.HasPrefix(bytes.TrimSpace(payload.EditedArgs), []byte("{")) {
		return nil, true, errors.New(`a preview asked again takes the payload {"action": "preview", "editedArgs": {…}}, the full arguments of the call`)
	}
	return payload.EditedArgs, true, nil
}

func (entry ResumeEntry) Approval() (approved bool, editedArgs json.RawMessage, err error) {
	switch entry.Status {
	case ResumeCancelled:
		if len(bytes.TrimSpace(entry.Payload)) > 0 && !bytes.Equal(bytes.TrimSpace(entry.Payload), []byte("null")) {
			var payload struct {
				EditedArgs json.RawMessage `json:"editedArgs"`
			}
			if json.Unmarshal(entry.Payload, &payload) == nil && payload.EditedArgs != nil {
				return false, nil, errors.New("edited arguments go with an approval only")
			}
		}
		return false, nil, nil
	case ResumeResolved:
		var payload struct {
			Approved   *bool           `json:"approved"`
			EditedArgs json.RawMessage `json:"editedArgs"`
		}
		decoder := json.NewDecoder(bytes.NewReader(entry.Payload))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&payload); err != nil || payload.Approved == nil {
			return false, nil, errors.New(`a resolved confirmation takes the payload {"approved": true}, {"approved": true, "editedArgs": {…}} or {"approved": false}`)
		}
		if payload.EditedArgs != nil && !*payload.Approved {
			return false, nil, errors.New("edited arguments go with an approval only")
		}
		if payload.EditedArgs != nil && !bytes.HasPrefix(bytes.TrimSpace(payload.EditedArgs), []byte("{")) {
			return false, nil, errors.New("editedArgs must be a JSON object, the full arguments of the call")
		}
		return *payload.Approved, payload.EditedArgs, nil
	}
	return false, nil, fmt.Errorf("resume status must be %q or %q", ResumeResolved, ResumeCancelled)
}

// Version reads the version of AG-UI the client declares, MAJOR.MINOR
// compared numerically: whether it is newer than the agent's, and an error
// for a declaration that is not a version, or of another major line.
func (input RunAgentInput) Version() (newer bool, err error) {
	if input.ProtocolVersion == "" {
		return false, nil
	}
	major, minor, ok := splitVersion(input.ProtocolVersion)
	if !ok {
		return false, ErrInvalidVersion
	}
	ownMajor, ownMinor, _ := splitVersion(ProtocolVersion)
	if major != ownMajor {
		return false, ErrUnsupportedVersion
	}
	return minor > ownMinor, nil
}

var (
	ErrInvalidVersion     = errors.New("protocolVersion must be a version MAJOR.MINOR, such as 1.0")
	ErrUnsupportedVersion = errors.New("the agent speaks the major version 1 of AG-UI only")
)

func splitVersion(version string) (int, int, bool) {
	major, minor, ok := strings.Cut(version, ".")
	if !ok {
		return 0, 0, false
	}
	a, errA := strconv.ParseUint(major, 10, 16)
	b, errB := strconv.ParseUint(minor, 10, 16)
	if errA != nil || errB != nil {
		return 0, 0, false
	}
	return int(a), int(b), true
}
