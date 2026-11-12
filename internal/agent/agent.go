package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/llm"
)

type MCPSession interface {
	// ServerInfo is the name and version the MCP server announced.
	ServerInfo() (string, string)
	ListTools(context.Context) ([]*mcp.Tool, error)
	CallTool(context.Context, string, map[string]any) (*mcp.CallToolResult, error)
	// ListResources and ReadResource give the reference texts of the
	// server, which the model reads with read_resource.
	ListResources(context.Context) ([]*mcp.Resource, error)
	ReadResource(context.Context, string) (*mcp.ReadResourceResult, error)
	Close() error
}

type MCPConnectFunc func(context.Context) (MCPSession, error)

type Config struct {
	// Backend names the backend the agent serves, such as om3: the origin
	// the agent gives the skills of its MCP, rather than the name the
	// server gives itself.
	Backend string
	// SystemPrompt instructs the model for the backend the agent serves.
	SystemPrompt  string
	MaxIterations int
	Timeout       time.Duration
	// SearchThreshold is the largest MCP catalog shown whole to the model;
	// beyond, it sees the core tools and searches the others with
	// search_tools. 0 means DefaultSearchThreshold, a negative value never
	// searches.
	SearchThreshold int
	// SearchScoreRatio is the share of the best score of a search under
	// which a result is not shown. 0 means DefaultSearchScoreRatio, a
	// negative value shows every result.
	SearchScoreRatio float64
	// Docs gives the documentation the model searches with search_docs;
	// nil offers no such tool.
	Docs DocsSource
	// Code gives the source code of the daemon the model searches with
	// search_code; nil offers no such tool.
	Code CodeSource
	// Now is the clock the events of a turn are stamped with, At; nil means
	// time.Now. The stream, the audit and the history carry its times.
	Now func() time.Time
}

type Agent struct {
	llm        llm.Client
	connectMCP MCPConnectFunc
	config     Config
	indexes    *indexCache
}

// TurnResult contains the provider-neutral messages produced by one completed
// user turn. It excludes the system prompt and the history supplied to
// RunTurn. Suspended is set instead when the turn waits for the user to
// confirm a tool call; ResumeTurn continues it.
type TurnResult struct {
	Messages     []llm.Message
	FinishReason llm.FinishReason
	Suspended    *SuspendedTurn
	// MCP describes the MCP server the turn used.
	MCP MCPServer
	// Exposure describes what the model was given.
	Exposure Exposure
}

// Exposure is what the model of a turn was given: a hash of the system
// prompt actually sent, whether the catalog was searched, and the tools
// sent, in the order they were first sent.
type Exposure struct {
	SystemPromptSHA256 string
	ToolSearch         bool
	// SearchScoreRatio is the ratio the searches of the turn used; 0 when
	// every result is shown or the catalog is not searched.
	SearchScoreRatio float64
	ShownTools       []string
	// Skills are the skills the turn loaded, with the digest of the text the
	// model was given.
	Skills []LoadedSkill
}

// MCPServer is the MCP server of a turn: the name and version it announced,
// and a hash of the tools it listed, which changes with any tool name,
// description, schema or annotation.
type MCPServer struct {
	Name        string
	Version     string
	ToolsSHA256 string
}

func New(llmClient llm.Client, connectMCP MCPConnectFunc, config Config) (*Agent, error) {
	if llmClient == nil {
		return nil, fmt.Errorf("agent LLM client is nil")
	}
	if connectMCP == nil {
		return nil, fmt.Errorf("agent MCP connector is nil")
	}
	if strings.TrimSpace(config.SystemPrompt) == "" {
		return nil, fmt.Errorf("agent system prompt is empty")
	}
	if config.MaxIterations <= 0 {
		return nil, fmt.Errorf("agent max iterations must be positive")
	}
	if config.Timeout <= 0 {
		return nil, fmt.Errorf("agent timeout must be positive")
	}
	if config.Now == nil {
		config.Now = time.Now
	}
	return &Agent{llm: llmClient, connectMCP: connectMCP, config: config, indexes: &indexCache{}}, nil
}

// RunTurn executes one user turn after a complete provider-neutral history.
// It returns only the new messages produced by the turn. A completed result may
// accompany an MCP cleanup error, allowing a future conversation service to
// distinguish completed model output from partial output. A tool call that
// requires the confirmation of the user suspends the turn instead of running.
func (a *Agent) RunTurn(ctx context.Context, history []llm.Message, prompt string, emit EmitFunc) (result TurnResult, err error) {
	if strings.TrimSpace(prompt) == "" {
		return result, fmt.Errorf("agent prompt is empty")
	}
	return a.run(ctx, history, &SuspendedTurn{Messages: []llm.Message{{Role: llm.RoleUser, Text: prompt}}}, "", emit)
}

// ResumeTurn continues a suspended turn with the decision of the user on its
// pending tool call: approved, the call runs as it was suspended; rejected,
// the model is told it did not run.
func (a *Agent) ResumeTurn(ctx context.Context, history []llm.Message, suspended SuspendedTurn, decision Decision, emit EmitFunc) (result TurnResult, err error) {
	// A turn waiting on the client resumes with its answers, without a
	// decision; one waiting on a confirmation, with the decision.
	if suspended.WaitsOnClient() != (decision == "") || (decision != "" && !decision.Valid()) {
		return result, fmt.Errorf("agent confirmation decision %q is invalid for the suspended turn", decision)
	}
	if err := suspended.validate(a.config.MaxIterations); err != nil {
		return result, err
	}
	state := suspended
	if state.Messages, err = prepareSuspendedMessages(suspended.Messages); err != nil {
		return result, fmt.Errorf("suspended turn: %w", err)
	}
	state.Calls = append([]llm.ToolCall(nil), suspended.Calls...)
	state.Results = append([]llm.ToolResult(nil), suspended.Results...)
	state.ClientCalls = append([]string(nil), suspended.ClientCalls...)
	state.ClientResults = append([]llm.ToolResult(nil), suspended.ClientResults...)
	return a.run(ctx, history, &state, decision, emit)
}

// run drives the turn held by state until the model answers, or until a tool
// call requires a confirmation. decision applies to the first call left to
// run, the one a suspended turn waits on.
func (a *Agent) run(ctx context.Context, history []llm.Message, state *SuspendedTurn, decision Decision, consume EmitFunc) (result TurnResult, err error) {
	if consume == nil {
		return result, fmt.Errorf("agent event consumer is nil")
	}
	// Each event is stamped once, as it is emitted: the stream, the audit
	// and the history all carry this time.
	emit := func(event Event) error {
		if event.At.IsZero() {
			event.At = a.config.Now()
		}
		return consume(event)
	}
	history, err = prepareHistory(history)
	if err != nil {
		return result, err
	}
	ctx, cancel := context.WithTimeout(ctx, a.config.Timeout)
	defer cancel()

	session, err := a.connectMCP(ctx)
	if err != nil {
		return result, fmt.Errorf("connect agent to MCP: %w", err)
	}
	defer func() {
		if closeErr := session.Close(); closeErr != nil {
			err = errors.Join(err, fmt.Errorf("close agent MCP session: %w", closeErr))
		}
	}()

	mcpTools, err := session.ListTools(ctx)
	if err != nil {
		return result, fmt.Errorf("list agent MCP tools: %w", err)
	}
	result.MCP.Name, result.MCP.Version = session.ServerInfo()
	if result.MCP.ToolsSHA256, err = toolCatalogHash(mcpTools); err != nil {
		return result, err
	}
	// The tools for the client only, such as the previews, are never
	// offered to the model; the confirmation of a call calls them.
	allTools := make(map[string]*mcp.Tool, len(mcpTools))
	for _, tool := range mcpTools {
		allTools[tool.Name] = tool
	}
	offeredTools := modelTools(mcpTools)
	// An unattended turn offers the read-only tools only, and none of the
	// client: nobody is there to confirm a call, or to answer one.
	unattended := Unattended(ctx)
	if unattended {
		if len(clientToolsFrom(ctx)) > 0 {
			return result, fmt.Errorf("an unattended turn takes no tools of the client")
		}
		offeredTools = readOnlyTools(offeredTools)
	}
	tools, _, err := convertTools(offeredTools)
	if err != nil {
		return result, err
	}
	toolsByName := make(map[string]*mcp.Tool, len(offeredTools))
	titles := make(map[string]string, len(offeredTools))
	for _, tool := range offeredTools {
		toolsByName[tool.Name] = tool
		titles[tool.Name] = toolTitle(tool)
	}
	selection, err := a.selectTools(offeredTools, tools, result.MCP.ToolsSHA256, history, state)
	if err != nil {
		return result, err
	}
	// The skills of the MCP are loaded with load_skill, and its resources
	// read with read_resource, each offered in every request of the turn
	// when the server has some. The files of the skills are load_skill's.
	skills, err := newSkillShelf(ctx, session, selection.catalog)
	if err != nil {
		return result, err
	}
	skills.carryTools(history, selection)
	shelf, err := newResourceShelf(ctx, session, selection.catalog, skills.files)
	if err != nil {
		return result, err
	}
	// The documentation is searched with search_docs, offered in every
	// request of the turn when the backend has some.
	if _, taken := selection.catalog[SearchDocsToolName]; taken && a.config.Docs != nil {
		return result, fmt.Errorf("MCP tool %q conflicts with the documentation search of the agent", SearchDocsToolName)
	}
	if _, taken := selection.catalog[SearchCodeToolName]; taken && a.config.Code != nil {
		return result, fmt.Errorf("MCP tool %q conflicts with the code search of the agent", SearchCodeToolName)
	}
	// The tools of the client are offered in every request of the turn,
	// never searched; one named as a tool of the agent is refused.
	offered, offeredNames, err := clientTools(clientToolsFrom(ctx), selection.known)
	if err != nil {
		return result, err
	}
	systemPrompt := a.config.SystemPrompt
	if selection.search {
		systemPrompt += "\n" + searchInstruction
	}
	if unattended {
		systemPrompt += "\n" + unattendedInstruction
	}
	promptSum := sha256.Sum256([]byte(systemPrompt))
	// finish completes a result with what the turn used and was given.
	finish := func(turn TurnResult) TurnResult {
		turn.MCP = result.MCP
		turn.Exposure = Exposure{
			SystemPromptSHA256: hex.EncodeToString(promptSum[:]), ToolSearch: selection.search,
			SearchScoreRatio: selection.scoreRatio, ShownTools: slices.Clone(selection.shown),
			Skills: slices.Clone(skills.loaded),
		}
		return turn
	}
	messages := make([]llm.Message, 0, len(history)+len(state.Messages)+3)
	messages = append(messages, llm.Message{Role: llm.RoleSystem, Text: systemPrompt + "\n" + confirmationState(state, decision)})
	messages = append(messages, history...)
	messages = append(messages, state.Messages...)
	llmContext := auth.WithoutAuthentication(ctx)

	// executeCalls runs the calls of the current iteration left to run, and
	// reports whether one waits for a confirmation.
	executeCalls := func() (bool, error) {
		for len(state.Results) < len(state.Calls) {
			call := state.Calls[len(state.Results)]
			// The decision of the user applies to the call they were shown,
			// the first left to run, and to no other. When its tool is no
			// longer offered, such as after a change of the catalog of the
			// MCP since the turn waited, the turn fails rather than answer the
			// call as unoffered: a later call of the answer, never shown,
			// would otherwise take the approval, or the arguments the user
			// wrote for another tool.
			callDecision := decision
			decision = ""
			if callDecision != "" {
				tool, offered := toolsByName[call.Name]
				if !offered {
					state.EditedArguments = nil
					return false, fmt.Errorf("the confirmed tool %q is no longer offered: the decision cannot apply", call.Name)
				}
				// The decision was taken on the definition the user saw:
				// a tool the server changed since, its annotations, schema
				// or preview, takes a new confirmation.
				if sum, err := toolCatalogHash([]*mcp.Tool{tool}); err != nil || sum != state.ToolSHA256 {
					state.EditedArguments = nil
					return false, fmt.Errorf("the confirmed tool %q changed since its confirmation: the decision cannot apply", call.Name)
				}
			}
			// A call of a tool of the client is answered by the client.
			if slices.Contains(state.ClientCalls, call.ID) {
				answer, ok := state.clientResult(call.ID)
				if !ok {
					return false, fmt.Errorf("the client did not answer the call %q of its tool %q", call.ID, call.Name)
				}
				state.Results = append(state.Results, answer)
				continue
			}
			if offeredNames[call.Name] {
				// The calls of tools of the client left in the answer go to
				// the client at once; the calls of the MCP after them run
				// once the client answered.
				for _, pending := range state.Calls[len(state.Results):] {
					if !offeredNames[pending.Name] {
						continue
					}
					if err := emitAgentEvent(emit, Event{Type: EventToolStarted, ToolName: pending.Name, ToolCallID: pending.ID, ToolArguments: pending.Arguments, Iteration: state.Iteration}); err != nil {
						return false, err
					}
					state.ClientCalls = append(state.ClientCalls, pending.ID)
				}
				return true, nil
			}
			if skills.known(call.Name) {
				if err := emitAgentEvent(emit, Event{Type: EventToolStarted, ToolName: call.Name, ToolCallID: call.ID, ToolArguments: call.Arguments, Iteration: state.Iteration}); err != nil {
					return false, err
				}
				loadResult := skills.load(ctx, session, a.config.Backend, call, selection)
				// The tools a skill showed stay shown when the turn resumes.
				if selection.search {
					state.ActiveTools = slices.Clone(selection.active)
				}
				if err := emitAgentEvent(emit, Event{Type: EventToolFinished, ToolName: call.Name, ToolError: loadResult.IsError, ToolCallID: call.ID, ToolResult: loadResult.Content, Iteration: state.Iteration}); err != nil {
					return false, err
				}
				state.Results = append(state.Results, loadResult)
				continue
			}
			if shelf.known(call.Name) {
				if err := emitAgentEvent(emit, Event{Type: EventToolStarted, ToolName: call.Name, ToolCallID: call.ID, ToolArguments: call.Arguments, Iteration: state.Iteration}); err != nil {
					return false, err
				}
				readResult := shelf.read(ctx, session, call)
				if err := emitAgentEvent(emit, Event{Type: EventToolFinished, ToolName: call.Name, ToolError: readResult.IsError, ToolCallID: call.ID, ToolResult: readResult.Content, Iteration: state.Iteration}); err != nil {
					return false, err
				}
				state.Results = append(state.Results, readResult)
				continue
			}
			if a.config.Docs != nil && call.Name == SearchDocsToolName {
				if err := emitAgentEvent(emit, Event{Type: EventToolStarted, ToolName: call.Name, ToolCallID: call.ID, ToolArguments: call.Arguments, Iteration: state.Iteration}); err != nil {
					return false, err
				}
				// The calls the source makes to give the book, such as the
				// read of the version of the cluster, are audited.
				audited := &auditedSession{MCPSession: session, emit: emit, iteration: state.Iteration, purpose: "documentation_version", listed: selection.catalog}
				// With the code searchable too, a search tells where to look
				// for what the book does not say.
				next := ""
				if a.config.Code != nil {
					next = docsThenCode
				}
				docsResult, err := searchDocs(ctx, a.config.Docs, audited, call, next)
				if err != nil {
					return false, fmt.Errorf("search the documentation: %w", err)
				}
				if err := emitAgentEvent(emit, Event{Type: EventToolFinished, ToolName: call.Name, ToolError: docsResult.IsError, ToolCallID: call.ID, ToolResult: docsResult.Content, Iteration: state.Iteration}); err != nil {
					return false, err
				}
				state.Results = append(state.Results, docsResult)
				continue
			}
			if a.config.Code != nil && call.Name == SearchCodeToolName {
				if err := emitAgentEvent(emit, Event{Type: EventToolStarted, ToolName: call.Name, ToolCallID: call.ID, ToolArguments: call.Arguments, Iteration: state.Iteration}); err != nil {
					return false, err
				}
				codeResult := llm.ToolResult{CallID: call.ID, Content: textToolResult(codeSearchesSpent), IsError: true}
				if state.CodeSearches < maxCodeSearchesPerAsk {
					state.CodeSearches++
					// The read of the version of the cluster, the code
					// searched by default, is audited.
					audited := &auditedSession{MCPSession: session, emit: emit, iteration: state.Iteration, purpose: "code_version", listed: selection.catalog}
					var err error
					if codeResult, err = searchCode(ctx, a.config.Code, audited, call); err != nil {
						return false, fmt.Errorf("search the code: %w", err)
					}
				}
				if err := emitAgentEvent(emit, Event{Type: EventToolFinished, ToolName: call.Name, ToolError: codeResult.IsError, ToolCallID: call.ID, ToolResult: codeResult.Content, Iteration: state.Iteration}); err != nil {
					return false, err
				}
				state.Results = append(state.Results, codeResult)
				continue
			}
			if selection.search && call.Name == SearchToolName {
				if err := emitAgentEvent(emit, Event{Type: EventToolStarted, ToolName: call.Name, ToolCallID: call.ID, ToolArguments: call.Arguments, Iteration: state.Iteration}); err != nil {
					return false, err
				}
				searchResult := selection.run(call, titles)
				state.ActiveTools, state.Searches = slices.Clone(selection.active), selection.searches
				if err := emitAgentEvent(emit, Event{Type: EventToolFinished, ToolName: call.Name, ToolError: searchResult.IsError, ToolCallID: call.ID, ToolResult: searchResult.Content, Iteration: state.Iteration}); err != nil {
					return false, err
				}
				state.Results = append(state.Results, searchResult)
				continue
			}
			tool, ok := toolsByName[call.Name]
			if !ok {
				if err := emitAgentEvent(emit, Event{Type: EventToolStarted, ToolName: call.Name, ToolCallID: call.ID, ToolArguments: call.Arguments, Iteration: state.Iteration}); err != nil {
					return false, err
				}
				unoffered := llm.ToolResult{CallID: call.ID, Content: textToolResult(unofferedToolResult), IsError: true}
				if err := emitAgentEvent(emit, Event{Type: EventToolFinished, ToolName: call.Name, ToolError: true, NotRun: NotRunUnoffered, ToolCallID: call.ID, ToolResult: unoffered.Content, Iteration: state.Iteration}); err != nil {
					return false, err
				}
				state.Results = append(state.Results, unoffered)
				continue
			}
			arguments, err := decodeToolArguments(call)
			if err != nil {
				return false, err
			}
			if callDecision == DecisionReject {
				state.Results = append(state.Results, llm.ToolResult{CallID: call.ID, Content: rejectedToolResult, IsError: true})
				continue
			}
			if unattended && requiresConfirmation(tool) {
				refused := llm.ToolResult{CallID: call.ID, Content: textToolResult(unattendedToolResult), IsError: true}
				if err := emitAgentEvent(emit, Event{Type: EventToolStarted, ToolName: call.Name, ToolCallID: call.ID, ToolArguments: call.Arguments, Iteration: state.Iteration}); err != nil {
					return false, err
				}
				if err := emitAgentEvent(emit, Event{Type: EventToolFinished, ToolName: call.Name, ToolError: true, NotRun: NotRunNeedsConfirmation, ToolCallID: call.ID, ToolResult: refused.Content, Iteration: state.Iteration}); err != nil {
					return false, err
				}
				state.Results = append(state.Results, refused)
				continue
			}
			if callDecision != DecisionApprove && requiresConfirmation(tool) {
				state.ToolTitle, state.ArgumentTypes, state.InputSchema = toolTitle(tool), argumentTypes(tool), editableSchema(tool)
				if state.ToolSHA256, err = toolCatalogHash([]*mcp.Tool{tool}); err != nil {
					return false, err
				}
				state.Preview, state.PreviewError = preview(ctx, session, allTools, tool, call.Arguments)
				state.PreviewArguments = nil
				return true, nil
			}
			// The user approved the call with arguments of their own, in place
			// of those the model proposed.
			edited := json.RawMessage(nil)
			if callDecision == DecisionApprove && len(state.EditedArguments) > 0 {
				edited, state.EditedArguments = state.EditedArguments, nil
				if arguments, err = decodeToolArguments(llm.ToolCall{Name: call.Name, Arguments: edited}); err != nil {
					return false, err
				}
			}
			ranWith := call.Arguments
			if edited != nil {
				ranWith = edited
			}
			if err := emitAgentEvent(emit, Event{Type: EventToolStarted, ToolName: call.Name, ToolCallID: call.ID, ToolArguments: ranWith, Iteration: state.Iteration}); err != nil {
				return false, err
			}
			toolResult, err := session.CallTool(ctx, call.Name, arguments)
			if err != nil {
				return false, fmt.Errorf("call agent MCP tool %q: %w", call.Name, err)
			}
			content, err := encodeToolResult(toolResult)
			if err != nil {
				return false, fmt.Errorf("process agent MCP tool %q result: %w", call.Name, err)
			}
			if edited != nil {
				content = editedToolResult(content, edited)
			}
			if err := emitAgentEvent(emit, Event{Type: EventToolFinished, ToolName: call.Name, ToolError: toolResult.IsError, ToolCallID: call.ID, ToolResult: content, Iteration: state.Iteration}); err != nil {
				return false, err
			}
			state.Results = append(state.Results, llm.ToolResult{CallID: call.ID, Content: content, IsError: toolResult.IsError})
		}
		return false, nil
	}

	// A preview asked again runs nothing: the pending call is told with the
	// arguments the user edited, and the turn waits again.
	if decision == DecisionPreview {
		call, err := state.PendingCall()
		if err != nil {
			return result, err
		}
		tool, ok := toolsByName[call.Name]
		if !ok || len(state.EditedArguments) == 0 {
			return result, fmt.Errorf("the pending call %q has no preview to compute again", call.Name)
		}
		state.Preview, state.PreviewError = preview(ctx, session, allTools, tool, state.EditedArguments)
		state.PreviewArguments, state.EditedArguments = state.EditedArguments, nil
		snapshot := *state
		return finish(TurnResult{Suspended: &snapshot}), nil
	}
	for {
		if len(state.Calls) > 0 {
			suspended, err := executeCalls()
			if err != nil {
				return result, err
			}
			if suspended {
				snapshot := *state
				return finish(TurnResult{Suspended: &snapshot}), nil
			}
			assistantMessage := llm.Message{Role: llm.RoleAssistant, Text: state.Text, ToolCalls: state.Calls}
			toolMessage := llm.Message{Role: llm.RoleTool, ToolResults: state.Results}
			messages = append(messages, assistantMessage, toolMessage)
			state.Messages = append(state.Messages, assistantMessage, toolMessage)
			state.Calls, state.Results, state.Text, state.ToolTitle, state.ArgumentTypes = nil, nil, "", "", nil
			state.InputSchema, state.EditedArguments, state.ToolSHA256 = nil, nil, ""
			state.ClientCalls, state.ClientResults = nil, nil
		}
		if state.Iteration >= a.config.MaxIterations {
			return result, fmt.Errorf("agent reached maximum of %d iterations", a.config.MaxIterations)
		}
		state.Iteration++
		iteration := state.Iteration
		var (
			calls        []llm.ToolCall
			turnText     strings.Builder
			finishReason llm.FinishReason
			completion   bool
		)
		// The last iteration asks for the answer: a turn that would need
		// more calls tells what it has and what is left, rather than fail.
		requestTools := selection.tools()
		if skills.tool != nil {
			requestTools = append(requestTools, *skills.tool)
			if !slices.Contains(selection.shown, LoadSkillToolName) {
				selection.shown = append(selection.shown, LoadSkillToolName)
			}
		}
		if shelf.tool != nil {
			requestTools = append(requestTools, *shelf.tool)
			if !slices.Contains(selection.shown, ReadResourceToolName) {
				selection.shown = append(selection.shown, ReadResourceToolName)
			}
		}
		if a.config.Docs != nil {
			requestTools = append(requestTools, docsTool(a.config.Docs))
			if !slices.Contains(selection.shown, SearchDocsToolName) {
				selection.shown = append(selection.shown, SearchDocsToolName)
			}
		}
		if a.config.Code != nil {
			requestTools = append(requestTools, codeTool(a.config.Code))
			if !slices.Contains(selection.shown, SearchCodeToolName) {
				selection.shown = append(selection.shown, SearchCodeToolName)
			}
		}
		requestTools = append(requestTools, offered...)
		for _, tool := range offered {
			if !slices.Contains(selection.shown, tool.Name) {
				selection.shown = append(selection.shown, tool.Name)
			}
		}
		// The question that ran every tool it may asks for the answer too.
		lastCall := iteration == a.config.MaxIterations || state.TotalToolCalls >= maxToolCallsPerAsk
		request := llm.Request{Messages: messages, Tools: requestTools, NoToolCall: lastCall}
		// The model call is timed on the clock of the agent: until its first
		// event, the wait of the provider, and until its usage, the whole call.
		modelStart := time.Now()
		var firstEvent time.Duration
		// The calls past the bound of an answer are counted, not kept: the
		// answer is refused as a whole, with its size.
		extraCalls := 0
		if err := a.llm.Stream(llmContext, request, func(event llm.Event) error {
			if err := event.Validate(); err != nil {
				return fmt.Errorf("validate LLM event: %w", err)
			}
			if firstEvent == 0 {
				firstEvent = max(time.Since(modelStart), time.Nanosecond)
			}
			switch event.Type {
			case llm.EventTextDelta:
				turnText.WriteString(event.TextDelta)
				return emitAgentEvent(emit, Event{Type: EventTextDelta, TextDelta: event.TextDelta, Iteration: iteration})
			case llm.EventToolCall:
				if len(calls) >= maxToolCallsPerTurn {
					extraCalls++
					return nil
				}
				call := *event.ToolCall
				call.Arguments = append(json.RawMessage(nil), event.ToolCall.Arguments...)
				calls = append(calls, call)
			case llm.EventUsage:
				usage := *event.Usage
				return emitAgentEvent(emit, Event{Type: EventUsage, Usage: &usage, Iteration: iteration, ModelDuration: time.Since(modelStart), ModelFirstEvent: firstEvent})
			case llm.EventCompleted:
				if completion {
					return fmt.Errorf("LLM emitted multiple completion events")
				}
				completion = true
				finishReason = event.FinishReason
			}
			return nil
		}); err != nil {
			return result, fmt.Errorf("run LLM iteration %d: %w", iteration, err)
		}
		if !completion {
			return result, fmt.Errorf("LLM iteration %d ended without completion", iteration)
		}
		if len(calls) == 0 {
			if finishReason == llm.FinishReasonToolCalls {
				return result, fmt.Errorf("LLM iteration %d completed for tool calls without a tool call", iteration)
			}
			if turnText.Len() == 0 {
				return result, fmt.Errorf("LLM iteration %d completed without text or tool calls", iteration)
			}
			if err := emitAgentEvent(emit, Event{Type: EventCompleted, FinishReason: finishReason, Iteration: iteration}); err != nil {
				return result, err
			}
			state.Messages = append(state.Messages, llm.Message{Role: llm.RoleAssistant, Text: turnText.String()})
			return finish(TurnResult{Messages: state.Messages, FinishReason: finishReason}), nil
		}
		if finishReason != llm.FinishReasonToolCalls {
			return result, fmt.Errorf("LLM iteration %d emitted tool calls with finish reason %q", iteration, finishReason)
		}
		if iteration == a.config.MaxIterations {
			return result, fmt.Errorf("agent reached maximum of %d iterations before a final answer", a.config.MaxIterations)
		}
		for _, call := range calls {
			// A tool the turn does not offer is answered with a tool error
			// when its call runs. The bound of the arguments a turn stores, for every call, the
			// calls of the agent included.
			if len(call.Arguments) > maxToolArguments {
				return result, fmt.Errorf("tool %q arguments exceed %d bytes", call.Name, maxToolArguments)
			}
		}
		// An answer past a bound runs none of its calls: each is answered
		// with the bound, for the model to ask less or to answer.
		if bound, refusal := callLimitRefusal(len(calls)+extraCalls, state.TotalToolCalls); refusal != nil {
			refused := &RefusedCalls{Bound: bound, Asked: len(calls) + extraCalls, Ran: state.TotalToolCalls}
			if err := emitAgentEvent(emit, Event{Type: EventToolCallsRefused, Refused: refused, Iteration: iteration}); err != nil {
				return result, err
			}
			results := make([]llm.ToolResult, 0, len(calls))
			for _, call := range calls {
				results = append(results, llm.ToolResult{CallID: call.ID, Content: refusal, IsError: true})
			}
			state.Calls, state.Results, state.Text = calls, results, turnText.String()
			continue
		}
		state.TotalToolCalls += len(calls)
		state.Calls, state.Results, state.Text = calls, nil, turnText.String()
	}
}

func emitAgentEvent(emit EmitFunc, event Event) error {
	if err := event.Validate(); err != nil {
		return fmt.Errorf("validate agent event: %w", err)
	}
	if err := emit(event); err != nil {
		return fmt.Errorf("consume agent event: %w", err)
	}
	return nil
}

// callLimitRefusal is the bound a model answer asking for `asked` tools
// passes, the question having run `ran`, and the answer to each of its calls:
// "answer", more than maxToolCallsPerTurn at once, or "question", more than
// maxToolCallsPerAsk in all. Nil within the bounds.
func callLimitRefusal(asked, ran int) (string, json.RawMessage) {
	switch {
	case asked > maxToolCallsPerTurn:
		return "answer", textToolResult(fmt.Sprintf("Not run: this answer asked for %d tools at once, and the agent runs at most %d per answer, so none of them ran. "+
			"Ask for %d at most, or use a tool that reads many objects in one call.", asked, maxToolCallsPerTurn, maxToolCallsPerTurn))
	case ran+asked > maxToolCallsPerAsk:
		return "question", textToolResult(fmt.Sprintf("Not run: this question already ran %d tools, and the agent runs at most %d per question, so %d more at most can run and none of these %d ran. "+
			"Ask for fewer, or answer with what you have.", ran, maxToolCallsPerAsk, maxToolCallsPerAsk-ran, asked))
	}
	return "", nil
}
