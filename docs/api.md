# Client API

`opensvc-ai-agentd` serves its clients over HTTPS. The OpenSVC webapp is its
client; see the [browser client guide](webapp.md) for the headers, message
history, action confirmation and CORS, and the
[integration guide](integration.md) for an interface speaking AG-UI.

## Backends

The agent serves each backend under its own routes, `/v1/{backend}`. A backend
is one product: its MCP, the identity check of its users and the instructions
given to the model. The route selects the backend; nothing in the request can
select another one.

| Backend | Routes | MCP | Configuration |
|---|---|---|---|
| `om3` | `/v1/om3/...` | OpenSVC daemon MCP, on its local Unix socket | `OPENSVC_AI_OM3_MCP_SOCKET`, required |
| `oc3` | `/v1/oc3/...` | OpenSVC collector MCP, on its local Unix socket | `OPENSVC_AI_OC3_MCP_SOCKET`, optional: the backend is served only when set |

A conversation belongs to the backend it was created on. It is not listed,
read or continued through the routes of another backend.

## Routes

The routes of a backend, here `om3`; `oc3` has the same routes under
`/v1/oc3`:

| Method and path | Purpose |
|---|---|
| `POST /v1/om3/conversations` | Create a conversation |
| `GET /v1/om3/conversations` | List the conversations of the caller |
| `GET /v1/om3/conversations/{id}` | Read a conversation and its pending confirmation |
| `GET /v1/om3/conversations/{id}/messages` | Read its stored messages |
| `PATCH /v1/om3/conversations/{id}` | Rename it |
| `DELETE /v1/om3/conversations/{id}` | Delete it |
| `POST /v1/om3/conversations/{id}/runs` | Send a prompt, or answer a pending action, streamed in AG-UI; see [Runs](#runs) |
| `GET /v1/om3/capabilities` | What the runs route can do, as an AG-UI `AgentCapabilities` |
| `GET /v1/om3/tools` | The tools the assistant may call for the caller; see [Tools](#tools) |
| `PUT /v1/om3/conversations/{id}/turns/{turn_id}/feedback` | Give or replace the feedback on a turn |
| `DELETE /v1/om3/conversations/{id}/turns/{turn_id}/feedback` | Withdraw it |

`GET /health` is shared by all backends and needs no authentication.

Every response carries `X-Request-ID`, the ID the agent gives the request.
Every audit line of the agent carries it as `request_id`, and the agent
forwards it to the MCP of the backend, whose audit lines carry it too: one ID
follows a request from the agent to its tool calls.

Each model call of a run has its audit line `llm_usage`: its `iteration`, its
tokens, and its timing on the clock of the agent, `duration_ms` from the
request to the usage the provider reports, and `first_event_ms` to the first
event of its stream, the wait of the provider before it answers. A slow run
tells there whether the provider or the tools took the time.

`GET /openapi.json` returns the OpenAPI 3.1 description of these routes, their
bodies, errors and stream events, without authentication. The source is
`internal/api/openapi.json`, embedded in the binary; a test keeps its routes
equal to the routes the agent registers.

## Authentication

Each backend declares the headers its routes read. Every `om3` route requires:

```http
Authorization: Bearer <OpenID Connect access token>
X-OpenSVC-Cluster-ID: <ID of the cluster it is intended for>
```

and may add `X-OpenSVC-Node: <node name>`, an optional routing hint read by
om3 only.

The client obtains the token by signing in with the OpenID provider the
daemons of the cluster trust; the om3 MCP accepts no other token. Before
each operation, the agent verifies the token through the om3 MCP
`GET /mcp/auth/whoami`, served only on its local socket: MCP verifies the
token signature, issuer and audience it is configured with, selects the
cluster from the header and relays the token to that daemon's
`GET /api/auth/whoami`, which confirms the identity. The cluster header only
selects the target: a token intended for another cluster is refused by the
selected daemon.

The agent treats the token as opaque. It uses the returned identity and
expiry for conversation ownership and request deadlines, and delegates the
same token and cluster ID to MCP for the tools. The daemon enforces grants.
The token is never stored, logged or given to the model.

The answer of the MCP is reused, in the memory of the agent only, by the next
operations of the same token (and, for om3, the same cluster and node) for at
most one minute and never past its expiry. A refused token is refused again
for five seconds without asking the MCP; an unavailable verification is never
reused. The cache holds identities, never tokens: its keys are hashes. A
change of the account or of the grants on the target therefore applies within
a minute.

Every `oc3` route requires the OpenID Connect access token of the user, and
no other header:

```http
Authorization: Bearer <OpenID Connect access token>
```

Before each operation, the agent verifies the token through the oc3 MCP
`GET /mcp/auth/whoami`, served only on its local socket: MCP verifies the
token signature, issuer and audience it is configured with, relays the token
to the collector's `GET /api/auth/whoami`, and answers the issuer, subject
and expiry the collector confirmed. The agent forwards no other header to the
oc3 MCP, and delegates the same token to it for the tools. The collector
enforces the visibility of the user's account. The
answer is reused as for om3.

## Conversations

A conversation stores the successful turns of the caller, as context for the
next ones. A turn runs on a prompt, with a [run](#runs) streamed in AG-UI.
Only one turn runs at a time in a conversation.

- **Title.** After the first successful turn, the agent derives a title from
  its prompt: whitespace is normalized and titles longer than 80 characters
  are truncated, without an additional model request. A failed first turn
  sets no title. A title set with `PATCH` is not replaced by later prompts.
- **Failures.** Canceled, failed, timed-out and interrupted turns are not
  added to the model context and their partial output is not stored.
- **Tool calls.** A turn runs at most 8 tool calls per model answer and 16 in
  all. A model answer asking for more runs none of its calls: each is answered
  with a tool error naming the bound, and the model asks for fewer or answers
  with what it has. Once 16 ran, the next model request takes no call. The
  client is shown none of the refused calls; the audit records the answer as
  `tool_calls_refused`, with the `bound` it passed, `answer` or `question`,
  `calls_asked` and `calls_ran`.
- **Last iteration.** The last model request of a turn, the
  `OPENSVC_AI_AGENT_MAX_ITERATIONS`th, declares the tools but takes no call of
  them (`tool_choice` none): the model answers with what the turn read,
  rather than the turn failing. On dev5, a form submitted
  then its request approved took 7 iterations of 8, and failed for want of
  the eighth before this.
- **Idle stream.** Each write of the stream must end within 15 s, or the
  client is cut off; between writes, a tool or a model request runs as long
  as the turn may, such as an MCP tool waiting for an action to end. While
  nothing else is written, the stream sends an SSE comment, `: keepalive`,
  every 10 s, which clients ignore and which keeps idle proxies from closing
  it.
- **History.** `GET …/messages` gives the messages of the completed turns,
  the most recent last, by pages of at most 1 MiB. The last assistant message
  of a turn carries its `steps`, the calls of its tools in their order: `id`,
  `tool`, `arguments` (`null` with `arguments_omitted` past 16 KiB, or past
  192 KiB for the turn), `status` (`completed`, `failed` for an error of the
  tool, `rejected` by the user, `interrupted` without result),
  `result_excerpt`, the first 4 KiB of the result, cut on a character, with
  `result_truncated`, `refusal`, the text of an error of the tool, at most
  16 KiB, `edited_arguments` when the user edited the call they approved, and
  `started_at` and `ended_at`; `steps_omitted` past 64 steps, or all of them
  when the message holds in no page with its steps. It carries the
  `runs` of the turn too, `started_at` and `ended_at`, the waits for a
  decision left out, and its `usage`, `total_tokens`. Times are milliseconds
  of the clock of the agent, the timestamps the stream carried; a turn
  stored before they were has none, its steps in their order. A turn waiting
  on a confirmation is not in the history: its steps come back with the
  snapshot of the run that resumes it.
- **Busy.** A prompt while a turn runs, or while an action waits for a
  decision, returns 409 `conversation_busy`.
- **Expiry.** A conversation expires after its configured lifetime, counted
  from its creation and then from its last completed turn; an expired
  conversation returns 410.
- **Quotas.** A user, their OpenID issuer and subject, holds at most
  `OPENSVC_AI_CONVERSATION_MAX_PER_USER` conversations, 128 by default, on
  every backend and cluster together; beyond, a creation returns 409
  `conversation_quota`, until the user deletes one or one expires. The other
  users are not held back. The instance holds at most
  `OPENSVC_AI_CONVERSATION_MAX` conversations, 4096 by default, and
  `OPENSVC_AI_CONVERSATION_MAX_DATABASE_MIB` of stored messages, 256 MiB by
  default; beyond, 409 `conversation_limit`. The expired conversations are
  deleted before each creation counts.

## Runs

`POST /v1/{backend}/conversations/{id}/runs` runs the agent on a
conversation for a client of the [AG-UI protocol](https://docs.ag-ui.com),
such as the OpenSVC webapp or the `HttpAgent` of `@ag-ui/client`: a new
prompt, the answer to the confirmation the conversation waits on, or the
answers to the calls of the tools of the client. The client creates, lists,
renames, deletes and rates the conversation with the routes above.

The request is an AG-UI `RunAgentInput`, at most 4 MiB, with the
authentication headers of the backend:

- `threadId` is the ID of the conversation of the route, or 400
  `thread_mismatch`; `runId`, which the client gives, at most 128 printable
  bytes without spaces, or 400 `invalid_run_id`.
- `protocolVersion`, the version of AG-UI the client speaks, `MAJOR.MINOR`
  compared numerically, or 400 `invalid_protocol_version`: absent, a client
  from before the protocol carried one, is served; a newer minor of the line
  1 is served in 1.0 with a `CUSTOM` `opensvc.warning` after `RUN_STARTED`,
  as the versioning of AG-UI has a producer warn; another major line is
  refused, 400 `unsupported_protocol_version`. `RUN_STARTED` declares the
  version of the agent, never that of the client, and the audit of the run
  records the version the client declared, `protocol_version`.
- The prompt is the last message, which must be a message of the user, a
  text or text parts joined by line breaks, at most 32 KiB, or 400
  `invalid_prompt`. The other content parts, such as images, are skipped, as
  the protocol has a producer do rather than fail the run, and named in a
  `CUSTOM` `opensvc.warning` event after `RUN_STARTED`,
  `{message, skippedContentParts}`; a message without a text part is
  refused. The agent keeps its own history: the other messages are
  not read, so that a client cannot put words in the mouth of the assistant
  or of a tool, but the answers to the calls of its tools, below. Nor are
  `state`, `context` or `forwardedProps`.
- `tools` are the tools of the client, which the agent offers to the model;
  see [Client tools](#client-tools).
- `resume`, at most one entry, answers the confirmation the conversation
  waits on: `interruptId` is the confirmation ID, or 409
  `confirmation_not_pending`; `status` `resolved` with `payload`
  `{"approved": true}` approves the call, `{"approved": false}` rejects it,
  and `cancelled` rejects it. With an approval, `editedArgs` are the
  arguments the user edited, a full replacement of those of the call; see
  [Edited arguments](#edited-arguments). `{"action": "preview",
  "editedArgs": {…}}` decides nothing: it asks the preview of the call again
  with the edited arguments; see [Previews](#previews). Any other member of
  the payload, or `editedArgs` with a rejection, returns 400
  `invalid_decision`. The messages of a resume are not read.

Errors before the stream are JSON errors, `{"error": {"code", "message"}}`,
the codes the OpenAPI description lists. The stream is `text/event-stream`,
one `data: <event JSON>` per event, without event names, and the
`: keepalive` comments of an idle stream. Each event carries its `timestamp`
in milliseconds:

| Agent | AG-UI events |
|---|---|
| start | `RUN_STARTED` with `threadId`, `runId` and `protocolVersion` `1.0` |
| a client out of phase | `MESSAGES_SNAPSHOT` after `RUN_STARTED`; see below |
| text of one model answer | `TEXT_MESSAGE_START` (`messageId` `<turn ID>:<2 × iteration>`, the ID the turn stores the answer with, role `assistant`), `TEXT_MESSAGE_CONTENT`, `TEXT_MESSAGE_END` before a tool call or the end |
| tool call, of the MCP or of the agent | `TOOL_CALL_START` (`toolCallId`, `toolCallName`, `parentMessageId` the message of its answer), `TOOL_CALL_ARGS` with the JSON arguments whole, `TOOL_CALL_END` |
| tool result | `TOOL_CALL_RESULT` with the JSON result the model read, `{"content", "structuredContent", "isError"}`, `isError` `true` for a refusal of the MCP tool and `false` otherwise, or a note of its size beyond 64 KiB |
| completion | `RUN_FINISHED`, `outcome` `{"type":"success"}`, `result` `{"turnId": …}` for the feedback |
| confirmation | the call as a tool call, then `RUN_FINISHED`, `outcome` `interrupt` |
| failure | `RUN_ERROR` with a `code`, `request_timeout`, `tool_name_conflict` or `agent_failed`, and a `message` for the developer |

`RUN_FINISHED` and `RUN_ERROR` carry `usage`, one `TokenUsage`
`{inputTokens, outputTokens, totalTokens}` summing the model calls of the
run, `totalTokens` computed as input plus output; a resumed run reports its
own calls only, as the protocol accounts them, and a run without a model
call none.

The interrupt of a confirmation is
`{"id": <confirmation ID>, "reason": "tool_call", "toolCallId", "message", "responseSchema", "expiresAt", "metadata"}`,
`tool_call` being the core reason of an interrupt bound to a tool call:
`responseSchema` takes `{"approved": boolean}`, and `editedArgs`, the input
schema of the tool, when the user may edit the call; `expiresAt` is the
expiry of the confirmation, and `metadata` gives `turnId`, `tool` (`name`,
`title`, `destructive`), `editable` and `argumentTypes`, the media types of
the text arguments, and `preview`, `previewError` and `previewArguments`, see
[Previews](#previews), as the conversation read gives them in its
`pending_confirmation` after a reload. A resumed run gives the result of the
confirmed call against its `toolCallId`, without its start again; a
rejected one, a result telling the call did not run.

### Edited arguments

The user may edit the arguments of a call before approving it, when the
tool declares an input schema the agent can check them against: an object
schema of at most 64 KiB that resolves without loading another schema. The
interrupt then gives the schema as `editedArgs` in its `responseSchema`,
the signal AG-UI gives a client that it may show an edit form, and
`metadata` `editable` `true`; a call of another tool is confirmable as is,
`editable` `false`.

- **Validation.** `editedArgs` are one JSON object of at most 256 KiB, a
  full replacement of the arguments, not a merge, checked against the input
  schema before the confirmation is used: an invalid edit returns 400
  `invalid_edited_arguments` with the violation, its path and constraint,
  in `message`, and the confirmation still waits for the user to correct
  the form. An edit of a call that takes none returns the same code.
- **The call shown.** The decision binds to the call the user saw, by the
  hash of the arguments the model proposed; the edit replaces them for the
  run only.
- **The model.** Its message keeps the call it proposed: the result of the
  tool tells it, ahead of what the call returned, `the user edited the
  arguments before approving; the call ran with: {…}`. The history and the
  copy a feedback keeps hold the proposal, the edit and the result.
- **The client.** The resumed run gives the result against the original
  `toolCallId`, whose arguments the stream gave as proposed: the client,
  which sent the edit, shows the arguments the call ran with.
- **Audit.** `confirmation_decided` records `arguments_edited`, the hash
  and the size of the edited arguments, never their content.

### Previews

A tool the user confirms may name, in its `_meta`, the tool of the MCP that
tells what its call would do: `opensvc/preview_tool`, called with the
arguments of the call, or with `{"tool": <name>, "arguments": {…}}` when
`opensvc/preview_wraps_arguments` is `true`. Before it asks for the
confirmation, the agent calls it as the user, within 5 seconds, and gives
the structured content of its result in the `metadata` of the interrupt as
`preview`, at most 256 KiB: a summary, the keys changed with their current
and computed values, the sections deleted, a content written, as the MCP
documents them. A preview that fails blocks nothing: `previewError` gives
why, at most 1 KiB, and the client shows the arguments. A tool naming no
preview has neither.

- **A preview asked again.** After an edit, the resume payload
  `{"action": "preview", "editedArgs": {…}}`, the full arguments checked as
  for an approval, runs nothing: the run ends on a new interrupt of the
  same call, a new `id`, without its tool call again, whose `preview` is of
  the edited arguments, which `previewArguments` gives back. The call the
  confirmation binds to stays the one the model proposed; approve with the
  edit as before. `confirmation_decided` records the decision `preview`.
- **Hidden tools.** A tool whose `_meta` holds `opensvc/hidden: true`, and a
  tool another names as its preview, are never offered to the model nor
  returned by `search_tools`.

### Messages snapshot

The messages of the stream carry the IDs the conversation stores them
with, those of `GET …/messages`: `<turn ID>:<sequence>`, the message of the
user being message 1 of its turn, then each iteration adding the message of
the assistant, `2 × iteration`, and the message of its tools, `2 × iteration
+ 1`; a tool result is the message `<toolCallId>-result`. The message of the
user goes by the ID the client gave it, at most 128 printable bytes without
spaces, or 400 `invalid_message_id`, when the client is in phase and no other
message of the conversation holds it: the conversation stores it with the
turn, and serves it in its history, its display messages and its
snapshots.

A client that holds the last stored message of the conversation, by its ID,
and, for a resumed turn, the answer of the model it waits in, is in phase.
One that does not, such as another tab, or one sending its prompt alone on
a conversation with turns, receives a `MESSAGES_SNAPSHOT` right after
`RUN_STARTED`: the thread as the model reads it, the history compacted as
for the model, with the tool calls and their results, the turns failed or
canceled left out, then the messages of a resumed turn so far, the
iteration it waits in included, and the prompt of the run. `@ag-ui/client`
keeps the messages it knows in their place and adds the others after them:
the prompt it holds would stay before the turns it lacked. Out of phase,
the prompt therefore goes, and is stored, under `<turn ID>:1`, which the
client does not hold: the client drops its copy and adds the snapshot's
after the turns it lacked. Beyond 1 MiB, the oldest turns are left out, and `metadata`
`{"opensvc.truncated": {"omittedTurns": n}}` counts them with the turns
the bound of the history leaves out.

### Client tools

The tools of the client, the frontend tools of AG-UI, are actions of its
interface, such as opening a page or highlighting a row: the agent offers
them to the model, and the client runs them, under its own consent rules.
The agent never confirms them, and they act in the interface of the user,
not on the clusters or the collector.

- **Declaration.** At most 16 tools, each a name of 1 to 64 letters,
  digits, underscores or hyphens, the names the model providers take, a
  description of at most 4 KiB and parameters, a JSON schema object, of at
  most 64 KiB, 256 KiB in all, or 400 `invalid_tools`. A name of a tool of
  the agent, `search_tools`, `read_resource`, `load_skill` or `search_docs`, is refused there; a name of
  a tool of the MCP ends the run with `RUN_ERROR` `tool_name_conflict`, the
  turn reading the catalog of the MCP: a tool of the client must not take
  the place of one, such as a destructive tool the agent confirms. They are
  offered in every request of the turn, after `read_resource`, never
  searched, and counted in the tools shown of its runs.
- **Calls.** The calls of one answer of the model run in their order. At
  the first call of a tool of the client, the calls of tools of the client
  left in the answer are emitted, `TOOL_CALL_START`, `TOOL_CALL_ARGS` and
  `TOOL_CALL_END`, and the run ends with `RUN_FINISHED`, `outcome`
  `{"type": "success", "pendingToolCallIds": […]}`, in the order of the
  calls; the calls of the MCP before them ran, those after them run once the
  client answered. The agent never answers a call of a tool of the client:
  no `TOOL_CALL_RESULT` for it, in this run or the next. The run gives no
  `turnId`: the turn is not complete.
- **Wait.** The turn is stored as suspended, as for a confirmation, with
  the same lifetime, 10 minutes, after which it fails: a resume answers 409
  `confirmation_not_pending` meanwhile, and the conversation read gives no
  `pending_confirmation` for it.
- **Answers.** The next run answers the calls with the tool messages its
  messages end with, `{role: "tool", toolCallId, content, error?}`: the
  text of `content`, a string or text parts, the other parts skipped and
  named as for a prompt, 256 KiB in all, and an answer with `error` is a
  tool error for the model. Of these tool messages, only those answering a
  call waiting are read: the client keeps the results the agent gave in the
  stream among its messages. They must answer each call waiting once, or
  400 `invalid_tool_results`; on a conversation waiting on no call, 409
  `client_tools_not_pending`; on a wait past its lifetime, 410
  `client_tools_expired`, the turn having failed. The turn resumes, the model reading the
  answers as results of the tools, and the history compaction applies to
  them.
- **Abandon.** A run whose last message is a message of the user abandons
  the calls waiting, as AG-UI allows a client that leaves them: the turn is
  canceled, `client_tools_abandoned`, and not stored, then the new turn
  begins. A confirmation waiting is not abandoned: such a run answers 409
  `conversation_busy`.
- **Tools no longer offered.** A call of a tool the run does not offer,
  such as a tool of the client the history called and the client no longer
  declares, is answered with a tool error, `no tool of this name is offered
  in this run`, rather than failing the run.

`GET /v1/{backend}/capabilities` gives the AG-UI `AgentCapabilities` of the
route, for a client to shape its interface before a run: `transport`
`streaming`, `tools` `supported` and `clientProvided`, `parallelCalls` left
undeclared, `humanInTheLoop`
`approvals`, `interrupts` and `approveWithEdits`, and `multimodal`
`input` text only. `HttpAgent` does not read it: a client subclassing it
implements `getCapabilities()` with it.

### Tools

`GET /v1/{backend}/tools` lists the tools of the MCP the model may be
offered, for a client to show what the assistant can do. The agent opens a
session with the token of the caller, as for a run, but the list is the
catalog of the deployment: the MCP serves it whole to every user, without
the actions when they are disabled for the deployment, and the rights of the
user apply when a tool is called, not to the list. The tools for the client
only, such as the previews, are left out. Each tool gives its `name`, its
`title`, at most 200 bytes, its `summary`, the first sentence of its
description, at most 512 bytes, past which both are cut on a character and
end with an ellipsis, its `description`, whole, as the model reads it,
`arguments`, the description of each argument by its name, at most 64 of
1 KiB each, `read_only`, and
`needs_confirmation`, whether a call of it waits for the confirmation of the
user. A client shows the description and the arguments as plain text: they
come from the MCP. The whole catalog of om3-mcp weighs about 110 KiB. The answer is not cached (`Cache-Control: no-store`); an MCP that does
not answer gives 502 `tools_unavailable`.

`go test -tags agui_conformance ./internal/api/` runs the official client,
`@ag-ui/client` at the version `internal/api/testdata/agui` pins, 1.0.3,
against the route, after `npm ci --ignore-scripts` in that directory: its
event verification accepts a run with text, tool calls and a confirmation,
and its resume, and a run leaving a call of a tool of the client pending,
then the run answering it, and it rebuilds the messages of the thread from
them. Run
it again on each new version of the client.

## Resources

When the MCP of a backend declares the resource capability, the agent lists
its resources at the start of each turn, and offers the model a tool of its
own, `read_resource`, in every request of the turn: it reads one text
resource of the MCP, such as `oc3://docs/form-definition`, the reference of
the YAML of an oc3 form. Its description lists the text resources, at most
10, each with its description cut to 240 bytes, and its `uri` input takes
one of them only; a read gives the text, cut to 64 KiB, and a refused read
is a tool error for the model, not a failed turn. The name is the agent's:
an MCP tool named `read_resource` fails the turn, as one named
`search_tools` does. A listing that fails leaves the turn without the tool,
and a call of it, such as one waiting behind a confirmed call when the
turn resumes, is answered with a tool error. The listing is one more MCP
request per turn, resumptions included: the session is opened per turn,
and nothing caches it.

## Documentation

When the agent has a book, with `OPENSVC_AI_DOCS_URL`, the `om3` backend
offers the model a tool of the agent's own, `search_docs`, in every request
of the turn, and its system prompt says when to use it: for how OpenSVC
works, how to configure or do something, or what is advised. The book is
read from its `llms.txt`: the pages under `agent/`, and the history of the
keywords, `changes.md` and `removed.md` of the keyword reference, listed in
its optional section; the rest of the reference is left out, the daemon
serving it to the tools at the version it runs. The pages are indexed in
memory by section. A search takes English keywords, an optional `limit`, 1
to 5, 3 by default, and an optional `version`, the release line it is about;
it gives, the best first, the text holding each matching section, once each,
16 KiB in all, those far under the best score left out: the page when it is
short, else the run of its sections, up to 6 KiB, the section is in, cut
between sections of level 2 where it can be, each opening on the headings it
is under. The result names the book, its build and the agent release of its
keyword reference, which release line it is the book of and why, and ends on
how to read it: the sections describe OpenSVC, not the cluster, and a value
of an example is no default. A refused search is a tool error for the model,
not a failed turn. The name is the agent's: an MCP tool named `search_docs`
fails the turn. Without a book, the tool and its prompt are left out.

The book is published by release line, as its `versions.json` lists them
for the release picker of the book, under the configured root:

```json
[{"name": "v3.1", "path": "/"}, {"name": "v3.0", "path": "/v3.0/"}]
```

A name carries a release line, such as `v3.1`, and a path is a directory
under the root, `/` the latest book; an entry otherwise, or a second one at
`/`, is left out, said in the log. Without a `versions.json`, or with none
at `/`, the root holds one book, of the line its `Keyword reference` names.

The book at `/` is the latest, read at startup; the book of an earlier line
is under its path, such as `v3.0/llms.txt`, read the first time a search
asks for it, and kept. One that cannot be read is asked again 10 minutes
later, the search taking the latest book meanwhile. Every 15 minutes, on a
search, the agent reads `versions.json` and the `llms.txt` of the latest
book again: the latest book when its `Build` changed, the earlier ones from
their new paths when asked after `versions.json` changed. A documentation
that cannot be read at startup is read again on a search, every minute at
most, the tool offered meanwhile and answering with a tool error that it
cannot be read.

The tool lists the release lines with a book, and the model names the one
the question is about, as the tools reported the agent version of the nodes
concerned. When it names none, the `om3` backend searches the book of the
line the nodes of the cluster of the request run, the oldest when they
differ, read through the MCP once in 5 minutes per cluster, every page of
the nodes, a cluster of more than 1000 nodes or a version that cannot be
read, such as on a tool error, taking the latest book, the result saying so;
else the latest book. An MCP that does not list `get_cluster_status` gives
no version, the agent not calling it, and the latest book; a failed MCP
transport during the read ends the turn, as a failed call of the model does.
The version is not read while the agent has no book. A version of a line
newer than the latest book, of a line with no book, or naming no line, gets
the latest book, the result saying so.

## Skills

When the MCP of a backend declares the MCP Skills extension,
`io.modelcontextprotocol/skills`, the agent lists its skills with
`skills/list` at the start of each turn, and offers the model a tool of its
own, `load_skill`, in every request of the turn. A skill is a procedure for
a task, an Agent Skill whose `SKILL.md` the MCP serves as a resource. The
description of `load_skill` holds one line per skill, its name and its
description cut to 300 bytes, at most 8 skills; the prompt does not grow
with the skills. The files of the skills are left out of `read_resource`:
loading, never reading the file, activates a skill.

A load reads the `SKILL.md` from the MCP, unless its manifest gives it
beyond 64 KiB, and checks it against the entry `skills/list` gave: its
size, its SHA-256 digest and its frontmatter, field by field. The agent
reads the frontmatter in the form the specification gives it: strings,
plain, quoted or in a literal (`|`) or folded (`>`) block, and one level of
map of such strings. Any other YAML refuses the skill with the reason, any
difference with the listing too, a tool error for the model, not a failed
turn. A skill whose name is not the last segment of the path of its
`SKILL.md` is not listed. The model gets the body of the skill, the
frontmatter apart, with its origin, the backend of the agent, such as
`om3`, never the name the server gives itself. The tools the skill names in its `metadata` key
`opensvc/tools`, at most 24, are listed right after `search_tools`, where
the tools a search finds go, and stay listed when the turn resumes after a
confirmation; those the backend does not serve, such as actions it
disables, are named to the model. `allowed-tools` is never read. The text of a
skill loaded in an earlier turn is not kept, but its tools stay listed in
the next turns, at most 8, before the core tools, as the tools a search of
a recent turn showed: taken from the listing of the turn, never from the
history, so that a skill no longer served, or a refused load, carries
nothing. Only the `SKILL.md` reaches the
model: the other files of a skill, such as a `references/` page, are served
neither by `load_skill` nor by `read_resource`, which leaves out the files
of every skill listed. A server without the extension, or
whose listing fails, leaves the turn without the tool; an MCP tool named
`load_skill` fails the turn.

A text read is a tool result: in the next turns, the history keeps no
string longer than 256 bytes, so that a reference is read again in the
turn that needs it. The tool descriptions of the MCP tell when to read
one.

## Tool search

When the MCP of a backend lists more tools than
`OPENSVC_AI_TOOL_SEARCH_THRESHOLD` (20 by default), or tools too large for
one model request (1 MiB), the model does not see them all. A threshold of
`-1` searches only such a large catalog. The agent accepts catalogs of up to
1024 tools and 4 MiB, the size of one MCP response. It sees `search_tools`, a tool of the agent, and an active set:

- the core tools: those the MCP marks with `"opensvc/core": true` in their
  `_meta`. om3-mcp and oc3-mcp mark their own; the agent holds no list of
  them;
- the tools called in the recent history of the conversation, at most 8;
- the tools its searches found during the turn. They come right after
  `search_tools`, the most recent first, then the core tools and those of the
  history: listed last, after the core tools, the model searched again for a
  tool it had been shown in half of the replays of a first turn.

`search_tools` takes a query and an optional limit (5 by default, 10 at
most). Its description, the description of its query and the system prompt
ask the model for 3 to 8 English keywords describing the required
capability, without greetings, full sentences or resource identifiers: a
whole question brings words that favor the wrong tools. The query is bounded
to 512 UTF-8 bytes (its schema can only state 512 characters as
`maxLength`), its first 32 distinct terms are searched, and a query without
a searchable term is refused. The agent ranks the catalog with BM25 over the name, title,
description, parameters and `opensvc/keywords` metadata of each tool, and
returns the best tools with the first sentence of their description. Those
scoring at least `OPENSVC_AI_TOOL_SEARCH_SCORE_RATIO` (0.30 by default, an
experimental value) of the best score of the search are shown from the next
model call of the turn; the others are listed with `shown` false and
`reason` `low_score`. BM25 scores compare within one search only, hence a
ratio rather than an absolute score. The result lists the tools first, then
`next`, which names the tools shown and asks the model to call them directly
rather than search for them again, then `hint`, about the tools not shown: a
model reading the hint about the `low_score` tools first searched again for
the tools already shown, until its searches and iterations ran out. A tool already shown stays shown. A
query naming a tool of the catalog exactly returns that tool alone, marked
`exact`, without a BM25 ranking, and shows it within the budget, never
removing a core tool for it: the way to show a tool listed but not shown.
Only the hint of a result says so, beside the tools it concerns: said in the
system prompt and in the description of `search_tools`, it made the model
search again the exact name of the tools already shown. A ratio of `0` shows every
result. When the tools shown would
exceed the size of a model request, the oldest tools that are not core make
room, those of the history first; a tool that still does not fit is
returned with `shown` false and `reason` `budget`. A turn runs at most 3
valid searches, counted in its tool calls; a refused search is a tool error
for the model and does not count. The last search says so in `next`, and
`search_tools` is no longer offered to the model afterwards: it answers with
the tools shown. A search it still requests is refused as a tool error. A turn suspended while searching keeps
searching when it resumes. A tool of the catalog the model calls without having been shown it
runs as any other, with its annotations and confirmation. The searches are
stored with the tool calls of the turn, so that a feedback shows them, and
a turn resumed after a confirmation sees the tools it found.



## History compaction

The model receives the earlier turns of a conversation as a compact copy;
the stored messages, the feedback and the export keep the original results.
The tool calls, their arguments and call IDs are kept. Small results, errors
and results without structured content are kept as they are. A structured
result over 2 KiB is compacted to at most 2 KiB: its scalars and small
objects of scalars, such as a provenance, are kept, numbers as written and
strings unchanged; a string over 256 bytes is omitted whole; collections
keep their first items, 8 at first, fewer until the result fits, and an
array keeps the prefix before its first item it cannot keep whole, so that
the indexes stay true; an object of more than 32 entries, such as a status
per node, keeps its first entries; collections nested deeper than 6 levels
are omitted. Beside the structured content, the text copy the MCP SDK adds
is dropped: a single text, without annotations or metadata, holding the
value of the structured content and nothing after it. Distinct texts, such
as a warning, keep the prefix of those of at most 256 bytes, so that their
indexes stay true. The budget holds for the whole result, texts and marker
included: the texts kept are dropped before the structured content, and the
last resort is a marker alone. The keys of a compacted object are sorted.

Beyond 16 KiB of tool results in the history, the oldest results are
reduced further, down to a marker alone. When they still exceed it, the
oldest turn is left out whole and the turns kept are compacted again from
their original results. A policy given to the service is completed with the
default values of its fields left at zero, and refused when invalid.

A compacted result carries a `_history_compaction` field: the policy
(`v1`), `partial`, the original size, and its omissions, at most 16, each a
JSON pointer into the result with its kind (`array_tail`, `object_entries`,
`long_string`, `depth`, `content`, `value`) and the items kept and omitted;
no artificial item is added to the data. The system prompt tells the model
that omitted data is unknown, to read omitted details again with a read-only
tool, never again with a tool that changes state, and that a new read is a
new observation. Each run records the policy, its budgets, the sizes of the
history loaded, compacted and sent, the results compacted and the turns left
out.

## Feedback

The user rates a completed turn, to help review and improve the tools and the
prompts. `PUT …/turns/{turn_id}/feedback` takes:

```json
{"rating": "down", "reasons": ["missing_information"], "expected": "The number of nodes per OS"}
```

- `rating` is `up` or `down`.
- `reasons`, only with `down`, optional, any of `not_answered`,
  `missing_information`, `incorrect_or_outdated`, `hard_to_understand` and
  `action_not_done`.
- `expected`, optional, is what the user would have expected: at most 1000
  characters, kept as untrusted text.

A turn has one feedback: a new `PUT` replaces its rating, reasons and expected
text, and `DELETE` withdraws it. Only the owner of the conversation can rate
its turns; a turn still running or failed is refused with 400
`invalid_feedback`. The assistant messages of `GET …/messages` carry the
feedback of their turn, for the client to show it again.

The first feedback of a turn copies it for review: the question, the answer,
and the tool calls with their arguments, error flag and the start of their
results, each truncated. It also records what executed the turn, as the
turn recorded it while it ran: the agent version, the model, a hash of the
system prompt actually sent, whether the tool catalog was searched and the
tools shown to the model, the skills it loaded, each with the URI of its
`SKILL.md` and the SHA-256 digest of the text the model was given, and the
MCP server, with the name and version it announced
and a hash of the tools it listed, which changes with any tool name,
description, schema or annotation. A turn resumed after a confirmation on a
different agent or MCP records one run per execution. The feedback holds no
identity: it stays linked to its turn while the conversation exists, for the
user to change it, and is then kept anonymous
until `OPENSVC_AI_FEEDBACK_RETENTION` (180 days by default) ends. The
context is bounded to 16 KiB: beyond, it leaves out the shown tools of the
runs, counted in `shown_tools_omitted`, then the runs between the first and
the last, counted in `runs_omitted`, and sets `truncated`. The agent
keeps at most 4096 feedback; beyond, a new one returns 409 `feedback_limit`.

`opensvc-ai-agentd feedback export [-since 720h] [-with-tool-results]`, run on
the agent host with its environment, writes the feedback created within the
duration as JSON lines, without conversation or turn IDs. The tool results,
which may hold inventory data, are left out unless `-with-tool-results` is
given. Sending the export to OpenSVC is the choice of the operator.

## Identity and isolation

A conversation is bound to its backend and to the verified issuer and subject
of its creator, plus the verified cluster ID for om3; oc3 conversations have no
cluster. Listing returns only the conversations of the
caller on that backend. Reading, renaming or deleting another identity's
conversation returns 404, as for a missing one, without revealing whether it
exists.

Provider credentials remain in the agent configuration and are never returned
through the API.
