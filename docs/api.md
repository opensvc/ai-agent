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
| `POST /v1/om3/runs` | Run one prompt for a client no user watches, read-only, nothing stored; see [Unattended runs](#unattended-runs) |
| `GET /v1/om3/capabilities` | What the runs route can do, as an AG-UI `AgentCapabilities` |
| `GET /v1/om3/models` | The models a run chooses from; see [Models](#models) |
| `GET /v1/om3/tools` | The tools the assistant may call for the caller; see [Tools](#tools) |
| `PUT /v1/om3/conversations/{id}/turns/{turn_id}/feedback` | Give or replace the feedback on a turn |
| `DELETE /v1/om3/conversations/{id}/turns/{turn_id}/feedback` | Withdraw it |

`GET /health` is shared by all backends and needs no authentication.

Every response carries `X-Request-ID`, the ID the agent gives the request.
Every audit line of the agent carries it as `request_id`, and the agent
forwards it to the MCP of the backend, whose audit lines carry it too: one ID
follows a request from the agent to its tool calls.

Each model call of a run has its audit line `llm_usage`: its `model_id`, its
`iteration`, its tokens, `input_tokens` counting those the provider read from its cache,
`cache_read_tokens`, and wrote to it, `cache_write_tokens`, 0 when the
protocol does not tell, and its timing on the clock of the agent, `duration_ms` from the
request to the usage the provider reports, and `first_event_ms` to the first
event of its stream, the wait of the provider before it answers. A slow run
tells there whether the provider or the tools took the time.

A model request the provider fails for a transient reason before it gives
any event, an overload, a rate limit, an error of its own or a timeout
(HTTP 408, 429, 500, 502, 503, 504 or 529, or an `overloaded_error`,
`rate_limit_error` or `api_error` of the stream of the Messages protocol), is
sent again, twice at most, after the wait the provider asks for with
`Retry-After`, else 1 then 4 seconds: the run keeps the tool calls it made,
such as an action the user confirmed. A wait over 20 s, a request that
would start less than 10 s before the timeout of the run, a spent quota
(`insufficient_quota`) or a request that gave an event, such as text the
client was shown, fail the run as before. Each request sent again has its
audit line `llm_retried`: its `model_id`, its `iteration`, its `attempt`, the HTTP `status`
of the failure, 0 for an error of the stream, and its `wait_ms`.

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
  added to the model context and their partial output is not stored. A turn
  never stays running for its conversation: a run whose stream could not
  start, or whose first frame could not be written, cancels it,
  `streaming_unavailable` or `stream_not_started`; a run that panics fails
  it, `agent_panicked`; a turn left running by a stop of the agent is
  marked interrupted when it starts again.
- **Tool calls.** A turn runs at most 8 tool calls per model answer and 16 in
  all. A model answer asking for more runs none of its calls: each is answered
  with a tool error naming the bound, and the model asks for fewer or answers
  with what it has. Once 16 ran, the next model request takes no call. The
  client is shown none of the refused calls; the audit records the answer as
  `tool_calls_refused`, with the `bound` it passed, `answer` or `question`,
  `calls_asked` and `calls_ran`.
- **Reasoning.** The reasoning of the model, the thinking of the Messages
  protocol or the encrypted reasoning of the Responses protocol, is given
  back to the model in the next model calls of the run, never to the client,
  and never stored: neither the history nor a turn waiting on a
  confirmation or on the client keeps it, the run that resumes it starting
  without. A thinking block of the Messages protocol is bound to the system
  prompt and the tools of its request: once a `search_tools` shows more
  tools, the thinking before it is left out.
- **Calls of the agent.** An MCP call the agent makes of its own, not the
  model, such as reading the version of the cluster a documentation search
  is about, is not shown to the client nor counted in `tool_calls`; the
  audit records it as `agent_tool_called`, with its `tool_name`, its
  `purpose`, such as `documentation_version`, `tool_error` and
  `duration_ms`.
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
  decision left out, its `usage`, `total_tokens`, and its `model_id`, the
  model the turn last ran on, absent for a turn recorded before. Times are milliseconds
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
- **Stored bytes.** A user stores at most `OPENSVC_AI_CONVERSATION_MAX_USER_MIB`
  of messages on a backend and cluster, by default an eighth of the
  database, at least the size of one conversation, 32 MiB: one user cannot
  fill the database for the others. A run is refused with 409
  `conversation_quota` once the user's share is reached, `conversation_limit`
  once the database is full, before the model is called. That check avoids
  the certain case only: a turn that would pass either limit when stored ends
  with `RUN_ERROR` `conversation_quota` or `conversation_limit`, its model
  calls spent, and is not stored.

## Runs

The instance runs at most `OPENSVC_AI_MAX_CONCURRENT_ASKS` runs at once, 4
by default, every user, backend and cluster together, and a user at most
`OPENSVC_AI_MAX_CONCURRENT_ASKS_PER_USER`, 2 by default: one user cannot hold
every slot. A user is counted per backend and cluster, the tenant, by their
issuer and subject: a local user of om3 names its node as its issuer, which
the node of another cluster may share.
Beyond, a run is refused with 429 and `Retry-After`: `too_many_requests` for
the instance, `too_many_user_requests` for the user.

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
  or of a tool, but the answers to the calls of its tools, below. Nor is
  `state`; of `forwardedProps`, only `model`, the model of a new prompt:
  see [Models](#models).
- `context` is the context of the interface of a new prompt; see
  [Interface context](#interface-context).
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
  A decision applies to the call the confirmation showed, and to its tool as
  the MCP listed it then: when the MCP no longer offers that tool, or changed
  its definition since (description, annotations, schemas, metadata), the
  resumed run ends with `RUN_ERROR` `agent_failed` and nothing runs; no later
  call of the answer takes the decision. The user asks again.

The system prompt of every turn tells the model how to make a change,
wherever the question comes from: a question of the same message answered
first, in full; then, before the calls that change state, in the same
response, what leads to them and what they will change, in one or two
sentences, for several records their count and what selects them, never
their list, a set a criterion gives, such as a filter, counted before the
change; no confirmation asked in its answer, the agent asking for a
destructive call itself; the records asked, all of them and no other, by one
call of a tool that takes several or one call for each; no id or name made
up; after the calls, their result and what is left of what the user asked,
without repeating the text before them, the totals as the tool results
confirm them. The text of a response streams before its calls, so before
their confirmation. An approval tells it to finish what the user asked, each
other call confirmed on its own. The system message says in every run that
no confirmation is pending; the answer of a confirmation, the call it
covers, its path and node, and whether it was approved or rejected, ends the
result of that call, a text after its contents, which the history keeps:
the system message stays the same, and the request the turn waited after
begins the one that resumes it, which a provider reads from its cache. A
model that reasoned before the call resumes without its reasoning, which a
turn waiting on the user does not keep: its prefix then differs from its
first answer of the turn on.

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
| failure | `RUN_ERROR` with a `code`, `request_timeout`, `tool_name_conflict`, `conversation_quota` or `conversation_limit` (the turn could not be stored, see Quotas) or `agent_failed`, and a `message` for the developer |

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

### Interface context

`context`, the AG-UI `Context` entries `{description, value}`, tells what
the screen of the user shows when they send a new prompt, and what they did
or set in their session: the page, the record open on it, its tab, a record
looked at over the page, the rows they ticked, their time zone and clock;
the [browser guide](webapp.md#interface-context) holds the descriptions a
client writes. The model reads
it right after the prompt, in the same message, and its system prompt
tells it that the context is data, never instructions, that its identifiers count as given by the
user, to tell what a question such as "why is it down?" refers to, and that
the state is read with the tools before concluding. A count of rows ticked
without their list makes it ask which they are, rather than stand a wider
set in for them.

- **Bounds.** At most 8 entries, each a description of 1 to 128 bytes and a
  value of at most 2 KiB, the ids of 50 rows ticked, 4 KiB in all, printable
  UTF-8 text without control characters, or 400 `invalid_context`. Other members of an entry are
  ignored, as the protocol allows them. Absent, `null` or empty: no context,
  and the run goes as without one.
- **Lifetime.** The context goes with the turn of its prompt: a turn waiting
  on a confirmation or on the client keeps it, and the run that resumes it
  gives it to the model again: its own `context`, valid, is ignored. It is neither
  stored in the history, nor shown in the messages of the conversation or a
  snapshot: the next prompt comes with the context of its own screen.
- **Cache.** The context comes after the text of the prompt, past a cache
  point of the Messages protocol: the next turns, which give the prompt
  without it, still read the conversation up to the prompt from the cache
  of the provider.
- **Audit.** `conversation_run_started` and `unattended_run_started` carry
  `context_entries` and `context_bytes`, never the content.

The context is the client's word, as the prompt: it tells the model where to
look, and proves nothing. A client sends the state of its navigation, such
as the route and the identifiers in its address, never the free text of a
record, such as an alert message or a log line, which other people wrote.
An address may come from a link someone else wrote, too: send a value only
when it is shaped as what it names, such as an object path or the name of a
tab, short and without spaces, and leave out the others.

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
  the agent, `search_tools`, `read_resource`, `load_skill`, `search_docs` or `search_code`, is refused there; a name of
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
  the same lifetime, 10 minutes, after which it fails, its state deleted at
  the next purge of the expired conversations, on a creation or a listing:
  a resume answers 409
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
`approvals`, `interrupts` and `approveWithEdits`, `multimodal` `input` text
only, and `custom` `feedback`, always present: true, or false when the
operator turned the [feedback](#feedback) off, and `models`, true: the runs
choose their [model](#models). `HttpAgent` does not read it: a client
subclassing it implements `getCapabilities()` with it.

### Unattended runs

`POST /v1/{backend}/runs` runs the agent once for a client no user watches,
such as the scheduler of a collector sending a report: nobody can answer a
question or confirm an action, and nothing is stored.

- The request is an AG-UI `RunAgentInput` holding exactly one message of the
  user, the prompt, without history. `tools` must be empty or absent and
  `resume` absent, or 400 (`invalid_tools`, `invalid_decision`,
  `invalid_prompt`). `context`, optional, scopes the question as for a
  conversation run, with the same bounds (`invalid_context`), and
  `forwardedProps.model`, optional, chooses its model, as for a new prompt
  (`model_unknown`); see [Models](#models). `threadId` and `runId` are any IDs of the client, at
  most 128 printable bytes: nothing is stored under them, and the audit
  carries `run_id`, with the `request_id` the response returns, to find the
  run in the logs of both sides.
- The model is offered only the tools of the MCP whose annotations declare
  them read-only (`readOnlyHint: true`). A tool without annotations, a write,
  even one the conversations run without confirmation, a destructive tool
  and a hidden tool are left out, closed by default. A skill naming one is
  told it is not served, and a call of one is answered as unoffered without
  reaching the MCP. The system prompt tells the model the turn is
  unattended.
- The run never waits: it completes, or fails with `RUN_ERROR`. It takes a
  slot of the user as a conversation run does, and the same bounds: the
  prompt at most 32 KiB, the turn its timeout and iterations. A client
  running several at once for one account stays within
  `OPENSVC_AI_MAX_CONCURRENT_ASKS_PER_USER`, or gets 429.
- The stream speaks AG-UI, as a conversation run. `RUN_FINISHED` carries in
  its `result` the summary of the run, by identifiers only, never the
  arguments of a call nor its result:

```json
{"unattended": true, "finishReason": "completed",
 "toolCalls": [{"id": "call-1", "name": "list_alerts", "durationMs": 1500, "error": false},
               {"id": "call-2", "name": "attach_node_tag", "durationMs": 0, "error": true, "notRun": "unoffered"}],
 "toolCallCount": 2, "notRunCalls": 1, "refusedCalls": 0}
```

  `toolCalls` lists the first 256 calls; `toolCallCount` counts them all.
  A call that did not run carries `notRun`: `unoffered`, a tool the run does
  not offer, such as a write, or `needs_confirmation`; `notRunCalls` counts
  them, the writes the model asked for. `refusedCalls` counts the calls the
  bounds of the question refused before they ran, none of them listed.

The read-only tools are as safe as their annotations: a tool wrongly
declared read-only runs without confirmation. The account of the client is
the guarantee that does not depend on them, and its choice is the client's:
an account of read groups only, without privileges, changes nothing
whatever the tools.

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

### Models

`GET /v1/{backend}/models` lists the models the runs of the caller choose
from, the default first, for a client to offer the choice:

```json
{"models": [
  {"id": "default", "label": "GPT Luna", "default": true, "ready": true},
  {"id": "haiku", "label": "Claude Haiku", "default": false, "ready": true}
]}
```

The operator declares them, the default of the `OPENSVC_AI_LLM_*` variables,
of ID `default`, and those of `OPENSVC_AI_MODELS`; see the
[README](../README.md#several-models). Each gives its `id`, opaque to the
client, at most 64 bytes, its `label`, at most 64 characters, whether it is
the `default`, and whether the caller can run on it, `ready`, always true
today: a client shows a model that is not without offering it. Neither its
provider, nor its URL, nor its authentication leave the agent. The list is
the same for every backend and user; the answer is not cached
(`Cache-Control: no-store`). `custom.models`, true in the capabilities,
tells the route is served: a client of an agent without it offers no choice,
and sends no model.

A run names its model in `forwardedProps.model`, the `id` of the list:

```json
{"threadId": "...", "runId": "...", "messages": [...], "forwardedProps": {"model": "haiku"}}
```

- Absent, null, or with `forwardedProps` absent or not an object, the run
  is on the default model. The other members of `forwardedProps` are
  ignored.
- A model that is not a string of 1 to 64 bytes, or not of the list, is
  refused with 400 `model_unknown` before the conversation is read and
  before the stream starts; the audit keeps the code, not the value.
- A new turn runs on its model from its start to its end. A run answering a
  confirmation or the calls of the tools of the client resumes the turn on
  the model it started on, and does not read `model`; on the default model
  when the operator removed it meanwhile. The reasoning a model gives back
  lives as long as the run and is never stored, so the next turn of the
  conversation may run on another model: it reads the stored history.
- The audit line of a run of a new prompt carries `model_id`; `llm_usage`
  and `llm_retried` carry the `model_id` of each model call, resumed runs
  included. The last assistant message of a turn carries its `model_id`, for
  a client to offer again the model of the conversation it opens.
- The history of a conversation holds the IDs of the tool calls of each
  model it ran on, which the providers of the agent take from one another.
  An ID a provider would refuse, the ID of an earlier call, as a server
  numbering its calls from zero at each answer gives, or one out of letters,
  digits, `-` and `_`, is renamed in the requests only; the history, the
  stream and the audit keep it.

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

When the agent has a book, with `OPENSVC_AI_DOCS_URL`, both backends offer
the model a tool of the agent's own, `search_docs`, in every request of the
turn, and its system prompt says when to use it: for how OpenSVC works, how
to configure or do something, or what is advised. The book is read from its
`llms.txt`: the pages under `agent/`, and the history of the keywords,
`changes.md` and `removed.md` of the keyword reference, listed in its
optional section; the rest of the reference is left out, the daemon serving
it to the tools at the version it runs. The pages are indexed in memory by
section. A search takes English keywords, an optional `limit`, 1 to 5, 3 by
default, and an optional `version`, the release line it is about; it gives,
the best first, the text holding each matching section, once each, 16 KiB in
all, those far under the best score left out: the page when it is short,
else the run of its sections, up to 6 KiB, the section is in, cut between
sections of level 2 where it can be, each opening on the headings it is
under. The result names the book, its build and the agent release of its
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
line the nodes of the cluster of the request run, the newest when they
differ, read through the MCP once in 5 minutes per cluster, every page of
the nodes, a cluster of more than 1000 nodes or a version that cannot be
read, such as on a tool error, taking the latest book, the result saying so;
else the latest book. An MCP that does not list `get_cluster_status` gives
no version, the agent not calling it, and the latest book; a failed MCP
transport during the read ends the turn, as a failed call of the model does.
The version is not read while the agent has no book. The `oc3` backend has
no default, a fleet running several lines: its prompt asks the model to name
the line of the `agent_version` the collector reports for the nodes
concerned, and a search naming none gives the latest book, its result
telling the model, when the question is about given nodes, to search again
with their `agent_version`. A version of a line newer than the latest book,
of a line with no book, or naming no line, gets the latest book, the result
saying so.

## Code

When the agent has the code of the daemon, with `OPENSVC_AI_CODE_URL`, both
backends offer the model a second tool of the agent's own, `search_code`,
and their system prompt keeps it for why the daemon behaves as it does or
how a part of it works inside, usage questions being answered from the
documentation. On `om3`, a search that names no version reads the code of
the version the nodes of the cluster of the request run, the newest when
they differ, the version `search_docs` reads too, through the same MCP read,
audited as an agent call of purpose `code_version`; when that version
cannot be read, or its code cannot, the latest code, the result saying so.
On `oc3`, a fleet, a search that names no version reads the latest code of
the daemon, the archive of `OPENSVC_AI_CODE_REF`, `main` by default, read
again every 6 hours in the background, the searches served the code read
before until the new read ends, the result saying a cluster may run another
version. The model names a version only when the user names one, a tag such as `v3.0.0-rc56`, the
`v` added when it is missing, or a commit, or `v3.0.0-rc56-41-g451250d44`,
whose commit is read, else, when it is not published, its tag, the result
saying the code is not exactly that version; a version with no published
code is a tool error saying so.
The archive is read from the configured host only, in memory, at most 32 MB,
256 MiB unzipped, 8000 entries, 8 MiB of entry names, 2 MB a file and 1 KiB
a path, its Go files only, 64 MiB of them paths included, symbolic links left
out. A file left out is still read through to the next entry, and counts in
the 256 MiB: an archive of large files fails as a whole rather than be
filtered. A
file that does not parse, a generated file, marked `DO NOT EDIT`, and the
tests are left out. Each declaration is indexed with its doc comment, the
words of its identifiers searched, and a search gives, as `search_docs`
does, the file of the declarations found when it is short, else the run of
its declarations, up to 6 KiB, holding them, with its path and lines, 16 KiB
in all. A search takes English keywords or Go identifiers, an optional
`limit` and an optional `version`. The code of a version is read the first
time a search asks for it, a failed read asked again 10 minutes later;
three versions are kept or being read, the latest code always among them,
the least recently used other version leaving, never one still read: with
the three places taken by reads, a search for a fourth version is a tool
error telling the model to try again later. Two versions are read at once,
the others waiting their turn, the 2 minutes of a read counted from its
turn; a read that no search waits on any more, its runs canceled, leaves the
shelf and its download stops, an indexing under way finishing in its turn,
and the next search reads it again. A question searches the code 6 times
at most, across the confirmations it waits on: a search past them is a tool
error telling the model to answer with the code read.
The name is the agent's: an MCP tool named `search_code` fails the turn.

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
- the tools the searches of the recent turns showed without the model
  calling them, at most 15, the tools one search may show, in the order the
  turn listed them;
- the tools its searches found during the turn. They come right after
  `search_tools`, the most recent search first, the tools of one search in
  its order, the best first, then those of the recent searches, the core
  tools and those of the history: listed last, after the core tools, the
  model searched again for a tool it had been shown in half of the replays of
  a first turn.

A turn resumed after a confirmation starts with the tools of the request it
waited after, in the same order. A turn after one that searched once, and
showed at most 15 tools, starts with the tools of its last request: a
provider then reads the prefix of the request, tools, system message and
history, from its cache, where a change of the tools makes it write the
whole prefix again. The tools differ, and the next turn writes its prefix
again, when the turn before searched three times, `search_tools` then being
no longer listed, showed more than 15 tools, searched twice for tools in
common, or loaded a skill after a search, and when a tool called in an
earlier turn is called again, the tools of the history going the most
recent first.

`search_tools` takes a query and an optional limit (5 by default, 10 at
most). Its description, the description of its query and the system prompt
ask the model for 3 to 8 English keywords describing the required
capability, without greetings, full sentences or resource identifiers: a
whole question brings words that favor the wrong tools. The system prompt
also asks the model never to tell the user that something cannot be read or
done, or that no tool gives it, before `search_tools` found nothing for it,
a search of the documentation or of the code finding no tool: a model that
found no listed tool said none existed, when a search would have found it,
and another searched the documentation only. The query is bounded
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
The tool calls, their arguments and call IDs are kept. Small results, errors,
results with contents other than text, such as an image or a resource link,
and the results of `search_tools` of the last turn and of the last turn that
searched, from which the agent reads back the tools a former search showed,
are kept as they are. A structured result over 2 KiB is compacted to at
most 2 KiB: its scalars and small objects of scalars, such as a provenance,
are kept, numbers as written and strings unchanged; a string over 256 bytes is omitted whole; collections
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

Beyond 16 KiB of tool results in the history, the oldest structured
results are reduced further, to a quarter of their budget. When they still
exceed it, results are replaced by a marker alone, the oldest first, whether
structured or of text alone, without structured content, such as one of
`search_docs`, `search_code`, a skill, a resource or a former `search_tools`:
a text is never cut, and stays whole while the history fits once the
structured results are reduced. The call and its arguments stay, so that the
model can run it again, and the other results are compacted again from their
originals, so that a marker frees its room for them. When the results still
exceed 16 KiB, the oldest turn is left out whole and the turns kept are
compacted again from their original results. The compaction never leaves the
last turn out: over the budget alone, it is sent compacted, over 16 KiB, and
the turns before it are left out only while their own results exceed 16
KiB, so that the model reads the turn the user follows up and the earlier
ones. The bounds of the history, in messages and in bytes, still apply after
it, and leave the oldest turns out. A policy given to the service is
completed with the default values of its fields left at zero, and refused
when invalid.

A compacted result carries a `_history_compaction` field: the policy (`v2`;
`v1` kept the text results whole and could leave every turn out),
`partial`, the original size, and its omissions, at most 16, each a JSON
pointer into the result with its kind (`array_tail`, `object_entries`,
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
keeps at most 4096 feedback, a user at most an eighth of them on the turns of
their conversations of a backend and cluster; beyond their share, a new one
returns 409 `feedback_quota`. With the instance full, the oldest anonymous
feedback, of a deleted conversation, makes room; without any, 409
`feedback_limit`. The share counts the feedback of the conversations that
exist: one holding no identity once its conversation is deleted, a user who
rates then deletes again and again, one model turn per feedback, can fill the
anonymous feedback, and push out the oldest, the others' included. The
feedback stays anonymous by design; the cost of a turn per feedback is the
bound.

An operator whose users must not rate the answers sets
`OPENSVC_AI_FEEDBACK_ENABLED=false`: for every backend, `GET …/capabilities`
returns `custom.feedback` false, for the client to hide the rating, and `PUT`
and `DELETE …/feedback` return 403 `feedback_disabled`, after authentication
and without reading the body. The feedback stored before is kept until its
retention ends, still exported and still shown on the messages that carry it.

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
