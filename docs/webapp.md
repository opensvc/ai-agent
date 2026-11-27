# OpenID client headers

The [integration guide](integration.md) tells how an interface speaks to the
agent with AG-UI. This guide adds what a browser client of the om3 backend,
such as the OpenSVC webapp, needs: its headers, the verification of its
token, the history, the feedback, the choice of the model, the confirmation
of actions and CORS.

Each protected HTTPS request under `/v1/om3` supplies the user's access token
and cluster ID, and may name a daemon node:

```http
Authorization: Bearer <OpenID access_token>
X-OpenSVC-Cluster-ID: <cluster.config.id>
X-OpenSVC-Node: <daemon.nodename>
```

`X-OpenSVC-Node` is optional and om3-only. Requests under `/v1/oc3` send only
`Authorization`, and no other header. Use the headers of the backend for each
of its routes: conversation creation, listing, reading, renaming, deletion,
runs and feedback. The routes are listed in the
[client API](api.md#routes). The target belongs in the header so
authentication completes before reading the body.
Use the current token after renewal; the agent does not refresh tokens.

Each target header is one exact, nonempty value of at most 256 bytes, without
surrounding whitespace, control characters or commas. Duplicate headers are
rejected. The cluster ID is required; the node, when present, narrows the
target. The MCP resolves the target in its administrator-owned catalogue,
never a client-supplied URL or DNS lookup. Missing or unknown targets are
refused without fallback.
Only OpenID Connect access tokens are accepted: a token issued by the daemon
itself, after a local sign-in, is refused.

## Verification and forwarding

The MCP checks structure, required `iss/sub/aud/exp`, optional `nbf`, a
nonempty `kid`, and asymmetric algorithms RS256/384/512, PS256/384/512 or
ES256/384/512.

The agent treats the token as opaque and forwards the exact Bearer and target
headers to its configured MCP's `GET /mcp/auth/whoami` over the MCP's local
Unix socket. The MCP verifies the token signature, with the issuer and
audience configured in the MCP, before relaying it to the target daemon, which
confirms the identity and enforces grants. The om3 MCP then returns:

```json
{
  "cluster_id": "<requested cluster ID>",
  "issuer": "<JWT iss>",
  "subject": "<JWT sub>",
  "expires_at": "<JWT exp as RFC3339>"
}
```

The oc3 MCP returns the same fields without `cluster_id`, after the collector
confirms the identity. The agent validates this bounded JSON response: complete
identity fields, a future expiry and, for om3, the requested cluster ID. It does not decode the JWT or
compare claims locally. The returned subject remains the opaque OpenID `sub`,
not `preferred_username`. MCP credential refusals, expired identities or a
requested-cluster mismatch return 401; unavailable verification or malformed
bridge responses return 503. No conversation or model access is
allowed before this check. The agent caches a whoami answer at most one
minute, never past its expiry, and a refusal five seconds. Conversation
ownership is backend + issuer + subject, plus the cluster ID for om3; oc3
conversations have no cluster. Each subsequent MCP request carries the same
Bearer and targets from private request context over MCP's local Unix socket;
none is exposed to the model or stored as a credential. Origin binding applies to all headers.
The returned expiry bounds the protected operation. Authentication itself
uses a short timeout, without locally reading the token's expiry.

## Conversation messages

`GET /v1/om3/conversations/{id}/messages` returns persisted user/assistant display
text from completed turns only. Supply the same Bearer and target headers as
other conversation operations. Every read verifies identity again; access is
bound to the conversation owner. Missing and foreign IDs both return 404;
owned expired conversations return 410. Successful responses use
`Cache-Control: no-store`.

```json
{
  "messages": [
    {
      "id": "turn-id:1",
      "turn_id": "turn-id",
      "role": "user",
      "text": "Assess cluster health",
      "created_at": "2026-10-05T12:00:00Z"
    }
  ],
  "next_cursor": "1:1"
}
```

The initial page contains the latest messages, ordered chronologically within
that page. `limit` defaults to 50 and accepts 1 through 100. To load older
messages, pass the returned opaque cursor as `before` and prepend the resulting
page. An empty `next_cursor` means there are no older messages. Message IDs
remain stable across reads and restarts; pagination is exclusive and does not
shift when a new turn completes. Unknown, duplicate or invalid query parameters
return 400.

Pages are also bounded to 1 MiB of encoded JSON and may contain fewer than
`limit` messages. A single message that cannot fit is not truncated: the API
returns 413 with code `history_message_too_large`. An empty conversation returns
`{"messages":[],"next_cursor":""}`.

Only persisted text is projected. Empty assistant tool-call messages, tool
arguments/results, system prompts and provider credentials/state are excluded.
Tool activity and token usage are not returned in this V1. User timestamps use
the turn start time; assistant timestamps use its completion time, not the exact
time of each streaming chunk. Failed/canceled/interrupted turns do not persist
their prompt or partial answer, and therefore cannot be reconstructed here.

Reading does not invoke the model or tools, update retention, or change the
model context. All retained display messages can be paged, independently of
the smaller context window selected for model turns. Existing stored turns are
readable without a schema change.

The IDs of the messages are those the runs give them, so that a client
holding a page is in phase with the agent; a prompt goes by the ID the client
gave it, when the agent kept it. See [the API](api.md#messages-snapshot).

## Feedback

Under each assistant answer, offer 👍 and 👎, unless
`GET /v1/om3/capabilities` returns `custom.feedback` false: the operator
turned the feedback off, and its routes return 403 `feedback_disabled`. An
agent older than the setting returns no `custom`: the feedback is on.
After a 👎, offer the optional
reasons and a free text field "What did you expect?", then send them with
`PUT /v1/om3/conversations/{id}/turns/{turn_id}/feedback`, using the
`turn_id` of the message, or the `result.turnId` of `RUN_FINISHED` for an
answer just streamed:

| Reason | Code |
|---|---|
| Does not answer my question | `not_answered` |
| Missing information | `missing_information` |
| Incorrect or outdated information | `incorrect_or_outdated` |
| Hard to understand | `hard_to_understand` |
| Expected action not done | `action_not_done` |

`feedback` on the assistant messages of a page restores the choice after a
reload; `DELETE` on the same path withdraws it. With the feedback off, the
messages still carry the feedback given before: show it read-only, or not at
all, since changing or withdrawing it is refused. Tell the user that their
feedback and the exchange it rates are kept to improve the assistant, without
their identity. See [the API](api.md#feedback) for the limits.

## Model

When `GET /v1/om3/capabilities` returns `custom.models` true, read
`GET /v1/om3/models`, and offer a choice of the model beside the prompt when
it lists two models or more. With one model, an agent without
`custom.models`, or a list that cannot be read, offer no choice and send no
model: the runs are on the default model.

- Show each model by its `label`; a model whose `ready` is false, shown, is
  not offered.
- Choose first the `model_id` of the last assistant message of the
  conversation opened, then the last choice of the user, kept by the client
  for the backend, then the model whose `default` is true; an ID the list no
  longer holds is passed over.
- Send the choice in `forwardedProps.model` with each new prompt. The choice
  waits while a run streams: a confirmation or the answers to the calls of
  the tools of the client resume the turn on its model, whatever they send.
- On 400 `model_unknown`, the operator removed the model: read the list
  again, choose the default, tell the user, and keep their prompt for them to
  send it again.

See [the API](api.md#models).

## Skills

When `GET /v1/om3/capabilities` returns `custom.skills` true, read
`GET /v1/om3/skills`, and let the user ask one of its skills for a prompt.
An agent without `custom.skills`, or a list that cannot be read or is empty,
offers none.

- A `/` typed at the start of an empty message box opens the list of the
  skills, filtered by what follows, each by its `name` and its
  `description`, chosen with the arrows and Enter or Tab. Chosen, the skill
  is a tag over the message box, removable; the words typed after it are the
  prompt. A `/` elsewhere, or once a skill is chosen, is text.
- Send the skill in `forwardedProps.skill` with the new prompt. A skill
  chosen without words sends `/` and its name as the prompt, which the
  conversation shows as the message of the user.
- The agent tells the load as a step, `load_skill` and the skill, before the
  answer, and keeps it in the history; a load refused ends with an error
  result before the `RUN_ERROR`.
- On `RUN_ERROR` `skill_unavailable`, the skill is no longer served to the
  user, or could not be loaded: read the list again, tell the user, and keep
  their prompt for them to send it again. On 400 `invalid_skill`, the client
  sent a name not shaped as one.

See [the API](api.md#asked-skills).

## Interface context

With a new prompt, send in `context` what the screen shows and what the user
did or set in their session, so that a question such as "why is it down?" or
"stop these" needs no name: see [Interface context](api.md#interface-context)
for the bounds. Send what the user did or set, never the content of a
record.

The descriptions are a vocabulary the model reads, the same from every
client:

| Description | Value | Read from |
|---|---|---|
| `page` | The route, one segment: `/objects` | The address |
| `selected object`, `selected node`, `selected network`, `selected pool`, `selected store`, `selected namespace`; `selected record` elsewhere | The identifier of the row open in the panel of the list | The address |
| `open object`, `open node`, `open instance`, `open resource`, ... | The identifier of the record looked at over the page: `path`, `path@node`, `path:rid@node` | The address |
| `open tab` | The name of the tab of that panel | The address |
| `open view` | The name of the view within that tab | The address |
| `checked objects`, `checked nodes`, `checked resources` | The identifiers of the rows ticked in the list, separated by a space, in its order | The page |
| `checked objects count`, ... | The number of the rows ticked, in decimal | The page |
| `checked nodes`, ... | `all the rows of the list`: every row of the list ticked, on all its pages, when their ids do not fit and its filters are known whole | The page |
| `list filters` | The filters of the columns of the list, in words, separated by `; `: `status is In service; nodename contains dev` | The address |
| `session filterset` | The filterset the session narrows every list to, in oc3 | The collector |
| `time zone` | The IANA time zone of the browser: `Europe/Paris` | The browser |
| `local time` | The clock of the browser, ISO 8601 to the second with its offset: `2026-11-16T15:40:12+01:00` | The browser |

- **When.** Read the context when the prompt is sent, or queued: a prompt
  queued while a turn runs goes with the page and the rows of its time.
- **The assistant page.** On `/ai`, nothing is under the conversation: send
  the time zone and the local time only. A conversation continued there
  from the dock keeps what the earlier turns said of the record in its
  history.
- **All the rows or their count.** Send every row ticked or none: past the
  bounds of a value, or with one not shaped as an identifier, send the
  count alone, for the model to ask which; when they are every row of the
  list and its filters and session filterset are known whole, send
  `all the rows of the list` with them, a criterion the model counts with
  its tools before a change. A list cut to the bounds would
  have an action asked of "these" run on a part of them.
- **Never free text.** Never the text of a record, such as an alert message
  or a log line: other people wrote it, and the model must not take it as
  the user's word. Read from the address, a value goes only when shaped as
  what it names, an identifier or the name of a tab: a link may come from
  someone else. The rows ticked come from the page, which no link sets.
- **Resume.** The run that answers a confirmation or the calls of the tools
  of the client sends no context, or a valid one, which is ignored: the
  turn keeps the one of its prompt.

## Action confirmation

A tool that changes the cluster and that MCP annotates `destructiveHint: true`
never runs on the model's request alone: the user confirms or rejects it. A
tool that is not read-only and declares no `destructiveHint` is treated as
destructive, as the MCP specification defaults it. Read-only tools and
non-destructive actions run without confirmation.

### Request

When the model calls such a tool, the agent stores the turn in the
`awaiting_confirmation` state without running the call, gives the call as a
tool call, `TOOL_CALL_START`, `TOOL_CALL_ARGS` and `TOOL_CALL_END`, and ends
the run with `RUN_FINISHED`, `outcome` `interrupt`:

```json
{"type": "interrupt", "interrupts": [{
  "id": "<confirmation ID>", "reason": "tool_call", "toolCallId": "<call ID>",
  "expiresAt": "2026-10-09T17:45:00Z",
  "metadata": {"turnId": "<turn ID>",
    "tool": {"name": "stop_object", "title": "Stop object", "destructive": true},
    "editable": true, "argumentTypes": {"yaml": "application/yaml"}}}]}
```

Show the tool, its title and its exact arguments, the `TOOL_CALL_ARGS` of
the `toolCallId`, which are the call that runs if the user approves: never
the model text, which can describe the action differently. The model is told
to write before the call what leads to it and what it will change, after the
answer to a question of the same message: its `TEXT_MESSAGE_*` events come
before `TOOL_CALL_START`. Show that text above the confirmation, as its
explanation; the arguments alone say what runs. The confirmation expires
after 10 minutes.

`argumentTypes`, when present, gives the media type of text arguments, as
the input schema of the tool declares them in `contentMediaType`, such as
`{"yaml":"application/yaml"}` for the definition of a form: show such an
argument by its type, its lines and indentation kept, highlighted for a
type the client knows, rather than guess the format of a text. Show any
text holding line breaks with its whitespace kept, typed or not
(`white-space: pre-wrap`).

### Decision

Answer with a new run on the conversation, its `resume` the decision:

```json
{"threadId": "<conversation ID>", "runId": "<run ID>", "messages": [],
 "resume": [{"interruptId": "<confirmation ID>", "status": "resolved", "payload": {"approved": true}}]}
```

`{"approved": false}`, or the status `cancelled`, rejects the call. The
identity is verified again and must own the conversation. A confirmation is
used once.

When `editable` is `true`, the `responseSchema` gives `editedArgs`, the
input schema of the tool: show the arguments in a form built from it, its
labels and descriptions from the schema, the values the model proposed
filled in, and approve with `{"approved": true, "editedArgs": {…}}`, the
full arguments, only when the user changed a value. The `metadata` of the
interrupt may give `preview`, what the call would do as the MCP tells it,
or `previewError`: show it above the arguments; after an edit,
`{"action": "preview", "editedArgs": {…}}` asks it again for the edited
arguments, nothing running, the call waiting again under a new interrupt. Check the form against
the schema before sending it; the agent checks it again, and an edit it
refuses returns 400 `invalid_edited_arguments` with the violation, the
confirmation still waiting. Edited arguments with a rejection, or any other
payload, return 400 `invalid_decision`. Once approved, show the arguments
the call ran with: the stream gave those the model proposed.

The run streams the rest of the turn. Approved, the stored call runs, its
`TOOL_CALL_RESULT` given, then the model continues; rejected, its result
tells the action did not run, and the model answers. The run may end with
another interrupt when the model calls a further destructive tool.

| Case | Status | Code |
|---|---|---|
| Invalid resume or decision | 400 | `invalid_decision` |
| Edited arguments the input schema refuses | 400 | `invalid_edited_arguments`; the confirmation still waits |
| Unknown or foreign conversation | 404 | `conversation_not_found` |
| No such pending confirmation, or already decided | 409 | `confirmation_not_pending` |
| Confirmation expired | 410 | `confirmation_expired`; the turn failed and the call did not run |

### While it waits

- `GET /v1/om3/conversations/{id}` returns `pending_confirmation`, the
  confirmation with its `id`, `turn_id`, `tool`, `arguments`,
  `argument_types`, `editable`, `input_schema`, `expires_at`, and `preview`,
  `preview_error` and `preview_arguments` when the interrupt gave them,
  while it waits: show it again after a page reload, editable when it was, and answer
  it with its `id` as the `interruptId`. It holds the call, not the text the
  model wrote before it: after a reload, the confirmation comes back without
  its explanation.
- A prompt returns 409 `conversation_busy` until the user decides or the
  confirmation expires; an expired one no longer blocks the conversation.
- A pending confirmation survives an agent restart.

## CORS

Set `OPENSVC_AI_CORS_ALLOWED_ORIGINS` in the agent environment and restart:

```dotenv
# Restrictive: allow these webapp origins only.
OPENSVC_AI_CORS_ALLOWED_ORIGINS=https://webapp-a.example:1215,https://webapp-b.example:1215

# Alternatively: allow every browser origin.
OPENSVC_AI_CORS_ALLOWED_ORIGINS=*
```

Empty or unset disables CORS. Origins contain the scheme, hostname and optional
port, without `/ui`, a trailing slash, credentials, query or fragment. HTTP
origins are accepted for local frontend development; the agent API still uses
HTTPS. Hostnames are normalized to lowercase and default ports are omitted.
Partial wildcards are not supported, and `*` cannot be mixed with a list.
Malformed configuration prevents startup, without echoing its contents.

The browser client must use `fetch` with `credentials: "omit"` and supply its
Bearer token explicitly. The agent never sets `Access-Control-Allow-Credentials`.
Open mode permits use from any website; it does not relax JWT verification,
cluster/node catalogue restrictions, daemon grants or TLS checks. Use exact
origins when they are known and controlled.

Preflight `OPTIONS` requests are handled before JWT authentication. Allowed
methods are `GET`, `POST`, `PUT`, `PATCH` and `DELETE`; allowed request headers are
`Authorization`, `Content-Type`, `X-OpenSVC-Cluster-ID` and `X-OpenSVC-Node`.
Successful preflights return 204 and may be cached by the browser for 600
seconds. Actual requests still pass through normal authentication, and allowed
origins receive CORS headers on errors and streamed runs too. Unlisted origins
in restrictive mode return 403; invalid preflight methods/headers also return
403 without invoking authentication, conversation storage or the model.

Clients without `Origin` are unaffected. CORS is a browser
policy, not an authentication mechanism or protection against non-browser
clients. No CORS headers or origin registrations are sent to MCP or daemons.
