# Integrating an interface with the agent

This guide is for the developers of an interface that talks to
`opensvc-ai-agentd`: a chat panel in an application, a portal of a customer,
a command line client. The agent speaks [AG-UI](https://docs.ag-ui.com) 1.0
for its runs, and a REST API for its conversations. The
[client API](api.md) is the reference of the routes, their bodies and
errors; this guide tells how to put them together.

## 1. Prerequisites

The agent serves each backend under its own routes: `/v1/om3` for an
OpenSVC cluster, `/v1/oc3` for an OpenSVC collector. Each request carries the
OpenID Connect access token of the user, and, for om3, the target cluster:

```http
Authorization: Bearer <access token>
X-OpenSVC-Cluster-ID: <cluster.config.id>      om3 only
X-OpenSVC-Node: <node name>                    om3 only, optional
```

The agent verifies the token on every request, through the MCP of the
backend, and never stores it. Use the current token after a renewal. A
browser client calls the agent from an origin the operator allowed,
`OPENSVC_AI_CORS_ALLOWED_ORIGINS`, with `fetch` and `credentials: "omit"`;
see the [browser client guide](webapp.md).

The collector relays the routes of its backend under `/assistant/*`, so
that its web interface calls them with its own session, the runs included,
whose body it relays as it is, up to 4 MiB, the bound of the agent.

## 2. Conversations

A conversation holds the turns of one user on one backend. The REST routes
manage it; the runs speak AG-UI on it.

| Action | Route |
|---|---|
| Create | `POST /v1/{backend}/conversations`, answers `{"conversation": {"id", "title", …}}` |
| List | `GET /v1/{backend}/conversations` |
| Read | `GET /v1/{backend}/conversations/{id}` |
| Rename | `PATCH /v1/{backend}/conversations/{id}` `{"title": …}` |
| Delete | `DELETE /v1/{backend}/conversations/{id}` |
| Read its messages | `GET /v1/{backend}/conversations/{id}/messages`, paged, the texts of the completed turns |

To open a conversation, read its messages and give them to the client as
its initial messages: their IDs are those the runs give their messages, so
that the client and the agent stay in phase. The agent keeps the ID you
give a prompt, at most 128 printable bytes without spaces, while the client
is in phase; a client out of phase gets its prompt back under the ID the
agent gives it, `<turn ID>:1`, in the snapshot that resynchronizes it. A
conversation expires after its lifetime, counted from its last turn: 410
`conversation_expired`.

## 3. Runs

A run is one `POST /v1/{backend}/conversations/{id}/runs` of an AG-UI
`RunAgentInput`, answered with a stream of AG-UI events. The `HttpAgent` of
`@ag-ui/client` speaks it:

```javascript
import { HttpAgent } from "@ag-ui/client";

const base = "https://agent.example.com/v1/oc3";
const headers = { Authorization: "Bearer " + accessToken };
const { conversation } = await (await fetch(base + "/conversations", { method: "POST", headers })).json();

const agent = new HttpAgent({
  url: `${base}/conversations/${conversation.id}/runs`,
  headers,
  threadId: conversation.id,
});
agent.addMessage({ id: crypto.randomUUID(), role: "user", content: "How many nodes are there?" });
const result = await agent.runAgent({ runId: crypto.randomUUID() }, {
  onEvent: ({ event }) => render(event),
});
// result.result.turnId names the turn, for its feedback.
```

- `threadId` is the ID of the conversation, which the URL names too.
- `runId` is yours: at most 128 printable bytes without spaces.
- `protocolVersion`, which `@ag-ui/client` sends, is the version of AG-UI
  the client speaks: a version of the line 1 is served, a newer minor one
  with a `CUSTOM` `opensvc.warning`; another line is refused with 400
  `unsupported_protocol_version`. The agent declares its own, `1.0`, on
  `RUN_STARTED`.
- The prompt is the last message, a message of the user. The agent keeps
  the history of the conversation itself and reads no other message of the
  client, but the answers to the calls of its tools (section 6).
- `forwardedProps.model`, optional, is the ID of the model of the prompt,
  among those of `GET /v1/{backend}/models` (section 7), which the parameters of
  `runAgent` carry: `{ runId, forwardedProps: { model: "haiku" } }`. Absent, the default
  model; unknown, 400 `model_unknown`.
- `internal/api/testdata/agui/conformance.mjs`, in the repository of the
  agent, is a complete client: runs, a confirmation and its resume, a tool
  of the client, and a resynchronization.
- `internal/api/testdata/agui/streams.json` holds the streams the runs serve
  in the scenarios of a client, with the input of each: an interface can
  replay them in its tests, as the om3 webapp does, so that a change of the
  stream breaks them. `AGUI_GOLDEN_UPDATE=1 go test ./internal/api -run
  TestGoldenAGUIStreams` rewrites it.

## 4. What the events show

| Event | What the interface shows |
|---|---|
| `RUN_STARTED` | The run is under way |
| `MESSAGES_SNAPSHOT` | The thread as the agent holds it, sent to a client that lacks its last stored turn, such as another tab; `@ag-ui/client` merges it |
| `TEXT_MESSAGE_START`, `_CONTENT`, `_END` | The answer of the assistant, streamed; one message per answer of the model |
| `TOOL_CALL_START`, `_ARGS`, `_END` | A tool the agent calls, with its arguments; `search_tools`, `read_resource` and `load_skill` are tools of the agent itself, which an interface may show apart, or not at all |
| `TOOL_CALL_RESULT` | Its result, the JSON the model read, or a note of its size beyond 64 KiB |
| `CUSTOM` `opensvc.warning` | A note for the developer: content parts or a version the agent did not read |
| `RUN_FINISHED` | The end: `result.turnId` for the feedback, `usage` for the tokens, `outcome` success, success with calls of tools of the client pending, or an interrupt |
| `RUN_ERROR` | The run failed: `code` and `message`, such as `request_timeout`, `tool_name_conflict` |

A snapshot holds what the model reads: the texts, the tool calls and their
results as the history compaction leaves them, so a reloaded interface shows
what the model knows. Beyond 1 MiB, or past the history the model reads,
the oldest turns are left out, and `metadata` `opensvc.truncated`
`omittedTurns` counts them: read them with `GET …/messages` to show them.

## 5. Confirmations

An action that changes the cluster or the collector never runs on the
request of the model alone: the run ends with `RUN_FINISHED`, `outcome`
`interrupt`, after the call as a tool call:

```json
{"id": "<confirmation ID>", "reason": "tool_call", "toolCallId": "…",
 "message": "Confirm the action Stop service with the arguments of its call.",
 "responseSchema": {"type": "object", "properties": {"approved": {"type": "boolean"},
   "editedArgs": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}, "required": ["approved"]},
 "expiresAt": "2026-10-22T10:00:00Z",
 "metadata": {"turnId": "…", "tool": {"name": "stop_service", "title": "Stop service", "destructive": true},
              "editable": true, "argumentTypes": {"yaml": "application/yaml"}}}
```

- Show the tool, its title, and the exact arguments of the tool call, taken
  from its `TOOL_CALL_ARGS`, never from the text of the model, which may
  describe the action otherwise. Show an argument of `argumentTypes` by its
  type, a YAML as YAML with its lines and indentation, and any text holding
  line breaks with its whitespace kept.
- Count down to `expiresAt`, 10 minutes: past it, the turn fails and the
  call does not run.
- Answer with a new run on the same thread, `resume`:
  `{"interruptId": <id>, "status": "resolved", "payload": {"approved": true}}`
  runs the call, `{"approved": false}` or `"status": "cancelled"` rejects it.
- When `metadata.editable` is `true`, the `responseSchema` gives
  `editedArgs`, the input schema of the tool: offer a form built from it,
  and approve with `{"approved": true, "editedArgs": {…}}`, the full
  arguments, only when the user changed one. An edit the schema refuses
  returns 400 `invalid_edited_arguments` with the violation, and the
  confirmation still waits; check the form against the schema before
  sending it. See [the API](api.md#edited-arguments).
- Errors: 409 `confirmation_not_pending` when the conversation waits on no
  such interrupt, 410 `confirmation_expired`, 400 `invalid_edited_arguments`
  for an edit the schema refuses, 409 `conversation_busy` for a prompt
  while it waits.
- Never show a call as approved before the run that resumes it gives its
  `TOOL_CALL_RESULT`.

## 6. Tools of the interface

Give the agent tools of your interface, such as opening a page or
highlighting a row, in `tools` of the input:

```javascript
const tools = [{ name: "open_node_page", description: "Open the page of one node.",
  parameters: { type: "object", properties: { nodename: { type: "string" } }, required: ["nodename"] } }];
await agent.runAgent({ runId, tools }, subscriber);
```

- At most 16 tools, names of 1 to 64 letters, digits, underscores or
  hyphens, not a tool of the agent or of its MCP, descriptions of at most
  4 KiB, parameters of at most 64 KiB.
- When the model calls one, the run ends with `outcome`
  `{"type": "success", "pendingToolCallIds": […]}`: run the calls, under
  your own consent rules, then answer each in the next run with a tool
  message, `{role: "tool", toolCallId, content}`, or with `error` set for a
  failure. `HttpAgent` sends the tool messages you add.
- A prompt instead of the answers abandons the calls. The calls wait 10
  minutes, after which the answers are refused, 410 `client_tools_expired`.
- Declare your tools in every run: a tool absent from a run is answered to
  the model as no longer offered.

## 7. Capabilities

`GET /v1/{backend}/capabilities` answers the AG-UI `AgentCapabilities` of
the runs: streaming, the tools of the client, the interrupts with edits of
the arguments, in `custom.feedback` whether the turns take a feedback, and
in `custom.models` that the runs choose their model among those
`GET /v1/{backend}/models` lists, by `id` and `label`, the default first; see
the [API](api.md#models).
`HttpAgent` does not read them; a subclass exposes them:

```javascript
class OpenSVCAgent extends HttpAgent {
  async getCapabilities() {
    return (await fetch(this.url.replace(/\/conversations\/[^/]+\/runs$/, "/capabilities"), { headers: this.headers })).json();
  }
}
```

## 8. Operations

- **Feedback.** Rate a completed turn with
  `PUT …/turns/{turnId}/feedback`, the `turnId` of `RUN_FINISHED`, unless
  the capabilities give `custom.feedback` false: the operator turned it off,
  and the feedback routes return 403 `feedback_disabled`; see the
  [API](api.md#feedback).
- **Errors.** A request refused before the stream answers an HTTP error
  with a JSON body, `{"error": {"code", "message"}}`; once the stream
  started, a failure is a `RUN_ERROR`.
- **One run at a time.** A conversation runs one turn at a time: 409
  `conversation_busy` otherwise.
- **Proxies.** The stream sends `: keepalive` every 10 s while nothing else
  is written, and each write must reach the client within 15 s: a proxy
  must not buffer the stream.
- **No resumption.** A stream cut off is not resumed: the turn fails, and
  is not stored. Read the conversation again.
