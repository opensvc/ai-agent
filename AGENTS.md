# ai-agent: context for coding agents

## Purpose and scope

This project is a standalone Go daemon that orchestrates LLM turns and
authenticated OpenSVC MCP tool calls for cluster diagnostics. It exposes an
HTTPS API used by the OpenSVC webapp, with persistent conversations, under one route prefix per backend (`/v1/om3`, and `/v1/oc3` when the
collector MCP is configured). It does not issue OpenSVC tokens or implement daemon operations.

Keep the agent independent from the om3 binary, a local daemon, the MCP
implementation and any particular LLM provider. OpenSVC semantics and factual
data collection belong to MCP; orchestration and diagnostic reasoning belong
to the agent and model.

The canonical Go module is `github.com/opensvc/ai-agent`, including when
working in a fork. The binary remains `opensvc-ai-agentd`. Use the Go standard
library, the MCP Go SDK, and SQLite through `database/sql` and
`modernc.org/sqlite`. The project is licensed under Apache-2.0; preserve
attribution notices.

## Before editing

- Inspect the working tree and preserve unrelated user changes.
- Read the relevant implementation, [README](README.md) and
  [client contract](docs/api.md) before changing behavior.
- Identify the responsible layer and preserve its boundary. Prefer focused
  changes to existing capabilities over new frameworks or abstractions.
- Check actual API, MCP and provider contracts; do not invent compatibility,
  permissions or persistence guarantees.
- Changes to the agent API or SSE contract may affect the separate webapp
  client. Coordinate such changes explicitly; do not modify other projects
  implicitly.

## Architecture

- `cmd/opensvc-ai-agentd`: composition root, HTTPS listener and lifecycle.
  Construct shared clients, the orchestrator and conversation service here,
  not inside request handlers.
- `internal/config`: environment parsing and startup validation for the
  process, LLM, MCP, agent limits and conversation storage.
- `internal/api`: versioned HTTP routes and their OpenAPI document
  (`openapi.json`, served at `GET /openapi.json`; update it with any route,
  body, error code or stream event change), the backend-neutral authentication
  middleware, SSE, concurrency admission, stable public errors and structured
  audit events. It knows no backend header: each backend's `Authenticator`
  declares and reads its own. Keep handlers thin.
- `internal/auth`: verified identity and private request-scoped credentials,
  including the headers a backend forwards to its MCP; remove authentication
  data from LLM contexts.
- `internal/mcpclient`: backend-neutral local Unix socket transport, whoami
  transport, request-scoped MCP sessions and bounded tool discovery/results.
  It forwards only the headers its backend declares, and the request ID of
  `internal/requestid` as `X-Request-ID`, to join the MCP audit lines with
  those of the agent.
- `internal/backend/om3` and `internal/backend/oc3`: one package per
  backend, with its request headers, whoami contract, MCP client and system
  prompt. Keep backend rules in its package, never as backend switches in
  shared code.
- `internal/agent`: provider-neutral turn loop, history validation and
  sequential execution of model-requested MCP tools, with the system prompt
  its backend gives it. A tool call
  that MCP annotates destructive (`destructiveHint: true`, or not read-only
  without the hint) never runs on the model's request alone: the turn is
  suspended until the user confirms or rejects it through the API, and the
  decision applies to the exact stored call.
- `internal/llm`: neutral model contracts and events. Protocol adapters under
  `responses`, `chatcompletions` and `messages` implement these contracts;
  the neutral package must not import its adapters.
- `internal/llmfactory`: client construction selected by protocol, not by
  provider brand or model name.
- `internal/conversation`: ownership, turn lifecycle, retention and storage
  interface. `internal/conversation/sqlite` owns the current embedded schema,
  transactions and provider-neutral message encoding.

A backend is one product the agent serves: `om3`, always, and `oc3` when its
MCP socket is configured. The composition
root builds, per backend, its MCP client, its authenticator, an agent with its
system prompt and a conversation service bound to its name; the API registers
its routes under `/v1/{backend}`. The route alone selects the backend: never choose it from a
request header, body or token. The LLM models, storage, limits and audit are
shared; conversations stay bound to their backend. The operator declares the
models in the environment, and the composition root builds their clients; a
run chooses one by its ID, never by a URL, header or credential of the
request, and a turn keeps its model until it ends.

Each turn opens and closes its own authenticated MCP session.
Conversations must not retain MCP sessions or provider-specific state.

## Authentication and trust

- Treat the bearer token as opaque. Bound its size and reject missing,
  duplicate or malformed Authorization headers and query-string tokens.
  Do not decode JWTs or duplicate OpenSVC token rules: MCP owns profile
  selection, signature, issuer, audience, claims, algorithms, dates, target
  coherence and catalogue routing; the daemon or collector then confirms the
  identity and enforces grants.
- The om3 backend requires one bounded `X-OpenSVC-Cluster-ID` header and reads
  an optional `X-OpenSVC-Node` header, without interpreting them against token
  claims. It forwards them unchanged; the cluster ID selects the daemon that
  confirms the identity.
- The oc3 backend requires the user's OpenID Connect access token and no
  other header, and forwards none to the oc3 MCP.
- Reach each MCP only through its local Unix socket
  (`OPENSVC_AI_OM3_MCP_SOCKET`, `OPENSVC_AI_OC3_MCP_SOCKET`). MCP accepts
  only OpenID Connect access tokens there, never on an HTTPS listener. Do not
  add a network fallback.
- Authenticate through the MCP's `GET /mcp/auth/whoami` bridge on that socket.
  The MCP verifies the signature, issuer and audience itself, then relays the
  token to the daemon or collector `GET /api/auth/whoami`, which confirms the
  identity. Daemon-issued tokens are refused.
  Require a bounded JSON response with complete identity fields and a future
  expiry. This returned identity is authoritative; for om3, the requested
  cluster must match the response. Apply the returned expiry to the operation deadline.
- Complete authentication before reading prompts, accessing conversations or
  starting an SSE response. Invalid tokens return 401; unavailable verification
  returns 503. No offline fallback or local verification keys. The only cache
  is the in-memory whoami cache of `internal/mcpclient` (`identity_cache.go`):
  an MCP answer is reused at most a minute and never past its expiry, a
  refusal five seconds, an unavailable verification never; keys are hashes of
  the token and its MCP headers, and the cache holds identities, never tokens.
- Delegate the unchanged token and, for om3, the cluster/node headers to MCP
  in private request context; hide all from LLM contexts. Never retain it in a
  shared client or global state. Daemon or collector grants remain
  authoritative.
- Keep OpenSVC and provider credentials separate. Never place OpenSVC JWTs,
  identities or grants in LLM contexts, prompts, tool arguments or provider
  requests; preserve cancellation and deadlines when removing authentication.
- Never follow MCP redirects or use environment proxies on the socket.
- Browser CORS is opt-in through `OPENSVC_AI_CORS_ALLOWED_ORIGINS`: exact
  origins or a standalone `*`. Handle preflights before JWT authentication;
  real operations still authenticate. Never enable automatic browser credentials
  or derive CORS trust from JWT claims or client targets. Preserve streaming
  interfaces and put CORS headers on allowed-origin errors as well as successes.
- Never expose credentials in logs, errors, API responses or stored history.
  Audit records contain identifiers, tool names, counters, durations and stable
  codes, not prompts, model text, tool arguments/results or raw upstream errors.

## Turn and conversation invariants

- Preserve bounds on history, iterations, tool calls, arguments, results,
  catalogs, streams and concurrent turns. Keep discovered tools request-scoped;
  do not introduce a caller-independent catalog that bypasses visibility.
- Functional MCP tool errors can return to the model; MCP transport failures
  stop the turn. Do not invent daemon data or hide authorization failures.
- Protocol adapters translate wire events into neutral events, bound I/O,
  disable redirects and set provider storage to false where the protocol
  supports it; Chat Completions leaves the field out only when
  `OPENSVC_AI_LLM_STORE=omit` asks it, for a provider that refuses it; that
  provider's retention policy then applies, and the agent logs the omission at
  startup. Keep
  provider-specific parsing and completion semantics out of the orchestration
  loop.
- Bind all conversation operations to their backend and to verified
  `issuer` and `subject`, and `cluster_id` for a backend bound to clusters. Foreign and missing conversation IDs share the same public error.
- Serialize turns per conversation without holding a database transaction
  during LLM or MCP work. Commit complete messages before terminal completion;
  failed, canceled and interrupted turns must not replay partial output.
- Persist only the bounded provider-neutral history needed for later turns.
  Never persist JWTs, grants, provider credentials, system prompts or audit data.
  Keep the local SQLite directory and file owner-only; preserve atomic writes,
  retention, expiry and interrupted-turn recovery.
- Preserve the SSE event contract. Reject invalid requests before streaming;
  after streaming begins, emit stable generic errors, not raw provider failures,
  tool payloads or model chain-of-thought.
- Completion delivery can fail after persistence. Do not silently retry a turn
  or promise idempotency without an explicit API and storage design.
- On shutdown, stop admission, drain within the configured deadline, propagate
  cancellation to LLM/MCP, then close storage.

## Coding and change discipline

- Favor explicit Go and the standard library. Use `context.Context` for I/O,
  wrap errors with `fmt.Errorf` and `%w`, and format changes with `gofmt`.
- Keep types beside their owner and wire shapes private to protocol adapters.
  Avoid unnecessary interfaces, reflection, goroutines and frameworks.
- Add dependencies, protocols or operational scope only for an explicit need.
  Do not add migration or compatibility machinery without a requirement.
- Keep binaries, generated files, environment secrets and credentials out of
  Git. Keep documentation aligned with behavior and installation.
- Keep edits scoped to the request. Commit or push only when requested.
