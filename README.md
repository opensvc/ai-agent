# ai-agent

Standalone AI agent for OpenSVC cluster diagnostics, used by the OpenSVC webapp.

## Requirements

- Go 1.25.5 or later to build.
- An OpenSVC MCP server on the same host, with its local Unix socket.
- An LLM endpoint and model.
- A TLS certificate covering the agent's hostname, with its private key.

## Install

From the repository root, on a Linux host:

```bash
go build -o bin/opensvc-ai-agentd ./cmd/opensvc-ai-agentd
sudo useradd --system --user-group --no-create-home --shell /usr/sbin/nologin opensvc-ai
sudo install -Dm755 bin/opensvc-ai-agentd /usr/local/libexec/opensvc-ai-agentd
sudo install -d -m750 -o root -g opensvc-ai /etc/opensvc-ai
sudo install -d -m700 -o opensvc-ai -g opensvc-ai /var/lib/opensvc-ai-agent
```

Place the TLS certificate at `/etc/opensvc-ai/agent.crt` and the private key
at `/etc/opensvc-ai/agent.key`. Both must be readable by `opensvc-ai`; restrict
the private key to that user.

## Configure

Create `/etc/opensvc-ai/agent-llm.env`, owned by `root:opensvc-ai` with mode `0640`:

```dotenv
OPENSVC_AI_LISTEN_ADDR=0.0.0.0:8090
OPENSVC_AI_TLS_CERT_FILE=/etc/opensvc-ai/agent.crt
OPENSVC_AI_TLS_KEY_FILE=/etc/opensvc-ai/agent.key
OPENSVC_AI_CONVERSATION_DB_PATH=/var/lib/opensvc-ai-agent/conversations.db
OPENSVC_AI_OM3_MCP_SOCKET=/run/opensvc-mcp/delegated.sock

OPENSVC_AI_LLM_PROTOCOL=responses
OPENSVC_AI_LLM_BASE_URL=https://llm.example.test/v1
OPENSVC_AI_LLM_MODEL=your-model
OPENSVC_AI_LLM_AUTH_MODE=bearer
OPENSVC_AI_LLM_API_TOKEN=replace-me
```

Replace the example values. The conversation database is created on first
start. The agent refuses a database written by another schema version: remove
the file to start with an empty one. Use `chat_completions` instead of
`responses` when required by the provider. For a provider without authentication, set
`OPENSVC_AI_LLM_AUTH_MODE=none` and omit the API token.

For the Anthropic Messages API, use:

```dotenv
OPENSVC_AI_LLM_PROTOCOL=messages
OPENSVC_AI_LLM_BASE_URL=https://api.anthropic.com/v1
OPENSVC_AI_LLM_MODEL=your-anthropic-model
OPENSVC_AI_LLM_AUTH_MODE=api_key
OPENSVC_AI_LLM_API_TOKEN=replace-me
```

`api_key` sends `x-api-key`; `bearer` remains available for Messages endpoints.
Messages supports streamed text and MCP tool calls, without extended thinking
or provider-hosted tools. Keep the API key only in the protected environment
file, never in Git or conversation history.

The agent serves the `om3` backend under `/v1/om3` and reaches its MCP only
through the local Unix socket `OPENSVC_AI_OM3_MCP_SOCKET`. It also serves the
`oc3` backend under `/v1/oc3` when `OPENSVC_AI_OC3_MCP_SOCKET` names the local
Unix socket of the collector MCP. Run each MCP on the agent host, typically as
resources of one OpenSVC service, and give the `opensvc-ai` user access to
its socket through its group. A socket need not exist when the agent starts.
Restrict network access to the agent port.

With `OPENSVC_AI_DOCS_URL`, the root of the OpenSVC book, the `om3` backend
offers the model `search_docs`, a search of the book: at startup the agent
reads its `llms.txt` and the pages under `agent/` and the keyword history it
lists, from that host only, in 60 s at most, and holds their index in
memory, about 2 MiB. A book that cannot be read leaves the backend without
the search, said in the log, the rest served. The book is read once: a book
deployed again is read at the next start of the agent. A mirror of the book
serves its own `llms.txt`; see the
[client contract](docs/api.md#documentation).

For browser clients, set `OPENSVC_AI_CORS_ALLOWED_ORIGINS` to comma-separated
webapp origins, or `*` to allow all origins. Empty by default. See the
[browser client guide](docs/webapp.md#cors) for examples and restrictions.

### Optional environment variables

| Variable | Default | Bounds |
| --- | --- | --- |
| `OPENSVC_AI_LISTEN_ADDR` | `127.0.0.1:8090` | explicit IP address and port 1-65535 |
| `OPENSVC_AI_CONVERSATION_DB_PATH` | `/var/lib/opensvc-ai-agent/conversations.db` | |
| `OPENSVC_AI_CONVERSATION_LIFETIME` | `168h` | `1h` to `8760h` |
| `OPENSVC_AI_FEEDBACK_RETENTION` | `4320h` (180 days) | `24h` to `26280h` |
| `OPENSVC_AI_CONVERSATION_MAX` | `4096` conversations for the instance | 1 to 1000000 |
| `OPENSVC_AI_CONVERSATION_MAX_PER_USER` | `128`, or the instance limit if lower | 1 to the instance limit |
| `OPENSVC_AI_CONVERSATION_MAX_DATABASE_MIB` | `256` | `32` to `1048576` |
| `OPENSVC_AI_OC3_MCP_SOCKET` | unset: `oc3` not served | absolute path |
| `OPENSVC_AI_DOCS_URL` | unset: no documentation search | `https` URL of the root of the book, such as `https://book.opensvc.com/`; one naming its `llms.txt` is its directory |
| `OPENSVC_AI_CORS_ALLOWED_ORIGINS` | empty: no CORS | exact origins or `*` |
| `OPENSVC_AI_MAX_CONCURRENT_ASKS` | `4` | 1 to 128 |
| `OPENSVC_AI_SHUTDOWN_TIMEOUT` | `30s` | `1s` to `5m` |
| `OPENSVC_AI_AGENT_MAX_ITERATIONS` | `8` | 1 to 32 |
| `OPENSVC_AI_AGENT_TIMEOUT` | `5m` | `1s` to `30m` |
| `OPENSVC_AI_TOOL_SEARCH_SCORE_RATIO` | `0.30` | `0` to `1`; `0` shows every search result |
| `OPENSVC_AI_TOOL_SEARCH_THRESHOLD` | `20` | `1` to `1024`, or `-1` to search only catalogs too large for one model request |
| `OPENSVC_AI_LLM_TIMEOUT` | `2m` | any positive duration |
| `OPENSVC_AI_LLM_MAX_OUTPUT_TOKENS` | `4096` | 1 to 131072 |
| `OPENSVC_AI_LLM_REASONING_EFFORT` | unset: the model applies its own | `none`, `minimal`, `low`, `medium`, `high` or `xhigh`, as the model accepts; `responses` and `chat_completions` protocols only |

Durations use the Go syntax, such as `90s` or `2h30m`. An invalid value
prevents startup.

## Start

```bash
sudo -u opensvc-ai sh -c '
  set -a
  . /etc/opensvc-ai/agent-llm.env
  set +a
  exec /usr/local/libexec/opensvc-ai-agentd
'
```

For managed deployments, use an OpenSVC `app.simple` resource to launch the
binary with the same environment. The agent reads environment variables, not
the environment file itself; its launcher must load that file if used.

Check health using the hostname covered by the certificate:

```bash
curl https://agent.example.test:8090/health
```

Add `--cacert /path/to/ca.pem` if the agent uses a private CA.

## Use

The [client API](docs/api.md) lists the routes of each backend and how they
authenticate. Browser clients, such as the webapp, also follow the
[browser client guide](docs/webapp.md).

## Development

```bash
go fmt ./...
go test ./...
go vet ./...
go run honnef.co/go/tools/cmd/staticcheck@v0.7.0 ./...
```

The CI runs the same checks, plus module verification and a race-enabled test
run. `internal/llm/staticcheck.conf` keeps the capital letter of the provider
API names that start its error messages.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
