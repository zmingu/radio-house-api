# radio-house-api

An OpenAI-compatible local server.

## Quick start

```powershell
pip install -r requirements.txt
python rh_server.py
```

Or with uvicorn:

```powershell
uvicorn rh_server:app --host 127.0.0.1 --port 8002
```

In docker:

```powershell
copy .env.example .env      # optional, but keeps both sides on one config
copy proxy.json.example proxy.json
docker compose up -d --build
```

## Configuration

One `.env` configures both a local run and the container. compose loads it for the
container through `env_file`, and `rh_env.py` loads the same file for `uvicorn
rh_server:app` — a setting used to reach only the deployment, which is how the two
sides quietly drifted apart. A real environment variable still wins over the file, so
`RH_PROMPT_TOKEN_BUDGET=8000 uvicorn ...` overrides it for one run.

Copy `.env.example` to `.env` for the full list. Nothing in it is required; every
setting has a default, and the defaults are the same in both places.

| | |
| --- | --- |
| `RH_ANON_CREDENTIALS` | anonymous credentials to bootstrap when `credentials.json` holds none (default 8) |
| `RH_PROMPT_TOKEN_BUDGET` | upstream input ceiling, estimated tokens |
| `RH_PROXY_*` | egress proxy, overriding the matching `proxy.json` field |
| `HOST_PORT`, `TZ` | compose only |

`credentials.json`, `proxy.json` and `.env` are gitignored and never enter the image —
compose bind-mounts the first two at runtime. `credentials.json` must exist on the
docker host before `up`, or docker creates a directory with that name; an empty pool
(`{"credentials": []}`) is fine and is the normal state.

## Models

The current model list is not finalised for the final release. Expect models to not work, be removed, or be added.

`GET /v1/models` reports `web_search` per model. When it is on, the site runs its own web
search server-side and injects the pages it fetched into the prompt — measured on one
identical request: 4892 prompt tokens and $0.0073 with it on, against 2440 and $0.00022 with
it off, a 33x cost difference. That injection also crowds out the emulated tool spec and
pushes the request toward the 413 ceiling, so a model with `web_search: false` is the better
choice for tool-heavy work. It cannot be switched off per request — the site's own UI sends
only `model` and `effort` — so the model is the only control. Of the current catalogue only
`qwen3.8-flash` has it off, and it is what a request that names no model gets. A model the
caller does name is never swapped.

| ID | Context |
| --- | --- |
| `deepseek-v4-flash-0731` | 1M |
| `deepseek-v4-pro-0813` | 1M |
| `gemma-4-26b-a4b-it` | 256k |
| `gemma-4-31b-it` | 256k |
| `glm-5.2` | 1M |
| `glm-5.3-flash` | 1M |
| `gpt-oss-120b` | 128k |
| `inkling-small` | 524k |
| `minimax-m3` | 1M |
| `muse-glimmer-30b` | 131k |
| `nemotron-3.5-lightning` | 262k |
| `qwen3.8-27b` | 262k |
| `qwen3.8-flash` | 1M |

## Endpoints

- `POST /v1/chat/completions` — chat completions. `stream: true` for SSE. Supports `tools`, `tool_choice`, and `reasoning_effort`.
- `GET  /v1/models` — list models.
- `GET  /v1/models/{id}` — retrieve one model.
- `GET  /health` — liveness plus live credential-pool, egress and prompt-budget stats.
- `POST /admin/reload` — reload `credentials.json` and `proxy.json` from disk without
  restarting. Also clears cooldowns and revives retired credentials, so a pool taken out by
  a burst of rate limits recovers without a container restart.
- `POST /admin/egress/check` — report the egress IP each credential actually leaves from.

`/chat/completions` and `/api/v1/chat/completions` are accepted as aliases of `/v1/chat/completions`.

## Fitting the upstream's size limit

tryingopen.com refuses oversized submissions with HTTP 413 ("There's too much text in this
chat now"), a ceiling far below the models' advertised context. Prompts are shaped to fit
before sending, in three escalating steps — cheapest first:

1. **squeeze** — collapse indent runs, blank-line pileups and trailing spaces. Lossless.
2. **elide** — cut the middle out of oversized messages, keeping head *and* tail, with a
   `[... N characters elided ...]` marker. The end of a log or diff usually carries the
   conclusion, so tail-truncating would throw away the part that matters.
3. **drop** — discard the oldest turns, noting how many. The first message is never
   dropped: system instructions are folded into it.

Sizing is CJK-aware. Chinese runs about one token per character where latin text runs four
characters per token, so a plain `len()/4` underestimates Chinese prompts several-fold.
Attachments are counted at their wire length, since a base64 data URI costs real payload.

If the upstream still says 413, the prompt is reshaped smaller and retried on the same
route — no other route or cookie has a bigger ceiling, so cycling through them only burns
credits. The refused size is remembered, so later requests pre-trim instead of spending
another rejected round trip rediscovering the same limit. Once trimming bottoms out the
call returns `400 context_length_exceeded` naming what was dropped.

`RH_PROMPT_TOKEN_BUDGET` sets the starting budget (default 16000 estimated tokens).
`GET /health` reports the budget in effect and the size that was refused, if any.

## Tool calling

The upstream chat UI has no function-calling API, so `tools` is emulated: the tool schemas
go up as an instruction and a JSON reply is parsed back into `tool_calls`.

Three things this has to get right:

- **The spec is appended after the prompt is shaped, against a reserved token slice.**
  Shaping it together with the conversation let the trim cut the JSON format line and the
  tool names out of the middle, which turns tool calling off without any error — the model
  knows tools exist but not how to call them. It also lands at the *end* of the
  conversation, next to the request it governs, rather than folded into the first message.
- **A parsed name must match a declared tool.** Otherwise an answer that merely contains
  JSON — a `package.json`, a Kubernetes manifest — is read as a call to a tool named
  `nginx`, the real answer is discarded and the client gets a call it cannot execute.
  Names are matched exactly, case-insensitively, and with a `functions.` prefix stripped.
- **Tool results are relabelled to a role the upstream understands.** The UI protocol has
  only `user` and `assistant`; a `role: "tool"` message is dropped in transit, so the model
  never sees the result and calls the same tool again. Results go up as
  `[tool result: <name>]` on a user turn, and consecutive same-role turns are merged.

With `tool_choice: "required"` or a named function, a reply that contains no call is asked
once more, explicitly, before giving up — a client waiting for `tool_calls` would otherwise
break on the prose. With `"auto"`, calling remains the model's decision.

Large tool sets degrade rather than truncate: full JSON schema, then signatures, then bare
names, whichever fits `TOOL_SPEC_MAX_TOKENS`. A spec cut off mid-JSON reads as a broken
instruction and the model stops emitting calls entirely.

## Egress proxy

Upstream calls can leave through a forward proxy, addressed as **platform** (which node pool)
plus **account** (what a stable egress IP is pinned to). This is the credential convention
[Resin](https://github.com/Resinat/Resin) uses: username `Platform.Account`, password the proxy token.

Copy `proxy.json.example` to `proxy.json`:

```json
{
  "enabled": true,
  "url": "http://10.0.0.104:2260",
  "token": "your-proxy-token",
  "platform": "Default",
  "sticky": "credential"
}
```

With no `proxy.json` and no `RH_PROXY_*` variables set, everything goes direct as before.
`http`, `https`, `socks5` and `socks5h` URLs all work.

### Sticky modes

| `sticky` | Egress IP is pinned per | Use when |
| --- | --- | --- |
| `credential` (default) | each upstream cookie | you want the site to keep seeing one cookie from one IP |
| `client` | each downstream caller | callers should be isolated from each other |
| `fixed` | the whole instance | one IP for everything, named by `account` |
| `none` | nothing | rotate freely inside the platform |

### Per-request override

Callers can pick an egress identity per request:

```bash
curl http://10.0.0.104:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Egress-Platform: JP' \
  -H 'X-Egress-Account: user_tom' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"hi"}]}'
```

`X-Resin-Platform` / `X-Resin-Account` are accepted as aliases. Set `trust_headers: false` to
ignore these headers, so downstream clients cannot choose their own egress.

### Pinning specific credentials

`bindings` maps a credential id to a fixed identity, outranking the sticky mode:

```json
"bindings": {
  "cred-abc123": "JP.user_amy",
  "cred-def456": { "platform": "HK", "account": "user_bob" }
}
```

Precedence is request header → binding → sticky mode → config default.

### Verifying it works

```bash
curl -X POST http://10.0.0.104:8002/admin/egress/check
```

Each credential is probed and its real egress IP reported, with `distinct_ips` summarising the
spread — that is how you confirm stickiness is doing what you expect. Config changes need no
restart: edit `proxy.json`, then `POST /admin/reload`.

Every `RH_PROXY_*` environment variable (`_URL`, `_TOKEN`, `_PLATFORM`, `_ACCOUNT`, `_STICKY`,
`_ENABLED`, `_TRUST_HEADERS`, `_VERIFY_TLS`) overrides the matching file field. Empty counts as
unset. A proxy failure returns `502 egress_proxy_unavailable` and is never charged against the
credential's failure budget.

## Credential pool

The site issues cookies to anyone, so the pool needs no real accounts — but it does need
more than one entry. Under `sticky: credential` each credential draws its own egress IP,
and the site's rate limit is per IP, so a single credential stacked every request onto one
address and a burst of 429s looked like a broken deployment. An empty or missing
`credentials.json` therefore bootstraps `RH_ANON_CREDENTIALS` anonymous entries
(default 8) with stable ids `anon-1`…`anon-N` — stable because the sticky egress tag is
derived from the credential id, so reusing the names keeps each one's IP across restarts.

Bootstrapped entries are never written to the file: the file is for real cookies, and
writing the synthetic pool into it would turn config-derived state into hand-edited
state. Put real cookies in it and they are used instead of the bootstrap.

Without the egress proxy this setting changes nothing worth having — every credential
leaves from the same IP regardless.

## Credential health

A credential is retired only on a hard rejection — 401/403, or three unrelated failures
inside ten minutes. Two things explicitly do **not** retire one:

- **Rate limits.** A 429 means "later", not "never", so it sets a cooldown honouring
  `Retry-After` (floored at 15s, capped at 300s) and does not count toward retirement.
  Counting them was enough for a burst of 429s to take the whole service down until a
  restart, since retirement is otherwise permanent.
- **Egress and site-wide faults.** A dead proxy or a site outage says nothing about the
  cookie, so both are recorded as neutral.

Failure counts decay after ten minutes, so unrelated failures spread over hours no longer
add up to a retirement. A retired credential stays in the pool marked depleted rather than
being removed — selection skips it either way, and removing it meant the next write to
`credentials.json` saved only the survivors, eroding the file one entry per retirement.

`GET /health` shows each credential's failures, cooldown and last error. `POST
/admin/reload` re-reads the credential and proxy files, clears cooldowns and depletion,
and resets the prompt budget — a remembered 413 otherwise shrinks the budget for the rest
of the process lifetime, and a smaller budget squeezes the tool spec, so recovering from
one used to need a restart.
