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

## Models

The current model list is not finalised for the final release. Expect models to not work, be removed, or be added.

| ID | Context |
| --- | --- |
| `deepseek-v4-flash-0731` | 1M |
| `deepseek-v4-pro-0813` | 1M |
| `gemma-4-26b-a4b-it` | 262k |
| `gemma-4-31b-it` | 131k |
| `glm-5.2` | 1M |
| `glm-5.3` | 1M |
| `gpt-oss-120b` | 131k |
| `inkling` | 1M |
| `inkling-small` | 524k |
| `kimi-k3` | 1M |
| `minimax-m3` | 1M |
| `mistral-small-2603` | 262k |
| `muse-glimmer-30b` | 131k |
| `nemotron-3.5-lightning` | 1M |
| `qwen3.6-27b` | 262k |
| `qwen3.8-27b` | 262k |
| `qwen3.8-2.4t-a95b` | 1M |

## Endpoints

- `POST /v1/chat/completions` — chat completions. `stream: true` for SSE. Supports `tools`, `tool_choice`, and `reasoning_effort`.
- `GET  /v1/models` — list models.
- `GET  /v1/models/{id}` — retrieve one model.
- `GET  /health` — liveness plus live credential-pool stats.
- `POST /admin/reload` — reload `credentials.json` from disk without restarting.

`/chat/completions` and `/api/v1/chat/completions` are accepted as aliases of `/v1/chat/completions`.
