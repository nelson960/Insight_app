# Raw Engine Server — API Reference

Stateless local LLM server (streaming + non-streaming + embeddings + logs).  
OpenAI-ish schema, single fixed model, no sessions, no memory.

Base URL (default):
```
http://127.0.0.1:11435
```

If the configured port is busy, the server will try the next available port and print the fallback to stderr.

---

## Runtime & Config

### Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `INSIGHT_ENGINE_HOST` | `127.0.0.1` | Bind host |
| `INSIGHT_ENGINE_PORT` | `11435` | Bind port |
| `INSIGHT_ENGINE_MODEL_PATH` | (none) | **Required**. GGUF model path. If unset, tries `llm_model_path` from SQLite settings. |
| `INSIGHT_ENGINE_CTX` | model default | Context length (tokens) |
| `INSIGHT_ENGINE_THREADS` | model default | CPU threads |
| `INSIGHT_ENGINE_GPU_LAYERS` | model default | GPU layers |
| `INSIGHT_ENGINE_MAX_TOKENS` | `1024` | Default output cap |
| `INSIGHT_LOG_DIR` | `~/.insight/engine_logs` | JSONL log dir |
| `INSIGHT_LOG_PROMPTS` | `0` | Store full prompts (0/1) |
| `INSIGHT_LOG_COMPLETIONS` | `0` | Store full outputs (0/1) |
| `INSIGHT_LOG_PREVIEW_CHARS` | `400` | Preview chars stored in logs |
| `INSIGHT_ENGINE_EMBEDDING_PATH` | (optional) | Optional embedding model path |
| `INSIGHT_ENGINE_EMBEDDING_MODEL` | `nomic-embed-text-v1.5` | Embed model name |
| `INSIGHT_ENGINE_EMBEDDING_AUTO_DOWNLOAD` | `1` | Auto-download embeddings |

### Runtime Notes

- Chat model selection is fixed at startup (no per-request model switching).
- Embeddings model selection is fixed at startup; request `model` is echoed in responses but not used to select a model.
- Dependencies: chat requires `llama_cpp`; embeddings require `onnxruntime` and `tokenizers` plus the model files. Auto-download needs network access.

---

# Endpoints

## 1) `GET /health`

**Purpose**: status & model info.

**Response**
```json
{
  "ok": true,
  "status": "ok",
  "model_loaded": true,
  "model": "Qwen2.5 7B Instruct",
  "uptime_sec": 123.4,
  "ctx_size": 32768,
  "prompt_renderer": "qwen2"
}
```

**Example**
```bash
curl -s http://127.0.0.1:11435/health | jq
```

---

## 2) `POST /v1/chat/completions`

OpenAI-ish chat endpoint. Supports streaming via SSE.

### Request

```json
{
  "messages": [
    {"role": "system", "content": "You are Insight."},
    {"role": "user", "content": "Hello"}
  ],
  "stream": false,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "repeat_penalty": 1.1,
  "max_tokens": 512,
  "stop": ["\n\nUser:"]
}
```

**Fields**

| Field | Type | Default | Notes |
|---|---|---|---|
| `messages` | list | required | Chat messages |
| `stream` | bool | false | SSE streaming if true |
| `temperature` | float | model default | |
| `top_p` | float | model default | |
| `top_k` | int | model default | |
| `repeat_penalty` | float | model default | |
| `max_tokens` | int | config default | Output cap |
| `stop` | list[str] | [] | Additional stop strings |

**Server behavior**

- Builds a prompt from `messages` using the engine’s prompt renderer.
- Clamps `max_tokens` to avoid exceeding context.
- Adds internal stop markers (e.g., EOS/template tokens).
- Returns **429** if the single-model lock is busy.

### Non-stream response

```json
{
  "id": "chat_2e6…",
  "object": "chat.completion",
  "created": 1768125155,
  "model": "Qwen2.5 7B Instruct",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello!"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 12,
    "total_tokens": 54
  }
}
```

### Streaming response (SSE)

Each chunk is:
```
data: {json}
```
End:
```
data: [DONE]
```

Example:
```
data: {"id":"chat_x","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"content":"Hello"}}]}
data: [DONE]
```

### Streaming example (curl)

```bash
curl -N http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Stream test"}],"stream":true}'
```

### Error responses

| Status | Error |
|---|---|
| 400 | `prompt_too_long` |
| 400 | `max_tokens_exhausted` |
| 429 | `engine_busy` |
| 500 | internal |

---

## 3) `POST /v1/embeddings`

Returns vector embeddings.

### Request

```json
{
  "input": "hello world"
}
```

or
```json
{
  "input": ["hello", "world"],
  "model": "nomic-embed-text-v1.5"
}
```

### Response

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.01, 0.02, ...]}
  ],
  "model": "nomic-embed-text-v1.5"
}
```

### Example

```bash
curl -s http://127.0.0.1:11435/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":["hello","world"]}' | jq '.data[0].embedding[:8]'
```

---

## 4) `GET /v1/logs/recent?limit=50`

Returns recent log entries.

**Params**
- `limit` (1–500), default 50

**Response**
```json
{
  "data": [
    {
      "id": "req_abc",
      "timestamp": "2026-01-10T12:00:00Z",
      "endpoint": "/v1/chat/completions",
      "stream": true,
      "status": 200,
      "prompt_tokens": 123,
      "completion_tokens": 45,
      "latency_ms": 1220
    }
  ]
}
```

**Example**
```bash
curl -s "http://127.0.0.1:11435/v1/logs/recent?limit=10" | jq
```

---

## 5) `GET /v1/logs/{id}`

Returns a single log record.

**Example**
```bash
curl -s http://127.0.0.1:11435/v1/logs/req_abc | jq
```

**Errors**
- 404 `log_not_found`

---

# Streaming Notes

- SSE uses `text/event-stream`
- Server sets:
  - `Cache-Control: no-cache`
  - `X-Accel-Buffering: no`
- Cancellation: client disconnect stops generation (finish_reason = `cancelled`)

---

# Busy / Concurrency

Chat completions are single-flight; concurrent chat requests return 429.  
Embeddings use a separate lock and can run alongside chat, but concurrent embeddings return 429.

Busy responses look like:

```json
{"error":"engine_busy"}
```

HTTP 429.

---

# Running Multiple Models

To run multiple chat models, start multiple raw-engine processes on different ports with different `INSIGHT_ENGINE_MODEL_PATH`, then call each base URL with curl.  
Each process loads its own model (higher RAM/VRAM). For different embedding models, run separate processes with different `INSIGHT_ENGINE_EMBEDDING_MODEL`/`INSIGHT_ENGINE_EMBEDDING_PATH`.

Example:
```bash
INSIGHT_ENGINE_MODEL_PATH=/models/llama3.gguf INSIGHT_ENGINE_PORT=11435 python backend/raw_engine_server.py
INSIGHT_ENGINE_MODEL_PATH=/models/qwen2.gguf INSIGHT_ENGINE_PORT=11436 python backend/raw_engine_server.py
```

---

# Prompt Rules / Rendering

The server uses its internal prompt renderer based on GGUF metadata (ChatML/Qwen or Llama‑3).  
If the model template is unknown, the server will fail to start.  
It does **not** store session memory. Every call is fully stateless.

---

# Quick CLI Test Bundle

```bash
# health
curl -s http://127.0.0.1:11435/health | jq

# non-stream
curl -s http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":false}' \
  | jq -r '.choices[0].message.content'

# stream
curl -N http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Stream test"}],"stream":true}'

# embeddings
curl -s http://127.0.0.1:11435/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":["hello","world"]}' | jq '.data[0].embedding[:8]'

# logs
curl -s http://127.0.0.1:11435/v1/logs/recent?limit=5 | jq
```
