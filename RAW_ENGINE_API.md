# Raw Engine Server — API Reference

Stateless local LLM server (streaming + non-streaming + embeddings + logs).  
OpenAI-ish schema, single fixed model, no sessions, no memory.

Base URL (default):
```
http://127.0.0.1:11435
```

Auth (optional):
- If `INSIGHT_ENGINE_TOKEN` is set, include header `x-insight-token: <token>` on all requests.
- If `INSIGHT_ENGINE_HOST` is set to a non-loopback address without a token, the server will force `127.0.0.1`.

If the configured port is busy, the server will try the next available port and print the fallback to stderr.

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
  "prompt_renderer": "qwen2",
  "embeddings_ready": true,
  "embedding_model": "nomic-embed-text-v1.5",
  "embedding_path_present": true,
  "chat_busy": false,
  "embeddings_busy": false
}
```

**Example**
```bash
curl -s http://127.0.0.1:11435/health | jq
```

---

## 2) `GET /v1/model/info`

Full model metadata (same fields shown in Settings).

**Response**
```json
{
  "ok": true,
  "model": {
    "name": "DeepSeek R1 Distill Llama 8B",
    "architecture": "llama",
    "size_label": "8B",
    "file_type": 2,
    "quantization_version": 2,
    "ctx_train": 131072,
    "ctx_runtime": 32768,
    "n_layer": 32,
    "n_head": 32,
    "n_head_kv": 8,
    "n_embd": 4096,
    "rope_type": "yarn",
    "rope_freq_base": 500000,
    "vocab_size": 128256,
    "tokenizer_model": "gpt2",
    "kv_cache_gib": 4.0,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "prompt_renderer": "minja:default",
    "chat_template_name": "default",
    "path": "/models/model.gguf"
  }
}
```

**Example**
```bash
curl -s http://127.0.0.1:11435/v1/model/info | jq
```

---

## 3) `POST /v1/chat/completions`

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

## 4) `POST /v1/embeddings`

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

## 5) `GET /v1/logs/recent?limit=50`

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

## 6) `GET /v1/logs/{id}`

Returns a single log record.

**Example**
```bash
curl -s http://127.0.0.1:11435/v1/logs/req_abc | jq
```

**Errors**
- 404 `log_not_found`

---

## 7) `DELETE /v1/logs`

Clears the raw engine log file.

**Response**
```json
{
  "ok": true
}
```

**Example**
```bash
curl -X DELETE http://127.0.0.1:11435/v1/logs
```

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
