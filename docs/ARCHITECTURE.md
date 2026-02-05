# Architecture

This document is the engineering deep dive for Insight.

## Tech Stack

Tauri (Rust) • React/TypeScript • FastAPI (IPC-only) • llama.cpp (GGUF) • Qdrant • ONNX Runtime (Nomic embeddings) • SQLite • Local HTTP inference API (llama.cpp-based)

## Key Technical Ideas

- **Dual-access LLM architecture:** Same swappable GGUF model layer accessible via (1) IPC for desktop app with full RAG, (2) HTTP API for external tools
- **Drop‑in models (GGUF):** Works with any GGUF that embeds a valid chat template. No hard‑coded templates per model.
- **Clean KV architecture:** Retrieved chunks injected as one-time **CONTEXT PACK**, never persisted to chat history → reduces cross-turn contamination
- **KV session snapshotting:** llama.cpp state serialized per chat → instant resume without replaying prompts
- **RAG-ready enforcement:** `/chat` returns **HTTP 409** until embeddings/indexing complete → reduces low-quality answers during ingestion (observed in early prototypes)
- **Scope-aware retrieval:** Focused vs all-documents mode tied to UI state → enables multi-document queries when user selects broader scope
- **Raw-file fallback:** ripgrep-based line search for large text/log files without ingestion overhead

## System Prompt Policy (Why “No System Info”)

Insight does **not** rely on a heavy, permanent system prompt. Instead:

- **Model templates come from GGUF metadata** (minja renderer) — no hardcoded template strings.
- **Per‑turn instructions** (RAG rules, evidence policy, and selection context) are injected as an **ephemeral CONTEXT PACK** and never persisted in chat history.
- This keeps **model behavior consistent across different GGUFs** and avoids system‑prompt drift during compaction.

## System Architecture

The system is built around a **swappable local LLM backend** accessible via two interfaces:

1. **Internal IPC path** (desktop app with full RAG): Tauri → IPC → Orchestrator → LlamaSessionManager → llama.cpp
2. **External HTTP path** (local llama.cpp inference API): Tools → localhost:11435 → RawEngine → llama.cpp

Both paths share the same GGUF model layer, enabling **hot-swappable models** via environment variables or settings UI.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GGUF Model Layer (Swappable)                       │
│   Any GGUF model (Qwen2.5, Llama 3.1, Mistral, DeepSeek, GPT‑OSS, etc.)      │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           llama.cpp (KV + inference)                         │
└─────────────────────────────────────────────────────────────────────────────┘
          ▲                                       ▲
          │                                       │
┌─────────┴───────────────┐              ┌─────────┴─────────────────────────┐
│ Path 1: Desktop IPC      │              │ Path 2: Local HTTP API            │
│ (full RAG + UI)          │              │ Raw Engine Server (127.0.0.1)     │
└─────────┬───────────────┘              └─────────┬─────────────────────────┘
          │                                       │
┌─────────▼──────────────────────────────────────┐ │
│ Desktop UI (Tauri + React)                     │ │
│  - ChatWindow, DocumentsPane, SettingsModal    │ │
│  - State: chatUiStore (Zustand)                │ │
│  - API: engine.ts (IPC client wrappers)        │ │
└─────────┬──────────────────────────────────────┘ │
          │ Tauri IPC Commands (JSON over stdio)    │
┌─────────▼──────────────────────────────────────┐ │
│ Rust Tauri Layer (lib.rs, engine.rs)           │ │
│  - EngineProcess spawns Python sidecar         │ │
│  - StdoutRouter routes by request_id           │ │
│  - Direct SQLite access for session listing    │ │
└─────────┬──────────────────────────────────────┘ │
          │ stdin/stdout (JSON lines)               │
┌─────────▼──────────────────────────────────────┐ │
│ Python Engine (engine.py - IPC wrapper)         │ │
│  - RequestManager (anyio TaskGroup)             │ │
│  - StdoutWriter (thread-safe JSON)              │ │
│  - IPC allowlist enforcement                    │ │
└─────────┬──────────────────────────────────────┘ │
          │ FastAPI HTTP (ASGI)                    │
┌─────────▼──────────────────────────────────────┐ │
│ FastAPI Application (app.py)                   │ │
│  - x-insight-ipc header check                   │ │
│  - Routers: chat, files, settings, docs, search │ │
│  - Dependency container bound per app           │ │
└─────────┬──────────────────────────────────────┘ │
          │                                       │
          │                    ┌──────────────────▼──────────────────────┐
          │                    │ Raw Engine HTTP (uvicorn)               │
          │                    │  - /chat (text generation)              │
          │                    │  - /embeddings (optional)               │
          │                    └─────────────────────────────────────────┘
          │
┌─────────▼──────────────────────────────────────────────────────────────────┐
│ Ingestion + Planning + Storage                                              │
│  - Scheduler, Extractors, Chunker, Embedder                                 │
│  - Orchestrator (scope control, context budgeting, compaction)              │
│  - Retrieval (Qdrant dense + lexical + raw large-file windows)              │
│  - LlamaSessionManager (multi-session KV, snapshot/restore, rendering)      │
│  - Storage: SQLite, Qdrant, Files (disk)                                    │
└─────────┬──────────────────────────────────────────────────────────────────┘
          │
┌─────────▼─────────────────────────────────────────────────────────────┐
│ Local Storage (~/.insight/)                                            │
│  - SQLite (messages, files, metadata)                                  │
│  - Qdrant (vector embeddings, insight_chunks, insight_memories)        │
│  - KV Sessions (llama.cpp state snapshots)                             │
│  - ONNX models (Nomic Embed Text v1.5)                                 │
│  - Engine logs (raw_engine_server requests)                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Components

1. **Document Ingestion Pipeline**: File parsing (PDF with pytesseract OCR for scanned documents, plaintext), chunking with overlap, and background embedding using Nomic Embed Text v1.5 via ONNX Runtime
2. **Retrieval (Dense + Lexical + Raw-file fallback)**: Combines dense vector retrieval (Qdrant, K=12) with lexical search (DocSearchService) and raw file server (ripgrep-based for large text/log files)
3. **Context Assembly**: Retrieval orchestration that respects document scope (focused vs all-documents mode) and enforces RAG-ready state before generation
4. **Dual LLM Access**:
   - **LlamaSessionManager**: Multi-session KV cache management for desktop app (compaction, snapshots, resume)
   - **RawEngine**: Stateless single-chat API server for external HTTP access
5. **Desktop Shell**: Tauri-based desktop app with React frontend providing canvas-based note-taking and document management

## Design Choices

- **Dense retrieval + lexical fallback**: Semantic similarity via Nomic embeddings combined with keyword/substring search improves both recall and exact-match coverage
- **Raw file server for large text/logs**: ripgrep-based search with contextual windowing avoids ingestion overhead for massive files (e.g., 100MB+ logs)
- **llama.cpp over full Python inference**: GGUF quantization enables running 7-8B parameter models on 16GB RAM machines with interactive latency
- **Sidecar Python engine**: Tauri bundles Python via PyInstaller as an external binary, avoiding complex native module compilation while keeping all logic local
- **IPC-only FastAPI backend**: The Python engine rejects direct HTTP requests and only accepts IPC calls from the Tauri shell (enforced via `x-insight-ipc` header)
- **Ephemeral RAG context**: Retrieved document chunks are injected as a one-time system message ("CONTEXT PACK") rather than persisted into the chat history, preventing context pollution across turns
- **KV cache snapshotting**: Session state is serialized to disk (`~/.insight/kv_sessions/*.kv`) using llama.cpp's state API, enabling instant chat resumption without replaying prompts
- **RAG-ready enforcement**: The backend returns HTTP 409 when `/chat` is called before document ingestion completes, reducing low-quality answers during ingestion

## Raw Engine Server (Local HTTP Inference API)

The raw engine server provides Path 2 access to the swappable local LLM backend via a lightweight HTTP inference API on localhost.

- **Endpoints:** `/v1/chat/completions`, `/v1/embeddings`
- **Streaming:** Server-Sent Events supported
- **Bind address:** `127.0.0.1:11435` (localhost-only)
- **Auth (optional):** `INSIGHT_ENGINE_TOKEN` + `x-insight-token`
- **Lifecycle:** managed via Settings UI

### Usage examples

```bash
curl http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

```bash
curl http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain RAG"}],
    "stream": true
  }'
```

## Model Requirements

- **Supported LLMs (GGUF):**
  - Qwen2.5 7B Instruct (recommended)
  - Llama 3.1 8B Instruct
- **Memory:** 16GB RAM recommended for 8B models

## Results Snapshot

### Latency

| Metric | Target                             | Hardware    | Model             | Notes                      |
| ------ | ---------------------------------- | ----------- | ----------------- | -------------------------- |
| TTFT   | sub-second TTFT (tuning-dependent) | Apple M1/M2 | Qwen2.5 7B Q4_K_M | GPU layers enabled, 8K ctx |
| tok/s  | >35/s                              | Apple M1/M2 | Qwen2.5 7B Q4_K_M |                            |

### Retrieval behavior

- Dense retrieval + lexical fallback
- OCR integration for scanned PDFs
- Raw-file fallback for huge logs/text
- Scope-aware retrieval (focused vs all-documents)

### Reliability notes

- Ingestion gating reduced early, low-quality answers
- Ephemeral context injection reduced cross-turn contamination

## Evaluation Method

- Manual eval on **N≈30** curated queries across single-doc, multi-doc, quote lookup, summarization
- Judged on correctness, grounding, completeness, hallucination rate
- Retrieval sanity checks under focused vs all-doc scope

## Privacy

All inference, embeddings, and retrieval run locally. No document contents are sent to external services.
