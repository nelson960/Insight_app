# Insight — Privacy-First Local RAG with On-Device LLM Inference

**Local-first desktop RAG assistant**: document ingestion → dense retrieval → context assembly → **llama.cpp streaming generation**, designed for **privacy**, **low latency**, and **limited memory** on consumer hardware.

**Use case:** domain-specific QA over local documents (research, writing, sensitive files) **without cloud APIs**.

## TL;DR

- Runs fully offline: **ONNX embeddings + Qdrant retrieval + llama.cpp inference**
- Optimized for responsiveness: **~<500ms TTFT** (Qwen2.5 7B Q4_K_M, Apple M1/M2), **streaming output**
- Built-in safeguards: **RAG-ready gating (HTTP 409)** + **ephemeral "CONTEXT PACK" injection** to prevent context pollution
- Fast resume: **KV-cache snapshots** persisted per chat to avoid prompt replay

## Value Proposition

Insight is a **local research workspace**, not just a chat UI.  
It’s built for **parallel thinking** (cards + links), **strict grounding** (selection‑first + scope control),  
and **clean context** (one‑time CONTEXT PACK + KV resume) so answers stay tight and explainable.

## Tech Stack

Tauri (Rust) • React/TypeScript • FastAPI (IPC-only) • llama.cpp (GGUF) • Qdrant • ONNX Runtime (Nomic embeddings) • SQLite • Local HTTP inference API (llama.cpp-based)

## Highlights

- **Systems + ML:** retrieval quality ↔ latency/memory tradeoffs, context budgeting, multi-session KV cache
- **Applied research:** failure modes + mitigation (scope control, compaction drift, large-file fallback)
- **Production awareness:** isolation across chats/files (chat_id/file_id scoping), IPC-only backend, cancellation via native aborts

Recent hardening fixes after a full review:

- **SQLite safety:** thread‑local connections to prevent cross‑thread corruption
- **Upload safety:** size enforcement before read + concurrency backpressure
- **Ingestion reliability:** cancellation race fixed; jobs now respect cancel signals
- **LLM stability:** generation timeouts + abort callback to avoid hangs
- **IPC security:** allowlist path normalization to prevent traversal
- **Lifecycle sanity:** app‑scoped dependency container + clean shutdown hooks

## Demo

<p align="center">
  <img src="pics/Chat.png" width="900" alt="Compare & explain across two documents with scope control" />
</p>
<p align="center"><b>Compare & explain across two documents with scope control</b></p>

<br/>

<p align="center">
  <img src="pics/Canvas.png" width="900" alt="Canvas workspace with linked cards" />
</p>
<p align="center"><b>Canvas workspace with linked cards</b></p>

<br/>

<p align="center">
  <img src="pics/Raw Server.png" width="900" alt="Settings: local Raw API server running on localhost" />
</p>
<p align="center"><b>Settings: local Raw API server running on localhost</b></p>

## Product Features

- Selection-first + document-grounded QA: highlight text from files or model responses to ask follow-ups with precise grounding; supports focused/all-doc scope and compare mode.

- Transparent performance + trust HUD: real-time context window usage, remaining input budget, tok/s, TTFT, retrieval/context-pack visibility, and reliable streaming cancel.

- Canvas workspace + memory: branching/linkable cards with optional per-card attachments + export/share, plus KV snapshot/compaction and optional LTM-style recall to stay coherent in long chats.

---

## Canvas Workspace (Card-Based UI)

Insight’s canvas is a **spatial workspace** built around **cards**. Each card is an independent chat + document context you can place on the board, so you can branch ideas without losing state.

- **Card = workspace:** a single card pairs chat with focused documents for that thread.
- **Linked cards:** visually connect cards to show how ideas branch or relate.
- **Parallel threads:** keep multiple investigations open side‑by‑side without mixing context.
- **Context isolation:** each card maintains its own conversation state and context budget.

This is designed for research and analysis workflows where you want to explore multiple directions in parallel while keeping grounding tight per card.

---

## Beyond Retrieval

Insight is RAG‑enabled, but it’s designed as a **local, stateful workspace** rather than a single retrieval pipeline:

- **Canvas‑native workflow:** cards represent parallel investigative threads, not a single chat stream.
- **Session state + KV snapshots:** per‑chat llama.cpp state is persisted for fast resume (not just re‑prompting).
- **Context policy:** retrieved chunks are injected as a one‑time **CONTEXT PACK**, never saved to history.
- **Dual access paths:** full desktop RAG via IPC + a separate local HTTP API for raw, stateless inference.
- **Scope control:** focused vs all‑documents retrieval is first‑class in the UI, not a hidden backend toggle.

RAG is one ingredient; the product is a **local research workspace** with persistent state, branching, and strict context hygiene.

---

## Results (Local Hardware)

### Latency

| Metric | Target                             | Hardware    | Model             | Notes                      |
| ------ | ---------------------------------- | ----------- | ----------------- | -------------------------- |
| TTFT   | sub-second TTFT (tuning-dependent) | Apple M1/M2 | Qwen2.5 7B Q4_K_M | GPU layers enabled, 8K ctx |
| tok/s  | >35/s                              | Apple M1/M2 | Qwen2.5 7B Q4_K_M |                            |

### Retrieval Behavior

- **Dense retrieval + lexical fallback:** Qdrant dense retrieval (K=12) with keyword/substring search for large files and exact-match lookups
- **OCR integration:** pytesseract-based OCR for scanned PDFs
- **Raw-file fallback (logs/huge text):** for files marked `raw_large=True`, uses ripgrep + contextual windows to return evidence without embedding/ingestion
- **Scope + small-doc behavior:** focused vs all-documents scope is UI-controlled; small docs (<90% ctx) inject full text directly for maximum grounding

### Reliability

- Gating generation until ingestion completes reduced "confident wrong answers" observed in early prototypes
- Ephemeral context injection reduced citation/hallucination carryover across turns

### Qualitative Examples

**Success case (focused mode):**

- **Query:** "What is the return policy for electronics?"
- **Context:** User viewing "returns-policy.pdf" in documents pane
- **Result:** Correct answer extracted from focused document

**Success case (multi-hop with scope):**

- **Query:** "Compare the revenue growth mentioned in the Q2 report with the projections from the strategic plan"
- **Context:** User switches to "all-documents" scope
- **Result:** System retrieves from both documents, enabling cross-document synthesis

**Failure case (context overflow):**

- **Query:** Long conversation history + large retrieved chunks
- **Behavior:** Automatic compaction summarizes earlier turns; retrieved context trimmed head-first
- **Observation:** Summarizer occasionally loses nuanced details from early conversation

---

## Evaluation (Method)

- Manual evaluation on **N≈30** curated queries across {single-doc, multi-doc, quote lookup, summarization}
- Judged on: correctness, grounding, completeness, hallucination rate
- Retrieval sanity checks: inspected top-K chunk relevance and failure modes under focused vs all-docs scope

---

## Privacy

All inference, embeddings, and retrieval run locally. No document contents are sent to external services.

**Non-goal:** competing with frontier cloud models; the goal is private, offline, interactive document assistance.
**Non-goal:** providing a drop-in OpenAI-compatible API; the focus is tight integration with llama.cpp-backed models.

---

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

---

## Project Status

**Current state:** Production-ready onedir build for macOS (Apple Silicon)
**Supported:** macOS (Apple Silicon ARM64)
**Distribution:** Standalone .app bundle (no installer required)
**Demo:** [Add GIF/Loom/screenshot here showing chat streaming + doc focus switch]

---

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

---

### Core Components

1. **Document Ingestion Pipeline**: File parsing (PDF with pytesseract OCR for scanned documents, plaintext), chunking with overlap, and background embedding using Nomic Embed Text v1.5 via ONNX Runtime
2. **Retrieval (Dense + Lexical + Raw-file fallback)**: Combines dense vector retrieval (Qdrant, K=12) with lexical search (DocSearchService) and raw file server (ripgrep-based for large text/log files)
3. **Context Assembly**: Retrieval orchestration that respects document scope (focused vs all-documents mode) and enforces RAG-ready state before generation
4. **Dual LLM Access**:
   - **LlamaSessionManager**: Multi-session KV cache management for desktop app (compaction, snapshots, resume)
   - **RawEngine**: Stateless single-chat API server for external HTTP access
5. **Desktop Shell**: Tauri-based desktop app with React frontend providing canvas-based note-taking and document management

---

### Design Choices

- **Dense retrieval + lexical fallback**: Semantic similarity via Nomic embeddings combined with keyword/substring search improves both recall and exact-match coverage
- **Raw file server for large text/logs**: ripgrep-based search with contextual windowing avoids ingestion overhead for massive files (e.g., 100MB+ logs)
- **llama.cpp over full Python inference**: GGUF quantization enables running 7-8B parameter models on 16GB RAM machines with interactive latency
- **Sidecar Python engine**: Tauri bundles Python via PyInstaller as an external binary, avoiding complex native module compilation while keeping all logic local
- **IPC-only FastAPI backend**: The Python engine rejects direct HTTP requests and only accepts IPC calls from the Tauri shell (enforced via `x-insight-ipc` header)
- **Ephemeral RAG context**: Retrieved document chunks are injected as a one-time system message ("CONTEXT PACK") rather than persisted into the chat history, preventing context pollution across turns
- **KV cache snapshotting**: Session state is serialized to disk (`~/.insight/kv_sessions/*.kv`) using llama.cpp's state API, enabling instant chat resumption without replaying prompts
- **RAG-ready enforcement**: The backend returns HTTP 409 when `/chat` is called before document ingestion completes, reducing low-quality answers during ingestion

### Raw Engine Server (Local HTTP Inference API)

The **raw engine server** provides Path 2 access to the swappable local LLM backend via a lightweight HTTP inference API on localhost. This is how external tools (editors, IDEs, CLI scripts) access the local LLM without using the desktop app.

**Model scope (v1):** the HTTP API currently supports llama.cpp-backed models (e.g., Llama and Qwen families). The API surface is intentionally minimal and model-specific rather than a generic OpenAI replacement.

**Core concept:** Same swappable GGUF model layer, different interface. Models changed via `INSIGHT_ENGINE_MODEL_PATH` are immediately available to both the desktop app (IPC) and external tools (HTTP).

- **Endpoints:** `/v1/chat/completions` (text generation), `/v1/embeddings` (optional, llama.cpp/ONNX-backed)
- **Streaming:** Server-Sent Events supported
- **Bind address:** `127.0.0.1:11435` (localhost-only, no network exposure)
- **Auth (optional):** set `INSIGHT_ENGINE_TOKEN` and pass `x-insight-token: <token>`
- **Lifecycle:** managed via Settings UI (start/stop/status)
- **Model access:** shares the same GGUF model layer as the desktop app
- **Chat templates:** auto-detected from GGUF metadata (e.g., Llama 3, Qwen-style templates)

<details>
<summary>Usage examples</summary>

**Basic chat completion:**

```bash
curl http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

**Streaming:**

```bash
curl http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain RAG"}],
    "stream": true
  }'
```

**Switching models:**

```bash
# Stop server in Settings UI
export INSIGHT_ENGINE_MODEL_PATH=/path/to/qwen2.5-7b.gguf
# Start server in Settings UI
# Model is now available to both desktop app AND HTTP API
```

</details>

---

### Model Requirements

- **Supported LLMs** (GGUF format):
  - Qwen2.5 7B Instruct (recommended)
  - Llama 3.1 8B Instruct
- **Memory**: Minimum 16GB RAM recommended for 8B models

<details>
<summary>Storage layout (~/.insight)</summary>

All data is stored under `~/.insight/`:

- `db.sqlite`: User messages, file metadata, chat-file associations
- `qdrant/`: Vector database (collections: `insight_chunks`, `insight_memories`)
- `kv_sessions/`: llama.cpp KV cache snapshots per chat
- `em_models/`: ONNX embedding model weights
- `cache/`: Temporary file processing artifacts
- `logs/`: Application logs

</details>

---

## Repo Layout

- `insight/` — Tauri + React desktop app
- `backend/` — FastAPI IPC engine, retrieval + llama.cpp orchestration
  - `services/raw_engine_server/` — Standalone local HTTP inference server (SSE)
- `ONEDIR_BUILD_GUIDE.md` — build notes, packaging
- `RAW_ENGINE_API.md` — raw engine API reference
- `FRESH_MACHINE_CHECKLIST.md` — setup checklist

---

## Quick Start

### Production Build (macOS Onedir)

For production builds, use the automated build script:

```bash
./scripts/build_onedir_prod.sh
```

See [ONEDIR_BUILD_GUIDE.md](ONEDIR_BUILD_GUIDE.md) for detailed build instructions, troubleshooting, and distribution guidelines.

### Development Mode

```bash
# Frontend (React + Tauri)
cd insight
pnpm install
pnpm tauri dev

# Backend engine is spawned automatically by the Tauri dev app.
# If you need a custom interpreter, set PYTHON_BIN before running tauri dev.
# Avoid starting a second engine process or you'll hit Qdrant/SQLite locks.
# Example:
# PYTHON_BIN=/opt/miniconda3/envs/ml/bin/python pnpm tauri dev
```

---

## License

Licensed under the MIT License.

## Acknowledgments

- **llama.cpp**: George G. G. and contributors for efficient local LLM inference
- **Nomic AI**: Nomic Embed Text v1.5 model for embeddings
- **Qdrant**: High-performance vector database
- **Tauri**: Cross-platform desktop framework
- **ripgrep**: Fast regex search for raw file server
- **FastAPI & uvicorn**: HTTP API framework for raw engine server
