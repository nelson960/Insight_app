# Technical Decisions

This file captures major engineering tradeoffs in Insight.

## Why Qdrant + lexical fallback?

- Dense vectors are strong for semantic similarity.
- Lexical/rule-based search helps exact terms, IDs, and log-like text.
- Hybrid retrieval improves practical recall while keeping latency manageable.

## Why ephemeral CONTEXT PACK injection?

- Retrieved evidence is useful for one turn, but harmful if persisted long-term.
- Ephemeral injection keeps chat history cleaner and reduces context contamination.
- It also makes compaction and budgeting behavior more predictable.

## Why ripgrep raw-large mode?

- Very large files (logs, giant text dumps) are expensive to fully ingest/embed.
- ripgrep + contextual windows gives fast evidence retrieval with low overhead.
- This keeps UX responsive when files are too large for normal ingestion.

## Why ingestion gating (HTTP 409)?

- Answering before embedding/indexing is complete causes low-quality answers.
- Explicit gating makes the system fail clearly instead of hallucinating confidently.

## Why sidecar Python engine in a Tauri app?

- Python ML stack integration is straightforward (llama.cpp bindings, ONNX, extractors).
- Tauri remains lightweight for desktop shell/UI.
- Sidecar process isolates failures and simplifies backend lifecycle management.

## Why dual interfaces (IPC + localhost HTTP)?

- Desktop users need full RAG/session behavior via IPC.
- External tools need a simple local inference API.
- Same model/runtime, different access patterns.

## Why not an OpenAI-compatible API as a primary goal?

- Product focus is local workspace quality, not API surface parity.
- A narrow API keeps implementation simpler and behavior explicit.
- Avoids promising compatibility that can mask model/runtime differences.
