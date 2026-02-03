# Improvement List (Tracked)

Status legend: [DONE] implemented, [PARTIAL] implemented but needs tuning, [TODO] not started, [RISK] not recommended/too risky now.

## Core Context/Quality
1. [DONE] Small-doc full-file injection for small files (focused docs now eligible even without new upload; multi-doc capped). Test: attach a <50KB code file or focus an existing small file and confirm full-file context appears in prompt and answers include global structure.
2. [PARTIAL] Tree-sitter chunking for code files + line-range citations (small-doc). Test: ingest a code repo and verify chunks align to functions; in small-doc mode sources show line ranges. Note: requires tree_sitter_languages to be installed; otherwise falls back to block chunker.
3. [SKIP] Chunk summaries as metadata for retrieval (1-sentence per chunk). (Not pursuing now.)
4. [DONE] Hybrid retrieval (dense + keyword) with boosts. Test: query an exact identifier and verify lexical hits appear in sources.ask_with_context
5. [DONE] Locality-biased windows (neighbor chunk stitching). Test: retrieve a chunk in the middle of a file and confirm adjacent context appears.
6. [DONE] Persist LTM to SQLite (lightweight) so facts survive restarts. Test: restart app, ask a follow-up, and confirm LTM recall.
7. [DONE] Versioned sources per assistant regeneration version (sources stored per version and shown for active version). Test: regenerate with different context and verify sources match the active version.

## Chat Editing & Regeneration
8. [DONE] Assistant regeneration via `skip_user_message` + `target_assistant_id` (updates assistant message versions in SQLite). Test: regenerate and verify SQLite `messages.content_json.versions` updates.
9. [DONE] Persist edited user messages in SQLite + resync KV. Test: edit a user message, reload chat, confirm transcript reflects edits and KV rebuild uses updated text.
10. [DONE] Provide safe rollback on regeneration failure (UI + backend). Test: cancel regen mid-stream, ensure previous version is restored.
11. [DONE] Respect `skip_user_message` when committing KV (avoid duplicate user turns). Test: regenerate the same assistant reply twice and inspect KV/session transcript for duplicated user turns.
12. [DONE] Unify message IDs between UI and backend (or return backend IDs to UI). Test: regenerate a brand-new assistant message before reload and confirm versioning updates the correct SQLite row.

## Scope/Files/Docs
13. [DONE] Focus/selection/doc-pane scope resolution for RAG readiness. Test: selection forces single-file scope; doc-pane closed uses most recent file.
14. [DONE] Per-chat doc page edits stored in SQLite and used for search. Test: edit a doc, then search within it and ensure edited text is indexed.
15. [TODO] Raw-large fallback for huge text files + window search quality tuning. Test: huge file over raw_large threshold; verify window search is precise.

## Reliability & Performance
16. [PARTIAL] Clean KV mode with ephemeral context pack (dirty run) + clean commit. Test: verify context pack does not appear in saved transcript; KV snapshots persist.
17. [TODO] Optional KV reuse when scope/context pack unchanged (risk of prompt mismatch). Test: repeated same-query turns should reuse KV safely without drift.

## Not Recommended (for now)
- [RISK] Full entity-graph memory: heavy schema/maintenance cost vs current needs.
- [RISK] Always injecting full files: can overflow context and slow generation for large docs.

---

# Architecture + Standout Roadmap (Local RAG)

## Architecture Upgrades
1. [TODO] Model Orchestrator with pluggable adapters (llama.cpp, MLC, vLLM, OpenAI-compat local servers, remote). Test: swap chat model without restart; verify capability registry updates.
2. [TODO] Model Profiles (chat/embedding/rerank) with hot-swap and background pre-warm. Test: change embed model and observe retrieval without UI downtime.
3. [TODO] Background Job Pipeline for ingestion + downloads + indexing with progress + retry. Test: UI stays responsive while a large ingestion runs.
4. [TODO] Unified Retrieval Service (dense, BM25, hybrid, re-rank) with standardized evidence bundle. Test: mixed keyword + semantic query produces stable evidence ordering.
5. [TODO] Context Engine strategy router (full-file, focused, compare, raw-large). Test: same query on different file sizes routes to expected strategy.
6. [TODO] Storage split: SQLite for UI transcript + metadata; vector store for chunks; optional entity store; KV for inference only. Test: transcript reload does not require KV.
7. [TODO] Offline Evaluation Harness (retrieval + answer quality). Test: run a local suite and diff results across model upgrades.

## Standout Features (Beyond Typical Local UIs)
8. [TODO] "Full-file brain" for small docs with explicit mode indicator. Test: 20KB code file answers include global structure.
9. [TODO] Editable knowledge pages with provenance and retrieval honoring edits. Test: edit a doc, then ask; answer reflects edits.
10. [TODO] Multi-doc compare mode with structured output + citations. Test: two specs compared; differences cited.
11. [TODO] Retrieval explainability (why chunk selected). Test: UI shows dense vs keyword vs neighbor contributions.
12. [TODO] Per-project memory (preferences + decisions) stored locally. Test: restart; memory persists and affects responses.
13. [TODO] Local tool workflows (git diff, repo map, test runner hints). Test: request test summary; get command + output guidance.
14. [TODO] Multimodal indexing (OCR + audio + images). Test: search text inside a screenshot.
15. [TODO] One-click reindex + integrity audit (SQLite/Qdrant sync). Test: repair missing chunks and confirm counts match.



The biggest quality win is not new systems — it’s making the existing small‑doc strategy and hybrid retrieval fire more often and more intelligently. The rest (Tree‑Sitter chunking, dynamic hybrid weights, locality bias) brings the “ChatGPT‑like” jump without overhauling your architecture.
