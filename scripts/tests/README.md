# Insight Test Suite

These scripts run real integration checks against the packaged backend sidecar (`insight-engine`) inside `Insight.app`.

## Quick Start

```bash
# Smoke suite only (fast)
./scripts/tests/run_all_tests.sh --quick

# Full suite
./scripts/tests/run_all_tests.sh --all

# Specific suites
./scripts/tests/run_all_tests.sh --stress
./scripts/tests/run_all_tests.sh --concurrent --workers 12
./scripts/tests/run_all_tests.sh --memory --memory-duration 300
./scripts/tests/run_all_tests.sh --edge-case
```

## What Each Suite Tests

### Smoke (`smoke`)
- Sidecar binary smoketest (`INSIGHT_SMOKETEST=1`)
- Core IPC endpoints:
  - `GET /settings`
  - `GET /settings/health`
  - `GET /settings/storage`
  - `GET /settings/busy`
  - `GET /chat/session_summaries`

### Stress (`stress`)
- Real file ingestion via `POST /files/ingest_path`
- Ingestion progress polling via `GET /files/progress/{chat_id}`
- Document access via `GET /docs/page/{chat_id}/{file_id}`
- Search loop via `GET /search/doc/{chat_id}/{file_id}`
- Latency sampling (p50/p95)

### Concurrent (`concurrent`)
- Multi-threaded IPC request load on one sidecar process
- Repeated concurrent calls to:
  - `GET /settings/health`
  - `GET /chat/session_summaries`
- Failure-rate and latency summary

### Memory (`memory`)
- RSS sampling of sidecar process over time
- Request load during sampling
- Growth analysis and leak guard threshold
- CSV artifact output (`memory_samples.csv`)

### Edge (`edge`)
- Invalid payload validation checks
- Allowlist/method enforcement checks
- Invalid path ingestion checks
- Large-file policy behavior checks

## App Path

All scripts auto-detect the app in this order:
1. `dist/onedir/Insight.app`
2. `insight/src-tauri/target/release/bundle/macos/Insight.app`
3. First `Insight.app` found under `dist/`

You can always override with:

```bash
./scripts/tests/run_all_tests.sh --app /path/to/Insight.app --all
```

## Artifacts and Reports

- Master report:
  - `/tmp/insight_test_report_YYYYMMDD_HHMMSS.txt`
  - `TEST_REPORT_YYYYMMDD_HHMMSS.txt` at repo root
- Suite artifacts:
  - Temporary by default
  - Preserve with `--keep-artifacts`
  - Or choose a directory with `--artifacts-dir DIR`

## Individual Suite Scripts

Each suite has a wrapper script:

- `scripts/tests/stress_test.sh`
- `scripts/tests/concurrent_test.sh`
- `scripts/tests/memory_leak_test.sh`
- `scripts/tests/edge_case_test.sh`

All wrappers delegate to:

- `scripts/tests/engine_test_runner.py`

Example:

```bash
./scripts/tests/stress_test.sh --iterations 80 --keep-artifacts
./scripts/tests/memory_leak_test.sh --duration 600 --interval 5
```

## Build + Test Flow

```bash
./scripts/build_onedir_prod.sh --no-sign
./scripts/tests/run_all_tests.sh --all
```
