# Building a macOS `.dmg` (Tauri + bundled Python engine)

This repo is structured as:

- `backend/`: Python FastAPI engine (IPC via stdin/stdout; no HTTP required in desktop mode)
- `insight/`: Tauri + React desktop app

## Goals

- Package a macOS installable `.dmg` via Tauri.
- Bundle the Python engine as a **sidecar executable** (no Conda/venv required for end users).
- Store all runtime data under `~/.insight` (SQLite, Qdrant, cache, logs, KV sessions, embedding assets).

## 1) Build the Python sidecar (one-time per architecture)

Tauri copies sidecar binaries specified in `insight/src-tauri/tauri.conf.json` → `bundle.externalBin`.

This project expects a sidecar named like:

- `insight/src-tauri/bin/insight-engine-aarch64-apple-darwin` (Apple Silicon)
- `insight/src-tauri/bin/insight-engine-x86_64-apple-darwin` (Intel)

Tauri uses the pattern `name-{target-triple}` for `externalBin`.

### Option A (recommended): PyInstaller `--onefile`

From the repo root:

1. Install PyInstaller in your build Python:
   - `python -m pip install pyinstaller`

2. Build the engine sidecar:
   - Apple Silicon:
     - `pyinstaller --noconfirm --clean --onefile --name insight-engine backend/engine.py`
     - `cp dist/insight-engine insight/src-tauri/bin/insight-engine-aarch64-apple-darwin`
   - Intel (on an Intel Mac, or via cross-build tooling):
     - `pyinstaller --noconfirm --clean --onefile --name insight-engine backend/engine.py`
     - `cp dist/insight-engine insight/src-tauri/bin/insight-engine-x86_64-apple-darwin`

Notes:
- If PyInstaller misses dynamic libs (e.g. `llama_cpp`), add `--collect-submodules llama_cpp` or a `.spec` file.
- The embedding ONNX assets are downloaded on first run into `~/.insight/em_models/...` (not bundled).

## 2) Build the `.dmg` with Tauri

From the repo root:

- `cd insight`
- `pnpm install`
- `pnpm tauri build --bundles dmg --config src-tauri/tauri.conf.bundle.json`

"""
pyinstaller --noconfirm --clean --onefile \
  --name insight-engine \
  --collect-all onnxruntime \
  --collect-all llama_cpp \
  backend/engine.py

"""

Notes:
- Don’t use `TAURI_CONFIG=...` to point at a file: `TAURI_CONFIG` is an inline JSON *merge patch* used internally by the Tauri CLI/build script.
- `src-tauri/tauri.conf.json` is kept sidecar-free so `tauri dev` works without requiring the bundled engine binary.
- `--config src-tauri/tauri.conf.bundle.json` merges the sidecar settings (`bundle.externalBin`) into the default config for release builds.

Artifacts end up under `insight/src-tauri/target/release/bundle/`:
- Distribute the `.dmg` (single installer file).
- The `.app` is an intermediate bundle used to create the `.dmg`.

## 3) Runtime storage location

The desktop shell sets `INSIGHT_WORKSPACE_DIR` for the engine and uses the same directory for UI-side listing/watching:

- `~/.insight/`
  - `db.sqlite`
  - `qdrant/`
  - `kv_sessions/`
  - `cache/`
  - `logs/`
  - `em_models/`

## 4) Local testing (without a `.dmg`)

You can test the same behavior by setting the workspace dir explicitly:

- `INSIGHT_WORKSPACE_DIR=~/.insight-dev cargo tauri dev`
