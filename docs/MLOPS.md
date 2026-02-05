# MLOps Guide

Insight uses an artifact contract to make local RAG behavior reproducible.

## Contract files

- `configs/app.yaml` — runtime behavior knobs (embedding, chunking, retrieval, prompt policy)
- `configs/models.yaml` — model artifact registry + SHA256 checksums
- `configs/profiles/{dev,prod}.yaml` — profile overrides

## Install script dependencies

```bash
python3 -m pip install -r backend/requirements-ci.txt
```

## Run core checks (recommended)

```bash
./scripts/mlops/run_core.sh --workspace storage --allow-missing-artifacts
```

## Strict pre-release check

```bash
./scripts/mlops/run_core.sh --workspace ~/.insight --verify-hashes
```

## Individual commands

```bash
python3 scripts/mlops/validate_contract.py --workspace storage --allow-missing-artifacts --skip-hashes
python3 scripts/mlops/export_effective_contract.py --workspace storage --allow-missing-artifacts
python3 eval/run.py --dataset eval/datasets/smoke.jsonl --min-recall-at-k 0.66 --min-grounding-pass-rate 0.66
```

## Refresh model checksums

```bash
python3 scripts/mlops/compute_checksums.py --yaml \
  models/Qwen2.5-7B-Instruct_q4_0.gguf \
  ~/.insight/em_models/nomic-embed-text/onnx/model.onnx \
  ~/.insight/em_models/nomic-embed-text/tokenizer.json
```

## Effective contract output

Core checks write:

- `storage/contracts/effective_contract_latest.json` (dev workspace)
- `~/.insight/contracts/effective_contract_latest.json` (packaged runtime workspace)

This file includes:

- `contract_id` (reproducibility fingerprint)
- effective runtime settings from UI/config
- artifact existence/hash verification results
- validation status

## API endpoint

The backend exposes:

- `GET /settings/contract?verify_hashes=false&persist=true&allow_missing_artifacts=false`
