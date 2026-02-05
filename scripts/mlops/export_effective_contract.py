#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.mlops import build_effective_contract_snapshot, write_effective_contract_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export effective runtime contract snapshot from local settings DB.")
    parser.add_argument(
        "--workspace",
        default=str(Path.home() / ".insight"),
        help="Workspace root (default: ~/.insight)",
    )
    parser.add_argument(
        "--settings-db",
        default=None,
        help="Optional explicit SQLite settings DB path (default: <workspace>/db.sqlite)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional contract profile override (dev/prod)",
    )
    parser.add_argument(
        "--verify-hashes",
        action="store_true",
        help="Verify SHA256 hashes (can be slow for large model files)",
    )
    parser.add_argument(
        "--allow-missing-artifacts",
        action="store_true",
        help="Treat missing model artifacts as warnings instead of hard errors",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file path (default writes workspace contracts latest file)",
    )
    return parser.parse_args()


def load_settings(db_path: Path) -> Dict[str, Any]:
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT key, value_json FROM app_settings").fetchall()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()

    out: Dict[str, Any] = {}
    for key, raw in rows:
        if raw is None:
            continue
        try:
            out[str(key)] = json.loads(raw)
        except Exception:
            out[str(key)] = raw
    return out


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    db_path = Path(args.settings_db).expanduser().resolve() if args.settings_db else workspace / "db.sqlite"
    settings = load_settings(db_path)

    try:
        snapshot = build_effective_contract_snapshot(
            settings=settings,
            workspace=workspace,
            verify_hashes=bool(args.verify_hashes),
            allow_missing_artifacts=bool(args.allow_missing_artifacts),
            profile=args.profile,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2
    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        out = write_effective_contract_snapshot(snapshot, workspace=workspace)

    print(f"contract_id={snapshot.get('contract_id')}")
    print(f"ok={snapshot.get('validation', {}).get('ok')}")
    print(f"output={out}")
    return 0 if snapshot.get("validation", {}).get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
