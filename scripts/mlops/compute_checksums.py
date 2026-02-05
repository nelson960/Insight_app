#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute SHA256 checksums for model artifacts.")
    parser.add_argument("paths", nargs="+", help="File paths to hash")
    parser.add_argument(
        "--relative-to",
        default=None,
        help="Optional base directory used when printing paths",
    )
    parser.add_argument(
        "--yaml",
        action="store_true",
        help="Print YAML fragments suitable for configs/models.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failed = False
    base: Path | None = None
    if args.relative_to:
        base = Path(args.relative_to).expanduser().resolve()

    for raw in args.paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists() or not path.is_file():
            print(f"ERROR: not a file: {path}", file=sys.stderr)
            failed = True
            continue

        digest = sha256_file(path)
        display = str(path)
        if base is not None:
            try:
                display = str(path.relative_to(base))
            except ValueError:
                display = str(path)

        if args.yaml:
            print(f"- path: \"{display}\"")
            print(f"  sha256: \"{digest}\"")
        else:
            print(f"{display}  {digest}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
