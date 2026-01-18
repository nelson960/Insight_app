from __future__ import annotations

import json
from typing import Any


def sse_data(payload: str) -> bytes:
    return f"data: {payload}\n\n".encode("utf-8")


def sse_json(obj: Any) -> bytes:
    return sse_data(json.dumps(obj, ensure_ascii=False))
