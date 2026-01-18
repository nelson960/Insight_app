from __future__ import annotations

import json
import logging
import threading
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


class LogStore:
    def __init__(self, log_dir: Path, *, max_recent: int = 500) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / "raw_engine.jsonl"
        self._lock = threading.Lock()
        self._recent: Deque[Dict[str, Any]] = deque(maxlen=max_recent)
        self._index: Dict[str, Dict[str, Any]] = {}

    def record(self, entry: Dict[str, Any]) -> None:
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.write("\n")
            except Exception:
                logger.exception("raw_engine log write failed")
            self._recent.append(entry)
            entry_id = str(entry.get("id") or "")
            if entry_id:
                self._index[entry_id] = entry

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, min(limit, 500))
        with self._lock:
            if len(self._recent) >= limit:
                return list(self._recent)[-limit:]
        return self._read_recent_from_disk(limit)

    def get(self, entry_id: str) -> Optional[Dict[str, Any]]:
        entry_id = str(entry_id or "")
        if not entry_id:
            return None
        with self._lock:
            cached = self._index.get(entry_id)
            if cached:
                return cached
        return self._scan_disk(entry_id)

    def _read_recent_from_disk(self, limit: int) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        items: Deque[Dict[str, Any]] = deque(maxlen=limit)
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            logger.exception("raw_engine log read failed")
            return []
        return list(items)

    def _scan_disk(self, entry_id: str) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if str(obj.get("id") or "") == entry_id:
                        return obj
        except Exception:
            logger.exception("raw_engine log scan failed")
        return None


__all__ = ["LogStore"]
