from __future__ import annotations

import atexit
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from backend.runtime_utils import is_packaged


_UVICORN_RE = re.compile(r"Uvicorn running on https?://([^:]+):(\d+)")
_FALLBACK_RE = re.compile(r"falling back to (\d+)")

logger = logging.getLogger(__name__)


class RawEngineServerManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen[str]] = None
        self._logs: Deque[Dict[str, Any]] = deque(maxlen=250)
        self._requested_host = "127.0.0.1"
        self._requested_port = 11435
        self._host = self._requested_host
        self._port = self._requested_port
        self._log_dir = self._resolve_log_dir()
        self._command = "python -m backend.raw_engine_server"
        self._started_at: Optional[float] = None
        self._exit_code: Optional[int] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._starting = False
        self._start_error: Optional[str] = None
        self._start_requested_at: Optional[float] = None
        atexit.register(self.stop)

    def _build_command(self) -> List[str]:
        if is_packaged():
            return [sys.executable, "--raw-server"]
        return [sys.executable, "-m", "backend.raw_engine_server"]

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def _resolve_log_dir(self) -> Path:
        return Path(os.getenv("INSIGHT_LOG_DIR") or (Path.home() / ".insight" / "engine_logs"))

    def _record_line(self, line: str) -> None:
        line = line.strip("\n")
        if not line:
            return
        entry = {"ts": time.time(), "line": line}
        with self._lock:
            self._logs.append(entry)
            uvicorn_match = _UVICORN_RE.search(line)
            if uvicorn_match:
                self._host = uvicorn_match.group(1)
                try:
                    self._port = int(uvicorn_match.group(2))
                except Exception:
                    pass
                logger.info("Raw server ready on %s:%s", self._host, self._port)
            fallback_match = _FALLBACK_RE.search(line)
            if fallback_match:
                try:
                    self._port = int(fallback_match.group(1))
                except Exception:
                    pass
                logger.warning("Raw server port fallback to %s", self._port)
        if "Traceback" in line or "Error" in line or "ERROR" in line:
            logger.warning("Raw server log: %s", line)

    def _drain_output(self, process: subprocess.Popen[str]) -> None:
        stream = process.stdout
        if stream is None:
            return
        for line in stream:
            self._record_line(line)
        try:
            exit_code = process.poll()
        except Exception:
            exit_code = None
        with self._lock:
            self._exit_code = exit_code
            if exit_code is not None and exit_code != 0:
                self._start_error = f"Raw server exited with code {exit_code}"
        if exit_code is None:
            return
        if exit_code == 0:
            logger.info("Raw server exited cleanly")
        else:
            logger.error("Raw server exited with code %s", exit_code)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            running = bool(self._process and self._process.poll() is None)
            pid = self._process.pid if self._process and running else None
            if self._process and not running:
                try:
                    self._exit_code = self._process.poll()
                except Exception:
                    pass
            return {
                "ok": True,
                "running": running,
                "starting": bool(self._starting),
                "pid": pid,
                "host": self._host,
                "port": self._port,
                "requested_host": self._requested_host,
                "requested_port": self._requested_port,
                "log_dir": str(self._log_dir),
                "command": self._command,
                "started_at": self._started_at,
                "start_requested_at": self._start_requested_at,
                "exit_code": self._exit_code,
                "error": self._start_error,
            }

    def logs(self, limit: int = 250) -> List[Dict[str, Any]]:
        limit = max(1, min(limit, 250))
        with self._lock:
            return list(self._logs)[-limit:]

    def start(self, env_overrides: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        cmd: List[str] | None = None
        cmd_str: str | None = None
        with self._lock:
            if self._process and self._process.poll() is None:
                return self.status()
            if self._starting:
                return self.status()
            self._starting = True
            self._start_error = None
            self._start_requested_at = time.time()

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            if env_overrides:
                for key, val in env_overrides.items():
                    if val is None:
                        continue
                    env[str(key)] = str(val)
            requested_host = env.get("INSIGHT_ENGINE_HOST", "127.0.0.1")
            try:
                requested_port = int(env.get("INSIGHT_ENGINE_PORT", "11435"))
            except Exception:
                requested_port = 11435
            self._requested_host = requested_host
            self._requested_port = requested_port
            self._host = requested_host
            self._port = requested_port
            self._log_dir = self._resolve_log_dir()
            cmd = self._build_command()
            if is_packaged():
                env["INSIGHT_ENGINE_MODE"] = "raw"
            cmd_str = " ".join(cmd)
            self._command = cmd_str

        if cmd_str:
            self._record_line(f"[manager] starting raw server: {cmd_str}")
            logger.info(
                "Raw server start requested host=%s port=%s command=%s",
                requested_host,
                requested_port,
                cmd_str,
            )

        def _worker() -> None:
            try:
                proc = subprocess.Popen(
                    cmd or [sys.executable, "-m", "backend.raw_engine_server"],
                    cwd=str(self._project_root()),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
            except Exception as exc:
                with self._lock:
                    self._process = None
                    self._exit_code = None
                    self._started_at = None
                    self._starting = False
                    self._start_error = str(exc)
                self._record_line(f"[manager] raw server failed to start: {exc}")
                logger.exception("Failed to start raw server: %s", exc)
                return

            with self._lock:
                self._process = proc
                self._exit_code = None
                self._started_at = time.time()
                self._starting = False

            self._reader_thread = threading.Thread(
                target=self._drain_output,
                args=(proc,),
                daemon=True,
                name="raw-engine-log-drain",
            )
            self._reader_thread.start()

        threading.Thread(target=_worker, daemon=True, name="raw-engine-start").start()
        return self.status()

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            proc = self._process
            if self._starting and (proc is None or proc.poll() is not None):
                self._starting = False
        if not proc or proc.poll() is not None:
            return self.status()

        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass
        with self._lock:
            self._exit_code = proc.poll()
            self._starting = False
        return self.status()


_RAW_ENGINE_MANAGER: Optional[RawEngineServerManager] = None


def raw_engine_manager() -> RawEngineServerManager:
    global _RAW_ENGINE_MANAGER
    if _RAW_ENGINE_MANAGER is None:
        _RAW_ENGINE_MANAGER = RawEngineServerManager()
    return _RAW_ENGINE_MANAGER


__all__ = ["RawEngineServerManager", "raw_engine_manager"]
