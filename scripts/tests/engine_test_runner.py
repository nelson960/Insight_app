#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _now_ms() -> int:
    return int(time.time() * 1000)


def _chat_id(prefix: str) -> str:
    return f"{prefix}-{_now_ms()}-{uuid.uuid4().hex[:6]}"


class EngineIPC:
    """Thread-safe IPC client for insight-engine JSONL stdin/stdout protocol."""

    def __init__(
        self,
        engine_path: Path,
        cwd: Path,
        timeout_s: float = 30.0,
        *,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        self.engine_path = engine_path
        self.timeout_s = timeout_s
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        self.proc = subprocess.Popen(
            [str(engine_path)],
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        if self.proc.stdin is None or self.proc.stdout is None or self.proc.stderr is None:
            raise RuntimeError("Failed to start insight-engine IPC pipes")

        self._stdin = self.proc.stdin
        self._stdout = self.proc.stdout
        self._stderr = self.proc.stderr

        self._stdin_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: Dict[str, "queue.Queue[Dict[str, Any]]"] = {}

        self._stderr_lines: List[str] = []
        self._stderr_cap = 200
        self._stopped = threading.Event()

        self._stdout_thread = threading.Thread(target=self._read_stdout_loop, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr_loop, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    @property
    def pid(self) -> int:
        return self.proc.pid

    @property
    def stderr_tail(self) -> List[str]:
        return list(self._stderr_lines)

    def _read_stdout_loop(self) -> None:
        try:
            for raw in self._stdout:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                rid = obj.get("request_id")
                if not isinstance(rid, str) or not rid:
                    continue

                with self._pending_lock:
                    q = self._pending.get(rid)
                if q is not None:
                    q.put(obj)
        finally:
            self._stopped.set()

    def _read_stderr_loop(self) -> None:
        try:
            for raw in self._stderr:
                line = raw.rstrip("\n")
                if not line:
                    continue
                self._stderr_lines.append(line)
                if len(self._stderr_lines) > self._stderr_cap:
                    self._stderr_lines = self._stderr_lines[-self._stderr_cap :]
        finally:
            self._stopped.set()

    def request(
        self,
        endpoint: str,
        method: str = "POST",
        payload: Optional[Dict[str, Any]] = None,
        *,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        if payload is None:
            payload = {}
        rid = f"req-{_now_ms()}-{uuid.uuid4().hex[:10]}"
        msg = {
            "request_id": rid,
            "endpoint": endpoint,
            "method": method.upper(),
            "payload": payload,
            "stream": False,
        }
        q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        with self._pending_lock:
            self._pending[rid] = q

        try:
            with self._stdin_lock:
                self._stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
                self._stdin.flush()

            deadline = time.time() + (timeout_s if timeout_s is not None else self.timeout_s)
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(f"timeout waiting for response endpoint={endpoint} request_id={rid}")
                try:
                    obj = q.get(timeout=remaining)
                except queue.Empty as exc:
                    raise TimeoutError(
                        f"timeout waiting for response endpoint={endpoint} request_id={rid}"
                    ) from exc

                if "ok" in obj and "status" in obj:
                    return obj
                if obj.get("stream_error"):
                    return {
                        "request_id": rid,
                        "ok": False,
                        "status": 500,
                        "error": str(obj.get("stream_error")),
                        "data": {},
                    }
        finally:
            with self._pending_lock:
                self._pending.pop(rid, None)

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            with self._stdin_lock:
                self._stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self._stdin.flush()
        except Exception:
            pass

        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __enter__(self) -> "EngineIPC":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class SuiteContext:
    def __init__(
        self,
        app_path: Path,
        engine_path: Path,
        repo_root: Path,
        artifacts_dir: Path,
        workspace_dir: Path,
    ) -> None:
        self.app_path = app_path
        self.engine_path = engine_path
        self.repo_root = repo_root
        self.artifacts_dir = artifacts_dir
        self.workspace_dir = workspace_dir


class SuiteResult:
    def __init__(self, name: str) -> None:
        self.name = name
        self.failures: List[str] = []
        self.notes: List[str] = []

    @property
    def ok(self) -> bool:
        return not self.failures

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _detect_app_path(explicit: Optional[str], root: Path) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"App path does not exist: {p}")
        return p

    candidates = [
        root / "dist" / "onedir" / "Insight.app",
        root / "insight" / "src-tauri" / "target" / "release" / "bundle" / "macos" / "Insight.app",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    dist_dir = root / "dist"
    if dist_dir.exists():
        for c in sorted(dist_dir.rglob("Insight.app")):
            if c.is_dir():
                return c.resolve()

    raise FileNotFoundError("Could not auto-detect Insight.app. Pass --app /path/to/Insight.app")


def _find_engine_binary(app_path: Path) -> Path:
    patterns = [
        "Contents/Resources/bin/*/insight-engine",
        "Contents/Resources/bin/insight-engine",
        "Contents/MacOS/insight-engine",
    ]
    for pattern in patterns:
        matches = sorted(app_path.glob(pattern))
        for m in matches:
            if m.is_file() and os.access(m, os.X_OK):
                return m.resolve()

    raise FileNotFoundError(f"Could not find executable insight-engine inside app bundle: {app_path}")


def _run_smoketest_binary(
    engine_path: Path,
    cwd: Path,
    workspace_dir: Path,
    timeout_s: int = 180,
) -> Tuple[bool, str]:
    env = os.environ.copy()
    env["INSIGHT_SMOKETEST"] = "1"
    env["INSIGHT_WORKSPACE_DIR"] = str(workspace_dir)
    proc = subprocess.run(
        [str(engine_path)],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output[-4000:]


def _expect_ok(result: SuiteResult, resp: Dict[str, Any], label: str) -> bool:
    if not resp.get("ok"):
        result.fail(f"{label} failed status={resp.get('status')} error={resp.get('error')}")
        return False
    return True


def _expect_status(result: SuiteResult, resp: Dict[str, Any], status: int, label: str) -> bool:
    actual = int(resp.get("status") or 0)
    if actual != status:
        result.fail(
            f"{label} expected status={status} got={actual} ok={resp.get('ok')} error={resp.get('error')}"
        )
        return False
    return True


def _request_with_retries(
    client: EngineIPC,
    *,
    endpoint: str,
    method: str,
    payload: Dict[str, Any],
    timeout_s: float,
    attempts: int = 3,
    delay_s: float = 2.0,
) -> Dict[str, Any]:
    last_exc: Optional[Exception] = None
    for i in range(attempts):
        try:
            return client.request(endpoint, method=method, payload=payload, timeout_s=timeout_s)
        except Exception as exc:
            last_exc = exc
            if i + 1 < attempts:
                time.sleep(delay_s)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("request retry logic failed unexpectedly")


def _request_sessions(client: EngineIPC) -> Tuple[Dict[str, Any], str]:
    resp = client.request("/chat/session_summaries", method="GET", payload={})
    if int(resp.get("status") or 0) == 403:
        fallback = client.request("/chat/sessions", method="GET", payload={})
        return fallback, "GET /chat/sessions (fallback)"
    return resp, "GET /chat/session_summaries"


def run_smoke(ctx: SuiteContext, args: argparse.Namespace) -> SuiteResult:
    result = SuiteResult("smoke")

    ok, output = _run_smoketest_binary(ctx.engine_path, ctx.repo_root, ctx.workspace_dir)
    smoke_log = ctx.artifacts_dir / "smoketest.log"
    smoke_log.write_text(output or "", encoding="utf-8")
    if not ok:
        lowered = output.lower()
        offline_markers = ("huggingface.co", "name resolution", "maxretryerror", "temporary failure in name resolution")
        if any(m in lowered for m in offline_markers):
            result.note(f"sidecar smoketest skipped hard-fail due offline dependency; details in {smoke_log}")
        else:
            result.fail("sidecar smoketest failed")
            result.note(f"smoketest output saved at {smoke_log}")
            return result
    else:
        result.note("sidecar smoketest passed")

    with EngineIPC(
        ctx.engine_path,
        cwd=ctx.repo_root,
        timeout_s=45.0,
        env_overrides={"INSIGHT_WORKSPACE_DIR": str(ctx.workspace_dir)},
    ) as client:
        try:
            resp = _request_with_retries(
                client,
                endpoint="/settings",
                method="GET",
                payload={},
                timeout_s=90.0,
                attempts=3,
            )
        except Exception as exc:
            result.fail(f"GET /settings timed out or failed: {exc}")
            stderr_tail = "\\n".join(client.stderr_tail[-20:])
            if stderr_tail:
                result.note(f"engine stderr tail:\\n{stderr_tail}")
            return result
        if _expect_ok(result, resp, "GET /settings"):
            data = resp.get("data") or {}
            if not isinstance(data, dict) or "settings" not in data:
                result.fail("GET /settings response missing settings payload")

        resp = _request_with_retries(
            client,
            endpoint="/settings/health",
            method="GET",
            payload={},
            timeout_s=60.0,
            attempts=2,
        )
        if _expect_ok(result, resp, "GET /settings/health"):
            data = resp.get("data") or {}
            if isinstance(data, dict):
                result.note(f"health checks keys={sorted((data.get('checks') or {}).keys())[:6]}")

        resp = client.request("/settings/storage", method="GET", payload={})
        _expect_ok(result, resp, "GET /settings/storage")

        resp, label = _request_sessions(client)
        if not resp.get("ok"):
            result.note(
                f"{label} unavailable in this environment status={resp.get('status')} error={resp.get('error')}"
            )
        else:
            sessions = (resp.get("data") or {}).get("sessions") if isinstance(resp.get("data"), dict) else None
            if not isinstance(sessions, list):
                result.fail(f"{label} did not return sessions list")

        resp = client.request("/settings/busy", method="GET", payload={})
        _expect_ok(result, resp, "GET /settings/busy")

    return result


def _create_test_text_file(base: Path, name: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    p = base / name
    text = "\n".join(
        [
            "Insight integration test document.",
            "The purpose of this file is to validate ingestion and search.",
            "Keyword: ALPHA_BRAVO_CHARLIE.",
            "Line for search stability.",
        ]
        + [f"Extra line {i}: quick brown fox" for i in range(1, 80)]
    )
    p.write_text(text, encoding="utf-8")
    return p


def _wait_for_ingestion(client: EngineIPC, chat_id: str, timeout_s: int = 90) -> Dict[str, Any]:
    deadline = time.time() + timeout_s
    last: Dict[str, Any] = {}
    while time.time() < deadline:
        resp = client.request(f"/files/progress/{chat_id}", method="GET", payload={})
        last = resp
        if resp.get("ok"):
            data = resp.get("data") or {}
            total = int(data.get("total") or 0)
            done = int(data.get("done") or 0)
            active_jobs = int(data.get("active_jobs") or 0)
            if total > 0 and done >= total and active_jobs == 0:
                return resp
        time.sleep(0.5)
    return last


def run_stress(ctx: SuiteContext, args: argparse.Namespace) -> SuiteResult:
    result = SuiteResult("stress")
    loops = max(1, int(args.iterations))

    with EngineIPC(
        ctx.engine_path,
        cwd=ctx.repo_root,
        timeout_s=45.0,
        env_overrides={"INSIGHT_WORKSPACE_DIR": str(ctx.workspace_dir)},
    ) as client:
        health = client.request("/settings/health", method="GET", payload={})
        health_checks = (health.get("data") or {}).get("checks", {}) if isinstance(health.get("data"), dict) else {}
        embedding_present = bool(health_checks.get("embedding_present"))

        if not embedding_present:
            latencies_ms: List[float] = []
            for _ in range(loops):
                t0 = time.perf_counter()
                r1 = client.request("/settings/health", method="GET", payload={})
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                _expect_ok(result, r1, "GET /settings/health")

                t0 = time.perf_counter()
                r2 = client.request("/settings/storage", method="GET", payload={})
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                _expect_ok(result, r2, "GET /settings/storage")

            if latencies_ms:
                p50 = statistics.median(latencies_ms)
                p95 = sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))]
                result.note(f"latency_ms p50={p50:.1f} p95={p95:.1f} samples={len(latencies_ms)}")
            result.note("ingestion/search stress checks skipped because embedding model is unavailable")
            return result

        test_dir = ctx.artifacts_dir / "stress_inputs"
        source_file = _create_test_text_file(test_dir, "stress_input.txt")

        chat_id = _chat_id("stress")
        ingest_payload = {"chat_id": chat_id, "user_id": "test", "paths": [str(source_file)]}
        resp = client.request("/files/ingest_path", method="POST", payload=ingest_payload, timeout_s=60)
        if not _expect_ok(result, resp, "POST /files/ingest_path"):
            return result

        data = resp.get("data") or {}
        files = data.get("files") if isinstance(data, dict) else None
        if not isinstance(files, list) or not files:
            result.fail("ingest_path response missing files array")
            return result
        file_id = str(files[0].get("file_id") or "")
        if not file_id:
            result.fail("ingest_path did not return file_id")
            return result

        progress = _wait_for_ingestion(client, chat_id, timeout_s=120)
        if not _expect_ok(result, progress, "GET /files/progress/{chat_id}"):
            return result

        latencies_ms: List[float] = []
        for i in range(loops):
            t0 = time.perf_counter()
            r1 = client.request("/settings/health", method="GET", payload={})
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            _expect_ok(result, r1, "GET /settings/health")

            t0 = time.perf_counter()
            r2 = client.request(f"/search/doc/{chat_id}/{file_id}?q=ALPHA_BRAVO_CHARLIE&limit=5", method="GET", payload={})
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            if _expect_ok(result, r2, "GET /search/doc/{chat}/{file}"):
                matches = (r2.get("data") or {}).get("matches") if isinstance(r2.get("data"), dict) else None
                if not isinstance(matches, list):
                    result.fail("search response missing matches list")

            if i % 3 == 0:
                r3 = client.request(f"/docs/page/{chat_id}/{file_id}", method="GET", payload={})
                _expect_ok(result, r3, "GET /docs/page/{chat}/{file}")

        if latencies_ms:
            p50 = statistics.median(latencies_ms)
            p95 = sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))]
            result.note(f"latency_ms p50={p50:.1f} p95={p95:.1f} samples={len(latencies_ms)}")

        # Cleanup chat artifacts to keep workspace tidy.
        client.request(f"/chat/sessions/{chat_id}", method="DELETE", payload={})

    return result


def run_concurrent(ctx: SuiteContext, args: argparse.Namespace) -> SuiteResult:
    result = SuiteResult("concurrent")
    workers = max(1, int(args.workers))
    loops = max(1, int(args.iterations))

    with EngineIPC(
        ctx.engine_path,
        cwd=ctx.repo_root,
        timeout_s=45.0,
        env_overrides={"INSIGHT_WORKSPACE_DIR": str(ctx.workspace_dir)},
    ) as client:
        failures: List[str] = []
        latencies_ms: List[float] = []
        lock = threading.Lock()

        def worker(idx: int) -> None:
            for i in range(loops):
                try:
                    t0 = time.perf_counter()
                    resp = client.request("/settings/health", method="GET", payload={}, timeout_s=30)
                    dt = (time.perf_counter() - t0) * 1000.0
                    with lock:
                        latencies_ms.append(dt)
                    if not resp.get("ok"):
                        with lock:
                            failures.append(
                                f"worker={idx} step=health loop={i} status={resp.get('status')} error={resp.get('error')}"
                            )

                    resp = client.request("/settings/storage", method="GET", payload={}, timeout_s=30)
                    if not resp.get("ok"):
                        with lock:
                            failures.append(
                                f"worker={idx} step=storage loop={i} status={resp.get('status')} error={resp.get('error')}"
                            )
                except Exception as exc:
                    with lock:
                        failures.append(f"worker={idx} loop={i} exception={exc}")

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if failures:
            for item in failures[:10]:
                result.fail(item)
            if len(failures) > 10:
                result.fail(f"... plus {len(failures) - 10} more failures")

        if latencies_ms:
            p50 = statistics.median(latencies_ms)
            p95 = sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))]
            result.note(
                f"workers={workers} loops={loops} requests={len(latencies_ms)} latency_ms p50={p50:.1f} p95={p95:.1f}"
            )

    return result


def _rss_kb(pid: int) -> Optional[int]:
    try:
        out = subprocess.check_output(["ps", "-p", str(pid), "-o", "rss="], text=True).strip()
        return int(out) if out else None
    except Exception:
        return None


def run_memory(ctx: SuiteContext, args: argparse.Namespace) -> SuiteResult:
    result = SuiteResult("memory")
    duration_s = max(30, int(args.duration))
    interval_s = max(1, int(args.interval))

    with EngineIPC(
        ctx.engine_path,
        cwd=ctx.repo_root,
        timeout_s=45.0,
        env_overrides={"INSIGHT_WORKSPACE_DIR": str(ctx.workspace_dir)},
    ) as client:
        samples: List[Tuple[float, int]] = []
        start = time.time()
        loops = 0

        csv_path = ctx.artifacts_dir / "memory_samples.csv"
        with csv_path.open("w", encoding="utf-8") as fh:
            fh.write("seconds,rss_kb\n")
            while time.time() - start < duration_s:
                loops += 1
                client.request("/settings/health", method="GET", payload={})
                if loops % 2 == 0:
                    client.request("/settings/storage", method="GET", payload={})
                rss = _rss_kb(client.pid)
                elapsed = time.time() - start
                if rss is not None:
                    samples.append((elapsed, rss))
                    fh.write(f"{elapsed:.2f},{rss}\n")
                time.sleep(interval_s)

        if len(samples) < 5:
            result.note("memory RSS sampling unavailable or too sparse in this environment; skipped leak assertion")
            result.note(f"memory samples saved at {csv_path}")
            return result

        values = [v for _, v in samples if v > 0]
        if not values:
            result.note("memory RSS sampling unavailable in this environment; skipped leak assertion")
            result.note(f"memory samples saved at {csv_path}")
            return result

        first = values[0]
        last = values[-1]
        min_v = min(values)
        max_v = max(values)
        growth_pct = ((last - first) / first * 100.0) if first > 0 else 0.0

        result.note(
            "rss_mb first={:.1f} last={:.1f} min={:.1f} max={:.1f} growth_pct={:.1f} samples={}".format(
                first / 1024.0,
                last / 1024.0,
                min_v / 1024.0,
                max_v / 1024.0,
                growth_pct,
                len(values),
            )
        )
        result.note(f"memory samples saved at {csv_path}")

        # Keep threshold conservative to avoid false positives from cache warm-up.
        if growth_pct > 80.0:
            result.fail(f"potential leak: RSS growth {growth_pct:.1f}% exceeds 80% threshold")

    return result


def run_edge(ctx: SuiteContext, args: argparse.Namespace) -> SuiteResult:
    result = SuiteResult("edge")

    with EngineIPC(
        ctx.engine_path,
        cwd=ctx.repo_root,
        timeout_s=45.0,
        env_overrides={"INSIGHT_WORKSPACE_DIR": str(ctx.workspace_dir)},
    ) as client:
        health = client.request("/settings/health", method="GET", payload={})
        health_checks = (health.get("data") or {}).get("checks", {}) if isinstance(health.get("data"), dict) else {}
        embedding_present = bool(health_checks.get("embedding_present"))
        model_path_present = bool(health_checks.get("model_path"))

        resp = client.request("/chat/session_messages", method="POST", payload={})
        if int(resp.get("status") or 0) == 403:
            result.note("POST /chat/session_messages not allowed by this sidecar build; skipping this edge check")
        else:
            _expect_status(result, resp, 400, "POST /chat/session_messages missing chat_id")

        resp = client.request("/chat", method="POST", payload={})
        if int(resp.get("status") or 0) == 500 and "gguf model" in str(resp.get("error", "")).lower():
            if not model_path_present:
                result.note("POST /chat validation check skipped because no GGUF model is configured")
            else:
                result.fail(f"POST /chat returned model error unexpectedly: {resp.get('error')}")
        else:
            _expect_status(result, resp, 400, "POST /chat missing required fields")

        resp = client.request("/files/ingest_path", method="POST", payload={"chat_id": "", "paths": []})
        if int(resp.get("status") or 0) == 500 and "onnx model not found" in str(resp.get("error", "")).lower():
            if not embedding_present:
                result.note("ingest_path validation checks skipped because embedding model is not present")
            else:
                result.fail(f"ingest_path returned embedding error unexpectedly: {resp.get('error')}")
        else:
            _expect_status(result, resp, 400, "POST /files/ingest_path invalid payload")

        resp = client.request("/not_allowed", method="GET", payload={})
        _expect_status(result, resp, 403, "GET disallowed endpoint")

        resp = client.request("/settings/../health", method="GET", payload={})
        _expect_status(result, resp, 403, "GET normalized disallowed endpoint")

        resp = client.request("/settings", method="PATCH", payload={})
        _expect_status(result, resp, 403, "PATCH method disallowed by IPC allowlist")

        if embedding_present:
            bad_path = ctx.artifacts_dir / "does_not_exist.txt"
            payload = {"chat_id": _chat_id("edge"), "user_id": "test", "paths": [str(bad_path)]}
            resp = client.request("/files/ingest_path", method="POST", payload=payload)
            _expect_status(result, resp, 400, "POST /files/ingest_path nonexistent path")

            # Validate large-file policy in a chat that already has files.
            small_dir = ctx.artifacts_dir / "edge_inputs"
            small_file = _create_test_text_file(small_dir, "small.txt")
            chat_id = _chat_id("edge")

            resp = client.request(
                "/files/ingest_path",
                method="POST",
                payload={"chat_id": chat_id, "user_id": "test", "paths": [str(small_file)]},
                timeout_s=60,
            )
            if _expect_ok(result, resp, "POST /files/ingest_path small file"):
                _wait_for_ingestion(client, chat_id, timeout_s=90)

                # Default large-file threshold is 10 MiB; create a deterministic 11 MiB file.
                large_file = small_dir / "large.txt"
                with large_file.open("wb") as fh:
                    fh.write(b"a" * (11 * 1024 * 1024))

                resp2 = client.request(
                    "/files/ingest_path",
                    method="POST",
                    payload={"chat_id": chat_id, "user_id": "test", "paths": [str(large_file)]},
                    timeout_s=60,
                )
                status = int(resp2.get("status") or 0)
                if status not in {409, 413}:
                    result.fail(
                        "large-file policy test expected 409/413, got "
                        f"status={status} error={resp2.get('error')}"
                    )

            client.request(f"/chat/sessions/{chat_id}", method="DELETE", payload={})
        else:
            result.note("ingest_path edge checks skipped because embedding model is unavailable")

    return result


def _print_result(result: SuiteResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"[{status}] suite={result.name}")
    for note in result.notes:
        print(f"  - {note}")
    for failure in result.failures:
        print(f"  - ERROR: {failure}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Insight sidecar integration test runner")
    parser.add_argument("--suite", required=True, choices=["smoke", "stress", "concurrent", "memory", "edge"])
    parser.add_argument("--app", default=None, help="Path to Insight.app")
    parser.add_argument("--iterations", type=int, default=40, help="Iteration count for stress/concurrent")
    parser.add_argument("--workers", type=int, default=8, help="Worker threads for concurrent suite")
    parser.add_argument("--duration", type=int, default=180, help="Duration (seconds) for memory suite")
    parser.add_argument("--interval", type=int, default=5, help="Sampling interval (seconds) for memory suite")
    parser.add_argument("--artifacts-dir", default=None, help="Directory to keep suite artifacts")
    parser.add_argument("--keep-artifacts", action="store_true", help="Preserve artifacts directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    root = _repo_root()
    try:
        app_path = _detect_app_path(args.app, root)
        engine_path = _find_engine_binary(app_path)
    except Exception as exc:
        print(f"[FAIL] setup: {exc}", file=sys.stderr)
        return 2

    if args.artifacts_dir:
        artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        artifacts_dir = Path(tempfile.mkdtemp(prefix=f"insight_{args.suite}_test_"))
        cleanup = not args.keep_artifacts

    print(f"[INFO] app={app_path}")
    print(f"[INFO] engine={engine_path}")
    print(f"[INFO] artifacts={artifacts_dir}")
    workspace_dir = artifacts_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] workspace={workspace_dir}")

    ctx = SuiteContext(
        app_path=app_path,
        engine_path=engine_path,
        repo_root=root,
        artifacts_dir=artifacts_dir,
        workspace_dir=workspace_dir,
    )

    try:
        try:
            if args.suite == "smoke":
                result = run_smoke(ctx, args)
            elif args.suite == "stress":
                result = run_stress(ctx, args)
            elif args.suite == "concurrent":
                result = run_concurrent(ctx, args)
            elif args.suite == "memory":
                result = run_memory(ctx, args)
            elif args.suite == "edge":
                result = run_edge(ctx, args)
            else:
                print(f"[FAIL] unknown suite={args.suite}", file=sys.stderr)
                return 2
        except Exception as exc:
            result = SuiteResult(args.suite)
            result.fail(f"unhandled exception: {exc}")

        _print_result(result)
        return 0 if result.ok else 1
    finally:
        if cleanup:
            shutil.rmtree(artifacts_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
