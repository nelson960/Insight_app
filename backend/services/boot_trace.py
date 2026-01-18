"""
Boot trace logging for diagnosing packaging issues.
Logs every step of initialization to a stable location.
"""
import sys
import threading
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Any
import traceback as tb


_boot_trace_lock = threading.Lock()
_boot_trace_file = None
_boot_trace_initialized = False


def init_boot_trace() -> Path:
    """Initialize boot trace logging to a stable location."""
    global _boot_trace_file, _boot_trace_initialized

    with _boot_trace_lock:
        if _boot_trace_initialized:
            return _boot_trace_file

        # Determine log location
        # Try INSIGHT_WORKSPACE_DIR first (set by Rust sidecar)
        import os
        workspace_dir = os.getenv("INSIGHT_WORKSPACE_DIR")

        if workspace_dir:
            log_dir = Path(workspace_dir) / "logs"
        else:
            # Fallback to ~/.insight/logs
            log_dir = Path.home() / ".insight" / "logs"

        # Create log directory
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # Last resort: /tmp
            log_dir = Path("/tmp")

        # Create boot trace file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _boot_trace_file = log_dir / f"boot_trace_{timestamp}.log"

        _boot_trace_initialized = True

        # Write initial header
        with open(_boot_trace_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"INSIGHT BOOT TRACE - {datetime.now().isoformat()}\n")
            f.write(f"Log file: {_boot_trace_file}\n")
            f.write(f"{'='*80}\n\n")

            # Environment info
            f.write(f"Python executable: {sys.executable}\n")
            f.write(f"Python version: {sys.version}\n")
            f.write(f"Platform: {sys.platform}\n")
            f.write(f"frozen: {getattr(sys, 'frozen', False)}\n")
            f.write(f"_MEIPASS: {getattr(sys, '_MEIPASS', 'N/A')}\n")
            f.write(f"Working directory: {os.getcwd()}\n")
            f.write(f"INSIGHT_WORKSPACE_DIR: {os.getenv('INSIGHT_WORKSPACE_DIR', 'N/A')}\n")
            f.write(f"\n")

        return _boot_trace_file


def log_boot_step(step_name: str, status: str = "INFO", **details: Any) -> None:
    """Log a boot step with optional details."""
    try:
        log_file = init_boot_trace()

        timestamp = datetime.now().isoformat(timespec="milliseconds")

        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] [{status}] {step_name}\n")

            if details:
                for key, value in details.items():
                    f.write(f"    {key}: {value}\n")

            f.write("\n")

        # Also print to stderr for immediate visibility
        print(f"[BOOT] [{status}] {step_name}", file=sys.stderr)
        for key, value in details.items():
            print(f"[BOOT]     {key}: {value}", file=sys.stderr)

    except Exception as e:
        # Last resort: print to stderr
        print(f"[BOOT ERROR] Failed to write boot trace: {e}", file=sys.stderr)


def log_boot_error(step_name: str, exc: Exception) -> None:
    """Log an error with full traceback."""
    try:
        log_file = init_boot_trace()

        timestamp = datetime.now().isoformat(timespec="milliseconds")

        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] [ERROR] {step_name}\n")
            f.write(f"    Exception type: {type(exc).__name__}\n")
            f.write(f"    Exception message: {exc}\n")
            f.write(f"    Traceback:\n")
            f.write(f"{''.join(tb.format_exception(type(exc), exc, exc.__traceback__))}\n")
            f.write("\n")

        # Also print to stderr
        print(f"[BOOT] [ERROR] {step_name}: {exc}", file=sys.stderr)
        print("".join(tb.format_exception(type(exc), exc, exc.__traceback__)), file=sys.stderr)

    except Exception as e:
        print(f"[BOOT ERROR] Failed to write error: {e}", file=sys.stderr)


@contextmanager
def boot_trace_step(step_name: str):
    """Context manager for tracing a step with timing."""
    import time
    log_boot_step(step_name, status="START")

    start = time.time()
    try:
        yield
        elapsed = time.time() - start
        log_boot_step(step_name, status="OK", elapsed_seconds=f"{elapsed:.3f}")
    except Exception as e:
        elapsed = time.time() - start
        log_boot_error(f"{step_name} (failed after {elapsed:.3f}s)", e)
        raise


def get_boot_trace_path() -> Path | None:
    """Get the boot trace file path."""
    return _boot_trace_file


# Auto-initialize on import
init_boot_trace()
