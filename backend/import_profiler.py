"""Manual import profiler for debugging startup stall."""

import sys
import time
from pathlib import Path

# Timing storage
_import_times = {}


def timed_import(module_name, import_func):
    """Time an import and log it."""
    start = time.time()
    try:
        result = import_func()
        elapsed = time.time() - start
        _import_times[module_name] = elapsed

        # Log to stderr so it shows up in packaged mode
        if elapsed > 0.1:  # Only log imports taking > 100ms
            print(f"[IMPORT PROFILER] {module_name}: {elapsed:.2f}s", file=sys.stderr)

        return result
    except Exception as e:
        elapsed = time.time() - start
        print(f"[IMPORT PROFILER] {module_name}: FAILED after {elapsed:.2f}s - {e}", file=sys.stderr)
        raise


def get_import_report():
    """Get a report of all timed imports."""
    if not _import_times:
        return "No imports profiled"

    lines = ["\n=== IMPORT TIME REPORT ==="]
    sorted_times = sorted(_import_times.items(), key=lambda x: x[1], reverse=True)

    for module, elapsed in sorted_times:
        if elapsed > 0.1:  # Only show > 100ms
            lines.append(f"  {module}: {elapsed:.2f}s")

    total = sum(_import_times.values())
    lines.append(f"\n  Total profiled: {total:.2f}s")
    lines.append("=" * 30)
    return "\n".join(lines)


# Monkey-patch the built-in __import__ to track all imports
_original_import = __builtins__.__import__


def _profiling_import(name, *args, **kwargs):
    """Profile imports."""
    if name.startswith('backend') or name in {'onnxruntime', 'llama_cpp', 'tokenizers', 'qdrant_client', 'huggingface_hub'}:
        start = time.time()
        try:
            result = _original_import(name, *args, **kwargs)
            elapsed = time.time() - start
            if elapsed > 0.05:  # Only log > 50ms
                print(f"[IMPORT] {name}: {elapsed:.2f}s", file=sys.stderr)
                _import_times[name] = _import_times.get(name, 0) + elapsed
            return result
        except Exception as e:
            elapsed = time.time() - start
            print(f"[IMPORT] {name}: FAILED after {elapsed:.2f}s", file=sys.stderr)
            raise
    else:
        return _original_import(name, *args, **kwargs)


def enable_import_profiling():
    """Enable import profiling."""
    __builtins__.__import__ = _profiling_import
    print("[IMPORT PROFILER] Enabled", file=sys.stderr)
