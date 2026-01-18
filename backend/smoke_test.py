"""
Smoke test for verifying packaged binary functionality.

Run with:
    insight-engine --smoketest

This tests all critical components without blocking on downloads.
"""
from __future__ import annotations

import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  {title}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


def test(name: str, func) -> bool:
    """Run a test and print result."""
    try:
        func()
        print(f"  ✓ {name}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"  ✗ {name}: {e}", file=sys.stderr)
        return False


def test_environment() -> None:
    """Test Python environment setup."""
    import sys
    assert sys.version_info >= (3, 10), f"Python {sys.version} too old"

    # Check if packaged
    is_frozen = getattr(sys, 'frozen', False)
    meipass = getattr(sys, '_MEIPASS', None)

    if is_frozen:
        assert meipass, "sys._MEIPASS not set in frozen app"
        assert Path(meipass).exists(), f"sys._MEIPASS path doesn't exist: {meipass}"


def test_workspace() -> None:
    """Test workspace initialization."""
    # Import here to avoid issues
    from backend.core.workspace import get_workspace, Workspace

    # Create workspace if it doesn't exist
    ws = Workspace()
    assert ws.base.exists(), f"Workspace base doesn't exist: {ws.base}"

    # Test writability
    probe = ws.base / ".smoke_test.tmp"
    probe.write_text("test")
    probe.unlink()


def test_sqlite() -> None:
    """Test SQLite database."""
    import sqlite3
    from backend.core.workspace import Workspace

    ws = Workspace()
    db_path = ws.db

    # Open database
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO test (id) VALUES (1)")
    conn.commit()
    result = conn.execute("SELECT id FROM test WHERE id = 1").fetchone()
    assert result is not None, "SQLite write/read failed"
    conn.close()


def test_certificates() -> None:
    """Test SSL certificates are available."""
    import certifi
    import urllib.request
    import ssl
    import os

    bundle = certifi.where()
    assert Path(bundle).exists(), f"Certificate bundle not found: {bundle}"

    # Log certificate environment
    print(f"    [SSL] certifi.where(): {bundle}", file=sys.stderr)
    print(f"    [SSL] REQUESTS_CA_BUNDLE: {os.environ.get('REQUESTS_CA_BUNDLE', 'NOT SET')}", file=sys.stderr)
    print(f"    [SSL] SSL_CERT_FILE: {os.environ.get('SSL_CERT_FILE', 'NOT SET')}", file=sys.stderr)
    print(f"    [SSL] Active cert: {os.environ.get('REQUESTS_CA_BUNDLE') or os.environ.get('SSL_CERT_FILE') or bundle}", file=sys.stderr)

    # Verify active cert exists
    active_cert = os.environ.get('REQUESTS_CA_BUNDLE') or os.environ.get('SSL_CERT_FILE') or bundle
    assert Path(active_cert).exists(), f"Active certificate not found: {active_cert}"

    # Test SSL connection (non-blocking)
    context = ssl.create_default_context()
    with urllib.request.urlopen("https://www.google.com", context=context, timeout=5) as response:
        assert response.status == 200, "HTTPS connection failed"


def test_onnxruntime() -> None:
    """Test ONNX Runtime."""
    import onnxruntime as ort
    from backend.core.workspace import Workspace

    # Check version
    assert ort.__version__, "ONNX Runtime has no version"

    # Check providers
    providers = ort.get_available_providers()
    assert providers, "No ONNX Runtime providers available"

    # Try to create a session (if we have a model)
    ws = Workspace()
    model_dir = ws.base / "em_models" / "nomic-embed-text"
    model_path = model_dir / "onnx" / "model.onnx"

    if model_path.exists():
        session = ort.InferenceSession(str(model_path))
        assert session, "Failed to create ONNX session"
    else:
        print(f"    (ONNX model not found at {model_path}, skipping session test)", file=sys.stderr)


def test_tokenizers() -> None:
    """Test tokenizers library."""
    from tokenizers import Tokenizer

    # Try to load a simple tokenizer
    tok = Tokenizer.from_pretrained("bert-base-uncased")
    assert tok is not None, "Failed to load tokenizer"

    # Test encoding
    output = tok.encode("Hello world")
    assert output, "Tokenization failed"


def test_llama_cpp() -> None:
    """Test llama.cpp (if model configured)."""
    import llama_cpp
    from backend.core.workspace import Workspace
    from backend.services.storage.sqlite_store import SQLiteConfig, create_sqlite_store

    assert llama_cpp.__version__, "llama_cpp has no version"

    # Check if native library is accessible
    try:
        from llama_cpp import Llama
        print(f"    (llama_cpp Llama class accessible)", file=sys.stderr)
    except ImportError as e:
        print(f"    (llama_cpp Llama import failed: {e} - may be OK for packaged build)", file=sys.stderr)

    # Test loading a model if configured
    ws = Workspace()
    store = create_sqlite_store(ws.db, config=SQLiteConfig())

    try:
        model_path = store.get_setting("llm_model_path", "")
        if model_path and isinstance(model_path, str):
            p = Path(model_path).expanduser()
            if p.exists():
                # Try to load the model (this is the real test)
                from llama_cpp import Llama
                llm = Llama(
                    model_path=str(p),
                    n_ctx=512,  # Small context for smoke test
                    n_gpu_layers=0,  # Don't use GPU for test
                )
                assert llm, "Failed to load llama.cpp model"
            else:
                print(f"    (Model not found at {p}, skipping model load test)", file=sys.stderr)
        else:
            print(f"    (No model configured, skipping llama.cpp model test)", file=sys.stderr)
    finally:
        store.close()


def test_qdrant() -> None:
    """Test Qdrant local client."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from backend.core.workspace import Workspace

    ws = Workspace()
    qdrant_path = ws.qdrant / "smoke_test"

    # Create client
    client = QdrantClient(path=str(qdrant_path))

    # Create collection
    collection_name = "smoke_test_collection"
    vector_size = 768

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

    # Insert point
    client.upsert(
        collection_name=collection_name,
        points=[PointStruct(id=1, vector=[0.1] * vector_size, payload={"test": "data"})]
    )

    # Search
    results = client.search(
        collection_name=collection_name,
        query_vector=[0.1] * vector_size,
        limit=1
    )

    assert len(results) > 0, "Qdrant search returned no results"
    assert results[0].id == 1, "Qdrant search returned wrong result"

    # Cleanup
    client.delete_collection(collection_name)


def run_smoke_tests() -> int:
    """Run all smoke tests and return exit code (0=success, 1=failure)."""
    print_section("INSIGHT SMOKE TEST")
    print(f"Time: {datetime.now().isoformat()}", file=sys.stderr)
    print(f"Python: {sys.version}", file=sys.stderr)
    print(f"Executable: {sys.executable}", file=sys.stderr)

    is_frozen = getattr(sys, 'frozen', False)
    if is_frozen:
        print(f"Packaged: YES (sys._MEIPASS={getattr(sys, '_MEIPASS', 'N/A')})", file=sys.stderr)
    else:
        print(f"Packaged: NO (dev mode)", file=sys.stderr)

    tests = [
        ("Environment", test_environment),
        ("Workspace", test_workspace),
        ("SQLite", test_sqlite),
        ("SSL Certificates", test_certificates),
        ("ONNX Runtime", test_onnxruntime),
        ("Tokenizers", test_tokenizers),
        ("llama.cpp", test_llama_cpp),
        ("Qdrant", test_qdrant),
    ]

    results = []
    start_time = time.time()

    for name, func in tests:
        print_section(name)
        passed = test(name, func)
        results.append((name, passed))

    elapsed = time.time() - start_time

    # Summary
    print_section("SUMMARY")
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}", file=sys.stderr)

    print(f"\nTotal: {passed_count}/{total_count} tests passed in {elapsed:.1f}s", file=sys.stderr)

    if passed_count == total_count:
        print(f"\n✓ All tests passed!", file=sys.stderr)
        return 0
    else:
        print(f"\n✗ {total_count - passed_count} test(s) failed", file=sys.stderr)
        return 1


def main() -> int:
    """Entry point for smoke test."""
    # Check if --smoketest flag is present
    if "--smoketest" in sys.argv:
        return run_smoke_tests()
    return 0


if __name__ == "__main__":
    sys.exit(main())
