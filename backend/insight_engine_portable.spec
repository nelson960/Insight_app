# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Insight Engine (PORTABLE VERSION)

Build:
    pyinstaller --noconfirm --clean backend/insight_engine_portable.spec

Test:
    INSIGHT_SMOKETEST=1 ./dist/insight-engine
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    collect_dynamic_libs,
)

# -----------------------------------------------------------------------------
# Paths (IMPORTANT: __file__ is NOT defined in PyInstaller spec execution env)
# -----------------------------------------------------------------------------
SPEC_DIR = Path(globals().get("SPECPATH", os.getcwd())).resolve()  # .../backend
REPO_ROOT = SPEC_DIR.parent                                        # repo root
ENTRYPOINT = str(SPEC_DIR / "engine.py")                           # .../backend/engine.py

block_cipher = None

# -----------------------------------------------------------------------------
# Data files to include
# -----------------------------------------------------------------------------
datas = []

# ✅ Correct: bundle certifi's data files (includes certifi/cacert.pem)
try:
    datas += collect_data_files("certifi")
    print(f"[PyInstaller] Bundling certifi data files: {len(datas)} entries", file=sys.stderr)
except Exception as e:
    print(f"[PyInstaller] WARNING: collect_data_files('certifi') failed: {e}", file=sys.stderr)

# ✅ macOS SQLite: ensure _sqlite3.so is bundled
if sys.platform == "darwin":
    try:
        import sqlite3
        sqlite_path = sqlite3.__file__
        if sqlite_path.endswith("__init__.py"):
            # Get the parent directory which contains _sqlite3.so
            sqlite_dir = str(Path(sqlite_path).parent)
            # Add as data to preserve the shared library (don't use binaries to avoid stripping)
            datas.append((sqlite_dir, "sqlite3"))
            print(f"[PyInstaller] Added SQLite data files from {sqlite_dir}", file=sys.stderr)
    except Exception as e:
        print(f"[PyInstaller] WARNING: Failed to collect SQLite data files: {e}", file=sys.stderr)

# -----------------------------------------------------------------------------
# Hidden imports
# -----------------------------------------------------------------------------
hiddenimports = [
    # FastAPI & Starlette
    "fastapi",
    "fastapi.responses",
    "starlette",
    "starlette.responses",
    "starlette.middleware",

    # ONNX Runtime (native)
    "onnxruntime",
    "onnxruntime.capi",
    "onnxruntime.capi._pybind_state",
    "onnxruntime.capi.onnxruntime_pybind11_state",

    # llama_cpp (native) — don't force llama_cpp.llama_cpp_lib
    "llama_cpp",
    "llama_cpp.llama_chat_format",

    # tokenizers (native)
    "tokenizers",
    "tokenizers.models",
    "tokenizers.pre_tokenizers",
    "tokenizers.processors",

    # huggingface_hub (downloads)
    "huggingface_hub",
    "huggingface_hub._snapshot_download",
    "huggingface_hub.utils",
    "huggingface_hub.utils._typing",

    # anyio
    "anyio",
    "anyio.streams",
    "anyio.abc",
    "anyio.streams.memory",

    # Qdrant
    "qdrant_client",
    "qdrant_client.local",

    # Uvicorn
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",

    # SQLite
    "sqlite3",
    "_sqlite3",  # Explicitly include the native extension

    # HTTP stack
    "requests",
    "urllib3",
    "charset_normalizer",
    "idna",
    "certifi",

    # Utils
    "numpy",
    "pandas",
    "aiofiles",
    "python_multipart",
    "yaml",
]

# Collect all submodules for critical packages
for package in ["onnxruntime", "llama_cpp", "tokenizers", "huggingface_hub"]:
    try:
        subs = collect_submodules(package)
        hiddenimports.extend(subs)
        print(f"[PyInstaller] Collected {len(subs)} submodules from {package}", file=sys.stderr)
    except Exception as e:
        print(f"[PyInstaller] WARNING: collect_submodules failed for {package}: {e}", file=sys.stderr)

# Project modules
hiddenimports.extend([
    "backend",
    "backend.api",
    "backend.api.app",
    "backend.api.deps",
    "backend.api.routers",
    "backend.api.routers.chat",
    "backend.api.routers.docs",
    "backend.api.routers.files",
    "backend.api.routers.search",
    "backend.api.routers.settings",
    "backend.api.routers.diagnostics",
    "backend.core",
    "backend.core.workspace",
    "backend.services",
    "backend.services.boot_trace",
    "backend.services.runtime_utils",
    "backend.services.connectors",
    "backend.services.connectors.nomic",
    "backend.services.connectors.nomic_onnx",
    "backend.services.connectors.llama_session_manager",
    "backend.services.storage",
    "backend.services.storage.sqlite_store",
    "backend.services.storage.qdrant_index",
    "backend.services.ingestion",
    "backend.services.security",
    "backend.services.logging_config",
    "backend.services.health",
    "backend.services.gguf_metadata",
    "backend.services.ipc_events",
    "backend.raw_engine_server",
    "backend.services.raw_engine_server",
    "backend.services.raw_engine_server.app",
    "backend.services.raw_engine_server.config",
    "backend.services.raw_engine_server.engine",
    "backend.services.raw_engine_server.logging_store",
    "backend.services.raw_engine_server.manager",
    "backend.services.raw_engine_server.schemas",
    "backend.services.raw_engine_server.sse",
    "backend.engine",
])

# -----------------------------------------------------------------------------
# Binaries (native libs)
# -----------------------------------------------------------------------------
binaries = []

# Collect dynamic libraries for native packages
for package in ["onnxruntime", "llama_cpp", "tokenizers"]:
    try:
        libs = collect_dynamic_libs(package)
        binaries.extend(libs)
        print(f"[PyInstaller] Collected {len(libs)} dynamic libs from {package}", file=sys.stderr)
    except Exception as e:
        print(f"[PyInstaller] WARNING: collect_dynamic_libs failed for {package}: {e}", file=sys.stderr)



# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------
a = Analysis(
    [ENTRYPOINT],
    pathex=[str(SPEC_DIR), str(REPO_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Scientific computing (not needed)
        "matplotlib", "scipy", "pandas",
        "IPython", "jupyter", "jupyterlab", "notebook",
        # Testing
        "pytest", "test", "tests", "unittest", "doctest", "hypothesis",
        # Development tools
        "sphinx", "docs", "black", "isort", "flake8", "mypy", "pylint",
        "pydocstyle", "bandit",
        # GUI toolkits (not needed, we use Tauri)
        "tkinter", "tk",
        # ML frameworks not used directly
        "torch", "tensorflow",
        # Optional heavy dependencies
        "PIL", "Pillow", "cv2", "opencv",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

a.datas = [d for d in a.datas if d and d[0] is not None]
a.binaries = [b for b in a.binaries if b and b[0] is not None]

# -----------------------------------------------------------------------------
# Exclude debug/test files from bundle
# -----------------------------------------------------------------------------
print("[PyInstaller] Filtering out debug/test files...", file=sys.stderr)

# Files to exclude from bundling
_excluded_patterns = [
    'import_profiler',
    'raw_engine_server.py',
    'test_raw_engine_server',
]

# Filter out excluded files from data files (Python source files)
_original_datas_len = len(a.datas)
a.datas = [
    (name, path, type_code)
    for name, path, type_code in a.datas
    if not any(pattern in name for pattern in _excluded_patterns)
]
if len(a.datas) < _original_datas_len:
    print(f"[PyInstaller] Excluded {_original_datas_len - len(a.datas)} debug/test file(s) from bundle", file=sys.stderr)

# -----------------------------------------------------------------------------
# Version metadata (for Windows builds, ignored on macOS)
# -----------------------------------------------------------------------------
try:
    if sys.platform == "win32":
        version_info = VSVersionInfo(
            ffi=FixedFileInfo(
                filevers=(0, 1, 0, 0),
                prodvers=(0, 1, 0, 0),
                mask=0x3f,
                flags=0x0,
                OS=0x40004,
                fileType=0x1,
                subtype=0x0,
                date=(0, 0)
            ),
            kids=[
                StringFileInfo(
                    [
                        StringTable(
                            u'040904B0',
                            StringStruct(
                                [
                                    ('CompanyName', u'Insight AI'),
                                    ('FileDescription', u'Insight AI Engine'),
                                    ('FileVersion', u'0.1.0.0'),
                                    ('InternalName', u'insight-engine'),
                                    ('LegalCopyright', u'Copyright © 2025'),
                                    ('OriginalFilename', u'insight-engine.exe'),
                                    ('ProductName', u'Insight'),
                                    ('ProductVersion', u'0.1.0.0'),
                                ]
                            )
                        )
                    ]
                ),
                VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
            ]
        )
    else:
        version_info = None
except Exception:
    version_info = None

# -----------------------------------------------------------------------------
# Determine icon path based on platform
# -----------------------------------------------------------------------------
ICON_PATH = None
if sys.platform == "darwin":
    icon_path = REPO_ROOT / "insight" / "src-tauri" / "icons" / "icon.icns"
    if icon_path.exists():
        ICON_PATH = str(icon_path)
        print(f"[PyInstaller] Using icon: {ICON_PATH}", file=sys.stderr)
elif sys.platform == "win32":
    icon_path = REPO_ROOT / "insight" / "src-tauri" / "icons" / "icon.ico"
    if icon_path.exists():
        ICON_PATH = str(icon_path)
        print(f"[PyInstaller] Using icon: {ICON_PATH}", file=sys.stderr)

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="insight-engine",
    icon=ICON_PATH,
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,  # Remove debug symbols to reduce size
    upx=False,  # macOS: safer off, can cause issues with native libs
    runtime_tmpdir=None,
    console=True,  # Keep console on macOS for debugging; on Windows set to False
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,  # Don't strip collected binaries (breaks SQLite and other native libs)
    upx=False,
    name="insight-engine",
)

print("\n" + "=" * 70, file=sys.stderr)
print("PYINSTALLER BUILD COMPLETE", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print("\nOutput directory: dist/insight-engine", file=sys.stderr)
print("\nTo test:", file=sys.stderr)
print("  ./dist/insight-engine/insight-engine", file=sys.stderr)
print("  INSIGHT_SMOKETEST=1 ./dist/insight-engine/insight-engine", file=sys.stderr)
print("=" * 70 + "\n", file=sys.stderr)
