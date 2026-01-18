from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Optional

GGUF_MAGIC = b"GGUF"

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

_SCALAR_TYPE_SIZES = {
    GGUF_TYPE_UINT8: 1,
    GGUF_TYPE_INT8: 1,
    GGUF_TYPE_UINT16: 2,
    GGUF_TYPE_INT16: 2,
    GGUF_TYPE_UINT32: 4,
    GGUF_TYPE_INT32: 4,
    GGUF_TYPE_FLOAT32: 4,
    GGUF_TYPE_BOOL: 1,
    GGUF_TYPE_UINT64: 8,
    GGUF_TYPE_INT64: 8,
    GGUF_TYPE_FLOAT64: 8,
}


def _read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("Unexpected end of file")
    return data


def _read_uint32(handle) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _read_uint64(handle) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _read_str(handle) -> str:
    length = _read_uint64(handle)
    if length <= 0:
        return ""
    data = _read_exact(handle, length)
    return data.decode("utf-8", errors="ignore")


def _skip_str(handle) -> None:
    length = _read_uint64(handle)
    if length > 0:
        handle.seek(length, os.SEEK_CUR)


def _skip_value(handle, value_type: int) -> None:
    if value_type == GGUF_TYPE_STRING:
        _skip_str(handle)
        return
    if value_type == GGUF_TYPE_ARRAY:
        elem_type = _read_uint32(handle)
        length = _read_uint64(handle)
        if length <= 0:
            return
        if elem_type == GGUF_TYPE_STRING:
            for _ in range(length):
                _skip_str(handle)
            return
        size = _SCALAR_TYPE_SIZES.get(elem_type)
        if size is None:
            raise ValueError("Unsupported GGUF array element type")
        handle.seek(size * length, os.SEEK_CUR)
        return
    size = _SCALAR_TYPE_SIZES.get(value_type)
    if size is None:
        raise ValueError("Unsupported GGUF value type")
    handle.seek(size, os.SEEK_CUR)


def is_gguf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == GGUF_MAGIC
    except Exception:
        return False


def read_gguf_string_kv(path: Path, key: str) -> Optional[str]:
    try:
        with path.open("rb") as handle:
            if handle.read(4) != GGUF_MAGIC:
                return None
            _ = _read_uint32(handle)  # version
            _ = _read_uint64(handle)  # n_tensors
            n_kv = _read_uint64(handle)
            for _ in range(n_kv):
                k = _read_str(handle)
                value_type = _read_uint32(handle)
                if k == key:
                    if value_type == GGUF_TYPE_STRING:
                        return _read_str(handle)
                    _skip_value(handle, value_type)
                    return None
                _skip_value(handle, value_type)
    except Exception:
        return None
    return None


def detect_chat_template_kind(path: Path) -> str:
    template = read_gguf_string_kv(path, "tokenizer.chat_template")
    if not template:
        return "unknown"
    if "<|im_start|>" in template and "<|im_end|>" in template:
        return "chatml"
    if "<|start_header_id|>" in template and "<|eot_id|>" in template:
        return "llama3"
    return "unknown"


__all__ = ["is_gguf", "read_gguf_string_kv", "detect_chat_template_kind"]
