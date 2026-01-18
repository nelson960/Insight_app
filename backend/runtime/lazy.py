"""Lazy loading for heavy modules to avoid slow startup."""
import threading
from typing import Any, Callable, TypeVar

T = TypeVar('T')

# Global cache for lazy-loaded modules
_cache: dict[str, Any] = {}
_lock = threading.Lock()


def lazy_import(module_name: str) -> Any:
    """
    Lazy import a module and cache it.

    Args:
        module_name: Python module path (e.g., 'onnxruntime')

    Returns:
        The imported module

    Example:
        def get_onnxruntime():
            return lazy_import('onnxruntime')
    """
    with _lock:
        if module_name in _cache:
            return _cache[module_name]

        try:
            module = __import__(module_name, fromlist=[''])
            _cache[module_name] = module
            return module
        except ImportError as e:
            # Cache the failure so we don't retry repeatedly
            _cache[module_name] = None
            raise


def lazy_getter(module_name: str, attribute_name: str) -> Any:
    """
    Lazy import a specific attribute from a module.

    Args:
        module_name: Python module path
        attribute_name: Attribute to import (e.g., 'InferenceSession')

    Returns:
        The imported attribute

    Example:
        InferenceSession = lazy_getter('onnxruntime.capi.onnxruntime_pybind11_state', 'InferenceSession')
    """
    with _lock:
        key = f"{module_name}.{attribute_name}"
        if key in _cache:
            return _cache[key]

        try:
            module = __import__(module_name, fromlist=[attribute_name])
            attr = getattr(module, attribute_name)
            _cache[key] = attr
            return attr
        except (ImportError, AttributeError) as e:
            _cache[key] = None
            raise


# Convenience functions for commonly used heavy modules
def get_onnxruntime():
    """Lazy load onnxruntime."""
    return lazy_import('onnxruntime')


def get_llama_cpp():
    """Lazy load llama_cpp."""
    return lazy_import('llama_cpp')


def get_tokenizers():
    """Lazy load tokenizers."""
    return lazy_import('tokenizers')


def get_qdrant_client():
    """Lazy load qdrant_client."""
    return lazy_import('qdrant_client')
