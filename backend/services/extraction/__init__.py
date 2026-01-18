"""
Universal file extraction service for Insight.
"""

from .base import BaseExtractor, ExtractedDocument, ExtractionError, SimpleExtractor
from .registry import ExtractorRegistry, build_default_registry
from .service import FileExtractionResult, FileExtractionService, create_extraction_service

__all__ = [
    "BaseExtractor",
    "SimpleExtractor",
    "ExtractedDocument",
    "ExtractionError",
    "ExtractorRegistry",
    "build_default_registry",
    "FileExtractionService",
    "FileExtractionResult",
    "create_extraction_service",
]
