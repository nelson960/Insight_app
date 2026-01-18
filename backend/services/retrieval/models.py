from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass
class RetrievalQuery:
    """Encapsulates the vectorized query and retrieval params."""

    vector: Sequence[float]
    limit: int = 12
    with_chunks: bool = True
    with_metadata: bool = True


@dataclass
class RetrievalCandidate:
    """Intermediate candidate returned from the vector store."""

    chunk_id: str
    score: float
    payload: Dict[str, object]


@dataclass
class RetrievalResult:
    """Final retrieval response combining chunk text and metadata."""

    chunk_id: str
    score: float
    text: Optional[str]
    metadata: Dict[str, object]
    file_id: Optional[str]
    seq: Optional[int]


__all__ = ["RetrievalQuery", "RetrievalCandidate", "RetrievalResult"]
