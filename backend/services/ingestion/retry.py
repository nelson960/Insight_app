from __future__ import annotations

import random
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Dict, FrozenSet, Optional

from .models import IngestionErrorCode, RetryState


class RetryPolicy:
    """Centralised exponential backoff controller for ingestion tasks."""

    _DEFAULT_TRANSIENT: FrozenSet[IngestionErrorCode] = frozenset(
        {
            IngestionErrorCode.EMBEDDING_TIMEOUT,
            IngestionErrorCode.RATE_LIMIT,
            IngestionErrorCode.INDEX_WRITE_FAIL,
        }
    )

    def __init__(
        self,
        *,
        base_delay_seconds: float = 15.0,
        max_delay_seconds: float = 15 * 60.0,
        max_retries: int = 3,
        jitter_ratio: float = 0.2,
        transient_errors: Optional[
            FrozenSet[IngestionErrorCode]
        ] = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if base_delay_seconds <= 0:
            raise ValueError("base_delay_seconds must be positive")
        if max_delay_seconds < base_delay_seconds:
            raise ValueError("max_delay_seconds must be >= base_delay_seconds")
        if not (0.0 <= jitter_ratio <= 1.0):
            raise ValueError("jitter_ratio must be within [0, 1]")

        self._base_delay_seconds = base_delay_seconds
        self._max_delay_seconds = max_delay_seconds
        self._max_retries = max_retries
        self._jitter_ratio = jitter_ratio
        self._transient_errors = transient_errors or self._DEFAULT_TRANSIENT

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def is_transient(self, error_code: IngestionErrorCode) -> bool:
        return error_code in self._transient_errors

    def should_retry(
        self,
        *,
        error_code: IngestionErrorCode,
        state: RetryState,
    ) -> bool:
        """Decide whether another retry should be attempted."""
        if not self.is_transient(error_code):
            return False
        return state.retry_count < self._max_retries

    def compute_next_state(self, state: RetryState, *, now: Optional[datetime] = None) -> RetryState:
        """Return an updated retry state with incremented attempt count."""
        attempt = state.retry_count + 1
        delay_seconds = self._base_delay_seconds * (2 ** (attempt - 1))
        delay_seconds = min(delay_seconds, self._max_delay_seconds)

        if self._jitter_ratio > 0.0:
            jitter = delay_seconds * self._jitter_ratio
            delay_seconds = random.uniform(delay_seconds - jitter, delay_seconds + jitter)

        next_retry_at = (now or datetime.utcnow()) + timedelta(seconds=delay_seconds)
        return replace(state, retry_count=attempt, next_retry_at=next_retry_at)


def classify_exception(exc: BaseException) -> IngestionErrorCode:
    """Map arbitrary exceptions into a structured ingestion error code."""
    mapping: Dict[str, IngestionErrorCode] = {
        "DeadlineExceeded": IngestionErrorCode.EMBEDDING_TIMEOUT,
        "RateLimitError": IngestionErrorCode.RATE_LIMIT,
        "IndexWriteError": IngestionErrorCode.INDEX_WRITE_FAIL,
        "ExtractionError": IngestionErrorCode.EXTRACTION_FAILED,
        "CorruptFileError": IngestionErrorCode.CORRUPT_FILE,
    }
    exc_name = type(exc).__name__
    return mapping.get(exc_name, IngestionErrorCode.UNKNOWN)


__all__ = ["RetryPolicy", "classify_exception"]
