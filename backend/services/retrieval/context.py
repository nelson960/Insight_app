from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from qdrant_client.http import models as qmodels


@dataclass(slots=True)
class RetrievalContext:
    user_id: Optional[str] = None
    chat_id: Optional[str] = None
    file_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    date_range: Optional[Tuple[str, str]] = None  # ISO8601 strings


def build_filter(context: Optional[RetrievalContext]) -> Optional[qmodels.Filter]:
    if context is None:
        return None

    must: List[qmodels.FieldCondition] = []
    if context.user_id:
        must.append(qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=context.user_id)))
    if context.chat_id:
        must.append(qmodels.FieldCondition(key="chat_id", match=qmodels.MatchValue(value=context.chat_id)))
    if context.file_ids:
        must.append(qmodels.FieldCondition(key="file_id", match=qmodels.MatchAny(any=context.file_ids)))
    if context.tags:
        must.append(qmodels.FieldCondition(key="tags", match=qmodels.MatchAny(any=context.tags)))
    if context.date_range:
        start, end = context.date_range
        must.append(
            qmodels.FieldCondition(
                key="created_at",
                range=qmodels.Range(gte=start, lte=end),
            )
        )

    if not must:
        return None
    return qmodels.Filter(must=must)


__all__ = ["RetrievalContext", "build_filter"]
