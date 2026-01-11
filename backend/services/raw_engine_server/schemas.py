from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str = ""
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repeat_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None


class EmbeddingsRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="Input text or list of texts")
    model: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_path: Optional[str] = None
    embedding_auto_download: Optional[bool] = None
