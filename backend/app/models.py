from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Relevance = Literal["high", "medium", "low"]
Confidence = Literal["high", "medium", "low"]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=10)


class Source(BaseModel):
    document: str
    relevance: Relevance


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    confidence: Confidence
    retrieved_chunks: int
