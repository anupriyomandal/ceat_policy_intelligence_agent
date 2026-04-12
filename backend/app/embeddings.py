from __future__ import annotations

from typing import Sequence

import numpy as np
from openai import OpenAI

from app.config import Settings


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        client_kwargs = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        self.client = OpenAI(**client_kwargs)
        self.model = settings.embedding_model

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")

        response = self.client.embeddings.create(model=self.model, input=list(texts))
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
