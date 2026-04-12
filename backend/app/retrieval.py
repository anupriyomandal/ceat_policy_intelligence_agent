from __future__ import annotations

from dataclasses import dataclass

from app.embeddings import EmbeddingService
from app.schemas import ChunkRecord
from app.vector_store import LocalFaissStore


@dataclass(slots=True)
class RetrievedChunk:
    record: ChunkRecord
    score: float


class Retriever:
    def __init__(self, store: LocalFaissStore, embeddings: EmbeddingService) -> None:
        self.store = store
        self.embeddings = embeddings

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query_embedding = self.embeddings.embed_query(query)
        results = self.store.search(query_embedding, top_k=top_k)
        return [RetrievedChunk(record=record, score=score) for record, score in results]


def format_context(chunks: list[RetrievedChunk]) -> str:
    formatted = []
    for item in chunks:
        page_value = item.record.page_number if item.record.page_number is not None else "unknown"
        formatted.append(
            "\n".join(
                [
                    "<chunk>",
                    f"document: {item.record.document_name}",
                    f"chunk_id: {item.record.chunk_id}",
                    f"page: {page_value}",
                    "content:",
                    item.record.text,
                    "</chunk>",
                ]
            )
        )
    return "\n\n".join(formatted)
