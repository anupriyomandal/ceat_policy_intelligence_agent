from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from app.schemas import ChunkRecord


class LocalFaissStore:
    def __init__(self, index: faiss.Index, records: list[ChunkRecord]) -> None:
        self.index = index
        self.records = records

    @classmethod
    def create(cls, dimension: int) -> "LocalFaissStore":
        index = faiss.IndexFlatIP(dimension)
        return cls(index=index, records=[])

    @classmethod
    def load(cls, index_path: Path, store_path: Path) -> "LocalFaissStore":
        if not index_path.exists() or not store_path.exists():
            raise FileNotFoundError(
                f"Missing FAISS artifacts. Expected {index_path} and {store_path}."
            )

        index = faiss.read_index(str(index_path))
        payload = json.loads(store_path.read_text())
        records = [ChunkRecord.from_dict(item) for item in payload["records"]]
        return cls(index=index, records=records)

    def add(self, embeddings: np.ndarray, records: list[ChunkRecord]) -> None:
        normalized = self._normalize(embeddings)
        self.index.add(normalized)
        self.records.extend(records)

    def search(
        self, query_embedding: np.ndarray, top_k: int
    ) -> list[tuple[ChunkRecord, float]]:
        if self.index.ntotal == 0:
            return []

        query = self._normalize(query_embedding.reshape(1, -1))
        scores, indices = self.index.search(query, top_k)

        results: list[tuple[ChunkRecord, float]] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self.records):
                continue
            results.append((self.records[idx], float(score)))
        return results

    def save(self, index_path: Path, store_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        payload = {"records": [record.to_dict() for record in self.records]}
        store_path.write_text(json.dumps(payload, indent=2))

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        vectors = vectors.astype("float32")
        faiss.normalize_L2(vectors)
        return vectors
