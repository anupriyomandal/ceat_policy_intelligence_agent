from __future__ import annotations

import math
from typing import TypeVar

from app.config import get_settings
from app.embeddings import EmbeddingService
from app.ingestion import load_pdf_chunks
from app.vector_store import LocalFaissStore

T = TypeVar("T")


def batched(items: list[T], size: int) -> list[list[T]]:
    total = math.ceil(len(items) / size)
    return [items[index * size : (index + 1) * size] for index in range(total)]


def main() -> None:
    settings = get_settings()
    records = load_pdf_chunks(
        raw_dir=settings.raw_dir,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )

    embeddings = EmbeddingService(settings)
    vector_batches = []
    record_batches = batched(records, 64)
    for batch in record_batches:
        texts = [record.text for record in batch]
        vector_batches.append(embeddings.embed_texts(texts))

    if not vector_batches:
        raise RuntimeError("No embeddings generated.")

    import numpy as np

    matrix = np.vstack(vector_batches)
    store = LocalFaissStore.create(dimension=matrix.shape[1])
    store.add(matrix, records)
    store.save(settings.index_file, settings.store_file)

    print(
        f"Ingested {len(records)} chunks from {settings.raw_dir} into {settings.index_dir}"
    )


if __name__ == "__main__":
    main()
