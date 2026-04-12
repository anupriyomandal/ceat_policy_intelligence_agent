from __future__ import annotations

from pathlib import Path

import pdfplumber

from app.chunking import chunk_text, clean_text
from app.schemas import ChunkRecord


def load_pdf_chunks(
    raw_dir: Path,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> list[ChunkRecord]:
    pdf_paths = sorted(raw_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {raw_dir}")

    records: list[ChunkRecord] = []
    for pdf_path in pdf_paths:
        document_name = pdf_path.stem
        next_chunk_id = 1
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                extracted = page.extract_text() or ""
                cleaned = clean_text(extracted)
                if not cleaned:
                    continue

                chunks = chunk_text(
                    cleaned,
                    chunk_size_tokens=chunk_size_tokens,
                    overlap_tokens=chunk_overlap_tokens,
                )
                for chunk in chunks:
                    records.append(
                        ChunkRecord(
                            text=chunk,
                            document_name=document_name,
                            chunk_id=next_chunk_id,
                            page_number=page_idx,
                        )
                    )
                    next_chunk_id += 1

    return records
