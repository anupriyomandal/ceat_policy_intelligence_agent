from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class ChunkRecord:
    text: str
    document_name: str
    chunk_id: int
    page_number: int | None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkRecord":
        return cls(
            text=data["text"],
            document_name=data["document_name"],
            chunk_id=int(data["chunk_id"]),
            page_number=data.get("page_number"),
        )
