from __future__ import annotations

import re
from collections.abc import Iterable

import tiktoken


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n", text)
    return [block.strip() for block in blocks if block.strip()]


def chunk_text(
    text: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    if not text.strip():
        return []

    encoding = tiktoken.get_encoding(encoding_name)
    paragraphs = split_paragraphs(text)

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        para_tokens = len(encoding.encode(paragraph))
        if para_tokens > chunk_size_tokens:
            sentence_like_parts = _split_large_paragraph(paragraph)
        else:
            sentence_like_parts = [paragraph]

        for part in sentence_like_parts:
            part_tokens = len(encoding.encode(part))
            if current_parts and current_tokens + part_tokens > chunk_size_tokens:
                chunks.append("\n\n".join(current_parts))
                overlap_text = _build_overlap(current_parts, overlap_tokens, encoding)
                current_parts = overlap_text[:] if overlap_text else []
                current_tokens = sum(len(encoding.encode(item)) for item in current_parts)

            current_parts.append(part)
            current_tokens += part_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _split_large_paragraph(paragraph: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", paragraph)
    return [part.strip() for part in parts if part.strip()]


def _build_overlap(
    parts: Iterable[str], overlap_tokens: int, encoding: tiktoken.Encoding
) -> list[str]:
    selected: list[str] = []
    total = 0
    for part in reversed(list(parts)):
        tokens = len(encoding.encode(part))
        if total + tokens > overlap_tokens and selected:
            break
        selected.append(part)
        total += tokens
        if total >= overlap_tokens:
            break
    selected.reverse()
    return selected
