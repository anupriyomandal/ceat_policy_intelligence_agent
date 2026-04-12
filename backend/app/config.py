from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self) -> None:
        backend_root = Path(__file__).resolve().parents[1]
        project_root = backend_root.parent

        self.project_root = Path(os.getenv("PROJECT_ROOT", project_root))
        self.raw_dir = Path(os.getenv("RAW_DIR", self.project_root / "raw"))
        self.index_dir = Path(os.getenv("INDEX_DIR", self.project_root / "faiss_index"))
        self.index_file = self.index_dir / "index.faiss"
        self.store_file = self.index_dir / "store.json"

        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.chat_model = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL")

        self.chunk_size_tokens = int(os.getenv("CHUNK_SIZE_TOKENS", "700"))
        self.chunk_overlap_tokens = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))
        self.top_k = int(os.getenv("TOP_K", "4"))

        self.app_env = os.getenv("APP_ENV", "development")
        self.cors_origins = [
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
            if origin.strip()
        ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
