"""Application configuration.

All runtime settings come from environment variables (or a `.env` file in the
project root).  Nothing secret lives in code.  Access the single `settings`
instance everywhere::

    from chat_with_website.config import settings
    settings.database_url
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EmbeddingProvider = Literal["huggingface", "openai", "gemini"]
LLMProvider = Literal["ollama", "openai", "gemini"]

# Default model and vector size for each embedding provider.
# The vector size MUST match the `chunks.embedding` column in PostgreSQL.
EMBEDDING_DEFAULTS: dict[str, tuple[str, int]] = {
    "huggingface": ("sentence-transformers/all-MiniLM-L6-v2", 384),
    "openai": ("text-embedding-3-small", 1536),
    "gemini": ("models/gemini-embedding-001", 3072),
}

LLM_DEFAULTS: dict[str, str] = {
    "ollama": "llama3.2",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Database -----------------------------------------------------------
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/chat_with_website",
        description="psycopg connection string, e.g. postgresql://user:pass@host:5432/dbname",
    )
    db_pool_min: int = 1
    db_pool_max: int = 5

    # --- Embeddings ---------------------------------------------------------
    embedding_provider: EmbeddingProvider = "huggingface"
    embedding_model: str | None = None  # falls back to EMBEDDING_DEFAULTS
    embedding_dim: int | None = None  # falls back to EMBEDDING_DEFAULTS
    embedding_batch_size: int = 64

    # --- Chat LLM -----------------------------------------------------------
    llm_provider: LLMProvider = "ollama"
    llm_model: str | None = None  # falls back to LLM_DEFAULTS
    llm_temperature: float = 0.0
    ollama_base_url: str = "http://localhost:11434"

    # --- API keys (only needed for the paid providers) ----------------------
    openai_api_key: str | None = None
    google_api_key: str | None = None

    # --- Crawler ------------------------------------------------------------
    user_agent: str = "ChatWithWebsiteBot/0.1 (+https://github.com/niranjan-kumar-dev)"
    crawl_max_pages: int = 50
    crawl_max_depth: int = 3
    crawl_delay_seconds: float = 0.5
    crawl_timeout_seconds: float = 15.0
    crawl_respect_robots: bool = True
    crawl_allow_private_hosts: bool = False  # set True only to test against localhost
    min_page_words: int = 50
    boilerplate_line_ratio: float = 0.3  # a line on >30% of pages is treated as boilerplate

    # --- Chunking -----------------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Retrieval ----------------------------------------------------------
    top_k: int = 5
    score_threshold: float = 0.25  # cosine similarity below this is ignored
    history_turns: int = 6  # how many past Q/A pairs to keep in the prompt

    # --- Logging ------------------------------------------------------------
    log_level: str = "INFO"

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_size(cls, v: int, info):
        size = info.data.get("chunk_size", 800)
        if v >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v

    # Derived values ---------------------------------------------------------
    @property
    def resolved_embedding_model(self) -> str:
        return self.embedding_model or EMBEDDING_DEFAULTS[self.embedding_provider][0]

    @property
    def resolved_embedding_dim(self) -> int:
        return self.embedding_dim or EMBEDDING_DEFAULTS[self.embedding_provider][1]

    @property
    def resolved_llm_model(self) -> str:
        return self.llm_model or LLM_DEFAULTS[self.llm_provider]

    def safe_database_url(self) -> str:
        """Connection string with the password hidden, for logs."""
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(self.database_url)
        if parts.password:
            netloc = parts.netloc.replace(f":{parts.password}@", ":***@")
            return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
        return self.database_url


settings = Settings()
