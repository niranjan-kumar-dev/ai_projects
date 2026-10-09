"""Application configuration.

All runtime settings come from environment variables (or a `.env` file in the
project root).  Nothing secret lives in code.  Access the single `settings`
instance everywhere::

    from chat_with_website.config import settings
    settings.database_url
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AppEnv = Literal["development", "production"]
EmbeddingProvider = Literal["huggingface", "openai", "gemini"]
LLMProvider = Literal["ollama", "openai", "gemini"]

# Which *_DATABASE_URL variable each environment uses.
DATABASE_URL_BY_ENV: dict[str, str] = {
    "development": "LOCAL_DATABASE_URL",
    "production": "PRODUCTION_DATABASE_URL",
}

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
    "gemini": "gemini-3.8-flash",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Environment & database ---------------------------------------------
    # APP_ENV decides which database URL is used:
    #   development -> LOCAL_DATABASE_URL        (your machine)
    #   production  -> PRODUCTION_DATABASE_URL   (Streamlit Cloud / server)
    # DATABASE_URL, if set, overrides both (kept for backward compatibility).
    # The rest of the application only ever reads `settings.database_url`.
    app_env: AppEnv = "development"
    local_database_url: str | None = Field(default=None, description="used when APP_ENV=development")
    production_database_url: str | None = Field(default=None, description="used when APP_ENV=production")
    database_url: str | None = Field(
        default=None,
        description="resolved automatically from APP_ENV; set explicitly only to override",
    )
    db_pool_min: int = 1
    db_pool_max: int = 5

    # --- Embeddings ---------------------------------------------------------
    embedding_provider: EmbeddingProvider = "huggingface"
    embedding_model: str | None = None  # falls back to EMBEDDING_DEFAULTS
    embedding_dim: int | None = None  # falls back to EMBEDDING_DEFAULTS
    embedding_batch_size: int = 64

    # --- Chat LLM -----------------------------------------------------------
    llm_provider: LLMProvider = "openai"
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

    @field_validator("app_env", mode="before")
    @classmethod
    def _normalise_app_env(cls, v):
        if isinstance(v, str):
            v = v.strip().lower()
        if v not in DATABASE_URL_BY_ENV:
            raise ValueError(
                f"APP_ENV must be one of {sorted(DATABASE_URL_BY_ENV)}, got {v!r}. "
                "Set it in .env (local) or in the deployment's environment variables / secrets."
            )
        return v

    @field_validator("database_url", "local_database_url", "production_database_url", mode="before")
    @classmethod
    def _blank_is_none(cls, v):
        # `DATABASE_URL=` (empty) in .env should behave like "not set"
        return None if isinstance(v, str) and not v.strip() else v

    @model_validator(mode="after")
    def _resolve_database_url(self):
        """Pick the database for the current APP_ENV. This is the only place that knows
        about LOCAL_/PRODUCTION_ URLs; everything else uses `settings.database_url`."""
        if self.database_url:  # explicit override wins (backward compatible)
            return self
        var_name = DATABASE_URL_BY_ENV[self.app_env]
        chosen = self.local_database_url if self.app_env == "development" else self.production_database_url
        if not chosen:
            raise ValueError(
                f"APP_ENV={self.app_env} but {var_name} is not set. "
                f"Add {var_name}=postgresql://user:password@host:5432/dbname to .env "
                "(see .env.example) or to the deployment environment."
            )
        self.database_url = chosen
        return self

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

        parts = urlsplit(self.database_url or "")
        if parts.password:
            netloc = parts.netloc.replace(f":{parts.password}@", ":***@")
            return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
        return self.database_url or ""

    def database_label(self) -> str:
        """Short, secret-free description for logs and the UI, e.g.
        'development → localhost:5432/chat_with_website'."""
        from urllib.parse import urlsplit

        parts = urlsplit(self.database_url or "")
        host = parts.hostname or "?"
        port = f":{parts.port}" if parts.port else ""
        db = parts.path.lstrip("/") or "?"
        return f"{self.app_env} → {host}{port}/{db}"


def load_settings() -> Settings:
    """Build the Settings object, turning validation problems into one readable line."""
    try:
        return Settings()
    except ValidationError as exc:
        problems = "; ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors())
        print(f"Configuration error: {problems}", file=sys.stderr)
        sys.exit(1)


settings = load_settings()
