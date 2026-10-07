"""Embedding model factory + batched embedding helper.

An *embedding* is a list of numbers (a vector) that captures the meaning of a
piece of text.  Texts with similar meaning end up close together, which is what
lets us search by meaning instead of by keyword.

Providers (set `EMBEDDING_PROVIDER` in .env):

* `huggingface` - all-MiniLM-L6-v2 runs on your CPU, free, 384 dims (default)
* `openai`      - text-embedding-3-small, 1536 dims, needs OPENAI_API_KEY
* `gemini`      - gemini-embedding-001, 3072 dims, needs GOOGLE_API_KEY

The vector size must match the `chunks.embedding` column, so switching provider
means `python scripts/init_db.py --reset-chunks` and re-ingesting.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from chat_with_website.config import EMBEDDING_DEFAULTS, settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def get_embeddings(provider: str | None = None, model: str | None = None) -> Embeddings:
    """Return a LangChain `Embeddings` object for the configured provider (cached)."""
    provider = provider or settings.embedding_provider
    model = model or (settings.embedding_model if provider == settings.embedding_provider else None) \
        or EMBEDDING_DEFAULTS[provider][0]
    log.info("Loading embedding model %s (%s)", model, provider)

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model,
            encode_kwargs={"normalize_embeddings": True, "batch_size": settings.embedding_batch_size},
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")
        kwargs = {"model": model, "api_key": settings.openai_api_key}
        if settings.embedding_dim and settings.embedding_dim != EMBEDDING_DEFAULTS["openai"][1]:
            kwargs["dimensions"] = settings.embedding_dim
        return OpenAIEmbeddings(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set in .env")
        kwargs = {"model": model, "google_api_key": settings.google_api_key}
        if settings.embedding_dim and settings.embedding_dim != EMBEDDING_DEFAULTS["gemini"][1]:
            kwargs["output_dimensionality"] = settings.embedding_dim
        return GoogleGenerativeAIEmbeddings(**kwargs)

    raise ValueError(f"Unknown embedding provider: {provider}")


def embed_texts(texts: list[str], embeddings: Embeddings | None = None, retries: int = 3) -> list[list[float]]:
    """Embed many texts in batches, validating the vector size."""
    if not texts:
        return []
    embeddings = embeddings or get_embeddings()
    expected = settings.resolved_embedding_dim
    batch_size = settings.embedding_batch_size
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        for attempt in range(1, retries + 1):
            try:
                out = embeddings.embed_documents(batch)
                break
            except Exception as exc:  # network / rate-limit errors from hosted providers
                if attempt == retries:
                    raise
                wait = 2 ** attempt
                log.warning("Embedding batch failed (%s); retrying in %ss", exc, wait)
                time.sleep(wait)
        vectors.extend(out)

    bad = {len(v) for v in vectors if len(v) != expected}
    if bad:
        raise RuntimeError(
            f"Embedding model returned {bad} dims but EMBEDDING_DIM={expected}. "
            "Fix EMBEDDING_PROVIDER/EMBEDDING_DIM in .env and run init_db.py --reset-chunks."
        )
    return vectors


def embed_query(text: str, embeddings: Embeddings | None = None) -> list[float]:
    embeddings = embeddings or get_embeddings()
    return embeddings.embed_query(text)
