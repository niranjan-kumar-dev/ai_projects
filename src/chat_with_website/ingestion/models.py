"""Plain data containers passed between the ingestion steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class CrawledPage:
    """Raw HTML fetched by the crawler."""

    url: str
    html: str
    status_code: int
    depth: int
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PageContent:
    """Clean text produced by the extractor."""

    url: str
    title: str
    text: str
    content_hash: str
    lines: list[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class Chunk:
    """A piece of a page ready to be embedded and stored."""

    content: str
    chunk_index: int
    metadata: dict
    embedding: list[float] | None = None

    @property
    def embed_text(self) -> str:
        """Text actually sent to the embedding model (title gives extra context)."""
        title = self.metadata.get("title")
        return f"{title}\n\n{self.content}" if title else self.content


@dataclass
class IngestResult:
    website_id: int
    pages_crawled: int = 0
    pages_indexed: int = 0
    pages_skipped_unchanged: int = 0
    pages_skipped_short: int = 0
    chunks_created: int = 0
    errors: list[str] = field(default_factory=list)
