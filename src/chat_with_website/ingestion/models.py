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
class PageImage:
    """One image found on a page, with whatever text describes it."""

    url: str
    alt: str = ""       # alt / title attribute
    caption: str = ""   # <figcaption> text
    context: str = ""   # nearest heading or surrounding text

    @property
    def description(self) -> str:
        """Text used to embed/search this image (empty when nothing describes it)."""
        parts = [f"Image: {self.alt}" if self.alt else ""]
        if self.caption:
            parts.append(f"Caption: {self.caption}")
        if self.context:
            parts.append(f"Section: {self.context}")
        return ". ".join(p for p in parts if p)


@dataclass
class PageContent:
    """Clean text produced by the extractor."""

    url: str
    title: str
    text: str
    content_hash: str
    lines: list[str] = field(default_factory=list)
    images: list[PageImage] = field(default_factory=list)

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
