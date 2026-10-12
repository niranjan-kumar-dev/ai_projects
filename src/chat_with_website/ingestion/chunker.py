"""Split page text into overlapping chunks with rich metadata.

Why chunk?  Embedding models look at a limited window (all-MiniLM-L6-v2 reads
~256 word-piece tokens, roughly 800-1000 characters) and retrieval works best
when each vector represents one focused idea.  `RecursiveCharacterTextSplitter`
tries to cut on paragraph breaks first, then sentences, then words, so chunks
stay semantically coherent.  Overlap keeps a sentence that straddles a boundary
available in both neighbours.
"""

from __future__ import annotations

from datetime import datetime, timezone

from langchain_text_splitters import RecursiveCharacterTextSplitter

from chat_with_website.config import settings
from chat_with_website.ingestion.models import Chunk, PageContent

SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def build_splitter(chunk_size: int | None = None, chunk_overlap: int | None = None) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
        keep_separator="end",
        strip_whitespace=True,
    )


def chunk_page(
    page: PageContent,
    website_id: int,
    page_id: int,
    splitter: RecursiveCharacterTextSplitter | None = None,
    min_chars: int = 40,
) -> list[Chunk]:
    splitter = splitter or build_splitter()
    pieces = [p.strip() for p in splitter.split_text(page.text)]
    pieces = [p for p in pieces if len(p) >= min_chars]

    crawled_at = datetime.now(timezone.utc).isoformat()
    images = [img for img in page.images if img.description]
    total = len(pieces) + len(images)

    def _meta(idx: int, **extra) -> dict:
        return {
            "chunk_id": f"{page_id}:{idx}",
            "website_id": website_id,
            "page_id": page_id,
            "url": page.url,
            "title": page.title,
            "chunk_index": idx,
            "chunk_count": total,
            "source": "web",
            "crawled_at": crawled_at,
            **extra,
        }

    chunks: list[Chunk] = []
    for idx, piece in enumerate(pieces):
        chunks.append(Chunk(content=piece, chunk_index=idx, metadata=_meta(idx, type="text")))

    # Image chunks: the *description* (alt / caption / heading) is what gets embedded,
    # so "show me the logo" finds the logo through the same vector search as text.
    # The image URL rides along in metadata and is rendered by the UI.
    for img in images:
        idx = len(chunks)
        chunks.append(
            Chunk(
                content=img.description,
                chunk_index=idx,
                metadata=_meta(idx, type="image", image_url=img.url, alt=img.alt),
            )
        )
    return chunks
