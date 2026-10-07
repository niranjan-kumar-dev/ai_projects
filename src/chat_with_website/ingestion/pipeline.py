"""The ingestion pipeline: URL -> crawl -> extract -> chunk -> embed -> store.

`ingest_website` is the single entry point used by the CLI and the Streamlit UI.
"""

from __future__ import annotations

import logging
from typing import Callable

from chat_with_website.config import settings
from chat_with_website.db import repository as repo
from chat_with_website.db.connection import get_conn
from chat_with_website.ingestion.chunker import build_splitter, chunk_page
from chat_with_website.ingestion.crawler import Crawler
from chat_with_website.ingestion.embedder import embed_texts, get_embeddings
from chat_with_website.ingestion.extractor import extract_page, remove_boilerplate
from chat_with_website.ingestion.models import IngestResult, PageContent
from chat_with_website.ingestion.url_utils import registrable_host, validate_start_url

log = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None]


def _progress(cb: ProgressCallback | None, msg: str, done: int, total: int) -> None:
    log.info("%s (%d/%d)", msg, done, total)
    if cb:
        cb(msg, done, total)


def crawl_and_extract(
    start_url: str,
    max_pages: int | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[list[PageContent], int, int]:
    """Steps 1-2 only (no database).  Returns (pages, crawled_count, skipped_short)."""
    crawler = Crawler(start_url, max_pages=max_pages)
    extracted: list[PageContent] = []
    crawled = 0
    for raw in crawler.crawl(progress=progress):
        crawled += 1
        page = extract_page(raw)
        extracted.append(page)

    cleaned = remove_boilerplate(extracted)
    min_words = settings.min_page_words
    kept = [p for p in cleaned if p.word_count >= min_words]
    skipped = len(cleaned) - len(kept)
    if skipped:
        log.info("Skipped %d page(s) with fewer than %d words", skipped, min_words)
    return kept, crawled, skipped


def ingest_website(
    url: str,
    name: str | None = None,
    max_pages: int | None = None,
    progress: ProgressCallback | None = None,
    prune_missing: bool = True,
) -> IngestResult:
    """Crawl `url`, index its pages into PostgreSQL and return a summary.

    Re-running on the same URL is safe: unchanged pages (same content hash) are
    skipped, changed pages are re-embedded, pages that disappeared are removed
    when `prune_missing` is True.
    """
    start_url = validate_start_url(url, allow_private=settings.crawl_allow_private_hosts)
    name = name or registrable_host(start_url)
    max_pages = max_pages or settings.crawl_max_pages

    provider = settings.embedding_provider
    model = settings.resolved_embedding_model
    dim = settings.resolved_embedding_dim

    # -- 0. register the website and sanity-check the vector column -------
    with get_conn() as conn:
        table_dim = repo.chunk_embedding_dim(conn)
        if table_dim is None:
            raise RuntimeError("Table `chunks` does not exist. Run `python scripts/init_db.py` first.")
        if table_dim != dim:
            raise RuntimeError(
                f"chunks.embedding is vector({table_dim}) but the configured model {model} produces {dim} dims. "
                "Run `python scripts/init_db.py --reset-chunks` after changing EMBEDDING_PROVIDER."
            )
        website = repo.upsert_website(
            conn, name=name, base_url=start_url, max_pages=max_pages,
            embedding_provider=provider, embedding_model=model, embedding_dim=dim,
        )
        repo.set_website_status(conn, website["id"], "crawling")
    website_id = website["id"]
    result = IngestResult(website_id=website_id)

    try:
        # -- 1-2. crawl + extract (network only) ------------------------------
        pages, result.pages_crawled, result.pages_skipped_short = crawl_and_extract(
            start_url, max_pages=max_pages, progress=progress
        )
        if not pages:
            raise RuntimeError("No indexable pages found (site may be JavaScript-rendered or blocked).")

        # -- 3-5. chunk, embed, store - one page per transaction -------------
        embeddings = get_embeddings()
        splitter = build_splitter()
        total = len(pages)
        for i, page in enumerate(pages, start=1):
            _progress(progress, f"Indexing {page.url}", i, total)
            try:
                with get_conn() as conn:
                    page_row, changed = repo.upsert_page(
                        conn, website_id=website_id, url=page.url, title=page.title,
                        content_hash=page.content_hash, word_count=page.word_count,
                    )
                    if not changed:
                        result.pages_skipped_unchanged += 1
                        continue
                    chunks = chunk_page(page, website_id=website_id, page_id=page_row["id"], splitter=splitter)
                    vectors = embed_texts([c.embed_text for c in chunks], embeddings)
                    for c, v in zip(chunks, vectors):
                        c.embedding = v
                    result.chunks_created += repo.insert_chunks(conn, chunks)
                    result.pages_indexed += 1
            except Exception as exc:  # keep going with the other pages
                msg = f"{page.url}: {exc}"
                log.exception("Failed to index page %s", page.url)
                result.errors.append(msg)

        with get_conn() as conn:
            if prune_missing:
                removed = repo.delete_stale_pages(conn, website_id, [p.url for p in pages])
                if removed:
                    log.info("Removed %d page(s) no longer present on the site", removed)
            repo.refresh_website_counts(conn, website_id)
            status = "ready" if result.pages_indexed or result.pages_skipped_unchanged else "failed"
            repo.set_website_status(conn, website_id, status, "; ".join(result.errors[:5]) or None)
            conn.execute("ANALYZE chunks")
    except Exception as exc:
        log.exception("Ingestion failed for %s", start_url)
        with get_conn() as conn:
            repo.set_website_status(conn, website_id, "failed", str(exc))
        result.errors.append(str(exc))
        raise

    log.info(
        "Done: %d crawled, %d indexed, %d unchanged, %d too short, %d chunks, %d errors",
        result.pages_crawled, result.pages_indexed, result.pages_skipped_unchanged,
        result.pages_skipped_short, result.chunks_created, len(result.errors),
    )
    return result
