"""`ingest_website` run bookkeeping, with the crawler, embedder and database faked.

What is checked is *when* the pipeline records what: the run row before the crawl, the crawl
timestamp before indexing, the indexed timestamp only on success, page failures as structured
errors on a run that still completes, and an aborted run marked failed before the error is re-raised.
"""

from __future__ import annotations

import pytest

import chat_with_website.ingestion.pipeline as pipeline
from chat_with_website.config import settings
from chat_with_website.db import repository as repo
from chat_with_website.ingestion.models import Chunk, PageContent


def _page(url: str) -> PageContent:
    return PageContent(url=url, title="T", text="word " * 100, content_hash=f"hash-{url}")


class _Conn:
    """A stand-in connection: `ANALYZE` is the only raw statement the pipeline runs."""

    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.calls.append(("execute", sql))


@pytest.fixture
def fake_db(monkeypatch):
    calls: list[tuple] = []
    state = {"fail_pages": set(), "fail_crawl": None, "crawled": None, "existing": {}}

    def rec(_name, **kw):
        calls.append((_name, kw))

    monkeypatch.setattr(pipeline, "get_conn", lambda: _Conn(calls))
    monkeypatch.setattr(repo, "chunk_embedding_dim", lambda conn: settings.resolved_embedding_dim)
    monkeypatch.setattr(repo, "upsert_website", lambda conn, **kw: rec("upsert_website", **kw) or {"id": 42})
    monkeypatch.setattr(repo, "set_website_status", lambda conn, wid, status, error=None: rec("set_website_status", status=status, error=error))
    monkeypatch.setattr(repo, "start_ingestion_run", lambda conn, wid, **kw: rec("start_ingestion_run", website_id=wid, **kw) or 7)
    monkeypatch.setattr(repo, "mark_website_crawled", lambda conn, wid: rec("mark_website_crawled", website_id=wid))
    monkeypatch.setattr(repo, "mark_ingestion_run_crawled", lambda conn, rid, **kw: rec("mark_ingestion_run_crawled", run_id=rid, **kw))
    monkeypatch.setattr(repo, "mark_website_indexed", lambda conn, wid: rec("mark_website_indexed", website_id=wid))
    monkeypatch.setattr(repo, "finish_ingestion_run", lambda conn, rid, **kw: rec("finish_ingestion_run", run_id=rid, **kw))
    monkeypatch.setattr(repo, "refresh_website_counts", lambda conn, wid: rec("refresh_website_counts"))
    monkeypatch.setattr(repo, "delete_stale_pages", lambda conn, wid, urls: rec("delete_stale_pages", urls=list(urls)) or 0)
    monkeypatch.setattr(repo, "page_has_chunks", lambda conn, pid: False)
    monkeypatch.setattr(repo, "upsert_page", lambda conn, **kw: ({"id": hash(kw["url"]) % 1000}, True))
    monkeypatch.setattr(repo, "insert_chunks", lambda conn, chunks: rec("insert_chunks", n=len(chunks)) or len(chunks))

    def fake_crawl(start_url, max_pages=None, progress=None):
        if state["fail_crawl"]:
            raise state["fail_crawl"]
        pages = state["crawled"] if state["crawled"] is not None else [_page("https://example.com/a"), _page("https://example.com/b")]
        return pages, (len(pages) + 1 if pages else 0), (1 if pages else 0)  # one page was too short

    monkeypatch.setattr(pipeline, "crawl_and_extract", fake_crawl)
    monkeypatch.setattr(pipeline, "get_embeddings", lambda: object())
    monkeypatch.setattr(pipeline, "build_splitter", lambda: None)

    def fake_chunk_page(page, *, website_id, page_id, splitter):
        if page.url in state["fail_pages"]:
            raise RuntimeError("embedding exploded")
        return [Chunk(content="c", chunk_index=0, metadata={"page_id": page_id, "website_id": website_id, "url": page.url})]

    monkeypatch.setattr(pipeline, "chunk_page", fake_chunk_page)
    monkeypatch.setattr(pipeline, "embed_texts", lambda texts, emb: [[0.0] * 3 for _ in texts])
    monkeypatch.setattr(settings, "crawl_allow_private_hosts", False)
    return calls, state


def _names(calls):
    return [c[0] for c in calls if c[0] != "execute"]


def _call(calls, name):
    return [c[1] for c in calls if c[0] == name]


def test_successful_run_records_start_crawl_and_index_in_order(fake_db):
    calls, _ = fake_db
    result = pipeline.ingest_website("https://example.com")
    assert result.run_id == 7
    assert result.pages_crawled == 3 and result.pages_indexed == 2 and result.chunks_created == 2
    assert result.page_errors == [] and result.errors == []
    names = _names(calls)
    # the run row is created together with the "crawling" status, before anything else
    assert names[:3] == ["upsert_website", "set_website_status", "start_ingestion_run"]
    assert _call(calls, "start_ingestion_run")[0]["embedding_model"] == settings.resolved_embedding_model
    # the crawl is recorded (website + run) before any page is indexed
    assert names.index("mark_website_crawled") < names.index("insert_chunks")
    assert names.index("mark_ingestion_run_crawled") < names.index("insert_chunks")
    assert _call(calls, "mark_ingestion_run_crawled")[0] == {"run_id": 7, "pages_crawled": 3, "pages_skipped_short": 1}
    # ready -> last_indexed_at is written and the run completes
    assert _call(calls, "set_website_status")[-1] == {"status": "ready", "error": None}
    assert "mark_website_indexed" in names
    finish = _call(calls, "finish_ingestion_run")
    assert len(finish) == 1 and finish[0]["status"] == "completed" and finish[0]["page_errors"] == []
    assert finish[0]["pages_indexed"] == 2 and finish[0]["chunks_created"] == 2


def test_partial_failure_keeps_run_completed_and_stores_page_errors(fake_db):
    calls, state = fake_db
    state["fail_pages"] = {"https://example.com/b"}
    result = pipeline.ingest_website("https://example.com")
    assert result.pages_indexed == 1
    assert result.page_errors == [{"url": "https://example.com/b", "error": "embedding exploded"}]
    assert result.errors == ["https://example.com/b: embedding exploded"]
    assert _call(calls, "set_website_status")[-1] == {"status": "ready", "error": "https://example.com/b: embedding exploded"}
    assert "mark_website_indexed" in _names(calls)
    finish = _call(calls, "finish_ingestion_run")[0]
    assert finish["status"] == "completed" and finish["page_errors"] == result.page_errors


def test_every_page_failing_marks_website_and_run_failed_without_indexed_timestamp(fake_db):
    calls, state = fake_db
    state["fail_pages"] = {"https://example.com/a", "https://example.com/b"}
    pipeline.ingest_website("https://example.com")
    names = _names(calls)
    assert "mark_website_crawled" in names  # the crawl itself worked
    assert "mark_website_indexed" not in names
    assert _call(calls, "set_website_status")[-1]["status"] == "failed"
    assert _call(calls, "finish_ingestion_run")[0]["status"] == "failed"
    assert len(_call(calls, "finish_ingestion_run")[0]["page_errors"]) == 2


def test_crawl_failure_marks_run_failed_and_reraises(fake_db):
    calls, state = fake_db
    state["fail_crawl"] = RuntimeError("site unreachable")
    with pytest.raises(RuntimeError, match="site unreachable"):
        pipeline.ingest_website("https://example.com")
    names = _names(calls)
    assert "mark_website_crawled" not in names and "mark_ingestion_run_crawled" not in names
    assert "mark_website_indexed" not in names
    assert _call(calls, "set_website_status")[-1] == {"status": "failed", "error": "site unreachable"}
    finish = _call(calls, "finish_ingestion_run")
    assert len(finish) == 1 and finish[0]["status"] == "failed" and finish[0]["error"] == "site unreachable"


def test_empty_crawl_records_the_crawl_attempt_but_not_a_successful_crawl(fake_db):
    calls, state = fake_db
    state["crawled"] = []
    with pytest.raises(RuntimeError, match="No indexable pages"):
        pipeline.ingest_website("https://example.com")
    names = _names(calls)
    assert "mark_website_crawled" not in names  # zero pages: not a successful crawl
    assert "mark_ingestion_run_crawled" in names  # but the attempt is on the run row
    assert _call(calls, "finish_ingestion_run")[0]["status"] == "failed"


def test_db_outage_while_recording_failure_still_raises_the_original_error(fake_db, monkeypatch):
    calls, state = fake_db
    state["fail_crawl"] = RuntimeError("site unreachable")

    def boom(conn, wid, status, error=None):
        if status == "failed":  # the database dies exactly when the failure should be recorded
            raise ConnectionError("db gone")

    monkeypatch.setattr(repo, "set_website_status", boom)
    with pytest.raises(RuntimeError, match="site unreachable"):
        pipeline.ingest_website("https://example.com")
