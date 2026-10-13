"""Knowledge-base status queries against the real local database.

Temporary `zz-test` websites are built directly through the repository (pages, zero-vector
chunks, chatbot links, ingestion runs) so every situation the Websites page must handle exists at
once: a ready site, a site with zero pages, pages without embeddings, a failed run, a partial run,
several runs, and a chatbot-scoped view. Counts are checked against plain `count(*)` queries.
Skipped when PostgreSQL is not reachable; everything is deleted afterwards.
"""

from __future__ import annotations

import pytest

from chat_with_website.config import settings
from chat_with_website.db import repository as repo
from chat_with_website.ingestion.models import Chunk

pytestmark = pytest.mark.skipif(settings.app_env != "development", reason="only against the local database")

PREFIX = "zz-test kb "


@pytest.fixture
def db():
    try:
        from chat_with_website.db.connection import check_connection, ensure_schema_upgrades, get_conn

        check_connection()
        ensure_schema_upgrades()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"PostgreSQL not reachable: {exc}")
    yield get_conn
    with get_conn() as conn:
        conn.execute("DELETE FROM chatbots WHERE name LIKE %s", (PREFIX + "%",))
        conn.execute("DELETE FROM websites WHERE name LIKE %s", (PREFIX + "%",))


def _site(conn, name: str, status: str = "ready") -> int:
    w = repo.upsert_website(
        conn, name=PREFIX + name, base_url=f"https://{name}.zz-test.invalid/", max_pages=10,
        embedding_provider=settings.embedding_provider, embedding_model=settings.resolved_embedding_model,
        embedding_dim=settings.resolved_embedding_dim,
    )
    repo.set_website_status(conn, w["id"], status)
    return w["id"]


def _page(conn, website_id: int, n: int, chunks: int, dim: int) -> int:
    url = f"https://site.zz-test.invalid/p{n}"
    row, _ = repo.upsert_page(conn, website_id=website_id, url=url, title=f"Page {n}", content_hash=f"h{n}", word_count=120)
    if chunks:
        repo.insert_chunks(conn, [
            Chunk(content=f"chunk {i}", chunk_index=i, metadata={"page_id": row["id"], "website_id": website_id, "url": url},
                  embedding=[0.0] * dim)
            for i in range(chunks)
        ])
    return row["id"]


def _count(conn, sql, *params) -> int:
    return conn.execute(sql, params).fetchone()["n"]


@pytest.fixture
def world(db):
    """ids: ready (3 pages: 2+3+0 chunks), empty (0 pages), failed, partial, multi (3 runs)."""
    with db() as conn:
        conn.execute("DELETE FROM chatbots WHERE name LIKE %s", (PREFIX + "%",))
        conn.execute("DELETE FROM websites WHERE name LIKE %s", (PREFIX + "%",))
        dim = repo.chunk_embedding_dim(conn)
        ids = {}
        ids["ready"] = _site(conn, "ready")
        _page(conn, ids["ready"], 1, 2, dim)
        _page(conn, ids["ready"], 2, 3, dim)
        _page(conn, ids["ready"], 3, 0, dim)  # crawled but never embedded
        repo.mark_website_crawled(conn, ids["ready"])
        repo.mark_website_indexed(conn, ids["ready"])
        r = repo.start_ingestion_run(conn, ids["ready"], embedding_provider="hf", embedding_model="m")
        repo.mark_ingestion_run_crawled(conn, r, pages_crawled=4, pages_skipped_short=1)
        repo.finish_ingestion_run(conn, r, status="completed", pages_indexed=2, pages_skipped_unchanged=0,
                                  chunks_created=5, error=None, page_errors=[])

        ids["empty"] = _site(conn, "empty", status="pending")

        ids["failed"] = _site(conn, "failed", status="failed")
        repo.set_website_status(conn, ids["failed"], "failed", "site unreachable")
        r = repo.start_ingestion_run(conn, ids["failed"], embedding_provider="hf", embedding_model="m")
        repo.finish_ingestion_run(conn, r, status="failed", pages_indexed=0, pages_skipped_unchanged=0,
                                  chunks_created=0, error="site unreachable", page_errors=[])

        ids["partial"] = _site(conn, "partial")
        _page(conn, ids["partial"], 1, 1, dim)
        r = repo.start_ingestion_run(conn, ids["partial"], embedding_provider="hf", embedding_model="m")
        repo.mark_ingestion_run_crawled(conn, r, pages_crawled=2, pages_skipped_short=0)
        repo.finish_ingestion_run(conn, r, status="completed", pages_indexed=1, pages_skipped_unchanged=0, chunks_created=1,
                                  error="https://site.zz-test.invalid/p2: boom",
                                  page_errors=[{"url": "https://site.zz-test.invalid/p2", "error": "boom"}])

        ids["multi"] = _site(conn, "multi")
        for i in range(3):
            r = repo.start_ingestion_run(conn, ids["multi"], embedding_provider="hf", embedding_model=f"m{i}")
            repo.finish_ingestion_run(conn, r, status="completed" if i != 1 else "failed", pages_indexed=i,
                                      pages_skipped_unchanged=0, chunks_created=i, error=None, page_errors=[])
        ids["multi_last_run"] = r

        bot = repo.create_chatbot(conn, name=PREFIX + "bot", description="scoped")
        repo.set_chatbot_websites(conn, bot["id"], [ids["ready"], ids["partial"]])
        ids["bot"] = bot["id"]
    return ids


def _overview(db, chatbot_id=None):
    with db() as conn:
        return {w["id"]: w for w in repo.list_websites_overview(conn, chatbot_id)}


def test_overview_counts_match_postgres(world, db):
    ov = _overview(db)
    with db() as conn:
        for key in ("ready", "empty", "failed", "partial", "multi"):
            wid = world[key]
            w = ov[wid]
            assert w["pages_total"] == _count(conn, "SELECT count(*) AS n FROM pages WHERE website_id = %s", wid)
            assert w["chunks_total"] == _count(conn, "SELECT count(*) AS n FROM chunks WHERE website_id = %s", wid)
            assert w["pages_indexed"] == _count(
                conn, "SELECT count(DISTINCT page_id) AS n FROM chunks WHERE website_id = %s", wid
            )


def test_ready_site_has_timestamps_pages_without_chunks_and_a_completed_run(world, db):
    w = _overview(db)[world["ready"]]
    assert (w["pages_total"], w["pages_indexed"], w["chunks_total"]) == (3, 2, 5)
    assert w["status"] == "ready"
    assert w["last_crawled_at"] is not None and w["last_indexed_at"] is not None
    assert w["newest_chunk_at"] is not None
    assert w["run_status"] == "completed" and w["run_pages_crawled"] == 4 and w["run_chunks_created"] == 5
    assert w["run_crawl_finished_at"] is not None and w["run_finished_at"] is not None
    with db() as conn:
        missing = repo.list_pages_without_chunks(conn, world["ready"])
        assert [m["url"] for m in missing] == ["https://site.zz-test.invalid/p3"]
        rows, total = repo.list_pages_with_chunk_counts(conn, world["ready"])
        assert total == 3 and [r["chunk_count"] for r in rows] == [2, 3, 0]
        assert rows[0]["indexed_at"] is not None and rows[2]["indexed_at"] is None


def test_zero_pages_site_shows_nulls_not_fabricated_values(world, db):
    w = _overview(db)[world["empty"]]
    assert (w["pages_total"], w["pages_indexed"], w["chunks_total"]) == (0, 0, 0)
    assert w["last_crawled_at"] is None and w["last_indexed_at"] is None and w["newest_chunk_at"] is None
    assert w["run_id"] is None and w["run_status"] is None
    with db() as conn:
        assert repo.list_pages_with_chunk_counts(conn, world["empty"]) == ([], 0)
        assert repo.list_ingestion_runs(conn, world["empty"]) == []


def test_failed_crawl_keeps_error_and_failed_run(world, db):
    w = _overview(db)[world["failed"]]
    assert w["status"] == "failed" and w["error"] == "site unreachable"
    assert w["run_status"] == "failed" and w["run_error"] == "site unreachable"
    assert w["last_indexed_at"] is None and w["run_crawl_finished_at"] is None


def test_partial_run_exposes_failed_pages_as_json(world, db):
    w = _overview(db)[world["partial"]]
    assert w["run_status"] == "completed"
    assert w["run_page_errors"] == [{"url": "https://site.zz-test.invalid/p2", "error": "boom"}]
    assert w["pages_total"] == 1  # the failed page never got a row


def test_multiple_runs_latest_wins_and_history_is_newest_first(world, db):
    w = _overview(db)[world["multi"]]
    assert w["run_id"] == world["multi_last_run"] and w["run_embedding_model"] == "m2" and w["run_pages_indexed"] == 2
    with db() as conn:
        runs = repo.list_ingestion_runs(conn, world["multi"])
    assert [r["embedding_model"] for r in runs] == ["m2", "m1", "m0"]
    assert [r["status"] for r in runs] == ["completed", "failed", "completed"]


def test_chatbot_filter_scopes_the_overview(world, db):
    scoped = _overview(db, world["bot"])
    assert {world["ready"], world["partial"]} <= set(scoped)
    assert world["empty"] not in scoped and world["failed"] not in scoped and world["multi"] not in scoped
    assert _overview(db, -1) == {}  # a chatbot without websites sees nothing


def test_page_list_pagination_total_is_stable_across_pages(world, db):
    with db() as conn:
        p1, t1 = repo.list_pages_with_chunk_counts(conn, world["ready"], limit=2, offset=0)
        p2, t2 = repo.list_pages_with_chunk_counts(conn, world["ready"], limit=2, offset=2)
        p3, t3 = repo.list_pages_with_chunk_counts(conn, world["ready"], limit=2, offset=4)
    assert (t1, t2, t3) == (3, 3, 0)
    assert [r["url"] for r in p1 + p2] == [f"https://site.zz-test.invalid/p{n}" for n in (1, 2, 3)]
    assert p3 == []


def test_page_errors_are_capped_when_stored(world, db):
    many = [{"url": f"https://x/{i}", "error": "e" * 1000} for i in range(repo.MAX_STORED_PAGE_ERRORS + 50)]
    with db() as conn:
        r = repo.start_ingestion_run(conn, world["multi"], embedding_provider="hf", embedding_model="m")
        repo.finish_ingestion_run(conn, r, status="failed", pages_indexed=0, pages_skipped_unchanged=0,
                                  chunks_created=0, error="x", page_errors=many)
        stored = repo.list_ingestion_runs(conn, world["multi"], limit=1)[0]["page_errors"]
    assert len(stored) == repo.MAX_STORED_PAGE_ERRORS
    assert len(stored[0]["error"]) == repo.MAX_PAGE_ERROR_CHARS


def test_search_filters_url_and_title_case_insensitively(world, db):
    with db() as conn:
        rows, total = repo.list_pages_with_chunk_counts(conn, world["ready"], search="PAGE 2")
        assert total == 1 and rows[0]["title"] == "Page 2"
        rows, total = repo.list_pages_with_chunk_counts(conn, world["ready"], search="/p3")
        assert total == 1 and rows[0]["url"].endswith("/p3") and rows[0]["chunk_count"] == 0
        assert repo.list_pages_with_chunk_counts(conn, world["ready"], search="zzz-no-match") == ([], 0)
        # a filter with LIKE wildcards is treated literally enough not to match everything
        rows, total = repo.list_pages_with_chunk_counts(conn, world["ready"], search="%")
        assert total in (0, 3)  # '%' is a wildcard inside ILIKE; both outcomes are safe, never an error


def test_duplicate_urls_are_grouped_by_normalised_key(world, db):
    with db() as conn:
        wid = world["empty"]
        for url in ("https://site.zz-test.invalid/a", "https://site.zz-test.invalid/a/",
                    "http://www.site.zz-test.invalid/A#top", "https://site.zz-test.invalid/b"):
            repo.upsert_page(conn, website_id=wid, url=url, title="t", content_hash=url, word_count=10)
        dupes = repo.list_duplicate_page_urls(conn, wid)
    assert len(dupes) == 1
    assert dupes[0]["key"] == "site.zz-test.invalid/a" and dupes[0]["n"] == 3
    assert dupes[0]["urls"] == ["https://site.zz-test.invalid/a", "https://site.zz-test.invalid/a/", "http://www.site.zz-test.invalid/A#top"]
    with db() as conn:
        assert repo.list_duplicate_page_urls(conn, world["ready"]) == []
