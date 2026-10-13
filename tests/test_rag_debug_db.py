"""RAG Debugger retrieval against the real local database, without any embedding API.

A stored chunk's own vector is used as the query (`vector=`), so no OpenAI/HuggingFace call
happens: the chunk must come back first with a score of ~1.0, its text must equal the stored row,
and chunks of a website that is *not* attached to the chatbot must never appear.
"""

from __future__ import annotations

import numpy as np
import pytest

import chat_with_website.rag.debug as dbg
from chat_with_website.config import settings
from chat_with_website.db import repository as repo
from chat_with_website.ingestion.models import Chunk

pytestmark = pytest.mark.skipif(settings.app_env != "development", reason="only against the local database")
PREFIX = "zz-test rdbg "


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


def _site(conn, name):
    w = repo.upsert_website(conn, name=PREFIX + name, base_url=f"https://{name}.rdbg.zz-test.invalid/", max_pages=5,
                            embedding_provider=settings.embedding_provider,
                            embedding_model=settings.resolved_embedding_model, embedding_dim=settings.resolved_embedding_dim)
    repo.set_website_status(conn, w["id"], "ready")
    return w["id"]


def _vec(v) -> list[float]:
    """A stored `embedding` comes back as a pgvector Vector (or numpy array)."""
    return v.to_list() if hasattr(v, "to_list") else list(v)


def _unit(dim, rng):
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def world(db):
    rng = np.random.default_rng(42)
    with db() as conn:
        conn.execute("DELETE FROM chatbots WHERE name LIKE %s", (PREFIX + "%",))
        conn.execute("DELETE FROM websites WHERE name LIKE %s", (PREFIX + "%",))
        dim = repo.chunk_embedding_dim(conn)
        a, b = _site(conn, "attached"), _site(conn, "other")
        ids = {"a": a, "b": b, "chunks": {}}
        for site, tag in ((a, "a"), (b, "b")):
            for n in range(3):
                url = f"https://{tag}.rdbg.zz-test.invalid/p{n}"
                page, _ = repo.upsert_page(conn, website_id=site, url=url, title=f"{tag} page {n}" if n else None,
                                           content_hash=f"{tag}{n}", word_count=50)
                repo.insert_chunks(conn, [Chunk(content=f"stored text {tag}{n}", chunk_index=0,
                                                metadata={"page_id": page["id"], "website_id": site, "url": url,
                                                          "chunk_id": f"{page['id']}:0", "type": "text"},
                                                embedding=_unit(dim, rng))])
                ids["chunks"][f"{tag}{n}"] = conn.execute(
                    "SELECT id FROM chunks WHERE page_id = %s AND chunk_index = 0", (page["id"],)).fetchone()["id"]
        bot = repo.create_chatbot(conn, name=PREFIX + "bot", top_k=2)
        repo.set_chatbot_websites(conn, bot["id"], [a])
        ids["bot"] = bot["id"]
    return ids


def test_stored_vector_retrieves_itself_scoped_to_the_chatbot(world, db):
    cid = world["chunks"]["a1"]
    with db() as conn:
        row = conn.execute("SELECT content, embedding, page_id FROM chunks WHERE id = %s", (cid,)).fetchone()
    t = dbg.run_retrieval_test(world["bot"], "anything", k=2, vector=_vec(row["embedding"]))
    assert t.website_ids == [world["a"]] and t.skipped_reason is None
    assert len(t.chunks) == 2  # top_k respected
    first = t.chunks[0]
    assert first.db_id == cid and first.score == pytest.approx(1.0, abs=5e-3)
    assert first.text == row["content"] == "stored text a1"  # the card shows the stored row verbatim
    assert first.document.metadata["page_id"] == row["page_id"]
    assert first.kept and first.chunk_id == f"{row['page_id']}:0"
    assert all(c.website_id == world["a"] for c in t.chunks)  # website B never leaks in
    assert t.embed_ms == 0.0 and t.search_ms > 0 and t.total_ms >= t.search_ms


def test_missing_page_title_is_not_available(world, db):
    cid = world["chunks"]["a0"]  # page 0 has no title
    with db() as conn:
        vec = conn.execute("SELECT embedding FROM chunks WHERE id = %s", (cid,)).fetchone()["embedding"]
    t = dbg.run_retrieval_test(world["bot"], "q", k=1, vector=_vec(vec))
    assert t.chunks[0].db_id == cid and t.chunks[0].title is None


def test_compare_top_k_limits_hold_on_the_real_index(world, db, monkeypatch):
    cid = world["chunks"]["a2"]
    with db() as conn:
        vec = conn.execute("SELECT embedding FROM chunks WHERE id = %s", (cid,)).fetchone()["embedding"]
    monkeypatch.setattr(dbg, "get_embeddings", lambda: None)
    monkeypatch.setattr(dbg, "embed_query", lambda text, emb=None: _vec(vec))  # no API call
    tests = dbg.compare_top_k(world["bot"], "q", ks=(1, 2, 3, 10))
    assert [len(t.chunks) for t in tests] == [1, 2, 3, 3]  # only 3 chunks exist on the attached site
    assert all(c.website_id == world["a"] for t in tests for c in t.chunks)
