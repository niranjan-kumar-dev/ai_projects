"""All SQL lives here.  Every function takes an open psycopg connection.

Keeping SQL in one module makes it easy to review for injection issues (we only
ever use parameterised queries) and to swap storage later.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import psycopg
from psycopg.types.json import Jsonb

from chat_with_website.ingestion.models import Chunk

log = logging.getLogger(__name__)

# pgvector: HNSW on `vector` supports <= 2000 dims; above that the index (and the
# ORDER BY expression that must match it) use halfvec, which supports up to 4000.
HALFVEC_THRESHOLD = 2000


def _distance_expr(dim: int) -> str:
    """SQL cosine-distance expression matching the HNSW index built by init_db.py."""
    if dim > HALFVEC_THRESHOLD:
        return f"(c.embedding::halfvec({dim}) <=> %(q)s::halfvec({dim}))"
    return "(c.embedding <=> %(q)s)"


# ----------------------------------------------------------------- websites
def get_website_by_url(conn: psycopg.Connection, base_url: str) -> dict | None:
    return conn.execute("SELECT * FROM websites WHERE base_url = %s", (base_url,)).fetchone()


def get_website(conn: psycopg.Connection, website_id: int) -> dict | None:
    return conn.execute("SELECT * FROM websites WHERE id = %s", (website_id,)).fetchone()


def list_websites(conn: psycopg.Connection) -> list[dict]:
    return conn.execute("SELECT * FROM websites ORDER BY id").fetchall()


def upsert_website(
    conn: psycopg.Connection,
    *,
    name: str,
    base_url: str,
    max_pages: int,
    embedding_provider: str,
    embedding_model: str,
    embedding_dim: int,
) -> dict:
    """Create the website row or update its settings if it already exists."""
    return conn.execute(
        """
        INSERT INTO websites (name, base_url, max_pages, embedding_provider, embedding_model, embedding_dim)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (base_url) DO UPDATE
            SET name = EXCLUDED.name,
                max_pages = EXCLUDED.max_pages,
                embedding_provider = EXCLUDED.embedding_provider,
                embedding_model = EXCLUDED.embedding_model,
                embedding_dim = EXCLUDED.embedding_dim
        RETURNING *
        """,
        (name, base_url, max_pages, embedding_provider, embedding_model, embedding_dim),
    ).fetchone()


def set_website_status(conn: psycopg.Connection, website_id: int, status: str, error: str | None = None) -> None:
    conn.execute(
        "UPDATE websites SET status = %s, error = %s WHERE id = %s",
        (status, error, website_id),
    )


def refresh_website_counts(conn: psycopg.Connection, website_id: int) -> None:
    conn.execute(
        """
        UPDATE websites w SET
            page_count  = (SELECT count(*) FROM pages  WHERE website_id = w.id),
            chunk_count = (SELECT count(*) FROM chunks WHERE website_id = w.id),
            last_crawled_at = now()
        WHERE w.id = %s
        """,
        (website_id,),
    )


def delete_website(conn: psycopg.Connection, website_id: int) -> None:
    conn.execute("DELETE FROM websites WHERE id = %s", (website_id,))  # cascades to pages/chunks


# -------------------------------------------------------------------- pages
def get_page(conn: psycopg.Connection, website_id: int, url: str) -> dict | None:
    return conn.execute(
        "SELECT * FROM pages WHERE website_id = %s AND url = %s", (website_id, url)
    ).fetchone()


def upsert_page(
    conn: psycopg.Connection, *, website_id: int, url: str, title: str, content_hash: str, word_count: int
) -> tuple[dict, bool]:
    """Insert or update a page.  Returns (row, changed).

    `changed` is False when the page already existed with the same content
    hash, which lets the pipeline skip re-embedding unchanged pages.
    """
    existing = get_page(conn, website_id, url)
    if existing and existing["content_hash"] == content_hash:
        return existing, False

    row = conn.execute(
        """
        INSERT INTO pages (website_id, url, title, content_hash, word_count)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (website_id, url) DO UPDATE
            SET title = EXCLUDED.title,
                content_hash = EXCLUDED.content_hash,
                word_count = EXCLUDED.word_count,
                crawled_at = now()
        RETURNING *
        """,
        (website_id, url, title, content_hash, word_count),
    ).fetchone()
    if existing:  # content changed -> old chunks are stale
        conn.execute("DELETE FROM chunks WHERE page_id = %s", (row["id"],))
    return row, True


def page_has_chunks(conn: psycopg.Connection, page_id: int) -> bool:
    return conn.execute("SELECT 1 FROM chunks WHERE page_id = %s LIMIT 1", (page_id,)).fetchone() is not None


def list_pages(conn: psycopg.Connection, website_id: int) -> list[dict]:
    return conn.execute(
        "SELECT id, url, title, word_count, crawled_at FROM pages WHERE website_id = %s ORDER BY id",
        (website_id,),
    ).fetchall()


def delete_stale_pages(conn: psycopg.Connection, website_id: int, keep_urls: Sequence[str]) -> int:
    """Remove pages that were not seen in the latest crawl."""
    cur = conn.execute(
        "DELETE FROM pages WHERE website_id = %s AND NOT (url = ANY(%s))",
        (website_id, list(keep_urls)),
    )
    return cur.rowcount


# ------------------------------------------------------------------- chunks
def insert_chunks(conn: psycopg.Connection, chunks: list[Chunk]) -> int:
    if not chunks:
        return 0
    rows = [
        (
            c.metadata["page_id"],
            c.metadata["website_id"],
            c.chunk_index,
            c.content,
            len(c.content),
            np.asarray(c.embedding, dtype=np.float32),  # pgvector adapter understands numpy arrays
            Jsonb(c.metadata),
        )
        for c in chunks
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (page_id, website_id, chunk_index, content, char_count, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (page_id, chunk_index) DO UPDATE
                SET content = EXCLUDED.content,
                    char_count = EXCLUDED.char_count,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
            """,
            rows,
        )
    return len(rows)


def chunk_embedding_dim(conn: psycopg.Connection) -> int | None:
    row = conn.execute(
        """
        SELECT format_type(a.atttypid, a.atttypmod) AS t
        FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
        WHERE c.relname = 'chunks' AND a.attname = 'embedding'
        """
    ).fetchone()
    if not row:
        return None
    t = row["t"]
    try:
        return int(t[t.index("(") + 1 : t.index(")")])
    except ValueError:
        return None


def similarity_search(
    conn: psycopg.Connection,
    query_embedding: Sequence[float],
    website_ids: Sequence[int],
    k: int = 5,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Return the `k` most similar chunks restricted to the given websites.

    `<=>` is pgvector's cosine *distance* (0 = identical, 2 = opposite);
    score = 1 - distance is the cosine similarity people usually reason about.
    """
    if not website_ids:
        return []
    # pgvector 0.8: keep scanning the HNSW index until enough rows pass the WHERE filter.
    conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
    q = np.asarray(query_embedding, dtype=np.float32)
    dist = _distance_expr(len(q))
    rows = conn.execute(
        f"""
        SELECT c.id, c.content, c.metadata, c.chunk_index, p.url, p.title, c.website_id,
               1 - {dist} AS score
        FROM chunks c
        JOIN pages p ON p.id = c.page_id
        WHERE c.website_id = ANY(%(ids)s)
        ORDER BY {dist}
        LIMIT %(k)s
        """,
        {"q": q, "ids": list(website_ids), "k": k},
    ).fetchall()
    return [r for r in rows if r["score"] >= min_score]


# ----------------------------------------------------------------- chatbots
def create_chatbot(
    conn: psycopg.Connection,
    *,
    name: str,
    description: str | None = None,
    system_prompt: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    top_k: int = 5,
) -> dict:
    return conn.execute(
        """
        INSERT INTO chatbots (name, description, system_prompt, llm_provider, llm_model, top_k)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (name) DO UPDATE
            SET description = EXCLUDED.description, system_prompt = EXCLUDED.system_prompt,
                llm_provider = EXCLUDED.llm_provider, llm_model = EXCLUDED.llm_model, top_k = EXCLUDED.top_k
        RETURNING *
        """,
        (name, description, system_prompt, llm_provider, llm_model, top_k),
    ).fetchone()


def get_chatbot(conn: psycopg.Connection, chatbot_id: int) -> dict | None:
    return conn.execute("SELECT * FROM chatbots WHERE id = %s", (chatbot_id,)).fetchone()


def list_chatbots(conn: psycopg.Connection) -> list[dict]:
    return conn.execute(
        """
        SELECT b.*, COALESCE(array_agg(cw.website_id) FILTER (WHERE cw.website_id IS NOT NULL), '{}') AS website_ids
        FROM chatbots b LEFT JOIN chatbot_websites cw ON cw.chatbot_id = b.id
        GROUP BY b.id ORDER BY b.id
        """
    ).fetchall()


def set_chatbot_websites(conn: psycopg.Connection, chatbot_id: int, website_ids: Sequence[int]) -> None:
    """Replace the set of websites a chatbot can see.

    All attached websites must share one embedding model, otherwise their
    vectors live in different spaces and cannot be compared.
    """
    if website_ids:
        models = conn.execute(
            "SELECT DISTINCT embedding_provider || ':' || embedding_model AS m FROM websites WHERE id = ANY(%s)",
            (list(website_ids),),
        ).fetchall()
        if len(models) > 1:
            raise ValueError(f"Websites use different embedding models: {[m['m'] for m in models]}")
    conn.execute("DELETE FROM chatbot_websites WHERE chatbot_id = %s", (chatbot_id,))
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chatbot_websites (chatbot_id, website_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            [(chatbot_id, w) for w in website_ids],
        )


def get_chatbot_website_ids(conn: psycopg.Connection, chatbot_id: int) -> list[int]:
    rows = conn.execute(
        "SELECT website_id FROM chatbot_websites WHERE chatbot_id = %s ORDER BY website_id", (chatbot_id,)
    ).fetchall()
    return [r["website_id"] for r in rows]


def delete_chatbot(conn: psycopg.Connection, chatbot_id: int) -> None:
    conn.execute("DELETE FROM chatbots WHERE id = %s", (chatbot_id,))


# ------------------------------------------------------------ conversations
def create_conversation(conn: psycopg.Connection, conversation_id: str, chatbot_id: int, title: str | None = None) -> None:
    conn.execute(
        "INSERT INTO conversations (id, chatbot_id, title) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
        (conversation_id, chatbot_id, title),
    )


def add_message(
    conn: psycopg.Connection, conversation_id: str, role: str, content: str, sources: list[dict] | None = None
) -> None:
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, sources) VALUES (%s, %s, %s, %s)",
        (conversation_id, role, content, Jsonb(sources) if sources is not None else None),
    )


def get_messages(conn: psycopg.Connection, conversation_id: str, limit: int | None = None) -> list[dict]:
    sql = "SELECT role, content, sources, created_at FROM messages WHERE conversation_id = %s ORDER BY id"
    params: list[Any] = [conversation_id]
    if limit:
        sql = f"SELECT * FROM ({sql} DESC LIMIT %s) t ORDER BY created_at"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


def list_conversations(conn: psycopg.Connection, chatbot_id: int) -> list[dict]:
    return conn.execute(
        """
        SELECT c.id, c.title, c.created_at, count(m.id) AS message_count
        FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
        WHERE c.chatbot_id = %s GROUP BY c.id ORDER BY c.created_at DESC
        """,
        (chatbot_id,),
    ).fetchall()
