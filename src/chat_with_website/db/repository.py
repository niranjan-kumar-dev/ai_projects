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
    kinds: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the `k` most similar chunks restricted to the given websites.

    `<=>` is pgvector's cosine *distance* (0 = identical, 2 = opposite);
    score = 1 - distance is the cosine similarity people usually reason about.
    `kinds` restricts the chunk type ("text" / "image"); chunks indexed before
    image support have no type and count as text.
    """
    if not website_ids:
        return []
    # pgvector 0.8: keep scanning the HNSW index until enough rows pass the WHERE filter.
    conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
    q = np.asarray(query_embedding, dtype=np.float32)
    dist = _distance_expr(len(q))
    kind_filter = "AND COALESCE(c.metadata->>'type', 'text') = ANY(%(kinds)s)" if kinds else ""
    rows = conn.execute(
        f"""
        SELECT c.id, c.content, c.metadata, c.chunk_index, p.url, p.title, c.website_id,
               1 - {dist} AS score
        FROM chunks c
        JOIN pages p ON p.id = c.page_id
        WHERE c.website_id = ANY(%(ids)s) {kind_filter}
        ORDER BY {dist}
        LIMIT %(k)s
        """,
        {"q": q, "ids": list(website_ids), "k": k, "kinds": list(kinds or [])},
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


def update_chatbot(
    conn: psycopg.Connection,
    chatbot_id: int,
    *,
    name: str,
    description: str | None = None,
    system_prompt: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    top_k: int = 5,
) -> dict | None:
    """Edit an existing chatbot *by id* (so renaming keeps the same row and its websites).

    Raises `psycopg.errors.UniqueViolation` if `name` already belongs to another chatbot.
    """
    return conn.execute(
        """
        UPDATE chatbots
           SET name = %s, description = %s, system_prompt = %s,
               llm_provider = %s, llm_model = %s, top_k = %s
         WHERE id = %s
        RETURNING *
        """,
        (name, description, system_prompt, llm_provider, llm_model, top_k, chatbot_id),
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
    conn: psycopg.Connection,
    conversation_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
    edited_from_message_id: int | None = None,
) -> int:
    """Insert one message and return its id (the UI keeps it to attach feedback / regenerations).

    `edited_from_message_id` marks a question that is an edited version of a message in the
    parent conversation (see `fork_conversation`).
    """
    row = conn.execute(
        """
        INSERT INTO messages (conversation_id, role, content, sources, edited_from_message_id)
        VALUES (%s, %s, %s, %s, %s) RETURNING id
        """,
        (conversation_id, role, content, Jsonb(sources) if sources is not None else None, edited_from_message_id),
    ).fetchone()
    return row["id"]


def fork_conversation(
    conn: psycopg.Connection,
    *,
    new_id: str,
    chatbot_id: int,
    source_id: str,
    forked_from_message_id: int | None,
    turns: Sequence[tuple[str, str, list[dict] | None]],
) -> list[int]:
    """Create a branch of `source_id` containing copies of `turns` ([(role, content, sources), ...]).

    Used when the user edits a question: the turns *before* it are copied into the new
    conversation and the edited question is asked there, so the original conversation is never
    modified. The turns come from the UI (what the user actually sees), so the copy does not
    depend on parent rows that may be missing after an earlier failed save. Feedback is not
    copied: it belongs to the original answers. Returns the ids of the copied rows, in order.

    Raises `ValueError` when the source conversation belongs to another chatbot.
    """
    source = conn.execute("SELECT chatbot_id, title FROM conversations WHERE id = %s", (source_id,)).fetchone()
    if source is not None and source["chatbot_id"] != chatbot_id:
        raise ValueError(f"Conversation {source_id} does not belong to chatbot {chatbot_id}")
    if source is None:  # the parent was never saved (e.g. its first save failed): a plain new conversation
        title = next((c for r, c, _ in turns if r == "user"), None)
        parent_id, forked_from = None, None
    else:
        title, parent_id, forked_from = source["title"], source_id, forked_from_message_id
    conn.execute(
        """
        INSERT INTO conversations (id, chatbot_id, title, parent_conversation_id, forked_from_message_id)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (new_id, chatbot_id, title[:80] if title else None, parent_id, forked_from),
    )
    # One INSERT per turn keeps the returned ids in turn order (INSERT ... SELECT ... RETURNING
    # does not guarantee an order); the prefix is a handful of rows.
    return [add_message(conn, new_id, role, content, sources) for role, content, sources in turns]


def get_messages(conn: psycopg.Connection, conversation_id: str, limit: int | None = None) -> list[dict]:
    """Messages in order, each with its feedback `rating` (NULL when none was given)."""
    sql = """
        SELECT m.id, m.role, m.content, m.sources, m.created_at, m.regeneration_count,
               m.edited_from_message_id, f.rating
        FROM messages m LEFT JOIN message_feedback f ON f.message_id = m.id
        WHERE m.conversation_id = %s ORDER BY m.id
    """
    params: list[Any] = [conversation_id]
    if limit:
        sql = f"SELECT * FROM ({sql} DESC LIMIT %s) t ORDER BY id"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


def update_message(conn: psycopg.Connection, message_id: int, content: str, sources: list[dict] | None) -> int:
    """Replace an assistant message's text (a regeneration); returns the new regeneration_count.

    Feedback about the old text is dropped in the same transaction: it no longer applies.
    Raises `ValueError` if the message does not exist (e.g. the conversation was deleted meanwhile).
    """
    row = conn.execute(
        """
        UPDATE messages
           SET content = %s, sources = %s, regenerated_at = now(), regeneration_count = regeneration_count + 1
         WHERE id = %s AND role = 'assistant'
        RETURNING regeneration_count
        """,
        (content, Jsonb(sources) if sources is not None else None, message_id),
    ).fetchone()
    if row is None:
        raise ValueError(f"Assistant message {message_id} does not exist")
    conn.execute("DELETE FROM message_feedback WHERE message_id = %s", (message_id,))
    return row["regeneration_count"]


# ----------------------------------------------------------------- feedback
RATINGS = (-1, 1)  # thumbs down / thumbs up


def upsert_feedback(conn: psycopg.Connection, message_id: int, rating: int) -> bool:
    """Create or update the single feedback row of an assistant message.

    conversation_id / chatbot_id are taken from the message itself, so feedback can never be
    attached to the wrong chatbot. Returns False when the message no longer exists.
    """
    if rating not in RATINGS:
        raise ValueError(f"rating must be one of {RATINGS}, got {rating!r}")
    cur = conn.execute(
        """
        INSERT INTO message_feedback (message_id, conversation_id, chatbot_id, rating)
        SELECT m.id, m.conversation_id, c.chatbot_id, %s
          FROM messages m JOIN conversations c ON c.id = m.conversation_id
         WHERE m.id = %s AND m.role = 'assistant'
        ON CONFLICT (message_id) DO UPDATE SET rating = EXCLUDED.rating, updated_at = now()
        """,
        (rating, message_id),
    )
    return cur.rowcount == 1


def list_conversations(conn: psycopg.Connection, chatbot_id: int) -> list[dict]:
    """Conversations of one chatbot, newest activity first (`updated_at` = last message)."""
    return conn.execute(
        """
        SELECT c.id, c.title, c.created_at, c.parent_conversation_id, count(m.id) AS message_count,
               COALESCE(max(m.created_at), c.created_at) AS updated_at
        FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
        WHERE c.chatbot_id = %s GROUP BY c.id ORDER BY updated_at DESC
        """,
        (chatbot_id,),
    ).fetchall()


MAX_CONVERSATION_TITLE = 120


def rename_conversation(conn: psycopg.Connection, conversation_id: str, title: str) -> None:
    title = " ".join(title.split())[:MAX_CONVERSATION_TITLE]
    if not title:
        raise ValueError("Title cannot be empty")
    conn.execute("UPDATE conversations SET title = %s WHERE id = %s", (title, conversation_id))


def delete_conversation(conn: psycopg.Connection, conversation_id: str) -> None:
    """Messages are removed by the ON DELETE CASCADE foreign key."""
    conn.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))
