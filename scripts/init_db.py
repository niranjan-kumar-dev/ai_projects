"""Create the database (if needed), enable pgvector and apply schema.sql.

Usage (from the project root):

    python scripts/init_db.py                 # create DB + tables, idempotent
    python scripts/init_db.py --reset-chunks  # drop & recreate `chunks` (after changing the embedding model)
    python scripts/init_db.py --drop-all      # DANGER: drop every table and start over

The embedding vector size comes from .env (EMBEDDING_PROVIDER / EMBEDDING_DIM).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8")

import psycopg  # noqa: E402
from psycopg import sql  # noqa: E402

from chat_with_website.config import settings  # noqa: E402
from chat_with_website.db.repository import HALFVEC_THRESHOLD  # noqa: E402
from chat_with_website.logging_config import setup_logging  # noqa: E402

log = logging.getLogger("init_db")
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "src" / "chat_with_website" / "db" / "schema.sql"


def _maintenance_url(database_url: str) -> tuple[str, str]:
    """Return (url pointing at the `postgres` maintenance DB, target db name)."""
    parts = urlsplit(database_url)
    db_name = parts.path.lstrip("/") or "postgres"
    maint = urlunsplit((parts.scheme, parts.netloc, "/postgres", parts.query, parts.fragment))
    return maint, db_name


def ensure_database() -> None:
    maint_url, db_name = _maintenance_url(settings.database_url)
    with psycopg.connect(maint_url, autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,)).fetchone()
        if exists:
            log.info("Database '%s' already exists", db_name)
        else:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
            log.info("Created database '%s'", db_name)


def current_chunk_dim(conn: psycopg.Connection) -> int | None:
    row = conn.execute(
        """
        SELECT format_type(a.atttypid, a.atttypmod) AS t
        FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
        WHERE c.relname = 'chunks' AND a.attname = 'embedding'
        """
    ).fetchone()
    if not row:
        return None
    # format_type gives e.g. 'vector(384)'
    text = row[0]
    try:
        return int(text[text.index("(") + 1 : text.index(")")])
    except ValueError:
        return None


def apply_schema(reset_chunks: bool, drop_all: bool) -> None:
    dim = settings.resolved_embedding_dim
    if dim > HALFVEC_THRESHOLD:
        hnsw = (
            "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks "
            f"USING hnsw ((embedding::halfvec({dim})) halfvec_cosine_ops);"
        )
        log.info("Embedding dim %s > %s: using a halfvec HNSW index", dim, HALFVEC_THRESHOLD)
    else:
        hnsw = "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);"
    schema_sql = (
        SCHEMA_PATH.read_text(encoding="utf-8")
        .replace("{EMBEDDING_DIM}", str(dim))
        .replace("{HNSW_INDEX}", hnsw)
    )

    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        if drop_all:
            log.warning("Dropping ALL tables")
            conn.execute(
                "DROP TABLE IF EXISTS messages, conversations, chatbot_websites, chatbots, "
                "chunks, pages, websites CASCADE"
            )
        elif reset_chunks:
            log.warning("Dropping table `chunks` (embeddings will need to be re-ingested)")
            conn.execute("DROP TABLE IF EXISTS chunks CASCADE")
            conn.execute("DELETE FROM pages")  # pages without chunks are useless; re-ingest recreates them
            conn.execute("UPDATE websites SET chunk_count = 0, page_count = 0, status = 'pending'")

        existing = current_chunk_dim(conn)
        if existing is not None and existing != dim:
            log.error(
                "chunks.embedding is vector(%s) but .env is configured for %s dims (%s/%s).\n"
                "Either change EMBEDDING_PROVIDER back, or run with --reset-chunks and re-ingest.",
                existing, dim, settings.embedding_provider, settings.resolved_embedding_model,
            )
            sys.exit(1)

        conn.execute(schema_sql)
        log.info("Schema applied (embedding dimension = %s)", dim)

        ext = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY 1"
        ).fetchall()
        log.info("pgvector version: %s", ext[0] if ext else "NOT INSTALLED")
        log.info("Tables: %s", ", ".join(t[0] for t in tables))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset-chunks", action="store_true", help="drop and recreate the chunks table")
    parser.add_argument("--drop-all", action="store_true", help="drop every table (destructive)")
    args = parser.parse_args()

    setup_logging()
    log.info("Using %s", settings.safe_database_url())
    try:
        ensure_database()
        apply_schema(reset_chunks=args.reset_chunks, drop_all=args.drop_all)
    except psycopg.OperationalError as exc:
        log.error("Could not connect to PostgreSQL: %s", exc)
        log.error("Check DATABASE_URL in .env (user, password, host, port) and that the service is running.")
        sys.exit(1)
    except psycopg.Error as exc:
        log.error("PostgreSQL error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
