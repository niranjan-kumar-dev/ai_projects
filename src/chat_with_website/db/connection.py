"""PostgreSQL connection handling.

We use psycopg 3 with a small connection pool.  Opening a TCP connection to
Postgres costs tens of milliseconds, so a pool keeps a few connections ready.
`register_vector` teaches psycopg how to send/receive the pgvector `vector`
type, so we can pass plain Python lists / numpy arrays as embeddings.

Usage::

    from chat_with_website.db.connection import get_conn

    with get_conn() as conn:
        rows = conn.execute("SELECT 1").fetchall()
    # the transaction is committed (or rolled back on error) and the
    # connection goes back to the pool automatically.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from chat_with_website.config import settings

log = logging.getLogger(__name__)

_pool: ConnectionPool | None = None


def _configure(conn: psycopg.Connection) -> None:
    """Runs once for every new connection in the pool."""
    register_vector(conn)


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        log.info("Opening connection pool to %s", settings.safe_database_url())
        _pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=settings.db_pool_min,
            max_size=settings.db_pool_max,
            kwargs={"row_factory": dict_row},
            configure=_configure,
            open=True,
        )
    return _pool


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    """Borrow a connection; commit on success, roll back on exception."""
    pool = get_pool()
    with pool.connection() as conn:
        yield conn
        # psycopg_pool commits/rolls back for us when the block exits.


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def check_connection() -> dict:
    """Quick health check used by scripts and the UI."""
    with get_conn() as conn:
        version = conn.execute("SHOW server_version").fetchone()["server_version"]
        ext = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
    return {"server_version": version, "pgvector": ext["extversion"] if ext else None}
