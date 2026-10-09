"""PostgreSQL connection handling.

We use psycopg 3 with a small connection pool.  Opening a TCP connection to
Postgres costs tens of milliseconds, so a pool keeps a few connections ready.
`register_vector` teaches psycopg how to send/receive the pgvector `vector`
type, so we can pass plain Python lists / numpy arrays as embeddings.

Hosted Postgres (Neon in production) closes idle connections on its side:
Neon suspends an idle compute after a few minutes and its pooler drops idle
SSL sessions.  The pool cannot see that happen, so without precautions the
next request is handed a dead socket and fails with
``SSL connection has been closed unexpectedly``.  Three layers deal with it:

1. **Validation on checkout** – the pool runs a cheap round-trip (`check`)
   before handing out a connection and transparently replaces dead ones.
   Idle connections are also recycled (`max_idle`, `max_lifetime`) and TCP
   keepalives make the OS notice a vanished peer.
2. **Fresh connection per attempt** – `get_conn()` retries *obtaining* a
   healthy connection with exponential backoff (nothing has run yet, so this
   is always safe).  Once the body runs, a failure rolls back the transaction
   and the broken connection is discarded by the pool, never reused.
3. **Retrying the work itself** – a connection can still die between the
   check and the query.  `run_read()` re-runs a read-only transaction on a
   fresh connection; writes go through `get_conn()` / `run_write()` and are
   retried only when the caller declares them idempotent.

Usage::

    from chat_with_website.db.connection import get_conn, run_read

    with get_conn() as conn:                       # writes (not retried)
        conn.execute("INSERT ...")
    rows = run_read(lambda c: c.execute("SELECT 1").fetchall())   # retried
"""

from __future__ import annotations

import logging
import random
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator, TypeVar

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolClosed, PoolTimeout

from chat_with_website.config import settings

log = logging.getLogger(__name__)

T = TypeVar("T")

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()

# Errors that are *not* transient even though they subclass OperationalError.
_NON_TRANSIENT = (psycopg.errors.QueryCanceled,)


def _configure(conn: psycopg.Connection) -> None:
    """Runs once for every new connection in the pool."""
    register_vector(conn)


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:  # Streamlit serves sessions from several threads
            if _pool is None:
                log.info("Opening connection pool [%s] to %s", settings.database_label(), settings.safe_database_url())
                _pool = ConnectionPool(
                    conninfo=settings.database_url,
                    min_size=settings.db_pool_min,
                    max_size=settings.db_pool_max,
                    kwargs={
                        "row_factory": dict_row,
                        "connect_timeout": settings.db_connect_timeout,
                        # libpq TCP keepalives: detect a peer that vanished without a FIN.
                        "keepalives": 1,
                        "keepalives_idle": 30,
                        "keepalives_interval": 10,
                        "keepalives_count": 3,
                    },
                    configure=_configure,
                    check=ConnectionPool.check_connection,  # validate before every checkout
                    timeout=settings.db_pool_timeout,
                    max_idle=settings.db_max_idle,
                    max_lifetime=settings.db_max_lifetime,
                    open=True,
                )
    return _pool


# ------------------------------------------------------------------ retries
def is_transient_error(exc: BaseException) -> bool:
    """True for errors caused by the connection, not by the SQL.

    These are safe to retry on a *fresh* connection: dropped SSL/TCP sessions,
    "server closed the connection unexpectedly", Neon compute waking up, the
    pool failing to produce a healthy connection in time.  SQL errors
    (`ProgrammingError`, `DataError`, `IntegrityError`...) and statement
    timeouts are not transient and are never retried.
    """
    if isinstance(exc, _NON_TRANSIENT):
        return False
    if isinstance(exc, PoolClosed):
        return False
    return isinstance(exc, (psycopg.OperationalError, psycopg.InterfaceError, PoolTimeout))


def _describe(exc: BaseException) -> str:
    """Error summary for logs. psycopg never puts the password in messages; keep it that way."""
    return f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with full jitter: attempt 1 -> ~base, 2 -> ~2*base, ..."""
    cap = min(settings.db_retry_max_delay, settings.db_retry_base_delay * 2 ** (attempt - 1))
    return random.uniform(0, cap)


def _checkout(pool: ConnectionPool, attempts: int) -> psycopg.Connection:
    """Obtain a *validated* connection, retrying the checkout with backoff.

    Nothing has executed yet, so retrying here is always safe.  The pool's own
    `check` already discards dead connections; this loop covers the pool being
    unable to produce a healthy one in time (e.g. Neon waking from suspend).
    """
    for attempt in range(1, attempts + 1):
        try:
            return pool.getconn()
        except Exception as exc:
            if not is_transient_error(exc) or attempt >= attempts:
                log.error("db.checkout failed attempt=%d/%d error=%s", attempt, attempts, _describe(exc))
                raise
            delay = _backoff_delay(attempt)
            log.warning(
                "db.checkout retry attempt=%d/%d wait=%.2fs error=%s", attempt, attempts, delay, _describe(exc)
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    """Borrow a validated connection; commit on success, roll back on exception.

    The *body* is never retried (it may contain writes).  If it fails because
    the connection died, psycopg rolls back as far as it can and the pool
    discards the broken connection instead of reusing it, so the next
    `get_conn()` starts clean.
    """
    pool = get_pool()
    conn = _checkout(pool, settings.db_retry_attempts)
    try:
        with conn:  # psycopg: commit on success, rollback (errors ignored) on exception
            yield conn
    finally:
        pool.putconn(conn)  # always returned; broken connections are discarded here


def _run(fn: Callable[[psycopg.Connection], T], *, attempts: int, read_only: bool, label: str) -> T:
    for attempt in range(1, attempts + 1):
        try:
            with get_conn() as conn:
                if read_only:
                    # Enforced by Postgres: a retried function cannot accidentally write.
                    conn.execute("SET TRANSACTION READ ONLY")
                return fn(conn)
        except Exception as exc:
            if not is_transient_error(exc) or attempt >= attempts:
                if is_transient_error(exc):
                    log.error("db.%s failed attempt=%d/%d error=%s", label, attempt, attempts, _describe(exc))
                raise
            delay = _backoff_delay(attempt)
            log.warning(
                "db.%s retry attempt=%d/%d wait=%.2fs error=%s", label, attempt, attempts, delay, _describe(exc)
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def run_read(fn: Callable[[psycopg.Connection], T], *, attempts: int | None = None) -> T:
    """Run `fn(conn)` in a READ ONLY transaction, retrying transient connection errors.

    Each attempt uses a fresh, validated connection; the failed one is
    discarded.  Because the transaction is read-only, replaying it is safe.
    """
    return _run(fn, attempts=attempts or settings.db_retry_attempts, read_only=True, label="read")


def run_write(fn: Callable[[psycopg.Connection], T], *, idempotent: bool = False, attempts: int | None = None) -> T:
    """Run `fn(conn)` in a read-write transaction.

    Not retried unless `idempotent=True` - only pass that when replaying the
    whole function is provably harmless (e.g. `INSERT ... ON CONFLICT DO
    NOTHING`, `UPDATE ... SET x = const`).  A plain `INSERT` is not idempotent:
    the server may have committed before the connection dropped.
    """
    attempts = (attempts or settings.db_retry_attempts) if idempotent else 1
    return _run(fn, attempts=attempts, read_only=False, label="write")


# -------------------------------------------------------------------- misc
def close_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


def pool_stats() -> dict:
    """psycopg_pool counters (requests, connections lost, bad returns...) for diagnostics."""
    return get_pool().get_stats()


def check_connection() -> dict:
    """Quick health check used by scripts and the UI."""

    def _check(conn: psycopg.Connection) -> dict:
        version = conn.execute("SHOW server_version").fetchone()["server_version"]
        ext = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        return {"server_version": version, "pgvector": ext["extversion"] if ext else None}

    return run_read(_check)
