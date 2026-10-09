"""Connection-pool resilience: dropped SSL connections, retries, rollback, no leaks.

The unit tests use a fake pool so they need no database.  The last test talks
to the real local database and kills its own backend with
`pg_terminate_backend`; it runs only when CHAT_WITH_WEBSITE_DB_TESTS=1.
"""

from __future__ import annotations

import logging
import os

import psycopg
import pytest
from psycopg_pool import PoolClosed, PoolTimeout

from chat_with_website.db import connection as dbc

SSL_DROPPED = "consuming input failed: SSL connection has been closed unexpectedly"


# ------------------------------------------------------------------- fakes
class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class FakeConn:
    """Mimics the bits of psycopg.Connection the pool helpers touch.

    `errors` is a queue of exceptions raised by successive `execute()` calls
    (None = succeed).  A connection that raised a connection-level error is
    `broken`: rollback fails on it, exactly like a dead socket.
    """

    def __init__(self, errors=(), rows=({"ok": 1},)):
        self.errors = list(errors)
        self.rows = rows
        self.executed: list[str] = []
        self.commits = 0
        self.rollbacks = 0
        self.broken = False
        self.closed = False
        self.autocommit = False

    def execute(self, sql, params=None):
        self.executed.append(sql)
        if self.errors:
            exc = self.errors.pop(0)
            if exc is not None:
                if isinstance(exc, (psycopg.OperationalError, psycopg.InterfaceError)):
                    self.broken = True
                raise exc
        return FakeCursor(self.rows)

    def commit(self):
        if self.broken:
            raise psycopg.OperationalError("the connection is lost")
        self.commits += 1

    def rollback(self):
        if self.broken:
            raise psycopg.OperationalError("the connection is lost")
        self.rollbacks += 1

    # same contract as psycopg.Connection.__exit__: rollback errors are swallowed
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.closed:
            return
        if exc_type:
            try:
                self.rollback()
            except Exception:
                pass
        else:
            self.commit()


class FakePool:
    def __init__(self, conns, getconn_errors=()):
        self.conns = list(conns)
        self.getconn_errors = list(getconn_errors)
        self.checked_out: list[FakeConn] = []
        self.returned: list[FakeConn] = []

    def getconn(self):
        if self.getconn_errors:
            raise self.getconn_errors.pop(0)
        conn = self.conns.pop(0)
        self.checked_out.append(conn)
        return conn

    def putconn(self, conn):
        self.returned.append(conn)
        if conn.broken:  # psycopg_pool: "discarding closed connection"
            conn.closed = True


@pytest.fixture
def sleeps(monkeypatch):
    """Record backoff sleeps instead of waiting."""
    calls: list[float] = []
    monkeypatch.setattr(dbc.time, "sleep", calls.append)
    monkeypatch.setattr(dbc.settings, "db_retry_attempts", 3)
    monkeypatch.setattr(dbc.settings, "db_retry_base_delay", 0.25)
    monkeypatch.setattr(dbc.settings, "db_retry_max_delay", 3.0)
    return calls


def use_pool(monkeypatch, pool: FakePool) -> FakePool:
    monkeypatch.setattr(dbc, "get_pool", lambda: pool)
    return pool


# ------------------------------------------------------------- classification
def test_transient_error_classification():
    assert dbc.is_transient_error(psycopg.OperationalError(SSL_DROPPED))
    assert dbc.is_transient_error(psycopg.OperationalError("server closed the connection unexpectedly"))
    assert dbc.is_transient_error(psycopg.errors.AdminShutdown("terminating connection due to administrator command"))
    assert dbc.is_transient_error(psycopg.InterfaceError("the connection is closed"))
    assert dbc.is_transient_error(PoolTimeout("couldn't get a connection after 30.00 sec"))
    # SQL / logic errors and statement timeouts must never be replayed
    assert not dbc.is_transient_error(psycopg.errors.UndefinedTable("relation x does not exist"))
    assert not dbc.is_transient_error(psycopg.errors.UniqueViolation("duplicate key"))
    assert not dbc.is_transient_error(psycopg.errors.QueryCanceled("canceling statement due to statement timeout"))
    assert not dbc.is_transient_error(PoolClosed("the pool is closed"))
    assert not dbc.is_transient_error(ValueError("not a db error"))


# ------------------------------------------------------------------- reads
def test_run_read_retries_on_dropped_ssl_with_fresh_connection(monkeypatch, sleeps, caplog):
    dead = FakeConn(errors=[None, psycopg.OperationalError(SSL_DROPPED)])  # SET ok, query dies
    alive = FakeConn(rows=[{"n": 42}])
    pool = use_pool(monkeypatch, FakePool([dead, alive]))
    calls = []

    def fn(conn):
        calls.append(conn)
        return conn.execute("SELECT 42").fetchone()["n"]

    with caplog.at_level(logging.WARNING, logger="chat_with_website.db.connection"):
        assert dbc.run_read(fn) == 42

    assert calls == [dead, alive]  # second attempt got a different connection
    assert pool.returned == [dead, alive]  # both went back to the pool...
    assert dead.closed and not alive.closed  # ...and the broken one was discarded
    assert alive.commits == 1 and dead.commits == 0
    assert len(sleeps) == 1
    assert "db.read retry attempt=1/3" in caplog.text and "SSL connection has been closed" in caplog.text


def test_run_read_sets_read_only_transaction(monkeypatch, sleeps):
    conn = FakeConn()
    use_pool(monkeypatch, FakePool([conn]))
    dbc.run_read(lambda c: c.execute("SELECT 1").fetchall())
    assert conn.executed[0] == "SET TRANSACTION READ ONLY"


def test_run_read_gives_up_after_configured_attempts(monkeypatch, sleeps, caplog):
    conns = [FakeConn(errors=[psycopg.OperationalError(SSL_DROPPED)]) for _ in range(3)]
    pool = use_pool(monkeypatch, FakePool(conns))
    with caplog.at_level(logging.ERROR, logger="chat_with_website.db.connection"):
        with pytest.raises(psycopg.OperationalError, match="SSL connection has been closed"):
            dbc.run_read(lambda c: c.execute("SELECT 1"))
    assert len(pool.returned) == 3 and all(c.closed for c in conns)  # nothing leaked, all discarded
    assert len(sleeps) == 2  # no sleep after the final failure
    assert "db.read failed attempt=3/3" in caplog.text


def test_run_read_does_not_retry_sql_errors(monkeypatch, sleeps):
    conn = FakeConn(errors=[None, psycopg.errors.UndefinedTable("relation nope does not exist")])
    pool = use_pool(monkeypatch, FakePool([conn, FakeConn()]))
    with pytest.raises(psycopg.errors.UndefinedTable):
        dbc.run_read(lambda c: c.execute("SELECT * FROM nope"))
    assert len(pool.checked_out) == 1 and sleeps == []
    assert conn.rollbacks == 1 and conn.commits == 0 and not conn.closed  # healthy conn rolled back, kept


# ------------------------------------------------------------------ writes
def test_get_conn_write_body_is_not_retried(monkeypatch, sleeps):
    dead = FakeConn(errors=[psycopg.OperationalError(SSL_DROPPED)])
    pool = use_pool(monkeypatch, FakePool([dead, FakeConn()]))
    bodies = 0
    with pytest.raises(psycopg.OperationalError):
        with dbc.get_conn() as conn:
            bodies += 1
            conn.execute("INSERT INTO messages VALUES (1)")
    assert bodies == 1 and sleeps == []
    assert pool.returned == [dead] and dead.closed  # rolled back as far as possible, then discarded


def test_run_write_not_retried_unless_idempotent(monkeypatch, sleeps):
    pool = use_pool(monkeypatch, FakePool([FakeConn(errors=[psycopg.OperationalError(SSL_DROPPED)]), FakeConn()]))
    with pytest.raises(psycopg.OperationalError):
        dbc.run_write(lambda c: c.execute("INSERT ..."))
    assert len(pool.checked_out) == 1 and sleeps == []

    pool = use_pool(monkeypatch, FakePool([FakeConn(errors=[psycopg.OperationalError(SSL_DROPPED)]), FakeConn()]))
    dbc.run_write(lambda c: c.execute("INSERT ... ON CONFLICT DO NOTHING"), idempotent=True)
    assert len(pool.checked_out) == 2 and len(sleeps) == 1


def test_write_transaction_is_not_read_only(monkeypatch, sleeps):
    conn = FakeConn()
    use_pool(monkeypatch, FakePool([conn]))
    dbc.run_write(lambda c: c.execute("UPDATE x SET y = 1"))
    assert conn.executed == ["UPDATE x SET y = 1"] and conn.commits == 1


# ---------------------------------------------------------------- checkout
def test_checkout_retries_when_pool_cannot_supply_a_healthy_connection(monkeypatch, sleeps):
    good = FakeConn()
    pool = use_pool(monkeypatch, FakePool([good], getconn_errors=[PoolTimeout("couldn't get a connection")]))
    with dbc.get_conn() as conn:
        assert conn is good
    assert len(sleeps) == 1 and pool.returned == [good] and good.commits == 1


def test_checkout_does_not_retry_closed_pool(monkeypatch, sleeps):
    use_pool(monkeypatch, FakePool([FakeConn()], getconn_errors=[PoolClosed("the pool is closed")]))
    with pytest.raises(PoolClosed):
        with dbc.get_conn():
            pass
    assert sleeps == []


def test_checkout_exhausts_attempts(monkeypatch, sleeps):
    errors = [PoolTimeout("x")] * 3
    use_pool(monkeypatch, FakePool([], getconn_errors=errors))
    with pytest.raises(PoolTimeout):
        with dbc.get_conn():
            pass
    assert len(sleeps) == 2


# --------------------------------------------------------- leaks & backoff
def test_connection_returned_even_when_body_raises_unexpectedly(monkeypatch, sleeps):
    conn = FakeConn()
    pool = use_pool(monkeypatch, FakePool([conn]))

    class StreamlitRerun(BaseException):  # Streamlit interrupts reruns with a BaseException
        pass

    with pytest.raises(StreamlitRerun):
        with dbc.get_conn():
            raise StreamlitRerun()
    assert pool.returned == [conn] and conn.rollbacks == 1


def test_backoff_is_exponential_and_capped(monkeypatch):
    monkeypatch.setattr(dbc.settings, "db_retry_base_delay", 0.25)
    monkeypatch.setattr(dbc.settings, "db_retry_max_delay", 1.0)
    monkeypatch.setattr(dbc.random, "uniform", lambda lo, hi: hi)  # take the upper bound (no jitter)
    assert [dbc._backoff_delay(a) for a in (1, 2, 3, 4)] == [0.25, 0.5, 1.0, 1.0]


def test_error_description_is_compact():
    exc = psycopg.OperationalError("consuming input failed:\n   SSL connection\thas been closed unexpectedly")
    assert dbc._describe(exc) == "OperationalError: consuming input failed: SSL connection has been closed unexpectedly"


# -------------------------------------------------------- real database
@pytest.mark.skipif(not os.environ.get("CHAT_WITH_WEBSITE_DB_TESTS"), reason="set CHAT_WITH_WEBSITE_DB_TESTS=1")
def test_real_backend_killed_mid_transaction_is_retried():
    """Kill our own server backend between two statements, like Neon dropping the socket."""
    dbc.close_pool()
    pool = dbc.get_pool()
    killed = []

    def fn(conn):
        pid = conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"]
        if not killed:
            killed.append(pid)
            with psycopg.connect(dbc.settings.database_url, autocommit=True) as admin:
                admin.execute("SELECT pg_terminate_backend(%s)", (pid,))
        return conn.execute("SELECT 1 AS one").fetchone()["one"]

    try:
        assert dbc.run_read(fn) == 1
        stats = pool.get_stats()
        assert stats.get("returns_bad", 0) + stats.get("connections_lost", 0) >= 1
        # the pool's `check` also catches a connection killed while idle
        pid = dbc.run_read(lambda c: c.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"])
        with psycopg.connect(dbc.settings.database_url, autocommit=True) as admin:
            admin.execute("SELECT pg_terminate_backend(%s)", (pid,))
        assert dbc.run_read(lambda c: c.execute("SELECT 2 AS two").fetchone()["two"]) == 2
    finally:
        dbc.close_pool()
