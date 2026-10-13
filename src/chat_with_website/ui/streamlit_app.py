"""Streamlit UI: index websites, create chatbots, chat.

Run from the project root:

    python -m streamlit run src/chat_with_website/ui/streamlit_app.py

Streamlit re-runs this whole file on every click.  Anything expensive (the
embedding model, the connection pool) is cached with `st.cache_resource`;
anything that must survive a rerun (chat history, selected chatbot) lives in
`st.session_state`.
"""

from __future__ import annotations

import hmac
import io
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow running without `pip install -e .`

import psycopg
import streamlit as st
import streamlit.components.v2  # noqa: F401  (makes st.components.v2 available)
from langchain_core.messages import AIMessage, HumanMessage

from chat_with_website.config import EMBEDDING_DEFAULTS, LLM_DEFAULTS, settings
from chat_with_website.db import repository as repo
from chat_with_website.db.connection import (
    check_connection,
    ensure_schema_upgrades,
    get_conn,
    is_transient_error,
    run_read,
)
from chat_with_website.ingestion.embedder import get_embeddings
from chat_with_website.ingestion.pipeline import ingest_website
from chat_with_website.ingestion.url_utils import validate_start_url
from chat_with_website.logging_config import setup_logging
from chat_with_website.ui.branding import (
    DARK_BACKGROUND,
    MAX_DESCRIPTION,
    MAX_DISPLAY_NAME,
    MAX_HEADING,
    MAX_PLACEHOLDER,
    MAX_STARTERS,
    accent_css,
    branding_of,
    contrast_ratio,
    process_logo,
    starter_text,
    validate_branding,
)
from chat_with_website.rag.citations import (
    display_url,
    escape_markdown,
    group_sources,
    highlight_terms,
    linkify_citations,
    question_terms,
)
from chat_with_website.rag import debug as rag_debug
from chat_with_website.rag import evaluation as rag_eval
from chat_with_website.rag.chain import (
    build_chain_for_chatbot,
    format_context,
    history_from_rows,
    new_conversation_id,
    persist_turn,
)
from chat_with_website.rag.llm import PROVIDER_LABELS, describe_llm
from chat_with_website.ui.conversations import filter_conversations, group_conversations
from chat_with_website.ui import eval_io
from chat_with_website.ui import kb_status as kb

setup_logging()
log = logging.getLogger(__name__)
# Centered column for the chat (ChatGPT-like reading width); admin pages switch to "wide"
# themselves. 260 = compact sidebar width in px (still collapsible, "auto" behaviour).
st.set_page_config(
    page_title="Chat with website", page_icon=":material/chat:", layout="centered", initial_sidebar_state=260
)


# ----------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading embedding model…")
def _embeddings():
    return get_embeddings()


@st.cache_resource
def _db_health() -> dict:
    return check_connection()


@st.cache_resource(show_spinner="Checking database schema…")
def _schema_ready() -> bool:
    """Apply idempotent schema upgrades once per process (see `ensure_schema_upgrades`).

    A git push to Streamlit Cloud redeploys the code but nobody runs `init_db.py` against Neon;
    without this, code that reads a newly added column fails with `UndefinedColumn`.
    """
    ensure_schema_upgrades()
    return True


def _require_schema() -> bool:
    """Apply the idempotent upgrades (once per process) and explain when that fails.

    Every page calls this: a cold process whose *first* request lands on the Websites or
    Chatbots page must not read a column or table the database does not have yet."""
    try:
        _schema_ready()
    except Exception as exc:
        log.exception("Applying schema upgrades failed")
        st.error(
            "The database schema could not be updated. Run `python scripts/init_db.py` against this "
            f"database and reload. ({type(exc).__name__})", icon=":material/error:",
        )
        return False
    return True


def _rag_code_version() -> float:
    """Newest mtime of the RAG source files.

    Streamlit Cloud applies a git push as a *hot reload*: the script re-runs but the
    process (and everything in `st.cache_resource`) survives.  Keying the chain cache on
    this value means a push that changes the RAG code gets a freshly built chain instead
    of the stale object - no manual "Reboot app" needed.  Computed per rerun, cheap.
    """
    rag_dir = Path(__file__).resolve().parents[1] / "rag"
    return max((p.stat().st_mtime for p in rag_dir.glob("*.py")), default=0.0)


@st.cache_resource(show_spinner=False)
def _chain(chatbot_id: int, version: int, code_version: float):
    """Cached per chatbot; `version` is bumped when the chatbot is edited,
    `code_version` changes when the RAG code is redeployed.  (Parameters starting with an
    underscore are *not* part of a cache key, so these must not be underscored.)"""
    return build_chain_for_chatbot(chatbot_id)


def _bump_chain_version() -> None:
    st.session_state["chain_version"] = st.session_state.get("chain_version", 0) + 1
    _chain.clear()


# ------------------------------------------------------------- admin access
# ADMIN_PASSWORD (.env / Streamlit secrets) protects the Websites and Chatbots pages. Unset =
# open, which keeps local development unchanged. The Chat page is never gated.
ADMIN_KEY = "is_admin"


def _is_admin() -> bool:
    return not settings.admin_password or bool(st.session_state.get(ADMIN_KEY))


def _logout() -> None:
    st.session_state.pop(ADMIN_KEY, None)


def _require_admin() -> bool:
    """Draw the password prompt when needed; True when the caller may show the admin page."""
    if _is_admin():
        return True
    st.info("This is an admin page. Enter the admin password to continue.", icon=":material/lock:")
    with st.form("admin_login"):
        pw = st.text_input("Admin password", type="password", key="admin_password_input")
        ok = st.form_submit_button("Unlock", type="primary", icon=":material/lock_open:")
    if ok:
        if hmac.compare_digest(pw or "", settings.admin_password or ""):
            st.session_state[ADMIN_KEY] = True
            st.rerun()
        st.error("Wrong password.", icon=":material/error:")
    return False


# Reads go through `run_read` (retried on a fresh connection if Neon dropped the
# socket); writes keep `get_conn()` and are not retried.  Every borrow is a
# `with` block, so nothing leaks when Streamlit interrupts a rerun.
def _websites() -> list[dict]:
    return run_read(repo.list_websites)


def _chatbots() -> list[dict]:
    return run_read(repo.list_chatbots)


# ------------------------------------------------------------- page: websites
def page_websites() -> None:
    st.set_page_config(layout="wide")  # tables need the width
    if not _require_admin():
        return
    if not _require_schema():  # the overview reads `ingestion_runs` / `last_indexed_at`
        return
    st.title("Websites", icon=":material/language:")
    st.caption("Index a website: crawl → clean → chunk → embed → store in PostgreSQL/pgvector.")

    with st.form("ingest_form", clear_on_submit=False):
        c1, c2, c3 = st.columns([4, 2, 1])
        url = c1.text_input("Website URL", placeholder="https://example.com")
        name = c2.text_input("Name (optional)", placeholder="defaults to the domain")
        max_pages = c3.number_input("Max pages", min_value=1, max_value=1000, value=settings.crawl_max_pages)
        submitted = st.form_submit_button("Index website", type="primary")

    if submitted:
        try:
            validate_start_url(url, allow_private=settings.crawl_allow_private_hosts)
        except ValueError as exc:
            st.error(str(exc))
        else:
            _run_ingestion(url, name or None, int(max_pages))

    st.divider()
    _knowledge_base_overview()


# The dashboard reads through `st.cache_data` helpers keyed on a session "kb_version" counter: a
# plain rerun (typing in a search box, toggling an expander) does not touch the database, while
# "Refresh status", Re-index and Delete bump the counter so the next read is fresh. ttl=60 keeps
# other sessions from showing data older than a minute. Only metadata is cached, never chunk text.
KB_FILTER_KEY = "kb_chatbot_filter"
KB_SITE_KEY = "kb_site_select"
KB_VERSION_KEY = "kb_version"
KB_STALE_KEY = "kb_stale_days"
KB_FLASH_KEY = "kb_flash"  # result of a Re-index, shown once after the rerun that refreshes the overview
KB_CACHE_TTL = 60


def _kb_version() -> int:
    return int(st.session_state.get(KB_VERSION_KEY, 0))


def _bump_kb_version() -> None:
    """Invalidate the cached dashboard reads for this session (after Refresh / Re-index / Delete)."""
    st.session_state[KB_VERSION_KEY] = _kb_version() + 1


@st.cache_data(ttl=KB_CACHE_TTL, show_spinner=False)
def _kb_overview(chatbot_id: int | None, version: int) -> list[dict]:
    return run_read(lambda conn: repo.list_websites_overview(conn, chatbot_id))


@st.cache_data(ttl=KB_CACHE_TTL, show_spinner=False)
def _kb_pages(website_id: int, search: str, offset: int, version: int) -> tuple[list[dict], int]:
    return run_read(lambda conn: repo.list_pages_with_chunk_counts(
        conn, website_id, search=search, limit=repo.PAGE_LIST_SIZE, offset=offset))


@st.cache_data(ttl=KB_CACHE_TTL, show_spinner=False)
def _kb_missing(website_id: int, version: int) -> list[dict]:
    return run_read(lambda conn: repo.list_pages_without_chunks(conn, website_id))


@st.cache_data(ttl=KB_CACHE_TTL, show_spinner=False)
def _kb_duplicates(website_id: int, version: int) -> list[dict]:
    return run_read(lambda conn: repo.list_duplicate_page_urls(conn, website_id))


@st.cache_data(ttl=KB_CACHE_TTL, show_spinner=False)
def _kb_runs(website_id: int, version: int) -> list[dict]:
    return run_read(lambda conn: repo.list_ingestion_runs(conn, website_id, limit=10))


def _stale_after() -> timedelta:
    return timedelta(days=int(st.session_state.get(KB_STALE_KEY, settings.kb_stale_days)))


def _knowledge_base_overview() -> None:
    bots = _chatbots()
    head, tail = st.columns([5, 1], vertical_alignment="bottom")
    head.subheader("Knowledge base overview")
    if tail.button("Refresh status", icon=":material/refresh:", key="kb_refresh", width="stretch",
                   help="Re-read every number on this page from the database."):
        _bump_kb_version()
        _db_health.clear()
        st.toast("Status reloaded from the database", icon=":material/check:")

    bot_options: dict[int | None, str] = {None: "All chatbots"} | {b["id"]: _bot_label(b) for b in bots}
    f1, f2 = st.columns([3, 1], vertical_alignment="bottom")
    chatbot_id = f1.selectbox(
        "Show websites used by", list(bot_options), format_func=bot_options.get, key=KB_FILTER_KEY,
        help="Scope the overview to the websites attached to one chatbot.",
    )
    f2.number_input(
        "Stale after (days)", min_value=1, max_value=3650, value=settings.kb_stale_days, step=1, key=KB_STALE_KEY,
        help=f"Warn when a website was last indexed more than this many days ago (default KB_STALE_DAYS={settings.kb_stale_days}, "
             "this session only).",
    )
    sites = _kb_overview(chatbot_id, _kb_version())
    if flash := st.session_state.pop(KB_FLASH_KEY, None):
        st.success(flash, icon=":material/check:")
    st.caption(f"Loaded from the database at {datetime.now():%H:%M:%S} (cached for up to {KB_CACHE_TTL}s). "
               "Click *Refresh status* after an ingestion run elsewhere (CLI, another tab).")
    if not sites:
        if chatbot_id is None:
            st.info("No websites indexed yet. Add one above.")
        else:
            st.info("No websites are attached to this chatbot yet (Chatbots page).")
        return

    names = kb.chatbot_names_by_website(bots)
    stale_after = _stale_after()
    now = datetime.now(timezone.utc)
    st.dataframe(
        [
            {
                "id": w["id"],
                "name": w["name"],
                "status": kb.website_status_label(w),
                "discovered": kb.fmt_count(w["run_pages_crawled"]),
                "indexed": kb.fmt_count(w["pages_indexed"]),
                "stored": kb.fmt_count(w["pages_total"]),
                "failed": kb.fmt_count(len(kb.failed_pages(w))) if w.get("run_id") else kb.NOT_AVAILABLE,
                "chunks": kb.fmt_count(w["chunks_total"]),
                "last crawled": kb.fmt_time(w["last_crawled_at"]),
                "last indexed": kb.fmt_time(w["last_indexed_at"]),
                "freshness": {"fresh": "fresh", "stale": "stale", "unknown": kb.NOT_AVAILABLE}[
                    kb.freshness(w["last_indexed_at"], now, stale_after)],
                "used by": ", ".join(names.get(w["id"], [])) or "—",
                "url": w["base_url"],
            }
            for w in sites
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("url"),
            "discovered": st.column_config.TextColumn("discovered", help="Pages the crawler fetched in the latest run"),
            "indexed": st.column_config.TextColumn("indexed", help="Stored pages with at least one chunk"),
            "stored": st.column_config.TextColumn("stored", help="Rows in the pages table"),
            "failed": st.column_config.TextColumn("failed", help="Pages that failed in the latest run"),
        },
    )
    stale = [w["name"] for w in sites if kb.freshness(w["last_indexed_at"], now, stale_after) == "stale"]
    if stale:
        st.warning(f"Stale indexing data (older than {stale_after.days} days): {', '.join(stale)}", icon=":material/history:")

    by_id = {w["id"]: w for w in sites}
    if st.session_state.get(KB_SITE_KEY) not in by_id:  # deleted, or filtered out by the chatbot filter
        st.session_state[KB_SITE_KEY] = sites[0]["id"]
    chosen = st.selectbox(
        "Website details", list(by_id), format_func=lambda i: f"[{i}] {by_id[i]['name']} — {by_id[i]['base_url']}",
        key=KB_SITE_KEY,
    )
    _website_detail(by_id[chosen], names.get(chosen, []))


def _website_detail(w: dict, bot_names: list[str]) -> None:
    now = datetime.now(timezone.utc)
    label = kb.website_status_label(w)
    failed = kb.failed_pages(w)
    st.markdown(f"### {escape_markdown(w['name'])} {kb.status_badge_markdown(label)}")
    st.caption(f"{w['base_url']} · used by: {', '.join(bot_names) or 'no chatbot yet'}")

    # -- warnings (never silently "fixed": the admin decides whether to re-index) -----------
    if kb.run_is_stalled(w, now, timedelta(hours=settings.kb_stalled_run_hours)):
        st.warning(
            f"This run started {kb.fmt_time(w['run_started_at'])} and never finished. The indexing "
            "process has probably stopped; re-index to start a new run.", icon=":material/hourglass_empty:",
        )
    elif w["status"] == "crawling" and not w.get("run_id"):
        st.info("Processing — start time not available (the run began before run tracking existed).")
    if w["status"] == "failed" and w["error"]:
        st.error(f"Last run failed: {w['error']}", icon=":material/error:")
    elif label == kb.PARTIAL:
        missing = (w["pages_total"] or 0) - (w["pages_indexed"] or 0)
        parts = []
        if failed:
            parts.append(f"{len(failed)} page(s) failed in the latest run")
        if missing > 0:
            parts.append(f"{missing} stored page(s) have no embeddings")
        st.warning("Partially indexed: " + " and ".join(parts) + ". Those pages cannot be retrieved until a re-index succeeds.",
                   icon=":material/warning:")
    elif w["error"]:
        st.warning(f"Last run finished with errors: {w['error']}", icon=":material/warning:")
    stale_after = _stale_after()
    if kb.freshness(w["last_indexed_at"], now, stale_after) == "stale":
        st.warning(
            f"Indexed data may be outdated: last successful indexing was {kb.fmt_time(w['last_indexed_at'])}, "
            f"more than {stale_after.days} days ago. Re-index to pick up changes on the site.",
            icon=":material/history:",
        )
    drift = kb.embedding_drift(
        w, settings.embedding_provider, settings.resolved_embedding_model, settings.resolved_embedding_dim
    )
    if drift:
        st.warning(drift, icon=":material/sync_problem:")

    # -- compact stat cards -----------------------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Pages indexed", kb.fmt_count(w["pages_indexed"]), border=True,
        help=f"Stored pages with at least one chunk. {kb.fmt_count(w['pages_total'])} page(s) stored, "
             f"{kb.fmt_count(w['run_pages_crawled'])} discovered by the latest crawl.",
    )
    c2.metric("Chunks stored", kb.fmt_count(w["chunks_total"]), border=True, help="Embedded text/image vectors in pgvector.")
    c3.metric(
        "Failed pages", kb.fmt_count(len(failed)) if w.get("run_id") else kb.NOT_AVAILABLE, border=True,
        help="Pages the latest run could not index (see the table below).",
    )
    c4.metric(
        "Last successful crawl", kb.fmt_time(w["last_crawled_at"]), border=True,
        help="When the crawler last fetched the site's pages. A crawl can succeed while embedding fails.",
    )
    c5.metric(
        "Last successful indexing", kb.fmt_time(w["last_indexed_at"]), border=True,
        help="When pages were last embedded and the website became ready for chat. "
             "Websites indexed before run tracking existed show 'Not available' until re-indexed.",
    )

    # -- latest run ---------------------------------------------------------------------------
    st.markdown("**Latest ingestion run**")
    if w.get("run_id"):
        r1, r2 = st.columns(2)
        r1.markdown(
            f"- Status: {kb.run_summary_label(w)}\n"
            f"- Started: {kb.fmt_time(w['run_started_at'], '%Y-%m-%d %H:%M:%S')}\n"
            f"- Crawl finished: {kb.fmt_time(w['run_crawl_finished_at'], '%Y-%m-%d %H:%M:%S')}\n"
            f"- Finished: {kb.fmt_time(w['run_finished_at'], '%Y-%m-%d %H:%M:%S')}"
        )
        r2.markdown(
            f"- Pages discovered: {kb.fmt_count(w['run_pages_crawled'])} · indexed: {kb.fmt_count(w['run_pages_indexed'])}\n"
            f"- Unchanged: {kb.fmt_count(w['run_pages_skipped_unchanged'])} · too short: {kb.fmt_count(w['run_pages_skipped_short'])}\n"
            f"- Chunks created: {kb.fmt_count(w['run_chunks_created'])}\n"
            f"- Embedding model: {w['run_embedding_model'] or kb.NOT_AVAILABLE}"
        )
        if w["run_error"] and w["run_status"] == "failed":
            st.caption(f"Run error: {w['run_error']}")
        st.markdown("**Failed pages**")
        if failed:
            st.dataframe(failed, width="stretch", hide_index=True,
                         column_config={"url": st.column_config.LinkColumn("url")})
        else:
            st.caption("No page errors in the latest run.")
    else:
        st.caption(f"{kb.NOT_AVAILABLE} — no run has been recorded for this website yet. "
                   "Runs are tracked from the next (re-)index.")

    # -- pages ---------------------------------------------------------------------------------
    with st.expander(f"Indexed pages ({kb.fmt_count(w['pages_total'])})", icon=":material/list:"):
        _pages_table(w, "list")

    # -- diagnostics (operational detail: admin only, and never shown on the Chat page) ---------
    with st.expander("Advanced diagnostics", icon=":material/troubleshoot:"):
        if not _is_admin():
            st.info("Diagnostics are available to administrators only.", icon=":material/lock:")
        else:
            _diagnostics(w, now)

    # -- manage --------------------------------------------------------------------------------
    st.markdown("**Manage**")
    b1, b2, b3 = st.columns([1, 1, 4])
    if b1.button("Re-index", key=f"reindex_{w['id']}", icon=":material/refresh:"):
        summary = _run_ingestion(w["base_url"], w["name"], w["max_pages"])
        if summary:  # the overview above was drawn before the run: rerun so it shows the new data
            st.session_state[KB_FLASH_KEY] = f"Re-indexed {w['name']}: {summary}"
            st.rerun()
    confirm = b3.checkbox("I understand this deletes all its pages and chunks", key=f"confirm_{w['id']}")
    if b2.button("Delete", key=f"delete_{w['id']}", type="secondary", disabled=not confirm, icon=":material/delete:"):
        with get_conn() as conn:
            repo.delete_website(conn, w["id"])
        _bump_chain_version()
        _bump_kb_version()
        st.success(f"Deleted website {w['id']}")
        st.rerun()


def _pages_table(w: dict, slot: str) -> None:
    """Searchable, paginated page list: one LIMIT/OFFSET query per view, never the whole site.
    `slot` keeps the widgets of the two places this is shown (list / diagnostics) apart."""
    wid = w["id"]
    size = repo.PAGE_LIST_SIZE
    search_key, page_key, last_key = f"kb_search_{wid}_{slot}", f"kb_pages_{wid}_{slot}", f"kb_last_search_{wid}_{slot}"
    search = kb.normalise_search(st.text_input(
        "Search pages", key=search_key, placeholder="title or URL contains…", help="Case-insensitive substring match",
    ))
    if st.session_state.get(last_key) != search:  # new search: back to the first page
        st.session_state[last_key] = search
        st.session_state.pop(page_key, None)

    # first query decides how many pages exist (after the filter), the selector follows
    rows, total = _kb_pages(wid, search, 0, _kb_version())
    _, n_pages, _ = kb.page_window(total, 1, size)
    page_no = 1
    if n_pages > 1:
        page_no = int(st.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1, key=page_key,
                                      help=f"{size} pages per view"))
        if page_no > 1:
            offset, _, _ = kb.page_window(total, page_no, size)
            rows, total = _kb_pages(wid, search, offset, _kb_version())
    errors = kb.page_error_by_url(w)
    if not rows:
        st.caption("No pages match this search." if search else "No pages stored for this website.")
        if errors and not search:  # failed *new* pages have no row at all: they only exist in the run record
            st.caption(f"{len(errors)} page(s) failed in the latest run (see *Failed pages* above).")
        return
    st.caption(kb.page_window(total, page_no, size)[2])
    st.dataframe(
        [
            {
                "title": r["title"] or "",
                "url": r["url"],
                "status": kb.page_status(r["chunk_count"], r["url"], set(errors)),
                "chunks": int(r["chunk_count"] or 0),
                "words": r["word_count"],
                "last indexed": kb.fmt_time(r["indexed_at"]),
                "error": errors.get(r["url"], ""),
            }
            for r in rows
        ],
        width="stretch", hide_index=True,
        column_config={"url": st.column_config.LinkColumn("url", help="Opens the original page")},
    )


def _diagnostics(w: dict, now: datetime) -> None:
    st.caption("Read-only. Nothing here re-embeds or modifies the knowledge base.")
    d1, d2 = st.columns(2)
    d1.markdown(
        "**Embedding configuration of this website**\n"
        f"- Provider: {w['embedding_provider'] or kb.NOT_AVAILABLE}\n"
        f"- Model: {w['embedding_model'] or kb.NOT_AVAILABLE}\n"
        f"- Dimensions: {kb.fmt_count(w['embedding_dim'])}\n"
        f"- Total vector records: {kb.fmt_count(w['chunks_total'])}"
    )
    d2.markdown(
        "**App configuration now**\n"
        f"- Provider: {settings.embedding_provider}\n"
        f"- Model: {settings.resolved_embedding_model}\n"
        f"- Dimensions: {settings.resolved_embedding_dim}\n"
        f"- Last indexing execution: {kb.run_summary_label(w)}"
        + (f" (finished {kb.fmt_time(w['run_finished_at'], '%Y-%m-%d %H:%M:%S')})" if w.get('run_finished_at') else "")
    )
    stale_after = _stale_after()
    fresh = kb.freshness(w["last_indexed_at"], now, stale_after)
    st.markdown(
        "**Freshness**\n"
        f"- Newest stored chunk: {kb.fmt_time(w['newest_chunk_at'])}\n"
        f"- Last successful indexing: {kb.fmt_time(w['last_indexed_at'])} "
        f"({'older than ' + str(stale_after.days) + ' days' if fresh == 'stale' else fresh})"
    )

    st.markdown("**Pages without embeddings**")
    missing = _kb_missing(w["id"], _kb_version())
    if missing:
        st.warning(
            f"{len(missing)} page(s) are stored but have no embeddings, so they can never be retrieved. "
            "Re-index the website to embed them.", icon=":material/warning:",
        )
        st.dataframe(
            [{"url": m["url"], "title": m["title"] or "", "words": m["word_count"], "crawled": kb.fmt_time(m["crawled_at"])}
             for m in missing],
            width="stretch", hide_index=True, column_config={"url": st.column_config.LinkColumn("url")},
        )
    else:
        st.caption("Every stored page has at least one chunk.")

    st.markdown("**Duplicate page URLs**")
    dupes = _kb_duplicates(w["id"], _kb_version())
    if dupes:
        st.warning(
            f"{len(dupes)} group(s) of URLs look like the same page (differ only by scheme, www, case, "
            "fragment or trailing slash). Their chunks are stored twice and may be retrieved twice.", icon=":material/warning:",
        )
        st.dataframe(
            [{"page": d["key"], "copies": d["n"], "urls": "\n".join(d["urls"])} for d in dupes],
            width="stretch", hide_index=True,
        )
    else:
        st.caption("No duplicate URLs found.")

    st.markdown("**Chunks per page**")
    _pages_table(w, "diag")

    st.markdown("**Ingestion errors and run history**")
    runs = _kb_runs(w["id"], _kb_version())
    if runs:
        st.dataframe(
            [
                {
                    "run": r["id"], "status": r["status"],
                    "started": kb.fmt_time(r["started_at"], "%Y-%m-%d %H:%M:%S"),
                    "finished": kb.fmt_time(r["finished_at"], "%Y-%m-%d %H:%M:%S"),
                    "discovered": r["pages_crawled"], "indexed": r["pages_indexed"],
                    "unchanged": r["pages_skipped_unchanged"], "chunks": r["chunks_created"],
                    "page errors": len(kb.failed_pages({"page_errors": r["page_errors"]})),
                    "model": r["embedding_model"] or "", "error": r["error"] or "",
                }
                for r in runs
            ],
            width="stretch", hide_index=True,
        )
    else:
        st.caption(f"{kb.NOT_AVAILABLE} — no runs recorded yet.")


def _run_ingestion(url: str, name: str | None, max_pages: int) -> str | None:
    """Run the pipeline with a progress box; returns the one-line summary, or None when it failed."""
    _embeddings()  # make sure the model is loaded (and cached) before we start
    with st.status(f"Indexing {url}…", expanded=True) as status:
        bar = st.progress(0.0)
        line = st.empty()

        def progress(msg: str, done: int, total: int) -> None:
            bar.progress(min(done / max(total, 1), 1.0))
            line.write(msg)

        try:
            result = ingest_website(url, name=name, max_pages=max_pages, progress=progress)
        except Exception as exc:  # show the error in the UI instead of a stack trace
            status.update(label="Indexing failed", state="error")
            st.error(str(exc))
            _bump_kb_version()  # the failed run is recorded too
            return None
        bar.progress(1.0)
        status.update(label="Indexing finished", state="complete")
        run = f" · run #{result.run_id}" if result.run_id else ""
        summary = (
            f"Crawled {result.pages_crawled} page(s) · indexed {result.pages_indexed} · "
            f"unchanged {result.pages_skipped_unchanged} · too short {result.pages_skipped_short} · "
            f"{result.chunks_created} chunks{run}"
        )
        if result.errors:
            summary += f" · {len(result.errors)} error(s)"
            st.warning("\n".join(result.errors[:10]))
        st.success(summary)
    _bump_chain_version()
    _bump_kb_version()  # the dashboard must show the new run immediately
    return summary


# ------------------------------------------------------------- page: chatbots
NEW_CHATBOT = "➕ New chatbot"
EDIT_CHOICE_KEY = "chatbot_edit_choice"  # the selectbox's own state
EDIT_PENDING_KEY = "chatbot_edit_pending"  # set by buttons *below* the selectbox, applied on the next rerun


def _bot_label(b: dict) -> str:
    return f"[{b['id']}] {b['name']}"


def _select_for_edit(label: str) -> None:
    """Streamlit forbids changing a widget's state after it was drawn, so queue it and rerun."""
    st.session_state[EDIT_PENDING_KEY] = label
    st.rerun()


def page_chatbots() -> None:
    st.set_page_config(layout="wide")
    if not _require_admin():
        return
    if not _require_schema():
        return
    st.title("Chatbots", icon=":material/smart_toy:")
    st.caption("A chatbot answers only from the websites attached to it.")

    sites = _websites()
    ready = [w for w in sites if w["status"] == "ready"]
    if not ready:
        st.info("Index at least one website first (Websites page).")
        return
    site_labels = {w["id"]: f"[{w['id']}] {w['name']} ({w['chunk_count']} chunks)" for w in ready}

    bots = _chatbots()
    edit_options = {NEW_CHATBOT: None} | {_bot_label(b): b for b in bots}
    if EDIT_PENDING_KEY in st.session_state:  # an Edit/Save/Delete button asked to change the selection
        st.session_state[EDIT_CHOICE_KEY] = st.session_state.pop(EDIT_PENDING_KEY)
    if st.session_state.get(EDIT_CHOICE_KEY) not in edit_options:  # e.g. the bot was deleted
        st.session_state[EDIT_CHOICE_KEY] = NEW_CHATBOT
    choice = st.selectbox("Create new or edit existing", list(edit_options.keys()), key=EDIT_CHOICE_KEY)
    editing = edit_options[choice]
    if editing:
        st.caption(f"✏️ Editing chatbot [{editing['id']}] **{editing['name']}** — changes are saved to the same chatbot.")

    providers = list(PROVIDER_LABELS.keys())
    with st.form("chatbot_form"):
        name = st.text_input("Name *", value=editing["name"] if editing else "")
        description = st.text_input("Description", value=(editing or {}).get("description") or "")
        system_prompt = st.text_area(
            "Extra instructions (optional)",
            value=(editing or {}).get("system_prompt") or "",
            placeholder="e.g. You are Acme's support assistant. Be friendly and brief.",
        )
        default_sites = [w for w in (editing["website_ids"] if editing else []) if w in site_labels]
        website_ids = st.multiselect(
            "Websites *", options=list(site_labels.keys()), default=default_sites, format_func=lambda i: site_labels[i]
        )
        c1, c2, c3 = st.columns(3)
        cur_provider = (editing or {}).get("llm_provider") or settings.llm_provider
        provider = c1.selectbox(
            "Chat model provider", providers, index=providers.index(cur_provider), format_func=lambda p: PROVIDER_LABELS[p]
        )
        # Show the model the bot *actually* uses: the saved one, or the provider default when none was stored.
        cur_model = ((editing or {}).get("llm_model") or LLM_DEFAULTS[cur_provider]) if editing else ""
        model = c2.text_input(
            "Model", value=cur_model, placeholder=f"default: {LLM_DEFAULTS[settings.llm_provider]}"
        )
        top_k = c3.number_input("Chunks to retrieve (top_k)", 1, 20, value=(editing or {}).get("top_k") or settings.top_k)
        c1.caption("Defaults: " + ", ".join(f"{PROVIDER_LABELS[p]} → {m}" for p, m in LLM_DEFAULTS.items()))
        saved = st.form_submit_button("Update chatbot" if editing else "Create chatbot", type="primary")

    if saved:
        if not name.strip():
            st.error("Name is required.")
        elif not website_ids:
            st.error("Attach at least one website.")
        else:
            # Store provider and model explicitly (blank model -> that provider's default) so the
            # bot is pinned to what the user saw, and Edit always shows the same values back.
            fields = dict(
                name=name.strip(), description=description.strip() or None,
                system_prompt=system_prompt.strip() or None,
                llm_provider=provider,
                llm_model=model.strip() or LLM_DEFAULTS[provider], top_k=int(top_k),
            )
            try:
                with get_conn() as conn:  # a write: not retried
                    if editing:
                        bot = repo.update_chatbot(conn, editing["id"], **fields)
                    else:
                        bot = repo.create_chatbot(conn, **fields)
                    repo.set_chatbot_websites(conn, bot["id"], website_ids)
                _bump_chain_version()
                st.toast(f"{'Updated' if editing else 'Created'} chatbot [{bot['id']}] {bot['name']} — {describe_llm(bot)}", icon="✅")
                # keep the (possibly renamed) bot selected after the rerun; a new bot resets the form
                _select_for_edit(_bot_label(bot) if editing else NEW_CHATBOT)
            except psycopg.errors.UniqueViolation:
                st.error(f"A chatbot named “{name.strip()}” already exists. Choose another name.")
            except ValueError as exc:
                st.error(str(exc))

    if editing:
        _branding_panel(editing)
    else:
        st.caption("Branding & Appearance (logo, colours, welcome text, starter questions) becomes available once the chatbot is created.")

    st.divider()
    bots = _chatbots()
    if bots:
        st.subheader("Existing chatbots")
        for b in bots:
            c0, c1, c2, c3 = st.columns([0.5, 4.5, 1, 1])
            logo = _logo_bytes(b["id"], str(b.get("logo_updated_at") or "")) if b.get("has_logo") else None
            if logo:
                c0.image(logo, width=40)
            else:
                c0.markdown(":material/smart_toy:")
            shown = branding_of(b).display_name
            c1.markdown(
                f"**[{b['id']}] {b['name']}**" + (f" · shown as **{shown}**" if shown != b["name"].title() else "")
                + f" — {describe_llm(b)} · top_k {b['top_k']}  \n"
                + (f"_{b['description']}_  \n" if b.get("description") else "")
                + "Websites: " + (", ".join(site_labels.get(w, f"[{w}] (not ready)") for w in b["website_ids"]) or "none")
            )
            if c2.button("Edit", icon=":material/edit:", key=f"edit_bot_{b['id']}", width="stretch"):
                _select_for_edit(_bot_label(b))  # loads it into the form above
            if c3.button("Delete", icon=":material/delete:", key=f"del_bot_{b['id']}", width="stretch"):
                with get_conn() as conn:
                    repo.delete_chatbot(conn, b["id"])
                _bump_chain_version()
                st.toast(f"Deleted chatbot [{b['id']}] {b['name']} (with its evaluation test cases and results)", icon="🗑️")
                st.rerun()  # a deleted selection falls back to "New chatbot" above


# -------------------------------------------------------- chatbots: branding
@st.cache_data(show_spinner=False)
def _logo_bytes(chatbot_id: int, logo_version: str) -> bytes | None:
    """The chatbot's logo (small PNG) from the database, cached per upload.

    `logo_version` is `logo_updated_at` as text: every upload/removal changes it, so every
    session picks up the new image without clearing caches by hand.
    """
    return run_read(lambda conn: repo.get_chatbot_logo(conn, chatbot_id))


@st.cache_data(show_spinner=False)
def _processed_upload(file_id: str, data: bytes) -> bytes:
    """Validated + resized upload, cached per file so typing in the form does not re-decode it."""
    return process_logo(data)


THEME_PRIMARY = "#60a5fa"  # [theme.dark] primaryColor in .streamlit/config.toml


def _use_theme_colour(chatbot_id: int) -> None:
    """Button callback: put the picker back to the theme colour (= no custom accent)."""
    st.session_state[f"brand_accent_{chatbot_id}"] = THEME_PRIMARY


def _branding_preview(bot: dict, values: dict, logo: bytes | None) -> None:
    """What the Chat page's welcome screen will look like with these (unsaved) values."""
    merged = {**bot, **{k: v for k, v in values.items() if v}, "has_logo": logo is not None,
              "logo_updated_at": "preview" if logo else None}
    b = branding_of(merged)
    if b.accent:
        st.html(accent_css(b.accent))
    with st.container(border=True, horizontal_alignment="center"):
        if logo:
            st.image(logo, width=72)
        else:
            st.markdown(":material/smart_toy:", text_alignment="center")
        st.subheader(escape_markdown(b.heading), text_alignment="center")
        st.caption(escape_markdown(b.description), text_alignment="center")
        st.pills("Preview chips", [starter_text(s) for s in b.starters], label_visibility="collapsed",
                 key=f"brand_preview_pills_{bot['id']}", disabled=True)
        st.text_input("Preview input", placeholder=b.placeholder, label_visibility="collapsed",
                      key=f"brand_preview_input_{bot['id']}", disabled=True)
        st.button("Primary button", type="primary", key=f"brand_preview_btn_{bot['id']}", disabled=True)


def _branding_panel(bot: dict) -> None:
    """"Branding & Appearance" for the chatbot being edited: fields, live preview, save/reset.

    Lives outside the chatbot form so every change reruns and the preview updates live.
    Widget keys carry the chatbot id, so switching to another chatbot never shows stale values.
    """
    bid = bot["id"]
    b = branding_of(bot)
    with st.expander("Branding & Appearance", icon=":material/palette:", expanded=True):
        left, right = st.columns([3, 2], gap="large")
        with left:
            values = {
                "display_name": st.text_input("Display name", value=bot.get("display_name") or "", max_chars=MAX_DISPLAY_NAME,
                                              placeholder=b.display_name, key=f"brand_name_{bid}"),
                "welcome_heading": st.text_input("Welcome heading", value=bot.get("welcome_heading") or "", max_chars=MAX_HEADING,
                                                 placeholder=f"Ask anything about {b.display_name}", key=f"brand_heading_{bid}"),
                "description": st.text_area("Welcome description", value=bot.get("description") or "", max_chars=MAX_DESCRIPTION,
                                            height=80, key=f"brand_desc_{bid}"),
                "input_placeholder": st.text_input("Chat input placeholder", value=bot.get("input_placeholder") or "",
                                                   max_chars=MAX_PLACEHOLDER, placeholder=f"Message {b.display_name}…",
                                                   key=f"brand_placeholder_{bid}"),
                "starter_questions": st.text_area(
                    "Starter questions",
                    value="\n".join(starter_text(s) for s in (bot.get("starter_questions") or [])),
                    height=110, key=f"brand_starters_{bid}",
                    help=f"One question per line, up to {MAX_STARTERS}. The line order is the order shown; "
                         "edit, delete or move lines to change them. Leave empty for the default questions.",
                    placeholder="What do you offer?\nHow do I get started?",
                ),
            }
            st.caption(f"Up to {MAX_STARTERS} starter questions, one per line, shown in this order. "
                       "Empty = the default questions.")
            with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
                picked = st.color_picker("Accent colour", value=bot.get("accent_color") or THEME_PRIMARY,
                                         key=f"brand_accent_{bid}")
                st.button("Use theme colour", type="tertiary", icon=":material/format_color_reset:",
                          key=f"brand_accent_reset_{bid}", on_click=_use_theme_colour, args=(bid,))
            st.caption("Applied to the New chat button, starter chips, the open conversation, the chat input and links. "
                       "The theme colour means: no custom accent.")
            # the theme's own colour = no custom accent (nothing is injected on the Chat page)
            values["accent_color"] = "" if picked.lower() == THEME_PRIMARY else picked
            upload = st.file_uploader("Logo (PNG, JPG or WebP, up to 2 MB)", type=["png", "jpg", "jpeg", "webp"],
                                      max_upload_size=2, key=f"brand_logo_{bid}")
            remove_logo = False
            if bot.get("has_logo") and not upload:
                remove_logo = st.checkbox("Remove the current logo", key=f"brand_remove_logo_{bid}")

            # Validate on every rerun so the preview and the error messages are live.
            cleaned: dict | None = None
            new_logo: bytes | None = None
            try:
                cleaned = validate_branding(values)
                if upload is not None:
                    new_logo = _processed_upload(upload.file_id, upload.getvalue())
            except ValueError as exc:
                st.error(str(exc), icon=":material/error:")
            if cleaned and cleaned.get("accent_color") and contrast_ratio(cleaned["accent_color"], DARK_BACKGROUND) < 3:
                st.warning("That accent colour is hard to see on the dark background.", icon=":material/contrast:")

            a1, a2 = st.columns(2)
            if a1.button("Save branding", type="primary", icon=":material/save:", key=f"brand_save_{bid}",
                         disabled=cleaned is None):
                _save_branding(bid, cleaned, new_logo, remove_logo)
            if a2.button("Restore defaults", icon=":material/restart_alt:", key=f"brand_reset_{bid}"):
                _confirm_reset_branding(bot)
        with right:
            st.caption("Preview")
            preview_logo = new_logo if new_logo else (
                None if remove_logo else (_logo_bytes(bid, b.logo_version) if b.has_logo else None)
            )
            _branding_preview(bot, cleaned or {}, preview_logo)


def _save_branding(chatbot_id: int, cleaned: dict, new_logo: bytes | None, remove_logo: bool) -> None:
    if not _is_admin():  # defence in depth: the page is gated, but never write without the check
        st.error("Not allowed.", icon=":material/lock:")
        return
    try:
        with get_conn() as conn:  # one transaction: text fields + logo together
            row = repo.update_chatbot_branding(conn, chatbot_id, **cleaned)
            if row is None:
                raise ValueError("This chatbot no longer exists.")
            if new_logo is not None:
                repo.set_chatbot_logo(conn, chatbot_id, new_logo)
            elif remove_logo:
                repo.set_chatbot_logo(conn, chatbot_id, None)
    except ValueError as exc:
        st.error(str(exc), icon=":material/error:")
        return
    except Exception:
        log.exception("Saving branding failed")
        st.error("Branding could not be saved. Please try again.", icon=":material/error:")
        return
    _logo_bytes.clear()
    st.toast("Branding saved", icon=":material/check:")
    st.rerun()


@st.dialog("Restore default branding?")
def _confirm_reset_branding(bot: dict) -> None:
    st.write(f"Logo, display name, welcome text, accent colour and starter questions of **{bot['name']}** will be reset.")
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Cancel", key="brand_reset_cancel"):
            st.rerun()
        if st.button("Restore defaults", type="primary", icon=":material/restart_alt:", key="brand_reset_ok"):
            if not _is_admin():
                st.error("Not allowed.", icon=":material/lock:")
                return
            with get_conn() as conn:
                repo.reset_chatbot_branding(conn, bot["id"])
            _logo_bytes.clear()
            st.toast("Default branding restored", icon=":material/check:")
            st.rerun()


# ----------------------------------------------------------------- page: chat
SUGGESTIONS = [
    ":material/help: What is this website about?",
    ":material/image: Show me the logo",
    ":material/mail: How can I contact you?",
]
USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/smart_toy:"


def _assistant_avatar(bot: dict):
    """The chatbot's logo when it has one, else the default robot icon."""
    b = branding_of(bot)
    if b.has_logo:
        data = _logo_bytes(bot["id"], b.logo_version)
        if data:
            return io.BytesIO(data)
    return ASSISTANT_AVATAR


# ------------------------------------------------- sidebar: conversation list
CONVERSATION_LIST_HEIGHT = 360  # px; the list scrolls on its own, nav / "New chat" stay put
SEARCH_KEY = "conv_search"


@st.cache_data(ttl=300, show_spinner=False)
def _conversations(chatbot_id: int, version: int) -> list[dict]:
    """Conversation list for the sidebar, cached per chatbot.

    Every keystroke in the search box reruns the script; this keeps those reruns off
    PostgreSQL. `version` is bumped whenever a conversation is created, renamed or deleted.
    """
    rows = run_read(lambda conn: repo.list_conversations(conn, chatbot_id))
    return [c for c in rows if c["message_count"]]


def _bump_conversations() -> None:
    st.session_state["conv_version"] = st.session_state.get("conv_version", 0) + 1


def _clear_search() -> None:
    st.session_state[SEARCH_KEY] = ""


# The chat is three parallel lists in session state, one entry per message:
#   history      - LangChain HumanMessage / AIMessage (what the chain sees)
#   sources      - None for user turns, the list of source dicts for assistant turns
#   message_meta - {"id"} for user turns, {"id", "rating", "gen"} for assistant turns:
#                  the messages.id row (None when saving failed), the stored thumbs rating
#                  (1 / -1 / None) and how many times the answer was regenerated.
def _new_meta(
    message_id: int | None, rating: int | None = None, gen: int = 0, suggestions: list[str] | None = None
) -> dict:
    return {"id": message_id, "rating": rating, "gen": gen, "suggestions": list(suggestions or [])}


def _meta_from_row(r: dict) -> dict:
    if r["role"] != "assistant":
        return {"id": r.get("id")}
    return _new_meta(r.get("id"), r.get("rating"), r.get("regeneration_count") or 0, r.get("suggestions") or [])


# Per-conversation transient keys (edit mode, queued work); cleared whenever the chat changes.
PENDING_QUESTION_KEY = "pending_question"
PENDING_EDITED_FROM_KEY = "pending_edited_from"  # parent message id to stamp on an edited question
REGENERATE_KEY = "regenerate_index"
EDITING_KEY = "editing_index"  # index of the user message currently in edit mode
EDIT_ERROR_KEY = "edit_error"  # one-shot flag: "Save & Submit" was clicked with an empty message
FLASH_WARNING_KEY = "flash_warning"
LAST_DEBUG_KEY = "last_debug"  # retrieval details of the last turn, for "Show retrieved chunks"
FAILED_TURN_KEY = "failed_turn"  # {"question", "message"} of a question whose answer failed (not in history)
LAST_ERROR_KEY = "last_error_message"
_TRANSIENT_KEYS = (
    PENDING_QUESTION_KEY, PENDING_EDITED_FROM_KEY, REGENERATE_KEY, EDITING_KEY, EDIT_ERROR_KEY, LAST_DEBUG_KEY,
    FAILED_TURN_KEY, LAST_ERROR_KEY,
)


def _set_chat(conversation_id: str, history: list, sources: list, message_meta: list) -> None:
    for k in _TRANSIENT_KEYS:
        st.session_state.pop(k, None)
    st.session_state.update(
        conversation_id=conversation_id, history=history, sources=sources, message_meta=message_meta
    )


def _reset_chat() -> None:
    """Start an empty conversation (new chat, chatbot switch, deleting the open conversation)."""
    _set_chat(new_conversation_id(), [], [], [])


def _append_turn(question: str, answer: str, sources: list[dict], ids, suggestions: list[str] | None = None) -> None:
    """`ids` is the `TurnIds` returned by `persist_turn`, or None when saving failed."""
    st.session_state["history"].extend([HumanMessage(content=question), AIMessage(content=answer)])
    st.session_state["sources"].extend([None, sources])
    st.session_state["message_meta"].extend(
        [{"id": ids.user_id if ids else None}, _new_meta(ids.assistant_id if ids else None, suggestions=suggestions)]
    )


def _load_conversation(conversation_id: str) -> None:
    rows = run_read(lambda conn: repo.get_messages(conn, conversation_id))
    _set_chat(
        conversation_id, history_from_rows(rows),
        [(r["sources"] or []) if r["role"] == "assistant" else None for r in rows],
        [_meta_from_row(r) for r in rows],
    )
    st.rerun()


def _rename_conversation(conversation_id: str, title: str) -> None:
    try:
        with get_conn() as conn:  # a write: not retried
            repo.rename_conversation(conn, conversation_id, title)
    except ValueError as exc:
        st.error(str(exc))
        return
    _bump_conversations()
    st.toast("Conversation renamed", icon=":material/check:")
    st.rerun()


def _delete_conversation(conversation_id: str) -> None:
    with get_conn() as conn:  # a write: not retried
        repo.delete_conversation(conn, conversation_id)
    if st.session_state.get("conversation_id") == conversation_id:  # it was open: start fresh
        _reset_chat()
    _bump_conversations()
    st.toast("Conversation deleted", icon=":material/delete:")
    st.rerun()


@st.dialog("Delete conversation?")
def _confirm_delete(conversation: dict) -> None:
    st.write(f"**{conversation['title'] or 'untitled'}** and its messages will be permanently deleted.")
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Cancel", key="confirm_delete_cancel"):
            st.rerun()
        if st.button("Delete", type="primary", icon=":material/delete:", key="confirm_delete_ok"):
            _delete_conversation(str(conversation["id"]))


def _conversation_row(c: dict, active: bool) -> None:
    cid = str(c["id"])
    title = c["title"] or "untitled"
    when = (c.get("updated_at") or c["created_at"]).astimezone().strftime("%b %d, %H:%M")
    # wrap=False: the "..." popover must stay on the title's line even when the title is long.
    # The CSS in `_conversation_list` lets the title shrink and ellipsize, and slides a long title
    # on hover. No tooltip of any kind on the title: Streamlit's `help` tooltip and the browser's
    # native `title` tooltip both spill over the chat area (see `_TITLE_FIX_JS`); the full title
    # and the date live in the "..." popover. The open conversation gets the filled style.
    branched = bool(c.get("parent_conversation_id"))  # created by editing a question (stage 10)
    with st.container(horizontal=True, gap=None, vertical_alignment="center", wrap=False):
        if st.button(
            title, key=f"conv_{cid}", type="secondary" if active else "tertiary", width="stretch", wrap=False,
            icon=":material/alt_route:" if branched else None,
        ):
            _load_conversation(cid)
        with st.popover("", icon=":material/more_horiz:", type="tertiary", width="content", help="Conversation actions"):
            st.caption(f"Last activity: {when}" + (" · branched from an edited question" if branched else ""))
            new_title = st.text_input("Rename", value=title, key=f"rename_{cid}", max_chars=repo.MAX_CONVERSATION_TITLE)
            with st.container(horizontal=True):
                if st.button("Save", icon=":material/check:", key=f"save_{cid}", disabled=new_title.strip() == title):
                    _rename_conversation(cid, new_title)
                if st.button("Delete", icon=":material/delete:", key=f"delete_{cid}"):
                    _confirm_delete(c)


def _search_box() -> str:
    """Compact search input with a clear button; returns the current query."""
    with st.container(horizontal=True, gap="xsmall", vertical_alignment="center"):
        query = st.text_input(
            "Search chats", key=SEARCH_KEY, type="search", icon=":material/search:",
            placeholder="Search chats...", label_visibility="collapsed", width="stretch",
        ) or ""
        if query:
            st.button(
                "", icon=":material/close:", key="conv_search_clear", type="tertiary",
                help="Clear search", on_click=_clear_search,
            )
    return query


# Streamlit gives every single-row button without `help` a native `title` attribute (so hovering
# "reveals the full label"). That browser tooltip spills over the chat area, and no CSS can remove
# an attribute, so this tiny page script strips it from the conversation rows and keeps doing so
# after every rerun (Streamlit re-renders the buttons). It is scoped to the list's container key
# and the sidebar's data-testid; nothing else on the page is touched.
_TITLE_FIX_JS = """
export default function () {
  const strip = () =>
    document.querySelectorAll(".st-key-conv_list button[title]").forEach((b) => b.removeAttribute("title"));
  strip();
  if (window.__convTitleFixBound) return;  // the function runs on every mount; observe once
  const sidebar = document.querySelector('[data-testid="stSidebar"]');
  if (!sidebar) return;
  window.__convTitleFixBound = true;
  new MutationObserver(strip).observe(sidebar, { childList: true, subtree: true, attributes: true, attributeFilter: ["title"] });
}
"""
SIDEBAR_TITLE_FIX = st.components.v2.component("conv_title_fix", js=_TITLE_FIX_JS)


def _conversation_list(convs: list[dict], active_id: str | None, query: str = "") -> None:
    if not convs:
        st.caption("No conversations yet")
        return
    # Scoped (via the container key) CSS for the conversation rows:
    # - button labels are centered by default and there is no native option for left-aligned text;
    # - a long title must *shrink* instead of pushing the "..." popover out of the row: every
    #   wrapper between the row and the <button> gets min-width:0 (flex items default to
    #   min-width:auto = their content width) and the label ellipsizes on one line;
    # - on hover a long title slides left by exactly its overflow and back (ChatGPT-style), so
    #   the end becomes readable without any tooltip. The label box is a size container, so
    #   `100cqw` is its width and `calc(100cqw - 100%)` the (negative) overflow of the text;
    #   `min(0px, ...)` keeps short titles still.
    st.html(
        "<style>"
        ".st-key-conv_list [data-testid='stHorizontalBlock']{flex-wrap:nowrap}"
        ".st-key-conv_list [data-testid='stHorizontalBlock']>.stElementContainer:first-child{flex:1 1 0;min-width:0}"
        ".st-key-conv_list .stButton,.st-key-conv_list .stButton>div,.st-key-conv_list .stTooltipIcon,"
        ".st-key-conv_list .stTooltipHoverTarget{min-width:0;max-width:100%}"
        ".st-key-conv_list button{justify-content:flex-start;min-width:0}"
        ".st-key-conv_list button>div{min-width:0;overflow:hidden;justify-content:flex-start;margin:0;container-type:inline-size}"
        ".st-key-conv_list button p{display:inline-block;width:max-content;max-width:100%;text-align:left;"
        "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;vertical-align:top}"
        ".st-key-conv_list button:hover p{max-width:none;text-overflow:clip;"
        "animation:conv-title-slide 5s ease-in-out .4s infinite alternate}"
        "@keyframes conv-title-slide{from{transform:translateX(0)}to{transform:translateX(min(0px,calc(100cqw - 100%)))}}"
        "@media (prefers-reduced-motion:reduce){.st-key-conv_list button:hover p{animation:none;max-width:100%;text-overflow:ellipsis}}"
        "</style>"
    )
    with st.container(height=CONVERSATION_LIST_HEIGHT, border=False, key="conv_list"):
        SIDEBAR_TITLE_FIX(key="conv_title_fix", height=0)
        if query:
            # Search: a flat, in-memory filter of the loaded list (no extra DB query).
            matches = filter_conversations(convs, query)
            if not matches:
                st.caption("No matching conversations found")
                return
            st.caption(f"Results ({len(matches)})")
            with st.container(gap=None):
                for c in matches:
                    _conversation_row(c, active=str(c["id"]) == active_id)
            return
        for group, items in group_conversations(convs):
            st.caption(group)
            with st.container(gap=None):
                for c in items:
                    _conversation_row(c, active=str(c["id"]) == active_id)


def _about_details() -> None:
    """Environment details (DB, embeddings, LLM)."""
    try:
        h = _db_health()
        st.caption(f"PostgreSQL {h['server_version']} · pgvector {h['pgvector'] or 'missing'}  \n"
                   f"Database: {settings.database_label()}")
    except Exception as exc:
        st.error(f"Database unreachable: {exc}")
    st.caption(
        f"Embeddings: {settings.embedding_provider} · {settings.resolved_embedding_model} ({settings.resolved_embedding_dim}d)  \n"
        f"Default LLM: {describe_llm()}"
    )


# ------------------------------------------------------- editing a question
# The pencil under a user message is kept faint until that message is hovered or focused
# (keyboard users tab to it), so the chat stays uncluttered. Scoped to the `useract_` container
# keys; nothing is hidden, only dimmed.
_USER_ACTIONS_CSS = (
    "<style>"
    "[class*='st-key-useract_']{opacity:.45;transition:opacity .15s ease}"
    "[data-testid='stChatMessage']:hover [class*='st-key-useract_'],"
    "[class*='st-key-useract_']:focus-within{opacity:1}"
    "</style>"
)


def _start_edit(index: int) -> None:
    st.session_state[EDITING_KEY] = index
    st.session_state.pop(EDIT_ERROR_KEY, None)


def _cancel_edit() -> None:
    st.session_state.pop(EDITING_KEY, None)
    st.session_state.pop(EDIT_ERROR_KEY, None)


def _submit_edit(index: int, text_key: str) -> None:
    """"Save & Submit" callback: fork the conversation before the edited question, then ask it.

    Runs before the script (button callback). The turns *before* the edited message are copied
    into a new conversation in one transaction (`repo.fork_conversation`); the old answer, its
    sources and feedback, and every later turn stay only in the original conversation, which is
    never modified. The edited question is then queued exactly like a Retry, so the next run
    takes the ordinary new-question path: fresh retrieval, streaming, one save on success.
    """
    text = (st.session_state.get(text_key) or "").strip()
    if not text:
        st.session_state[EDIT_ERROR_KEY] = True  # stay in edit mode and complain
        return
    history: list = st.session_state["history"]
    if index > len(history):  # stale widget (the conversation changed under it)
        _cancel_edit()
        return
    sources: list = st.session_state["sources"]
    meta: list = st.session_state["message_meta"]
    prefix = history[:index]
    prefix_sources = [sources[k] if k < len(sources) else None for k in range(index)]
    turns = [
        ("user" if isinstance(m, HumanMessage) else "assistant", m.content, prefix_sources[k])
        for k, m in enumerate(prefix)
    ]
    edited_from = (meta[index] or {}).get("id") if index < len(meta) else None
    new_id = new_conversation_id()
    try:
        with get_conn() as conn:  # a write: not retried
            ids = repo.fork_conversation(
                conn, new_id=new_id, chatbot_id=st.session_state["active_bot"],
                source_id=st.session_state["conversation_id"], forked_from_message_id=edited_from, turns=turns,
            )
    except Exception:
        log.exception("Forking the conversation for an edited question failed")
        st.session_state[FLASH_WARNING_KEY] = (
            "The edited question could not be saved as a new branch. Please try again."
        )
        return
    new_meta = [
        {"id": ids[k]} if isinstance(m, HumanMessage) else _new_meta(ids[k]) for k, m in enumerate(prefix)
    ]
    _set_chat(new_id, prefix, prefix_sources, new_meta)
    st.session_state[PENDING_QUESTION_KEY] = text
    st.session_state[PENDING_EDITED_FROM_KEY] = edited_from
    _bump_conversations()


def _render_user_message(text: str, *, index: int | None = None, cid: str | None = None) -> None:
    """User turns sit in a rounded bordered bubble (radius comes from the theme).

    With `index` (a message from history) the bubble gets a pencil that switches it into edit
    mode: a text area with the original text plus Cancel / Save & Submit. The in-flight
    question (no index) has no actions.
    """
    editing = index is not None and st.session_state.get(EDITING_KEY) == index
    with st.chat_message("user", avatar=USER_AVATAR):
        if not editing:
            with st.container(border=True):
                st.markdown(text)
            if index is not None:
                with st.container(key=f"useract_{cid}_{index}", horizontal=True, gap="xsmall"):
                    st.button(
                        "", icon=":material/edit:", type="tertiary", help="Edit message",
                        key=f"edit_{cid}_{index}", on_click=_start_edit, args=(index,),
                    )
            return
        text_key = f"edit_text_{cid}_{index}"
        st.text_area("Edit your message", value=text, key=text_key, label_visibility="collapsed")
        if st.session_state.get(EDIT_ERROR_KEY):
            st.error("Message cannot be empty.", icon=":material/error:")
        with st.container(horizontal=True, gap="small", horizontal_alignment="right"):
            st.button("Cancel", type="tertiary", key=f"edit_cancel_{cid}_{index}", on_click=_cancel_edit)
            st.button(
                "Save & Submit", type="primary", icon=":material/send:", key=f"edit_save_{cid}_{index}",
                on_click=_submit_edit, args=(index, text_key),
            )


def _empty_state(bot: dict) -> str | None:
    """Centered welcome + suggestion chips; returns the chosen suggestion as the question."""
    b = branding_of(bot)
    st.space("large")
    with st.container(horizontal_alignment="center"):
        logo = _logo_bytes(bot["id"], b.logo_version) if b.has_logo else None
        if logo:
            st.image(logo, width=72)
        else:
            st.markdown(":material/forum:", text_alignment="center")
        # User-provided text goes through escape_markdown so it can only ever be plain text.
        st.subheader(escape_markdown(b.heading), text_alignment="center")
        st.caption(escape_markdown(b.description), text_alignment="center")
        # The widget only exists for the selected chatbot and `_reset_chat()` runs on every switch,
        # so one key is enough: a selection can never carry over to another chatbot.
        choice = st.pills("Try asking", b.starters, label_visibility="collapsed", key="chat_suggestion")
    if not choice:
        return None
    return starter_text(choice)  # drop the ":material/...:" icon prefix of the default chips


def page_chat() -> None:
    if not _require_schema():
        return
    bots = _chatbots()
    bots = [b for b in bots if b["website_ids"]]
    if not bots:
        st.title("Chat", icon=":material/chat:")
        st.info("Create a chatbot with at least one website first (Chatbots page).", icon=":material/info:")
        return

    with st.sidebar:
        labels = {b["id"]: branding_of(b).display_name for b in bots}
        chatbot_id = st.selectbox(
            "Chatbot", list(labels.keys()), format_func=lambda i: labels[i], key="chat_bot_select",
            label_visibility="collapsed",
        )
        bot = next(b for b in bots if b["id"] == chatbot_id)
        st.caption(describe_llm(bot))

        # reset the conversation when the chatbot changes
        if st.session_state.get("active_bot") != chatbot_id:
            st.session_state["active_bot"] = chatbot_id
            _reset_chat()

        if st.button("New chat", icon=":material/add:", type="primary", width="stretch"):
            _reset_chat()
            st.rerun()

        # Only this chatbot's conversations (list_conversations filters by chatbot_id),
        # cached so typing in the search box does not query the database.
        previous = _conversations(chatbot_id, st.session_state.get("conv_version", 0))
        query = _search_box()
        _conversation_list(previous, active_id=st.session_state.get("conversation_id"), query=query)

        with st.expander("Advanced settings", icon=":material/tune:"):
            show_debug = st.toggle("Show retrieved chunks", value=False)
            _about_details()

    history: list = st.session_state.setdefault("history", [])
    sources_by_turn: list = st.session_state.setdefault("sources", [])
    meta_by_turn: list = st.session_state.setdefault("message_meta", [])
    cid: str = st.session_state["conversation_id"]

    if flash := st.session_state.pop(FLASH_WARNING_KEY, None):  # a warning queued right before a rerun
        st.warning(flash, icon=":material/warning:")

    st.html(_USER_ACTIONS_CSS)
    # Branding is derived from the selected chatbot on every run (never kept in session state),
    # so switching chatbots can never show the previous chatbot's look.
    brand = branding_of(bot)
    if brand.accent:
        st.html(accent_css(brand.accent))
    avatar = _assistant_avatar(bot)
    if history:
        st.subheader(escape_markdown(brand.display_name))
        if bot.get("description"):
            st.caption(bot["description"])

    # The input is pinned to the bottom of the main area by Streamlit itself, so it can be
    # created before the history is drawn. submit_mode="disable": the input is disabled while
    # this run generates an answer, so a second question cannot be submitted mid-stream; the
    # same holds while a regeneration (queued by the Regenerate / Retry buttons) streams.
    # A question comes from the input on the run it was submitted, or from a queued Retry.
    # A rerun mid-generation (sidebar click, conversation switch, "New chat") simply stops this
    # script: the stream is abandoned and - because history and the DB are only updated after
    # a complete answer - nothing is saved or shown twice.
    regen_index = st.session_state.pop(REGENERATE_KEY, None)
    if regen_index is not None and not _is_last_assistant(history, regen_index):
        regen_index = None  # stale (the conversation changed meanwhile)
    question = st.chat_input(brand.placeholder, submit_mode="disable", disabled=regen_index is not None)
    if question:  # a freshly typed question cancels any open edit and never inherits an edit's parent link
        for k in (EDITING_KEY, EDIT_ERROR_KEY, PENDING_EDITED_FROM_KEY):
            st.session_state.pop(k, None)
    else:
        question = st.session_state.pop(PENDING_QUESTION_KEY, None)  # a queued Retry or an edited question

    for i, msg in enumerate(history):
        if isinstance(msg, HumanMessage):
            _render_user_message(msg.content, index=i, cid=cid)
            continue
        src = sources_by_turn[i] if i < len(sources_by_turn) else None
        meta = (meta_by_turn[i] if i < len(meta_by_turn) else None) or _new_meta(None)
        with st.chat_message("assistant", avatar=avatar):
            if i == regen_index:
                _regenerate(chatbot_id, i, msg.content, meta, cid, show_debug)  # reruns on success
                continue
            _render_answer(msg.content, src)
            _assistant_actions(i, msg.content, meta, cid, is_last=_is_last_assistant(history, i))
            if src:
                asked = history[i - 1].content if i >= 1 and isinstance(history[i - 1], HumanMessage) else None
                _render_sources(src, key=f"{cid}_{i}", question=asked, show_scores=show_debug)
            _render_debug(i, show_debug)
            if _is_last_assistant(history, i):  # suggestions belong to the latest answer only
                _render_followups(i, meta.get("suggestions") or [], cid, disabled=bool(question))

    # A question whose answer failed is not in history; keep showing it (with its error, Retry and
    # pencil) until it is retried, edited or replaced, so a rerun does not make it vanish.
    failed = st.session_state.get(FAILED_TURN_KEY)
    if not question and failed:
        _render_user_message(failed["question"], index=len(history), cid=cid)
        with st.chat_message("assistant", avatar=avatar):
            _error_state(failed["message"], on_click=_queue_retry, args=(failed["question"],), key="retry_turn")
        return

    if not question and not history:
        question = _empty_state(bot)  # a suggestion chip becomes the first question
    if not question:
        return
    st.session_state.pop(FAILED_TURN_KEY, None)

    # Shown immediately, before any work starts. `len(history)` is the index this question gets
    # once the turn is appended, so its pencil keeps working after the rerun below - and if the
    # answer fails, the pencil lets the user correct the question and resubmit it.
    _render_user_message(question, index=len(history), cid=cid)
    with st.chat_message("assistant", avatar=avatar):
        generated = _generate_answer(
            chatbot_id, question, history,
            retry_cb=_queue_retry, retry_args=(question,), retry_key="retry_turn", show_debug=show_debug,
        )
        if generated is None:  # nothing persisted, history unchanged; remember the question for the next run
            st.session_state[FAILED_TURN_KEY] = {"question": question, "message": st.session_state.get(LAST_ERROR_KEY, "")}
            return
        result, src, suggestions = generated

        # Phase 3: one write, only after a complete answer. An edited question also records which
        # message it replaced (the branch itself was created when "Save & Submit" was clicked).
        ids = None
        try:
            ids = persist_turn(
                cid, chatbot_id, question, result,
                edited_from_message_id=st.session_state.get(PENDING_EDITED_FROM_KEY), suggestions=suggestions,
            )
            st.session_state.pop(PENDING_EDITED_FROM_KEY, None)  # consumed; a Retry keeps it until saved
        except Exception:
            log.exception("Saving the conversation failed")
            st.session_state[FLASH_WARNING_KEY] = (  # shown by the run after the rerun below
                "The answer could not be saved to your conversation history. It will disappear on reload."
            )

    _remember_debug(len(history) + 1, result, show_debug)
    _append_turn(question, result.answer, src, ids, suggestions)
    _bump_conversations()  # a new or updated conversation must show up in the sidebar
    # Redraw from state: the empty-state header/chips disappear, the new turn gets its pencil and
    # answer actions from the normal loop, and the sidebar shows the conversation right away.
    st.rerun()


def _is_last_assistant(history: list, i: int) -> bool:
    return 1 <= i == len(history) - 1 and isinstance(history[i], AIMessage)


def _ask_followup(question: str) -> None:
    """Suggestion-button callback: ask the suggested question as an ordinary new message.

    It goes through the same queued-question path as Retry (one submission, popped once), but a
    follow-up is a *new* question, so any open edit and a pending edit link are dropped first.
    """
    for k in (EDITING_KEY, EDIT_ERROR_KEY, PENDING_EDITED_FROM_KEY):
        st.session_state.pop(k, None)
    st.session_state[PENDING_QUESTION_KEY] = question


def _render_followups(i: int, suggestions: list[str], cid: str, *, disabled: bool) -> None:
    """Compact "Suggested follow-up questions" row under the latest answer (nothing when empty)."""
    if not suggestions:
        return
    st.caption("Suggested follow-up questions")
    with st.container(horizontal=True, gap="small", wrap=True):
        for k, text in enumerate(suggestions):
            st.button(
                text, icon=":material/auto_awesome:", type="tertiary", wrap=True, key=f"fu_{cid}_{i}_{k}",
                on_click=_ask_followup, args=(text,), disabled=disabled,
            )


def _queue_retry(question: str) -> None:
    """Button callback: the next run re-asks the question as if it had just been submitted."""
    st.session_state[PENDING_QUESTION_KEY] = question


def _queue_regenerate(index: int) -> None:
    """Button callback: the next run regenerates the assistant message at `index` in place."""
    st.session_state[REGENERATE_KEY] = index


def _error_state(message: str, *, on_click, args: tuple, key: str) -> None:
    st.session_state[LAST_ERROR_KEY] = message  # so a failed question can be redrawn with its error
    st.error(message, icon=":material/error:")
    st.button("Retry", icon=":material/refresh:", key=key, on_click=on_click, args=args)


# --------------------------------------------------------- answer generation
def _generate_answer(
    chatbot_id: int, question: str, context_history: list, *, retry_cb, retry_args: tuple, retry_key: str,
    show_debug: bool,
) -> tuple | None:
    """Phases 1-2 of a turn, drawn into the current container: retrieval, then the streamed answer.

    Used both for a new question and for regenerating an existing answer. Returns
    `(RagAnswer, sources)` on success. On failure it has already shown the error and a Retry
    button (which calls `retry_cb(*retry_args)` on click) and returns None - nothing is saved.
    """
    indicator = st.empty()  # exactly one loading cue lives here, cleared on every exit path
    answer_slot = st.empty()
    tail = st.empty()  # below the answer: the "thinking of follow-ups" cue, cleared afterwards

    # Phase 1: retrieval (DB + at most one small rewrite call).
    indicator.markdown(":shimmer[Searching website content…]")
    try:
        chain, _ = _chain(chatbot_id, st.session_state.get("chain_version", 0), _rag_code_version())
        prepared = chain.prepare(question, context_history)
    except Exception as exc:
        log.exception("Preparing the answer failed")
        indicator.empty()
        _error_state(_friendly_error(exc), on_click=retry_cb, args=retry_args, key=retry_key)
        return None

    # Phase 2: stream the model's answer; nothing touches the database meanwhile.
    if prepared.ready is not None:  # greeting / nothing retrieved: canned text, no model call
        indicator.empty()
        result = prepared.ready
        answer_slot.markdown(result.answer)
    else:
        indicator.markdown(":shimmer[Preparing your answer…]")  # until the first token
        tokens: list[str] = []
        failure: list[BaseException] = []

        def guarded():
            try:
                for chunk in chain.stream_answer(prepared):
                    if not tokens:
                        indicator.empty()  # first token: the typing cue gives way to real text
                    tokens.append(chunk)
                    yield chunk
            except Exception as exc:  # surfaced after write_stream returns
                failure.append(exc)

        with answer_slot.container():
            st.write_stream(guarded)
        indicator.empty()
        if failure:
            log.exception("Streaming the answer failed", exc_info=failure[0])
            _error_state(
                _friendly_error(failure[0]) + (" The partial answer above was not saved." if tokens else ""),
                on_click=retry_cb, args=retry_args, key=retry_key,
            )
            return None
        result = chain.finalize(prepared, "".join(tokens))
        # The streamed text had plain [n] markers; redraw the final text with clickable citations
        # (and any post-processing changes).
        answer_slot.markdown(linkify_citations(result.answer, [s.to_dict() for s in result.sources]))

    # Phase 2b (optional): follow-up suggestions. One small extra model call that only sees the
    # passages the answer was built from; it never affects the answer, which is already on screen.
    suggestions: list[str] = []
    if result.grounded and settings.followup_count > 0 and hasattr(chain, "suggest_followups"):
        tail.markdown(":shimmer[Thinking of follow-up questions…]")
        try:
            suggestions = list(chain.suggest_followups(prepared, result, context_history))
        except Exception:
            log.exception("Follow-up suggestions failed")
            suggestions = []
        finally:
            tail.empty()
    return result, [s.to_dict() for s in result.sources], suggestions


def _remember_debug(index: int, result, show_debug: bool) -> None:
    """Keep the last turn's retrieval details so the debug expander survives the rerun."""
    if show_debug:
        st.session_state[LAST_DEBUG_KEY] = {"index": index, "result": result}


def _render_debug(index: int, show_debug: bool) -> None:
    """Draw the "Retrieved chunks" expander under the assistant message at `index` (if it is the last turn)."""
    debug = st.session_state.get(LAST_DEBUG_KEY)
    if not show_debug or not debug or debug["index"] != index:
        return
    result = debug["result"]
    with st.expander("Retrieved chunks (debug)", type="compact"):
        st.write(f"Standalone question: *{result.standalone_question}*")
        for d in result.documents:
            kind = d.metadata.get("type", "text")
            st.markdown(f"**[{d.metadata.get('chunk_id')}]** {kind} · score {d.metadata.get('score')} — {d.metadata.get('url')}")
            if kind == "image":
                st.image(d.metadata.get("image_url"), width=200)
            st.text(d.page_content[:700])


def _regenerate(chatbot_id: int, i: int, original: str, meta: dict, cid: str, show_debug: bool) -> None:
    """Regenerate the assistant message at `i` from its question and the turns before it.

    Called inside that message's chat bubble. The original answer stays in session state and
    in the database until a complete new answer exists; on failure it is shown again under
    the error so nothing is lost. On success the row is updated in place (no new user
    message) and the script reruns so the bubble is redrawn with fresh action buttons.
    """
    history: list = st.session_state["history"]
    question = history[i - 1].content
    generated = _generate_answer(
        chatbot_id, question, history[: i - 1],
        retry_cb=_queue_regenerate, retry_args=(i,), retry_key=f"retry_regen_{cid}_{i}", show_debug=show_debug,
    )
    if generated is None:
        st.caption("Your previous answer is kept:")
        old_sources = st.session_state["sources"]
        _render_answer(original, old_sources[i] if i < len(old_sources) else None)
        _assistant_actions(i, original, meta, cid, is_last=False)
        return
    result, src, suggestions = generated

    if meta["id"] is not None:
        try:
            with get_conn() as conn:  # a write: not retried (same policy as persist_turn)
                repo.update_message(conn, meta["id"], result.answer, src, suggestions)
        except Exception:
            log.exception("Saving the regenerated answer failed")
            st.session_state[FLASH_WARNING_KEY] = (
                "The regenerated answer could not be saved to your conversation history. "
                "It will disappear on reload."
            )
    history[i] = AIMessage(content=result.answer)
    st.session_state["sources"][i] = src
    st.session_state["message_meta"][i] = _new_meta(meta["id"], rating=None, gen=meta["gen"] + 1, suggestions=suggestions)
    _remember_debug(i, result, show_debug)
    _bump_conversations()
    st.rerun()


# ------------------------------------------------- assistant message actions
# Copy lives in a small st.components.v2 component: its JavaScript runs in the page itself (no
# iframe), so `navigator.clipboard.writeText` is called from a real click (every browser allows
# that) and the button can use the app theme through the --st-* CSS variables. Nothing is sent
# back to Python: copying never reruns the script.
_COPY_HTML = """
<button type="button" class="copy-btn" title="Copy response" aria-label="Copy response">
  <svg class="ic ic-copy" viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
  <svg class="ic ic-done" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>
  <span class="label">Copied</span>
</button>
"""
_COPY_CSS = """
.copy-btn { display: inline-flex; align-items: center; gap: .3rem; height: 2rem; min-width: 2rem;
  padding: 0 .45rem; border: 0; border-radius: 8px; background: transparent; cursor: pointer;
  color: var(--st-text-color, inherit); font-family: inherit; font-size: .8rem; line-height: 1;
  opacity: .7; transition: background-color .15s ease, opacity .15s ease; }
.copy-btn:hover, .copy-btn:focus-visible { background: var(--st-secondary-background-color, rgba(128,128,128,.18)); opacity: 1; outline: none; }
.copy-btn .ic { width: 1rem; height: 1rem; fill: none; stroke: currentColor; stroke-width: 2;
  stroke-linecap: round; stroke-linejoin: round; }
.copy-btn .ic-done, .copy-btn .label { display: none; }
.copy-btn.done { color: var(--st-primary-color, currentColor); opacity: 1; }
.copy-btn.done .ic-copy { display: none; }
.copy-btn.done .ic-done, .copy-btn.done .label { display: inline; }
"""
_COPY_JS = """
export default function (component) {
  const { data, parentElement } = component;
  const btn = parentElement.querySelector(".copy-btn");
  if (!btn) return;
  btn._text = (data && typeof data === "object" && "text" in data) ? String(data.text) : String(data ?? "");
  if (btn._bound) return;  // re-rendered with new data: keep the single click handler
  btn._bound = true;
  const legacyCopy = (text) => {  // for plain-http hosts where the Clipboard API is unavailable
    const ta = document.createElement("textarea");
    ta.value = text; ta.setAttribute("readonly", ""); ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
    document.body.removeChild(ta);
    return ok;
  };
  const showCopied = () => {
    btn.classList.add("done");
    clearTimeout(btn._timer);
    btn._timer = setTimeout(() => btn.classList.remove("done"), 1500);
  };
  btn.addEventListener("click", () => {
    const text = btn._text;
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(showCopied, () => { if (legacyCopy(text)) showCopied(); });
    } else if (legacyCopy(text)) {
      showCopied();
    }
  });
}
"""
COPY_BUTTON = st.components.v2.component("copy_button", html=_COPY_HTML, css=_COPY_CSS, js=_COPY_JS)


def _copy_button(text: str, key: str) -> None:
    # The text travels as data, never interpolated into the HTML (component HTML is not sanitised).
    COPY_BUTTON(data={"text": text}, key=key, width="content")


def _rating_to_feedback(rating: int | None) -> int | None:
    """DB rating (1 / -1 / None) -> st.feedback("thumbs") value (1 = up, 0 = down, None = unset)."""
    return None if rating is None else (1 if rating > 0 else 0)


def _save_feedback(message_id: int | None, widget_key: str) -> None:
    """st.feedback on_change callback: upsert the rating of one assistant message.

    Callbacks must never raise (that would abort the whole run), so errors become a toast.
    """
    value = st.session_state.get(widget_key)
    if message_id is None or value is None:
        return
    rating = 1 if value == 1 else -1
    try:
        with get_conn() as conn:  # a write; ON CONFLICT makes repeating it harmless
            ok = repo.upsert_feedback(conn, message_id, rating)
    except Exception:
        log.exception("Saving feedback failed")
        st.toast("Could not save your feedback. Please try again.", icon=":material/error:")
        return
    if ok:
        st.toast("Thanks for your feedback!", icon=":material/thumb_up:" if rating > 0 else ":material/thumb_down:")
    else:
        st.toast("This message no longer exists, so the feedback was not saved.", icon=":material/error:")


def _assistant_actions(i: int, text: str, meta: dict, cid: str, *, is_last: bool) -> None:
    """Compact action row under an assistant message: copy · regenerate (last answer only) · 👍 👎.

    Widget keys carry the conversation id and message index so they never clash across
    conversations; the feedback key also carries the regeneration counter so a regenerated
    answer starts with a blank rating.
    """
    with st.container(horizontal=True, gap="xsmall", vertical_alignment="center"):
        _copy_button(text, key=f"copy_{cid}_{i}")
        if is_last:
            st.button(
                "", icon=":material/refresh:", type="tertiary", help="Regenerate response",
                key=f"regen_{cid}_{i}", on_click=_queue_regenerate, args=(i,),
            )
        fb_key = f"fb_{cid}_{i}_{meta['gen']}"
        st.feedback(
            "thumbs", key=fb_key, default=_rating_to_feedback(meta["rating"]),
            on_change=_save_feedback, args=(meta["id"], fb_key), disabled=meta["id"] is None,
        )


def _friendly_error(exc: BaseException) -> str:
    """Short, user-facing message; the traceback is already in the log."""
    if is_transient_error(exc):
        return "The database connection dropped and could not be re-established. Please try again."
    module = type(exc).__module__ or ""
    if module.startswith(("openai", "google", "httpx", "httpcore", "ollama")) or isinstance(exc, TimeoutError):
        return "The AI service didn't respond or rejected the request. Please try again in a moment."
    if isinstance(exc, RuntimeError) and "API_KEY" in str(exc):
        return "The chat model is not configured (missing API key). Check the server's .env."
    return "Something went wrong while answering. Please try again."


def _render_answer(text: str, sources: list[dict] | None) -> None:
    """The answer with its [n] markers turned into links to the cited pages (render time only)."""
    st.markdown(linkify_citations(text, sources))


# Long URLs and titles must wrap inside the Sources panel instead of pushing it wider than the
# chat column (phones). Scoped to the sources containers by key prefix.
_SOURCES_CSS = "<style>[class*='st-key-sources_'] p,[class*='st-key-sources_'] a{overflow-wrap:anywhere}</style>"


def _render_sources(
    sources: list[dict], *, key: str, question: str | None = None, show_scores: bool = False
) -> None:
    """Compact source cards: one per page, with the citation numbers, the page link and an excerpt.

    `key` identifies the message (widget keys inside must be unique per message). `show_scores`
    (the "Show retrieved chunks" debug toggle) adds the cosine similarity; regular users never see
    it. Old messages whose sources have no excerpt get a fallback label.
    """
    cards = group_sources(sources)
    cited = [c for c in cards if c["cited"]]
    others = [c for c in cards if not c["cited"]]
    terms = question_terms(question)
    with st.expander(f"Sources ({len(cited) or len(cards)})", expanded=bool(cited)):
        st.html(_SOURCES_CSS)
        with st.container(key=f"sources_{key}"):
            for group, heading in ((cited, "Cited"), (others, "Also retrieved")):
                if not group:
                    continue
                st.markdown(f"**{heading}**")
                images = [c for c in group if c["image_url"]]
                for card in (c for c in group if not c["image_url"]):
                    _source_card(card, terms, show_scores)
                if images:
                    cols = st.columns(min(len(images), 4))
                    for col, c in zip(cols * ((len(images) + 3) // 4), images):
                        with col:
                            st.image(c["image_url"], caption=f"{_numbers(c)} {c['title']}", width="stretch")
                            st.caption((f"[page]({c['url']})" if c["url"] else "Source unavailable")
                                       + (f" · similarity {c['score']:.2f}" if show_scores and c["score"] is not None else ""))
        if show_scores:
            st.caption("Similarity = cosine similarity between the question and the passage (0–1); it is not a confidence.")


def _numbers(card: dict) -> str:
    return "".join(f"[{n}]" for n in card["numbers"])


def _source_card(card: dict, terms: list[str], show_scores: bool) -> None:
    link = f"[{card['title']}]({card['url']})" if card["url"] else f"{card['title']} _(source unavailable)_"
    with st.container(horizontal=True, gap="small", vertical_alignment="center", wrap=True):
        st.markdown(f"**{_numbers(card)}** {link}", width="stretch")
        if card["snippets"]:
            with st.popover("Excerpt", type="tertiary", width="content", help="The retrieved text this source contributed"):
                for snippet in card["snippets"]:
                    st.markdown(highlight_terms(snippet, terms))
        else:
            st.caption("Excerpt not available")
    meta = display_url(card["url"])
    if show_scores and card["score"] is not None:
        meta += f" · similarity {card['score']:.2f}"
    if meta:
        st.caption(meta)


# --------------------------------------------------------- page: RAG debugger
# Admin-only diagnostics: run the chatbot's own retrieval for a test question and inspect what
# the model would receive. Everything lives in session_state; nothing is written to the database
# and the chatbot's saved settings are never touched (top_k here is a what-if).
RDBG_BOT_KEY = "rdbg_bot"
RDBG_RESULT_KEY = "rdbg_result"
RDBG_COMPARE_KEY = "rdbg_compare_result"
RDBG_ANSWER_KEY = "rdbg_answer"
RDBG_LONG_CHUNK = 600  # characters; longer chunk text goes into an expander


def page_rag_debugger() -> None:
    st.set_page_config(layout="wide")
    if not _require_admin():
        return
    if not _require_schema():
        return
    st.title("RAG Debugger", icon=":material/troubleshoot:")
    st.caption(
        "Run a chatbot's retrieval for a test question and inspect the exact chunks it would answer from. "
        "Nothing here is saved, and the chatbot's settings are not changed."
    )
    bots = [b for b in _chatbots() if b["website_ids"]]
    if not bots:
        st.info("Create a chatbot with at least one website first (Chatbots page).", icon=":material/info:")
        return
    by_id = {b["id"]: b for b in bots}
    # Outside the form so the saved top_k / model caption follows the selection immediately.
    bot_id = st.selectbox("Chatbot", list(by_id), format_func=lambda i: _bot_label(by_id[i]), key=RDBG_BOT_KEY)
    bot = by_id[bot_id]
    saved_k = int(bot.get("top_k") or settings.top_k)
    has_instructions = bool((bot.get("system_prompt") or "").strip())
    st.caption(
        f"Saved top_k: **{saved_k}** · score threshold: **{settings.score_threshold}** · LLM: {describe_llm(bot)} · "
        f"custom instructions: {'set' if has_instructions else 'none'} · "
        f"websites: {', '.join(str(w) for w in bot['website_ids'])}"
    )

    with st.form("rdbg_form"):
        question = st.text_area("Test question", key="rdbg_question", height=80,
                                placeholder="A question a visitor might ask, e.g. What are your opening hours?")
        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
        ks = sorted({*rag_debug.DEFAULT_KS, saved_k})
        k = c1.radio("Top-K for this test", ks, index=ks.index(saved_k), horizontal=True, key=f"rdbg_k_{bot_id}",
                     help="How many chunks to ask the database for. Affects this test only.")
        compare = c2.checkbox("Compare top_k values", key="rdbg_compare",
                              help=f"Also run {', '.join(map(str, rag_debug.DEFAULT_KS))} and show how the results change.")
        run = st.form_submit_button("Run retrieval test", type="primary", icon=":material/search:", key="rdbg_run")

    if run:
        q = (question or "").strip()
        if not q:
            st.warning("Type a question first.", icon=":material/warning:")
        else:
            st.session_state.pop(RDBG_ANSWER_KEY, None)  # an old answer must not sit under new evidence
            st.session_state.pop(RDBG_COMPARE_KEY, None)
            try:
                with st.spinner("Embedding the question and searching…", show_time=True):
                    st.session_state[RDBG_RESULT_KEY] = rag_debug.run_retrieval_test(bot_id, q, int(k))
                    if compare:
                        st.session_state[RDBG_COMPARE_KEY] = rag_debug.compare_top_k(bot_id, q, ks)
            except Exception as exc:
                log.exception("Retrieval test failed")
                st.session_state.pop(RDBG_RESULT_KEY, None)
                st.error(f"Retrieval failed: {_friendly_error(exc)}", icon=":material/error:")

    test = st.session_state.get(RDBG_RESULT_KEY)
    if not test or test.chatbot_id != bot_id:
        return  # results belong to another chatbot (or none yet)
    _rdbg_header(test, bot)
    if test.skipped_reason:
        st.warning(test.skipped_reason, icon=":material/warning:")
        return
    st.subheader(f"Retrieved chunks ({len(test.chunks)})")
    st.caption("In the order the database returned them – the same order the model sees them numbered [1], [2], … "
               "Chunks below the threshold are shown here but dropped before the model is called.")
    if not test.chunks:
        st.info("No chunks came back: the attached websites have no embeddings of this kind, or the index is empty "
                "(see the Websites page).", icon=":material/info:")
    terms = question_terms(test.question)
    for chunk in test.chunks:
        _rdbg_chunk_card(chunk, terms)
    compare_tests = st.session_state.get(RDBG_COMPARE_KEY)
    if compare_tests:
        _rdbg_compare_table(compare_tests)
    _rdbg_answer_section(test, bot_id)


def _rdbg_header(test: "rag_debug.RetrievalTest", bot: dict) -> None:
    saved_k = int(bot.get("top_k") or settings.top_k)
    if test.k != saved_k:
        st.info(f"Tested with top_k = {test.k}; the chatbot's saved value is {saved_k}. The saved value is unchanged.",
                icon=":material/science:")
    if test.greeting:
        st.info("This looks like a bare greeting. The chatbot answers greetings with a canned welcome and does not "
                "retrieve; the search below is shown for information only.", icon=":material/waving_hand:")
    m = st.columns(6)
    m[0].metric("Chunks retrieved", len(test.chunks), border=True, help=f"Rows returned for top_k = {test.k}")
    m[1].metric("Kept for the model", test.kept_count, border=True,
                help=f"Chunks with score ≥ {test.threshold}; the rest are dropped by the chatbot")
    m[2].metric("Embedding", f"{test.embed_ms:.0f} ms", border=True, help="Time to embed the question (measured)")
    m[3].metric("Vector search", f"{test.search_ms:.0f} ms", border=True, help="PostgreSQL round trip(s) (measured)")
    m[4].metric("Total retrieval", f"{test.total_ms:.0f} ms", border=True)
    m[5].metric("Context", f"{test.context_chars:,} chars", border=True,
                help=f"≈ {test.est_tokens:,} tokens (chars ÷ {rag_debug.CHARS_PER_TOKEN}, a rough estimate, not the provider's count)")
    st.caption(
        f"Embedding model: {settings.embedding_provider} · {settings.resolved_embedding_model} "
        f"({settings.resolved_embedding_dim}d) · score: {rag_debug.metric_label(settings.resolved_embedding_dim)} · "
        f"threshold {test.threshold} · websites searched: {', '.join(map(str, test.website_ids)) or 'none'}"
        + (" · picture question: image chunks searched separately and listed first" if test.image_question else "")
    )


def _rdbg_chunk_card(chunk: "rag_debug.RetrievedChunk", terms: list[str]) -> None:
    na = rag_debug.NOT_AVAILABLE
    badge = ":green-badge[kept]" if chunk.kept else ":gray-badge[dropped – below threshold]"
    title = escape_markdown(chunk.title) if chunk.title else na
    link = f"[{title}]({chunk.url})" if chunk.url else f"{title} · URL {na}"
    with st.container(border=True):
        st.markdown(f"**[{chunk.rank}]** {badge} · score **{chunk.score:.4f}** · {link}")
        st.caption(
            f"chunk id: {chunk.chunk_id or na} · db id: {chunk.db_id if chunk.db_id is not None else na} · "
            f"website: {chunk.website_id if chunk.website_id is not None else na} · type: {chunk.kind} · "
            f"{chunk.chars:,} chars"
        )
        if chunk.kind == "image" and chunk.document.metadata.get("image_url"):
            st.image(chunk.document.metadata["image_url"], width=200)
        body = highlight_terms(chunk.text, terms) if chunk.text else f"_{na}_"
        if chunk.chars > RDBG_LONG_CHUNK:
            st.markdown(highlight_terms(chunk.text[:RDBG_LONG_CHUNK], terms) + " …")
            with st.expander("Full chunk text"):
                st.markdown(body)
        else:
            st.markdown(body)


def _rdbg_compare_table(tests: list) -> None:
    st.subheader("Top-K comparison")
    st.dataframe(rag_debug.compare_rows(tests), width="stretch", hide_index=True)
    st.caption(
        "One real search per top_k with the question embedded once. pgvector's HNSW scan is approximate, so the "
        "smaller result is not guaranteed to be a prefix of the larger one; hosted embedding APIs can also differ "
        "in the last decimals between calls."
    )


def _rdbg_answer_section(test: "rag_debug.RetrievalTest", bot_id: int) -> None:
    st.subheader("Answer accuracy")
    st.caption("Optional: generate an answer from the kept chunks with the chatbot's own model. "
               "One model call; nothing is saved to the conversation history.")
    if st.button("Generate test answer", key="rdbg_generate", icon=":material/smart_toy:"):
        try:
            chain, _ = _chain(bot_id, st.session_state.get("chain_version", 0), _rag_code_version())
            with st.spinner("Generating…", show_time=True):
                st.session_state[RDBG_ANSWER_KEY] = rag_debug.generate_test_answer(chain, test)
        except Exception as exc:
            log.exception("Test answer failed")
            st.error(f"Generation failed: {_friendly_error(exc)}", icon=":material/error:")
    answer = st.session_state.get(RDBG_ANSWER_KEY)
    if not answer:
        return
    left, right = st.columns(2)
    with left:
        st.markdown("#### Generated answer (LLM)")
        if answer.note:
            st.info(answer.note, icon=":material/info:")
        with st.container(border=True):
            _render_answer(answer.answer, [s.to_dict() for s in answer.sources])
        tokens = (f"{answer.model_tokens:,} tokens (model tokenizer)" if answer.model_tokens is not None
                  else f"≈ {answer.context_chars // rag_debug.CHARS_PER_TOKEN:,} tokens (estimate)")
        st.caption(
            f"grounded: {'yes' if answer.grounded else 'no'} · model called: {'yes' if answer.llm_called else 'no'} · "
            f"generation: {answer.gen_ms:.0f} ms · context: {answer.context_chars:,} chars, {tokens}"
        )
    with right:
        st.markdown("#### Retrieved evidence")
        kept = [c for c in test.chunks if c.kept]
        cited = rag_debug.cited_ranks(answer)
        if not kept:
            st.caption("No chunk passed the threshold, so the model received no evidence.")
        for n, chunk in enumerate(kept, start=1):
            mark = ":green-badge[cited in answer]" if n in cited else ":gray-badge[not cited]"
            title = escape_markdown(chunk.title) if chunk.title else rag_debug.NOT_AVAILABLE
            with st.container(border=True):
                st.markdown(f"**[{n}]** {mark} · score {chunk.score:.4f} · "
                            + (f"[{title}]({chunk.url})" if chunk.url else title))
                st.markdown(highlight_terms(chunk.text[:RDBG_LONG_CHUNK], question_terms(test.question))
                            + (" …" if chunk.chars > RDBG_LONG_CHUNK else ""))
    st.caption("A citation shows which passage the model pointed at. It does not make the answer correct, and a "
               "high score does not either: read the passage and compare it with the answer.")
    with st.expander("Exact context sent to the model", type="compact"):
        st.text(format_context(test.kept) if test.kept else "(empty)")


# ------------------------------------------------------------ page: RAG evaluation
# Test cases, runs and results live in the database (eval_* tables), so everything survives a
# refresh. A run is executed by a background thread (rag/evaluation.py); the page only starts it
# and polls the run row, so clicking around or reloading never interrupts a batch.
EVAL_BOT_KEY = "eval_bot"
EVAL_EDIT_KEY = "eval_edit_case"  # (chatbot_id, case_id) being edited, absent = new case
EVAL_FLASH_KEY = "eval_flash"
EVAL_POLL_SECONDS = 2
EVAL_STATUS_ICONS = {
    "passed": ":material/check_circle:", "failed": ":material/cancel:",
    "needs_review": ":material/rate_review:", "error": ":material/error:",
}


def page_rag_evaluation() -> None:
    st.set_page_config(layout="wide")
    if not _require_admin():
        return
    if not _require_schema():  # the eval_* tables are created by the schema upgrades
        return
    st.title("RAG Evaluation", icon=":material/fact_check:")
    st.caption(
        "Run saved test questions through a chatbot's real pipeline (same prompt, retrieval settings and model as "
        "the Chat page) and grade the answers. Test questions never appear in conversation history."
    )
    bots = [b for b in _chatbots() if b["website_ids"]]
    if not bots:
        st.info("Create a chatbot with at least one website first (Chatbots page).", icon=":material/info:")
        return
    by_id = {b["id"]: b for b in bots}
    bot_id = st.selectbox("Chatbot", list(by_id), format_func=lambda i: _bot_label(by_id[i]), key=EVAL_BOT_KEY)
    bot = by_id[bot_id]
    judge = rag_eval.judge_label()
    st.caption(
        f"Answers: {describe_llm(bot)} · top_k {int(bot.get('top_k') or settings.top_k)} · score threshold "
        f"{settings.score_threshold} · judge: {judge or 'disabled (every graded case needs review)'} · "
        f"rubric v{rag_eval.RUBRIC_VERSION}"
    )
    if flash := st.session_state.pop(EVAL_FLASH_KEY, None):
        kind, text = flash
        (st.success if kind == "success" else st.warning)(text, icon=":material/info:")
    dashboard, cases, results = st.tabs(["Dashboard", "Test cases", "Results"])
    with dashboard:
        _eval_dashboard(bot)
    with cases:
        _eval_cases(bot)
    with results:
        _eval_results(bot)


def _eval_tokens(run: dict) -> tuple[int | None, int | None]:
    if not run.get("results_with_usage"):
        return None, None
    tin = (run.get("answer_input_tokens") or 0) + (run.get("judge_input_tokens") or 0)
    tout = (run.get("answer_output_tokens") or 0) + (run.get("judge_output_tokens") or 0)
    return tin, tout


def _eval_cost_text(run: dict) -> str:
    tin, tout = _eval_tokens(run)
    cost = rag_eval.estimate_cost(tin, tout)
    return rag_eval.NOT_AVAILABLE if cost is None else f"≈ ${cost:,.4f}"


def _eval_dashboard(bot: dict) -> None:
    bid = bot["id"]
    counts = run_read(lambda conn: repo.count_eval_cases(conn, bid))
    runs = run_read(lambda conn: repo.list_eval_runs(conn, bid, 20))
    running = next((r for r in runs if r["status"] == "running"), None)
    latest = runs[0] if runs else None

    r1 = st.columns(4)
    r1[0].metric("Test cases", int(counts["total"] or 0), border=True,
                 help=f"{int(counts['enabled'] or 0)} enabled (included in runs)")
    r1[1].metric("Latest run", (latest["status"].title() if latest else rag_eval.NOT_AVAILABLE), border=True,
                 help=f"Run #{latest['id']} started {kb.fmt_time(latest['started_at'])}" if latest else "No run yet")
    r1[2].metric("Passed", latest["passed"] if latest else rag_eval.NOT_AVAILABLE, border=True)
    r1[3].metric("Failed", latest["failed"] if latest else rag_eval.NOT_AVAILABLE, border=True)
    r2 = st.columns(4)
    r2[0].metric("Needs review", latest["needs_review"] if latest else rag_eval.NOT_AVAILABLE, border=True,
                 help="Uncertain outcomes are left to a person instead of being passed or failed automatically")
    r2[1].metric("Errors", latest["errored"] if latest else rag_eval.NOT_AVAILABLE, border=True,
                 help="Cases whose pipeline raised (network, provider); not graded")
    r2[2].metric("Avg retrieval latency", eval_io.fmt_ms(latest.get("avg_prepare_ms")) if latest else rag_eval.NOT_AVAILABLE,
                 border=True, help="chain.prepare per case: embedding + vector search (+ question rewriting for "
                                   "follow-up cases). Median: " + (eval_io.fmt_ms(latest.get("median_prepare_ms")) if latest else "–"))
    r2[3].metric("Avg generation latency", eval_io.fmt_ms(latest.get("avg_generation_ms")) if latest else rag_eval.NOT_AVAILABLE,
                 border=True, help="One answer-model call per case (cases answered without a model call excluded). "
                                   "Median: " + (eval_io.fmt_ms(latest.get("median_generation_ms")) if latest else "–"))
    if latest:
        tin, tout = _eval_tokens(latest)
        tokens = f"{tin:,} in / {tout:,} out" if tin is not None else f"{rag_eval.NOT_AVAILABLE} (the provider reported no usage)"
        st.caption(f"Latest run tokens (answer + judge): {tokens} · estimated cost: {_eval_cost_text(latest)}"
                   + ("" if settings.eval_price_input_per_1m is not None and settings.eval_price_output_per_1m is not None
                      else " (set EVAL_PRICE_INPUT_PER_1M / EVAL_PRICE_OUTPUT_PER_1M to estimate)")
                   + f" · judge: {latest.get('judge_model') or 'disabled'} · rubric v{latest.get('rubric_version') or '?'}")
        if latest.get("error"):
            st.warning(latest["error"], icon=":material/warning:")

    st.subheader("Run evaluation")
    if running:
        _eval_progress(bid, running["id"])
    else:
        _eval_run_form(bot, int(counts["enabled"] or 0))

    st.subheader("Evaluation history")
    if not runs:
        st.caption("No evaluation has been run for this chatbot yet.")
        return
    st.dataframe(
        [
            {
                "run": r["id"], "started": kb.fmt_time(r["started_at"]), "status": r["status"],
                "cases": f"{r['done']}/{r['total_cases']}", "passed": r["passed"], "failed": r["failed"],
                "needs review": r["needs_review"], "errors": r["errored"],
                "pass rate": (f"{eval_io.pass_rate(r):.0%}" if eval_io.pass_rate(r) is not None else rag_eval.NOT_AVAILABLE),
                "avg retrieval": eval_io.fmt_ms(r.get("avg_prepare_ms")),
                "avg generation": eval_io.fmt_ms(r.get("avg_generation_ms")),
                "avg judge": eval_io.fmt_ms(r.get("avg_judge_ms")),
                "tokens in/out": (lambda t: f"{t[0]:,}/{t[1]:,}" if t[0] is not None else rag_eval.NOT_AVAILABLE)(_eval_tokens(r)),
                "est. cost": _eval_cost_text(r),
                "model": (r.get("config") or {}).get("llm_model") or rag_eval.NOT_AVAILABLE,
                "top_k": (r.get("config") or {}).get("top_k"),
                "judge": r.get("judge_model") or "disabled",
                "rubric": r.get("rubric_version") or "",
            }
            for r in runs
        ],
        width="stretch", hide_index=True,
    )
    rates = [eval_io.pass_rate(r) for r in reversed(runs) if r["status"] == "completed" and eval_io.pass_rate(r) is not None]
    if len(rates) > 1:
        st.caption("Pass-rate trend (oldest → newest completed runs): " + " → ".join(f"{p:.0%}" for p in rates[-10:]))
    completed = next((r for r in runs if r["status"] == "completed"), None)
    if completed:
        regressions = run_read(lambda conn: repo.eval_regressions(conn, bid, completed["id"]))
        if regressions:
            st.warning(
                f"{len(regressions)} case(s) passed in run #{regressions[0]['previous_run_id']} but not in run "
                f"#{completed['id']}: " + "; ".join(f"#{r['test_case_id']} {r['question'][:60]}" for r in regressions[:10]),
                icon=":material/trending_down:",
            )
        _eval_export_button(bid, completed["id"], key=f"eval_export_latest_{bid}",
                            label=f"Export results of run #{completed['id']} (CSV)")


def _eval_run_form(bot: dict, enabled: int) -> None:
    bid = bot["id"]
    with st.form(f"eval_run_form_{bid}"):
        cats = st.multiselect(
            "Categories to run", list(rag_eval.CATEGORIES), format_func=eval_io.category_label,
            key=f"eval_run_categories_{bid}", placeholder="All categories",
        )
        start = st.form_submit_button("Run evaluation", type="primary", icon=":material/play_arrow:",
                                      key=f"eval_run_{bid}", disabled=enabled == 0)
    calls = "1 answer call" + (" + 1 judge call" if settings.eval_judge_enabled else "")
    st.caption(
        f"{enabled} enabled case(s) · at most {settings.eval_max_batch} per run · one case at a time with a "
        f"{settings.eval_case_delay_seconds:g}s pause · per case: retrieval + {calls} (none for greetings or when "
        "nothing is retrieved). Only one run per chatbot at a time."
    )
    if enabled == 0:
        st.caption("Add test cases on the *Test cases* tab first.")
    if not start:
        return
    all_cases = run_read(lambda conn: repo.list_all_eval_cases(conn, bid))
    ids = [c["id"] for c in all_cases if c["enabled"] and (not cats or c["category"] in cats)]
    if not ids:
        st.warning("No enabled test case in the selected categories.", icon=":material/warning:")
        return
    try:
        chain, _ = _chain(bid, st.session_state.get("chain_version", 0), _rag_code_version())
        judge = rag_eval.build_judge()
        run_id = rag_eval.start_eval_run_in_background(bid, ids, chain=chain, judge=judge)
    except rag_eval.EvalRunInProgress:
        st.warning("An evaluation is already running for this chatbot.", icon=":material/hourglass_top:")
        return
    except rag_eval.NothingToEvaluate as exc:
        st.warning(str(exc), icon=":material/warning:")
        return
    except Exception as exc:
        log.exception("Starting the evaluation failed")
        st.error(f"Could not start the evaluation: {_friendly_error(exc)}", icon=":material/error:")
        return
    skipped = max(len(ids) - settings.eval_max_batch, 0)
    st.toast(f"Evaluation run #{run_id} started" + (f" (first {settings.eval_max_batch} cases)" if skipped else ""),
             icon=":material/play_arrow:")
    st.rerun()


@st.fragment(run_every=EVAL_POLL_SECONDS)
def _eval_progress(bot_id: int, run_id: int) -> None:
    """Re-reads the run row every few seconds while it is running; a full rerun when it ends."""
    run = run_read(lambda conn: repo.get_eval_run(conn, run_id, bot_id))
    if run is None:
        return
    if run["status"] != "running":
        text = (f"Run #{run_id} {run['status']}: {run['passed']} passed · {run['failed']} failed · "
                f"{run['needs_review']} need review · {run['errored']} error(s).")
        st.session_state[EVAL_FLASH_KEY] = ("success" if run["status"] == "completed" else "warning",
                                            text + (f" {run['error']}" if run.get("error") else ""))
        st.rerun()
    total, done = int(run["total_cases"] or 0), int(run["done"] or 0)
    with st.status(f"Evaluating run #{run_id}: {done} of {total} case(s) done", expanded=True, state="running"):
        st.progress(done / total if total else 0.0)
        st.caption(
            f"{run['passed']} passed · {run['failed']} failed · {run['needs_review']} need review · {run['errored']} error(s)"
            + (f" · last case: {run['last_question'][:100]}" if run.get("last_question") else "")
        )
        st.caption("The run continues on the server if you leave this page; every case is saved as soon as it is graded.")


def _eval_export_button(bot_id: int, run_id: int, *, key: str, label: str = "Export results (CSV)") -> None:
    results = run_read(lambda conn: repo.list_eval_results(conn, run_id, bot_id))
    st.download_button(label, data=eval_io.results_to_csv(results), file_name=f"rag-eval-chatbot{bot_id}-run{run_id}.csv",
                       mime="text/csv", icon=":material/download:", key=key, disabled=not results)


# ------------------------------------------------------------ evaluation: test cases
def _eval_cases(bot: dict) -> None:
    bid = bot["id"]
    edit = st.session_state.get(EVAL_EDIT_KEY)
    editing = None
    if edit and edit[0] == bid:
        editing = run_read(lambda conn: repo.get_eval_case(conn, edit[1], bid))
        if editing is None:
            st.session_state.pop(EVAL_EDIT_KEY, None)
    _eval_case_form(bid, editing)
    st.divider()
    _eval_case_list(bid)
    st.divider()
    _eval_import_export(bid)


def _eval_case_form(bid: int, editing: dict | None) -> None:
    sfx = f"{bid}_{editing['id']}" if editing else f"{bid}_new"
    e = editing or {}
    st.subheader(f"Edit test case #{e['id']}" if editing else "New test case")
    st.caption("Write the expected answer yourself from the website (or copy it from an indexed page). It must be "
               "verified by a person: the evaluator compares against it and never invents one.")
    keys = list(rag_eval.CATEGORIES)
    with st.form(f"eval_case_form_{sfx}", clear_on_submit=not editing):
        question = st.text_area("Test question", value=e.get("question", ""), key=f"eval_q_{sfx}", height=70)
        expected = st.text_area("Verified expected answer", value=e.get("expected_answer", ""), key=f"eval_a_{sfx}",
                                height=100, help="Leave empty only for questions the website cannot answer.")
        c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
        category = c1.selectbox("Category", keys, index=keys.index(e.get("category", "factual")),
                                format_func=eval_io.category_label, key=f"eval_cat_{sfx}")
        answerable = c2.checkbox("Answerable from the website", value=e.get("answerable", True), key=f"eval_ans_{sfx}",
                                 help="Unticked: the correct behaviour is to decline. Always off for 'Unanswerable Questions'.")
        enabled = c3.checkbox("Include in runs", value=e.get("enabled", True), key=f"eval_en_{sfx}")
        c4, c5 = st.columns(2)
        urls = c4.text_area("Supporting source URLs (optional, one per line)", key=f"eval_urls_{sfx}",
                            value="\n".join(e.get("source_urls") or []), height=80,
                            help="Pages that contain the answer. Recall@K is computed only for cases with these labels.")
        facts = c5.text_area("Key facts that must appear (optional, one per line)", key=f"eval_facts_{sfx}",
                             value="\n".join(e.get("key_facts") or []), height=80,
                             help="Exact names, numbers, prices or dates. A missing key fact fails the case.")
        turns = st.text_area("Previous turns (follow-up questions only)", key=f"eval_turns_{sfx}",
                             value=eval_io.turns_to_lines(e.get("previous_turns")), height=70,
                             placeholder="Q: What services do you offer?\nA: We offer web development [1].")
        notes = st.text_input("Notes (optional)", value=e.get("notes") or "", key=f"eval_notes_{sfx}")
        save = st.form_submit_button("Save changes" if editing else "Create test case", type="primary",
                                     icon=":material/save:", key=f"eval_save_{sfx}")
    if editing and st.button("Cancel editing", key=f"eval_cancel_{sfx}", icon=":material/close:"):
        st.session_state.pop(EVAL_EDIT_KEY, None)
        st.rerun()
    if not save:
        return
    try:
        fields = {
            "question": question.strip(), "expected_answer": expected.strip(), "category": category,
            "answerable": False if category == "unanswerable" else bool(answerable), "enabled": bool(enabled),
            "source_urls": eval_io.parse_list(urls), "key_facts": eval_io.parse_list(facts),
            "previous_turns": eval_io.parse_turn_lines(turns), "notes": notes.strip() or None,
        }
    except ValueError as exc:
        st.error(f"Could not read the form: {exc}", icon=":material/error:")
        return
    problems = eval_io.validate_case(fields)
    if problems:
        st.error("Not saved: " + "; ".join(problems) + ".", icon=":material/error:")
        return
    try:
        with get_conn() as conn:
            if editing:
                saved = repo.update_eval_case(conn, editing["id"], bid, **fields)
            else:
                saved = repo.create_eval_case(conn, bid, **fields)
    except psycopg.errors.UniqueViolation:
        st.error("This chatbot already has a test case with that question.", icon=":material/error:")
        return
    except ValueError as exc:
        st.error(f"Not saved: {exc}", icon=":material/error:")
        return
    if saved is None:
        st.error("That test case no longer exists.", icon=":material/error:")
        return
    st.session_state.pop(EVAL_EDIT_KEY, None)
    st.toast(f"{'Updated' if editing else 'Created'} test case #{saved['id']}", icon=":material/check:")
    st.rerun()


def _eval_case_list(bid: int) -> None:
    st.subheader("Saved test cases")
    f1, f2, f3 = st.columns([3, 2, 2], vertical_alignment="bottom")
    search = kb.normalise_search(f1.text_input("Search", key=f"eval_search_{bid}", placeholder="question or answer contains…"))
    cat_opts = [None, *rag_eval.CATEGORIES]
    category = f2.selectbox("Category", cat_opts, format_func=lambda k: "All categories" if k is None else eval_io.category_label(k),
                            key=f"eval_filter_cat_{bid}")
    ans_opts = {None: "Answerable or not", True: "Answerable", False: "Not answerable"}
    answerable = f3.selectbox("Answerable", list(ans_opts), format_func=ans_opts.get, key=f"eval_filter_ans_{bid}")
    size = repo.EVAL_LIST_SIZE
    page_key, last_key = f"eval_page_{bid}", f"eval_last_filter_{bid}"
    flt = (search, category, answerable)
    if st.session_state.get(last_key) != flt:
        st.session_state[last_key] = flt
        st.session_state.pop(page_key, None)

    def _read(offset: int):
        return run_read(lambda conn: repo.list_eval_cases(conn, bid, search=search, category=category,
                                                          answerable=answerable, limit=size, offset=offset))

    rows, total = _read(0)
    _, n_pages, _ = kb.page_window(total, 1, size)
    page_no = 1
    if n_pages > 1:
        page_no = int(st.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1, key=page_key))
        if page_no > 1:
            rows, total = _read(kb.page_window(total, page_no, size)[0])
    if not rows:
        st.caption("No test case matches these filters." if (search or category or answerable is not None)
                   else "No test cases yet. Create one above or import a CSV below.")
        return
    st.caption(kb.page_window(total, page_no, size)[2].replace("No pages", "No test cases"))
    for c in rows:
        with st.container(border=True):
            left, b1, b2, b3 = st.columns([7, 1.2, 1.4, 1.2], vertical_alignment="center")
            flags = [eval_io.category_label(c["category"]), "answerable" if c["answerable"] else "not answerable"]
            if not c["enabled"]:
                flags.append("excluded from runs")
            if c.get("key_facts"):
                flags.append(f"{len(c['key_facts'])} key fact(s)")
            if c.get("source_urls"):
                flags.append(f"{len(c['source_urls'])} source URL(s)")
            if c.get("previous_turns"):
                flags.append(f"{len(c['previous_turns'])} previous turn(s)")
            left.markdown(f"**#{c['id']}** {escape_markdown(c['question'])}")
            left.caption(" · ".join(flags) + f" · updated {kb.fmt_time(c['updated_at'])}")
            if b1.button("Edit", key=f"eval_edit_{c['id']}", icon=":material/edit:", width="stretch"):
                st.session_state[EVAL_EDIT_KEY] = (bid, c["id"])
                st.rerun()
            confirm = b2.checkbox("Confirm", key=f"eval_confirm_{c['id']}", help="Tick to enable Delete. Past results are kept.")
            if b3.button("Delete", key=f"eval_delete_{c['id']}", icon=":material/delete:", disabled=not confirm, width="stretch"):
                with get_conn() as conn:
                    deleted = repo.delete_eval_case(conn, c["id"], bid)
                if st.session_state.get(EVAL_EDIT_KEY) == (bid, c["id"]):
                    st.session_state.pop(EVAL_EDIT_KEY, None)
                st.toast(f"Deleted test case #{c['id']}" if deleted else "That test case was already deleted",
                         icon=":material/delete:")
                st.rerun()


def _eval_import_export(bid: int) -> None:
    st.subheader("Import / export")
    nonce_key = f"eval_upload_nonce_{bid}"
    nonce = st.session_state.get(nonce_key, 0)
    left, right = st.columns([3, 1])
    with left:
        upload = st.file_uploader(
            "Import test cases from CSV", type=["csv"], key=f"eval_upload_{bid}_{nonce}",
            help="Columns: " + ", ".join(eval_io.CSV_COLUMNS) + ". Existing questions are updated, new ones added.",
        )
    with right:
        cases = run_read(lambda conn: repo.list_all_eval_cases(conn, bid))
        st.download_button("Export test cases (CSV)", data=eval_io.cases_to_csv(cases),
                           file_name=f"rag-eval-cases-chatbot{bid}.csv", mime="text/csv", icon=":material/download:",
                           key=f"eval_export_cases_{bid}", disabled=not cases, width="stretch")
    st.caption("Lists (source_urls, key_facts) are JSON arrays or values separated by `|`; previous_turns is a JSON "
               'array of {"role": "user"|"assistant", "content": …}; answerable is yes/no; category is a key or a label. '
               "The export uses the same format.")
    if upload is None:
        return
    rows, errors = eval_io.parse_cases_csv(upload.getvalue())
    if errors:
        st.error(f"{len(errors)} problem(s) in the file:\n\n" + "\n".join(f"- {e}" for e in errors[:20]),
                 icon=":material/error:")
    if rows:
        st.dataframe(
            [{"question": r["question"], "category": eval_io.category_label(r["category"]),
              "answerable": r["answerable"], "expected answer": r["expected_answer"][:120],
              "key facts": len(r["key_facts"]), "source URLs": len(r["source_urls"])} for r in rows],
            width="stretch", hide_index=True,
        )
    valid_only = bool(errors) and st.checkbox("Import the valid rows only", key=f"eval_valid_only_{bid}_{nonce}")
    can_import = bool(rows) and (not errors or valid_only)
    if st.button(f"Import {len(rows)} test case(s)", key=f"eval_import_{bid}", type="primary", disabled=not can_import,
                 icon=":material/upload:"):
        try:
            with get_conn() as conn:
                n = repo.upsert_eval_cases(conn, bid, rows)
        except ValueError as exc:
            st.error(f"Nothing imported: {exc}", icon=":material/error:")
            return
        st.session_state[nonce_key] = nonce + 1  # clears the uploader
        st.session_state[EVAL_FLASH_KEY] = ("success", f"Imported {n} test case(s) (new questions added, existing ones updated).")
        st.rerun()


# ------------------------------------------------------------ evaluation: results
def _eval_results(bot: dict) -> None:
    bid = bot["id"]
    runs = run_read(lambda conn: repo.list_eval_runs(conn, bid, 50))
    if not runs:
        st.info("No results yet. Run an evaluation from the *Dashboard* tab.", icon=":material/info:")
        return
    by_id = {r["id"]: r for r in runs}
    run_id = st.selectbox(
        "Evaluation run", list(by_id), key=f"eval_run_select_{bid}",
        format_func=lambda i: (f"Run #{i} · {kb.fmt_time(by_id[i]['started_at'])} · {by_id[i]['status']} · "
                               f"{by_id[i]['passed']} passed / {by_id[i]['done']} done"),
    )
    run = by_id[run_id]
    cfg = run.get("config") or {}
    st.caption(
        f"Model {cfg.get('llm_provider', '?')} · {cfg.get('llm_model', '?')} (temperature {cfg.get('llm_temperature', '?')}) · "
        f"top_k {cfg.get('top_k', '?')} · threshold {cfg.get('score_threshold', '?')} · embeddings "
        f"{cfg.get('embedding_model', '?')} ({cfg.get('embedding_dim', '?')}d) · custom instructions "
        f"{'set' if cfg.get('custom_instructions_set') else 'none'} · judge {run.get('judge_model') or 'disabled'} · "
        f"rubric v{run.get('rubric_version') or '?'} ({run.get('judge_prompt_hash') or '?'})"
    )
    st.markdown(eval_io.RUN_STATUS_BADGES.get(run["status"], run["status"])
                + f" {eval_io.status_badge('passed')} {run['passed']} · {eval_io.status_badge('failed')} {run['failed']} · "
                + f"{eval_io.status_badge('needs_review')} {run['needs_review']} · {eval_io.status_badge('error')} {run['errored']}"
                + (f" · {run['total_cases'] - run['done']} not run" if run["status"] != "running" and run["done"] < run["total_cases"] else ""))
    if run["status"] == "running":
        st.info("This run is still in progress; results appear here as cases finish.", icon=":material/hourglass_top:")
    if run.get("error"):
        st.warning(run["error"], icon=":material/warning:")
    f1, f2, f3, f4 = st.columns([2, 2, 3, 2], vertical_alignment="bottom")
    st_opts = [None, *repo.EVAL_RESULT_STATUSES]
    status = f1.selectbox("Status", st_opts, key=f"eval_res_status_{bid}",
                          format_func=lambda s: "All statuses" if s is None else rag_eval.STATUS_LABELS[s])
    cat_opts = [None, *rag_eval.CATEGORIES]
    category = f2.selectbox("Category", cat_opts, key=f"eval_res_cat_{bid}",
                            format_func=lambda k: "All categories" if k is None else eval_io.category_label(k))
    search = kb.normalise_search(f3.text_input("Search", key=f"eval_res_search_{bid}", placeholder="question or answer contains…"))
    with f4:
        _eval_export_button(bid, run_id, key=f"eval_export_{bid}_{run_id}")
    results = run_read(lambda conn: repo.list_eval_results(conn, run_id, bid, status=status, category=category, search=search))
    if not results:
        st.caption("No result matches these filters." if (status or category or search) else "No results stored for this run yet.")
        return
    for r in results:
        _eval_result_card(r)


def _eval_result_card(r: dict) -> None:
    na = rag_eval.NOT_AVAILABLE
    label = f"{rag_eval.STATUS_LABELS.get(r['status'], r['status'])} · {eval_io.category_label(r['category'])} · {r['question'][:120]}"
    with st.expander(label, icon=EVAL_STATUS_ICONS.get(r["status"])):
        st.markdown(f"{eval_io.status_badge(r['status'])} · case "
                    + (f"#{r['test_case_id']}" if r.get("test_case_id") else "deleted")
                    + f" · {'answerable' if r['answerable'] else 'not answerable (should decline)'}")
        if r.get("failure_reason"):
            box = st.error if r["status"] in ("failed", "error") else st.warning
            box(r["failure_reason"], icon=":material/report:")
        if r.get("error"):
            st.caption(f"Error: {r['error']}")
        if r.get("test_case_id") is None:
            st.caption("The test case was deleted after this run; this result is kept exactly as it was evaluated.")
        elif r.get("case_current_updated_at") and r.get("case_updated_at") and r["case_current_updated_at"] != r["case_updated_at"]:
            st.caption("The test case was edited after this run. The expected answer below is the version that was used.")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Expected answer (verified)")
            with st.container(border=True):
                st.markdown(escape_markdown(r["expected_answer"]) if r["expected_answer"].strip() else f"_{'none – the chatbot should decline' if not r['answerable'] else na}_")
            if r.get("key_facts"):
                st.caption("Key facts: " + " · ".join(escape_markdown(f) for f in r["key_facts"]))
        with right:
            st.markdown("#### Generated answer (chatbot)")
            with st.container(border=True):
                if r.get("generated_answer"):
                    _render_answer(r["generated_answer"], r.get("retrieved") or [])
                else:
                    st.markdown(f"_{na}_")
            if r.get("standalone_question") and r["standalone_question"] != r["question"]:
                st.caption(f"Rewritten for retrieval as: {r['standalone_question']}")

        corr, faith = r.get("correctness") or {}, r.get("faithfulness") or {}
        retr, cites = r.get("retrieval") or {}, r.get("citations") or {}
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown("**Correctness**")
            st.caption((corr.get("verdict") or "not judged").replace("_", " ")
                       + (f" ({corr['confidence']} confidence)" if corr.get("confidence") else "")
                       + (f": {corr['reason']}" if corr.get("reason") else ""))
            if corr.get("key_facts_missing"):
                st.caption("Missing key facts: " + ", ".join(corr["key_facts_missing"]))
            if corr.get("auto_facts_missing"):
                st.caption("Values from the expected answer not found verbatim: " + ", ".join(corr["auto_facts_missing"]))
            if corr.get("missing_facts"):
                st.caption("Judge – missing: " + "; ".join(corr["missing_facts"]))
            if corr.get("contradictions"):
                st.caption("Judge – contradictions: " + "; ".join(corr["contradictions"]))
            if corr.get("judge_error"):
                st.caption(f"Judge unavailable: {corr['judge_error']}")
        with m2:
            st.markdown("**Faithfulness**")
            st.caption((faith.get("verdict") or "not judged").replace("_", " "))
            for claim in (faith.get("unsupported_claims") or [])[:5]:
                st.caption(f"Unsupported: {claim}")
        with m3:
            st.markdown("**Retrieval**")
            if retr.get("recall_at_k") is not None:
                st.caption(f"Recall@{retr.get('k')}: {retr['recall_at_k']:.2f} ({len(retr.get('matched_urls') or [])} of "
                           f"{len([u for u in retr.get('expected_urls') or [] if u.strip()])} labelled pages retrieved)")
            else:
                st.caption("No labelled source pages: Recall@K not computed.")
            st.caption(f"{retr.get('k', 0)} passage(s) retrieved"
                       + (f" · top score {retr['top_score']:.3f}" if retr.get("top_score") is not None else ""))
        with m4:
            st.markdown("**Citations**")
            if not cites.get("checked"):
                st.caption("No model answer to check.")
            else:
                st.caption(f"Cited: {' '.join(f'[{n}]' for n in cites.get('cited') or []) or 'none'}"
                           + (f" · invalid: {' '.join(f'[{n}]' for n in cites['invalid'])}" if cites.get("invalid") else " · all map to retrieved passages"))
                if cites.get("unsupported_by_passage"):
                    st.caption("Judge – passage does not support: " + " ".join(f"[{n}]" for n in cites["unsupported_by_passage"]))

        retrieved = r.get("retrieved") or []
        st.markdown(f"**Retrieved evidence ({len(retrieved)})** – untrusted website content, shown verbatim")
        for s in retrieved:
            title = escape_markdown(s.get("title") or "") or na
            link = f"[{title}]({s['url']})" if s.get("url") else title
            badge = ":green-badge[cited]" if s.get("cited") else ":gray-badge[not cited]"
            with st.container(border=True):
                st.markdown(f"**[{s.get('n')}]** {badge} · score {float(s.get('score') or 0):.4f} · {link}")
                if s.get("snippet"):
                    st.caption(escape_markdown(s["snippet"]))
        usage = r.get("usage") or {}
        tok = " · ".join(f"{k}: {v.get('input_tokens', '?')} in / {v.get('output_tokens', '?')} out" for k, v in usage.items() if v)
        st.caption(
            f"retrieval {eval_io.fmt_ms(r.get('prepare_ms'))} · generation {eval_io.fmt_ms(r.get('generation_ms'))} · "
            f"judge {eval_io.fmt_ms(r.get('judge_ms'))} · model called: {'yes' if r.get('model_called') else 'no'} · "
            f"tokens: {tok or na} · judge {r.get('judge_model') or 'disabled'} · rubric v{r.get('rubric_version') or '?'} · "
            f"evaluated {kb.fmt_time(r.get('created_at'))}"
        )
        st.json({"correctness": corr, "faithfulness": faith, "retrieval": retr, "citations": cites}, expanded=False)


# ---------------------------------------------------------------------- shell
def sidebar_footer() -> None:
    """Environment details in one popover (the Chat page shows them under Advanced settings)."""
    with st.sidebar:
        with st.popover("About this app", icon=":material/info:", width="stretch"):
            _about_details()
        if settings.admin_password and st.session_state.get(ADMIN_KEY):
            st.button("Log out of admin", icon=":material/logout:", type="tertiary", key="admin_logout", on_click=_logout)


pages = st.navigation(
    [
        st.Page(page_chat, title="Chat", icon=":material/chat:", default=True),
        st.Page(page_websites, title="Websites", icon=":material/language:"),
        st.Page(page_chatbots, title="Chatbots", icon=":material/smart_toy:"),
        st.Page(page_rag_debugger, title="RAG Debugger", icon=":material/troubleshoot:"),
        st.Page(page_rag_evaluation, title="RAG Evaluation", icon=":material/fact_check:"),
    ]
)
pages.run()
if pages.title != "Chat":
    sidebar_footer()
