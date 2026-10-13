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
from chat_with_website.rag.chain import (
    build_chain_for_chatbot,
    history_from_rows,
    new_conversation_id,
    persist_turn,
)
from chat_with_website.rag.llm import PROVIDER_LABELS, describe_llm
from chat_with_website.ui.conversations import filter_conversations, group_conversations
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
    st.info("This page changes the chatbot setup. Enter the admin password to continue.", icon=":material/lock:")
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


# The dashboard is one aggregate query per rerun (`list_websites_overview`) plus one paginated
# page query for the selected website. Nothing is cached on purpose: a Re-index / Delete must
# show up immediately, and "Refresh status" simply reruns the script, which re-reads the database.
KB_FILTER_KEY = "kb_chatbot_filter"
KB_SITE_KEY = "kb_site_select"


def _knowledge_base_overview() -> None:
    bots = _chatbots()
    head, tail = st.columns([5, 1], vertical_alignment="bottom")
    head.subheader("Knowledge base overview")
    if tail.button("Refresh status", icon=":material/refresh:", key="kb_refresh", width="stretch"):
        _db_health.clear()  # the click already reruns the script; also re-check the server
        st.toast("Status reloaded from the database", icon=":material/check:")

    bot_options: dict[int | None, str] = {None: "All chatbots"} | {b["id"]: _bot_label(b) for b in bots}
    chatbot_id = st.selectbox(
        "Show websites used by", list(bot_options), format_func=bot_options.get, key=KB_FILTER_KEY,
        help="Scope the overview to the websites attached to one chatbot.",
    )
    sites = run_read(lambda conn: repo.list_websites_overview(conn, chatbot_id))
    st.caption(f"Loaded from the database at {datetime.now():%H:%M:%S}. "
               "Click *Refresh status* to reload after an ingestion run elsewhere (CLI, another tab).")
    if not sites:
        if chatbot_id is None:
            st.info("No websites indexed yet. Add one above.")
        else:
            st.info("No websites are attached to this chatbot yet (Chatbots page).")
        return

    names = kb.chatbot_names_by_website(bots)
    st.dataframe(
        [
            {
                "id": w["id"],
                "name": w["name"],
                "status": kb.status_label(w["status"]),
                "pages indexed": f"{kb.fmt_count(w['pages_indexed'])} / {kb.fmt_count(w['pages_total'])}",
                "chunks": kb.fmt_count(w["chunks_total"]),
                "last crawled": kb.fmt_time(w["last_crawled_at"]),
                "last indexed": kb.fmt_time(w["last_indexed_at"]),
                "latest run": kb.run_summary_label(w),
                "used by": ", ".join(names.get(w["id"], [])) or "—",
                "url": w["base_url"],
            }
            for w in sites
        ],
        width="stretch",
        hide_index=True,
        column_config={"url": st.column_config.LinkColumn("url")},
    )

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
    label = kb.status_label(w["status"])
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
    elif w["error"]:
        st.warning(f"Last run finished with errors: {w['error']}", icon=":material/warning:")
    fresh = kb.freshness(w["last_indexed_at"], now, timedelta(days=settings.kb_stale_days))
    if fresh == "stale":
        st.warning(
            f"Indexed data may be outdated: last successful indexing was {kb.fmt_time(w['last_indexed_at'])}, "
            f"more than {settings.kb_stale_days} days ago. Re-index to pick up changes on the site.",
            icon=":material/history:",
        )
    drift = kb.embedding_drift(
        w, settings.embedding_provider, settings.resolved_embedding_model, settings.resolved_embedding_dim
    )
    if drift:
        st.warning(drift, icon=":material/sync_problem:")

    # -- headline metrics -------------------------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Pages indexed", kb.fmt_count(w["pages_indexed"]),
        help=f"Pages with at least one stored chunk. {kb.fmt_count(w['pages_total'])} page(s) are stored in total.",
    )
    m2.metric("Chunks stored", kb.fmt_count(w["chunks_total"]), help="Embedded text/image chunks in pgvector.")
    m3.metric(
        "Last successful crawl", kb.fmt_time(w["last_crawled_at"]),
        help="When the crawler last fetched the site's pages. A crawl can succeed while embedding fails.",
    )
    m4.metric(
        "Last successful indexing", kb.fmt_time(w["last_indexed_at"]),
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
            f"- Pages crawled: {kb.fmt_count(w['run_pages_crawled'])} · indexed: {kb.fmt_count(w['run_pages_indexed'])}\n"
            f"- Unchanged: {kb.fmt_count(w['run_pages_skipped_unchanged'])} · too short: {kb.fmt_count(w['run_pages_skipped_short'])}\n"
            f"- Chunks created: {kb.fmt_count(w['run_chunks_created'])}\n"
            f"- Embedding model: {w['run_embedding_model'] or kb.NOT_AVAILABLE}"
        )
        if w["run_error"] and w["run_status"] == "failed":
            st.caption(f"Run error: {w['run_error']}")
        failed = kb.failed_pages(w)
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
    failed_urls = {f["url"] for f in kb.failed_pages(w)}
    with st.expander(f"Indexed pages ({kb.fmt_count(w['pages_total'])})", icon=":material/list:"):
        _pages_table(w, failed_urls, show_chunks=False)

    # -- diagnostics (operational detail: admin only, and never shown on the Chat page) ---------
    with st.expander("Advanced diagnostics", icon=":material/troubleshoot:"):
        if not _is_admin():
            st.info("Diagnostics are available to administrators only.", icon=":material/lock:")
        else:
            _diagnostics(w, failed_urls, now)

    # -- manage --------------------------------------------------------------------------------
    st.markdown("**Manage**")
    b1, b2, b3 = st.columns([1, 1, 4])
    if b1.button("Re-index", key=f"reindex_{w['id']}", icon=":material/refresh:"):
        _run_ingestion(w["base_url"], w["name"], w["max_pages"])
    confirm = b3.checkbox("I understand this deletes all its pages and chunks", key=f"confirm_{w['id']}")
    if b2.button("Delete", key=f"delete_{w['id']}", type="secondary", disabled=not confirm, icon=":material/delete:"):
        with get_conn() as conn:
            repo.delete_website(conn, w["id"])
        _bump_chain_version()
        st.success(f"Deleted website {w['id']}")
        st.rerun()


def _pages_table(w: dict, failed_urls: set[str], *, show_chunks: bool) -> None:
    """Paginated page list: one LIMIT/OFFSET query per rerun, never the whole site."""
    size = repo.PAGE_LIST_SIZE
    _, n_pages, _ = kb.page_window(w["pages_total"], 1, size)
    page_no = 1
    if n_pages > 1:
        page_no = int(st.number_input(
            "Page", min_value=1, max_value=n_pages, value=1, step=1, key=f"kb_pages_{w['id']}_{'diag' if show_chunks else 'list'}",
            help=f"{size} pages per view",
        ))
    offset, _, caption = kb.page_window(w["pages_total"], page_no, size)
    rows, total = run_read(lambda conn: repo.list_pages_with_chunk_counts(conn, w["id"], limit=size, offset=offset))
    if not rows:
        st.caption("No pages stored for this website.")
        # failed *new* pages have no row at all: they only exist in the run record
        if failed_urls:
            st.caption(f"{len(failed_urls)} page(s) failed in the latest run (see *Failed pages* above).")
        return
    st.caption(kb.page_window(total, page_no, size)[2] if total else caption)
    table = []
    for r in rows:
        row = {
            "url": r["url"],
            "title": r["title"] or "",
            "words": r["word_count"],
            "status": kb.page_status(r["chunk_count"], r["url"], failed_urls),
        }
        if show_chunks:
            row["chunks"] = int(r["chunk_count"] or 0)
        row["indexed at"] = kb.fmt_time(r["indexed_at"])
        table.append(row)
    st.dataframe(table, width="stretch", hide_index=True, column_config={"url": st.column_config.LinkColumn("url")})


def _diagnostics(w: dict, failed_urls: set[str], now: datetime) -> None:
    st.caption("Read-only. Nothing here re-embeds or modifies the knowledge base.")
    d1, d2 = st.columns(2)
    d1.markdown(
        "**Embedding configuration of this website**\n"
        f"- Provider: {w['embedding_provider'] or kb.NOT_AVAILABLE}\n"
        f"- Model: {w['embedding_model'] or kb.NOT_AVAILABLE}\n"
        f"- Dimensions: {kb.fmt_count(w['embedding_dim'])}"
    )
    d2.markdown(
        "**App configuration now**\n"
        f"- Provider: {settings.embedding_provider}\n"
        f"- Model: {settings.resolved_embedding_model}\n"
        f"- Dimensions: {settings.resolved_embedding_dim}"
    )
    fresh = kb.freshness(w["last_indexed_at"], now, timedelta(days=settings.kb_stale_days))
    newest = kb.fmt_time(w["newest_chunk_at"])
    st.markdown(
        "**Freshness**\n"
        f"- Newest stored chunk: {newest}\n"
        f"- Last successful indexing: {kb.fmt_time(w['last_indexed_at'])} "
        f"({'older than ' + str(settings.kb_stale_days) + ' days' if fresh == 'stale' else fresh})"
    )

    st.markdown("**Pages without chunks**")
    missing = run_read(lambda conn: repo.list_pages_without_chunks(conn, w["id"]))
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

    st.markdown("**Chunks per page**")
    _pages_table(w, failed_urls, show_chunks=True)

    st.markdown("**Run history**")
    runs = run_read(lambda conn: repo.list_ingestion_runs(conn, w["id"], limit=10))
    if runs:
        st.dataframe(
            [
                {
                    "run": r["id"], "status": r["status"],
                    "started": kb.fmt_time(r["started_at"], "%Y-%m-%d %H:%M:%S"),
                    "finished": kb.fmt_time(r["finished_at"], "%Y-%m-%d %H:%M:%S"),
                    "crawled": r["pages_crawled"], "indexed": r["pages_indexed"],
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


def _run_ingestion(url: str, name: str | None, max_pages: int) -> None:
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
            return
        bar.progress(1.0)
        status.update(label="Indexing finished", state="complete")
        run = f" · run #{result.run_id}" if result.run_id else ""
        st.success(
            f"Crawled {result.pages_crawled} page(s) · indexed {result.pages_indexed} · "
            f"unchanged {result.pages_skipped_unchanged} · too short {result.pages_skipped_short} · "
            f"{result.chunks_created} chunks{run}"
        )
        if result.errors:
            st.warning("\n".join(result.errors[:10]))
    _bump_chain_version()


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
                st.toast(f"Deleted chatbot [{b['id']}] {b['name']}", icon="🗑️")
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
    ]
)
pages.run()
if pages.title != "Chat":
    sidebar_footer()
