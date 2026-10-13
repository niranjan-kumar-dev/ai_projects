"""Streamlit UI: index websites, create chatbots, chat.

Run from the project root:

    python -m streamlit run src/chat_with_website/ui/streamlit_app.py

Streamlit re-runs this whole file on every click.  Anything expensive (the
embedding model, the connection pool) is cached with `st.cache_resource`;
anything that must survive a rerun (chat history, selected chatbot) lives in
`st.session_state`.
"""

from __future__ import annotations

import logging
import sys
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
from chat_with_website.rag.citations import (
    display_url,
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
    sites = _websites()
    if not sites:
        st.info("No websites indexed yet. Add one above.")
        return

    st.subheader("Indexed websites")
    st.dataframe(
        [
            {
                "id": w["id"], "name": w["name"], "status": w["status"], "pages": w["page_count"],
                "chunks": w["chunk_count"], "embedding model": w["embedding_model"],
                "last crawled": w["last_crawled_at"].strftime("%Y-%m-%d %H:%M") if w["last_crawled_at"] else "",
                "url": w["base_url"],
            }
            for w in sites
        ],
        width="stretch",
        hide_index=True,
    )

    for w in sites:
        with st.expander(f"[{w['id']}] {w['name']} — {w['base_url']}"):
            if w["error"]:
                st.warning(f"Last error: {w['error']}")
            pages = run_read(lambda conn: repo.list_pages(conn, w["id"]))
            st.write(f"{len(pages)} page(s):")
            st.dataframe(
                [{"url": p["url"], "title": p["title"], "words": p["word_count"]} for p in pages],
                width="stretch", hide_index=True,
            )
            b1, b2, b3 = st.columns([1, 1, 4])
            if b1.button("Re-index", key=f"reindex_{w['id']}"):
                _run_ingestion(w["base_url"], w["name"], w["max_pages"])
            confirm = b3.checkbox("I understand this deletes all its pages and chunks", key=f"confirm_{w['id']}")
            if b2.button("Delete", key=f"delete_{w['id']}", type="secondary", disabled=not confirm):
                with get_conn() as conn:
                    repo.delete_website(conn, w["id"])
                _bump_chain_version()
                st.success(f"Deleted website {w['id']}")
                st.rerun()


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
        st.success(
            f"Crawled {result.pages_crawled} page(s) · indexed {result.pages_indexed} · "
            f"unchanged {result.pages_skipped_unchanged} · too short {result.pages_skipped_short} · "
            f"{result.chunks_created} chunks"
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

    st.divider()
    bots = _chatbots()
    if bots:
        st.subheader("Existing chatbots")
        for b in bots:
            c1, c2, c3 = st.columns([5, 1, 1])
            c1.markdown(
                f"**[{b['id']}] {b['name']}** — {describe_llm(b)} · top_k {b['top_k']}  \n"
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


# ----------------------------------------------------------------- page: chat
SUGGESTIONS = [
    ":material/help: What is this website about?",
    ":material/image: Show me the logo",
    ":material/mail: How can I contact you?",
]
USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/smart_toy:"


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
def _new_meta(message_id: int | None, rating: int | None = None, gen: int = 0) -> dict:
    return {"id": message_id, "rating": rating, "gen": gen}


def _meta_from_row(r: dict) -> dict:
    if r["role"] != "assistant":
        return {"id": r.get("id")}
    return _new_meta(r.get("id"), r.get("rating"), r.get("regeneration_count") or 0)


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


def _append_turn(question: str, answer: str, sources: list[dict], ids) -> None:
    """`ids` is the `TurnIds` returned by `persist_turn`, or None when saving failed."""
    st.session_state["history"].extend([HumanMessage(content=question), AIMessage(content=answer)])
    st.session_state["sources"].extend([None, sources])
    st.session_state["message_meta"].extend(
        [{"id": ids.user_id if ids else None}, _new_meta(ids.assistant_id if ids else None)]
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
    st.space("large")
    with st.container(horizontal_alignment="center"):
        st.markdown(":material/forum:", text_alignment="center")
        st.subheader(f"Ask anything about {bot['name']}", text_alignment="center")
        st.caption(
            bot.get("description") or "Answers come only from the website content indexed for this chatbot.",
            text_alignment="center",
        )
        choice = st.pills("Try asking", SUGGESTIONS, label_visibility="collapsed", key="chat_suggestion")
    if not choice:
        return None
    return choice.split(" ", 1)[1]  # drop the ":material/...:" icon prefix


def page_chat() -> None:
    try:
        _schema_ready()
    except Exception as exc:
        log.exception("Applying schema upgrades failed")
        st.error(
            "The database schema could not be updated. Run `python scripts/init_db.py` against this "
            f"database and reload. ({type(exc).__name__})", icon=":material/error:",
        )
        return
    bots = _chatbots()
    bots = [b for b in bots if b["website_ids"]]
    if not bots:
        st.title("Chat", icon=":material/chat:")
        st.info("Create a chatbot with at least one website first (Chatbots page).", icon=":material/info:")
        return

    with st.sidebar:
        labels = {b["id"]: f"{b['name'].title()}" for b in bots}
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
    if history:
        st.subheader(bot["name"])
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
    question = st.chat_input(f"Message {bot['name']}…", submit_mode="disable", disabled=regen_index is not None)
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
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            if i == regen_index:
                _regenerate(chatbot_id, i, msg.content, meta, cid, show_debug)  # reruns on success
                continue
            _render_answer(msg.content, src)
            _assistant_actions(i, msg.content, meta, cid, is_last=_is_last_assistant(history, i))
            if src:
                asked = history[i - 1].content if i >= 1 and isinstance(history[i - 1], HumanMessage) else None
                _render_sources(src, key=f"{cid}_{i}", question=asked, show_scores=show_debug)
            _render_debug(i, show_debug)

    # A question whose answer failed is not in history; keep showing it (with its error, Retry and
    # pencil) until it is retried, edited or replaced, so a rerun does not make it vanish.
    failed = st.session_state.get(FAILED_TURN_KEY)
    if not question and failed:
        _render_user_message(failed["question"], index=len(history), cid=cid)
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
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
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        generated = _generate_answer(
            chatbot_id, question, history,
            retry_cb=_queue_retry, retry_args=(question,), retry_key="retry_turn", show_debug=show_debug,
        )
        if generated is None:  # nothing persisted, history unchanged; remember the question for the next run
            st.session_state[FAILED_TURN_KEY] = {"question": question, "message": st.session_state.get(LAST_ERROR_KEY, "")}
            return
        result, src = generated

        # Phase 3: one write, only after a complete answer. An edited question also records which
        # message it replaced (the branch itself was created when "Save & Submit" was clicked).
        ids = None
        try:
            ids = persist_turn(
                cid, chatbot_id, question, result,
                edited_from_message_id=st.session_state.get(PENDING_EDITED_FROM_KEY),
            )
            st.session_state.pop(PENDING_EDITED_FROM_KEY, None)  # consumed; a Retry keeps it until saved
        except Exception:
            log.exception("Saving the conversation failed")
            st.session_state[FLASH_WARNING_KEY] = (  # shown by the run after the rerun below
                "The answer could not be saved to your conversation history. It will disappear on reload."
            )

    _remember_debug(len(history) + 1, result, show_debug)
    _append_turn(question, result.answer, src, ids)
    _bump_conversations()  # a new or updated conversation must show up in the sidebar
    # Redraw from state: the empty-state header/chips disappear, the new turn gets its pencil and
    # answer actions from the normal loop, and the sidebar shows the conversation right away.
    st.rerun()


def _is_last_assistant(history: list, i: int) -> bool:
    return 1 <= i == len(history) - 1 and isinstance(history[i], AIMessage)


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

    return result, [s.to_dict() for s in result.sources]


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
    result, src = generated

    if meta["id"] is not None:
        try:
            with get_conn() as conn:  # a write: not retried (same policy as persist_turn)
                repo.update_message(conn, meta["id"], result.answer, src)
        except Exception:
            log.exception("Saving the regenerated answer failed")
            st.session_state[FLASH_WARNING_KEY] = (
                "The regenerated answer could not be saved to your conversation history. "
                "It will disappear on reload."
            )
    history[i] = AIMessage(content=result.answer)
    st.session_state["sources"][i] = src
    st.session_state["message_meta"][i] = _new_meta(meta["id"], rating=None, gen=meta["gen"] + 1)
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
