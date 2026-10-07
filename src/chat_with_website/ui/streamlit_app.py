"""Streamlit UI: index websites, create chatbots, chat.

Run from the project root:

    python -m streamlit run src/chat_with_website/ui/streamlit_app.py

Streamlit re-runs this whole file on every click.  Anything expensive (the
embedding model, the connection pool) is cached with `st.cache_resource`;
anything that must survive a rerun (chat history, selected chatbot) lives in
`st.session_state`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow running without `pip install -e .`

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from chat_with_website.config import EMBEDDING_DEFAULTS, LLM_DEFAULTS, settings
from chat_with_website.db import repository as repo
from chat_with_website.db.connection import check_connection, get_conn
from chat_with_website.ingestion.embedder import get_embeddings
from chat_with_website.ingestion.pipeline import ingest_website
from chat_with_website.ingestion.url_utils import validate_start_url
from chat_with_website.logging_config import setup_logging
from chat_with_website.rag.chain import (
    answer_question,
    build_chain_for_chatbot,
    history_from_rows,
    new_conversation_id,
)
from chat_with_website.rag.llm import PROVIDER_LABELS, describe_llm

setup_logging()
st.set_page_config(page_title="Chat with Website", page_icon="🌐", layout="wide")


# ----------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading embedding model…")
def _embeddings():
    return get_embeddings()


@st.cache_resource
def _db_health() -> dict:
    return check_connection()


@st.cache_resource(show_spinner=False)
def _chain(chatbot_id: int, _version: int):
    """Cached per chatbot; `_version` is bumped when the chatbot is edited."""
    return build_chain_for_chatbot(chatbot_id)


def _bump_chain_version() -> None:
    st.session_state["chain_version"] = st.session_state.get("chain_version", 0) + 1
    _chain.clear()


def _websites() -> list[dict]:
    with get_conn() as conn:
        return repo.list_websites(conn)


def _chatbots() -> list[dict]:
    with get_conn() as conn:
        return repo.list_chatbots(conn)


# ------------------------------------------------------------- page: websites
def page_websites() -> None:
    st.title("🌐 Websites")
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
        use_container_width=True,
        hide_index=True,
    )

    for w in sites:
        with st.expander(f"[{w['id']}] {w['name']} — {w['base_url']}"):
            if w["error"]:
                st.warning(f"Last error: {w['error']}")
            with get_conn() as conn:
                pages = repo.list_pages(conn, w["id"])
            st.write(f"{len(pages)} page(s):")
            st.dataframe(
                [{"url": p["url"], "title": p["title"], "words": p["word_count"]} for p in pages],
                use_container_width=True, hide_index=True,
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
def page_chatbots() -> None:
    st.title("🤖 Chatbots")
    st.caption("A chatbot answers only from the websites attached to it.")

    sites = _websites()
    ready = [w for w in sites if w["status"] == "ready"]
    if not ready:
        st.info("Index at least one website first (Websites page).")
        return
    site_labels = {w["id"]: f"[{w['id']}] {w['name']} ({w['chunk_count']} chunks)" for w in ready}

    bots = _chatbots()
    edit_options = {"➕ New chatbot": None} | {f"[{b['id']}] {b['name']}": b for b in bots}
    choice = st.selectbox("Create new or edit existing", list(edit_options.keys()))
    editing = edit_options[choice]

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
        model = c2.text_input("Model (blank = provider default)", value=(editing or {}).get("llm_model") or "")
        top_k = c3.number_input("Chunks to retrieve (top_k)", 1, 20, value=(editing or {}).get("top_k") or settings.top_k)
        c1.caption("Defaults: " + ", ".join(f"{PROVIDER_LABELS[p]} → {m}" for p, m in LLM_DEFAULTS.items()))
        saved = st.form_submit_button("Save chatbot", type="primary")

    if saved:
        if not name.strip():
            st.error("Name is required.")
        elif not website_ids:
            st.error("Attach at least one website.")
        else:
            try:
                with get_conn() as conn:
                    bot = repo.create_chatbot(
                        conn, name=name.strip(), description=description.strip() or None,
                        system_prompt=system_prompt.strip() or None,
                        llm_provider=provider if provider != settings.llm_provider or model else None,
                        llm_model=model.strip() or None, top_k=int(top_k),
                    )
                    repo.set_chatbot_websites(conn, bot["id"], website_ids)
                _bump_chain_version()
                st.success(f"Saved chatbot [{bot['id']}] {bot['name']} — {describe_llm(bot)}")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    st.divider()
    bots = _chatbots()
    if bots:
        st.subheader("Existing chatbots")
        for b in bots:
            c1, c2 = st.columns([6, 1])
            c1.markdown(
                f"**[{b['id']}] {b['name']}** — {describe_llm(b)} · top_k {b['top_k']}  \n"
                + (f"_{b['description']}_  \n" if b.get("description") else "")
                + "Websites: " + (", ".join(site_labels.get(w, f"[{w}] (not ready)") for w in b["website_ids"]) or "none")
            )
            if c2.button("Delete", key=f"del_bot_{b['id']}"):
                with get_conn() as conn:
                    repo.delete_chatbot(conn, b["id"])
                _bump_chain_version()
                st.rerun()


# ----------------------------------------------------------------- page: chat
def page_chat() -> None:
    bots = _chatbots()
    bots = [b for b in bots if b["website_ids"]]
    if not bots:
        st.title("💬 Chat")
        st.info("Create a chatbot with at least one website first (Chatbots page).")
        return

    with st.sidebar:
        st.header("Chat settings")
        labels = {b["id"]: f"[{b['id']}] {b['name']}" for b in bots}
        chatbot_id = st.selectbox("Chatbot", list(labels.keys()), format_func=lambda i: labels[i], key="chat_bot_select")
        bot = next(b for b in bots if b["id"] == chatbot_id)
        st.caption(describe_llm(bot))

        # reset the conversation when the chatbot changes
        if st.session_state.get("active_bot") != chatbot_id:
            st.session_state.update(active_bot=chatbot_id, conversation_id=new_conversation_id(), history=[], sources=[])

        if st.button("🆕 New conversation", use_container_width=True):
            st.session_state.update(conversation_id=new_conversation_id(), history=[], sources=[])
            st.rerun()

        with get_conn() as conn:
            previous = repo.list_conversations(conn, chatbot_id)
        previous = [c for c in previous if c["message_count"]]
        if previous:
            st.subheader("Previous conversations")
            for c in previous[:15]:
                label = f"{c['created_at'].strftime('%m-%d %H:%M')} · {c['title'] or 'untitled'}"
                if st.button(label, key=f"conv_{c['id']}", use_container_width=True):
                    with get_conn() as conn:
                        rows = repo.get_messages(conn, str(c["id"]))
                    st.session_state.update(
                        conversation_id=str(c["id"]), history=history_from_rows(rows),
                        sources=[(r["sources"] or []) if r["role"] == "assistant" else None for r in rows],
                    )
                    st.rerun()

        show_debug = st.toggle("Show retrieved chunks", value=False)

    st.title(f"💬 {bot['name']}")
    if bot.get("description"):
        st.caption(bot["description"])

    history: list = st.session_state.setdefault("history", [])
    sources_by_turn: list = st.session_state.setdefault("sources", [])

    for i, msg in enumerate(history):
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)
            if role == "assistant" and i < len(sources_by_turn) and sources_by_turn[i]:
                _render_sources(sources_by_turn[i])

    question = st.chat_input("Ask something about the website…")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the website and writing an answer…"):
            try:
                chain, _ = _chain(chatbot_id, st.session_state.get("chain_version", 0))
                result = answer_question(
                    chatbot_id, question, history, conversation_id=st.session_state["conversation_id"], chain=chain
                )
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                return
        st.markdown(result.answer)
        src = [s.to_dict() for s in result.sources]
        if src:
            _render_sources(src)
        if show_debug:
            with st.expander("Retrieved chunks (debug)"):
                st.write(f"Standalone question: *{result.standalone_question}*")
                for d in result.documents:
                    st.markdown(f"**[{d.metadata.get('chunk_id')}]** score {d.metadata.get('score')} — {d.metadata.get('url')}")
                    st.text(d.page_content[:700])

    history.extend([HumanMessage(content=question), AIMessage(content=result.answer)])
    sources_by_turn.extend([None, src])


def _render_sources(sources: list[dict]) -> None:
    cited = [s for s in sources if s.get("cited")]
    others = [s for s in sources if not s.get("cited")]
    seen: set[str] = set()
    with st.expander(f"Sources ({len(cited) or len(sources)})", expanded=bool(cited)):
        for group, title in ((cited, "Cited"), (others, "Also retrieved")):
            lines = []
            for s in group:
                if s["url"] in seen:
                    continue
                seen.add(s["url"])
                lines.append(f"- [{s['n']}] [{s['title']}]({s['url']}) · score {s['score']:.2f}")
            if lines:
                st.markdown(f"**{title}**  \n" + "\n".join(lines))


# ---------------------------------------------------------------------- shell
def sidebar_footer() -> None:
    with st.sidebar:
        st.divider()
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


pages = st.navigation(
    [
        st.Page(page_chat, title="Chat", icon="💬", default=True),
        st.Page(page_websites, title="Websites", icon="🌐"),
        st.Page(page_chatbots, title="Chatbots", icon="🤖"),
    ]
)
pages.run()
sidebar_footer()
