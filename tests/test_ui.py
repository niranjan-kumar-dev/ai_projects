"""Headless UI tests with `st.testing.v1.AppTest` - no database, no LLM.

The app script imports `run_read`, `check_connection`, the repository module and
the chain helpers at run time, so patching those module attributes *before*
`.run()` is enough to feed it fake data.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import chat_with_website.db.connection as db_conn
import chat_with_website.db.repository as repo
import chat_with_website.rag.chain as chain_mod
from chat_with_website.rag.chain import RagAnswer, Source
from chat_with_website.ui.conversations import group_conversations

APP = Path(__file__).resolve().parents[1] / "src" / "chat_with_website" / "ui" / "streamlit_app.py"

BOTS = [
    {"id": 7, "name": "acme support", "description": "Ask about Acme", "website_ids": [1],
     "llm_provider": "openai", "llm_model": "gpt-4o-mini", "top_k": 5, "system_prompt": None},
]
NOW = datetime.now(timezone.utc)


def _conv(n: int, title: str, days_ago: int, bot_id: int = 7) -> dict:
    when = NOW - timedelta(days=days_ago)
    return {"id": f"{n:08d}-1111-1111-1111-111111111111", "title": title, "message_count": 2,
            "created_at": when, "updated_at": when, "chatbot_id": bot_id}


CONVERSATIONS = [
    _conv(1, "What services?", 0),
    _conv(2, "A very long conversation title that keeps going and going so it must be truncated", 1),
    _conv(3, "Pricing question", 5),
    _conv(4, "Old chat", 30),
]
CONV1 = CONVERSATIONS[0]["id"]
FAKE_ANSWER = "Hi! I'm the acme support assistant. Ask me anything about acme support and I'll answer from its website content."


@pytest.fixture
def fake_backend(monkeypatch):
    monkeypatch.setattr(db_conn, "run_read", lambda fn: fn(None))
    monkeypatch.setattr(db_conn, "check_connection", lambda: {"server_version": "17", "pgvector": "0.8.0"})
    monkeypatch.setattr(repo, "list_chatbots", lambda conn: list(BOTS))
    monkeypatch.setattr(repo, "list_websites", lambda conn: [])
    monkeypatch.setattr(repo, "list_conversations", lambda conn, chatbot_id: [c for c in CONVERSATIONS if c["chatbot_id"] == chatbot_id])
    monkeypatch.setattr(repo, "get_messages", lambda conn, cid, limit=None: [
        {"role": "user", "content": "What services?", "sources": None, "created_at": NOW},
        {"role": "assistant", "content": "Web dev [1].", "sources": [{"n": 1, "url": "https://acme.example/", "title": "Home", "score": 0.5, "cited": True}], "created_at": NOW},
    ])
    calls: dict[str, list] = {"rename": [], "delete": []}
    monkeypatch.setattr(repo, "rename_conversation", lambda conn, cid, title: calls["rename"].append((cid, title)))
    monkeypatch.setattr(repo, "delete_conversation", lambda conn, cid: calls["delete"].append(cid))

    class _Conn:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(db_conn, "get_conn", lambda: _Conn())
    monkeypatch.calls = calls
    monkeypatch.setattr(chain_mod, "build_chain_for_chatbot", lambda chatbot_id: (object(), BOTS[0]))
    monkeypatch.setattr(
        chain_mod,
        "answer_question",
        lambda chatbot_id, question, history=(), conversation_id=None, chain=None: RagAnswer(
            answer=FAKE_ANSWER,
            sources=[Source(n=1, url="https://acme.example/", title="Home", score=0.5, cited=True)],
            grounded=True,
        ),
    )
    return monkeypatch


def _run_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def test_chat_page_renders_empty_state(fake_backend):
    at = _run_app()
    assert at.selectbox(key="chat_bot_select").value == 7
    assert "Ask anything about acme support" in at.subheader[0].value
    assert at.pills(key="chat_suggestion")  # suggestion chips are shown on an empty chat
    assert not at.chat_message
    assert at.button(key=f"conv_{CONV1}")  # recent conversation listed
    assert at.session_state["history"] == []


def test_group_conversations_buckets_and_order():
    groups = group_conversations(CONVERSATIONS, now=NOW)
    assert [g for g, _ in groups] == ["Today", "Yesterday", "Previous 7 days", "Older"]
    assert [len(items) for _, items in groups] == [1, 1, 1, 1]
    assert group_conversations([CONVERSATIONS[0]], now=NOW) == [("Today", [CONVERSATIONS[0]])]
    assert group_conversations([], now=NOW) == []


def test_sidebar_groups_highlights_and_offers_actions(fake_backend):
    at = _run_app()
    captions = [c.value for c in at.caption]
    for g in ("Today", "Yesterday", "Previous 7 days", "Older"):
        assert g in captions
    # every conversation has a title button, a rename input and save/delete actions
    for c in CONVERSATIONS:
        assert at.button(key=f"conv_{c['id']}").value is False
        assert at.text_input(key=f"rename_{c['id']}").value == c["title"]
        assert at.button(key=f"save_{c['id']}")
        assert at.button(key=f"delete_{c['id']}")
    # load one -> it becomes the active (highlighted) conversation and history is restored
    at.button(key=f"conv_{CONV1}").click().run()
    assert not at.exception, at.exception
    assert at.session_state["conversation_id"] == CONV1
    assert len(at.session_state["history"]) == 2
    assert at.button(key=f"conv_{CONV1}").proto.type == "secondary"
    assert at.button(key=f"conv_{CONVERSATIONS[1]['id']}").proto.type == "tertiary"


def test_rename_conversation_calls_repository(fake_backend):
    at = _run_app()
    at.text_input(key=f"rename_{CONV1}").set_value("Services overview").run()
    at.button(key=f"save_{CONV1}").click().run()
    assert not at.exception, at.exception
    assert fake_backend.calls["rename"] == [(CONV1, "Services overview")]


def test_delete_conversation_resets_open_chat(fake_backend):
    at = _run_app()
    at.button(key=f"conv_{CONV1}").click().run()
    assert at.session_state["conversation_id"] == CONV1
    at.button(key=f"delete_{CONV1}").click().run()  # opens the confirmation dialog
    assert not at.exception, at.exception
    assert at.button(key="confirm_delete_ok") and at.button(key="confirm_delete_cancel")
    assert fake_backend.calls["delete"] == []  # nothing deleted until confirmed


def test_chat_page_answers_and_keeps_history(fake_backend):
    at = _run_app()
    at.chat_input[0].set_value("Hi").run()
    assert not at.exception, at.exception
    assert len(at.chat_message) == 2
    assert at.chat_message[0].markdown[0].value == "Hi"
    assert at.chat_message[1].markdown[0].value == FAKE_ANSWER
    assert len(at.session_state["history"]) == 2
    assert at.session_state["sources"][1][0]["url"] == "https://acme.example/"
    assert not at.pills  # chips disappear once the conversation has started


def test_suggestion_chip_becomes_first_question(fake_backend):
    at = _run_app()
    at.pills(key="chat_suggestion").select(":material/image: Show me the logo").run()
    assert not at.exception, at.exception
    assert at.chat_message[0].markdown[0].value == "Show me the logo"
    assert at.session_state["history"][0].content == "Show me the logo"


def test_chat_page_without_bots_shows_hint(fake_backend, monkeypatch):
    monkeypatch.setattr(repo, "list_chatbots", lambda conn: [])
    at = _run_app()
    assert at.info
    assert not at.chat_input
