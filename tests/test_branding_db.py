"""End-to-end branding check against the real local database and the real Chat page.

Two temporary chatbots are branded through the repository, the Streamlit app is run with the
*real* database (only the LLM chain is faked, so no OpenAI call), and the welcome screen, chips,
logo, avatar and accent stylesheet are checked for each bot, for an unbranded bot, and after a
reset. Skipped when PostgreSQL is not reachable. Everything is deleted afterwards.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

import chat_with_website.rag.chain as chain_mod
from chat_with_website.config import settings
from chat_with_website.db import repository as repo
from chat_with_website.ui.branding import DEFAULT_STARTERS, starter_text

APP = Path(__file__).resolve().parents[1] / "src" / "chat_with_website" / "ui" / "streamlit_app.py"

pytestmark = pytest.mark.skipif(settings.app_env != "development", reason="only against the local database")


def _png(colour) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (120, 60), colour).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def db():
    try:
        from chat_with_website.db.connection import check_connection, get_conn

        check_connection()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"PostgreSQL not reachable: {exc}")
    return get_conn


@pytest.fixture
def two_branded_bots(db, monkeypatch):
    monkeypatch.setattr(settings, "admin_password", None)
    with db() as conn:
        site = conn.execute("SELECT id FROM websites WHERE status = 'ready' ORDER BY id LIMIT 1").fetchone()
        if not site:
            pytest.skip("no indexed website to attach the test chatbots to")
        conn.execute("DELETE FROM chatbots WHERE name LIKE 'zz-test %'")  # leftovers from an aborted run
        a = repo.create_chatbot(conn, name="zz-test GMR", description="Transcription you can trust.")
        b = repo.create_chatbot(conn, name="zz-test DictaAI", description="Dictate, we type.")
        for bot in (a, b):
            repo.set_chatbot_websites(conn, bot["id"], [site["id"]])
        repo.update_chatbot_branding(
            conn, a["id"], display_name="GMR Helper", welcome_heading="How can GMR help?", description="Transcription you can trust.",
            accent_color="#ff6600", input_placeholder="Ask GMR…",
            starter_questions=["What do you transcribe?", "How fast is turnaround?", "Do you offer captions?", "How do I upload?"],
        )
        repo.update_chatbot_branding(
            conn, b["id"], display_name="Dicta Assistant", welcome_heading="Welcome to DictaAI", description="Dictate, we type.",
            accent_color="#22c55e", input_placeholder="Ask Dicta…", starter_questions=["Which languages?", "Is there an app?"],
        )
        logo_a, logo_b = _png((255, 102, 0)), _png((34, 197, 94))
        repo.set_chatbot_logo(conn, a["id"], logo_a)
        repo.set_chatbot_logo(conn, b["id"], logo_b)
    try:
        yield a["id"], b["id"], logo_a, logo_b
    finally:
        with db() as conn:
            conn.execute("DELETE FROM chatbots WHERE name LIKE 'zz-test %'")


def _fake_chain(monkeypatch):
    from langchain_core.documents import Document

    from chat_with_website.rag.chain import Prepared, RagAnswer, Source

    class FakeChain:
        def prepare(self, question, history=()):
            return Prepared(question=question, history=list(history), standalone=question,
                            documents=[Document(page_content="x", metadata={"url": "https://a/", "title": "A", "score": 0.5})])

        def stream_answer(self, p):
            yield "Answer [1]."

        def finalize(self, p, raw):
            return RagAnswer(answer=raw, sources=[Source(n=1, url="https://a/", title="A", score=0.5, cited=True)])

        def suggest_followups(self, p, result, history=()):
            return []

    monkeypatch.setattr(chain_mod, "build_chain_for_chatbot", lambda chatbot_id: (FakeChain(), {}))


def _html(at) -> str:
    return " ".join(str(h.proto.body) for h in at.get("html"))


def _select(at, bot_id):
    at.selectbox(key="chat_bot_select").set_value(bot_id).run()
    assert not at.exception, at.exception
    return at


def test_two_chatbots_get_independent_branding_on_the_real_page(two_branded_bots, monkeypatch, db):
    a, b, logo_a, logo_b = two_branded_bots
    _fake_chain(monkeypatch)
    import streamlit as st

    st.cache_data.clear()
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, at.exception

    # --- chatbot A: its own heading, description, placeholder, chips, logo and accent
    _select(at, a)
    assert "How can GMR help?" in [s.value for s in at.subheader]
    assert "Transcription you can trust." in [c.value for c in at.caption]
    assert at.chat_input[0].placeholder == "Ask GMR…"
    assert list(at.pills(key="chat_suggestion").options) == ["What do you transcribe?", "How fast is turnaround?", "Do you offer captions?", "How do I upload?"]
    assert at.image, "welcome logo missing"
    assert "#ff6600" in _html(at) and "#22c55e" not in _html(at)
    assert "!important" in _html(at)
    # a starter chip goes through the normal pipeline and the answer uses the logo as avatar
    at.pills(key="chat_suggestion").select("How do I upload?").run()
    assert not at.exception, at.exception
    assert at.chat_message[0].markdown[0].value == "How do I upload?"
    assert at.chat_message[1].proto.avatar not in ("", ":material/smart_toy:")
    with db() as conn:  # the turn was persisted against chatbot A
        row = conn.execute("SELECT count(*) AS n FROM messages m JOIN conversations c ON c.id = m.conversation_id WHERE c.chatbot_id = %s", (a,)).fetchone()
    assert row["n"] == 2

    # --- chatbot B: everything swaps, history is separate (empty)
    _select(at, b)
    assert "Welcome to DictaAI" in [s.value for s in at.subheader]
    assert at.chat_input[0].placeholder == "Ask Dicta…"
    assert list(at.pills(key="chat_suggestion").options) == ["Which languages?", "Is there an app?"]
    assert "#22c55e" in _html(at) and "#ff6600" not in _html(at)
    assert at.session_state["history"] == [] and at.image

    # --- back to A: its conversation is listed, its look is back
    _select(at, a)
    assert "#ff6600" in _html(at) and at.chat_input[0].placeholder == "Ask GMR…"
    assert any(b_.label == "How do I upload?" for b_ in at.button if b_.key and b_.key.startswith("conv_"))

    # --- an unbranded chatbot: defaults and no accent stylesheet
    with db() as conn:
        plain = conn.execute("SELECT id FROM chatbots WHERE name NOT LIKE 'zz-test %' AND accent_color IS NULL ORDER BY id LIMIT 1").fetchone()
    if plain:
        _select(at, plain["id"])
        assert "stBaseButton-primary" not in _html(at)
        assert list(at.pills(key="chat_suggestion").options) == [starter_text(s) for s in DEFAULT_STARTERS]

    # --- restore defaults on A: logo gone, defaults back, no stale cache
    with db() as conn:
        assert repo.reset_chatbot_branding(conn, a)
    st.cache_data.clear()
    _select(at, a)
    assert "Ask anything about Zz-Test Gmr" in [s.value for s in at.subheader]
    assert "#ff6600" not in _html(at) and not at.image
    assert at.chat_input[0].placeholder == "Message Zz-Test Gmr…"
