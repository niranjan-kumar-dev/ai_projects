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
from langchain_core.documents import Document

from chat_with_website.rag.chain import Prepared, RagAnswer, Source, TurnIds
from chat_with_website.ui.conversations import filter_conversations, group_conversations

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


SAVED_RATING: dict = {"value": None}  # rating the fake `get_messages` reports for the assistant message
MESSAGE_ROWS: dict = {"rows": None}  # a test may replace the rows the fake `get_messages` returns
FU = ["How do you price it?", "Which stack do you use?"]  # what the FakeChain suggests
SOURCES_HOME = [{"n": 1, "url": "https://acme.example/", "title": "Home", "score": 0.5, "cited": True}]


def _row(mid: int, role: str, content: str, sources=None, rating=None) -> dict:
    return {"id": mid, "role": role, "content": content, "sources": sources, "created_at": NOW,
            "regeneration_count": 0, "rating": rating, "edited_from_message_id": None}


@pytest.fixture
def fake_backend(monkeypatch):
    calls: dict[str, list] = {"rename": [], "delete": [], "persist": [], "stream": [], "prepare": [],
                              "update": [], "feedback": [], "fork": [], "edited_from": [],
                              "update_suggestions": [], "followups": [], "persist_suggestions": []}
    SAVED_RATING["value"] = None
    MESSAGE_ROWS["rows"] = None
    monkeypatch.setattr(db_conn, "run_read", lambda fn: fn(None))
    monkeypatch.setattr(db_conn, "ensure_schema_upgrades", lambda: None)  # no DB in UI tests
    monkeypatch.setattr(db_conn, "check_connection", lambda: {"server_version": "17", "pgvector": "0.8.0"})
    monkeypatch.setattr(repo, "list_chatbots", lambda conn: list(BOTS))
    monkeypatch.setattr(repo, "list_websites", lambda conn: [])
    # knowledge-base status page (Websites): no sites by default
    monkeypatch.setattr(repo, "list_websites_overview", lambda conn, chatbot_id=None: [])
    monkeypatch.setattr(repo, "list_pages_with_chunk_counts", lambda conn, wid, search="", limit=100, offset=0: ([], 0))
    monkeypatch.setattr(repo, "list_duplicate_page_urls", lambda conn, wid, limit=100: [])
    monkeypatch.setattr(repo, "list_pages_without_chunks", lambda conn, wid, limit=100: [])
    monkeypatch.setattr(repo, "list_ingestion_runs", lambda conn, wid, limit=10: [])
    monkeypatch.setattr(repo, "list_conversations", lambda conn, chatbot_id: [c for c in CONVERSATIONS if c["chatbot_id"] == chatbot_id])
    monkeypatch.setattr(repo, "get_messages", lambda conn, cid, limit=None: MESSAGE_ROWS["rows"] or [
        _row(1, "user", "What services?"),
        _row(2, "assistant", "Web dev [1].", SOURCES_HOME, rating=SAVED_RATING["value"]),
    ])

    def fake_fork(conn, *, new_id, chatbot_id, source_id, forked_from_message_id, turns):
        calls["fork"].append({"new_id": new_id, "chatbot_id": chatbot_id, "source_id": source_id,
                              "forked_from_message_id": forked_from_message_id, "turns": list(turns)})
        return list(range(101, 101 + len(turns)))

    monkeypatch.setattr(repo, "fork_conversation", fake_fork)
    monkeypatch.setattr(repo, "rename_conversation", lambda conn, cid, title: calls["rename"].append((cid, title)))
    monkeypatch.setattr(repo, "delete_conversation", lambda conn, cid: calls["delete"].append(cid))

    def fake_update_message(conn, message_id, content, sources, suggestions=None):
        calls["update"].append((message_id, content, sources))
        calls["update_suggestions"].append(suggestions)
        return len(calls["update"])

    def fake_upsert_feedback(conn, message_id, rating):
        calls["feedback"].append((message_id, rating))
        return True

    monkeypatch.setattr(repo, "update_message", fake_update_message)
    monkeypatch.setattr(repo, "upsert_feedback", fake_upsert_feedback)

    class _Conn:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(db_conn, "get_conn", lambda: _Conn())
    monkeypatch.calls = calls
    import streamlit as st

    st.cache_data.clear()  # the sidebar caches the conversation list per chatbot
    st.cache_resource.clear()  # the chat page caches the chain per chatbot
    class FakeChain:
        """Mimics RagChain.prepare / stream_answer / finalize without DB or LLM."""

        fail_stream = False

        def prepare(self, question, history=()):
            calls["prepare"].append((question, [m.content for m in history]))
            if question.strip().lower() in ("hi", "hello"):
                return Prepared(question=question, history=[], standalone=question,
                                ready=RagAnswer(answer=FAKE_ANSWER, grounded=False))
            return Prepared(question=question, history=list(history), standalone=question,
                            documents=[Document(page_content="Web dev", metadata={"url": "https://acme.example/", "title": "Home", "score": 0.5})])

        def stream_answer(self, p):
            calls["stream"].append(p.question)
            for chunk in ["We build ", "**websites** ", "[1]."]:
                yield chunk
            if self.fail_stream:
                raise RuntimeError("boom from the model")

        fail_followups = False
        followups = ["How do you price it?", "Which stack do you use?"]

        def suggest_followups(self, p, result, history=()):
            calls["followups"].append((p.question, [m.content for m in history]))
            if self.fail_followups:
                raise RuntimeError("followups boom")
            return list(self.followups)

        def finalize(self, p, raw):
            return RagAnswer(answer=raw.strip(), sources=[Source(n=1, url="https://acme.example/", title="Home", score=0.5, cited=True)], grounded=True)

    fake_chain = FakeChain()
    monkeypatch.setattr(chain_mod, "build_chain_for_chatbot", lambda chatbot_id: (fake_chain, BOTS[0]))
    def fake_persist(cid, bot_id, q, result, edited_from_message_id=None, suggestions=None):
        calls["persist"].append((cid, q, result.answer))
        calls["edited_from"].append(edited_from_message_id)
        calls["persist_suggestions"].append(list(suggestions or []))
        return TurnIds(user_id=41, assistant_id=42)  # ids of the stored rows

    monkeypatch.setattr(chain_mod, "persist_turn", fake_persist)
    monkeypatch.fake_chain = fake_chain
    return monkeypatch


def _run_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def test_chat_page_renders_empty_state(fake_backend):
    at = _run_app()
    assert at.selectbox(key="chat_bot_select").value == 7
    assert "Ask anything about Acme Support" in at.subheader[0].value
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


def test_filter_conversations_is_case_insensitive_and_multi_term():
    assert [c["id"] for c in filter_conversations(CONVERSATIONS, "PRICING")] == [CONVERSATIONS[2]["id"]]
    assert [c["id"] for c in filter_conversations(CONVERSATIONS, "long title")] == [CONVERSATIONS[1]["id"]]
    assert filter_conversations(CONVERSATIONS, "") == CONVERSATIONS
    assert filter_conversations(CONVERSATIONS, "   ") == CONVERSATIONS
    assert filter_conversations(CONVERSATIONS, "nothing-like-this") == []
    assert filter_conversations([{"id": "x", "title": None}], "untitled") == [{"id": "x", "title": None}]


def test_search_filters_without_touching_selection_or_db(fake_backend, monkeypatch):
    db_calls = {"n": 0}
    orig = repo.list_conversations

    def counted(conn, chatbot_id):
        db_calls["n"] += 1
        return orig(conn, chatbot_id)

    monkeypatch.setattr(repo, "list_conversations", counted)
    at = _run_app()
    at.button(key=f"conv_{CONV1}").click().run()  # open a conversation first
    assert at.session_state["conversation_id"] == CONV1
    queries_before = db_calls["n"]

    at.text_input(key="conv_search").set_value("pricing").run()
    assert not at.exception, at.exception
    assert db_calls["n"] == queries_before  # typing reruns the script but not the query
    shown = [b.key for b in at.button if b.key and b.key.startswith("conv_") and b.key != "conv_search_clear"]
    assert shown == [f"conv_{CONVERSATIONS[2]['id']}"]
    assert at.button(key="conv_search_clear")  # clear button appears once there is a query
    assert "Results (1)" in [c.value for c in at.caption]
    assert "Today" not in [c.value for c in at.caption]  # flat results while searching
    assert at.session_state["conversation_id"] == CONV1  # selection untouched

    at.button(key=f"conv_{CONVERSATIONS[2]['id']}").click().run()  # a result opens that conversation
    assert at.session_state["conversation_id"] == CONVERSATIONS[2]["id"]

    at.text_input(key="conv_search").set_value("zzz").run()
    assert "No matching conversations found" in [c.value for c in at.caption]

    at.button(key="conv_search_clear").click().run()  # clear -> grouping is back
    assert at.text_input(key="conv_search").value == ""
    assert "Today" in [c.value for c in at.caption]
    assert len([b for b in at.button if b.key and b.key.startswith("conv_") and b.key != "conv_search_clear"]) == len(CONVERSATIONS)


def test_sidebar_groups_highlights_and_offers_actions(fake_backend):
    at = _run_app()
    captions = [c.value for c in at.caption]
    for g in ("Today", "Yesterday", "Previous 7 days", "Older"):
        assert g in captions
    # every conversation has a title button, a rename input and save/delete actions
    for c in CONVERSATIONS:
        assert at.button(key=f"conv_{c['id']}").value is False
        assert at.button(key=f"conv_{c['id']}").proto.help == ""  # no hover tooltip (it spilled over the chat)
        assert at.text_input(key=f"rename_{c['id']}").value == c["title"]
        assert at.button(key=f"save_{c['id']}")
        assert at.button(key=f"delete_{c['id']}")
    assert len([c for c in captions if c.startswith("Last activity:")]) == len(CONVERSATIONS)  # date moved into the popover
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


def test_greeting_is_answered_without_streaming(fake_backend):
    at = _run_app()
    at.chat_input[0].set_value("Hi").run()
    assert not at.exception, at.exception
    assert len(at.chat_message) == 2
    assert at.chat_message[0].markdown[0].value == "Hi"
    assert at.chat_message[1].markdown[0].value == FAKE_ANSWER
    assert len(at.session_state["history"]) == 2
    assert fake_backend.calls["stream"] == []  # no model call for a greeting
    assert [c[2] for c in fake_backend.calls["persist"]] == [FAKE_ANSWER]  # saved exactly once
    assert not at.pills  # chips disappear once the conversation has started


def test_streamed_answer_is_shown_saved_once_and_kept_in_history(fake_backend):
    at = _run_app()
    at.chat_input[0].set_value("What do you build?").run()
    assert not at.exception, at.exception
    assert fake_backend.calls["stream"] == ["What do you build?"]  # exactly one model call
    assert "We build **websites** [[1]](https://acme.example/)." in [m.value for m in at.chat_message[1].markdown]
    assert fake_backend.calls["persist"] == [(at.session_state["conversation_id"], "What do you build?", "We build **websites** [1].")]
    assert at.session_state["history"][1].content == "We build **websites** [1]."
    assert at.session_state["sources"][1][0]["url"] == "https://acme.example/"


def _texts(at) -> list[str]:
    return [m.value for m in at.markdown]


def test_no_loading_cue_is_left_after_a_streamed_answer(fake_backend):
    at = _run_app()
    at.chat_input[0].set_value("What do you build?").run()
    assert not at.exception, at.exception
    assert not any(":shimmer[" in t for t in _texts(at))  # indicator slot was cleared
    assert "We build **websites** [[1]](https://acme.example/)." in [m.value for m in at.chat_message[1].markdown]


def test_greeting_leaves_no_indicator_and_no_stream(fake_backend):
    at = _run_app()
    at.chat_input[0].set_value("hello").run()
    assert not at.exception, at.exception
    assert not any(":shimmer[" in t for t in _texts(at))
    assert fake_backend.calls["stream"] == []
    assert at.chat_message[1].markdown[0].value == FAKE_ANSWER


def test_stream_failure_shows_error_and_retry_then_succeeds(fake_backend):
    fake_backend.fake_chain.fail_stream = True
    at = _run_app()
    at.chat_input[0].set_value("What do you build?").run()
    assert not at.exception, at.exception
    assert at.error and "Something went wrong" in at.error[0].value
    assert "not saved" in at.error[0].value  # partial text was streamed, user is told it is discarded
    assert not any(":shimmer[" in t for t in _texts(at))
    assert at.button(key="retry_turn")
    assert fake_backend.calls["persist"] == []
    assert at.session_state["history"] == []

    fake_backend.fake_chain.fail_stream = False  # "the API is back"
    at.button(key="retry_turn").click().run()
    assert not at.exception, at.exception
    assert fake_backend.calls["stream"] == ["What do you build?", "What do you build?"]  # one new attempt
    assert [c[1] for c in fake_backend.calls["persist"]] == ["What do you build?"]  # saved once, on success
    assert at.session_state["history"][1].content == "We build **websites** [1]."
    assert not at.error  # error state is gone after a successful retry
    assert "retry_turn" not in [b.key for b in at.button]
    assert "pending_question" not in at.session_state  # the queued question was consumed


def test_persist_failure_warns_but_keeps_answer(fake_backend, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("db down")

    monkeypatch.setattr(chain_mod, "persist_turn", boom)
    at = _run_app()
    at.chat_input[0].set_value("What do you build?").run()
    assert not at.exception, at.exception
    assert at.warning and "could not be saved" in at.warning[0].value
    assert at.session_state["history"][1].content == "We build **websites** [1]."


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


# ------------------------------------------------- message actions (stage 09)
def _feedback(at, key):
    found = [f for f in at.get("feedback") if f.key == key]
    assert found, f"no feedback widget with key {key}; have {[f.key for f in at.get('feedback')]}"
    return found[0]


def test_loaded_conversation_shows_action_row_on_assistant_message_only(fake_backend):
    at = _run_app()
    at.button(key=f"conv_{CONV1}").click().run()
    assert not at.exception, at.exception
    # user turn (index 0) has no actions; the last assistant turn (index 1) has regenerate + feedback
    assert f"regen_{CONV1}_0" not in [b.key for b in at.button]
    assert at.button(key=f"regen_{CONV1}_1")
    fb = _feedback(at, f"fb_{CONV1}_1_0")
    assert fb.value is None and not fb.proto.disabled
    assert at.chat_message[1].markdown[0].value == "Web dev [[1]](https://acme.example/)."  # the answer stays the first markdown
    assert at.session_state["message_meta"] == [{"id": 1}, {"id": 2, "rating": None, "gen": 0, "suggestions": []}]


def test_fresh_answer_gets_action_row_with_saved_id(fake_backend):
    at = _run_app()
    at.chat_input[0].set_value("What do you build?").run()
    assert not at.exception, at.exception
    cid = at.session_state["conversation_id"]
    assert at.button(key=f"regen_{cid}_1")
    assert not _feedback(at, f"fb_{cid}_1_0").proto.disabled
    assert at.session_state["message_meta"] == [{"id": 41}, {"id": 42, "rating": None, "gen": 0, "suggestions": FU}]


def test_feedback_is_disabled_when_the_answer_was_not_saved(fake_backend, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("db down")

    monkeypatch.setattr(chain_mod, "persist_turn", boom)
    at = _run_app()
    at.chat_input[0].set_value("What do you build?").run()
    assert not at.exception, at.exception
    cid = at.session_state["conversation_id"]
    assert _feedback(at, f"fb_{cid}_1_0").proto.disabled
    assert at.button(key=f"regen_{cid}_1")  # regenerating still works without a DB row


def test_regenerate_replaces_last_answer_in_place(fake_backend):
    at = _run_app()
    at.button(key=f"conv_{CONV1}").click().run()
    at.button(key=f"regen_{CONV1}_1").click().run()
    assert not at.exception, at.exception
    # exactly one new model call, asked with the original question and the turns *before* it
    assert fake_backend.calls["stream"] == ["What services?"]
    assert fake_backend.calls["prepare"] == [("What services?", [])]
    assert fake_backend.calls["update"] == [(2, "We build **websites** [1].", [
        {"n": 1, "url": "https://acme.example/", "title": "Home", "score": 0.5, "cited": True, "kind": "text", "image_url": None,
         "snippet": None, "chunk_id": None}
    ])]
    assert fake_backend.calls["persist"] == []  # no new rows, the existing one was updated
    assert len(at.session_state["history"]) == 2  # no duplicate user message
    assert at.session_state["history"][1].content == "We build **websites** [1]."
    assert at.session_state["sources"][1][0]["url"] == "https://acme.example/"
    assert at.session_state["message_meta"][1] == {"id": 2, "rating": None, "gen": 1, "suggestions": FU}
    assert at.chat_message[1].markdown[0].value == "We build **websites** [[1]](https://acme.example/)."
    assert _feedback(at, f"fb_{CONV1}_1_1").value is None  # fresh rating for the new text
    assert at.button(key=f"regen_{CONV1}_1")  # can be regenerated again
    assert not any(":shimmer[" in t for t in _texts(at))
    assert "regenerate_index" not in at.session_state


def test_regenerate_can_be_repeated(fake_backend):
    at = _run_app()
    at.button(key=f"conv_{CONV1}").click().run()
    at.button(key=f"regen_{CONV1}_1").click().run()
    at.button(key=f"regen_{CONV1}_1").click().run()
    assert not at.exception, at.exception
    assert fake_backend.calls["stream"] == ["What services?", "What services?"]
    assert [u[0] for u in fake_backend.calls["update"]] == [2, 2]
    assert at.session_state["message_meta"][1]["gen"] == 2
    assert _feedback(at, f"fb_{CONV1}_1_2")


def test_regenerate_failure_keeps_original_and_offers_retry(fake_backend):
    fake_backend.fake_chain.fail_stream = True
    at = _run_app()
    at.button(key=f"conv_{CONV1}").click().run()
    at.button(key=f"regen_{CONV1}_1").click().run()
    assert not at.exception, at.exception
    assert at.error and "not saved" in at.error[0].value
    assert at.button(key=f"retry_regen_{CONV1}_1")
    assert fake_backend.calls["update"] == []
    assert at.session_state["history"][1].content == "Web dev [1]."  # untouched
    assert "Web dev [[1]](https://acme.example/)." in [m.value for m in at.chat_message[1].markdown]  # still visible
    assert at.chat_input[0].proto.disabled  # no new question while the regeneration ran

    fake_backend.fake_chain.fail_stream = False  # "the API is back"
    at.button(key=f"retry_regen_{CONV1}_1").click().run()
    assert not at.exception, at.exception
    assert not at.error
    assert at.session_state["history"][1].content == "We build **websites** [1]."
    assert [u[0] for u in fake_backend.calls["update"]] == [2]
    assert not at.chat_input[0].proto.disabled


def test_regenerate_without_db_row_updates_session_only(fake_backend, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("db down")

    monkeypatch.setattr(chain_mod, "persist_turn", boom)
    at = _run_app()
    at.chat_input[0].set_value("What do you build?").run()
    cid = at.session_state["conversation_id"]
    at.button(key=f"regen_{cid}_1").click().run()
    assert not at.exception, at.exception
    assert fake_backend.calls["update"] == []  # nothing to update
    assert at.session_state["message_meta"][1] == {"id": None, "rating": None, "gen": 1, "suggestions": FU}


def test_feedback_is_saved_and_can_be_changed(fake_backend):
    at = _run_app()
    at.button(key=f"conv_{CONV1}").click().run()
    _feedback(at, f"fb_{CONV1}_1_0").set_value(1).run()  # thumbs up
    assert not at.exception, at.exception
    assert fake_backend.calls["feedback"] == [(2, 1)]
    _feedback(at, f"fb_{CONV1}_1_0").set_value(0).run()  # changed to thumbs down -> same row, new rating
    assert fake_backend.calls["feedback"] == [(2, 1), (2, -1)]
    assert _feedback(at, f"fb_{CONV1}_1_0").value == 0
    assert fake_backend.calls["stream"] == []  # feedback never calls the model


def test_saved_feedback_is_highlighted_after_reload(fake_backend):
    SAVED_RATING["value"] = -1
    at = _run_app()
    at.button(key=f"conv_{CONV1}").click().run()
    assert _feedback(at, f"fb_{CONV1}_1_0").value == 0  # thumbs down shown as selected
    assert at.session_state["message_meta"][1]["rating"] == -1
    assert fake_backend.calls["feedback"] == []  # showing the saved value does not write


def test_feedback_db_error_is_reported_not_raised(fake_backend, monkeypatch):
    def boom(conn, message_id, rating):
        raise ConnectionError("db down")

    monkeypatch.setattr(repo, "upsert_feedback", boom)
    at = _run_app()
    at.button(key=f"conv_{CONV1}").click().run()
    _feedback(at, f"fb_{CONV1}_1_0").set_value(1).run()
    assert not at.exception, at.exception
    assert at.toast and "Could not save" in at.toast[0].value


def test_action_keys_follow_the_open_conversation(fake_backend):
    at = _run_app()
    at.button(key=f"conv_{CONV1}").click().run()
    assert at.button(key=f"regen_{CONV1}_1")
    other = CONVERSATIONS[2]["id"]
    at.button(key=f"conv_{other}").click().run()
    assert not at.exception, at.exception
    assert at.button(key=f"regen_{other}_1")
    assert f"regen_{CONV1}_1" not in [b.key for b in at.button]
    assert [f.key for f in at.get("feedback")] == [f"fb_{other}_1_0"]


# ------------------------------------------------ edit a question (stage 10)
FOUR_ROWS = [
    _row(1, "user", "What services?"),
    _row(2, "assistant", "Web dev [1].", SOURCES_HOME),
    _row(3, "user", "Which languages?"),
    _row(4, "assistant", "Python and JS [1].", SOURCES_HOME),
]


def _open(at, cid=CONV1):
    at.button(key=f"conv_{cid}").click().run()
    assert not at.exception, at.exception
    return at


def test_pencil_only_under_history_user_messages(fake_backend):
    at = _open(_run_app())
    keys = [b.key for b in at.button]
    assert f"edit_{CONV1}_0" in keys and f"edit_{CONV1}_1" not in keys
    assert not at.text_area  # not editing yet
    at.chat_input[0].set_value("Another one").run()  # the new question gets its pencil right away
    assert not at.exception, at.exception
    assert f"edit_{CONV1}_2" in [b.key for b in at.button]
    assert f"edit_{CONV1}_3" not in [b.key for b in at.button]


def test_first_question_redraws_without_empty_state_and_with_pencil(fake_backend):
    at = _run_app()
    assert at.pills(key="chat_suggestion")
    at.chat_input[0].set_value("What do you build?").run()
    assert not at.exception, at.exception
    cid = at.session_state["conversation_id"]
    assert not at.pills  # the welcome chips are gone after the turn
    assert "Ask anything about Acme Support" not in [s.value for s in at.subheader]
    assert at.button(key=f"edit_{cid}_0") and at.button(key=f"regen_{cid}_1")
    assert fake_backend.calls["stream"] == ["What do you build?"]  # the rerun did not ask again
    assert len(fake_backend.calls["persist"]) == 1
    assert at.session_state["conv_version"] == 1  # the sidebar list was refreshed


def test_failed_question_can_be_edited_and_resubmitted(fake_backend):
    fake_backend.fake_chain.fail_stream = True
    at = _run_app()
    at.chat_input[0].set_value("What do you buidl?").run()
    assert not at.exception, at.exception
    cid = at.session_state["conversation_id"]
    assert at.error and at.button(key=f"edit_{cid}_0")  # the failed question keeps a pencil
    fake_backend.fake_chain.fail_stream = False
    at.button(key=f"edit_{cid}_0").click().run()
    at.text_area(key=f"edit_text_{cid}_0").set_value("What do you build?").run()
    at.button(key=f"edit_save_{cid}_0").click().run()
    assert not at.exception, at.exception
    fork = fake_backend.calls["fork"][0]
    assert fork["turns"] == [] and fork["forked_from_message_id"] is None  # nothing was saved before
    assert fake_backend.calls["stream"] == ["What do you buidl?", "What do you build?"]
    assert [c[1] for c in fake_backend.calls["persist"]] == ["What do you build?"]
    assert fake_backend.calls["edited_from"] == [None]
    assert [m.content for m in at.session_state["history"]] == ["What do you build?", "We build **websites** [1]."]
    assert not at.error


def test_debug_expander_survives_the_rerun(fake_backend):
    at = _run_app()
    at.toggle[0].set_value(True).run()  # "Show retrieved chunks"
    at.chat_input[0].set_value("What do you build?").run()
    assert not at.exception, at.exception
    assert any(e.label.startswith("Retrieved chunks") for e in at.expander)
    assert "Standalone question" in " ".join(_texts(at)) or any("Standalone question" in str(m.value) for m in at.markdown)


def test_pencil_opens_editor_and_cancel_restores(fake_backend):
    at = _open(_run_app())
    at.button(key=f"edit_{CONV1}_0").click().run()
    assert not at.exception, at.exception
    ta = at.text_area(key=f"edit_text_{CONV1}_0")
    assert ta.value == "What services?"
    assert at.button(key=f"edit_cancel_{CONV1}_0") and at.button(key=f"edit_save_{CONV1}_0")
    assert at.session_state["editing_index"] == 0
    at.button(key=f"edit_cancel_{CONV1}_0").click().run()
    assert not at.exception, at.exception
    assert not at.text_area and "editing_index" not in at.session_state
    assert at.session_state["conversation_id"] == CONV1
    assert [m.content for m in at.session_state["history"]] == ["What services?", "Web dev [1]."]
    assert fake_backend.calls["fork"] == [] and fake_backend.calls["stream"] == []


def test_empty_edit_is_rejected(fake_backend):
    at = _open(_run_app())
    at.button(key=f"edit_{CONV1}_0").click().run()
    at.text_area(key=f"edit_text_{CONV1}_0").set_value("   ").run()
    at.button(key=f"edit_save_{CONV1}_0").click().run()
    assert not at.exception, at.exception
    assert at.error and "cannot be empty" in at.error[0].value
    assert at.text_area(key=f"edit_text_{CONV1}_0")  # still editing
    assert fake_backend.calls["fork"] == [] and fake_backend.calls["stream"] == [] and fake_backend.calls["persist"] == []
    assert at.session_state["conversation_id"] == CONV1


def test_editing_latest_question_forks_and_regenerates(fake_backend):
    at = _open(_run_app())
    at.button(key=f"edit_{CONV1}_0").click().run()
    at.text_area(key=f"edit_text_{CONV1}_0").set_value("What do you build?\nList them.").run()
    at.button(key=f"edit_save_{CONV1}_0").click().run()
    assert not at.exception, at.exception
    new_cid = at.session_state["conversation_id"]
    assert new_cid != CONV1
    # the branch was created before the edited question, with nothing before it
    assert fake_backend.calls["fork"] == [{
        "new_id": new_cid, "chatbot_id": 7, "source_id": CONV1, "forked_from_message_id": 1, "turns": [],
    }]
    # fresh retrieval with the multiline question and no context; one model call; one save with the edit link
    assert fake_backend.calls["prepare"] == [("What do you build?\nList them.", [])]
    assert fake_backend.calls["stream"] == ["What do you build?\nList them."]
    assert fake_backend.calls["persist"] == [(new_cid, "What do you build?\nList them.", "We build **websites** [1].")]
    assert fake_backend.calls["edited_from"] == [1]
    assert fake_backend.calls["update"] == []  # the old answer row is never touched
    assert [m.content for m in at.session_state["history"]] == ["What do you build?\nList them.", "We build **websites** [1]."]
    assert at.session_state["message_meta"] == [{"id": 41}, {"id": 42, "rating": None, "gen": 0, "suggestions": FU}]
    assert "Web dev [1]." not in _texts(at)  # the outdated answer is gone from the active branch
    assert at.session_state["sources"][1][0]["url"] == "https://acme.example/"
    assert "pending_edited_from" not in at.session_state and "editing_index" not in at.session_state
    assert not at.text_area


def test_editing_older_question_keeps_only_the_turns_before_it(fake_backend):
    MESSAGE_ROWS["rows"] = FOUR_ROWS
    at = _open(_run_app())
    assert at.button(key=f"edit_{CONV1}_2")
    at.button(key=f"edit_{CONV1}_2").click().run()
    at.text_area(key=f"edit_text_{CONV1}_2").set_value("Which frameworks?").run()
    at.button(key=f"edit_save_{CONV1}_2").click().run()
    assert not at.exception, at.exception
    new_cid = at.session_state["conversation_id"]
    fork = fake_backend.calls["fork"][0]
    assert fork["source_id"] == CONV1 and fork["forked_from_message_id"] == 3
    assert fork["turns"] == [("user", "What services?", None), ("assistant", "Web dev [1].", SOURCES_HOME)]
    assert fake_backend.calls["prepare"] == [("Which frameworks?", ["What services?", "Web dev [1]."])]
    assert [m.content for m in at.session_state["history"]] == [
        "What services?", "Web dev [1].", "Which frameworks?", "We build **websites** [1]."
    ]
    assert "Python and JS [1]." not in _texts(at)  # the later turn is not part of the branch
    # the copied turns carry the branch's own ids, so feedback targets the copies, not the originals
    assert at.session_state["message_meta"][:2] == [{"id": 101}, {"id": 102, "rating": None, "gen": 0, "suggestions": []}]
    assert _feedback(at, f"fb_{new_cid}_1_0")
    assert fake_backend.calls["edited_from"] == [3]


def test_editing_first_of_several_questions_copies_nothing(fake_backend):
    MESSAGE_ROWS["rows"] = FOUR_ROWS
    at = _open(_run_app())
    at.button(key=f"edit_{CONV1}_0").click().run()
    at.text_area(key=f"edit_text_{CONV1}_0").set_value("What do you offer?").run()
    at.button(key=f"edit_save_{CONV1}_0").click().run()
    assert not at.exception, at.exception
    assert fake_backend.calls["fork"][0]["turns"] == []
    assert len(at.session_state["history"]) == 2


def test_edit_failure_keeps_question_and_retry_saves_once(fake_backend):
    fake_backend.fake_chain.fail_stream = True
    at = _open(_run_app())
    at.button(key=f"edit_{CONV1}_0").click().run()
    at.text_area(key=f"edit_text_{CONV1}_0").set_value("What do you build?").run()
    at.button(key=f"edit_save_{CONV1}_0").click().run()
    assert not at.exception, at.exception
    assert at.error and at.button(key="retry_turn")
    assert len(fake_backend.calls["fork"]) == 1 and fake_backend.calls["persist"] == []
    assert "What do you build?" in [m.value for m in at.chat_message[0].markdown]  # edited question still shown
    assert at.session_state["pending_edited_from"] == 1  # kept for the retry
    assert at.session_state["history"] == []  # nothing paired with the old answer

    fake_backend.fake_chain.fail_stream = False
    at.button(key="retry_turn").click().run()
    assert not at.exception, at.exception
    assert len(fake_backend.calls["fork"]) == 1  # no second branch
    assert fake_backend.calls["persist"] == [(at.session_state["conversation_id"], "What do you build?", "We build **websites** [1].")]
    assert fake_backend.calls["edited_from"] == [1]
    assert "pending_edited_from" not in at.session_state


def test_typing_a_new_question_cancels_the_edit(fake_backend):
    at = _open(_run_app())
    at.button(key=f"edit_{CONV1}_0").click().run()
    assert at.text_area
    at.chat_input[0].set_value("Something else").run()
    assert not at.exception, at.exception
    assert not at.text_area and "editing_index" not in at.session_state
    assert fake_backend.calls["fork"] == []
    assert fake_backend.calls["edited_from"] == [None]  # a typed question never inherits an edit link
    assert at.session_state["conversation_id"] == CONV1


def test_sidebar_marks_branched_conversations(fake_backend, monkeypatch):
    branched = dict(CONVERSATIONS[3], parent_conversation_id=CONV1)
    monkeypatch.setattr(repo, "list_conversations", lambda conn, chatbot_id: CONVERSATIONS[:3] + [branched])
    at = _run_app()
    assert at.button(key=f"conv_{branched['id']}").proto.icon == ":material/alt_route:"
    assert at.button(key=f"conv_{CONV1}").proto.icon == ""
    assert any("branched from an edited question" in c.value for c in at.caption)


# ---------------------------------------------------- citations & sources (stage 11)
LINKED_HOME = "[[1]](https://acme.example/)"


def test_loaded_answer_has_clickable_citations_and_compact_sources(fake_backend):
    at = _open(_run_app())
    assert at.chat_message[1].markdown[0].value == f"Web dev {LINKED_HOME}."
    assert at.session_state["history"][1].content == "Web dev [1]."  # stored text keeps plain markers
    texts = _texts(at)
    assert any(t == "**[1]** [Home](https://acme.example/)" for t in texts)  # one card, numbers + link
    assert "score" not in " ".join(texts).lower() and "similarity" not in " ".join(texts).lower()
    assert "Excerpt not available" in [c.value for c in at.caption]  # old row without a snippet
    assert "acme.example" in [c.value for c in at.caption]  # readable url, no raw scheme


def test_scores_only_with_the_debug_toggle(fake_backend):
    at = _run_app()
    at.toggle[0].set_value(True).run()
    at.button(key=f"conv_{CONV1}").click().run()
    assert not at.exception, at.exception
    assert any("similarity 0.50" in c.value for c in at.caption)
    assert any("cosine similarity" in c.value for c in at.caption)


def test_excerpt_popover_shows_highlighted_snippet_and_merges_page_chunks(fake_backend):
    MESSAGE_ROWS["rows"] = [
        _row(1, "user", "Which services do you offer?"),
        _row(2, "assistant", "Web and mobile [1][2].", [
            {"n": 1, "url": "https://acme.example/", "title": "Home", "score": 0.5, "cited": True,
             "kind": "text", "image_url": None, "snippet": "We offer web services *fast*.", "chunk_id": "1:0"},
            {"n": 2, "url": "https://acme.example/", "title": "Home", "score": 0.4, "cited": True,
             "kind": "text", "image_url": None, "snippet": "Also mobile services.", "chunk_id": "1:1"},
            {"n": 3, "url": "", "title": "", "score": 0.3, "cited": False, "kind": "text", "image_url": None,
             "snippet": None, "chunk_id": None},
        ]),
    ]
    at = _open(_run_app())
    texts = _texts(at)
    assert at.chat_message[1].markdown[0].value == f"Web and mobile {LINKED_HOME}[[2]](https://acme.example/)."
    assert "**[1][2]** [Home](https://acme.example/)" in texts  # two chunks of one page -> one card
    assert "We **offer** web **services** \\*fast\\*." in texts  # escaped + question terms highlighted
    assert "Also mobile **services**." in texts
    assert "**[3]** Untitled page _(source unavailable)_" in texts  # missing metadata fallback
    assert any("Sources (1)" == e.label for e in at.expander)  # cited pages counted, not chunks


def test_fresh_answer_is_redrawn_with_links_and_invented_numbers_dropped(fake_backend):
    at = _run_app()
    at.chat_input[0].set_value("What do you build?").run()
    assert not at.exception, at.exception
    assert at.chat_message[1].markdown[0].value == f"We build **websites** {LINKED_HOME}."
    assert fake_backend.calls["persist"][0][2] == "We build **websites** [1]."  # stored plain


def test_regenerated_and_branched_answers_use_their_own_sources(fake_backend):
    at = _open(_run_app())
    at.button(key=f"regen_{CONV1}_1").click().run()
    assert at.chat_message[1].markdown[0].value == f"We build **websites** {LINKED_HOME}."
    assert at.session_state["sources"][1][0]["url"] == "https://acme.example/"
    # edit -> new branch; the new answer gets fresh sources, nothing is inherited from the old answer
    at.button(key=f"edit_{at.session_state['conversation_id']}_0").click().run()
    cid = at.session_state["conversation_id"]
    at.text_area(key=f"edit_text_{cid}_0").set_value("What else?").run()
    at.button(key=f"edit_save_{cid}_0").click().run()
    assert not at.exception, at.exception
    assert len(at.session_state["sources"]) == 2 and at.session_state["sources"][1][0]["cited"] is True


# ------------------------------------------------ follow-up suggestions (stage 12)
FU = ["How do you price it?", "Which stack do you use?"]


def _fu_keys(at):
    return [b.key for b in at.button if b.key and b.key.startswith("fu_")]


def test_fresh_answer_shows_followups_for_the_latest_answer_only(fake_backend):
    at = _run_app()
    at.chat_input[0].set_value("What do you build?").run()
    assert not at.exception, at.exception
    cid = at.session_state["conversation_id"]
    assert _fu_keys(at) == [f"fu_{cid}_1_0", f"fu_{cid}_1_1"]
    assert at.button(key=f"fu_{cid}_1_0").label == FU[0]
    assert "Suggested follow-up questions" in [c.value for c in at.caption]
    assert fake_backend.calls["followups"] == [("What do you build?", [])]  # one call, after the answer
    assert fake_backend.calls["persist_suggestions"] == [FU]  # stored with the answer
    assert at.session_state["message_meta"][1]["suggestions"] == FU
    assert not any(":shimmer[" in t for t in _texts(at))


def test_clicking_a_followup_asks_it_in_the_same_conversation(fake_backend):
    at = _run_app()
    at.chat_input[0].set_value("What do you build?").run()
    cid = at.session_state["conversation_id"]
    at.button(key=f"fu_{cid}_1_0").click().run()
    assert not at.exception, at.exception
    assert at.session_state["conversation_id"] == cid  # same conversation
    assert [m.content for m in at.session_state["history"]] == [
        "What do you build?", "We build **websites** [1].", FU[0], "We build **websites** [1].",
    ]
    # fresh retrieval with the previous turns as context, one model call, one save, no duplicates
    assert fake_backend.calls["prepare"][-1] == (FU[0], ["What do you build?", "We build **websites** [1]."])
    assert fake_backend.calls["stream"] == ["What do you build?", FU[0]]
    assert [p[1] for p in fake_backend.calls["persist"]] == ["What do you build?", FU[0]]
    assert "pending_question" not in at.session_state
    # suggestions now hang under the new (last) answer only
    assert _fu_keys(at) == [f"fu_{cid}_3_0", f"fu_{cid}_3_1"]
    assert fake_backend.calls["edited_from"] == [None, None]


def test_followups_from_storage_show_only_under_the_last_answer(fake_backend):
    MESSAGE_ROWS["rows"] = [
        _row(1, "user", "What services?"),
        dict(_row(2, "assistant", "Web dev [1].", SOURCES_HOME), suggestions=["Old one?"]),
        _row(3, "user", "Which languages?"),
        dict(_row(4, "assistant", "Python [1].", SOURCES_HOME), suggestions=["Do you use Django?"]),
    ]
    at = _open(_run_app())
    assert _fu_keys(at) == [f"fu_{CONV1}_3_0"]
    assert at.button(key=f"fu_{CONV1}_3_0").label == "Do you use Django?"
    assert fake_backend.calls["followups"] == []  # nothing regenerated on load


def test_greeting_and_failure_produce_no_followups_and_keep_the_answer(fake_backend):
    at = _run_app()
    at.chat_input[0].set_value("hello").run()
    assert fake_backend.calls["followups"] == [] and _fu_keys(at) == []  # ungrounded: no call
    fake_backend.fake_chain.fail_followups = True
    at.chat_input[0].set_value("What do you build?").run()
    assert not at.exception, at.exception
    assert _fu_keys(at) == []
    assert at.session_state["history"][-1].content == "We build **websites** [1]."  # answer unaffected
    assert fake_backend.calls["persist_suggestions"][-1] == []
    assert not any(":shimmer[" in t for t in _texts(at))


def test_regenerate_replaces_followups(fake_backend):
    at = _open(_run_app())
    fake_backend.fake_chain.followups = ["Fresh one?"]
    at.button(key=f"regen_{CONV1}_1").click().run()
    assert not at.exception, at.exception
    assert fake_backend.calls["update_suggestions"] == [["Fresh one?"]]
    assert _fu_keys(at) == [f"fu_{CONV1}_1_0"] and at.button(key=f"fu_{CONV1}_1_0").label == "Fresh one?"


def test_edit_branch_drops_old_followups_and_gets_new_ones(fake_backend):
    MESSAGE_ROWS["rows"] = [
        _row(1, "user", "What services?"),
        dict(_row(2, "assistant", "Web dev [1].", SOURCES_HOME), suggestions=["Old one?"]),
        _row(3, "user", "Which languages?"),
        dict(_row(4, "assistant", "Python [1].", SOURCES_HOME), suggestions=["Stale?"]),
    ]
    at = _open(_run_app())
    at.button(key=f"edit_{CONV1}_2").click().run()
    at.text_area(key=f"edit_text_{CONV1}_2").set_value("Which frameworks?").run()
    at.button(key=f"edit_save_{CONV1}_2").click().run()
    assert not at.exception, at.exception
    cid = at.session_state["conversation_id"]
    assert at.session_state["message_meta"][1]["suggestions"] == []  # copied prefix answer: invalidated
    assert _fu_keys(at) == [f"fu_{cid}_3_0", f"fu_{cid}_3_1"]  # the new answer's own suggestions
    assert "Stale?" not in [b.label for b in at.button]


def test_followup_click_never_inherits_a_pending_edit_link(fake_backend):
    fake_backend.fake_chain.fail_stream = True
    at = _open(_run_app())
    at.button(key=f"edit_{CONV1}_0").click().run()
    at.text_area(key=f"edit_text_{CONV1}_0").set_value("What do you build?").run()
    at.button(key=f"edit_save_{CONV1}_0").click().run()  # generation fails, edit link stays pending
    assert at.session_state.get("pending_edited_from") == 1
    fake_backend.fake_chain.fail_stream = False
    at.button(key="retry_turn").click().run()  # retry succeeds and consumes the link
    assert fake_backend.calls["edited_from"] == [1]
    cid = at.session_state["conversation_id"]
    at.button(key=f"fu_{cid}_1_0").click().run()  # a follow-up is a plain new question
    assert fake_backend.calls["edited_from"] == [1, None]


# ----------------------------------------------------------- branding (stage 13)
import io as _io

from PIL import Image as _Image

from chat_with_website.config import settings as _settings
from chat_with_website.ui.branding import DEFAULT_STARTERS, starter_text

BRANDED = {
    "id": 8, "name": "gmr transcription", "description": "Fast, accurate transcription.", "website_ids": [1],
    "llm_provider": "openai", "llm_model": "gpt-4o-mini", "top_k": 5, "system_prompt": None,
    "display_name": "GMR Helper", "welcome_heading": "How can we help *you*?", "accent_color": "#ff6600",
    "starter_questions": ["What do you transcribe?", "How fast is turnaround?", "Do you offer captions?", "How do I upload?"],
    "input_placeholder": "Ask GMR…", "logo_updated_at": "2026-10-13T10:00:00", "has_logo": True,
}


def _png(size=(64, 64)) -> bytes:
    buf = _io.BytesIO()
    _Image.new("RGB", size, (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def branded_backend(fake_backend, monkeypatch):
    logo = _png()
    monkeypatch.setattr(repo, "list_chatbots", lambda conn: [dict(BOTS[0]), dict(BRANDED)])
    monkeypatch.setattr(repo, "get_chatbot_logo", lambda conn, cid: logo if cid == 8 else None)
    fake_backend.calls.update({"branding": [], "logo": [], "reset": []})
    monkeypatch.setattr(repo, "update_chatbot_branding",
                        lambda conn, cid, **f: fake_backend.calls["branding"].append((cid, f)) or {**BRANDED, **f})
    monkeypatch.setattr(repo, "set_chatbot_logo", lambda conn, cid, data: fake_backend.calls["logo"].append((cid, data)) or True)
    monkeypatch.setattr(repo, "reset_chatbot_branding", lambda conn, cid: fake_backend.calls["reset"].append(cid) or True)
    monkeypatch.setattr(repo, "list_websites", lambda conn: [{"id": 1, "name": "site", "status": "ready", "chunk_count": 3}])
    monkeypatch.setattr(_settings, "admin_password", None)
    fake_backend.logo = logo
    return fake_backend


def _html_bodies(at) -> str:
    return " ".join(str(h.proto.body) for h in at.get("html"))


def test_unbranded_chatbot_keeps_the_defaults(branded_backend):
    at = _run_app()
    assert at.selectbox(key="chat_bot_select").value == 7
    assert "Ask anything about Acme Support" in [s.value for s in at.subheader]
    assert at.chat_input[0].placeholder == "Message Acme Support…"
    assert list(at.pills(key="chat_suggestion").options) == [starter_text(x) for x in DEFAULT_STARTERS]
    assert "stBaseButton-primary" not in _html_bodies(at)  # no accent css for the default look


def test_branded_chatbot_applies_its_own_look_and_switching_swaps_everything(branded_backend):
    at = _run_app()
    at.selectbox(key="chat_bot_select").set_value(8).run()
    assert not at.exception, at.exception
    assert "How can we help \\*you\\*?" in [s.value for s in at.subheader]  # user text is escaped, never markdown
    assert "Fast, accurate transcription." in [c.value for c in at.caption]
    assert at.chat_input[0].placeholder == "Ask GMR…"
    assert list(at.pills(key="chat_suggestion").options) == BRANDED["starter_questions"]
    assert "#ff6600" in _html_bodies(at)  # accent css present
    assert at.session_state["history"] == []  # switching reset the chat
    assert at.image  # the logo is shown on the welcome screen
    at.pills(key="chat_suggestion").select("How do I upload?").run()
    assert not at.exception, at.exception
    assert at.chat_message[1].proto.avatar not in ("", ":material/smart_toy:")  # logo used as avatar
    # back to the unbranded bot: nothing of the previous branding remains
    at.selectbox(key="chat_bot_select").set_value(7).run()
    assert "Ask anything about Acme Support" in [s.value for s in at.subheader]
    assert at.chat_input[0].placeholder == "Message Acme Support…"
    assert "#ff6600" not in _html_bodies(at)
    assert list(at.pills(key="chat_suggestion").options) == [starter_text(x) for x in DEFAULT_STARTERS]


def _chatbots_page(label: str | None = None) -> AppTest:
    """Open the Chatbots page. `AppTest.switch_page` only knows file pages, so the function page is
    selected by the hash Streamlit derives from its url path (the function name)."""
    from streamlit.navigation.page import calc_hash

    at = AppTest.from_file(str(APP), default_timeout=30)
    if label:
        at.session_state["chatbot_edit_choice"] = label
    at.run()  # registers the st.navigation pages
    at._page_hash = calc_hash("page_chatbots")
    at.run()
    assert not at.exception, at.exception
    return at


def test_branding_panel_is_only_offered_for_existing_chatbots(branded_backend):
    at = _chatbots_page()
    assert not [t for t in at.text_input if t.key and t.key.startswith("brand_name_")]
    assert any("becomes available once the chatbot is created" in c.value for c in at.caption)
    at = _chatbots_page("[8] gmr transcription")
    assert at.text_input(key="brand_name_8").value == "GMR Helper"
    assert at.color_picker(key="brand_accent_8").value == "#ff6600"
    assert at.text_area(key="brand_starters_8").value == "\n".join(BRANDED["starter_questions"])
    assert at.checkbox(key="brand_remove_logo_8")  # a logo exists -> can be removed
    assert "How can we help \\*you\\*?" in [s.value for s in at.subheader]  # live preview


def test_branding_save_writes_cleaned_values_once(branded_backend):
    at = _chatbots_page("[8] gmr transcription")
    at.text_input(key="brand_name_8").set_value("  GMR  Assistant ").run()
    at.text_area(key="brand_starters_8").set_value("One\nTwo?").run()
    at.checkbox(key="brand_remove_logo_8").check().run()
    at.button(key="brand_save_8").click().run()
    assert not at.exception, at.exception
    cid, fields = branded_backend.calls["branding"][0]
    assert cid == 8 and fields["display_name"] == "GMR Assistant" and fields["starter_questions"] == ["One?", "Two?"]
    assert fields["accent_color"] == "#ff6600"
    assert branded_backend.calls["logo"] == [(8, None)]  # logo removed in the same save
    assert len(branded_backend.calls["branding"]) == 1


def test_branding_rejects_too_many_starters_without_writing(branded_backend):
    at = _chatbots_page("[8] gmr transcription")
    at.text_area(key="brand_starters_8").set_value("a?\nb?\nc?\nd?\ne?").run()
    assert at.error and "At most 4" in at.error[0].value
    assert at.button(key="brand_save_8").proto.disabled
    assert branded_backend.calls["branding"] == []


def test_restore_defaults_asks_for_confirmation_first(branded_backend):
    # Like the delete-conversation dialog: AppTest cannot click inside an st.dialog (the rerun does
    # not re-open it), so this checks that nothing is reset before the confirmation is shown.
    at = _chatbots_page("[8] gmr transcription")
    at.button(key="brand_reset_8").click().run()
    assert not at.exception, at.exception
    assert at.button(key="brand_reset_ok") and at.button(key="brand_reset_cancel")
    assert branded_backend.calls["reset"] == []


def test_admin_password_gates_the_chatbots_page_but_not_chat(branded_backend, monkeypatch):
    monkeypatch.setattr(_settings, "admin_password", "s3cret")
    at = _chatbots_page("[8] gmr transcription")
    assert at.text_input(key="admin_password_input")
    assert not [t for t in at.text_input if t.key and t.key.startswith("brand_name_")]  # nothing editable yet
    at.text_input(key="admin_password_input").set_value("wrong").run()
    at.button[0].click().run()  # the form's Unlock button
    assert at.error and "Wrong password" in at.error[0].value
    at.text_input(key="admin_password_input").set_value("s3cret").run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert at.session_state["is_admin"] is True
    assert at.text_input(key="brand_name_8")  # unlocked
    chat = _run_app()  # the Chat page never asks
    assert not [t for t in chat.text_input if t.key == "admin_password_input"] and chat.chat_input


# ------------------------------------------------------- Websites page: knowledge-base status
def _site(**over) -> dict:
    """One `list_websites_overview` row (a ready site, no run recorded) with overrides."""
    base = {
        "id": 1, "name": "acme", "base_url": "https://acme.example/", "status": "ready", "error": None,
        "max_pages": 50, "embedding_provider": _settings.embedding_provider,
        "embedding_model": _settings.resolved_embedding_model, "embedding_dim": _settings.resolved_embedding_dim,
        "last_crawled_at": NOW - timedelta(hours=1), "last_indexed_at": None, "created_at": NOW - timedelta(days=3),
        "pages_total": 12, "pages_indexed": 11, "chunks_total": 340, "newest_chunk_at": NOW - timedelta(hours=1),
        "run_id": None, "run_status": None, "run_started_at": None, "run_crawl_finished_at": None,
        "run_finished_at": None, "run_pages_crawled": None, "run_pages_indexed": None,
        "run_pages_skipped_unchanged": None, "run_pages_skipped_short": None, "run_chunks_created": None,
        "run_error": None, "run_page_errors": None, "run_embedding_model": None,
    }
    return {**base, **over}


def _page_row(n: int, chunks: int) -> dict:
    return {"id": n, "url": f"https://acme.example/p{n}", "title": f"Page {n}", "word_count": 100 + n,
            "crawled_at": NOW, "chunk_count": chunks, "indexed_at": NOW if chunks else None, "total": 0}


@pytest.fixture
def websites_backend(fake_backend, monkeypatch):
    monkeypatch.setattr(_settings, "admin_password", None)
    calls = fake_backend.calls
    calls.update({"overview": [], "pages": []})
    state = {"sites": [_site()], "pages": [_page_row(1, 3), _page_row(2, 0)], "missing": [], "runs": [], "dupes": []}

    def overview(conn, chatbot_id=None):
        calls["overview"].append(chatbot_id)
        if chatbot_id is None:
            return list(state["sites"])
        wanted = {w for b in BOTS if b["id"] == chatbot_id for w in b["website_ids"]}
        return [w for w in state["sites"] if w["id"] in wanted]

    def pages(conn, wid, search="", limit=100, offset=0):
        calls["pages"].append((wid, limit, offset, search))
        hits = [r for r in state["pages"] if search.lower() in r["url"].lower() or search.lower() in (r["title"] or "").lower()]
        rows = [dict(r, total=len(hits)) for r in hits[offset:offset + limit]]
        return rows, (len(hits) if rows else 0)

    monkeypatch.setattr(repo, "list_websites_overview", overview)
    monkeypatch.setattr(repo, "list_pages_with_chunk_counts", pages)
    monkeypatch.setattr(repo, "list_pages_without_chunks", lambda conn, wid, limit=100: list(state["missing"]))
    monkeypatch.setattr(repo, "list_ingestion_runs", lambda conn, wid, limit=10: list(state["runs"]))
    monkeypatch.setattr(repo, "list_duplicate_page_urls", lambda conn, wid, limit=100: list(state["dupes"]))
    fake_backend.state = state
    return fake_backend


def _websites_page() -> AppTest:
    import streamlit as st
    from streamlit.navigation.page import calc_hash

    st.cache_data.clear()  # the dashboard caches its reads per process; every page open starts clean
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    at._page_hash = calc_hash("page_websites")
    at.run()
    assert not at.exception, at.exception
    return at


def _kb_texts(at) -> str:
    """Everything the page says: markdown, captions, metrics, warnings, infos, errors, dataframes."""
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [f"{m.label}={m.value}" for m in at.metric]
    parts += [w.value for w in at.warning] + [i.value for i in at.info] + [e.value for e in at.error]
    parts += [str(d.value) for d in at.dataframe]
    return " | ".join(parts)


def test_websites_page_empty_state(fake_backend, monkeypatch):
    monkeypatch.setattr(_settings, "admin_password", None)
    at = _websites_page()
    assert any("No websites indexed yet" in i.value for i in at.info)
    assert not at.metric


def test_ready_site_shows_live_counts_and_not_available_for_untracked_data(websites_backend):
    at = _websites_page()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Pages indexed"] == "11"
    assert metrics["Chunks stored"] == "340"
    assert metrics["Last successful crawl"] != "Not available"
    assert metrics["Last successful indexing"] == "Not available"  # never derived from the crawl time
    assert "Partially indexed" in _kb_texts(at)  # 11 of 12 stored pages have chunks
    assert "no run has been recorded" in _kb_texts(at)
    assert [w.value for w in at.warning if "Partially indexed" not in w.value] == []  # only the partial warning
    # the page list ran exactly one paginated query for the selected website
    assert websites_backend.calls["pages"][0] == (1, 100, 0, "")
    table = str(at.dataframe[1].value)  # overview table is [0]; the pages table follows
    assert "Indexed" in table and "No chunks" in table


def test_stalled_processing_run_is_flagged(websites_backend):
    websites_backend.state["sites"] = [_site(
        status="crawling", run_id=5, run_status="processing", run_started_at=NOW - timedelta(hours=5),
    )]
    at = _websites_page()
    assert any("never finished" in w.value for w in at.warning)
    assert "Processing" in _kb_texts(at)


def test_failed_run_and_failed_pages_are_listed(websites_backend):
    errs = [{"url": "https://acme.example/p9", "error": "timeout"}]
    websites_backend.state["sites"] = [_site(
        status="failed", error="https://acme.example/p9: timeout", last_crawled_at=NOW,
        run_id=6, run_status="failed", run_started_at=NOW - timedelta(minutes=5), run_finished_at=NOW,
        run_pages_crawled=1, run_pages_indexed=0, run_pages_skipped_unchanged=0, run_pages_skipped_short=0,
        run_chunks_created=0, run_error="https://acme.example/p9: timeout", run_page_errors=errs,
        run_embedding_model="m",
    )]
    at = _websites_page()
    assert any("Last run failed" in e.value for e in at.error)
    assert "Failed" in _kb_texts(at)
    assert any("p9" in str(d.value) and "timeout" in str(d.value) for d in at.dataframe)


def test_stale_data_and_embedding_drift_warnings(websites_backend, monkeypatch):
    monkeypatch.setattr(_settings, "kb_stale_days", 7)
    websites_backend.state["sites"] = [_site(
        last_indexed_at=NOW - timedelta(days=30), embedding_model="old-model", embedding_dim=99,
    )]
    at = _websites_page()
    texts = [w.value for w in at.warning]
    assert any("may be outdated" in t for t in texts)
    assert any("old-model" in t and "99 dims" in t for t in texts)


def test_diagnostics_lists_pages_without_chunks_and_run_history(websites_backend):
    websites_backend.state["missing"] = [{"url": "https://acme.example/p2", "title": "Page 2", "word_count": 102, "crawled_at": NOW}]
    websites_backend.state["runs"] = [{
        "id": 3, "status": "completed", "started_at": NOW, "finished_at": NOW, "pages_crawled": 2, "pages_indexed": 1,
        "pages_skipped_unchanged": 0, "chunks_created": 3, "page_errors": [{"url": "https://acme.example/p2", "error": "x"}],
        "embedding_model": "m", "error": None,
    }]
    at = _websites_page()
    assert any("have no embeddings" in w.value for w in at.warning)
    assert "Every stored page" not in _kb_texts(at)
    assert any("'run': 3" in str(d.value) or "3" in str(d.value) for d in at.dataframe)
    assert "Chunks per page" in _kb_texts(at)


def test_chatbot_filter_scopes_the_overview(websites_backend):
    websites_backend.state["sites"] = [_site(), _site(id=2, name="other", base_url="https://other.example/")]
    at = _websites_page()
    assert len(at.dataframe[0].value) == 2
    at.selectbox(key="kb_chatbot_filter").set_value(7).run()
    assert not at.exception, at.exception
    assert websites_backend.calls["overview"][-1] == 7
    assert len(at.dataframe[0].value) == 1  # bot 7 only uses website 1


def test_switching_websites_and_refreshing_requery_the_database(websites_backend):
    websites_backend.state["sites"] = [_site(), _site(id=2, name="other", base_url="https://other.example/", chunks_total=9)]
    at = _websites_page()
    at.selectbox(key="kb_site_select").set_value(2).run()
    assert not at.exception, at.exception
    assert {m.label: m.value for m in at.metric}["Chunks stored"] == "9"
    assert websites_backend.calls["pages"][-1][0] == 2
    before = len(websites_backend.calls["overview"])
    at.button(key="kb_refresh").click().run()
    assert not at.exception, at.exception
    assert len(websites_backend.calls["overview"]) > before


def test_large_site_paginates_the_page_list(websites_backend):
    websites_backend.state["pages"] = [_page_row(n, 1) for n in range(1, 251)]
    websites_backend.state["sites"] = [_site(pages_total=250, pages_indexed=250)]
    at = _websites_page()
    assert websites_backend.calls["pages"][0] == (1, 100, 0, "")  # never the whole site
    assert any("Showing 1–100 of 250" in c.value for c in at.caption)
    at.number_input(key="kb_pages_1_list").set_value(3).run()
    assert not at.exception, at.exception
    assert (1, 100, 200, "") in websites_backend.calls["pages"]
    assert any("Showing 201–250 of 250" in c.value for c in at.caption)


def test_admin_password_gates_the_websites_page(websites_backend, monkeypatch):
    monkeypatch.setattr(_settings, "admin_password", "s3cret")
    at = _websites_page()
    assert at.text_input(key="admin_password_input")
    assert not at.metric and not at.dataframe  # no operational data before unlocking
    assert websites_backend.calls["overview"] == []


def test_fully_indexed_site_is_completed_and_partial_sites_are_not(websites_backend):
    websites_backend.state["sites"] = [_site(pages_total=11, pages_indexed=11)]
    at = _websites_page()
    assert at.dataframe[0].value.iloc[0]["status"] == "Completed" and "Partially" not in _kb_texts(at)
    assert not at.warning
    errs = [{"url": "https://acme.example/p9", "error": "timeout"}]
    websites_backend.state["sites"] = [_site(
        pages_total=11, pages_indexed=11, run_id=6, run_status="completed", run_started_at=NOW, run_finished_at=NOW,
        run_pages_crawled=12, run_pages_indexed=11, run_pages_skipped_unchanged=0, run_pages_skipped_short=0,
        run_chunks_created=30, run_page_errors=errs, run_embedding_model="m",
    )]
    at = _websites_page()
    assert at.dataframe[0].value.iloc[0]["status"] == "Partially indexed"
    assert any("1 page(s) failed in the latest run" in w.value for w in at.warning)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Failed pages"] == "1"
    row = at.dataframe[0].value.iloc[0]
    assert row["discovered"] == "12" and row["failed"] == "1" and row["stored"] == "11"


def test_overview_shows_not_available_for_discovered_and_failed_without_a_run(websites_backend):
    at = _websites_page()
    row = at.dataframe[0].value.iloc[0]
    assert row["discovered"] == "Not available" and row["failed"] == "Not available"
    assert {m.label: m.value for m in at.metric}["Failed pages"] == "Not available"


def test_page_search_filters_server_side_and_resets_pagination(websites_backend):
    websites_backend.state["pages"] = [_page_row(n, 1) for n in range(1, 151)] + [dict(_page_row(999, 2), title="Pricing plans")]
    websites_backend.state["sites"] = [_site(pages_total=151, pages_indexed=151)]
    at = _websites_page()
    at.number_input(key="kb_pages_1_list").set_value(2).run()
    assert (1, 100, 100, "") in websites_backend.calls["pages"]
    at.text_input(key="kb_search_1_list").set_value("PRICING").run()
    assert not at.exception, at.exception
    assert websites_backend.calls["pages"][-1] == (1, 100, 0, "PRICING")  # back to page 1, filter sent to SQL
    table = at.dataframe[1].value
    assert len(table) == 1 and table.iloc[0]["title"] == "Pricing plans" and table.iloc[0]["chunks"] == 2
    assert any("Showing 1–1 of 1" in c.value for c in at.caption)
    at.text_input(key="kb_search_1_list").set_value("nothing-matches").run()
    assert any("No pages match this search" in c.value for c in at.caption)


def test_page_list_shows_error_details_for_failed_pages(websites_backend):
    errs = [{"url": "https://acme.example/p2", "error": "embedding timeout"}]
    websites_backend.state["sites"] = [_site(run_id=6, run_status="completed", run_started_at=NOW, run_finished_at=NOW,
                                             run_pages_crawled=2, run_pages_indexed=1, run_pages_skipped_unchanged=0,
                                             run_pages_skipped_short=0, run_chunks_created=3, run_page_errors=errs,
                                             run_embedding_model="m")]
    at = _websites_page()
    # the page list is the table with both an error and a chunks column (overview has chunks, Failed pages has error)
    pages_table = [d.value for d in at.dataframe if {"error", "chunks"} <= set(d.value.columns)][0]
    p2 = pages_table[pages_table["url"] == "https://acme.example/p2"].iloc[0]
    assert p2["status"] == "Failed" and p2["error"] == "embedding timeout" and p2["chunks"] == 0
    p1 = pages_table[pages_table["url"] == "https://acme.example/p1"].iloc[0]
    assert p1["status"] == "Indexed" and p1["error"] == ""


def test_stale_threshold_is_configurable_on_the_page(websites_backend):
    websites_backend.state["sites"] = [_site(last_indexed_at=NOW - timedelta(days=10))]
    at = _websites_page()
    assert not [w for w in at.warning if "outdated" in w.value]  # default 30 days
    at.number_input(key="kb_stale_days").set_value(5).run()
    assert not at.exception, at.exception
    assert any("more than 5 days" in w.value for w in at.warning)
    assert at.dataframe[0].value.iloc[0]["freshness"] == "stale"


def test_dashboard_reads_are_cached_until_refresh_or_reindex(websites_backend, monkeypatch):
    at = _websites_page()
    n = len(websites_backend.calls["overview"])
    at.text_input(key="kb_search_1_list").set_value("p1").run()  # an ordinary rerun
    assert len(websites_backend.calls["overview"]) == n  # overview served from cache
    at.button(key="kb_refresh").click().run()
    assert len(websites_backend.calls["overview"]) == n + 1  # refresh bypasses the cache
    assert at.session_state["kb_version"] == 1
    # a (faked) re-index bumps the version too, so the new run shows up without clicking Refresh
    # AppTest re-executes the script file on every run, so its `from ... import` picks these up
    import chat_with_website.ingestion.embedder as embedder_mod
    import chat_with_website.ingestion.pipeline as pipeline_mod
    from chat_with_website.ingestion.models import IngestResult
    monkeypatch.setattr(pipeline_mod, "ingest_website",
                        lambda url, name=None, max_pages=None, progress=None: IngestResult(website_id=1, run_id=9))
    monkeypatch.setattr(embedder_mod, "get_embeddings", lambda *a, **k: object())
    at.button(key="reindex_1").click().run()  # runs, bumps the version, flashes the summary, reruns
    assert not at.exception, at.exception
    assert at.session_state["kb_version"] == 2
    assert len(websites_backend.calls["overview"]) == n + 2
    assert any("run #9" in s.value for s in at.success)


def test_diagnostics_lists_duplicate_urls(websites_backend):
    websites_backend.state["dupes"] = [{"key": "acme.example/a", "n": 2, "urls": ["https://acme.example/a", "https://acme.example/a/"]}]
    at = _websites_page()
    assert any("look like the same page" in w.value for w in at.warning)
    assert any("acme.example/a/" in str(d.value) for d in at.dataframe)
    assert "Total vector records: 340" in _kb_texts(at)
    assert "Last indexing execution" in _kb_texts(at)


# ------------------------------------------------------------------- RAG Debugger page
import chat_with_website.rag.debug as _rdbg
from langchain_core.documents import Document as _Doc


def _rdbg_test(bot_id=7, k=5, docs=None, **over) -> _rdbg.RetrievalTest:
    docs = docs if docs is not None else [
        _Doc(page_content="Acme builds websites.", metadata={"url": "https://acme.example/services", "title": "Services",
                                                            "score": 0.81, "chunk_id": "3:0", "db_id": 30, "website_id": 1, "type": "text"}),
        _Doc(page_content="Long " * 200, metadata={"url": "https://acme.example/long", "title": None, "score": 0.1,
                                                    "chunk_id": "4:0", "db_id": 40, "website_id": 1, "type": "text"}),
    ]
    t = _rdbg.RetrievalTest(bot_id, "What do you build?", k, 0.25, [1], False, False, embed_ms=12.5, search_ms=3.2,
                            total_ms=16.0, context_chars=120, est_tokens=30, distinct_pages=1)
    t.chunks = [_rdbg.chunk_from_document(i, d, 0.25) for i, d in enumerate(docs, start=1)]
    for key, value in over.items():
        setattr(t, key, value)
    return t


@pytest.fixture
def debugger_backend(fake_backend, monkeypatch):
    monkeypatch.setattr(_settings, "admin_password", None)
    calls = fake_backend.calls
    calls.update({"retrieval": [], "compare": [], "generate": []})
    state = {"test": _rdbg_test()}

    def run(bot_id, question, k):
        calls["retrieval"].append((bot_id, question, k))
        return _rdbg_test(bot_id=bot_id, k=k, docs=[c.document for c in state["test"].chunks])

    def compare(bot_id, question, ks=_rdbg.DEFAULT_KS):
        calls["compare"].append((bot_id, question, tuple(ks)))
        return [_rdbg_test(bot_id=bot_id, k=k) for k in ks]

    def generate(chain, test):
        calls["generate"].append(test.k)
        from chat_with_website.rag.chain import Source
        return _rdbg.AnswerTest(answer="Acme builds websites [1].", sources=[Source(n=1, url="https://acme.example/services",
                                title="Services", score=0.81, cited=True)], grounded=True, gen_ms=250.0, llm_called=True,
                                context_chars=120, model_tokens=None)

    monkeypatch.setattr(_rdbg, "run_retrieval_test", run)
    monkeypatch.setattr(_rdbg, "compare_top_k", compare)
    monkeypatch.setattr(_rdbg, "generate_test_answer", generate)

    def boom(*a, **k):
        raise AssertionError("diagnostics must not modify chatbots or persist messages")

    monkeypatch.setattr(repo, "update_chatbot", boom)
    monkeypatch.setattr(chain_mod, "persist_turn", boom)
    fake_backend.state = state
    return fake_backend


def _debugger_page() -> AppTest:
    from streamlit.navigation.page import calc_hash

    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    at._page_hash = calc_hash("page_rag_debugger")
    at.run()
    assert not at.exception, at.exception
    return at


def _run_test(at, question="What do you build?") -> AppTest:
    at.text_area(key="rdbg_question").set_value(question).run()
    at.button(key="rdbg_run").click().run()
    assert not at.exception, at.exception
    return at


def test_debugger_requires_the_admin_password(debugger_backend, monkeypatch):
    monkeypatch.setattr(_settings, "admin_password", "s3cret")
    at = _debugger_page()
    assert at.text_input(key="admin_password_input")
    assert not [s for s in at.selectbox if s.key == "rdbg_bot"] and not at.metric
    assert debugger_backend.calls["retrieval"] == []


def test_debugger_runs_retrieval_and_renders_chunks_honestly(debugger_backend):
    at = _debugger_page()
    assert at.selectbox(key="rdbg_bot").value == 7
    assert at.radio(key="rdbg_k_7").value == 5  # defaults to the chatbot's saved top_k
    assert any("custom instructions: none" in c.value for c in at.caption)
    _run_test(at)
    assert debugger_backend.calls["retrieval"] == [(7, "What do you build?", 5)]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Chunks retrieved"] == "2" and metrics["Kept for the model"] == "1"
    assert metrics["Embedding"] == "12 ms" and metrics["Vector search"] == "3 ms"
    md = " ".join(m.value for m in at.markdown)
    assert ":green-badge[kept]" in md and "dropped – below threshold" in md
    assert "0.8100" in md and "[Services](https://acme.example/services)" in md
    assert "Not available" in md  # the second chunk has no title
    assert any("chunk id: 3:0" in c.value and "db id: 30" in c.value for c in at.caption)
    assert any("Full chunk text" in e.label for e in at.expander)  # long chunk collapsed
    assert any("higher is better" in c.value for c in at.caption)
    assert not at.warning


def test_debugger_top_k_choice_only_affects_the_test(debugger_backend):
    at = _debugger_page()
    at.radio(key="rdbg_k_7").set_value(8).run()
    _run_test(at)
    assert debugger_backend.calls["retrieval"][-1][2] == 8
    assert any("Tested with top_k = 8" in i.value and "saved value is 5" in i.value for i in at.info)
    # the Chatbots page still shows the saved value and nothing was written
    bots = _chatbots_page("[7] acme support")
    assert bots.number_input[0].value == 5 if bots.number_input else True


def test_debugger_compare_runs_every_k(debugger_backend):
    at = _debugger_page()
    at.checkbox(key="rdbg_compare").check().run()
    _run_test(at)
    assert debugger_backend.calls["compare"] == [(7, "What do you build?", (3, 5, 8, 10))]
    table = at.dataframe[0].value
    assert list(table["top_k"]) == [3, 5, 8, 10] and "SQL ms" in table.columns
    assert any("Top-K comparison" in h.value for h in at.subheader)


def test_debugger_generates_an_answer_and_separates_it_from_the_evidence(debugger_backend):
    at = _debugger_page()
    _run_test(at)
    at.button(key="rdbg_generate").click().run()
    assert not at.exception, at.exception
    assert debugger_backend.calls["generate"] == [5]
    md = " ".join(m.value for m in at.markdown)
    assert "Generated answer (LLM)" in md and "Retrieved evidence" in md
    assert "cited in answer" in md
    assert any("generation: 250 ms" in c.value and "model called: yes" in c.value for c in at.caption)
    assert any("does not make the answer correct" in c.value for c in at.caption)
    assert any("Exact context sent" in e.label for e in at.expander)


def test_debugger_empty_question_and_no_websites(debugger_backend):
    at = _debugger_page()
    at.button(key="rdbg_run").click().run()
    assert any("Type a question" in w.value for w in at.warning)
    assert debugger_backend.calls["retrieval"] == []
    debugger_backend.state["test"] = _rdbg_test(docs=[])
    _run_test(at)
    assert any("No chunks came back" in i.value for i in at.info)


def test_debugger_results_are_hidden_when_the_chatbot_changes(debugger_backend, monkeypatch):
    monkeypatch.setattr(repo, "list_chatbots", lambda conn: list(BOTS) + [dict(BOTS[0], id=8, name="other", website_ids=[2])])
    at = _debugger_page()
    _run_test(at)
    assert at.metric
    at.selectbox(key="rdbg_bot").set_value(8).run()
    assert not at.exception, at.exception
    assert not at.metric  # the stored result belongs to chatbot 7


# ------------------------------------------------------------------- RAG Evaluation page
import chat_with_website.rag.evaluation as _eval


def _eval_case_row(cid: int, question: str, bot: int = 7, **over) -> dict:
    row = {"id": cid, "chatbot_id": bot, "question": question, "expected_answer": "Web development.", "answerable": True,
           "category": "factual", "source_urls": [], "key_facts": [], "previous_turns": [], "notes": None,
           "enabled": True, "created_at": NOW, "updated_at": NOW}
    row.update(over)
    return row


def _eval_run_row(rid: int, bot: int = 7, status: str = "completed", **over) -> dict:
    row = {"id": rid, "chatbot_id": bot, "status": status, "started_at": NOW, "finished_at": NOW, "total_cases": 2,
           "done": 2, "passed": 1, "failed": 1, "needs_review": 0, "errored": 0, "last_question": "Pricing?",
           "config": {"llm_provider": "openai", "llm_model": "gpt-4o-mini", "llm_temperature": 0.0, "top_k": 5,
                      "score_threshold": 0.25, "embedding_model": "text-embedding-3-large", "embedding_dim": 3072,
                      "custom_instructions_set": False},
           "judge_model": "OpenAI · gpt-4o-mini", "rubric_version": "1.0", "judge_prompt_hash": "abc123", "error": None,
           "avg_prepare_ms": 41.0, "median_prepare_ms": 40.0, "avg_generation_ms": 950.0, "median_generation_ms": 900.0,
           "avg_judge_ms": 600.0, "answer_input_tokens": 2000, "answer_output_tokens": 100, "judge_input_tokens": 0,
           "judge_output_tokens": 0, "results": 2, "results_with_usage": 2, "judge_failures": 0}
    row.update(over)
    return row


def _eval_result_row(rid: int, run_id: int, status: str, question: str, **over) -> dict:
    row = {"id": rid, "run_id": run_id, "chatbot_id": 7, "test_case_id": rid, "question": question,
           "expected_answer": "From $50 per month.", "answerable": True, "category": "pricing", "source_urls": [],
           "key_facts": ["$50"], "case_updated_at": NOW, "case_current_updated_at": NOW, "case_exists": True,
           "standalone_question": question, "raw_answer": "It is cheap [1].", "generated_answer": "It is cheap [1].",
           "grounded": True, "model_called": True,
           "retrieved": [{"n": 1, "url": "https://acme.example/pricing", "title": "Pricing", "score": 0.71, "cited": True,
                          "snippet": "Plans start at $50 per month."}],
           "status": status, "failure_code": "key_fact_missing" if status == "failed" else None,
           "failure_reason": "Required key facts missing from the answer: $50." if status == "failed" else None,
           "correctness": {"verdict": "incorrect", "confidence": "high", "reason": "No price given.",
                           "key_facts_missing": ["$50"] if status == "failed" else []},
           "faithfulness": {"verdict": "supported", "unsupported_claims": []},
           "retrieval": {"k": 1, "recall_at_k": None, "expected_urls": [], "matched_urls": [], "top_score": 0.71},
           "citations": {"checked": True, "cited": [1], "invalid": []}, "prepare_ms": 40.0, "generation_ms": 900.0,
           "judge_ms": 600.0, "usage": {"answer": {"input_tokens": 1000, "output_tokens": 50}},
           "judge_model": "OpenAI · gpt-4o-mini", "rubric_version": "1.0", "error": None, "created_at": NOW}
    row.update(over)
    return row


@pytest.fixture
def evaluation_backend(fake_backend, monkeypatch):
    """In-memory eval_* tables; the batch runner is replaced by a synchronous fake."""
    monkeypatch.setattr(_settings, "admin_password", None)
    monkeypatch.setattr(_settings, "eval_judge_enabled", True)
    calls = fake_backend.calls
    calls.update({"eval_create": [], "eval_update": [], "eval_delete": [], "eval_upsert": [], "eval_start": [],
                  "eval_list": []})
    state = {"cases": {1: _eval_case_row(1, "What services do you offer?"),
                       2: _eval_case_row(2, "Pricing?", category="pricing", key_facts=["$50"]),
                       3: _eval_case_row(3, "Do you sell cars?", category="unanswerable", answerable=False,
                                         expected_answer="", enabled=False),
                       9: _eval_case_row(9, "Other bot question", bot=8)},
             "runs": [], "results": {}}

    def cases_of(bot):
        return [c for c in sorted(state["cases"].values(), key=lambda c: c["id"]) if c["chatbot_id"] == bot]

    def list_cases(conn, bot, *, search="", category=None, answerable=None, enabled=None, limit=50, offset=0):
        calls["eval_list"].append({"bot": bot, "search": search, "category": category, "answerable": answerable})
        rows = [c for c in cases_of(bot)
                if (not search or search.lower() in (c["question"] + c["expected_answer"]).lower())
                and (category is None or c["category"] == category)
                and (answerable is None or c["answerable"] == answerable)]
        return rows[offset: offset + limit], len(rows)

    def create(conn, bot, **fields):
        calls["eval_create"].append((bot, fields))
        cid = max(state["cases"]) + 1
        state["cases"][cid] = _eval_case_row(cid, bot=bot, **fields)
        return state["cases"][cid]

    def update(conn, cid, bot, **fields):
        calls["eval_update"].append((cid, bot, fields))
        if cid not in state["cases"] or state["cases"][cid]["chatbot_id"] != bot:
            return None
        state["cases"][cid].update(fields)
        return state["cases"][cid]

    def delete(conn, cid, bot):
        calls["eval_delete"].append((cid, bot))
        return state["cases"].pop(cid, None) is not None

    def upsert(conn, bot, rows):
        calls["eval_upsert"].append((bot, list(rows)))
        for r in rows:
            create(conn, bot, **r)
        return len(rows)

    def runs_of(conn, bot, limit=20):
        return [r for r in state["runs"] if r["chatbot_id"] == bot][:limit]

    def start(bot, ids, *, chain, judge=None):
        calls["eval_start"].append((bot, list(ids)))
        rid = 100 + len(state["runs"])
        state["runs"].insert(0, _eval_run_row(rid, bot=bot, total_cases=len(ids), done=len(ids), passed=len(ids), failed=0))
        state["results"][rid] = [_eval_result_row(i, rid, "passed", state["cases"][i]["question"]) for i in ids]
        return rid

    monkeypatch.setattr(repo, "count_eval_cases", lambda conn, bot: {"total": len(cases_of(bot)),
                                                                     "enabled": sum(c["enabled"] for c in cases_of(bot))})
    monkeypatch.setattr(repo, "list_eval_cases", list_cases)
    monkeypatch.setattr(repo, "list_all_eval_cases", lambda conn, bot: cases_of(bot))
    monkeypatch.setattr(repo, "get_eval_case", lambda conn, cid, bot: (
        state["cases"].get(cid) if state["cases"].get(cid, {}).get("chatbot_id") == bot else None))
    monkeypatch.setattr(repo, "create_eval_case", create)
    monkeypatch.setattr(repo, "update_eval_case", update)
    monkeypatch.setattr(repo, "delete_eval_case", delete)
    monkeypatch.setattr(repo, "upsert_eval_cases", upsert)
    monkeypatch.setattr(repo, "list_eval_runs", runs_of)
    monkeypatch.setattr(repo, "get_eval_run", lambda conn, rid, bot: next(
        (r for r in state["runs"] if r["id"] == rid and r["chatbot_id"] == bot), None))
    monkeypatch.setattr(repo, "list_eval_results", lambda conn, rid, bot, status=None, category=None, search="": [
        r for r in state["results"].get(rid, []) if status is None or r["status"] == status])
    monkeypatch.setattr(repo, "eval_regressions", lambda conn, bot, rid: [])
    monkeypatch.setattr(_eval, "start_eval_run_in_background", start)
    monkeypatch.setattr(_eval, "build_judge", lambda: None)

    def boom(*a, **k):
        raise AssertionError("evaluation must not write conversation history")

    monkeypatch.setattr(repo, "create_conversation", boom)
    monkeypatch.setattr(repo, "add_message", boom)
    fake_backend.state = state
    return fake_backend


def _evaluation_page() -> AppTest:
    from streamlit.navigation.page import calc_hash

    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    at._page_hash = calc_hash("page_rag_evaluation")
    at.run()
    assert not at.exception, at.exception
    return at


def _eval_texts(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [w.value for w in at.warning]
    parts += [e.value for e in at.error] + [i.value for i in at.info] + [s.value for s in at.success]
    # AppTest parses an expander that has an icon (the result cards) as a Status element
    parts += [f"{m.label}={m.value}" for m in at.metric] + [e.label for e in at.expander] + [e.label for e in at.status]
    return "\n".join(str(p) for p in parts)


def test_evaluation_requires_the_admin_password(evaluation_backend, monkeypatch):
    monkeypatch.setattr(_settings, "admin_password", "s3cret")
    called = []
    monkeypatch.setattr(repo, "count_eval_cases", lambda conn, bot: called.append(bot))
    at = _evaluation_page()
    assert at.text_input(key="admin_password_input")
    assert not at.tabs and not at.metric and called == []
    assert any("admin page" in i.value for i in at.info)


def test_evaluation_dashboard_without_runs(evaluation_backend):
    at = _evaluation_page()
    assert [t.label for t in at.tabs] == ["Dashboard", "Test cases", "Results"]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Test cases"] == "3" and metrics["Latest run"] == "Not available"
    assert metrics["Avg retrieval latency"] == "Not available"
    text = _eval_texts(at)
    assert "2 enabled case(s)" in text and "No evaluation has been run" in text and "judge: OpenAI · gpt-4o-mini" in text
    assert "Other bot question" not in text  # chatbot 8's case is not listed for chatbot 7


def test_evaluation_creates_a_case_and_validates_it(evaluation_backend):
    at = _evaluation_page()
    at.text_area(key="eval_q_7_new").set_value("Where are you based?")
    at.button(key="eval_save_7_new").click().run()
    assert any("needs a verified expected answer" in e.value for e in at.error)
    assert evaluation_backend.calls["eval_create"] == []
    at.text_area(key="eval_q_7_new").set_value("Where are you based?")
    at.text_area(key="eval_a_7_new").set_value("In Hyderabad, India.")
    at.text_area(key="eval_urls_7_new").set_value("https://acme.example/about\nhttps://acme.example/contact")
    at.text_area(key="eval_facts_7_new").set_value("Hyderabad")
    at.selectbox(key="eval_cat_7_new").set_value("company")
    at.button(key="eval_save_7_new").click().run()
    assert not at.exception, at.exception
    bot, fields = evaluation_backend.calls["eval_create"][0]
    assert bot == 7 and fields["category"] == "company" and fields["key_facts"] == ["Hyderabad"]
    assert fields["source_urls"] == ["https://acme.example/about", "https://acme.example/contact"]
    assert "#10 Where are you based?" in _eval_texts(at).replace("**", "")


def test_evaluation_unanswerable_category_forces_answerable_off(evaluation_backend):
    at = _evaluation_page()
    at.text_area(key="eval_q_7_new").set_value("Do you sell boats?")
    at.selectbox(key="eval_cat_7_new").set_value("unanswerable")
    at.button(key="eval_save_7_new").click().run()
    _, fields = evaluation_backend.calls["eval_create"][0]
    assert fields["answerable"] is False and fields["expected_answer"] == ""


def test_evaluation_edits_a_case(evaluation_backend):
    at = _evaluation_page()
    at.button(key="eval_edit_2").click().run()
    assert at.text_area(key="eval_q_7_2").value == "Pricing?"
    assert at.text_area(key="eval_facts_7_2").value == "$50"
    at.text_area(key="eval_a_7_2").set_value("From $60 per month.")
    at.text_area(key="eval_facts_7_2").set_value("$60")
    at.button(key="eval_save_7_2").click().run()
    assert not at.exception, at.exception
    cid, bot, fields = evaluation_backend.calls["eval_update"][0]
    assert (cid, bot, fields["expected_answer"], fields["key_facts"]) == (2, 7, "From $60 per month.", ["$60"])
    assert not [t for t in at.text_area if t.key == "eval_q_7_2"]  # back to the "new" form


def test_evaluation_delete_needs_confirmation(evaluation_backend):
    at = _evaluation_page()
    assert at.button(key="eval_delete_1").disabled
    at.checkbox(key="eval_confirm_1").check().run()
    at.button(key="eval_delete_1").click().run()
    assert evaluation_backend.calls["eval_delete"] == [(1, 7)]
    assert 1 not in evaluation_backend.state["cases"] and not [b for b in at.button if b.key == "eval_delete_1"]


def test_evaluation_search_and_filters(evaluation_backend):
    at = _evaluation_page()
    at.text_input(key="eval_search_7").set_value("pric").run()
    assert evaluation_backend.calls["eval_list"][-1]["search"] == "pric"
    assert [b.key for b in at.button if b.key and b.key.startswith("eval_edit_")] == ["eval_edit_2"]
    at.text_input(key="eval_search_7").set_value("").run()
    at.selectbox(key="eval_filter_ans_7").set_value(False).run()
    assert [b.key for b in at.button if b.key and b.key.startswith("eval_edit_")] == ["eval_edit_3"]
    assert "excluded from runs" in _eval_texts(at)


def test_evaluation_csv_import_preview_and_errors(evaluation_backend):
    at = _evaluation_page()
    bad = b"question,expected_answer,answerable,category\nOpening hours?,9 to 5,yes,factual\nBroken,,maybe,factual\n"
    at.file_uploader(key="eval_upload_7_0").upload("cases.csv", bad, "text/csv").run()
    assert not at.exception, at.exception
    assert any("Line 3: answerable must be yes/no" in e.value for e in at.error)
    assert at.button(key="eval_import_7").disabled  # an invalid row blocks the import...
    at.checkbox(key="eval_valid_only_7_0").check().run()
    at.button(key="eval_import_7").click().run()  # ...unless only the valid rows are imported
    assert not at.exception, at.exception
    bot, rows = evaluation_backend.calls["eval_upsert"][0]
    assert bot == 7 and [r["question"] for r in rows] == ["Opening hours?"]
    assert any("Imported 1 test case(s)" in s.value for s in at.success)
    assert at.download_button(key="eval_export_cases_7")


def test_evaluation_run_starts_with_enabled_cases_and_shows_counts(evaluation_backend):
    at = _evaluation_page()
    at.button(key="eval_run_7").click().run()
    assert not at.exception, at.exception
    assert evaluation_backend.calls["eval_start"] == [(7, [1, 2])]  # case 3 is disabled, case 9 is another bot's
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Latest run"] == "Completed" and metrics["Passed"] == "2" and metrics["Failed"] == "0"
    assert metrics["Avg generation latency"] == "950 ms"
    df = at.dataframe[0].value
    assert list(df["run"]) == [100] and list(df["pass rate"]) == ["100%"]
    assert evaluation_backend.calls["persist"] == []


def test_evaluation_run_category_filter(evaluation_backend):
    at = _evaluation_page()
    at.multiselect(key="eval_run_categories_7").set_value(["pricing"])
    at.button(key="eval_run_7").click().run()
    assert evaluation_backend.calls["eval_start"] == [(7, [2])]


def test_evaluation_running_run_blocks_a_second_one(evaluation_backend):
    evaluation_backend.state["runs"] = [_eval_run_row(5, status="running", done=1, total_cases=4, passed=1, failed=0)]
    at = _evaluation_page()
    assert not [b for b in at.button if b.key == "eval_run_7"]
    assert any("Evaluating run #5: 1 of 4" in s.label for s in at.status)
    assert evaluation_backend.calls["eval_start"] == []


def test_evaluation_results_show_expected_vs_generated_and_evidence(evaluation_backend):
    evaluation_backend.state["runs"] = [_eval_run_row(5)]
    evaluation_backend.state["results"][5] = [
        _eval_result_row(1, 5, "passed", "What services do you offer?", key_facts=[], category="service"),
        _eval_result_row(2, 5, "failed", "Pricing?", test_case_id=None, case_exists=False),
    ]
    at = _evaluation_page()
    text = _eval_texts(at)
    assert "Failed · Pricing · Pricing?" in text and "Passed · Service Information · What services" in text
    assert "#### Expected answer (verified)" in text and "#### Generated answer (chatbot)" in text
    assert "Required key facts missing from the answer: $50." in text
    assert "The test case was deleted after this run" in text
    assert "No labelled source pages: Recall@K not computed." in text
    assert "[Pricing](https://acme.example/pricing)" in text and ":green-badge[cited]" in text
    assert "rubric v1.0 (abc123)" in text and "custom instructions none" in text
    assert at.download_button(key="eval_export_7_5") and at.download_button(key="eval_export_latest_7")
    at.selectbox(key="eval_res_status_7").set_value("failed").run()
    assert [e.label for e in at.status if e.label.startswith(("Passed", "Failed"))] == ["Failed · Pricing · Pricing?"]


def test_evaluation_data_follows_the_selected_chatbot(evaluation_backend, monkeypatch):
    monkeypatch.setattr(repo, "list_chatbots", lambda conn: list(BOTS) + [dict(BOTS[0], id=8, name="other", website_ids=[2])])
    at = _evaluation_page()
    at.selectbox(key="eval_bot").set_value(8).run()
    assert not at.exception, at.exception
    text = _eval_texts(at)
    assert "Other bot question" in text and "Pricing?" not in text
    assert {m.label: m.value for m in at.metric}["Test cases"] == "1"
