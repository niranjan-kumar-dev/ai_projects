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
