"""RAG Debugger logic with the embedder and the database faked (no network, no PostgreSQL).

Also covers `search_chunks` / `search_chunks_timed`, the single retrieval path the chatbot and
the debugger share.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import chat_with_website.rag.debug as dbg
import chat_with_website.rag.retriever as retriever_mod
from chat_with_website.config import settings
from chat_with_website.db import repository as repo
from chat_with_website.rag.chain import RagChain
from tests.test_chain import DOCS, FakeRetriever


def _row(i: int, score: float, kind: str = "text", website_id: int = 1, title="Page", url=None) -> dict:
    return {
        "id": 100 + i, "content": f"passage {i} about services", "chunk_index": i,
        "metadata": {"chunk_id": f"{10 + i}:{i}", "page_id": 10 + i, "type": kind,
                     **({"image_url": f"https://a/img{i}.png"} if kind == "image" else {})},
        "url": url or f"https://a.example/p{i}", "title": title, "website_id": website_id, "score": score,
    }


@pytest.fixture
def fake_search(monkeypatch):
    calls: dict[str, list] = {"search": [], "embed": [], "website_ids": []}
    state = {"rows": [_row(1, 0.81), _row(2, 0.40), _row(3, 0.10)], "website_ids": [1, 2]}

    def website_ids(conn, chatbot_id):
        calls["website_ids"].append(chatbot_id)
        return list(state["website_ids"])

    def search(conn, vector, website_ids, k=5, min_score=0.0, kinds=None):
        calls["search"].append({"website_ids": list(website_ids), "k": k, "min_score": min_score, "kinds": kinds})
        rows = [r for r in state["rows"] if kinds is None or r["metadata"].get("type", "text") in kinds]
        return [r for r in rows[:k] if r["score"] >= min_score]

    monkeypatch.setattr(retriever_mod, "run_read", lambda fn: fn(None))
    monkeypatch.setattr(dbg, "run_read", lambda fn: fn(None))
    monkeypatch.setattr(repo, "get_chatbot_website_ids", website_ids)
    monkeypatch.setattr(repo, "similarity_search", search)
    monkeypatch.setattr(retriever_mod, "get_embeddings", lambda: "emb")
    monkeypatch.setattr(dbg, "get_embeddings", lambda: "emb")

    def embed(text, emb=None):
        calls["embed"].append(text)
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(retriever_mod, "embed_query", embed)
    monkeypatch.setattr(dbg, "embed_query", embed)
    # any write would be a bug
    import chat_with_website.db.connection as conn_mod
    import chat_with_website.rag.chain as chain_mod

    def boom(*a, **k):
        raise AssertionError("the debugger must not write")

    monkeypatch.setattr(conn_mod, "get_conn", boom)
    monkeypatch.setattr(chain_mod, "persist_turn", boom)
    monkeypatch.setattr(settings, "score_threshold", 0.25)
    return calls, state


# ------------------------------------------------------------ retriever path
def test_search_chunks_forwards_scope_k_threshold_and_kinds(fake_search):
    calls, _ = fake_search
    docs = retriever_mod.search_chunks(7, "what services?", k=2, kinds=["text"])
    assert calls["website_ids"] == [7]
    assert calls["search"] == [{"website_ids": [1, 2], "k": 2, "min_score": 0.25, "kinds": ["text"]}]
    assert [d.metadata["score"] for d in docs] == [0.81, 0.4]
    assert docs[0].metadata["db_id"] == 101 and docs[0].metadata["chunk_id"] == "11:1"
    assert docs[0].page_content == "passage 1 about services"


def test_search_chunks_timed_measures_and_can_skip_embedding(fake_search):
    calls, _ = fake_search
    docs, t = retriever_mod.search_chunks_timed(7, "q", k=3, min_score=0.0)
    assert len(docs) == 3 and t["embed_ms"] >= 0 and t["search_ms"] >= 0 and calls["embed"] == ["q"]
    docs, t = retriever_mod.search_chunks_timed(7, "q", k=3, min_score=0.0, vector=[0.0, 0.0, 0.0])
    assert t["embed_ms"] == 0.0 and calls["embed"] == ["q"]  # not embedded again


def test_search_chunks_without_websites_returns_nothing_and_never_embeds(fake_search):
    calls, state = fake_search
    state["website_ids"] = []
    assert retriever_mod.search_chunks(7, "q") == []
    assert calls["embed"] == [] and calls["search"] == []


# ------------------------------------------------------------- retrieval test
def test_run_retrieval_test_shows_every_row_and_flags_dropped_ones(fake_search):
    calls, _ = fake_search
    t = dbg.run_retrieval_test(7, "  What services do you offer?  ", k=5)
    assert t.question == "What services do you offer?"
    assert calls["search"] == [{"website_ids": [1, 2], "k": 5, "min_score": -1.0, "kinds": ["text"]}]
    assert [c.rank for c in t.chunks] == [1, 2, 3]
    assert [c.kept for c in t.chunks] == [True, True, False]  # 0.10 < 0.25
    assert t.kept_count == 2 and len(t.kept) == 2 and t.distinct_pages == 2
    assert (t.min_score, t.max_score) == (0.10, 0.81)
    assert t.context_chars > 0 and t.est_tokens == t.context_chars // dbg.CHARS_PER_TOKEN
    assert t.total_ms >= t.search_ms >= 0 and t.embed_ms >= 0
    assert not t.greeting and not t.image_question and t.skipped_reason is None
    c = t.chunks[0]
    assert (c.db_id, c.chunk_id, c.url, c.title, c.website_id, c.kind) == (101, "11:1", "https://a.example/p1", "Page", 1, "text")


def test_missing_metadata_is_reported_not_guessed(fake_search):
    _, state = fake_search
    state["rows"] = [{"id": 5, "content": "x", "chunk_index": 0, "metadata": {}, "url": None, "title": None,
                      "website_id": 1, "score": 0.9}]
    t = dbg.run_retrieval_test(7, "q", k=3)
    c = t.chunks[0]
    assert c.title is None and c.url is None and c.kind == "text"
    assert c.chunk_id == "?:0"  # the retriever's fallback when the JSONB has no chunk_id
    assert t.distinct_pages == 0  # no url -> no page counted


def test_no_websites_gives_a_reason_and_no_embedding_call(fake_search):
    calls, state = fake_search
    state["website_ids"] = []
    t = dbg.run_retrieval_test(7, "q", k=5)
    assert t.skipped_reason and t.chunks == [] and calls["embed"] == []


def test_image_question_runs_a_second_search_with_images_first(fake_search):
    calls, state = fake_search
    state["rows"] = [_row(1, 0.9), _row(2, 0.5, kind="image")]
    t = dbg.run_retrieval_test(7, "Do you have a photo of the office?", k=4)
    assert t.image_question
    assert [c["kinds"] for c in calls["search"]] == [["text"], ["image"]]
    assert [c.kind for c in t.chunks] == ["image", "text"]
    assert calls["embed"] == ["Do you have a photo of the office?"]  # embedded once, vector reused
    assert t.chunks[0].document.metadata["image_url"] == "https://a/img2.png"


def test_greeting_is_flagged_but_still_searched(fake_search):
    t = dbg.run_retrieval_test(7, "hello!", k=3)
    assert t.greeting and len(t.chunks) == 3


# ---------------------------------------------------------------- compare
def test_compare_top_k_embeds_once_and_searches_once_per_k(fake_search):
    calls, state = fake_search
    state["rows"] = [_row(i, 1 - i / 20) for i in range(1, 12)]
    tests = dbg.compare_top_k(7, "q", ks=(10, 3, 5, 8, 5))
    assert [t.k for t in tests] == [3, 5, 8, 10]
    assert calls["embed"] == ["q"]
    assert [c["k"] for c in calls["search"]] == [3, 5, 8, 10]
    assert [len(t.chunks) for t in tests] == [3, 5, 8, 10]
    assert len({t.embed_ms for t in tests}) == 1  # the shared, measured value
    rows = dbg.compare_rows(tests)
    assert [r["top_k"] for r in rows] == [3, 5, 8, 10]
    assert rows[3]["chunks returned"] == 10 and rows[3]["kept (≥ threshold)"] == 10 and rows[3]["dropped"] == 0
    assert rows[0]["lowest score"] == 0.85 and rows[0]["highest score"] == 0.95
    assert dbg.compare_rows([dbg.RetrievalTest(7, "q", 3, 0.25, [], False, False)])[0]["lowest score"] == dbg.NOT_AVAILABLE


# -------------------------------------------------------------- generation
def _chain(responses, docs=DOCS):
    return RagChain(FakeRetriever(docs=docs), FakeListChatModel(responses=responses), chatbot_name="Acme Support")


def _test_with(docs, threshold=0.25, question="What do you offer?"):
    t = dbg.RetrievalTest(7, question, 5, threshold, [1], image_question=False, greeting=dbg.is_greeting(question))
    t.chunks = [dbg.chunk_from_document(i, d, threshold) for i, d in enumerate(docs, start=1)]
    return t


def test_generate_test_answer_uses_the_kept_chunks_and_measures(fake_search):
    t = _test_with(DOCS)
    a = dbg.generate_test_answer(_chain(["We offer web development [1]."]), t)
    assert a.llm_called and a.grounded and a.gen_ms >= 0 and a.note is None
    assert a.answer == "We offer web development [1]."
    assert [s.cited for s in a.sources] == [True, False]
    assert dbg.cited_ranks(a) == {1}
    assert a.context_chars == len(dbg.format_context(DOCS))
    assert a.model_tokens is None or a.model_tokens > 0


def test_generate_test_answer_without_kept_chunks_makes_no_model_call(fake_search):
    low = [d.model_copy(update={"metadata": {**d.metadata, "score": 0.05}}) for d in DOCS]
    t = _test_with(low)
    assert t.kept == []
    a = dbg.generate_test_answer(_chain([]), t)  # FakeListChatModel with no responses would fail if called
    assert not a.llm_called and not a.grounded and "Acme Support" in a.answer and a.note


def test_generate_test_answer_for_a_greeting_makes_no_model_call(fake_search):
    t = _test_with(DOCS, question="hi there")
    assert t.greeting
    a = dbg.generate_test_answer(_chain([]), t)
    assert not a.llm_called and "Acme Support" in a.answer and a.sources == []


def test_metric_label_mentions_precision():
    assert "32-bit" in dbg.metric_label(1536) and "higher is better" in dbg.metric_label(1536)
    assert "16-bit" in dbg.metric_label(3072)
