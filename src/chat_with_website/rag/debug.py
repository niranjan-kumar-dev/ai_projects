"""RAG Retrieval Debugger: run the chatbot's own retrieval for a test question and report what
came back, with real timings. Optionally generate a test answer through the chatbot's chain.

Rules this module keeps:
- **One retrieval path.** `search_chunks_timed` (rag/retriever.py) is the function the chatbot's
  retriever uses; the debugger calls it with `min_score=-1` so the k rows the database returned
  are all visible, and flags the ones the chatbot would drop (`score < settings.score_threshold`,
  which `similarity_search` applies *after* the SQL `LIMIT k`). Nothing is re-sorted: the order
  shown is the `[1]..[k]` numbering the model sees.
- **No writes.** Nothing here touches `get_conn`, `persist_turn` or `suggest_followups`; test
  questions are never stored.
- **No fabricated numbers.** Timings come from `perf_counter`; token counts are an estimate
  unless the model exposes a tokenizer; missing metadata is reported as `NOT_AVAILABLE`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Sequence

from langchain_core.documents import Document

from chat_with_website.config import settings
from chat_with_website.db import repository as repo
from chat_with_website.db.connection import run_read
from chat_with_website.ingestion.embedder import embed_query, get_embeddings
from chat_with_website.rag.chain import Prepared, RagAnswer, Source, format_context
from chat_with_website.rag.prompts import greeting_reply, is_greeting, is_image_question, out_of_scope_reply
from chat_with_website.rag.retriever import search_chunks_timed

log = logging.getLogger(__name__)

NOT_AVAILABLE = "Not available"
DEFAULT_KS = (3, 5, 8, 10)
# Cosine similarity can be negative; -1 keeps every row the SQL LIMIT returned (0 would hide some).
MIN_SCORE_ALL = -1.0
CHARS_PER_TOKEN = 4  # rough English rule of thumb, labelled as such in the UI


def metric_label(dim: int) -> str:
    """How the score is computed, so the page can say which direction is 'better'."""
    precision = "halfvec, 16-bit" if dim > repo.HALFVEC_THRESHOLD else "vector, 32-bit"
    return f"cosine similarity = 1 − pgvector `<=>` distance ({precision}); higher is better, 1.0 = identical"


# ---------------------------------------------------------------------- data
@dataclass(frozen=True)
class RetrievedChunk:
    rank: int  # 1-based, database return order = the chain's [n] numbering
    document: Document
    score: float
    kept: bool  # score >= threshold -> the chatbot passes it to the model
    kind: str
    db_id: int | None
    chunk_id: str | None
    url: str | None
    title: str | None
    website_id: int | None
    chunk_index: int | None
    chars: int

    @property
    def text(self) -> str:
        return self.document.page_content


def chunk_from_document(rank: int, doc: Document, threshold: float) -> RetrievedChunk:
    m = doc.metadata
    score = float(m.get("score", 0.0))
    return RetrievedChunk(
        rank=rank, document=doc, score=score, kept=score >= threshold,
        kind=str(m.get("type") or "text"),
        db_id=m.get("db_id"), chunk_id=(str(m["chunk_id"]) if m.get("chunk_id") is not None else None),
        url=m.get("url") or None, title=(m.get("title") or None), website_id=m.get("website_id"),
        chunk_index=m.get("chunk_index"), chars=len(doc.page_content),
    )


@dataclass
class RetrievalTest:
    chatbot_id: int
    question: str
    k: int
    threshold: float
    website_ids: list[int]
    image_question: bool
    greeting: bool  # the Chat page would not retrieve for this; shown for information only
    chunks: list[RetrievedChunk] = field(default_factory=list)
    embed_ms: float = 0.0
    search_ms: float = 0.0  # one or two SQL round trips (text, plus images for picture questions)
    total_ms: float = 0.0
    context_chars: int = 0  # len(format_context(kept)): the exact string the model would get
    est_tokens: int = 0  # context_chars // CHARS_PER_TOKEN
    distinct_pages: int = 0
    skipped_reason: str | None = None

    @property
    def kept(self) -> list[Document]:
        return [c.document for c in self.chunks if c.kept]

    @property
    def kept_count(self) -> int:
        return sum(1 for c in self.chunks if c.kept)

    @property
    def min_score(self) -> float | None:
        return min((c.score for c in self.chunks), default=None)

    @property
    def max_score(self) -> float | None:
        return max((c.score for c in self.chunks), default=None)


@dataclass
class AnswerTest:
    answer: str
    sources: list[Source]
    grounded: bool
    gen_ms: float
    llm_called: bool
    context_chars: int
    model_tokens: int | None  # from the model's own tokenizer when it has one, else None
    note: str | None = None  # why the LLM was not called


# ------------------------------------------------------------------ retrieval
def _finish(test: RetrievalTest, docs: list[Document], timings: dict[str, float], t_start: float) -> RetrievalTest:
    test.chunks = [chunk_from_document(i, d, test.threshold) for i, d in enumerate(docs, start=1)]
    test.embed_ms = round(timings.get("embed_ms", 0.0), 2)
    test.search_ms = round(timings.get("search_ms", 0.0), 2)
    test.total_ms = round((perf_counter() - t_start) * 1000, 2)
    kept = test.kept
    test.context_chars = len(format_context(kept)) if kept else 0
    test.est_tokens = test.context_chars // CHARS_PER_TOKEN
    test.distinct_pages = len({d.metadata.get("url") for d in kept if d.metadata.get("url")})
    return test


def run_retrieval_test(
    chatbot_id: int, question: str, k: int, *, threshold: float | None = None, vector: list[float] | None = None
) -> RetrievalTest:
    """The chatbot's retrieval for `question` at top_k `k`, every returned row visible.

    Mirrors `RagChain.retrieve`: text chunks only, plus a separate image search (images first)
    when the question is about pictures. `vector` skips the embedding call (used by
    `compare_top_k`, which embeds once for all k values).
    """
    t_start = perf_counter()
    threshold = settings.score_threshold if threshold is None else threshold
    question = (question or "").strip()
    test = RetrievalTest(
        chatbot_id=chatbot_id, question=question, k=int(k), threshold=threshold, website_ids=[],
        image_question=is_image_question(question), greeting=is_greeting(question),
    )
    test.website_ids = list(run_read(lambda conn: repo.get_chatbot_website_ids(conn, chatbot_id)))
    if not test.website_ids:
        test.skipped_reason = "This chatbot has no websites attached, so the chatbot retrieves nothing."
        test.total_ms = round((perf_counter() - t_start) * 1000, 2)
        return test

    embed_ms = 0.0
    if vector is None:  # embed once here so a picture question (two searches) pays for one embedding
        embeddings = get_embeddings()  # model load is not "embed time"
        t0 = perf_counter()
        vector = embed_query(question, embeddings)
        embed_ms = (perf_counter() - t0) * 1000
    text_docs, timings = search_chunks_timed(chatbot_id, question, k=k, min_score=MIN_SCORE_ALL, kinds=["text"], vector=vector)
    docs = text_docs
    search_ms = timings["search_ms"]
    if test.image_question:
        image_docs, t2 = search_chunks_timed(chatbot_id, question, k=k, min_score=MIN_SCORE_ALL, kinds=["image"], vector=vector)
        search_ms += t2["search_ms"]
        docs = image_docs + text_docs
    return _finish(test, docs, {"embed_ms": embed_ms, "search_ms": search_ms}, t_start)


def compare_top_k(chatbot_id: int, question: str, ks: Sequence[int] = DEFAULT_KS) -> list[RetrievalTest]:
    """One real search per k (never a slice of the largest), with the query embedded once."""
    ks = sorted({int(k) for k in ks if int(k) > 0})
    website_ids = run_read(lambda conn: repo.get_chatbot_website_ids(conn, chatbot_id))
    if not website_ids:
        return [run_retrieval_test(chatbot_id, question, k) for k in ks]
    embeddings = get_embeddings()
    t0 = perf_counter()
    vector = embed_query((question or "").strip(), embeddings)
    embed_ms = round((perf_counter() - t0) * 1000, 2)
    tests = []
    for k in ks:
        t = run_retrieval_test(chatbot_id, question, k, vector=vector)
        t.embed_ms = embed_ms  # shared, measured once
        tests.append(t)
    return tests


def compare_rows(tests: Sequence[RetrievalTest]) -> list[dict]:
    """Table rows for the top_k comparison (pure)."""
    rows = []
    for t in tests:
        rows.append({
            "top_k": t.k,
            "chunks returned": len(t.chunks),
            "kept (≥ threshold)": t.kept_count,
            "dropped": len(t.chunks) - t.kept_count,
            "distinct pages": t.distinct_pages,
            "context chars": t.context_chars,
            "≈ tokens": t.est_tokens,
            "SQL ms": t.search_ms,
            "lowest score": round(t.min_score, 4) if t.min_score is not None else NOT_AVAILABLE,
            "highest score": round(t.max_score, 4) if t.max_score is not None else NOT_AVAILABLE,
        })
    return rows


# ----------------------------------------------------------------- generation
def generate_test_answer(chain, test: RetrievalTest) -> AnswerTest:
    """Answer `test.question` from the kept chunks exactly as the Chat page would for a first
    question (empty history: no question rewriting, one model call). Nothing is persisted."""
    name = getattr(chain, "chatbot_name", "") or ""
    if test.greeting:
        return AnswerTest(answer=greeting_reply(name), sources=[], grounded=False, gen_ms=0.0, llm_called=False,
                          context_chars=0, model_tokens=None,
                          note="A bare greeting gets the canned welcome; the chatbot makes no retrieval and no model call.")
    kept = test.kept
    if not kept:
        return AnswerTest(answer=out_of_scope_reply(name), sources=[], grounded=False, gen_ms=0.0, llm_called=False,
                          context_chars=0, model_tokens=None,
                          note="No chunk reached the score threshold, so the chatbot replies out-of-scope without calling the model.")
    p = Prepared(question=test.question, history=[], standalone=test.question, documents=list(kept),
                 image_question=test.image_question)
    context = format_context(kept)
    model_tokens = None
    llm = getattr(chain, "llm", None)
    if llm is not None and hasattr(llm, "get_num_tokens"):
        try:
            model_tokens = int(llm.get_num_tokens(context))
        except Exception:  # local models without a tokenizer: keep the estimate
            model_tokens = None
    t0 = perf_counter()
    raw = "".join(chain.stream_answer(p))
    result: RagAnswer = chain.finalize(p, raw)
    gen_ms = round((perf_counter() - t0) * 1000, 2)
    return AnswerTest(answer=result.answer, sources=list(result.sources), grounded=result.grounded, gen_ms=gen_ms,
                      llm_called=True, context_chars=len(context), model_tokens=model_tokens)


def cited_ranks(answer: AnswerTest) -> set[int]:
    """Which context numbers `[n]` the answer cited (n is the rank among the *kept* chunks)."""
    return {s.n for s in answer.sources if s.cited}
