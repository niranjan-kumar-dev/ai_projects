"""LangChain retriever backed by our pgvector `chunks` table.

A *retriever* is anything with `invoke(query) -> list[Document]`.  Ours embeds
the query, runs the cosine search in PostgreSQL and - crucially - restricts the
search to the websites attached to one chatbot.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from chat_with_website.config import settings
from chat_with_website.db import repository as repo
from chat_with_website.db.connection import run_read
from chat_with_website.ingestion.embedder import embed_query, get_embeddings

log = logging.getLogger(__name__)


def row_to_document(row: dict[str, Any]) -> Document:
    meta = dict(row.get("metadata") or {})
    meta.update(
        {
            "url": row["url"],
            "title": row["title"],
            "score": round(float(row["score"]), 4),
            "chunk_id": meta.get("chunk_id", f"{row.get('page_id', '?')}:{row['chunk_index']}"),
            "chunk_index": row["chunk_index"],
            "website_id": row["website_id"],
            "db_id": row["id"],
        }
    )
    return Document(page_content=row["content"], metadata=meta)


_row_to_document = row_to_document  # old private name, kept for callers/tests


def search_chunks_timed(
    chatbot_id: int,
    query: str,
    k: int | None = None,
    min_score: float | None = None,
    kinds: list[str] | None = None,
    *,
    vector: list[float] | None = None,
) -> tuple[list[Document], dict[str, float]]:
    """`search_chunks` plus how long each step took: `{"embed_ms", "search_ms"}`.

    The chatbot and the RAG Debugger page share this one code path, so what the debugger shows
    is what the chatbot retrieves. `vector` lets a caller embed the query once and run several
    searches (the debugger's top_k comparison); `embed_ms` is then 0.
    """
    k = k or settings.top_k
    min_score = settings.score_threshold if min_score is None else min_score
    website_ids = run_read(lambda conn: repo.get_chatbot_website_ids(conn, chatbot_id))
    if not website_ids:
        log.warning("Chatbot %s has no websites attached", chatbot_id)
        return [], {"embed_ms": 0.0, "search_ms": 0.0}
    # Embed *outside* the transaction: no pooled connection is held during the
    # (slow, remote) embedding call, and a DB retry never re-embeds.
    embed_ms = 0.0
    if vector is None:
        embeddings = get_embeddings()  # model load (HuggingFace: seconds, once) is not "embed time"
        t0 = perf_counter()
        vector = embed_query(query, embeddings)
        embed_ms = (perf_counter() - t0) * 1000
    t1 = perf_counter()
    rows = run_read(
        lambda conn: repo.similarity_search(conn, vector, website_ids, k=k, min_score=min_score, kinds=kinds)
    )
    search_ms = (perf_counter() - t1) * 1000
    docs = [row_to_document(r) for r in rows]
    log.debug("Retrieved %d chunk(s) for %r (chatbot %s)", len(docs), query, chatbot_id)
    return docs, {"embed_ms": embed_ms, "search_ms": search_ms}


def search_chunks(
    chatbot_id: int,
    query: str,
    k: int | None = None,
    min_score: float | None = None,
    kinds: list[str] | None = None,
) -> list[Document]:
    """Plain-function version of the retriever (used by the CLI, the chain's retriever and the
    RAG Debugger). `kinds` limits the search to chunk types ("text", "image"); None = all."""
    return search_chunks_timed(chatbot_id, query, k=k, min_score=min_score, kinds=kinds)[0]


class PgVectorChatbotRetriever(BaseRetriever):
    """Retriever scoped to one chatbot's websites."""

    chatbot_id: int
    k: int = Field(default_factory=lambda: settings.top_k)
    min_score: float = Field(default_factory=lambda: settings.score_threshold)
    kinds: list[str] | None = None  # None = every chunk type

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        return search_chunks(self.chatbot_id, query, k=self.k, min_score=self.min_score, kinds=self.kinds)
