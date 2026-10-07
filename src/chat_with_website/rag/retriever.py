"""LangChain retriever backed by our pgvector `chunks` table.

A *retriever* is anything with `invoke(query) -> list[Document]`.  Ours embeds
the query, runs the cosine search in PostgreSQL and - crucially - restricts the
search to the websites attached to one chatbot.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from chat_with_website.config import settings
from chat_with_website.db import repository as repo
from chat_with_website.db.connection import get_conn
from chat_with_website.ingestion.embedder import embed_query

log = logging.getLogger(__name__)


def _row_to_document(row: dict[str, Any]) -> Document:
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


def search_chunks(
    chatbot_id: int,
    query: str,
    k: int | None = None,
    min_score: float | None = None,
) -> list[Document]:
    """Plain-function version of the retriever (used by the CLI and the UI debug view)."""
    k = k or settings.top_k
    min_score = settings.score_threshold if min_score is None else min_score
    with get_conn() as conn:
        website_ids = repo.get_chatbot_website_ids(conn, chatbot_id)
        if not website_ids:
            log.warning("Chatbot %s has no websites attached", chatbot_id)
            return []
        vector = embed_query(query)
        rows = repo.similarity_search(conn, vector, website_ids, k=k, min_score=min_score)
    docs = [_row_to_document(r) for r in rows]
    log.debug("Retrieved %d chunk(s) for %r (chatbot %s)", len(docs), query, chatbot_id)
    return docs


class PgVectorChatbotRetriever(BaseRetriever):
    """Retriever scoped to one chatbot's websites."""

    chatbot_id: int
    k: int = Field(default_factory=lambda: settings.top_k)
    min_score: float = Field(default_factory=lambda: settings.score_threshold)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        return search_chunks(self.chatbot_id, query, k=self.k, min_score=self.min_score)
