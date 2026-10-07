"""The RAG chain: history-aware question rewriting -> retrieval -> grounded answer.

Built with LCEL (LangChain Expression Language): small runnables piped together.

    standalone = CONDENSE_PROMPT | llm | StrOutputParser
    docs       = retriever.invoke(standalone)
    answer     = ANSWER_PROMPT(context=format(docs)) | llm | StrOutputParser
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

from chat_with_website.config import settings
from chat_with_website.db import repository as repo
from chat_with_website.db.connection import get_conn
from chat_with_website.rag.llm import resolve_llm_for_chatbot
from chat_with_website.rag.prompts import ANSWER_PROMPT, CONDENSE_QUESTION_PROMPT, NOT_ENOUGH_INFO
from chat_with_website.rag.retriever import PgVectorChatbotRetriever

log = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Source:
    n: int
    url: str
    title: str
    score: float
    cited: bool = False

    def to_dict(self) -> dict:
        return {"n": self.n, "url": self.url, "title": self.title, "score": self.score, "cited": self.cited}


@dataclass
class RagAnswer:
    answer: str
    sources: list[Source] = field(default_factory=list)
    standalone_question: str = ""
    documents: list[Document] = field(default_factory=list)
    grounded: bool = True  # False when we answered "not enough information"

    @property
    def cited_sources(self) -> list[Source]:
        return [s for s in self.sources if s.cited]


# --------------------------------------------------------------------- helpers
def trim_history(history: Sequence[BaseMessage], turns: int | None = None) -> list[BaseMessage]:
    """Keep only the last N question/answer pairs so the prompt stays small."""
    turns = settings.history_turns if turns is None else turns
    return list(history)[-(2 * turns):] if turns > 0 else []


def format_context(docs: Sequence[Document]) -> str:
    """Number the passages so the model can cite them as [1], [2]..."""
    parts = []
    for i, d in enumerate(docs, start=1):
        title = d.metadata.get("title") or d.metadata.get("url", "")
        parts.append(f"[{i}] {title} ({d.metadata.get('url', '')})\n{d.page_content}")
    return "\n\n".join(parts)


def build_sources(docs: Sequence[Document], answer: str) -> list[Source]:
    """One source per passage, flagged `cited` when its [n] appears in the answer.

    Passages are numbered as the model saw them, so the UI can show
    "[2] Title - url"; duplicates of the same URL are kept as separate numbers
    because the model may cite either one.
    """
    cited_numbers = {int(n) for n in _CITATION_RE.findall(answer)}
    sources = []
    for i, d in enumerate(docs, start=1):
        sources.append(
            Source(
                n=i,
                url=d.metadata.get("url", ""),
                title=d.metadata.get("title") or d.metadata.get("url", ""),
                score=float(d.metadata.get("score", 0.0)),
                cited=i in cited_numbers,
            )
        )
    return sources


def history_from_rows(rows: Sequence[dict]) -> list[BaseMessage]:
    """Convert `messages` table rows into LangChain messages."""
    out: list[BaseMessage] = []
    for r in rows:
        out.append(HumanMessage(content=r["content"]) if r["role"] == "user" else AIMessage(content=r["content"]))
    return out


# ----------------------------------------------------------------------- chain
class RagChain:
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseChatModel,
        system_prompt: str | None = None,
        history_turns: int | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.custom_instructions = (system_prompt or "").strip()
        self.history_turns = history_turns
        self._condense = CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
        self._answer = ANSWER_PROMPT | llm | StrOutputParser()

    # step 1
    def condense_question(self, question: str, history: Sequence[BaseMessage]) -> str:
        if not history:
            return question
        try:
            rewritten = self._condense.invoke({"question": question, "chat_history": list(history)}).strip()
        except Exception as exc:  # fall back to the raw question rather than failing the turn
            log.warning("Question rewriting failed (%s); using the original question", exc)
            return question
        rewritten = rewritten.strip().strip('"').splitlines()[0] if rewritten else question
        log.info("Standalone question: %r", rewritten)
        return rewritten or question

    # steps 2-4
    def invoke(self, question: str, history: Sequence[BaseMessage] = ()) -> RagAnswer:
        history = trim_history(history, self.history_turns)
        standalone = self.condense_question(question, history)
        docs = self.retriever.invoke(standalone)

        if not docs:
            return RagAnswer(
                answer=NOT_ENOUGH_INFO, sources=[], standalone_question=standalone, documents=[], grounded=False
            )

        answer = self._answer.invoke(
            {
                "context": format_context(docs),
                "question": question,
                "chat_history": history,
                "custom_instructions": self.custom_instructions,
                "not_enough_info": NOT_ENOUGH_INFO,
            }
        ).strip()
        grounded = NOT_ENOUGH_INFO.lower() not in answer.lower()
        sources = build_sources(docs, answer) if grounded else []
        return RagAnswer(answer=answer, sources=sources, standalone_question=standalone, documents=list(docs), grounded=grounded)


# ------------------------------------------------------------- public helpers
def build_chain_for_chatbot(chatbot_id: int) -> tuple[RagChain, dict]:
    with get_conn() as conn:
        chatbot = repo.get_chatbot(conn, chatbot_id)
    if not chatbot:
        raise ValueError(f"Chatbot {chatbot_id} does not exist")
    retriever = PgVectorChatbotRetriever(chatbot_id=chatbot_id, k=chatbot.get("top_k") or settings.top_k)
    llm = resolve_llm_for_chatbot(chatbot)
    return RagChain(retriever, llm, system_prompt=chatbot.get("system_prompt")), chatbot


def answer_question(
    chatbot_id: int,
    question: str,
    history: Sequence[BaseMessage] = (),
    conversation_id: str | None = None,
    chain: RagChain | None = None,
) -> RagAnswer:
    """Answer one question; if `conversation_id` is given, persist both messages."""
    if chain is None:
        chain, _ = build_chain_for_chatbot(chatbot_id)
    result = chain.invoke(question, history)

    if conversation_id:
        with get_conn() as conn:
            repo.create_conversation(conn, conversation_id, chatbot_id, title=question[:80])
            repo.add_message(conn, conversation_id, "user", question)
            repo.add_message(conn, conversation_id, "assistant", result.answer, [s.to_dict() for s in result.sources])
    return result


def new_conversation_id() -> str:
    return str(uuid.uuid4())
