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
from typing import Iterator, Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

from chat_with_website.config import settings
from chat_with_website.db import repository as repo
from chat_with_website.db.connection import get_conn, run_read
from chat_with_website.rag.citations import normalize_citations
from chat_with_website.rag.followups import clean_followups, parse_followups
from chat_with_website.rag.llm import resolve_llm_for_chatbot
from chat_with_website.rag.prompts import (
    ANSWER_PROMPT,
    CONDENSE_QUESTION_PROMPT,
    FOLLOWUP_PROMPT,
    NOT_ENOUGH_INFO,
    greeting_reply,
    is_greeting,
    is_image_question,
    out_of_scope_reply,
)
from chat_with_website.rag.retriever import PgVectorChatbotRetriever

log = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(\s*<?([^)\s>]+)>?[^)]*\)")


@dataclass
class Source:
    n: int
    url: str
    title: str
    score: float
    cited: bool = False
    kind: str = "text"  # "text" | "image"
    image_url: str | None = None
    snippet: str | None = None  # the retrieved text the model saw (trimmed), shown as evidence
    chunk_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "url": self.url,
            "title": self.title,
            "score": self.score,
            "cited": self.cited,
            "kind": self.kind,
            "image_url": self.image_url,
            "snippet": self.snippet,
            "chunk_id": self.chunk_id,
        }


SNIPPET_CHARS = 500  # how much of a passage is stored with the answer as its excerpt


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


@dataclass
class Prepared:
    """Everything needed to generate an answer, computed before the model call."""

    question: str
    history: list[BaseMessage]
    standalone: str
    documents: list[Document] = field(default_factory=list)
    image_question: bool = False
    ready: RagAnswer | None = None  # set when no model call is needed (greeting, nothing retrieved)


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
        if is_image_doc(d):
            parts.append(
                f"[{i}] IMAGE on {title} ({d.metadata.get('url', '')})\n"
                f"{d.page_content}\nImage URL: {d.metadata['image_url']}"
            )
        else:
            parts.append(f"[{i}] {title} ({d.metadata.get('url', '')})\n{d.page_content}")
    return "\n\n".join(parts)


def is_image_doc(doc: Document) -> bool:
    return doc.metadata.get("type") == "image" and bool(doc.metadata.get("image_url"))


def build_sources(docs: Sequence[Document], answer: str) -> list[Source]:
    """One source per passage, flagged `cited` when its [n] appears in the answer.

    Passages are numbered as the model saw them, so the UI can show
    "[2] Title - url"; duplicates of the same URL are kept as separate numbers
    because the model may cite either one.
    """
    cited_numbers = {int(n) for n in _CITATION_RE.findall(answer)}
    sources = []
    for i, d in enumerate(docs, start=1):
        image = is_image_doc(d)
        text = " ".join((d.page_content or "").split())
        sources.append(
            Source(
                n=i,
                url=d.metadata.get("url", ""),
                title=d.metadata.get("title") or d.metadata.get("url", ""),
                score=float(d.metadata.get("score", 0.0)),
                cited=i in cited_numbers,
                kind="image" if image else "text",
                image_url=d.metadata.get("image_url") if image else None,
                snippet=(text[:SNIPPET_CHARS] + ("…" if len(text) > SNIPPET_CHARS else "")) if text and not image else None,
                chunk_id=str(d.metadata["chunk_id"]) if d.metadata.get("chunk_id") is not None else None,
            )
        )
    return sources


def drop_unknown_images(answer: str, docs: Sequence[Document]) -> str:
    """Remove markdown images whose URL was not supplied by a retrieved image chunk.

    Models sometimes "helpfully" invent a plausible URL such as /images/logo.png
    when asked for a picture.  That renders as a broken image, so we keep only
    URLs we actually retrieved and replace the rest with their alt text.
    """
    allowed = {d.metadata["image_url"] for d in docs if is_image_doc(d)}

    def _keep_or_strip(m: re.Match) -> str:
        url = m.group(2).strip()
        if url in allowed:
            return m.group(0)
        log.warning("Dropping image URL not present in retrieved chunks: %s", url)
        return m.group(1).strip()

    return _MD_IMAGE_RE.sub(_keep_or_strip, answer).strip()


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
        chatbot_name: str | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.custom_instructions = (system_prompt or "").strip()
        self.history_turns = history_turns
        self.chatbot_name = (chatbot_name or "").strip()
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

    # step 2
    def retrieve(self, standalone: str, image_question: bool = False) -> list[Document]:
        """Vector search with text and images kept apart.

        Image chunks are short ("Image: Acme logo. Section: site logo") and score close to
        almost any generic question, so a mixed search lets them fill the whole top-k and
        the model is left with no text to answer from.  Therefore:
        - normal question  -> text chunks only;
        - picture question -> k image chunks *and* k text chunks, images first.
        Retrievers without a `kinds` field (e.g. test fakes) are used as-is.
        """
        retriever = self.retriever
        if not hasattr(retriever, "kinds"):
            docs = list(retriever.invoke(standalone))
            if image_question:
                docs.sort(key=lambda d: not is_image_doc(d))  # stable: keeps score order within each group
            return docs

        text_docs = list(retriever.model_copy(update={"kinds": ["text"]}).invoke(standalone))
        if not image_question:
            return text_docs
        image_docs = list(retriever.model_copy(update={"kinds": ["image"]}).invoke(standalone))
        return image_docs + text_docs

    # The answer is produced in three phases so the UI can stream tokens:
    #   prepare()  -> greeting check, history trim, question rewrite, retrieval  (DB, 0-1 LLM calls)
    #   stream_answer() -> the single answer-model call, yielding text chunks
    #   finalize() -> post-processing of the full text (image guard, out-of-scope reply, sources)
    # invoke() composes them for callers that do not stream (CLI, tests).

    # steps 0-2
    def prepare(self, question: str, history: Sequence[BaseMessage] = ()) -> Prepared:
        # step 0: a bare greeting gets a canned welcome - no retrieval, no LLM call
        if is_greeting(question):
            ready = RagAnswer(answer=greeting_reply(self.chatbot_name), standalone_question=question, grounded=False)
            return Prepared(question=question, history=[], standalone=question, ready=ready)

        history = trim_history(history, self.history_turns)
        standalone = self.condense_question(question, history)
        image_question = is_image_question(question)
        docs = self.retrieve(standalone, image_question=image_question)
        p = Prepared(question=question, history=history, standalone=standalone, documents=docs, image_question=image_question)
        if not docs:
            p.ready = RagAnswer(
                answer=out_of_scope_reply(self.chatbot_name), sources=[], standalone_question=standalone,
                documents=[], grounded=False,
            )
        return p

    def _answer_inputs(self, p: Prepared) -> dict:
        """The exact prompt variables of the answer call (shared by streaming and one-shot)."""
        return {
            "context": format_context(p.documents),
            "question": p.question,
            "chat_history": p.history,
            "custom_instructions": self.custom_instructions,
            "not_enough_info": NOT_ENOUGH_INFO,
        }

    # step 3
    def stream_answer(self, p: Prepared) -> Iterator[str]:
        """Yield the model's answer as text chunks (exactly one model call)."""
        yield from self._answer.stream(self._answer_inputs(p))

    def answer_once(self, p: Prepared) -> tuple[str, dict | None]:
        """Non-streaming twin of `stream_answer`: same prompt, same model, one call.

        Returns the text and the provider's `usage_metadata` (token counts) when it reports
        any - the Chat page streams and never sees usage; the evaluation page needs it. `text`
        is used instead of `content` because some providers return a list of content blocks.
        """
        msg = (ANSWER_PROMPT | self.llm).invoke(self._answer_inputs(p))
        text = msg.text if isinstance(getattr(msg, "text", None), str) else str(msg.content)
        usage = getattr(msg, "usage_metadata", None)
        return text, (dict(usage) if usage else None)

    # step 4
    def finalize(self, p: Prepared, raw_answer: str) -> RagAnswer:
        """Turn the model's full text into the user-facing answer + sources."""
        docs = p.documents
        answer = drop_unknown_images(raw_answer.strip(), docs)
        grounded = NOT_ENOUGH_INFO.lower() not in answer.lower()
        if not grounded:
            # The model's sentinel is for code; the user gets a friendly, chatbot-named reply.
            # The prompt allows one extra sentence about related content - keep it (minus
            # citation markers, since sources are not shown for ungrounded answers).
            log.info("Model declined (out of scope): %r", answer)
            hint = _CITATION_RE.sub("", re.sub(re.escape(NOT_ENOUGH_INFO), "", answer, flags=re.I))
            hint = re.sub(r"\s+([.,;:!?])", r"\1", hint).strip()
            answer = out_of_scope_reply(self.chatbot_name) + (f"\n\n{hint}" if hint else "")
        else:
            # Only numbers that exist in the context survive; "[1, 3]" / "[1-3]" become "[1][3]" /
            # "[1][2][3]" so the stored text, the links and the Sources panel agree.
            answer, _ = normalize_citations(answer, len(docs))
        sources = build_sources(docs, answer) if grounded else []
        return RagAnswer(answer=answer, sources=sources, standalone_question=p.standalone, documents=list(docs), grounded=grounded)

    # step 5 (optional, after the answer is complete): follow-up question suggestions
    def suggest_followups(self, p: Prepared, result: RagAnswer, history: Sequence[BaseMessage] = ()) -> list[str]:
        """Up to `settings.followup_count` short questions the user might ask next.

        One small non-streaming model call that only sees the passages the answer was built
        from (no new retrieval). Returns [] for ungrounded answers, when disabled, or on any
        failure - suggestions are a convenience and must never affect the answer itself.
        """
        count = settings.followup_count
        if count <= 0 or not result.grounded or not p.documents:
            return []
        asked = [m.content for m in history if isinstance(m, HumanMessage)][-6:] + [p.question]
        try:
            passages = "\n\n".join(
                f"[{i}] {d.metadata.get('title') or d.metadata.get('url', '')}\n{' '.join(d.page_content.split())[:300]}"
                for i, d in enumerate(p.documents, start=1)
            )
            raw = (FOLLOWUP_PROMPT | self.llm | StrOutputParser()).invoke(
                {
                    "count": count,
                    "context": passages,
                    "asked": "; ".join(asked[:-1]) or "(nothing)",
                    "question": p.question,
                    "answer": result.answer[:1500],
                }
            )
        except Exception as exc:  # never let a suggestion failure touch the answer
            log.warning("Follow-up suggestions failed (%s); showing none", exc)
            return []
        suggestions = clean_followups(parse_followups(raw), asked=asked, limit=count)
        log.info("Follow-up suggestions: %s", suggestions)
        return suggestions

    def invoke(self, question: str, history: Sequence[BaseMessage] = ()) -> RagAnswer:
        """Non-streaming convenience: prepare -> stream -> finalize."""
        p = self.prepare(question, history)
        if p.ready is not None:
            return p.ready
        return self.finalize(p, "".join(self.stream_answer(p)))


# ------------------------------------------------------------- public helpers
def build_chain_for_chatbot(chatbot_id: int) -> tuple[RagChain, dict]:
    chatbot = run_read(lambda conn: repo.get_chatbot(conn, chatbot_id))
    if not chatbot:
        raise ValueError(f"Chatbot {chatbot_id} does not exist")
    retriever = PgVectorChatbotRetriever(chatbot_id=chatbot_id, k=chatbot.get("top_k") or settings.top_k)
    llm = resolve_llm_for_chatbot(chatbot)
    chain = RagChain(retriever, llm, system_prompt=chatbot.get("system_prompt"), chatbot_name=chatbot.get("name"))
    return chain, chatbot


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
        persist_turn(conversation_id, chatbot_id, question, result)
    return result


@dataclass
class TurnIds:
    """Ids of the two rows a saved turn produced (the UI attaches edits / feedback to them)."""

    user_id: int
    assistant_id: int


def persist_turn(
    conversation_id: str,
    chatbot_id: int,
    question: str,
    result: RagAnswer,
    *,
    edited_from_message_id: int | None = None,
    suggestions: list[str] | None = None,
) -> TurnIds:
    """Save one completed question/answer pair - called exactly once, after generation.

    `edited_from_message_id` is set when the question is an edited version of a message in the
    parent conversation (the branch itself was created by `repo.fork_conversation`);
    `suggestions` are the follow-up questions generated for this answer.
    Plain INSERTs are not idempotent (the server may have committed before the socket
    dropped), so this write is deliberately *not* retried.
    """
    with get_conn() as conn:
        repo.create_conversation(conn, conversation_id, chatbot_id, title=question[:80])
        user_id = repo.add_message(
            conn, conversation_id, "user", question, edited_from_message_id=edited_from_message_id
        )
        assistant_id = repo.add_message(
            conn, conversation_id, "assistant", result.answer, [s.to_dict() for s in result.sources],
            suggestions=suggestions or None,
        )
        return TurnIds(user_id=user_id, assistant_id=assistant_id)


def new_conversation_id() -> str:
    return str(uuid.uuid4())
