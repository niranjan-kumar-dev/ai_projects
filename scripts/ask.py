"""Chat with a chatbot from the terminal (no UI needed).

Usage (from the project root):

    python scripts/ask.py --chatbot 1                        # interactive chat, new conversation
    python scripts/ask.py --chatbot 1 --conversation <uuid>  # resume a saved conversation
    python scripts/ask.py --chatbot 1 --search-only "how do I get started"   # show retrieved chunks only
    python scripts/ask.py --chatbot 1 -q "What is Python good for?"          # one question, then exit

Inside the chat:  /sources  show sources of the last answer   /history  show the conversation
                  /debug    toggle retrieved-chunk display    /quit     exit
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from chat_with_website.db import repository as repo  # noqa: E402
from chat_with_website.db.connection import get_conn  # noqa: E402
from chat_with_website.logging_config import setup_logging  # noqa: E402
from chat_with_website.rag.chain import (  # noqa: E402
    RagAnswer,
    answer_question,
    build_chain_for_chatbot,
    history_from_rows,
    new_conversation_id,
)
from chat_with_website.rag.llm import describe_llm  # noqa: E402
from chat_with_website.rag.retriever import search_chunks  # noqa: E402

log = logging.getLogger("ask")


def print_sources(result: RagAnswer) -> None:
    if not result.sources:
        print("  (no sources)")
        return
    for s in result.sources:
        mark = "*" if s.cited else " "
        print(f"  {mark}[{s.n}] {s.title}  —  {s.url}  (score {s.score:.2f})")
    print("  * = cited in the answer")


def print_docs(docs) -> None:
    for i, d in enumerate(docs, start=1):
        print(f"\n--- [{i}] score={d.metadata.get('score'):.3f}  {d.metadata.get('url')}")
        print(d.page_content[:600] + ("..." if len(d.page_content) > 600 else ""))


def cmd_search(chatbot_id: int, query: str, k: int | None) -> None:
    docs = search_chunks(chatbot_id, query, k=k, min_score=0.0)
    print(f"\nTop {len(docs)} chunk(s) for: {query!r}")
    print_docs(docs)


def chat_loop(chatbot_id: int, conversation_id: str | None, one_question: str | None, debug: bool) -> None:
    chain, chatbot = build_chain_for_chatbot(chatbot_id)
    history = []
    if conversation_id:
        with get_conn() as conn:
            history = history_from_rows(repo.get_messages(conn, conversation_id))
        print(f"Resumed conversation {conversation_id} with {len(history)} message(s).")
    else:
        conversation_id = new_conversation_id()

    print(f"\nChatbot [{chatbot['id']}] {chatbot['name']}  —  {describe_llm(chatbot)}")
    print(f"Conversation id: {conversation_id}  (resume later with --conversation {conversation_id})")
    print("Type your question, or /quit.\n")

    last: RagAnswer | None = None
    while True:
        try:
            question = one_question or input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question == "/quit":
            break
        if question == "/sources":
            print_sources(last) if last else print("  ask something first")
            continue
        if question == "/history":
            for m in history:
                print(f"  {m.type:>5}: {m.content[:200]}")
            continue
        if question == "/debug":
            debug = not debug
            print(f"  debug {'on' if debug else 'off'}")
            continue

        last = answer_question(chatbot_id, question, history, conversation_id=conversation_id, chain=chain)
        if debug:
            print(f"\n[standalone question] {last.standalone_question}")
            print_docs(last.documents)
        print(f"\nBot: {last.answer}\n")
        if last.sources:
            print("Sources:")
            print_sources(last)
            print()

        from langchain_core.messages import AIMessage, HumanMessage

        history.extend([HumanMessage(content=question), AIMessage(content=last.answer)])
        if one_question:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chatbot", type=int, required=True, help="chatbot id (see create_chatbot.py --list)")
    parser.add_argument("--conversation", help="conversation uuid to resume")
    parser.add_argument("-q", "--question", help="ask one question and exit")
    parser.add_argument("--search-only", metavar="QUERY", help="only show the retrieved chunks for QUERY")
    parser.add_argument("-k", type=int, help="number of chunks for --search-only")
    parser.add_argument("--debug", action="store_true", help="print the standalone question and retrieved chunks")
    args = parser.parse_args()
    setup_logging()

    try:
        if args.search_only:
            cmd_search(args.chatbot, args.search_only, args.k)
        else:
            chat_loop(args.chatbot, args.conversation, args.question, args.debug)
    except (ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
