"""Create or update a chatbot and attach indexed websites to it.

Usage (from the project root):

    python scripts/create_chatbot.py --name python --websites 1
    python scripts/create_chatbot.py --name acme --websites 2,3 --description "Acme support bot" \
        --system-prompt "You are Acme's friendly assistant." --llm-provider openai --llm-model gpt-4o-mini
    python scripts/create_chatbot.py --list
    python scripts/create_chatbot.py --delete 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from chat_with_website.config import settings  # noqa: E402
from chat_with_website.db import repository as repo  # noqa: E402
from chat_with_website.db.connection import get_conn  # noqa: E402
from chat_with_website.logging_config import setup_logging  # noqa: E402
from chat_with_website.rag.llm import describe_llm  # noqa: E402

log = logging.getLogger("create_chatbot")


def cmd_list() -> None:
    with get_conn() as conn:
        bots = repo.list_chatbots(conn)
        sites = {w["id"]: w for w in repo.list_websites(conn)}
    if not bots:
        print("No chatbots yet.")
        return
    for b in bots:
        print(f"[{b['id']}] {b['name']}  —  {describe_llm(b)}  —  top_k={b['top_k']}")
        if b.get("description"):
            print(f"     {b['description']}")
        for wid in b["website_ids"]:
            w = sites.get(wid)
            print(f"     website {wid}: {w['base_url'] if w else '?'} ({w['status'] if w else '?'}, {w['chunk_count'] if w else 0} chunks)")


def cmd_create(args: argparse.Namespace) -> None:
    website_ids = [int(x) for x in args.websites.split(",") if x.strip()] if args.websites else []
    with get_conn() as conn:
        existing = {w["id"]: w for w in repo.list_websites(conn)}
        missing = [w for w in website_ids if w not in existing]
        if missing:
            raise ValueError(f"Unknown website id(s): {missing}. Run `python scripts/ingest.py --list`.")
        not_ready = [w for w in website_ids if existing[w]["status"] != "ready"]
        if not_ready:
            log.warning("Website(s) %s are not in status 'ready' yet", not_ready)

        bot = repo.create_chatbot(
            conn,
            name=args.name,
            description=args.description,
            system_prompt=args.system_prompt,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            top_k=args.top_k or settings.top_k,
        )
        if website_ids:
            repo.set_chatbot_websites(conn, bot["id"], website_ids)
    print(f"Chatbot [{bot['id']}] '{bot['name']}' saved with websites {website_ids}; LLM: {describe_llm(bot)}")
    print(f"Try it:  python scripts/ask.py --chatbot {bot['id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", help="unique chatbot name (re-using a name updates that chatbot)")
    parser.add_argument("--websites", help="comma-separated website ids, e.g. 1,2")
    parser.add_argument("--description")
    parser.add_argument("--system-prompt", help="extra instructions prepended to the answer prompt")
    parser.add_argument("--llm-provider", choices=["ollama", "openai", "gemini"], help="override LLM_PROVIDER for this bot")
    parser.add_argument("--llm-model", help="override the model name for this bot")
    parser.add_argument("--top-k", type=int, help=f"chunks to retrieve (default {settings.top_k})")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--delete", type=int, metavar="ID")
    args = parser.parse_args()
    setup_logging()

    try:
        if args.list:
            cmd_list()
        elif args.delete:
            with get_conn() as conn:
                repo.delete_chatbot(conn, args.delete)
            print(f"Deleted chatbot {args.delete}")
        elif args.name:
            cmd_create(args)
        else:
            parser.error("--name is required (or use --list / --delete)")
    except (ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
