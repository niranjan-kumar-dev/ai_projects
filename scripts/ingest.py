"""Index a website into PostgreSQL/pgvector.

Usage (from the project root):

    python scripts/ingest.py https://www.python.org/about/
    python scripts/ingest.py https://example.com --name "Example" --max-pages 20
    python scripts/ingest.py https://example.com --dry-run      # crawl + clean only, print what would be indexed
    python scripts/ingest.py --list                              # show indexed websites
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8")

from chat_with_website.config import settings  # noqa: E402
from chat_with_website.logging_config import setup_logging  # noqa: E402

log = logging.getLogger("ingest")


def cmd_list() -> None:
    from chat_with_website.db import repository as repo
    from chat_with_website.db.connection import get_conn

    with get_conn() as conn:
        rows = repo.list_websites(conn)
    if not rows:
        print("No websites indexed yet.")
        return
    print(f"{'id':>3}  {'status':<9} {'pages':>5} {'chunks':>6}  {'model':<40} base_url")
    for w in rows:
        print(f"{w['id']:>3}  {w['status']:<9} {w['page_count']:>5} {w['chunk_count']:>6}  {w['embedding_model']:<40} {w['base_url']}")


def cmd_dry_run(url: str, max_pages: int | None) -> None:
    from chat_with_website.ingestion.pipeline import crawl_and_extract
    from chat_with_website.ingestion.url_utils import validate_start_url

    start = validate_start_url(url, allow_private=settings.crawl_allow_private_hosts)
    pages, crawled, skipped = crawl_and_extract(start, max_pages=max_pages)
    print(f"\nCrawled {crawled} page(s); {len(pages)} indexable, {skipped} too short.\n")
    for p in pages:
        preview = p.text[:200].replace("\n", " ")
        print(f"- {p.url}\n  title: {p.title}\n  words: {p.word_count}\n  text : {preview}...\n")


def cmd_ingest(url: str, name: str | None, max_pages: int | None) -> None:
    from chat_with_website.ingestion.pipeline import ingest_website

    result = ingest_website(url, name=name, max_pages=max_pages)
    print("\nIngestion summary")
    print(f"  website id        : {result.website_id}")
    print(f"  pages crawled     : {result.pages_crawled}")
    print(f"  pages indexed     : {result.pages_indexed}")
    print(f"  unchanged (skipped): {result.pages_skipped_unchanged}")
    print(f"  too short (skipped): {result.pages_skipped_short}")
    print(f"  chunks created    : {result.chunks_created}")
    if result.errors:
        print(f"  errors            : {len(result.errors)}")
        for e in result.errors[:10]:
            print(f"    - {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", nargs="?", help="start URL of the website to index")
    parser.add_argument("--name", help="display name for the website (default: domain)")
    parser.add_argument("--max-pages", type=int, help=f"page limit (default {settings.crawl_max_pages})")
    parser.add_argument("--dry-run", action="store_true", help="crawl and clean only; do not touch the database")
    parser.add_argument("--list", action="store_true", help="list indexed websites and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging("DEBUG" if args.verbose else None)

    if args.list:
        cmd_list()
        return
    if not args.url:
        parser.error("url is required (or use --list)")
    try:
        if args.dry_run:
            cmd_dry_run(args.url, args.max_pages)
        else:
            cmd_ingest(args.url, args.name, args.max_pages)
    except (ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
