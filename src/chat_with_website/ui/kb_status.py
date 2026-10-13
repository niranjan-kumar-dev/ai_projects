"""Pure helpers for the knowledge-base status dashboard on the Websites page.

No Streamlit and no database here: the page feeds these functions the rows returned by
`repository.list_websites_overview` / `list_pages_with_chunk_counts` and renders what comes back.
Keeping the logic pure makes every rule (status labels, freshness, stalled runs, "Not available")
unit-testable without a browser or PostgreSQL.

Guiding rule: **never invent a value**. Anything the database does not record for a website is
shown as `NOT_AVAILABLE`, never derived from a neighbouring timestamp.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

NOT_AVAILABLE = "Not available"

# websites.status (set by the ingestion pipeline) -> what administrators see
STATUS_LABELS: dict[str, str] = {
    "pending": "Pending",
    "crawling": "Processing",
    "ready": "Completed",
    "failed": "Failed",
}
# Streamlit badge colours (`:color-badge[text]`), see status_badge_markdown
STATUS_BADGES: dict[str, str] = {
    "Pending": "gray",
    "Processing": "blue",
    "Completed": "green",
    "Failed": "red",
    NOT_AVAILABLE: "gray",
}

PAGE_INDEXED = "Indexed"
PAGE_NO_CHUNKS = "No chunks"
PAGE_FAILED = "Failed"
PAGE_FAILED_KEPT = "Failed · previous version kept"

Freshness = Literal["fresh", "stale", "unknown"]


# ------------------------------------------------------------------ labels
def status_label(website_status: str | None) -> str:
    """Pending / Processing / Completed / Failed; anything unknown is reported, not guessed."""
    if not website_status:
        return NOT_AVAILABLE
    return STATUS_LABELS.get(website_status, NOT_AVAILABLE)


def status_badge_markdown(label: str) -> str:
    colour = STATUS_BADGES.get(label, "gray")
    return f":{colour}-badge[{label}]"


def fmt_time(dt: datetime | None, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Timestamps come back timezone-aware from psycopg; show them in the server's local time."""
    if dt is None:
        return NOT_AVAILABLE
    if dt.tzinfo is not None:
        dt = dt.astimezone()
    return dt.strftime(fmt)


def fmt_count(value: int | None) -> str:
    return NOT_AVAILABLE if value is None else f"{int(value):,}"


def run_summary_label(run: dict | None) -> str:
    """One line describing the latest run, e.g. 'Completed · 2 page(s) failed'."""
    if not run or not run.get("run_status"):
        return NOT_AVAILABLE
    status = {"processing": "Processing", "completed": "Completed", "failed": "Failed"}.get(
        run["run_status"], NOT_AVAILABLE
    )
    failed = len(failed_pages(run))
    if status == "Completed" and failed:
        return f"{status} · {failed} page(s) failed"
    return status


# ------------------------------------------------------------- page status
def failed_pages(run: dict | None) -> list[dict]:
    """The `[{"url", "error"}]` list stored on the latest run; tolerant of missing/legacy data."""
    if not run:
        return []
    raw = run.get("run_page_errors", run.get("page_errors"))
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, dict) and item.get("url"):
            out.append({"url": str(item["url"]), "error": str(item.get("error") or "")})
    return out


def page_status(chunk_count: int | None, url: str, failed_urls: set[str]) -> str:
    """Derived per-page status (the `pages` table has no status column, on purpose)."""
    has_chunks = bool(chunk_count)
    if url in failed_urls:
        return PAGE_FAILED_KEPT if has_chunks else PAGE_FAILED
    return PAGE_INDEXED if has_chunks else PAGE_NO_CHUNKS


# ------------------------------------------------------------- freshness
def run_is_stalled(run: dict | None, now: datetime, max_age: timedelta) -> bool:
    """A run still `processing` after `max_age` almost certainly belongs to a process that died."""
    if not run or run.get("run_status") != "processing":
        return False
    started = run.get("run_started_at")
    if started is None:
        return False
    return now - started > max_age


def freshness(last_indexed_at: datetime | None, now: datetime, stale_after: timedelta) -> Freshness:
    if last_indexed_at is None:
        return "unknown"
    return "stale" if now - last_indexed_at > stale_after else "fresh"


def embedding_drift(website: dict, provider: str, model: str, dim: int) -> str | None:
    """Warning text when the website was embedded with a different model than the app now uses
    (its chunks would be searched with incompatible query vectors), else None."""
    site_provider = website.get("embedding_provider")
    site_model = website.get("embedding_model")
    site_dim = website.get("embedding_dim")
    if site_provider is None and site_model is None and site_dim is None:
        return None  # nothing recorded, nothing to compare
    problems = []
    if site_provider and site_provider != provider:
        problems.append(f"provider {site_provider} (app: {provider})")
    if site_model and site_model != model:
        problems.append(f"model {site_model} (app: {model})")
    if site_dim and site_dim != dim:
        problems.append(f"{site_dim} dims (app: {dim})")
    if not problems:
        return None
    return (
        "This website was embedded with a different configuration than the app currently uses: "
        + ", ".join(problems)
        + ". Re-index it so queries and chunks share one embedding space."
    )


# -------------------------------------------------------- chatbot scoping
def websites_for_chatbot(bots: list[dict], chatbot_id: int | None) -> set[int] | None:
    """The website ids attached to `chatbot_id`; None means "no filter"."""
    if chatbot_id is None:
        return None
    for b in bots:
        if b.get("id") == chatbot_id:
            return {int(w) for w in (b.get("website_ids") or []) if w is not None}
    return set()


def chatbot_names_by_website(bots: list[dict]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for b in bots:
        for w in b.get("website_ids") or []:
            if w is None:
                continue
            out.setdefault(int(w), []).append(str(b.get("name", "")))
    return {k: sorted(v) for k, v in out.items()}


# ------------------------------------------------------------- pagination
def page_window(total: int, page: int, size: int) -> tuple[int, int, str]:
    """(offset, number_of_pages, caption) for a 1-based `page` of `size` rows out of `total`."""
    size = max(int(size), 1)
    total = max(int(total), 0)
    n_pages = max((total + size - 1) // size, 1)
    page = min(max(int(page), 1), n_pages)
    offset = (page - 1) * size
    if total == 0:
        return 0, 1, "No pages"
    first, last = offset + 1, min(offset + size, total)
    return offset, n_pages, f"Showing {first:,}–{last:,} of {total:,}"
