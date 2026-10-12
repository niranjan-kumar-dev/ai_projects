"""Pure helpers for the sidebar conversation list (no Streamlit imports, unit-testable)."""

from __future__ import annotations

from datetime import datetime, timezone

CONVERSATION_GROUPS = ("Today", "Yesterday", "Previous 7 days", "Older")


def filter_conversations(convs: list[dict], query: str) -> list[dict]:
    """Case-insensitive title search; every whitespace-separated term must appear.

    Works on the already-loaded list, so typing never touches the database.
    """
    terms = [t for t in (query or "").lower().split() if t]
    if not terms:
        return list(convs)
    return [c for c in convs if all(t in (c.get("title") or "untitled").lower() for t in terms)]


def group_conversations(convs: list[dict], now: datetime | None = None) -> list[tuple[str, list[dict]]]:
    """Bucket conversations by last activity, ChatGPT-style. Empty groups are omitted.

    `updated_at` (or `created_at`) is timezone-aware; calendar days are compared in the
    server's local zone so "Today" matches what the user sees on the clock.
    """
    now = (now or datetime.now(timezone.utc)).astimezone()
    today = now.date()
    buckets: dict[str, list[dict]] = {g: [] for g in CONVERSATION_GROUPS}
    for c in convs:
        when = (c.get("updated_at") or c["created_at"]).astimezone().date()
        age = (today - when).days
        if age <= 0:
            buckets["Today"].append(c)
        elif age == 1:
            buckets["Yesterday"].append(c)
        elif age <= 7:
            buckets["Previous 7 days"].append(c)
        else:
            buckets["Older"].append(c)
    return [(g, items) for g, items in buckets.items() if items]
