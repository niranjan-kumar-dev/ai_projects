"""Unit tests for the pure knowledge-base status helpers (no Streamlit, no database)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from chat_with_website.ui import kb_status as kb

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)


# ------------------------------------------------------------------ labels
@pytest.mark.parametrize(
    "raw,label",
    [("pending", "Pending"), ("crawling", "Processing"), ("ready", "Completed"), ("failed", "Failed"),
     ("weird", kb.NOT_AVAILABLE), (None, kb.NOT_AVAILABLE), ("", kb.NOT_AVAILABLE)],
)
def test_status_label_maps_pipeline_states_and_never_guesses(raw, label):
    assert kb.status_label(raw) == label


def test_status_badge_markdown_uses_streamlit_badge_syntax():
    assert kb.status_badge_markdown("Completed") == ":green-badge[Completed]"
    assert kb.status_badge_markdown("Failed") == ":red-badge[Failed]"
    assert kb.status_badge_markdown("something else") == ":gray-badge[something else]"


def test_fmt_time_and_fmt_count_show_not_available_for_missing_values():
    assert kb.fmt_time(None) == kb.NOT_AVAILABLE
    assert kb.fmt_count(None) == kb.NOT_AVAILABLE
    assert kb.fmt_count(0) == "0"
    assert kb.fmt_count(12345) == "12,345"
    naive = datetime(2026, 1, 2, 3, 4)
    assert kb.fmt_time(naive) == "2026-01-02 03:04"
    aware = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    assert kb.fmt_time(aware) == aware.astimezone().strftime("%Y-%m-%d %H:%M")  # local time, no guessing


def test_run_summary_label_counts_failed_pages_on_a_completed_run():
    assert kb.run_summary_label(None) == kb.NOT_AVAILABLE
    assert kb.run_summary_label({"run_status": None}) == kb.NOT_AVAILABLE
    assert kb.run_summary_label({"run_status": "processing"}) == "Processing"
    assert kb.run_summary_label({"run_status": "completed", "run_page_errors": []}) == "Completed"
    errs = [{"url": "https://a/x", "error": "boom"}, {"url": "https://a/y", "error": "boom"}]
    assert kb.run_summary_label({"run_status": "completed", "run_page_errors": errs}) == "Completed · 2 page(s) failed"
    assert kb.run_summary_label({"run_status": "failed", "run_page_errors": errs}) == "Failed"


# ------------------------------------------------------------- page status
def test_failed_pages_is_tolerant_of_missing_or_malformed_data():
    assert kb.failed_pages(None) == []
    assert kb.failed_pages({}) == []
    assert kb.failed_pages({"run_page_errors": None}) == []
    assert kb.failed_pages({"run_page_errors": "not a list"}) == []
    assert kb.failed_pages({"run_page_errors": [{"error": "no url"}, "junk", {"url": "https://a/", "error": None}]}) == [
        {"url": "https://a/", "error": ""}
    ]
    # the run-history rows use the raw column name
    assert kb.failed_pages({"page_errors": [{"url": "https://b/", "error": "x"}]}) == [{"url": "https://b/", "error": "x"}]


@pytest.mark.parametrize(
    "chunks,url,failed,expected",
    [
        (3, "https://a/ok", set(), kb.PAGE_INDEXED),
        (0, "https://a/none", set(), kb.PAGE_NO_CHUNKS),
        (None, "https://a/none", set(), kb.PAGE_NO_CHUNKS),
        (0, "https://a/new", {"https://a/new"}, kb.PAGE_FAILED),
        (5, "https://a/old", {"https://a/old"}, kb.PAGE_FAILED_KEPT),
    ],
)
def test_page_status_matrix(chunks, url, failed, expected):
    assert kb.page_status(chunks, url, failed) == expected


# ------------------------------------------------------------- freshness
def test_run_is_stalled_only_for_old_processing_runs():
    limit = timedelta(hours=3)
    assert not kb.run_is_stalled(None, NOW, limit)
    assert not kb.run_is_stalled({"run_status": "completed", "run_started_at": NOW - timedelta(days=2)}, NOW, limit)
    assert not kb.run_is_stalled({"run_status": "processing", "run_started_at": None}, NOW, limit)
    assert not kb.run_is_stalled({"run_status": "processing", "run_started_at": NOW - timedelta(hours=2, minutes=59)}, NOW, limit)
    assert kb.run_is_stalled({"run_status": "processing", "run_started_at": NOW - timedelta(hours=3, minutes=1)}, NOW, limit)


def test_freshness_unknown_fresh_stale():
    stale_after = timedelta(days=30)
    assert kb.freshness(None, NOW, stale_after) == "unknown"
    assert kb.freshness(NOW - timedelta(days=29), NOW, stale_after) == "fresh"
    assert kb.freshness(NOW - timedelta(days=31), NOW, stale_after) == "stale"


def test_embedding_drift_reports_each_difference_and_nothing_when_aligned():
    site = {"embedding_provider": "huggingface", "embedding_model": "m1", "embedding_dim": 384}
    assert kb.embedding_drift(site, "huggingface", "m1", 384) is None
    msg = kb.embedding_drift(site, "openai", "text-embedding-3-small", 1536)
    assert "provider huggingface (app: openai)" in msg
    assert "model m1 (app: text-embedding-3-small)" in msg
    assert "384 dims (app: 1536)" in msg
    assert kb.embedding_drift({}, "openai", "m", 1) is None  # nothing recorded -> nothing to compare
    assert "model m1 (app: m2)" in kb.embedding_drift({"embedding_model": "m1"}, "huggingface", "m2", 384)


# -------------------------------------------------------- chatbot scoping
BOTS = [
    {"id": 1, "name": "beta", "website_ids": [10, 20]},
    {"id": 2, "name": "alpha", "website_ids": [20]},
    {"id": 3, "name": "empty", "website_ids": [None]},  # array_agg of a bot without websites
]


def test_websites_for_chatbot():
    assert kb.websites_for_chatbot(BOTS, None) is None
    assert kb.websites_for_chatbot(BOTS, 1) == {10, 20}
    assert kb.websites_for_chatbot(BOTS, 3) == set()
    assert kb.websites_for_chatbot(BOTS, 99) == set()


def test_chatbot_names_by_website_is_sorted_and_skips_nulls():
    assert kb.chatbot_names_by_website(BOTS) == {10: ["beta"], 20: ["alpha", "beta"]}


# ------------------------------------------------------------- pagination
def test_page_window_edges():
    assert kb.page_window(0, 1, 100) == (0, 1, "No pages")
    assert kb.page_window(100, 1, 100) == (0, 1, "Showing 1–100 of 100")
    assert kb.page_window(101, 2, 100) == (100, 2, "Showing 101–101 of 101")
    assert kb.page_window(250, 3, 100) == (200, 3, "Showing 201–250 of 250")
    assert kb.page_window(250, 99, 100) == (200, 3, "Showing 201–250 of 250")  # clamped
    assert kb.page_window(250, 0, 100) == (0, 3, "Showing 1–100 of 250")  # clamped
    assert kb.page_window(5, 1, 0)[0] == 0  # size never below 1
