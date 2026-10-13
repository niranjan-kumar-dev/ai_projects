"""Pure citation helpers: marker validation, links, source grouping, excerpt highlighting."""

from __future__ import annotations

from chat_with_website.rag.citations import (
    display_url,
    escape_markdown,
    group_sources,
    highlight_terms,
    linkify_citations,
    normalize_citations,
    question_terms,
)

SRC = [
    {"n": 1, "url": "https://acme.example/", "title": "Home", "score": 0.5, "cited": True, "snippet": "We build websites."},
    {"n": 2, "url": "https://acme.example/about", "title": "About", "score": 0.4, "cited": False, "snippet": "Founded in 2010."},
    {"n": 3, "url": "https://acme.example/", "title": "Home", "score": 0.3, "cited": True, "snippet": "Second chunk of home."},
]


def test_normalize_expands_runs_and_drops_invented_numbers():
    text, cited = normalize_citations("A [1, 3]. B [2-4]. C [9]. D [0][2].", n_docs=4)
    assert text == "A [1][3]. B [2][3][4]. C. D [2]."
    assert cited == {1, 2, 3, 4}


def test_normalize_keeps_valid_markers_and_code_untouched():
    text, cited = normalize_citations("Use `x[9]` [1] and ![logo [7]](https://a/logo.png) [2]", n_docs=2)
    assert text == "Use `x[9]` [1] and ![logo [7]](https://a/logo.png) [2]"
    assert cited == {1, 2}


def test_normalize_without_docs_removes_everything():
    assert normalize_citations("Fact [1][2].", n_docs=0) == ("Fact.", set())


def test_linkify_links_known_numbers_only():
    out = linkify_citations("We build sites [1][2][5].", SRC)
    assert out == "We build sites [[1]](https://acme.example/)[[2]](https://acme.example/about)[5]."


def test_linkify_handles_ranges_images_and_missing_sources():
    assert linkify_citations("See [1-2].", SRC) == "See [[1]](https://acme.example/)[[2]](https://acme.example/about)."
    assert linkify_citations("![alt [1]](https://a/x.png) [1]", SRC) == "![alt [1]](https://a/x.png) [[1]](https://acme.example/)"
    assert linkify_citations("Plain [1].", None) == "Plain [1]."
    assert linkify_citations("No url [1].", [{"n": 1, "url": "", "title": "x"}]) == "No url [1]."


def test_group_sources_merges_chunks_of_one_page_and_keeps_order():
    cards = group_sources(SRC)
    assert [c["title"] for c in cards] == ["Home", "About"]
    assert cards[0]["numbers"] == [1, 3] and cards[0]["cited"] is True
    assert cards[0]["snippets"] == ["We build websites.", "Second chunk of home."]
    assert cards[0]["score"] == 0.5
    assert cards[1]["numbers"] == [2] and cards[1]["cited"] is False


def test_group_sources_fallbacks_for_missing_metadata():
    cards = group_sources([
        {"n": 1, "url": "https://www.example.com/docs/page", "title": "", "score": 0.3},
        {"n": 2, "url": "", "title": "", "score": None},
        {"n": 3, "url": "https://x.example/", "title": "X"},  # old row: no snippet / cited / kind
    ])
    assert cards[0]["title"] == "example.com/docs/page"
    assert cards[1]["title"] == "Untitled page" and cards[1]["url"] == "" and cards[1]["score"] is None
    assert cards[2]["snippets"] == [] and cards[2]["cited"] is False and cards[2]["kind"] == "text"


def test_display_url_trims_and_strips_www():
    assert display_url("https://www.example.com/docs/a/b/?x=1") == "example.com/docs/a/b"
    long = display_url("https://example.com/" + "x" * 100)
    assert len(long) == 60 and long.endswith("…")
    assert display_url(None) == ""


def test_highlight_escapes_markdown_and_bolds_whole_words():
    out = highlight_terms("Python *rocks* for web_dev and pythonic code", ["python", "web_dev"])
    assert out == "**Python** \\*rocks\\* for **web\\_dev** and pythonic code"
    assert highlight_terms("plain", []) == "plain"
    assert escape_markdown("a\nb") == "a  \nb"


def test_question_terms_skips_short_and_stop_words():
    assert question_terms("What services does this website offer in Python?") == ["services", "offer", "python"]
    assert question_terms(None) == []
