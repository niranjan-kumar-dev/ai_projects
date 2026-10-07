"""Turn raw HTML into clean, readable text.

Two levels of cleaning:

* **Per page** (`extract_page`): remove scripts, styles, navigation, footers,
  cookie banners...; prefer the `<main>`/`<article>` region; normalise whitespace.
* **Across pages** (`remove_boilerplate`): a line that appears on many pages of
  the same site (menu items, copyright notices, "Subscribe to our newsletter")
  is boilerplate and is removed from every page.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter

from bs4 import BeautifulSoup, Comment

from chat_with_website.config import settings
from chat_with_website.ingestion.models import CrawledPage, PageContent

log = logging.getLogger(__name__)

# Tags whose content is never useful for answering questions.
REMOVE_TAGS = [
    "script", "style", "noscript", "template", "iframe", "svg", "canvas", "video", "audio",
    "nav", "header", "footer", "aside", "form", "button", "input", "select", "textarea",
]
# CSS selectors for common boilerplate containers.
REMOVE_SELECTORS = [
    "[role=navigation]", "[role=banner]", "[role=contentinfo]", "[role=complementary]",
    "[aria-hidden=true]", ".cookie", "#cookie", "[class*=cookie]", "[id*=cookie]",
    "[class*=breadcrumb]", "[class*=sidebar]", "[id*=sidebar]", "[class*=share]",
    "[class*=social]", "[class*=newsletter]", "[class*=popup]", "[class*=modal]",
    ".sr-only", ".visually-hidden", ".skip-link",
]
MAIN_SELECTORS = ["main", "[role=main]", "article", "#content", "#main", ".content", ".main"]

# Block-level tags: we insert newlines around them so paragraphs stay separate.
BLOCK_TAGS = {
    "p", "div", "section", "article", "li", "ul", "ol", "br", "hr", "table", "tr", "td", "th",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "dd", "dt", "dl", "figure", "figcaption",
}

_WS_RE = re.compile(r"[ \t \r\f\v]+")
_NL_RE = re.compile(r"\n{3,}")


def _page_title(soup: BeautifulSoup, url: str) -> str:
    # <title> is page-specific on most sites; og:title is often the same for the whole site.
    if soup.title and soup.title.string and soup.title.string.strip():
        return _WS_RE.sub(" ", soup.title.string).strip()
    og = soup.find("meta", property="og:title")
    if og and og.get("content", "").strip():
        return og["content"].strip()
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(" ", strip=True)
    return url


def _clean_text(text: str) -> str:
    text = _WS_RE.sub(" ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


def html_to_text(html: str) -> tuple[str, BeautifulSoup]:
    """Return (clean text, soup) for one HTML document."""
    soup = BeautifulSoup(html, "lxml")

    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    for tag in soup.find_all(REMOVE_TAGS):
        tag.decompose()
    for selector in REMOVE_SELECTORS:
        try:
            for tag in soup.select(selector):
                tag.decompose()
        except Exception:  # invalid selector for this parser - ignore
            continue

    root = None
    for selector in MAIN_SELECTORS:
        root = soup.select_one(selector)
        if root is not None and len(root.get_text(strip=True)) > 200:
            break
        root = None
    if root is None:
        root = soup.body or soup

    # Add newlines around block elements so get_text keeps paragraph boundaries.
    for tag in root.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    # Headings: keep them on their own line and visually mark them.
    for h in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        h.insert_before("\n\n")
        h.insert_after("\n")

    text = root.get_text(" ")
    return _clean_text(text), soup


def extract_page(page: CrawledPage) -> PageContent:
    text, soup = html_to_text(page.html)
    title = _page_title(soup, page.url)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return PageContent(
        url=page.url,
        title=title,
        text=text,
        content_hash=content_hash(text),
        lines=lines,
    )


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def remove_boilerplate(pages: list[PageContent], ratio: float | None = None) -> list[PageContent]:
    """Drop lines that repeat across many pages of the same crawl.

    Only applied when there are at least 3 pages; with fewer pages a repeated
    line may legitimately be content.
    """
    ratio = ratio if ratio is not None else settings.boilerplate_line_ratio
    if len(pages) < 3:
        return pages

    counts: Counter[str] = Counter()
    for p in pages:
        counts.update(set(ln.strip() for ln in p.lines if len(ln.strip()) >= 3))
    threshold = max(3, math.ceil(len(pages) * ratio))
    boilerplate = {ln for ln, c in counts.items() if c >= threshold}
    if not boilerplate:
        return pages
    log.info("Removing %d boilerplate lines seen on >= %d pages", len(boilerplate), threshold)

    cleaned: list[PageContent] = []
    for p in pages:
        kept = [ln for ln in p.lines if ln.strip() not in boilerplate]
        text = _clean_text("\n".join(kept))
        cleaned.append(PageContent(url=p.url, title=p.title, text=text, content_hash=content_hash(text), lines=kept))
    return cleaned
