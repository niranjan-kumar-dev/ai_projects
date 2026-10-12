"""Turn raw HTML into clean, readable text.

Two levels of cleaning:

* **Per page** (`extract_page`): remove scripts, styles, navigation, footers,
  cookie banners...; prefer the `<main>`/`<article>` region; normalise whitespace.
* **Across pages** (`remove_boilerplate`): a line that appears on many pages of
  the same site (menu items, copyright notices, "Subscribe to our newsletter")
  is boilerplate and is removed from every page.

Images (`extract_images`) are collected separately - logos, `og:image` and the
pictures inside the main content - with their alt text / caption / nearest
heading, so the chunker can index them as searchable "image chunks".
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter

from bs4 import BeautifulSoup, Comment, Tag

from chat_with_website.config import settings
from chat_with_website.ingestion.models import CrawledPage, PageContent, PageImage
from chat_with_website.ingestion.url_utils import normalize_url

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


def _clean_soup(soup: BeautifulSoup) -> Tag:
    """Strip noise tags in place and return the main-content root."""
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
    return root if root is not None else (soup.body or soup)


def _root_text(root: Tag) -> str:
    # Add newlines around block elements so get_text keeps paragraph boundaries.
    for tag in root.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    # Headings: keep them on their own line and visually mark them.
    for h in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        h.insert_before("\n\n")
        h.insert_after("\n")
    return _clean_text(root.get_text(" "))


def html_to_text(html: str) -> tuple[str, BeautifulSoup]:
    """Return (clean text, soup) for one HTML document."""
    soup = BeautifulSoup(html, "lxml")
    root = _clean_soup(soup)
    return _root_text(root), soup


# ------------------------------------------------------------------- images
# `get_text()` ignores attributes, so <img src/alt> would vanish.  We collect
# images separately so they can become searchable "image chunks" (see chunker).
IMAGE_SRC_ATTRS = ("src", "data-src", "data-lazy-src", "data-original")
MAX_IMAGES_PER_PAGE = 20
MIN_IMAGE_SIZE = 64  # px; smaller declared sizes are icons / tracking pixels
_PIXEL_RE = re.compile(r"(pixel|spacer|1x1|blank\.|transparent\.|tracking|/ads?/)", re.I)
_HEADINGS = ["h1", "h2", "h3", "h4", "h5", "h6"]


def _img_url(img: Tag, base_url: str) -> str | None:
    candidate = next((img.get(a) for a in IMAGE_SRC_ATTRS if img.get(a)), None)
    if not candidate:
        srcset = img.get("srcset") or img.get("data-srcset") or ""
        candidate = srcset.split(",")[0].strip().split(" ")[0] if srcset else None
    if not candidate or candidate.startswith("data:"):
        return None
    url = normalize_url(candidate, base=base_url)  # resolves relative paths, drops non-http
    if not url or _PIXEL_RE.search(url):
        return None
    return url


def _is_tiny(img: Tag) -> bool:
    for attr in ("width", "height"):
        m = re.match(r"\d+", str(img.get(attr, "")))
        if m and int(m.group()) < MIN_IMAGE_SIZE:
            return True
    return False


def _attr_text(tag: Tag, name: str) -> str:
    return _WS_RE.sub(" ", str(tag.get(name) or "")).strip()


def _image_from_tag(img: Tag, base_url: str, context: str = "") -> PageImage | None:
    url = _img_url(img, base_url)
    if not url or _is_tiny(img):
        return None
    alt = _attr_text(img, "alt") or _attr_text(img, "title")
    caption = ""
    figure = img.find_parent("figure")
    if figure is not None:
        fc = figure.find("figcaption")
        if fc is not None:
            caption = fc.get_text(" ", strip=True)
    if not context:
        heading = img.find_previous(_HEADINGS)
        if heading is not None:
            context = heading.get_text(" ", strip=True)
        elif img.parent is not None:
            context = img.parent.get_text(" ", strip=True)[:100]
    return PageImage(url=url, alt=alt, caption=caption, context=_WS_RE.sub(" ", context).strip())


def _logo_images(soup: BeautifulSoup, base_url: str) -> list[PageImage]:
    """Logos live in <header>/<nav>, which are removed as noise - grab them first."""
    out = []
    for img in soup.find_all("img"):
        hint = " ".join([_attr_text(img, "alt"), _attr_text(img, "class"), _attr_text(img, "id"), _attr_text(img, "src")]).lower()
        if "logo" in hint:
            pi = _image_from_tag(img, base_url, context="site logo")
            if pi:
                pi.alt = pi.alt or "logo"
                out.append(pi)
    return out


def extract_images(soup: BeautifulSoup, root: Tag, base_url: str, logos: list[PageImage] = ()) -> list[PageImage]:
    """Logos + og:image + images in the main content, de-duplicated, capped."""
    images: list[PageImage] = list(logos)
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        url = normalize_url(og["content"], base=base_url)
        if url:
            images.append(PageImage(url=url, alt=_page_title(soup, base_url), context="page preview image"))
    for img in root.find_all("img"):
        pi = _image_from_tag(img, base_url)
        if pi:
            images.append(pi)

    seen: set[str] = set()
    unique = []
    for pi in images:
        if pi.url not in seen and pi.description:
            seen.add(pi.url)
            unique.append(pi)
    return unique[:MAX_IMAGES_PER_PAGE]


def extract_page(page: CrawledPage) -> PageContent:
    soup = BeautifulSoup(page.html, "lxml")
    logos = _logo_images(soup, page.url)  # before <header>/<nav> are stripped
    root = _clean_soup(soup)
    images = extract_images(soup, root, page.url, logos)  # before newlines are injected
    text = _root_text(root)
    title = _page_title(soup, page.url)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return PageContent(
        url=page.url,
        title=title,
        text=text,
        content_hash=content_hash(text, images),
        lines=lines,
        images=images,
    )


def content_hash(text: str, images: list[PageImage] = ()) -> str:
    """Change-detection hash; includes image URLs so an image-only edit re-indexes."""
    h = hashlib.sha256(text.encode("utf-8"))
    for url in sorted(i.url for i in images):
        h.update(b"\n" + url.encode("utf-8"))
    return h.hexdigest()


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
        cleaned.append(
            PageContent(
                url=p.url, title=p.title, text=text, content_hash=content_hash(text, p.images), lines=kept, images=p.images
            )
        )
    return cleaned
