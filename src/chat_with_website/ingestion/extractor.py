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
from urllib.parse import unquote, urlsplit

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


NEARBY_MAX_CHARS = 160
NEARBY_MAX_LEVELS = 3
# File-name stems that say nothing about the subject.
_GENERIC_STEMS = {"image", "images", "img", "photo", "picture", "pic", "logo", "banner", "icon", "thumb", "thumbnail", "default", "placeholder"}
_HEX_RE = re.compile(r"^[0-9a-f]{8,}$|^[0-9a-f-]{32,}$", re.I)
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")


def image_filename_words(url: str) -> str:
    """'/img/AuthorImage/Suyash.jpg' -> 'Suyash'; 'human-transcriptionists-1.svg' -> 'human transcriptionists'.

    Returns '' for meaningless names (logo.svg, a1b2c3d4.png, 1x1.gif).
    """
    stem = unquote(urlsplit(url).path.rsplit("/", 1)[-1]).rsplit(".", 1)[0]
    if not stem or _HEX_RE.match(stem):
        return ""
    stem = _CAMEL_RE.sub(" ", stem)
    words = [w for w in re.split(r"[-_.\s]+|\d+", stem) if len(w) > 1]
    words = [w for w in words if w.lower() not in _GENERIC_STEMS]
    return " ".join(words)


def _nearby_text(img: Tag) -> str:
    """Text of the closest ancestor (up to 3 levels) that has words other than the image's own."""
    node = img
    for _ in range(NEARBY_MAX_LEVELS):
        node = node.parent
        if node is None or node.name in ("body", "html", "[document]"):
            break
        text = _WS_RE.sub(" ", node.get_text(" ", strip=True)).strip()
        if text:
            return text[:NEARBY_MAX_CHARS].rsplit(" ", 1)[0] if len(text) > NEARBY_MAX_CHARS else text
    return ""


def _image_from_tag(img: Tag, base_url: str, context: str = "", nearby: bool = True) -> PageImage | None:
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
    near = _nearby_text(img) if nearby else ""
    if near and (near == caption or near == alt):
        near = ""
    return PageImage(
        url=url,
        alt=alt,
        caption=caption,
        context=_WS_RE.sub(" ", context).strip(),
        nearby=near,
        filename=image_filename_words(url),
    )


MAX_LOGOS_PER_PAGE = 2
_BRAND_CLASS_RE = re.compile(r"navbar|brand|site-logo|header-logo|masthead", re.I)


def _in_site_header(img: Tag) -> bool:
    """True when the image sits in the page header/nav (where the site logo lives)."""
    if img.find_parent(["header", "nav"]) is not None:
        return True
    for anc in img.parents:
        if anc.name in ("body", "html", "[document]", None):
            break
        if _BRAND_CLASS_RE.search(_attr_text(anc, "class") + " " + _attr_text(anc, "id")):
            return True
    return False


def _logo_images(soup: BeautifulSoup, base_url: str) -> list[PageImage]:
    """The site logo lives in <header>/<nav>, which are removed as noise - grab it first.

    Only header/nav images that look like a logo qualify; a "client logos" carousel in
    the body is ordinary content and is handled by `extract_images` with its own alt text.
    """
    out: list[PageImage] = []
    for img in soup.find_all("img"):
        hint = " ".join([_attr_text(img, "alt"), _attr_text(img, "class"), _attr_text(img, "id"), _attr_text(img, "src")]).lower()
        if "logo" in hint and _in_site_header(img):
            pi = _image_from_tag(img, base_url, context="site logo", nearby=False)  # nav text is noise
            if pi:
                pi.alt = pi.alt or "logo"
                out.append(pi)
                if len(out) >= MAX_LOGOS_PER_PAGE:
                    break
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


IMAGE_FORMAT_VERSION = b"img-v3"  # bump when image *descriptions* change, so pages re-index once


def content_hash(text: str, images: list[PageImage] = ()) -> str:
    """Change-detection hash; includes image URLs so an image-only edit re-indexes."""
    h = hashlib.sha256(text.encode("utf-8"))
    if images:
        h.update(b"\n" + IMAGE_FORMAT_VERSION)
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

    threshold = max(3, math.ceil(len(pages) * ratio))
    counts: Counter[str] = Counter()
    image_counts: Counter[str] = Counter()
    for p in pages:
        counts.update(set(ln.strip() for ln in p.lines if len(ln.strip()) >= 3))
        image_counts.update({img.url for img in p.images})
    boilerplate = {ln for ln, c in counts.items() if c >= threshold}
    # An image on many pages (site logo, badges, author photo in a sidebar) is indexed on the
    # *first* page only; otherwise it would become one near-identical chunk per page.
    repeated_images = {url for url, c in image_counts.items() if c >= threshold}
    if not boilerplate and not repeated_images:
        return pages
    log.info(
        "Removing %d boilerplate lines and de-duplicating %d repeated images seen on >= %d pages",
        len(boilerplate), len(repeated_images), threshold,
    )

    seen_images: set[str] = set()
    cleaned: list[PageContent] = []
    for p in pages:
        kept = [ln for ln in p.lines if ln.strip() not in boilerplate]
        text = _clean_text("\n".join(kept))
        images = []
        for img in p.images:
            if img.url in repeated_images:
                if img.url in seen_images:
                    continue
                seen_images.add(img.url)
            images.append(img)
        cleaned.append(
            PageContent(
                url=p.url, title=p.title, text=text, content_hash=content_hash(text, images), lines=kept, images=images
            )
        )
    return cleaned
