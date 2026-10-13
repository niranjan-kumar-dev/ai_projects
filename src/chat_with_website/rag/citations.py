"""Pure helpers for citations: validate `[n]` markers, turn them into links, group sources.

Everything here is plain string processing (no Streamlit, no database), so it is unit-tested
directly. The contract with the rest of the chain:

- `n` in `[n]` is the 1-based position of a passage in the context the model received
  (`format_context` in chain.py numbers them). `build_sources` records one `Source` per passage,
  so the message's own `sources` list is the only mapping ever needed to resolve a marker.
- `normalize_citations` runs once, when an answer is finalised: it rewrites `[1, 2]` / `[1-3]`
  into `[1][2][3]` and drops numbers the model invented (outside 1..n_docs). The cleaned text is
  what gets stored, so every later render (reload, branch, copy) works from valid markers.
- `linkify_citations` runs at render time only: the stored text keeps plain `[n]` markers.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlsplit

log = logging.getLogger(__name__)

# One or more comma/space-separated numbers or ranges inside one pair of brackets: [1] [2, 3] [1-3] [2][5]
CITATION_RE = re.compile(r"\[(\d{1,3}(?:\s*(?:,|-|–)\s*\d{1,3})*)\]")

# Regions where markers must never be touched: fenced code, inline code, markdown images (alt text,
# which may itself contain one level of [brackets]).
_PROTECTED_RE = re.compile(r"```.*?```|`[^`\n]*`|!\[(?:[^\[\]]|\[[^\]]*\])*\]\([^)]*\)", re.S)
_REMOVED = "\x00"  # placeholder for a marker that lost all its numbers

_STOP_WORDS = {
    "about", "above", "after", "again", "also", "because", "before", "being", "below", "between",
    "could", "does", "doing", "down", "during", "each", "from", "further", "have", "having", "here",
    "into", "itself", "just", "more", "most", "once", "only", "other", "over", "same", "should",
    "some", "such", "than", "that", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "under", "until", "very", "what", "when", "where", "which", "while", "will",
    "with", "would", "your", "tell", "show", "give", "list", "explain", "please", "website", "page",
}
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>#|~])")


def _split_protected(text: str) -> list[tuple[str, bool]]:
    """Split `text` into (segment, protected) pairs; protected segments are returned verbatim."""
    parts: list[tuple[str, bool]] = []
    pos = 0
    for m in _PROTECTED_RE.finditer(text):
        if m.start() > pos:
            parts.append((text[pos : m.start()], False))
        parts.append((m.group(0), True))
        pos = m.end()
    if pos < len(text):
        parts.append((text[pos:], False))
    return parts


def _expand(group: str) -> list[int]:
    """'1, 3-5' -> [1, 3, 4, 5] (order kept, duplicates dropped)."""
    out: list[int] = []
    for piece in re.split(r"\s*,\s*", group.strip()):
        rng = re.split(r"\s*[-–]\s*", piece)
        if len(rng) == 2 and rng[0].isdigit() and rng[1].isdigit():
            lo, hi = int(rng[0]), int(rng[1])
            nums = list(range(lo, hi + 1)) if lo <= hi and hi - lo < 50 else [lo, hi]
        else:
            nums = [int(x) for x in rng if x.isdigit()]
        for n in nums:
            if n not in out:
                out.append(n)
    return out


def normalize_citations(text: str, n_docs: int) -> tuple[str, set[int]]:
    """Rewrite marker runs into `[a][b]` form and drop numbers outside 1..n_docs.

    Returns the cleaned text and the set of valid numbers it cites. Markers inside code or
    image alt text are left alone. A marker that becomes empty is removed together with one
    preceding space, so "fact [9]." becomes "fact.".
    """
    cited: set[int] = set()
    invented: set[int] = set()

    def _fix(m: re.Match) -> str:
        nums = _expand(m.group(1))
        valid = [n for n in nums if 1 <= n <= n_docs]
        invented.update(n for n in nums if not 1 <= n <= n_docs)
        cited.update(valid)
        return "".join(f"[{n}]" for n in valid) or _REMOVED

    out = []
    for seg, protected in _split_protected(text):
        if protected:
            out.append(seg)
            continue
        seg = CITATION_RE.sub(_fix, seg)
        # Drop the placeholder; take the space before it only when punctuation follows
        # ("fact [9]." -> "fact."), otherwise leave one space ("D [0][2]" -> "D [2]").
        seg = re.sub(r"[ \t]*" + _REMOVED + r"+(?=[.,;:!?])", "", seg).replace(_REMOVED, "")
        out.append(seg)
    cleaned = "".join(out)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    if invented:
        log.warning("Dropped citation number(s) not in the context: %s", sorted(invented))
    return cleaned, cited


def extract_citation_numbers(text: str) -> list[int]:
    """Every number the text cites, in order of first appearance, *without* validating it.

    `normalize_citations` drops invented numbers, so anything that needs to know what the model
    originally wrote (the evaluation's citation-integrity check) must run this on the raw answer.
    Markers inside code or image alt text are ignored, as everywhere else.
    """
    out: list[int] = []
    for seg, protected in _split_protected(text or ""):
        if protected:
            continue
        for m in CITATION_RE.finditer(seg):
            for n in _expand(m.group(1)):
                if n not in out:
                    out.append(n)
    return out


def strip_citations(text: str) -> str:
    """Remove `[n]` markers (outside code / images) so text comparisons see only the prose."""
    out = []
    for seg, protected in _split_protected(text or ""):
        out.append(seg if protected else CITATION_RE.sub("", seg))
    return re.sub(r"[ \t]+(?=[.,;:!?])", "", re.sub(r"[ \t]{2,}", " ", "".join(out))).strip()


def linkify_citations(text: str, sources: list[dict] | None) -> str:
    """Turn `[n]` into a markdown link to source n's page: `[[n]](url)`.

    CommonMark renders the inner brackets as the link text, so the reader still sees "[n]".
    Numbers without a known url stay plain text. Never called on stored text - render time only.
    """
    urls = {int(s["n"]): s.get("url") for s in (sources or []) if s.get("n") is not None}
    if not urls:
        return text

    def _link(m: re.Match) -> str:
        parts = []
        for n in _expand(m.group(1)):
            url = urls.get(n)
            parts.append(f"[[{n}]]({url})" if url else f"[{n}]")
        return "".join(parts)

    return "".join(seg if protected else CITATION_RE.sub(_link, seg) for seg, protected in _split_protected(text))


def display_url(url: str | None, max_len: int = 60) -> str:
    """'https://www.example.com/docs/a/b?x=1' -> 'example.com/docs/a/b' (trimmed with an ellipsis)."""
    if not url:
        return ""
    parts = urlsplit(url)
    host = (parts.netloc or "").removeprefix("www.")
    path = parts.path.rstrip("/")
    shown = f"{host}{path}" if host else url
    return shown if len(shown) <= max_len else shown[: max_len - 1] + "…"


def group_sources(sources: list[dict] | None) -> list[dict]:
    """One card per page url (several chunks of one page merge), in first-seen order.

    Each card: numbers (sorted), title, url, cited, kind, image_url, score (max), snippets
    (deduped, in number order). Missing metadata gets a readable fallback.
    """
    cards: dict[str, dict] = {}
    for s in sources or []:
        key = s.get("image_url") or s.get("url") or f"#{s.get('n')}"
        card = cards.get(key)
        if card is None:
            url = s.get("url") or ""
            title = (s.get("title") or "").strip() or display_url(url) or "Untitled page"
            card = cards[key] = {
                "numbers": [], "title": title, "url": url, "cited": False, "kind": s.get("kind") or "text",
                "image_url": s.get("image_url"), "score": None, "snippets": [],
            }
        if s.get("n") is not None and s["n"] not in card["numbers"]:
            card["numbers"].append(int(s["n"]))
        card["cited"] = card["cited"] or bool(s.get("cited"))
        if s.get("score") is not None:
            card["score"] = max(card["score"] or 0.0, float(s["score"]))
        snippet = (s.get("snippet") or "").strip()
        if snippet and snippet not in card["snippets"]:
            card["snippets"].append(snippet)
    out = list(cards.values())
    for card in out:
        card["numbers"].sort()
    return out


def escape_markdown(text: str) -> str:
    """Escape characters that markdown would otherwise interpret, so evidence is shown verbatim."""
    return _MD_SPECIAL.sub(r"\\\1", text).replace("\n", "  \n")


def question_terms(question: str | None, limit: int = 10) -> list[str]:
    """Distinct words (>= 4 letters, not stop words) from the question, for highlighting."""
    if not question:
        return []
    seen: list[str] = []
    for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]{3,}", question):
        lw = w.lower()
        if lw not in _STOP_WORDS and lw not in seen:
            seen.append(lw)
        if len(seen) >= limit:
            break
    return seen


def highlight_terms(snippet: str, terms: list[str]) -> str:
    """Escape the snippet for markdown and bold whole-word matches of `terms` (case-insensitive).

    The evidence text itself is unchanged; only emphasis is added.
    """
    escaped = escape_markdown(snippet)
    if not terms:
        return escaped
    # Match the terms in their escaped form too ("web_dev" became "web\_dev").
    pattern = re.compile(r"\b(" + "|".join(re.escape(escape_markdown(t)) for t in terms) + r")\b", re.I)
    return pattern.sub(lambda m: f"**{m.group(1)}**", escaped)
