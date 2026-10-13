"""Follow-up question suggestions: parsing and validation of what the model returns.

Pure string handling (no model, no Streamlit) so it is unit-tested directly. The model is asked
for a JSON array of short questions; `parse_followups` extracts it defensively and
`clean_followups` enforces the rules (short, unique, end with "?", not something already asked).
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

MAX_FOLLOWUP_CHARS = 120


class FollowUps(BaseModel):
    """Schema the model's output is validated against."""

    questions: list[str]


_JSON_ARRAY_RE = re.compile(r"\[.*?\]", re.S)


def parse_followups(text: str) -> list[str]:
    """Extract the list of questions from the model's reply.

    Accepts a bare JSON array, an array embedded in prose or a code fence, or (fallback) one
    question per line. Never raises: unusable output gives an empty list.
    """
    if not text:
        return []
    m = _JSON_ARRAY_RE.search(text)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return FollowUps(questions=[str(x) for x in data if isinstance(x, (str, int, float))]).questions
        except (json.JSONDecodeError, ValidationError):
            pass
    # Fallback: lines that look like questions ("- What ...?" / "1. How ...?")
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip().strip('"') for ln in text.splitlines()]
    return [ln for ln in lines if ln.endswith("?")]


def _normalize(q: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", q.lower()).strip()


def clean_followups(raw: list[str], *, asked: list[str] | tuple[str, ...] = (), limit: int = 3) -> list[str]:
    """Validate and de-duplicate candidate questions, keeping their order.

    Drops empties, anything longer than MAX_FOLLOWUP_CHARS, near-duplicates of each other and of
    questions already asked in the conversation (case/punctuation-insensitive), and makes sure
    each one ends with a question mark.
    """
    seen = {_normalize(a) for a in asked if a}
    out: list[str] = []
    for q in raw:
        q = " ".join((q or "").split()).strip().strip('"').strip()
        if not q or len(q) > MAX_FOLLOWUP_CHARS:
            continue
        if not q.endswith("?"):
            q = q.rstrip(".!:;") + "?"
        key = _normalize(q)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= limit:
            break
    return out
