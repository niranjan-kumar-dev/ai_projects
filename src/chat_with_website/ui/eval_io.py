"""Pure helpers for the RAG Evaluation page: CSV import/export, form parsing, labels.

No Streamlit and no database (like `ui/kb_status.py`), so every rule is unit-tested directly.

CSV format (UTF-8, header row required, extra columns ignored):

    question, expected_answer, answerable, category, source_urls, key_facts, previous_turns, notes

- `answerable`: yes/no, true/false, 1/0 (empty = yes; always "no" for the unanswerable category)
- `category`: a key (`pricing`) or a label (`Pricing`); empty = factual
- `source_urls`, `key_facts`: a JSON array (`["a", "b"]`) or values separated by `|`
- `previous_turns`: a JSON array of `{"role": "user"|"assistant", "content": "..."}` (follow-up cases)
"""

from __future__ import annotations

import csv
import io
import json
from typing import Iterable, Sequence
from urllib.parse import urlsplit

from chat_with_website.db.repository import EVAL_CATEGORIES

CSV_COLUMNS = ["question", "expected_answer", "answerable", "category", "source_urls", "key_facts", "previous_turns", "notes"]
MAX_IMPORT_ROWS = 500
MAX_IMPORT_BYTES = 1_000_000
MAX_QUESTION_CHARS = 2000
MAX_ANSWER_CHARS = 8000
MAX_LIST_ITEMS = 20
STATUS_BADGES = {
    "passed": ":green-badge[Passed]",
    "failed": ":red-badge[Failed]",
    "needs_review": ":orange-badge[Needs review]",
    "error": ":gray-badge[Error]",
}
RUN_STATUS_BADGES = {"running": ":blue-badge[Running]", "completed": ":green-badge[Completed]", "failed": ":red-badge[Failed]"}
_TRUE = {"yes", "y", "true", "t", "1"}
_FALSE = {"no", "n", "false", "f", "0"}
_LABEL_TO_KEY = {v.lower(): k for k, v in EVAL_CATEGORIES.items()}


def status_badge(status: str | None) -> str:
    return STATUS_BADGES.get(status or "", ":gray-badge[Not available]")


def category_label(key: str | None) -> str:
    return EVAL_CATEGORIES.get(key or "", key or "Not available")


def category_key(value: str | None) -> str | None:
    """Accept a key or a label (case-insensitive); empty -> 'factual'; unknown -> None."""
    v = (value or "").strip()
    if not v:
        return "factual"
    if v in EVAL_CATEGORIES:
        return v
    return _LABEL_TO_KEY.get(v.lower()) or _LABEL_TO_KEY.get(v.lower() + " questions")


def parse_bool(value: str | None, default: bool = True) -> bool | None:
    v = (value or "").strip().lower()
    if not v:
        return default
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return None


def parse_list(value: str | None) -> list[str]:
    """JSON array or `|`-separated values (also used for "one per line" form fields)."""
    v = (value or "").strip()
    if not v:
        return []
    if v.startswith("["):
        data = json.loads(v)
        if not isinstance(data, list) or not all(isinstance(x, (str, int, float)) for x in data):
            raise ValueError("must be a JSON array of strings")
        items = [str(x).strip() for x in data]
    else:
        items = [x.strip() for x in v.replace("\n", "|").split("|")]
    return [x for x in items if x]


def parse_turns(value: str | None) -> list[dict]:
    v = (value or "").strip()
    if not v:
        return []
    data = json.loads(v)
    if not isinstance(data, list):
        raise ValueError("must be a JSON array")
    turns = []
    for t in data:
        if not isinstance(t, dict) or t.get("role") not in ("user", "assistant") or not str(t.get("content", "")).strip():
            raise ValueError('each turn needs "role" (user/assistant) and non-empty "content"')
        turns.append({"role": t["role"], "content": str(t["content"]).strip()})
    return turns


def parse_turn_lines(text: str | None) -> list[dict]:
    """Form field: lines starting with `Q:` (visitor) or `A:` (assistant); other lines continue
    the previous turn."""
    turns: list[dict] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        head = s[:2].upper()
        if head in ("Q:", "A:"):
            turns.append({"role": "user" if head == "Q:" else "assistant", "content": s[2:].strip()})
        elif turns:
            turns[-1]["content"] = (turns[-1]["content"] + " " + s).strip()
        else:
            raise ValueError("start each previous turn with 'Q:' or 'A:'")
    for t in turns:
        if not t["content"]:
            raise ValueError("a previous turn is empty")
    return turns


def turns_to_lines(turns: Sequence[dict] | None) -> str:
    return "\n".join(("Q: " if t.get("role") == "user" else "A: ") + str(t.get("content", "")) for t in turns or [])


def valid_url(url: str) -> bool:
    parts = urlsplit(url if "://" in url else "http://" + url)
    return parts.scheme in ("http", "https") and bool(parts.hostname) and "." in (parts.hostname or "")


def validate_case(fields: dict) -> list[str]:
    """Problems with one case (form or CSV row); [] when it can be saved."""
    errors = []
    q = (fields.get("question") or "").strip()
    if not q:
        errors.append("question is empty")
    elif len(q) > MAX_QUESTION_CHARS:
        errors.append(f"question is longer than {MAX_QUESTION_CHARS} characters")
    if len(fields.get("expected_answer") or "") > MAX_ANSWER_CHARS:
        errors.append(f"expected answer is longer than {MAX_ANSWER_CHARS} characters")
    category = fields.get("category")
    if category not in EVAL_CATEGORIES:
        errors.append(f"unknown category {category!r}")
    if category == "unanswerable" and fields.get("answerable"):
        errors.append("an unanswerable-category case must have answerable = no")
    if fields.get("answerable") and not (fields.get("expected_answer") or "").strip() and not fields.get("key_facts"):
        errors.append("an answerable case needs a verified expected answer or key facts")
    for name in ("source_urls", "key_facts"):
        if len(fields.get(name) or []) > MAX_LIST_ITEMS:
            errors.append(f"more than {MAX_LIST_ITEMS} {name.replace('_', ' ')}")
    bad = [u for u in fields.get("source_urls") or [] if not valid_url(u)]
    if bad:
        errors.append(f"invalid source URL(s): {', '.join(bad[:3])}")
    if category == "follow_up" and not fields.get("previous_turns"):
        errors.append("a follow-up case needs previous turns")
    return errors


def parse_cases_csv(data: bytes | str, *, max_rows: int = MAX_IMPORT_ROWS) -> tuple[list[dict], list[str]]:
    """(valid rows, error messages with line numbers). Rows with errors are not returned."""
    if isinstance(data, bytes):
        if len(data) > MAX_IMPORT_BYTES:
            return [], [f"The file is larger than {MAX_IMPORT_BYTES // 1_000_000} MB."]
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return [], ["The file is not UTF-8 text (save it as 'CSV UTF-8')."]
    else:
        text = data.lstrip("﻿")
    reader = csv.DictReader(io.StringIO(text))
    header = [h.strip().lower() for h in (reader.fieldnames or [])]
    if "question" not in header:
        return [], [f"Missing header row with a 'question' column. Expected columns: {', '.join(CSV_COLUMNS)}."]
    reader.fieldnames = header
    rows, errors, seen = [], [], set()
    for line_no, raw in enumerate(reader, start=2):
        if len(rows) + len(errors) >= max_rows:
            errors.append(f"Only the first {max_rows} rows are read; the rest were ignored.")
            break
        r = {k: (v or "") for k, v in raw.items() if k}
        if not any(str(v).strip() for v in r.values()):
            continue
        problems = []
        category = category_key(r.get("category"))
        if category is None:
            problems.append(f"unknown category {r.get('category')!r}")
        answerable = parse_bool(r.get("answerable"), default=category != "unanswerable")
        if answerable is None:
            problems.append(f"answerable must be yes/no, got {r.get('answerable')!r}")
        source_urls = _parse_cell(parse_list, r, "source_urls", problems)
        key_facts = _parse_cell(parse_list, r, "key_facts", problems)
        turns = _parse_cell(parse_turns, r, "previous_turns", problems)
        fields = {
            "question": r.get("question", "").strip(), "expected_answer": r.get("expected_answer", "").strip(),
            "answerable": bool(answerable), "category": category, "source_urls": source_urls, "key_facts": key_facts,
            "previous_turns": turns, "notes": r.get("notes", "").strip() or None,
        }
        if category is not None and answerable is not None:
            problems += validate_case(fields)
        key = fields["question"].lower()
        if key and key in seen:
            problems.append("duplicate question in this file")
        if problems:
            errors.append(f"Line {line_no}: " + "; ".join(problems))
            continue
        seen.add(key)
        rows.append(fields)
    return rows, errors


def _parse_cell(parser, row: dict, name: str, problems: list[str]) -> list:
    """Parse one list cell; a malformed cell adds a problem and yields an empty list."""
    try:
        return parser(row.get(name))
    except ValueError as exc:  # json.JSONDecodeError is a ValueError
        problems.append(f"{name}: {exc}")
        return []


def _safe_cell(value) -> str:
    """Spreadsheet formula injection guard: a cell starting with = + - @ is prefixed with '."""
    s = "" if value is None else str(value)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


def _write_csv(header: Sequence[str], rows: Iterable[Sequence]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    for row in rows:
        w.writerow([_safe_cell(v) for v in row])
    return buf.getvalue().encode("utf-8-sig")  # BOM so Excel opens it as UTF-8


def cases_to_csv(cases: Sequence[dict]) -> bytes:
    """Export in the import format (round-trips through parse_cases_csv)."""
    return _write_csv(CSV_COLUMNS, (
        [c.get("question"), c.get("expected_answer"), "yes" if c.get("answerable", True) else "no", c.get("category"),
         json.dumps(list(c.get("source_urls") or []), ensure_ascii=False),
         json.dumps(list(c.get("key_facts") or []), ensure_ascii=False),
         json.dumps(list(c.get("previous_turns") or []), ensure_ascii=False) if c.get("previous_turns") else "",
         c.get("notes") or ""]
        for c in cases
    ))


RESULT_COLUMNS = [
    "run_id", "result_id", "test_case_id", "status", "failure_code", "failure_reason", "category", "answerable",
    "question", "expected_answer", "generated_answer", "correctness", "faithfulness", "judge_confidence",
    "key_facts_missing", "recall_at_k", "invalid_citations", "retrieved_urls", "prepare_ms", "generation_ms",
    "judge_ms", "input_tokens", "output_tokens", "judge_model", "rubric_version", "evaluated_at",
]


def _tokens(usage: dict | None, key: str) -> int | str:
    if not usage:
        return ""
    vals = [(u or {}).get(key) for u in usage.values()]
    vals = [v for v in vals if v is not None]
    return sum(vals) if vals else ""


def result_row(r: dict) -> list:
    corr, faith = r.get("correctness") or {}, r.get("faithfulness") or {}
    retr, cites = r.get("retrieval") or {}, r.get("citations") or {}
    recall = retr.get("recall_at_k")
    return [
        r.get("run_id"), r.get("id"), r.get("test_case_id"), r.get("status"), r.get("failure_code") or "",
        r.get("failure_reason") or "", r.get("category"), "yes" if r.get("answerable") else "no", r.get("question"),
        r.get("expected_answer"), r.get("generated_answer") or "", corr.get("verdict") or "",
        faith.get("verdict") or "", corr.get("confidence") or "", " | ".join(corr.get("key_facts_missing") or []),
        "" if recall is None else round(recall, 3), " ".join(f"[{n}]" for n in cites.get("invalid") or []),
        " ".join(s.get("url", "") for s in r.get("retrieved") or []),
        _ms(r.get("prepare_ms")), _ms(r.get("generation_ms")), _ms(r.get("judge_ms")),
        _tokens(r.get("usage"), "input_tokens"), _tokens(r.get("usage"), "output_tokens"),
        r.get("judge_model") or "", r.get("rubric_version") or "",
        r["created_at"].isoformat() if hasattr(r.get("created_at"), "isoformat") else (r.get("created_at") or ""),
    ]


def _ms(v) -> str | float:
    return "" if v is None else round(float(v), 1)


def results_to_csv(results: Sequence[dict]) -> bytes:
    return _write_csv(RESULT_COLUMNS, (result_row(r) for r in results))


def fmt_ms(value) -> str:
    return "Not available" if value is None else f"{float(value):,.0f} ms"


def pass_rate(run: dict) -> float | None:
    """Passed share of graded cases (errors excluded), None when nothing was graded."""
    graded = (run.get("passed") or 0) + (run.get("failed") or 0) + (run.get("needs_review") or 0)
    return (run.get("passed") or 0) / graded if graded else None
