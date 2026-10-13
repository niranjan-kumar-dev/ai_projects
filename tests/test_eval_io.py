"""CSV import/export and form parsing for the RAG Evaluation page (pure)."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone

import pytest

from chat_with_website.ui import eval_io

GOOD = (
    "question,expected_answer,answerable,category,source_urls,key_facts,previous_turns,notes\n"
    'What do you offer?,Web development.,yes,Service Information,https://a.example/services|https://a.example/,"[""web development""]",,from site\n'
    "Do you sell cars?,,no,unanswerable,,,,\n"
    'How much is it?,From $50.,1,follow_up,,$50,"[{""role"": ""user"", ""content"": ""What services?""}]",\n'
)


def test_parse_good_csv():
    rows, errors = eval_io.parse_cases_csv(GOOD.encode("utf-8-sig"))
    assert errors == [] and len(rows) == 3
    assert rows[0]["category"] == "service" and rows[0]["source_urls"] == ["https://a.example/services", "https://a.example/"]
    assert rows[0]["key_facts"] == ["web development"] and rows[0]["notes"] == "from site"
    assert rows[1]["answerable"] is False and rows[1]["category"] == "unanswerable"
    assert rows[2]["previous_turns"] == [{"role": "user", "content": "What services?"}] and rows[2]["key_facts"] == ["$50"]


def test_parse_reports_row_errors_with_line_numbers():
    bad = (
        "question,expected_answer,answerable,category,source_urls\n"
        "Q1,A1,maybe,factual,\n"
        "Q2,A2,yes,Weather,\n"
        "Q3,,yes,factual,\n"
        "Q4,A4,yes,unanswerable,\n"
        "Q5,A5,yes,factual,not a url\n"
        "q1,A,yes,factual,\n"
        "Q1,A,yes,factual,\n"
    )
    rows, errors = eval_io.parse_cases_csv(bad)
    assert [r["question"] for r in rows] == ["q1"]
    text = "\n".join(errors)
    assert "Line 2: answerable must be yes/no" in text and "Line 3: unknown category 'Weather'" in text
    assert "Line 4: an answerable case needs a verified expected answer" in text
    assert "Line 5: an unanswerable-category case must have answerable = no" in text
    assert "Line 6: invalid source URL" in text and "Line 8: duplicate question" in text


def test_parse_rejects_missing_header_bad_encoding_and_huge_files():
    assert "Missing header" in eval_io.parse_cases_csv("a,b\n1,2\n")[1][0]
    assert "not UTF-8" in eval_io.parse_cases_csv(b"question\n\xff\xfe\xfa")[1][0]
    assert "larger than" in eval_io.parse_cases_csv(b"x" * (eval_io.MAX_IMPORT_BYTES + 1))[1][0]
    many = "question,expected_answer\n" + "".join(f"Q{i},A\n" for i in range(5))
    rows, errors = eval_io.parse_cases_csv(many, max_rows=3)
    assert len(rows) == 3 and "first 3 rows" in errors[0]


def test_cases_round_trip_through_export_and_import():
    rows, _ = eval_io.parse_cases_csv(GOOD)
    again, errors = eval_io.parse_cases_csv(eval_io.cases_to_csv(rows))
    assert errors == [] and again == rows


def test_exports_neutralise_spreadsheet_formulas():
    data = eval_io.cases_to_csv([{"question": "=HYPERLINK(\"http://evil\")", "expected_answer": "+1", "answerable": True,
                                  "category": "factual", "source_urls": [], "key_facts": [], "previous_turns": []}])
    assert data.startswith("﻿".encode("utf-8"))
    row = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))[1]
    assert row[0].startswith("'=") and row[1] == "'+1"


def test_results_csv_columns_and_values():
    r = {"run_id": 3, "id": 11, "test_case_id": None, "status": "failed", "failure_code": "key_fact_missing",
         "failure_reason": "Required key facts missing: $50.", "category": "pricing", "answerable": True,
         "question": "Price?", "expected_answer": "$50", "generated_answer": "It is cheap [1].",
         "correctness": {"verdict": "incorrect", "confidence": "high", "key_facts_missing": ["$50"]},
         "faithfulness": {"verdict": "supported"}, "retrieval": {"recall_at_k": 0.5}, "citations": {"invalid": [4]},
         "retrieved": [{"url": "https://a.example/p"}], "prepare_ms": 12.34, "generation_ms": 456.7, "judge_ms": None,
         "usage": {"answer": {"input_tokens": 10, "output_tokens": 5}, "judge": {"input_tokens": 20, "output_tokens": 3}},
         "judge_model": "OpenAI · gpt-4o-mini", "rubric_version": "1.0", "created_at": datetime(2026, 10, 13, tzinfo=timezone.utc)}
    rows = list(csv.DictReader(io.StringIO(eval_io.results_to_csv([r]).decode("utf-8-sig"))))
    assert list(rows[0]) == eval_io.RESULT_COLUMNS
    got = rows[0]
    assert got["status"] == "failed" and got["key_facts_missing"] == "$50" and got["recall_at_k"] == "0.5"
    assert got["invalid_citations"] == "[4]" and got["input_tokens"] == "30" and got["output_tokens"] == "8"
    assert got["judge_ms"] == "" and got["test_case_id"] == ""


def test_form_helpers():
    assert eval_io.parse_list("a\n\n b |c") == ["a", "b", "c"]
    assert eval_io.parse_list('["x", 2]') == ["x", "2"]
    with pytest.raises(ValueError):
        eval_io.parse_list('[{"a": 1}]')
    turns = eval_io.parse_turn_lines("Q: What services?\nA: Web dev\ncontinued [1].")
    assert turns == [{"role": "user", "content": "What services?"}, {"role": "assistant", "content": "Web dev continued [1]."}]
    assert eval_io.parse_turn_lines(eval_io.turns_to_lines(turns)) == turns
    with pytest.raises(ValueError):
        eval_io.parse_turn_lines("no prefix")
    assert eval_io.category_key("Pricing") == "pricing" and eval_io.category_key("factual") == "factual"
    assert eval_io.category_key("Multi-Page") == "multi_page" and eval_io.category_key("") == "factual"
    assert eval_io.category_key("weather") is None
    assert eval_io.validate_case({"question": "Q", "category": "follow_up", "answerable": True, "expected_answer": "A"}) == [
        "a follow-up case needs previous turns"]
    assert eval_io.status_badge("needs_review") == ":orange-badge[Needs review]"
    assert eval_io.pass_rate({"passed": 3, "failed": 1, "needs_review": 0}) == 0.75
    assert eval_io.pass_rate({"passed": 0, "failed": 0, "needs_review": 0}) is None
    assert json.loads(json.dumps(eval_io.CSV_COLUMNS))[0] == "question"
