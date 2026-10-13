"""RAG evaluation storage against the real local database (no model API calls).

Covers what only PostgreSQL can prove: chatbot isolation in every query, the one-running-run
partial unique index, history kept when a case is edited or deleted, CSV upserts, and the
latency / token aggregates of the history table.
"""

from __future__ import annotations

import psycopg
import pytest

import chat_with_website.rag.evaluation as ev
from chat_with_website.config import settings
from chat_with_website.db import repository as repo

pytestmark = pytest.mark.skipif(settings.app_env != "development", reason="only against the local database")
PREFIX = "zz-test eval "


@pytest.fixture
def db():
    try:
        from chat_with_website.db.connection import check_connection, ensure_schema_upgrades, get_conn

        check_connection()
        ensure_schema_upgrades()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"PostgreSQL not reachable: {exc}")
    with get_conn() as conn:
        conn.execute("DELETE FROM chatbots WHERE name LIKE %s", (PREFIX + "%",))
    yield get_conn
    with get_conn() as conn:
        conn.execute("DELETE FROM chatbots WHERE name LIKE %s", (PREFIX + "%",))  # cascades to eval_*


@pytest.fixture
def bots(db):
    with db() as conn:
        a = repo.create_chatbot(conn, name=PREFIX + "a", top_k=3)["id"]
        b = repo.create_chatbot(conn, name=PREFIX + "b")["id"]
    return a, b


def _case(conn, bot, q, **kw):
    kw.setdefault("expected_answer", "Web development.")
    return repo.create_eval_case(conn, bot, question=q, **kw)


def test_cases_crud_is_scoped_to_the_chatbot(db, bots):
    a, b = bots
    with db() as conn:
        c1 = _case(conn, a, "What services?", source_urls=["https://a.example/s"], key_facts=["web development"],
                   previous_turns=[{"role": "user", "content": "hi"}], category="service")
        _case(conn, a, "Pricing?", category="pricing", expected_answer="From $50.")
        _case(conn, a, "Do you sell cars?", category="unanswerable", answerable=False, expected_answer="")
        cb = _case(conn, b, "What services?")  # same question, other chatbot: allowed
        assert c1["source_urls"] == ["https://a.example/s"] and c1["previous_turns"][0]["role"] == "user"
        with pytest.raises(psycopg.errors.UniqueViolation):
            with conn.transaction():
                _case(conn, a, "WHAT SERVICES?")  # duplicate (case-insensitive) within one chatbot
        # another chatbot's id cannot be read, edited or deleted through chatbot a
        assert repo.get_eval_case(conn, cb["id"], a) is None
        assert repo.update_eval_case(conn, cb["id"], a, question="hijack", expected_answer="x") is None
        assert repo.delete_eval_case(conn, cb["id"], a) is False
        assert [c["id"] for c in repo.get_eval_cases(conn, a, [c1["id"], cb["id"]])] == [c1["id"]]
        rows, total = repo.list_eval_cases(conn, a, search="servic")
        assert total == 1 and rows[0]["id"] == c1["id"]
        rows, total = repo.list_eval_cases(conn, a, category="pricing")
        assert total == 1 and rows[0]["question"] == "Pricing?"
        assert repo.list_eval_cases(conn, a, answerable=False)[1] == 1
        rows, total = repo.list_eval_cases(conn, a, limit=2, offset=2)
        assert total == 3 and len(rows) == 1
        assert repo.count_eval_cases(conn, a) == {"total": 3, "enabled": 3}
        with pytest.raises(ValueError):
            _case(conn, a, "Bad", category="unanswerable", answerable=True)
        with pytest.raises(psycopg.errors.CheckViolation):  # the table enforces it too
            with conn.transaction():
                conn.execute("INSERT INTO eval_test_cases (chatbot_id, question, category, answerable) "
                             "VALUES (%s, 'x', 'unanswerable', TRUE)", (a,))
        updated = repo.update_eval_case(conn, c1["id"], a, question="What services do you offer?",
                                        expected_answer="Web dev and cloud.", category="service", enabled=False)
        assert updated["updated_at"] >= c1["updated_at"] and updated["enabled"] is False


def test_csv_upsert_updates_existing_questions_and_enforces_the_limit(db, bots, monkeypatch):
    a, _ = bots
    with db() as conn:
        _case(conn, a, "What services?", expected_answer="old")
        n = repo.upsert_eval_cases(conn, a, [
            {"question": "what services?", "expected_answer": "new", "answerable": True, "category": "service"},
            {"question": "Opening hours?", "expected_answer": "9-5", "answerable": True, "category": "factual",
             "key_facts": ["9-5"]},
        ])
        assert n == 2
        cases = repo.list_all_eval_cases(conn, a)
        assert [(c["question"], c["expected_answer"]) for c in cases] == [("What services?", "new"), ("Opening hours?", "9-5")]
    monkeypatch.setattr(repo, "MAX_EVAL_CASES_PER_CHATBOT", 2)
    with pytest.raises(ValueError, match="limit"):
        with db() as conn:
            repo.upsert_eval_cases(conn, a, [{"question": "Third?", "expected_answer": "x", "answerable": True,
                                              "category": "factual"}])
    with db() as conn:  # rolled back as a whole
        assert repo.count_eval_cases(conn, a)["total"] == 2


def _record(case_row, status="passed", **over):
    rec = ev.CaseOutcome(case=ev.EvalCase.from_row(case_row), status=status, generated_answer="We build sites [1].",
                         grounded=True, model_called=True, prepare_ms=40.0, generation_ms=900.0, judge_ms=500.0,
                         retrieved=[{"n": 1, "url": "https://a.example/s", "title": "S", "score": 0.8, "cited": True}],
                         correctness={"verdict": "correct"}, citations={"checked": True, "cited": [1], "invalid": []},
                         usage={"answer": {"input_tokens": 1000, "output_tokens": 50}}).to_record()
    rec.update(over)
    return rec


def test_runs_results_history_and_aggregates(db, bots):
    a, b = bots
    with db() as conn:
        c1 = _case(conn, a, "What services?")
        c2 = _case(conn, a, "Pricing?", category="pricing")
        run1 = repo.start_eval_run(conn, a, total_cases=2, config={"top_k": 3}, judge_model="OpenAI · gpt-4o-mini",
                                   rubric_version="1.0", judge_prompt_hash="abc")
    with pytest.raises(psycopg.errors.UniqueViolation):  # second running run for the same chatbot
        with db() as conn:
            repo.start_eval_run(conn, a, total_cases=1, config={}, judge_model=None, rubric_version="1.0")
    with db() as conn:
        other = repo.start_eval_run(conn, b, total_cases=0, config={}, judge_model=None, rubric_version="1.0")  # other bot: fine
        assert repo.get_running_eval_run(conn, a)["id"] == run1
        for row, status in ((c1, "passed"), (c2, "passed")):
            repo.add_eval_result(conn, run1, a, _record(row, status))
            repo.bump_eval_run_progress(conn, run1, status, row["question"])
        repo.finish_eval_run(conn, run1, status="completed")
        run2 = repo.start_eval_run(conn, a, total_cases=2, config={}, judge_model=None, rubric_version="1.0")
        repo.add_eval_result(conn, run2, a, _record(c1, "passed", generation_ms=1100.0))
        repo.add_eval_result(conn, run2, a, _record(c2, "failed", failure_code="judge_incorrect",
                                                    failure_reason="Incorrect: price differs.", usage=None))
        for s in ("passed", "failed"):
            repo.bump_eval_run_progress(conn, run2, s)
        repo.finish_eval_run(conn, run2, status="completed")
        repo.finish_eval_run(conn, run2, status="failed")  # no effect: only a running run can be finished

        # edit c1 and delete c2: history keeps the snapshot
        repo.update_eval_case(conn, c1["id"], a, question="What services do you offer?", expected_answer="Changed.")
        repo.delete_eval_case(conn, c2["id"], a)
        res = repo.list_eval_results(conn, run2, a)
        assert [r["question"] for r in res] == ["What services?", "Pricing?"]
        assert res[0]["expected_answer"] == "Web development." and res[0]["case_current_updated_at"] != res[0]["case_updated_at"]
        assert res[1]["test_case_id"] is None and res[1]["case_exists"] is False
        assert res[1]["retrieved"][0]["url"] == "https://a.example/s" and res[1]["usage"] is None
        assert repo.list_eval_results(conn, run2, b) == []  # scoped
        assert [r["question"] for r in repo.list_eval_results(conn, run2, a, status="failed")] == ["Pricing?"]
        assert [r["question"] for r in repo.list_eval_results(conn, run2, a, search="pric")] == ["Pricing?"]

        runs = repo.list_eval_runs(conn, a)
        assert [r["id"] for r in runs] == [run2, run1]
        r2 = runs[0]
        assert (r2["status"], r2["done"], r2["passed"], r2["failed"]) == ("completed", 2, 1, 1)
        assert r2["avg_generation_ms"] == pytest.approx(1000.0) and r2["median_generation_ms"] == pytest.approx(1000.0)
        assert r2["answer_input_tokens"] == 1000 and r2["results_with_usage"] == 1
        assert runs[1]["last_question"] == "Pricing?"
        # c2 was deleted, so no regression can be attributed to it; c1 passed both times
        assert repo.eval_regressions(conn, a, run2) == []
        assert repo.get_eval_run(conn, run1, b) is None and repo.get_eval_run(conn, other, b)["status"] == "running"


def test_regressions_and_stalled_runs(db, bots):
    a, _ = bots
    with db() as conn:
        c1 = _case(conn, a, "What services?")
        run1 = repo.start_eval_run(conn, a, total_cases=1, config={}, judge_model=None, rubric_version="1.0")
        repo.add_eval_result(conn, run1, a, _record(c1, "passed"))
        repo.finish_eval_run(conn, run1, status="completed")
        run2 = repo.start_eval_run(conn, a, total_cases=1, config={}, judge_model=None, rubric_version="1.0")
        repo.add_eval_result(conn, run2, a, _record(c1, "failed", failure_reason="Incorrect."))
        repo.finish_eval_run(conn, run2, status="completed")
        reg = repo.eval_regressions(conn, a, run2)
        assert [(r["test_case_id"], r["previous_run_id"]) for r in reg] == [(c1["id"], run1)]

        run3 = repo.start_eval_run(conn, a, total_cases=1, config={}, judge_model=None, rubric_version="1.0")
        assert repo.mark_stalled_eval_runs(conn, a, 1) == 0  # just started
        conn.execute("UPDATE eval_runs SET started_at = now() - interval '2 hours' WHERE id = %s", (run3,))
        assert repo.mark_stalled_eval_runs(conn, a, 1) == 1
        stalled = repo.get_eval_run(conn, run3, a)
        assert stalled["status"] == "failed" and "Stalled" in stalled["error"]
        assert repo.get_running_eval_run(conn, a) is None


def test_evaluate_batch_end_to_end_on_the_real_tables(db, bots, monkeypatch):
    """The real repository + a fake chain/judge: rows land in eval_* and nothing in conversations."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from chat_with_website.rag.chain import RagChain
    from tests.test_chain import DOCS, FakeRetriever
    from tests.test_evaluation import VERDICT_OK, judge_model

    a, _ = bots
    with db() as conn:
        c1 = _case(conn, a, "What services do you offer?", expected_answer="Web development and cloud consulting.")
        c2 = _case(conn, a, "Do you sell cars?", category="unanswerable", answerable=False, expected_answer="")
        convs = conn.execute("SELECT count(*) AS n FROM conversations").fetchone()["n"]
    monkeypatch.setattr(ev, "_sleep", lambda s: None)
    chain = RagChain(FakeRetriever(docs=DOCS), FakeListChatModel(responses=[
        "We offer web development and cloud consulting [1].",
        "I don't have enough information in the indexed website content to answer that."]), chatbot_name="Acme")
    s = ev.evaluate_batch(a, [c1["id"], c2["id"]], chain=chain, judge=judge_model(VERDICT_OK))
    assert (s.status, s.passed, s.done) == ("completed", 2, 2)
    with db() as conn:
        run = repo.get_eval_run(conn, s.run_id, a)
        assert (run["status"], run["done"], run["passed"]) == ("completed", 2, 2)
        assert run["config"]["top_k"] == 3 and run["rubric_version"] == ev.RUBRIC_VERSION
        res = repo.list_eval_results(conn, s.run_id, a)
        assert [r["status"] for r in res] == ["passed", "passed"] and res[0]["citations"]["cited"] == [1]
        assert conn.execute("SELECT count(*) AS n FROM conversations").fetchone()["n"] == convs
