# Step 16 – Automated RAG Evaluation

The RAG Debugger (step 15) explains *one* question. This step answers a different question:
**does the chatbot still answer the questions we care about correctly** after a re-crawl, a new
custom instruction, a different model or a new `top_k`? The **RAG Evaluation** page (administrators
only) keeps a library of verified test cases per chatbot, runs them in a batch through the real
chatbot pipeline and grades every answer as **Passed**, **Failed** or **Needs review**, with the
reason, the evidence and the timings stored in PostgreSQL.

## 1. What a run does

For each selected test case, `run_case` in `rag/evaluation.py` runs exactly what the Chat page runs
for that question, with the chatbot's cached chain (`_chain(...)`), so prompt, custom instructions,
model, temperature, `top_k` and score threshold are the real ones:

| Step | Code | Stored as |
|---|---|---|
| Retrieval (and question rewriting when the case has previous turns) | `chain.prepare(question, history)` | `prepare_ms`, `standalone_question`, `retrieved` |
| Answer (one model call, same prompt as streaming) | `chain.answer_once(p)` – new, non-streaming twin of `stream_answer` built from the same `_answer_inputs(p)` | `raw_answer`, `generation_ms`, `usage.answer` |
| Post-processing (citation clean-up, out-of-scope reply) | `chain.finalize(p, raw)` | `generated_answer`, `grounded` |
| Deterministic checks | key facts, exact values, citation integrity, Recall@K | `correctness`, `citations`, `retrieval` |
| LLM judge (separate model call, optional) | `run_judge(...)` | `correctness`, `faithfulness`, `citations.support`, `judge_ms`, `usage.judge` |
| Verdict | `decide_status(...)` | `status`, `failure_code`, `failure_reason` |

Greetings get the canned welcome and questions with nothing above the threshold get the
out-of-scope reply without a model call, exactly as in chat. `persist_turn` and
`suggest_followups` are never called: **test questions never reach `conversations` / `messages`**
(a test asserts this against the real database).

## 2. Metrics

**Answer correctness** – two layers:
- *Deterministic*: the case's **key facts** (exact names, numbers, prices, dates) must appear in the
  answer after normalisation (Unicode NFKC, case, quotes/dashes, citation markers and markdown
  emphasis removed, `1,500` = `1500`, whole-token match so `50` matches `$50` but not `1500`).
  Without key facts, numbers (2+ digits), money, percentages and dates are extracted from the
  expected answer automatically (URLs, e-mail addresses and phone numbers skipped); a missing
  extracted value only ever sends a case to review, never fails it.
- *Semantic*: the judge compares the meaning of the answer with the verified expected answer
  (`correct` / `partially_correct` / `incorrect` / `cannot_judge`), listing missing facts and
  contradictions.

**Faithfulness** – the judge checks every factual claim against the retrieved passages only
(`supported` / `partially_supported` / `unsupported`) and quotes unsupported claims. Confidence
wording in the answer is explicitly not evidence.

**Retrieval quality** – **Recall@K** = share of the case's labelled source URLs that appear among the
retrieved passages (URLs compared without scheme, `www.`, fragment or trailing slash). It is computed
**only** when the case has labelled URLs; otherwise the page says "not computed". It is stored and
shown but never decides the status: the answer may be on a page nobody labelled.

**Citation accuracy** – (1) *integrity*: every `[n]` in the model's **raw** output must be a passage
number that was in its context (`finalize` silently drops invented numbers, so the check runs before
it); (2) *support*: the judge says, per cited number, whether that passage supports the sentence.

**Unanswerable questions** – a case marked *not answerable* passes only when the chatbot declines.

## 3. Status rules (`decide_status`, rubric 1.0)

Ordered; the first matching rule wins.

| # | Situation | Status / code |
|---|---|---|
| 1 | The pipeline raised (network, provider, database) | Error / exception name |
| 2 | Greeting | Passed when the canned welcome came back, else Needs review |
| 3 | Not answerable and the chatbot declined | Passed |
| 4 | Not answerable, answered, judge: unsupported (confidence ≥ medium) | Failed / `hallucination` |
| 5 | Not answerable, answered, anything else | Needs review / `answered_unanswerable` (label or answer is wrong) |
| 6 | Answerable, nothing retrieved above the threshold | Failed / `retrieval_empty` |
| 7 | Answerable, the model declined | Failed / `model_declined` (+ "Recall@K = 0" when labelled) |
| 8 | A key fact is missing but the judge says correct with high confidence | Needs review / `key_fact_conflict` |
| 9 | A key fact is missing | Failed / `key_fact_missing` |
| 10 | Picture question without a cited image | Needs review / `no_image` |
| 11 | Invented citation numbers | Needs review / `invalid_citation` |
| 12 | No expected answer and no key facts | Needs review / `no_expectation` |
| 13 | Judge disabled, failed, unparsable or `cannot_judge` | Needs review / `judge_unavailable` |
| 14 | Judge: correct + supported + high confidence (and no extracted value missing) | **Passed** |
| 15 | Judge: incorrect or unsupported with high confidence | Failed / `judge_incorrect` / `judge_unsupported` |
| 16 | Everything else (partially correct, medium/low confidence, …) | Needs review / `uncertain` |

A pass is downgraded to Needs review when the judge says a cited passage does not support its
sentence (`citation_unsupported`). Key facts alone never produce a pass: without a judge every
graded case goes to review.

**Why so conservative?** An LLM judge is a second model with its own errors. It is used for what
string matching cannot do (meaning, support), but it only passes or fails a case on its own when it
is confident, and any disagreement with the deterministic checks goes to a person.

## 4. The judge

- Model: `EVAL_JUDGE_PROVIDER` / `EVAL_JUDGE_MODEL`, default the `.env` LLM (gpt-4o-mini),
  temperature 0, a different call from generation. Disable with `EVAL_JUDGE_ENABLED=false`.
- Output: a Pydantic `JudgeVerdict` (all fields required, enums as `Literal`s). Providers that
  support it use `with_structured_output(..., include_raw=True)` (OpenAI strict JSON schema, Gemini
  response schema, Ollama `format`); otherwise, or when that fails, plain JSON prompting with code
  fences stripped and one retry. An unusable reply never crashes the run: it becomes
  `judge_unavailable`.
- Prompt injection: question, expected answer, passages and generated answer are placed in tags
  (`<PASSAGES>…</PASSAGES>`), closing tags inside the data are neutralised, and the rubric says the
  tagged content is data, never instructions.
- Versioning: every run and result stores `rubric_version` (`RUBRIC_VERSION = "1.0"`), the judge
  model and a hash of the judge prompt. **Bump `RUBRIC_VERSION` whenever the rubric text or the rules
  change**, so the history never compares results graded by different rules without saying so.

## 5. Database (`schema_upgrades.sql`, stage 16)

| Table | Holds |
|---|---|
| `eval_test_cases` | per chatbot: question (unique per chatbot, case-insensitive), verified expected answer, answerable, category (7 values), `source_urls`, `key_facts`, `previous_turns`, notes, enabled |
| `eval_runs` | one row per batch: status (`running` / `completed` / `failed`), counters updated after every case, `config` (provider, model, temperature, top_k, threshold, embedding model, judge, rubric, `custom_instructions_set` – never the instructions text), error |
| `eval_results` | one row per evaluated case: a **snapshot** of the case, the answers, retrieved passages with scores, the four metric JSON objects, timings, provider token usage, judge model, rubric version |

- **History is preserved**: results copy the case; `test_case_id` is `ON DELETE SET NULL`, so
  editing a case leaves old results untouched (the page says "edited since this run") and deleting
  it keeps them ("case deleted").
- **One running run per chatbot**: a partial unique index (`WHERE status = 'running'`) makes a
  duplicate batch impossible, even from two browser sessions.
- Averages, medians and token sums are computed in SQL (`list_eval_runs`), never stored.
- Deleting a chatbot deletes its test cases and results (cascade); the Chatbots page says so.

## 6. Running a batch

*Run evaluation* creates the run row (refused if one is running), then a background thread evaluates
the cases one at a time and saves each result in its own transaction as soon as it is graded. The
page polls the run row every 2 seconds (`st.fragment(run_every=2)`), so leaving or reloading the page
does not stop the batch, and a crash keeps every result already saved.

Cost and rate limits:
- at most `EVAL_MAX_BATCH` (50) cases per run, `EVAL_CASE_DELAY_SECONDS` (0.5 s) between cases,
  no parallel model calls;
- the SDK retries 429s twice; the evaluator waits 10 s and tries once more; after 3 rate-limited
  cases in a row, or on `insufficient_quota`, the run stops as *failed* with the reason and keeps the
  saved results (the rest show as "not run");
- per case: the retrieval embedding, one answer call and at most one judge call (plus one rewriting
  call for follow-up cases);
- tokens come only from the provider's `usage_metadata`; the cost is shown only when
  `EVAL_PRICE_INPUT_PER_1M` and `EVAL_PRICE_OUTPUT_PER_1M` are set, otherwise "Not available".
  The same prices are applied to answer and judge tokens.
- A run still `running` after `EVAL_STALLED_RUN_HOURS` (process restarted) is closed as stalled
  before the next one starts.

## 7. The page (`page_rag_evaluation`)

- **Dashboard**: cards (test cases, latest run status, passed, failed, needs review, errors, average
  retrieval and generation latency, tokens and cost), *Run evaluation* with a category filter, live
  progress while running, the evaluation history table, pass-rate trend, regressions (cases that
  passed in the previous completed run and not in the latest), results export.
- **Test cases**: create / edit form (question, verified expected answer, category, answerable,
  include in runs, source URLs, key facts, previous turns as `Q:` / `A:` lines, notes); searchable,
  filterable, paginated list with Edit and Delete (Delete needs the *Confirm* tick); CSV import with
  a preview and per-line errors (nothing is imported while a row is invalid unless *Import the valid
  rows only* is ticked); CSV export.
- **Results**: pick a run (its configuration is shown), filter by status / category / text, export
  CSV; one card per case with the failure reason, **expected and generated answers side by side**,
  the four metrics, the retrieved evidence (verbatim, marked as untrusted website content), timings,
  tokens and the raw evaluation JSON.

### CSV format

```text
question,expected_answer,answerable,category,source_urls,key_facts,previous_turns,notes
What do you offer?,Web development and hosting.,yes,Service Information,https://acme.example/services,"[""web development""]",,
Do you sell cars?,,no,unanswerable,,,,
How much is it?,From $50 per month.,yes,follow_up,,$50,"[{""role"": ""user"", ""content"": ""What services?""}]",
```

`answerable` is yes/no (true/false, 1/0); `category` is a key (`pricing`) or a label (`Pricing`);
lists are JSON arrays or `|`-separated; existing questions are updated, new ones added; at most 500
rows / 1 MB per file and 500 cases per chatbot. Exports are UTF-8 with BOM (Excel) and cells that
start with `=`, `+`, `-` or `@` are prefixed with `'` (spreadsheet formula injection).

### Writing good test cases

The evaluator never invents ground truth. Write the expected answer from the website yourself (or
copy it from an indexed page shown in the RAG Debugger), add **key facts** for values that must be
exact, label **source URLs** when you know which page holds the answer, and add a few *unanswerable*
questions so the "declines instead of inventing" behaviour is measured too.

## 8. Code

| Where | What |
|---|---|
| `rag/chain.py` | `_answer_inputs(p)` shared by `stream_answer` and the new `answer_once(p) -> (text, usage)` |
| `rag/citations.py` | `extract_citation_numbers` (raw markers, unvalidated), `strip_citations` |
| `rag/evaluation.py` | `JudgeVerdict`, `JUDGE_SYSTEM_PROMPT`, `RUBRIC_VERSION`, deterministic checks, `run_judge`, `decide_status`, `run_case`, `prepare_run`, `execute_run`, `evaluate_batch`, `start_eval_run_in_background`, `estimate_cost` |
| `ui/eval_io.py` | CSV import/export, form parsing, validation, badges (pure) |
| `db/repository.py` | `*_eval_case(s)`, `start/finish_eval_run`, `bump_eval_run_progress`, `mark_stalled_eval_runs`, `list_eval_runs`, `add_eval_result`, `list_eval_results`, `eval_regressions` |
| `ui/streamlit_app.py` | `page_rag_evaluation` and the `_eval_*` helpers; registered after RAG Debugger |

## Test this step

Automated (no model calls): `pytest -q tests/test_evaluation.py tests/test_eval_io.py` (rules,
judge parsing, fake chains, batch bookkeeping, rate limits, CSV), `pytest -q tests/test_ui.py -k
evaluation` (the page), and with local PostgreSQL `pytest -q tests/test_evaluation_db.py`
(isolation by chatbot, one running run, history after edit/delete, aggregates, no conversation rows).

Manual:
1. Create a case whose answer is on the site, with a key fact and a source URL → Run → Passed,
   Recall@K 1.0.
2. Change its key fact to a value that is not on the site → Run → Failed / key fact missing.
3. Add an unanswerable question → Passed when the chatbot declines.
4. Make an expected answer vague → the judge is unsure → Needs review with its reason.
5. Edit a case after a run → the old result still shows the old expected answer, marked "edited".
6. Delete a case → its old results stay, marked "case deleted".
7. Import a CSV with one bad row → the line number and reason are shown; nothing is imported until
   fixed or *Import the valid rows only* is ticked.
8. Click *Run evaluation*, then reload the page → the progress continues; a second run is refused.
9. Export cases and results → open in a spreadsheet.
10. Open the Chat page → no test question appears in the conversation list.
11. Set `ADMIN_PASSWORD` → the page asks for it.
