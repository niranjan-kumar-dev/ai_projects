# Step 15 – RAG Retrieval Debugger

"The answer is on the website, so why did the chatbot get it wrong?" Almost always one of three
things: the right passage was never retrieved, it was retrieved but scored below the threshold
and dropped, or it reached the model and the model still answered badly. The **RAG Debugger**
page (administrators only) shows which one it was for any test question, with real timings,
without writing anything to the database and without touching the chatbot's settings.

## 1. What it runs

The page calls the chatbot's own retrieval. `rag/retriever.py` now has one function,
`search_chunks_timed`, used by the chain's `PgVectorChatbotRetriever`, by `search_chunks`
(CLI) and by the debugger, so the debugger cannot drift from what the chatbot does. The only
differences from a live chat turn:

| | Chat page | Debugger |
|---|---|---|
| Question rewriting | only when there is history (one model call) | none: the test is a first question, `standalone = question` |
| `top_k` | the chatbot's saved value | the value chosen for the test (3 / 5 / 8 / 10 / saved) |
| Threshold | passages with `score < SCORE_THRESHOLD` are dropped before the model is called | the same passages are **shown and flagged** "dropped – below threshold" |
| Persistence | question, answer and sources are stored in `messages` | nothing is stored; results live in `st.session_state` |

The flag is exact, not an approximation: `similarity_search` applies `LIMIT k` in SQL and the
threshold in Python afterwards, so retrieving with `min_score = -1` returns precisely the k rows
the chain sees before it filters. Chunks are listed in the order the database returned them,
which is the `[1]..[k]` numbering the model sees in its context; nothing is re-sorted.

Picture questions (`is_image_question`) run the same two searches as the chain (text, then
images) with the question embedded once, images listed first.

## 2. Reading the scores

`score = 1 − (embedding <=> query)` where `<=>` is pgvector's **cosine distance**. So the score
is the cosine similarity: **higher is better**, `1.0` means the query vector equals the stored
vector, `0` means unrelated, and negative values are possible (that is why the debugger retrieves
with `min_score = -1`). The page states the metric and its direction next to the embedding model.

Two honesty notes shown on the page:

- With more than 2000 dimensions (e.g. Gemini's 3072, OpenAI `text-embedding-3-large`) the HNSW
  index and the `ORDER BY` use `halfvec` (16-bit floats). A self-match then reads ~0.999 instead
  of 1.000 and two close passages can swap places in the third decimal.
- The score is a similarity, **not** a probability that the answer is right. A high score on an
  irrelevant passage happens; a correct passage with 0.3 happens. The page never converts the
  score into a percentage or a "confidence".

The stored vector was computed over *title + content* (`Chunk.embed_text`); the card shows the
content the model receives. HNSW with `hnsw.iterative_scan = relaxed_order` is approximate, and
hosted embedding APIs are not bit-reproducible, so the fourth decimal can differ between runs.

## 3. The page (`page_rag_debugger` in `ui/streamlit_app.py`)

1. **Chatbot** (only bots with websites) – the caption shows its saved `top_k`, the threshold,
   the LLM (`describe_llm`), whether custom instructions are set (never the text) and the
   website ids it searches. Website scoping comes from `chatbot_websites`, exactly as in chat.
2. **Form**: test question, **Top-K for this test** (3 / 5 / 8 / 10 plus the saved value,
   defaulting to the saved value), **Compare top_k values**, *Run retrieval test*.
3. **Header cards** (measured with `perf_counter`): chunks retrieved, kept for the model,
   embedding ms, vector search ms (the SQL round trip(s)), total retrieval ms (includes the
   website lookup and the pool checkout), context size in characters with an estimated token
   count (`chars ÷ 4`, labelled as an estimate). A note appears when the tested `top_k` differs
   from the saved one, and when the question is a bare greeting (the chatbot would not retrieve).
4. **Retrieved chunks** – one bordered card per chunk: rank, kept / dropped badge, score, page
   title linked to the URL, chunk id, database id, website id, type, length; the text with the
   question's terms highlighted, collapsed into an expander beyond 600 characters. Image chunks
   show the image. Missing title or URL reads **Not available**.
5. **Top-K comparison** (optional): the question is embedded **once** and one real search runs
   per `top_k`; the table shows chunks returned / kept / dropped, distinct pages, context size,
   SQL ms and the lowest and highest score. Nothing is sliced from the largest result because
   the HNSW scan is approximate.
6. **Answer accuracy** – *Generate test answer* runs the kept chunks through the chatbot's
   cached chain (`_chain`, the same object the Chat page uses, so model, temperature and custom
   instructions are the real ones): `Prepared(history=[]) → stream_answer → finalize`, one model
   call, timed. Two columns keep the **generated answer** apart from the **retrieved
   evidence**; each kept chunk is marked *cited in answer* or *not cited*, and an expander shows
   the exact context string sent. No chunk above the threshold → the out-of-scope reply with no
   model call; a greeting → the canned welcome with no model call. Follow-up suggestions are not
   generated and nothing is persisted.

### How top_k testing works

`top_k` is only the `k` argument of the test's `similarity_search` call. The chatbot row is never
read for writing and `repo.update_chatbot` is never called from this page; the Chatbots page
remains the only place that changes a chatbot. The UI tests assert that `update_chatbot` and
`persist_turn` are never invoked by the debugger.

### Security

The page sits behind `_require_admin` (same `ADMIN_PASSWORD` prompt as Websites and Chatbots)
and `_require_schema`. It shows provider and model names only – no API keys, no connection
strings (`database_label` is host/db only, shown elsewhere), no system prompt text. All SQL is
the existing parameterised `similarity_search` / `get_chatbot_website_ids`.

## 4. Code

| Where | What |
|---|---|
| `rag/retriever.py` | `search_chunks_timed(chatbot_id, query, k, min_score, kinds, *, vector) -> (docs, {"embed_ms", "search_ms"})`; `search_chunks` delegates to it; `row_to_document` (public name) |
| `rag/debug.py` | `RetrievedChunk`, `RetrievalTest`, `AnswerTest`; `run_retrieval_test`, `compare_top_k`, `compare_rows`, `generate_test_answer`, `cited_ranks`, `metric_label`. No Streamlit, no writes |
| `ui/streamlit_app.py` | `page_rag_debugger` + `_rdbg_header`, `_rdbg_chunk_card`, `_rdbg_compare_table`, `_rdbg_answer_section`; registered in `st.navigation` after Chatbots |

Cost: one embedding call per *Run retrieval test* (also one for the comparison) and one chat
model call per *Generate test answer*.

## Test this step

Automated: `pytest -q tests/test_rag_debug.py tests/test_ui.py` (no database, no model) and,
with the local PostgreSQL running, `pytest -q tests/test_rag_debug_db.py`, which stores random
vectors, uses one stored vector as the query (no embedding API) and checks that it comes back
first with score ≈ 1.0, that the text equals the stored row, that `top_k` limits hold and that a
website not attached to the chatbot never appears.

Manual:
1. Ask a question whose answer is on the site: the right passage is kept and the test answer
   cites it.
2. Ask something the site does not cover: low scores, dropped chunks or none; *Generate test
   answer* returns the out-of-scope reply with "model called: no".
3. Run the comparison for 3 / 5 / 8 / 10 and see how context size and the lowest score change.
4. Every URL on the cards belongs to the chatbot's websites.
5. Open a chunk's URL: the card text is on that page.
6. A chunk whose page has no title shows **Not available**.
7. A chatbot whose websites have no embeddings shows "No chunks came back".
8. Compare retrieval ms with generation ms.
9. Open the Chatbots page afterwards: `top_k` is unchanged.
10. Set `ADMIN_PASSWORD`: the page asks for it; the Chat page is unchanged.
