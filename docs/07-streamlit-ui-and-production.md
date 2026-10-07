# Step 7 – Streamlit UI and production notes

## Run the UI

```powershell
python -m streamlit run src/chat_with_website/ui/streamlit_app.py
```
Opens http://localhost:8501 with three pages (left sidebar):

| Page | What you do there |
|---|---|
| **Websites** | Enter a URL + page limit → watch the crawl progress → see pages/chunks per site. Re-index or delete. |
| **Chatbots** | Create/edit a chatbot: name, instructions, websites (multi-select), chat provider + model, top_k. |
| **Chat** | Pick a chatbot, ask questions, expand **Sources** under each answer, start a new conversation or resume an old one. Toggle *Show retrieved chunks* to debug. |

## How the Streamlit app is built (`ui/streamlit_app.py`)

Streamlit **re-runs the whole script on every interaction**. Three tools make that workable:

* `st.cache_resource` – expensive singletons: the embedding model (`_embeddings`), the RAG chain
  per chatbot (`_chain`), the DB health check. The chain cache is cleared (`_bump_chain_version`)
  whenever a website or chatbot changes.
* `st.session_state` – per-browser-tab memory: `history`, `conversation_id`, `sources`, the
  selected chatbot. Changing chatbot resets the conversation.
* `st.navigation` + `st.Page` – multi-page app from plain functions in one file.

Ingestion runs inside `st.status` with a progress bar fed by the `progress` callback of
`ingest_website`. Errors are caught and shown with `st.error` instead of a stack trace.
Deleting a website requires ticking a confirmation box first.

## Test the whole project

1. **Websites** → index `https://www.python.org/about/` with max pages 10 (or your own site).
   Expect status `ready` and a non-zero chunk count.
2. **Chatbots** → create `python` attached to that website. Save.
3. **Chat** → ask the three-step sequence: a broad question, a follow-up ("tell me more about
   the second point"), a different topic ("how can I contact / join …"). Check that
   *Sources* shows the right page and that the follow-up was understood.
4. Ask something unrelated → the bot must reply that it does not have enough information,
   with no sources.
5. Click **New conversation**, ask again, refresh the browser → pick the conversation from
   *Previous conversations* and continue it.
6. Index a second website, create a second chatbot, and confirm neither bot cites the other's domain.
7. `pytest -q` → all tests pass.

## Production checklist

Already in place: parameterised SQL, URL validation with SSRF protection, robots.txt, request
timeouts and retries, content-type and size limits, batch embedding with retry, per-page
transactions, content-hash skip on re-index, HNSW index, `ANALYZE` after ingest, connection pool,
secrets only in `.env` (git-ignored), temperature 0, fixed refusal sentence, per-chatbot website
isolation enforced in SQL.

Next steps when you go live:

| Topic | Recommendation |
|---|---|
| **Background ingestion** | Crawling 50 pages blocks the Streamlit request. Move `ingest_website` to a worker (RQ/Celery, or a plain `threading.Thread` + status polling via `websites.status`). |
| **JavaScript sites** | Add a Playwright-based fetcher behind a per-website flag; everything after `CrawledPage` stays the same. |
| **Hybrid search** | Add `content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED` + GIN index; combine BM25 and vector ranks with Reciprocal Rank Fusion. Helps with exact names, codes, phone numbers. |
| **Re-ranking** | Retrieve 20, re-rank to 5 with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for better precision. |
| **Better embeddings** | `BAAI/bge-small-en-v1.5` or `bge-m3` (multilingual) are drop-in HuggingFace upgrades; OpenAI/Gemini via `.env`. Remember: change model → `init_db.py --reset-chunks` → re-ingest. |
| **Evaluation** | Keep a `tests/eval_questions.yaml` of question → expected URL pairs per website and measure retrieval hit-rate after every change. |
| **Auth & multi-tenant** | Put the UI behind a login (Streamlit-Authenticator, or a reverse proxy with SSO). Add `owner_id` to `websites`/`chatbots`. |
| **API** | Expose `answer_question` through FastAPI for website widgets; the chain code is UI-agnostic. |
| **Scheduling** | Cron `scripts/ingest.py <url>` nightly; unchanged pages are skipped by content hash. |
| **Database** | Back up PostgreSQL normally – vectors are just a column. Tune `hnsw.ef_search` (default 40) up for recall, down for speed. Pool size ≈ concurrent users. |
| **Observability** | Log `standalone_question`, scores and latency per turn; LangSmith or OpenTelemetry if you want traces. |
| **Deployment** | Docker image with the package + `streamlit run`; `pgvector/pgvector:pg17` or a managed Postgres. Never bake `.env` into the image – pass env vars. |
