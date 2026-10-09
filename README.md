# Chat with Website

A **multi-website RAG (Retrieval-Augmented Generation) chatbot** built with Python, LangChain,
PostgreSQL + pgvector and Streamlit.

Give it a website URL. It crawls the site, cleans the pages, splits them into chunks, turns the
chunks into vector embeddings and stores everything in PostgreSQL. A *chatbot* is then attached
to one or more indexed websites and answers questions **only from that content**, with source
URLs and support for follow-up questions. If the answer is not on the site, it says so.

```
Website URL ─► Crawl ─► Extract & clean ─► Chunk ─► Embed ─► PostgreSQL / pgvector
                                                                     │
question + chat history ─► standalone question ─► embed ─► vector search (this chatbot's websites only)
                                                                     │
                                   LLM answers from retrieved chunks ─► answer + source URLs
```

## Status

| Stage | What | Status |
|---|---|---|
| 0 | Architecture, package layout, config | ✅ |
| 1 | PostgreSQL + pgvector schema and `init_db.py` | ✅ |
| 2 | Crawler + content extraction | ✅ |
| 3 | Chunking, embeddings, storage (`ingest.py`) | ✅ |
| 4 | Chatbots + filtered vector search (`create_chatbot.py`) | ✅ |
| 5 | RAG chain with conversation memory and sources (`ask.py`) | ✅ |
| 6 | Streamlit UI (Websites / Chatbots / Chat) | ✅ |
| 7 | Production notes and hardening checklist | ✅ |

The step-by-step guides live in [`docs/`](docs/00-architecture.md) – each explains the concept,
the code and how to test before moving on.

## Providers

| Role | Default | Alternatives (switch in `.env`) |
|---|---|---|
| Embeddings `EMBEDDING_PROVIDER` | `huggingface` – all-MiniLM-L6-v2, local, 384 dims | `openai` (text-embedding-3-small), `gemini` (gemini-embedding-001) |
| Chat LLM `LLM_PROVIDER` | `openai` – gpt-4o-mini (needs `OPENAI_API_KEY`) | `ollama` (llama3.2, free, local), `gemini` (gemini-3.8-flash) |

The chat LLM can be changed at any time. Changing the **embedding** provider changes the vector
size, so run `python scripts/init_db.py --reset-chunks` and re-ingest your websites.

## Setup

Requirements: Python 3.11 (conda env `chat-with-website`), PostgreSQL 16/17 with the pgvector
extension (already installed on this machine as service `postgresql-x64-17`), and
an `OPENAI_API_KEY` for the default chat model (or [Ollama](https://ollama.com) with `llama3.2`
pulled if you set `LLM_PROVIDER=ollama`).

```powershell
conda activate chat-with-website
pip install -r requirements.txt
pip install -e .

copy .env.example .env     # set LOCAL_DATABASE_URL (your postgres password) and optional API keys
python scripts/init_db.py  # creates the database, enables pgvector, creates tables
```

If `conda activate` does not work in PowerShell, call the interpreter directly:
`& "C:\Anaconda3\envs\chat-with-website\python.exe" scripts/init_db.py`.

## Run the UI

```powershell
python -m streamlit run src/chat_with_website/ui/streamlit_app.py
```
Then in the browser (http://localhost:8501):

1. **Websites** – enter a URL (e.g. `https://www.python.org/about/`), set max pages, click *Index website* and watch the progress.
2. **Chatbots** – create a chatbot, attach one or more indexed websites, pick the chat provider (Ollama / OpenAI / Gemini).
3. **Chat** – select the chatbot and ask questions. Every answer has a *Sources* list with the page URLs; follow-up questions use the conversation; unrelated questions get "I don't have enough information…". Conversations are saved and can be resumed from the sidebar.

## Command-line usage

```powershell
python scripts/ingest.py https://www.python.org/about/ --dry-run --max-pages 6   # crawl + clean only
python scripts/ingest.py https://www.python.org/about/ --name "Python.org" --max-pages 20
python scripts/ingest.py --list

python scripts/create_chatbot.py --name python --websites 1
python scripts/create_chatbot.py --list

python scripts/ask.py --chatbot 1 --search-only "how do I get started"   # inspect retrieval
python scripts/ask.py --chatbot 1 --debug                                 # interactive chat
python scripts/ask.py --chatbot 1 -q "What is Python good for?"

pytest -q                                                                  # unit tests (no DB/LLM needed)
```

## Test the project end to end

1. `python scripts/init_db.py` → tables created, pgvector 0.8.x reported.
2. Index a website (UI or `ingest.py`); re-run it → `unchanged (skipped)` count equals the page count, chunk count unchanged.
3. Create a chatbot attached to it.
4. Ask a broad question → follow-up ("tell me more about the second one") → a different topic ("how can I contact them?"). Sources must point at the right pages.
5. Ask something unrelated → refusal, no sources.
6. Index a second website + second chatbot → neither cites the other's domain.

## Configuration (`.env`)

| Key | Default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | `development` → `LOCAL_DATABASE_URL`, `production` → `PRODUCTION_DATABASE_URL` |
| `LOCAL_DATABASE_URL` | – | PostgreSQL connection string for your machine |
| `PRODUCTION_DATABASE_URL` | – | PostgreSQL connection string for the deployed app (set via Streamlit secrets / env vars) |
| `DATABASE_URL` | – | Optional: overrides both of the above |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `huggingface` / per-provider default | Embedding model (vector size must match the `chunks` table) |
| `LLM_PROVIDER` / `LLM_MODEL` | `openai` / `gpt-4o-mini` | Chat model (`ollama` and `gemini` also supported) |
| `OPENAI_API_KEY`, `GOOGLE_API_KEY` | – | `OPENAI_API_KEY` is required for the default chat model; `GOOGLE_API_KEY` only for Gemini |
| `USER_AGENT` | `ChatWithWebsiteBot/0.1 ...` | Identifies the crawler to websites |
| `CRAWL_MAX_PAGES`, `CRAWL_MAX_DEPTH`, `CRAWL_DELAY_SECONDS`, `CRAWL_RESPECT_ROBOTS` | 50, 3, 0.5, true | Crawl limits and politeness |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | 800, 120 | Chunking (characters) |
| `TOP_K`, `SCORE_THRESHOLD` | 5, 0.25 | Retrieval |
| `LOG_LEVEL` | `INFO` | Logging |

See `.env.example` for the full list with comments.

## Project layout

```
scripts/                 CLI entry points (init_db.py, ingest.py, ...)
src/chat_with_website/
  config.py              settings from .env
  db/                    schema.sql, connection pool, repository (all SQL)
  ingestion/             url_utils, crawler, extractor, chunker, embedder, pipeline
  rag/                   (stage 4-5) retriever, prompts, chain
  ui/                    (stage 6) Streamlit app
tests/                   pytest unit tests
docs/                    step-by-step guides
```

## Test URLs

Server-rendered sites work best; JavaScript-only sites return little text.

- https://www.python.org/about/
- https://docs.python.org/3/tutorial/
- http://paulgraham.com/articles.html
