# Step 8 – Project walkthrough: which file does what, and where to change things

Read this when you want to maintain or modify the project yourself.

## 1. Folders and files

One rule holds the project together: **scripts and the UI are thin entry points; all real logic
lives in the package `src/chat_with_website/`**. The database stores the results. The same package
code is used by the command line, the tests and the Streamlit app.

```
CHAT_WITH_WEBSITE/
├── .env                      ← YOUR settings and secrets (password, API keys). Read by config.py
├── .env.example              ← template for .env with comments
├── requirements.txt          ← pip packages
├── pyproject.toml            ← makes `chat_with_website` importable (pip install -e .), pytest config
├── README.md
│
├── docs/                     ← learning material only, no code runs from here
│
├── scripts/                  ← things YOU run from the terminal
│   ├── init_db.py            create database + tables            (run once)
│   ├── ingest.py             index a website                     (per website)
│   ├── create_chatbot.py     create a chatbot, attach websites   (per chatbot)
│   └── ask.py                chat in the terminal                (testing)
│
├── src/chat_with_website/    ← THE APPLICATION
│   ├── config.py             reads .env → `settings` object used everywhere
│   ├── logging_config.py     log format
│   ├── db/
│   │   ├── schema.sql        tables + indexes (the database design)
│   │   ├── connection.py     opens the connection pool to PostgreSQL
│   │   └── repository.py     EVERY SQL query (insert, search, delete…)
│   ├── ingestion/            "website → database" pipeline
│   │   ├── models.py         small data holders: CrawledPage, PageContent, Chunk, IngestResult
│   │   ├── url_utils.py      clean URLs, same-domain check, block private IPs
│   │   ├── crawler.py        download pages (robots.txt, sitemap, link following)
│   │   ├── extractor.py      HTML → clean text, remove menus/footers
│   │   ├── chunker.py        text → ~800-char chunks with metadata
│   │   ├── embedder.py       chunks → vectors (HuggingFace / OpenAI / Gemini)
│   │   └── pipeline.py       runs the steps above in order
│   ├── rag/                  "question → answer" pipeline
│   │   ├── llm.py            picks the chat model (Ollama / OpenAI / Gemini)
│   │   ├── prompts.py        the two prompt texts (rewrite question, answer)
│   │   ├── retriever.py      vector search for one chatbot
│   │   └── chain.py          glues rewrite → retrieve → answer → sources
│   └── ui/
│       └── streamlit_app.py  the web UI (Websites / Chatbots / Chat pages)
│
└── tests/                    pytest unit tests (run without DB or LLM)
```

**Which file runs first?** Whatever you type in the terminal: `scripts/init_db.py`,
`scripts/ingest.py`, `scripts/ask.py`, or `streamlit run src/chat_with_website/ui/streamlit_app.py`.
Each of those imports `config.py` first (so `.env` is loaded), then calls into `ingestion/` or
`rag/`. Nothing in `src/` ever runs on its own.

Three files you will touch most often:

| Question | File |
|---|---|
| Where is `.env` loaded? | `src/chat_with_website/config.py`. The `Settings` class lists every setting with its default. `settings = Settings()` at the bottom reads `.env` once. Everywhere else does `from chat_with_website.config import settings`. |
| Where is the database connection created? | `src/chat_with_website/db/connection.py`. `get_pool()` opens a psycopg connection pool using `settings.database_url`. `get_conn()` hands out one connection and commits or rolls back for you. |
| Where are the SQL queries? | `src/chat_with_website/db/repository.py`, and nowhere else. Functions like `upsert_page`, `insert_chunks`, `similarity_search`, `create_chatbot`, `add_message`. |

## 2. The two flows

### Flow A – indexing a website (`scripts/ingest.py` or the Websites page)

```
python scripts/ingest.py https://site.com
  │
  ▼
scripts/ingest.py        cmd_ingest()           parses arguments, prints the summary
  │
  ▼
ingestion/pipeline.py    ingest_website(url)    the conductor
  ├─ url_utils.py        validate_start_url()   http(s) only, not a private IP
  ├─ db/repository.py    upsert_website()       websites row, status='crawling'
  │
  ├─ crawl_and_extract()
  │    ├─ crawler.py     Crawler.crawl()        robots.txt → sitemap → BFS over same-domain links
  │    │                                         yields CrawledPage(url, html)
  │    ├─ extractor.py   extract_page()         HTML → PageContent(title, text, content_hash)
  │    └─ extractor.py   remove_boilerplate()   drop lines repeated on ≥30 % of pages
  │
  └─ for each page:
       ├─ db/repository.py  upsert_page()       skip if content_hash unchanged
       ├─ chunker.py        chunk_page()        → list[Chunk] with metadata {url, title, chunk_id…}
       ├─ embedder.py       embed_texts()       → list of 384-number vectors
       └─ db/repository.py  insert_chunks()     INSERT INTO chunks (content, embedding, metadata)
  │
  ▼
PostgreSQL: websites → pages → chunks (vector column + HNSW index)
```

Data shape at each step: a URL string → `CrawledPage` (raw HTML) → `PageContent` (clean text)
→ `list[Chunk]` (text pieces + metadata) → the same chunks with `.embedding` filled → rows in the
`chunks` table.

### Flow B – answering a question (Chat page or `scripts/ask.py`)

```
User types a question in the Chat page
  │
  ▼
ui/streamlit_app.py      page_chat()            adds message to session_state, calls answer_question()
  │
  ▼
rag/chain.py             answer_question()      builds/reuses a RagChain for this chatbot
  ├─ db/repository.py    get_chatbot()          which LLM, which system prompt, top_k
  ├─ rag/llm.py          get_chat_model()       Ollama / OpenAI / Gemini object
  │
  └─ RagChain.invoke(question, history)
       ├─ trim_history()                         keep the last 6 exchanges
       ├─ condense_question()                    prompts.CONDENSE_QUESTION_PROMPT → LLM
       │                                         "tell me more about the second one"
       │                                         → "what is included in managed hosting?"
       ├─ rag/retriever.py  search_chunks()
       │    ├─ db/repository.py get_chatbot_website_ids()   which websites this bot may read
       │    ├─ ingestion/embedder.py embed_query()          question → vector (SAME model as indexing)
       │    └─ db/repository.py similarity_search()         SELECT … WHERE website_id = ANY(…)
       │                                                     ORDER BY embedding <=> query LIMIT k
       ├─ format_context()                       "[1] Title (url)\n text …"
       ├─ prompts.ANSWER_PROMPT → LLM            "answer ONLY from the passages, cite [n],
       │                                          otherwise say 'not enough information'"
       └─ build_sources()                        which [n] the answer actually cited
  │
  ▼
RagAnswer(answer, sources, standalone_question, documents)
  │
  ├─ db/repository.py    add_message() ×2        saved to conversations / messages
  ▼
ui/streamlit_app.py      st.markdown(answer) + "Sources" expander
```

The one idea to hold on to: the embedding model is used twice, once in Flow A for chunks and
once in Flow B for the question. The search only works because both sides use the same model,
which is why `websites` records the model and why changing it requires re-indexing.

The Streamlit file itself is three functions, `page_websites`, `page_chatbots`, `page_chat`,
wired together by `st.navigation` at the bottom. The Websites page calls the same
`ingest_website` as the script. Chat history lives in `st.session_state` while the tab is open
and in the `messages` table permanently.

## 3. Where to change what

| I want to… | Edit | Notes |
|---|---|---|
| Change chunk size or overlap | `.env`: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=120` | Logic in `ingestion/chunker.py` (`build_splitter`, separators list). Re-index afterwards. Keep under ~1000 chars for MiniLM. |
| Switch the LLM from Ollama to OpenAI | `.env`: `LLM_PROVIDER=openai`, `OPENAI_API_KEY=…`, optional `LLM_MODEL=gpt-4o` | No re-indexing. Per-chatbot override on the Chatbots page or `create_chatbot.py --llm-provider`. Add a new provider in `rag/llm.py` `get_chat_model` plus `LLM_DEFAULTS` in `config.py`. |
| Switch embeddings to OpenAI or Gemini | `.env`: `EMBEDDING_PROVIDER=openai` and the key, then `python scripts/init_db.py --reset-chunks` and re-ingest | Vector size changes (384 → 1536). Model list in `config.py` `EMBEDDING_DEFAULTS`; construction in `ingestion/embedder.py` `get_embeddings`. |
| Change how pages are crawled | `ingestion/crawler.py` | `_sitemap_urls` (sitemap), `_allowed` (robots), `_extract_links` (which links to follow), `crawl` (BFS loop). Limits via `.env`: `CRAWL_MAX_PAGES`, `CRAWL_MAX_DEPTH`, `CRAWL_DELAY_SECONDS`. Allowed/blocked URL rules in `url_utils.py` (`SKIP_EXTENSIONS`, `is_same_domain`). |
| Change what text is kept from a page | `ingestion/extractor.py` | `REMOVE_TAGS`, `REMOVE_SELECTORS` (what to delete), `MAIN_SELECTORS` (where content lives), `remove_boilerplate` threshold via `.env` `BOILERPLATE_LINE_RATIO`, `MIN_PAGE_WORDS`. |
| Change the website indexing logic (order of steps, error handling, re-index behaviour) | `ingestion/pipeline.py` `ingest_website` | Skip-unchanged logic is in `repository.upsert_page` (content hash). |
| Change vector search (how many chunks, threshold, the SQL) | `.env`: `TOP_K`, `SCORE_THRESHOLD`; SQL in `db/repository.py` `similarity_search`; Document conversion in `rag/retriever.py` | For hybrid keyword + vector search, change the SQL here and add a `tsvector` column in `schema.sql`. |
| Change the prompts or the refusal sentence | `rag/prompts.py` | `CONDENSE_QUESTION_PROMPT`, `ANSWER_SYSTEM_PROMPT`, `NOT_ENOUGH_INFO`. The chain in `rag/chain.py` detects the refusal by this exact sentence. |
| Change how many past turns the bot remembers | `.env`: `HISTORY_TURNS=6` | Applied by `trim_history` in `rag/chain.py`. |
| Change the database schema | `db/schema.sql`, then the matching functions in `db/repository.py` | Statements use `IF NOT EXISTS`, so new tables apply with `python scripts/init_db.py`. Changing an existing column needs an `ALTER TABLE` run manually or `--drop-all`. |
| Change the chatbot UI | `ui/streamlit_app.py` | `page_chat` for the chat screen, `_render_sources` for the source list, `page_websites` / `page_chatbots` for the admin pages. |
| Change the database password or host | `.env`: `DATABASE_URL` | Read by `config.py`, used by `db/connection.py`. |
| Change log verbosity | `.env`: `LOG_LEVEL=DEBUG` | `logging_config.py`. |

Two habits that keep this maintainable. First, if a value might ever differ between machines,
it belongs in `.env` and `config.py`, not in code. Second, when you change anything in
`ingestion/`, re-run `python scripts/ingest.py <url>`; unchanged pages are skipped by hash, so
it is cheap, and run `pytest -q` to catch breakage in the pure functions.
