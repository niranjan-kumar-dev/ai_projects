# Step 4 – Embeddings and storage (the ingestion pipeline is complete)

## Embeddings in plain words

An embedding model reads text and outputs a fixed-length list of numbers – 384 for
all-MiniLM-L6-v2. It is trained so that texts with similar meaning get similar numbers:
"How do I reach you?" and "Contact us by phone or email" end up close together even though
they share no words. Searching by embedding is therefore **semantic search**.

Important properties:

* **Same model for indexing and querying.** Vectors from different models live in different
  spaces and cannot be compared. That is why `websites` records the model and why switching
  provider requires re-ingesting.
* **Fixed size per model.** 384 (MiniLM), 1536 (OpenAI text-embedding-3-small),
  3072 (Gemini gemini-embedding-001). The `chunks.embedding` column is `vector(N)`.
* **Normalised** vectors (length 1) make cosine distance the natural measure.

## Providers (`ingestion/embedder.py`)

| `EMBEDDING_PROVIDER` | Class | Needs | Dims | Cost |
|---|---|---|---|---|
| `huggingface` | `HuggingFaceEmbeddings` | ~90 MB download, CPU | 384 | free |
| `openai` | `OpenAIEmbeddings` | `OPENAI_API_KEY` | 1536 | about $0.02 per 1M tokens |
| `gemini` | `GoogleGenerativeAIEmbeddings` | `GOOGLE_API_KEY` | 3072 | free tier available |

`get_embeddings()` builds the right LangChain object and is cached, so the model loads once.
`embed_texts()` batches (64 at a time), retries hosted APIs with back-off and **validates the
vector size** against `EMBEDDING_DIM` before anything is written.

### Switching provider later
```powershell
# .env: EMBEDDING_PROVIDER=openai  (+ OPENAI_API_KEY)
python scripts/init_db.py --reset-chunks     # recreates chunks as vector(1536)
python scripts/ingest.py https://your-site    # re-embed
```
`init_db.py` refuses to run if the table's dimension and `.env` disagree, and `ingest.py`
checks again, so you cannot mix vector sizes by accident.

## Storage (`db/repository.py`)

* `upsert_website` – create/update the website row, recording the embedding model.
* `upsert_page` – returns `(row, changed)`. If a page's `content_hash` is unchanged nothing else
  happens; if it changed, its old chunks are deleted first.
* `insert_chunks` – `executemany` with `ON CONFLICT (page_id, chunk_index) DO UPDATE`.
  Embeddings are passed as numpy `float32` arrays, which the pgvector adapter sends natively.
* `delete_stale_pages` – pages that vanished from the site are removed after a re-crawl.
* `refresh_website_counts` + `ANALYZE chunks` – keeps counts and planner statistics fresh.

## Pipeline (`ingestion/pipeline.py`)

```
ingest_website(url)
  validate URL → upsert website (status='crawling')
  crawl_and_extract → pages
  for each page (own transaction):
      upsert_page → skip if unchanged
      chunk_page → embed_texts → insert_chunks
  prune stale pages, refresh counts, status='ready' (or 'failed' with the error text)
```
A failure on one page is recorded in `IngestResult.errors` and the crawl continues.
`progress` callbacks feed the CLI log now and the Streamlit progress bar later.

## Test this step

```powershell
python scripts/ingest.py https://www.python.org/about/ --max-pages 10
python scripts/ingest.py --list
```
The first run downloads the MiniLM model (~90 MB). Expect a summary like
`pages indexed: 8, chunks created: 60`.

In pgAdmin / psql:
```sql
SELECT id, name, status, page_count, chunk_count, embedding_model FROM websites;
SELECT url, title, word_count FROM pages ORDER BY id;
SELECT page_id, chunk_index, left(content, 80) FROM chunks ORDER BY page_id, chunk_index LIMIT 10;

-- a first semantic search without any Python: the chunks closest to chunk #1
SELECT p.url, left(c.content, 80),
       1 - (c.embedding <=> (SELECT embedding FROM chunks WHERE id = 1)) AS score
FROM chunks c JOIN pages p ON p.id = c.page_id
ORDER BY c.embedding <=> (SELECT embedding FROM chunks WHERE id = 1)
LIMIT 5;
```

Re-run the same `ingest.py` command: the summary should show `unchanged (skipped): N` and the
chunk count must **not** grow. That is the content-hash check working.

Next: Step 5 – Chatbots and vector search (`docs/05-...`, built in the next stage).
