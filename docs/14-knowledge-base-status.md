# Step 14 – Knowledge base status and data freshness

The **Websites** page now answers the questions an administrator asks about a RAG knowledge
base: how many pages are really indexed, how many chunks (embeddings) are stored, when the site
was last crawled, when it was last successfully indexed, whether the last run failed and on
which pages, and whether the stored data is getting old. Everything is read from PostgreSQL
with aggregate queries; nothing is re-crawled or re-embedded to produce a number.

## 1. Where the status lives

| Fact | Source | Notes |
|---|---|---|
| Ingestion status | `websites.status` | `pending` → **Pending**, `crawling` → **Processing**, `ready` → **Completed**, `failed` → **Failed** |
| Indexed pages / stored chunks | `count(*)` over `pages` / `chunks` per website | Computed live (`list_websites_overview`), not read from `websites.page_count` / `chunk_count`, which are only refreshed on a successful run |
| Last successful crawl | `websites.last_crawled_at` | Set right after the crawler finished with at least one page |
| Last successful indexing | `websites.last_indexed_at` (**new**) | Set only when the run ends `ready` |
| Latest run (counts, errors, failed pages) | `ingestion_runs` (**new table**) | One row per `ingest_website` call, newest shown; history under *Advanced diagnostics* |
| Per-page status | derived | has chunks → *Indexed*; no chunks → *No chunks*; URL in the latest run's `page_errors` → *Failed* |

Both additions come from `db/schema_upgrades.sql` and are applied automatically at start (and by
`init_db.py`), so Neon does not need a manual migration:

```sql
ALTER TABLE websites ADD COLUMN IF NOT EXISTS last_indexed_at TIMESTAMPTZ;
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id BIGSERIAL PRIMARY KEY, website_id INT REFERENCES websites(id) ON DELETE CASCADE,
    status TEXT CHECK (status IN ('processing','completed','failed')),
    started_at, crawl_finished_at, finished_at TIMESTAMPTZ,
    pages_crawled, pages_indexed, pages_skipped_unchanged, pages_skipped_short, chunks_created INT,
    error TEXT, page_errors JSONB, embedding_provider TEXT, embedding_model TEXT);
```

### Why two timestamps

A crawl can succeed while embedding fails (API key expired, model mismatch, database down).
Before this step `last_crawled_at` was written at the very end of a *successful* run, so it
really meant "last indexed". Now:

- `last_crawled_at` is written as soon as `crawl_and_extract` returns pages (`mark_website_crawled`);
- `last_indexed_at` is written only when the run ends `ready` (`mark_website_indexed`).

Websites indexed before this step have `last_indexed_at = NULL` and show **Not available** until
they are re-indexed. The value is never derived from the crawl time: the page shows what the
database knows, and says so when it knows nothing.

### Why no per-page status column

In `ingest_website` each page is its own transaction. When a page's content changed, its old
chunks are deleted *inside* that transaction; if embedding then fails, the rollback restores the
old chunks and the old hash. A failed *new* page leaves no row at all. So the honest per-page
status is: "does it have chunks?" plus "did the latest run report it?" – both already in the
database. The run row stores the failed URLs and messages as JSON (`page_errors`, capped at 200
entries of 500 characters), which also covers pages that have no row.

## 2. The pipeline's bookkeeping (`ingestion/pipeline.py`)

Only status writes were added; crawl, chunk, embed and retrieval code is untouched.

1. **Step 0** (same transaction as `upsert_website` + status `crawling`): `start_ingestion_run`.
   The row is committed before the crawl, so a crash still leaves a `processing` run.
2. **After the crawl**: `mark_website_crawled` (if ≥ 1 page) + `mark_ingestion_run_crawled`
   (pages crawled, too-short count, `crawl_finished_at`). Done *before* the "no indexable pages"
   check so a crawl that found nothing usable is still recorded as attempted.
3. **Per page**: a failure appends to `result.errors` (text, for the CLI) and to
   `result.page_errors` (`{"url", "error"}`, for the run row). The loop continues.
4. **End**: `refresh_website_counts`, status `ready`/`failed`, `mark_website_indexed` only on
   `ready`, `finish_ingestion_run` with status `completed` (even with some page errors) or
   `failed`.
5. **Abort** (exception anywhere): website `failed`, run `failed` with the message, then the
   original exception is re-raised. If recording the failure itself fails (database down) that
   is logged and the *original* error still surfaces.

`IngestResult` gained `run_id` and `page_errors`; `scripts/ingest.py` prints the run id and its
`--list` output shows live counts plus both timestamps.

### Stalled runs

If the process dies mid-run (Streamlit Cloud reboot, Ctrl-C), the website stays `crawling` and
the run stays `processing`. The page shows **Processing** with a warning once the run is older
than `KB_STALLED_RUN_HOURS` (default 3). Nothing is repaired automatically: re-indexing starts a
new run, and the previous one stays in the history as what it was.

## 3. The Websites page

Below the ingest form:

- **Knowledge base overview** – one table for all websites: status, pages *discovered* (fetched
  by the latest crawl), *indexed* (stored pages with chunks), *stored* (rows in `pages`), *failed*
  (latest run), chunks, last crawled, last indexed, freshness, which chatbots use it. A **Show
  websites used by** selectbox scopes the table to one chatbot (`chatbot_websites`); **Stale after
  (days)** sets the freshness threshold for this browser session (default `KB_STALE_DAYS`).
  **Refresh status** re-reads everything from the database; the caption shows when the data was
  loaded.
- **Website details** – pick one site:
  - status badge and warnings: stalled run, last error, *partially indexed* (which pages and why),
    data older than the threshold, embedding configuration different from the app's;
  - five compact cards: *Pages indexed*, *Chunks stored*, *Failed pages*, *Last successful crawl*,
    *Last successful indexing* (hover for what each means);
  - latest run summary and its **Failed pages** table (URL + error);
  - **Indexed pages** (collapsed): a search box (title or URL, case-insensitive, applied in SQL),
    then title, URL (opens the original page), status, chunks, words, last indexed, error – 100
    per view with a page selector (`LIMIT/OFFSET`, never the whole site);
  - **Advanced diagnostics** (collapsed, administrators only): the website's embedding
    provider/model/dims next to the app's, total vector records, last indexing execution,
    freshness, pages without embeddings, duplicate page URLs, chunks per page, and the last 10
    runs with their errors. Read-only: no button here re-embeds anything;
  - **Manage**: Re-index and Delete. A Re-index refreshes the overview above it when it finishes
    (the summary is shown once at the top).

The Chat page never shows any of this. With `ADMIN_PASSWORD` set, the Websites page (and so
every number above) sits behind the password prompt; without it the page is open, as before.

### Status rules

| Shown as | When |
|---|---|
| Pending / Processing / Failed | `websites.status` is `pending` / `crawling` / `failed` |
| **Completed** | `ready`, no failed page in the latest run, every stored page has chunks |
| **Partially indexed** | `ready`, but the latest run has failed pages *or* some stored pages have no chunks |

A website is therefore never reported as fully indexed while some of its pages failed.
`websites.status` itself stays `ready` so chatbots can still use the pages that did index.

### Queries (`db/repository.py`)

| Function | SQL (essentials) |
|---|---|
| `list_websites_overview(conn, chatbot_id=None)` | `SELECT w.*, (SELECT count(*) FROM pages p WHERE p.website_id = w.id) AS pages_total, (SELECT count(*) FROM pages p WHERE p.website_id = w.id AND EXISTS (SELECT 1 FROM chunks c WHERE c.page_id = p.id)) AS pages_indexed, (SELECT count(*) FROM chunks c WHERE c.website_id = w.id) AS chunks_total, (SELECT max(c.created_at) FROM chunks c WHERE c.website_id = w.id) AS newest_chunk_at, r.* FROM websites w LEFT JOIN LATERAL (SELECT * FROM ingestion_runs ir WHERE ir.website_id = w.id ORDER BY ir.started_at DESC, ir.id DESC LIMIT 1) r ON TRUE WHERE %(chatbot_id)s IS NULL OR w.id IN (SELECT website_id FROM chatbot_websites WHERE chatbot_id = %(chatbot_id)s) ORDER BY w.id` |
| `list_pages_with_chunk_counts(conn, website_id, search, limit, offset)` | `SELECT p.*, (SELECT count(*) FROM chunks c WHERE c.page_id = p.id) AS chunk_count, (SELECT max(c.created_at) FROM chunks c WHERE c.page_id = p.id) AS indexed_at, count(*) OVER () AS total FROM pages p WHERE p.website_id = %(wid)s AND (%(q)s = '' OR p.url ILIKE '%' || %(q)s || '%' OR p.title ILIKE '%' || %(q)s || '%') ORDER BY p.id LIMIT %(limit)s OFFSET %(offset)s` |
| `list_pages_without_chunks` | `SELECT … FROM pages p WHERE p.website_id = %s AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.page_id = p.id)` |
| `list_duplicate_page_urls` | `SELECT regexp_replace(lower(split_part(url,'#',1)), '^https?://(www\.)?|/+$', '', 'g') AS key, count(*) AS n, array_agg(url ORDER BY id) AS urls FROM pages WHERE website_id = %s GROUP BY key HAVING count(*) > 1` |
| `list_ingestion_runs` | `SELECT … FROM ingestion_runs WHERE website_id = %s ORDER BY started_at DESC, id DESC LIMIT %s` |
| `start_/mark_/finish_ingestion_run`, `mark_website_crawled/indexed` | The pipeline's writes |

Chunk text is never selected for any statistic. Pure presentation rules (labels, badges, *Not
available*, partial status, stalled, freshness, drift, search normalisation, pagination maths)
live in `ui/kb_status.py`, which imports neither Streamlit nor psycopg, so they are unit tested
directly.

### Settings

| Variable | Default | Meaning |
|---|---|---|
| `KB_STALE_DAYS` | 30 | Default for the *Stale after (days)* control on the page |
| `KB_STALLED_RUN_HOURS` | 3 | A run still `processing` after this long is flagged as stalled |

## 4. Performance and caching

- The dashboard reads go through `st.cache_data(ttl=60)` helpers keyed on a per-session
  `kb_version` counter. A plain rerun (typing in the search box, opening an expander) serves the
  overview, page list, duplicates, missing pages and run history from the cache and opens no
  database connection. **Refresh status**, a Re-index and a Delete bump the counter, so the next
  read is fresh; the 60 s TTL bounds what *other* sessions can see. Only metadata is cached.
- The overview is a single query; each count is an index-only aggregate per website. With a few
  hundred websites this is still a few milliseconds on Neon.
- The page list and the search are paginated in SQL. A site with 10 000 pages costs one 100-row
  query per view.
- The pipeline adds two small writes per run (run start, crawl mark) and a few columns on the
  existing final transaction. The per-page hot loop is unchanged.
- `init_db.py --reset-chunks` now also clears `last_indexed_at`; `--drop-all` drops
  `ingestion_runs`.

## 5. Limitations (data that is not tracked)

- Websites indexed before this step show **Not available** for last indexed, discovered and
  failed pages until they are re-indexed; nothing is back-filled or guessed.
- A page that failed on its *first* visit has no `pages` row (its transaction rolled back); it
  appears only in the latest run's Failed pages table, not in the page list.
- "Discovered" is the number of pages the crawler fetched in the latest run, not a sitemap total.
- The freshness threshold on the page is per browser session; the durable default is
  `KB_STALE_DAYS`.
- Duplicate detection compares URL spellings; it cannot tell that two different URLs serve the
  same content.

## Test this step

Automated: `pytest -q tests/test_kb_status.py tests/test_pipeline_runs.py tests/test_ui.py`
(no database) and, with the local PostgreSQL running, `pytest -q tests/test_kb_status_db.py`,
which builds a ready site, an empty site, pages without chunks, a failed run, a partial run,
several runs, a chatbot-scoped view, search and duplicate URLs, and checks every count against `count(*)`.

Manual (Websites page):
1. Index a small site: status **Completed**, pages and chunks > 0, both timestamps filled, a run
   with crawled/indexed counts. Compare with `SELECT count(*) FROM pages/chunks WHERE website_id = …`.
2. Add a website whose crawl finds nothing (an unreachable URL): **Failed**, the error is shown,
   no "last indexed", the run is `failed`.
3. Run `python scripts/init_db.py --reset-chunks`, reload: pages show *No chunks*, chunks 0, last
   indexed **Not available**, diagnostics list every page under *Pages without chunks*.
4. Re-index with an invalid `OPENAI_API_KEY` (OpenAI embeddings): crawl succeeds, every page fails,
   status **Failed**, *Failed pages* lists them, *Last successful crawl* is new while *Last
   successful indexing* is unchanged.
5. Fix the key and re-index: **Completed**, a second run in the history, both timestamps updated.
6. Switch websites in the selectbox and filter by chatbot: the table and details follow.
   Type part of a title in *Search pages*: the list filters and the pager resets. Lower *Stale
   after (days)* to 1: sites indexed earlier than yesterday are flagged.
7. Run `python scripts/ingest.py <url>` in a terminal, click **Refresh status**: the numbers change.
8. Index a site with 300+ pages: the page list shows 100 at a time with a page selector.
9. Set `ADMIN_PASSWORD`: the Websites page asks for it; the Chat page shows no diagnostics.
10. Websites indexed before this step show **Not available** for last indexed and "no run has
    been recorded" until re-indexed; nothing is invented.
