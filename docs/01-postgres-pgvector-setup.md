# Step 1 – PostgreSQL + pgvector

## What is a vector database?

An **embedding** is a list of a few hundred numbers describing the meaning of a text.
To answer a question we need to find the stored texts whose vectors are *closest*
to the question's vector. A vector database is anything that can do that search fast.

**pgvector** is a PostgreSQL extension that adds:

* a `vector(N)` column type (N = number of dimensions, 384 for all-MiniLM-L6-v2),
* distance operators – `<=>` cosine distance, `<->` Euclidean, `<#>` inner product,
* index types (**HNSW**, IVFFlat) so nearest-neighbour search stays fast as data grows.

Using PostgreSQL instead of a separate vector store means one database, one backup,
real foreign keys between websites, pages, chunks and chatbots, and plain SQL for
filtering ("only chunks of websites 3 and 7").

### Cosine distance in one sentence
Two vectors pointing in the same direction have cosine distance 0; unrelated ones
are near 1. We normalise embeddings to length 1, so cosine is the right measure.
`score = 1 - distance` is the familiar "similarity" between 0 and 1.

### HNSW vs IVFFlat
* **HNSW** (what we use): a graph index; excellent recall, works on an empty table,
  slower inserts. Right choice for up to a few million rows.
* **IVFFlat**: clusters vectors; must be built *after* data exists and retrained
  when data changes a lot. Skip it for now.

## Schema

`src/chat_with_website/db/schema.sql` (simplified):

```
websites ──< pages ──< chunks (embedding vector(N), metadata jsonb)
    ▲
    │ chatbot_websites (many-to-many)
chatbots ──< conversations ──< messages
```

* `websites` – one row per crawl root. Records which embedding model was used.
* `pages` – one row per URL with a `content_hash`, so re-crawling only re-embeds changed pages.
* `chunks` – the searchable unit. `website_id` is copied here (denormalised) so the
  search filter is a plain indexed column instead of a join.
* `chatbots` + `chatbot_websites` – a chatbot sees only the websites attached to it.
* `conversations` / `messages` – chat history (used from stage 5 on).

The placeholder `{EMBEDDING_DIM}` is replaced by `init_db.py` with the dimension from `.env`.

## Setup on this machine

PostgreSQL 17 is already installed as the Windows service `postgresql-x64-17`
(port 5432) and pgvector 0.8.0 ships with it – you only need the password you set
during installation.

1. Edit `.env`:
   ```
   DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/chat_with_website
   ```
   Forgot the password? In pgAdmin you can reset it from a query window with
   `ALTER USER postgres PASSWORD 'newpass';` (pgAdmin itself stores the saved password).
2. Run
   ```powershell
   python scripts/init_db.py
   ```
   It creates the database if missing, enables `vector`, and creates all tables and indexes.
   It is idempotent – run it as often as you like.

Other environments: Docker `docker run -e POSTGRES_PASSWORD=pw -p 5432:5432 pgvector/pgvector:pg17`,
or any hosted Postgres with pgvector (Neon, Supabase, RDS).

### Local vs production database (`APP_ENV`)

You never edit a URL to switch environments. `config.py` picks the database from `APP_ENV`:

```
APP_ENV=development  →  LOCAL_DATABASE_URL        (default; your machine)
APP_ENV=production   →  PRODUCTION_DATABASE_URL   (Streamlit Community Cloud / server)
DATABASE_URL         →  optional override of both (old single-URL behaviour still works)
```

The rest of the code only reads `settings.database_url`; the choice is made once, in
`Settings._resolve_database_url`. Missing URL or an unknown `APP_ENV` value stops the program with
one line, e.g. `Configuration error: APP_ENV=production but PRODUCTION_DATABASE_URL is not set…`.

Every connection logs the environment without secrets:
`Opening connection pool [development → localhost:5432/chat_with_website] to postgresql://postgres:***@…`
and the Streamlit sidebar footer shows the same `Database: development → …` label.

Each database is independent: run `python scripts/init_db.py` and index websites once per
environment. To do that for production from your machine, prefix one command:
`$env:APP_ENV="production"; python scripts/init_db.py` (PowerShell), then remove the variable.

## Code walkthrough

* `db/connection.py` – a `psycopg_pool.ConnectionPool`. `register_vector(conn)` teaches psycopg
  to send numpy arrays as `vector` values. `get_conn()` is a context manager that commits on
  success and rolls back on error.
* `db/repository.py` – *all* SQL. Only parameterised queries (`%s` placeholders), never string
  formatting with user input, which rules out SQL injection.
* `scripts/init_db.py` – connects to the maintenance DB `postgres` to `CREATE DATABASE`, then
  applies `schema.sql`. Flags: `--reset-chunks` (after changing the embedding model),
  `--drop-all` (start over).

## Test this step

```powershell
python scripts/init_db.py
```
Expected log lines: `Schema applied (embedding dimension = 384)`, `pgvector version: 0.8.0`,
`Tables: chatbot_websites, chatbots, chunks, conversations, messages, pages, websites`.

In pgAdmin (or `psql -U postgres -d chat_with_website`):
```sql
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';   -- vector | 0.8.0
SELECT '[1,2,3]'::vector <=> '[1,2,4]'::vector;                          -- a small cosine distance
```

Next: [Step 2 – Crawling and extraction](02-crawling-and-extraction.md)
