-- Chat with Website: PostgreSQL + pgvector schema.
-- Applied by scripts/init_db.py, which replaces {EMBEDDING_DIM} with the configured vector size.
-- Every statement is idempotent so the script can be re-run safely.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per indexed website (a crawl root).
CREATE TABLE IF NOT EXISTS websites (
    id                 SERIAL PRIMARY KEY,
    name               TEXT        NOT NULL,
    base_url           TEXT        NOT NULL UNIQUE,
    status             TEXT        NOT NULL DEFAULT 'pending',   -- pending | crawling | ready | failed
    embedding_provider TEXT        NOT NULL,
    embedding_model    TEXT        NOT NULL,
    embedding_dim      INT         NOT NULL,
    max_pages          INT         NOT NULL DEFAULT 50,
    page_count         INT         NOT NULL DEFAULT 0,
    chunk_count        INT         NOT NULL DEFAULT 0,
    error              TEXT,
    last_crawled_at    TIMESTAMPTZ,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per crawled page. content_hash lets us skip unchanged pages on re-crawl.
CREATE TABLE IF NOT EXISTS pages (
    id           SERIAL PRIMARY KEY,
    website_id   INT         NOT NULL REFERENCES websites(id) ON DELETE CASCADE,
    url          TEXT        NOT NULL,
    title        TEXT,
    content_hash TEXT        NOT NULL,
    word_count   INT         NOT NULL DEFAULT 0,
    crawled_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (website_id, url)
);

-- One row per text chunk with its embedding. website_id is denormalised so the
-- vector search can filter on a plain indexed column.
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    page_id     INT         NOT NULL REFERENCES pages(id)    ON DELETE CASCADE,
    website_id  INT         NOT NULL REFERENCES websites(id) ON DELETE CASCADE,
    chunk_index INT         NOT NULL,
    content     TEXT        NOT NULL,
    char_count  INT         NOT NULL,
    embedding   vector({EMBEDDING_DIM}) NOT NULL,
    metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (page_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_website_idx   ON chunks (website_id);
CREATE INDEX IF NOT EXISTS chunks_page_idx      ON chunks (page_id);
-- HNSW = fast approximate nearest-neighbour index. Cosine distance because embeddings are normalised.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);

-- A chatbot is a named configuration that can see one or more websites.
CREATE TABLE IF NOT EXISTS chatbots (
    id            SERIAL PRIMARY KEY,
    name          TEXT        NOT NULL UNIQUE,
    description   TEXT,
    system_prompt TEXT,
    llm_provider  TEXT,                    -- NULL = use .env default
    llm_model     TEXT,                    -- NULL = use provider default
    top_k         INT         NOT NULL DEFAULT 5,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chatbot_websites (
    chatbot_id INT NOT NULL REFERENCES chatbots(id) ON DELETE CASCADE,
    website_id INT NOT NULL REFERENCES websites(id) ON DELETE CASCADE,
    PRIMARY KEY (chatbot_id, website_id)
);

-- Persisted conversations (used from the RAG/UI stages onward).
CREATE TABLE IF NOT EXISTS conversations (
    id         UUID PRIMARY KEY,
    chatbot_id INT         NOT NULL REFERENCES chatbots(id) ON DELETE CASCADE,
    title      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id UUID        NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT        NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT        NOT NULL,
    sources         JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_conversation_idx ON messages (conversation_id, id);
