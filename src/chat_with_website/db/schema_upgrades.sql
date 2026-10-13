-- Incremental, idempotent schema upgrades (stage 09 onwards).
-- Every statement MUST be safe to replay (IF NOT EXISTS / ADD COLUMN IF NOT EXISTS): the app runs
-- this file on every cold start (`ensure_schema_upgrades`) so a deploy that adds a column never
-- breaks production because the database was not migrated by hand. No placeholders here.

-- Message actions (stage 09): regeneration bookkeeping on messages + one rating per assistant message.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS regenerated_at     TIMESTAMPTZ;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS regeneration_count INT NOT NULL DEFAULT 0;

-- The primary key on message_id guarantees at most one feedback row per message (no duplicates);
-- chatbot_id / conversation_id are denormalised so feedback can be reported per chatbot without joins.
CREATE TABLE IF NOT EXISTS message_feedback (
    message_id      BIGINT      PRIMARY KEY REFERENCES messages(id)      ON DELETE CASCADE,
    conversation_id UUID        NOT NULL    REFERENCES conversations(id) ON DELETE CASCADE,
    chatbot_id      INT         NOT NULL    REFERENCES chatbots(id)      ON DELETE CASCADE,
    rating          SMALLINT    NOT NULL CHECK (rating IN (-1, 1)),       -- 1 = thumbs up, -1 = thumbs down
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS message_feedback_chatbot_idx ON message_feedback (chatbot_id, rating);

-- Edited questions (stage 10): an edit forks the conversation. The branch records where it came
-- from, and the edited question records which message it replaced. SET NULL: deleting a parent
-- never touches its branches.
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS parent_conversation_id UUID   REFERENCES conversations(id) ON DELETE SET NULL;
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS forked_from_message_id BIGINT REFERENCES messages(id)      ON DELETE SET NULL;
ALTER TABLE messages      ADD COLUMN IF NOT EXISTS edited_from_message_id BIGINT REFERENCES messages(id)      ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS conversations_parent_idx ON conversations (parent_conversation_id);

-- Follow-up suggestions (stage 12): the questions suggested under an assistant answer.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS suggestions JSONB;

-- Per-chatbot branding (stage 13). All nullable: existing chatbots keep the defaults.
ALTER TABLE chatbots ADD COLUMN IF NOT EXISTS display_name      TEXT;
ALTER TABLE chatbots ADD COLUMN IF NOT EXISTS welcome_heading   TEXT;
ALTER TABLE chatbots ADD COLUMN IF NOT EXISTS accent_color      TEXT;
ALTER TABLE chatbots ADD COLUMN IF NOT EXISTS starter_questions JSONB;
ALTER TABLE chatbots ADD COLUMN IF NOT EXISTS input_placeholder TEXT;
ALTER TABLE chatbots ADD COLUMN IF NOT EXISTS logo              BYTEA;   -- normalised 256px PNG
ALTER TABLE chatbots ADD COLUMN IF NOT EXISTS logo_updated_at   TIMESTAMPTZ;

-- Knowledge-base status (stage 14): one row per ingestion run, plus when the website was last
-- fully indexed. `last_crawled_at` (schema.sql) now means "the crawl succeeded"; `last_indexed_at`
-- means "pages were embedded and the website became ready". Existing rows keep NULL (never guessed).
ALTER TABLE websites ADD COLUMN IF NOT EXISTS last_indexed_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id                      BIGSERIAL   PRIMARY KEY,
    website_id              INT         NOT NULL REFERENCES websites(id) ON DELETE CASCADE,
    status                  TEXT        NOT NULL DEFAULT 'processing' CHECK (status IN ('processing', 'completed', 'failed')),
    started_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    crawl_finished_at       TIMESTAMPTZ,
    finished_at             TIMESTAMPTZ,
    pages_crawled           INT         NOT NULL DEFAULT 0,
    pages_indexed           INT         NOT NULL DEFAULT 0,
    pages_skipped_unchanged INT         NOT NULL DEFAULT 0,
    pages_skipped_short     INT         NOT NULL DEFAULT 0,
    chunks_created          INT         NOT NULL DEFAULT 0,
    error                   TEXT,
    page_errors             JSONB       NOT NULL DEFAULT '[]'::jsonb,   -- list of url + error objects
    embedding_provider      TEXT,
    embedding_model         TEXT
);
CREATE INDEX IF NOT EXISTS ingestion_runs_website_idx ON ingestion_runs (website_id, started_at DESC);

-- RAG evaluation (stage 16): a test-case library per chatbot, one row per batch run and one row per
-- evaluated case. Results snapshot the case (question, expected answer, labels) so editing or
-- deleting a case never rewrites history; `test_case_id` is only a pointer (SET NULL on delete).
-- JSON defaults avoid curly braces on purpose: this file is applied verbatim, without substitution.
CREATE TABLE IF NOT EXISTS eval_test_cases (
    id              BIGSERIAL   PRIMARY KEY,
    chatbot_id      INT         NOT NULL REFERENCES chatbots(id) ON DELETE CASCADE,
    question        TEXT        NOT NULL,
    expected_answer TEXT        NOT NULL DEFAULT '',
    answerable      BOOLEAN     NOT NULL DEFAULT TRUE,
    category        TEXT        NOT NULL DEFAULT 'factual'
                    CHECK (category IN ('factual', 'service', 'pricing', 'company', 'multi_page', 'follow_up', 'unanswerable')),
    source_urls     JSONB       NOT NULL DEFAULT '[]'::jsonb,   -- labelled pages that should be retrieved
    key_facts       JSONB       NOT NULL DEFAULT '[]'::jsonb,   -- exact strings the answer must contain
    previous_turns  JSONB       NOT NULL DEFAULT '[]'::jsonb,   -- [role/content] history for follow-up cases
    notes           TEXT,
    enabled         BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (category <> 'unanswerable' OR answerable = FALSE)
);
CREATE INDEX IF NOT EXISTS eval_test_cases_chatbot_idx ON eval_test_cases (chatbot_id, id);
CREATE UNIQUE INDEX IF NOT EXISTS eval_test_cases_question_idx ON eval_test_cases (chatbot_id, lower(question));

CREATE TABLE IF NOT EXISTS eval_runs (
    id                BIGSERIAL   PRIMARY KEY,
    chatbot_id        INT         NOT NULL REFERENCES chatbots(id) ON DELETE CASCADE,
    status            TEXT        NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
    started_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at       TIMESTAMPTZ,
    total_cases       INT         NOT NULL DEFAULT 0,
    done              INT         NOT NULL DEFAULT 0,
    passed            INT         NOT NULL DEFAULT 0,
    failed            INT         NOT NULL DEFAULT 0,
    needs_review      INT         NOT NULL DEFAULT 0,
    errored           INT         NOT NULL DEFAULT 0,
    last_question     TEXT,                                      -- progress line for the dashboard
    config            JSONB       NOT NULL DEFAULT jsonb_build_object(),   -- model / retrieval settings used
    judge_model       TEXT,
    rubric_version    TEXT,
    judge_prompt_hash TEXT,
    error             TEXT
);
CREATE INDEX IF NOT EXISTS eval_runs_chatbot_idx ON eval_runs (chatbot_id, started_at DESC);
-- At most one running evaluation per chatbot, enforced by the database (not by the UI session).
CREATE UNIQUE INDEX IF NOT EXISTS eval_runs_one_running_idx ON eval_runs (chatbot_id) WHERE status = 'running';

CREATE TABLE IF NOT EXISTS eval_results (
    id                  BIGSERIAL   PRIMARY KEY,
    run_id              BIGINT      NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    test_case_id        BIGINT      REFERENCES eval_test_cases(id) ON DELETE SET NULL,
    chatbot_id          INT         NOT NULL REFERENCES chatbots(id) ON DELETE CASCADE,
    question            TEXT        NOT NULL,                     -- snapshot of the case at run time
    expected_answer     TEXT        NOT NULL DEFAULT '',
    answerable          BOOLEAN     NOT NULL DEFAULT TRUE,
    category            TEXT        NOT NULL,
    source_urls         JSONB       NOT NULL DEFAULT '[]'::jsonb,
    key_facts           JSONB       NOT NULL DEFAULT '[]'::jsonb,
    case_updated_at     TIMESTAMPTZ,
    standalone_question TEXT,
    raw_answer          TEXT,                                     -- model output before finalize (citation check)
    generated_answer    TEXT,
    grounded            BOOLEAN,
    model_called        BOOLEAN     NOT NULL DEFAULT FALSE,
    retrieved           JSONB       NOT NULL DEFAULT '[]'::jsonb,   -- passages the model saw, with scores
    status              TEXT        NOT NULL CHECK (status IN ('passed', 'failed', 'needs_review', 'error')),
    failure_code        TEXT,
    failure_reason      TEXT,
    correctness         JSONB,
    faithfulness        JSONB,
    retrieval           JSONB,
    citations           JSONB,
    prepare_ms          REAL,
    generation_ms       REAL,
    judge_ms            REAL,
    usage               JSONB,                                    -- provider usage_metadata, when returned
    judge_model         TEXT,
    rubric_version      TEXT,
    error               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS eval_results_run_idx ON eval_results (run_id, id);
CREATE INDEX IF NOT EXISTS eval_results_case_idx ON eval_results (test_case_id);
CREATE INDEX IF NOT EXISTS eval_results_chatbot_idx ON eval_results (chatbot_id, created_at DESC);
