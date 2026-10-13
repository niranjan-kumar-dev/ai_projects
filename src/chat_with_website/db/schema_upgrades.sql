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
