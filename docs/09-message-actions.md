# Step 9 – Message actions: copy, regenerate, feedback

Every answer the chatbot gives now has a small row of actions underneath it:

| Action | What it does | Where the work happens |
|---|---|---|
| **Copy** | Puts that answer's markdown on the clipboard and shows "Copied" for 1.5 s | Browser only (JavaScript); nothing is sent to Python |
| **Regenerate** (last answer only) | Asks the model again with the same question and the same earlier turns, then replaces the answer in place | `_regenerate` → `_generate_answer` → `repo.update_message` |
| **👍 / 👎** | Saves one rating per answer; clicking the other thumb changes it | `_save_feedback` → `repo.upsert_feedback` |

Nothing about retrieval, embeddings or the LangChain pipeline changed: regeneration runs the
exact same `chain.prepare → stream_answer → finalize` phases as a normal question.

## 1. Database: two columns and one table

`src/chat_with_website/db/schema.sql` grew three idempotent statements, so the "migration" is
simply `python scripts/init_db.py` again (locally and on Neon).

```sql
ALTER TABLE messages ADD COLUMN IF NOT EXISTS regenerated_at     TIMESTAMPTZ;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS regeneration_count INT NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS message_feedback (
    message_id      BIGINT   PRIMARY KEY REFERENCES messages(id)      ON DELETE CASCADE,
    conversation_id UUID     NOT NULL    REFERENCES conversations(id) ON DELETE CASCADE,
    chatbot_id      INT      NOT NULL    REFERENCES chatbots(id)      ON DELETE CASCADE,
    rating          SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Why a separate table and not a `rating` column on `messages`?

- **No duplicates by construction.** `message_id` is the primary key, so a message can have at
  most one feedback row. Saving feedback is an *upsert*: `INSERT … ON CONFLICT (message_id) DO
  UPDATE SET rating = …, updated_at = now()`. Changing 👍 to 👎 updates the same row.
- **The ids are never trusted from the UI.** `upsert_feedback` is an `INSERT … SELECT` that reads
  `conversation_id` and `chatbot_id` from the message itself, so feedback can only ever be
  attached to the chatbot that actually produced the answer.
- **It is easy to extend** (a free-text comment, a user id) and easy to report on
  (`SELECT chatbot_id, rating, count(*) FROM message_feedback GROUP BY 1, 2`).

A regeneration deletes the old feedback row in the same transaction as the `UPDATE messages`
(`repo.update_message`): the thumbs were about text that no longer exists.

## 2. The UI needs to know each message's id

Before this step the chat page only kept two parallel lists in `st.session_state`: `history`
(LangChain messages) and `sources`. Feedback and regeneration must address a specific
database row, so:

- `repo.add_message` now returns the new id (`RETURNING id`) and `persist_turn` passes the
  assistant message's id back to the page.
- `repo.get_messages` also returns `id`, `regeneration_count` and the saved `rating`
  (`LEFT JOIN message_feedback`).
- A third parallel list, `message_meta`, holds `{"id", "rating", "gen"}` for every assistant
  turn (`None` for user turns). `gen` counts regenerations. All five places that reset or extend
  the lists go through two helpers, `_reset_chat()` and `_append_turn()`, so they can't drift apart.

If saving the answer failed (database down), the message has no id: the thumbs are shown
disabled, Copy and Regenerate still work.

## 3. Regenerate = the same three phases, drawn into the old bubble

The streaming code that used to live inline in `page_chat` moved into
`_generate_answer(chatbot_id, question, context_history, …)`. It draws the loading cue, runs
phase 1 (`chain.prepare`: question rewrite + retrieval) and phase 2 (`st.write_stream` over
`chain.stream_answer`, then `chain.finalize`) and returns `(RagAnswer, sources)`, or `None` after
showing the error and a **Retry** button. A new question and a regeneration both call it; the
only difference is *what* they pass:

| | question | context_history | afterwards |
|---|---|---|---|
| New question | the text just typed | the whole `history` | `persist_turn` → two new rows |
| Regenerate answer `i` | `history[i-1]` (its own question) | `history[:i-1]` (the turns *before* that question) | `update_message(id)` → same row, new text |

Details worth knowing:

- **Only the last answer** has a Regenerate button. Regenerating an older answer would leave
  the later turns conditioned on text that no longer exists (ChatGPT behaves the same way).
- The button's callback stores the message index in `st.session_state["regenerate_index"]`;
  the next run pops it, disables the chat input (`st.chat_input(..., disabled=True)`) and, when the
  loop reaches that bubble, streams the new answer into it instead of drawing the old text.
- **The original is kept until a complete new answer exists.** `history[i]` and the database
  row are only touched after `finalize` succeeded. If the model call fails or times out
  (`_friendly_error` maps OpenAI/httpx errors and `TimeoutError`), the error and **Retry** appear
  in the bubble and the previous answer is shown again under them.
- On success the page calls `st.rerun()` so the bubble is redrawn with a fresh action row (the
  feedback key contains `gen`, so the thumbs start blank) and the chat input is enabled again.
- No duplicate user message: the question is read from history, never re-inserted.

## 4. Feedback with `st.feedback`

`st.feedback("thumbs", key=..., default=..., on_change=_save_feedback, args=(message_id, key))`
is Streamlit's built-in thumbs widget: it highlights the selected thumb, matches the theme and
needs no custom HTML. Its value is `1` for 👍 and `0` for 👎, which the app maps to the stored
rating `1` / `-1`.

Two Streamlit rules shaped the code:

- **Widget keys must be unique on the page.** Every key carries the conversation id and the
  message index (`fb_<conversation>_<i>_<gen>`, `regen_<conversation>_<i>`, `copy_<conversation>_<i>`),
  so switching conversations never reuses a stale widget.
- **Callbacks must not raise.** An exception inside `on_change` aborts the whole run, so
  `_save_feedback` catches database errors and shows a toast instead.

The saved rating is read back by `_load_conversation` and passed as `default`, which is why the
highlighted thumb survives a page refresh.

## 5. Copy without an iframe: `st.components.v2`

Streamlit has no clipboard API, and browsers only allow writing to the clipboard from a real
user click. The classic `st.components.v1.html` runs inside an iframe that knows nothing about
the app theme. Streamlit 1.64's `st.components.v2.component(name, html=, css=, js=)` mounts
HTML directly in the page (inside a shadow root for style isolation), so:

- the click handler calls `navigator.clipboard.writeText(text)` (with an `execCommand("copy")`
  fallback for plain-http hosts) and swaps the icon to a check mark plus "Copied" for 1.5 s;
- the button is styled with the theme variables `--st-text-color`,
  `--st-secondary-background-color` and `--st-primary-color`, so it matches the tertiary
  Streamlit buttons beside it in light and dark mode;
- nothing is sent back to Python, so copying never reruns the script or calls the model.

The answer text is passed as `data={"text": ...}`, never pasted into the HTML string: component
HTML is not sanitised, and answers come from a model.

## 6. A cache bug fixed along the way

`st.cache_data` and `st.cache_resource` **ignore parameters whose name starts with an
underscore**. The sidebar list was cached as `_conversations(chatbot_id, _version)`, so bumping
the version never invalidated it and a new conversation could take up to five minutes (the TTL)
to appear. The parameters are now `version` / `code_version`.

## Test this step

Automated (no database, no model):

```powershell
pytest -q tests/test_ui.py tests/test_chain.py
```

Manual, after `python scripts/init_db.py` and starting the UI:

1. Ask a short question → **Copy** → paste into the chat input: the text matches.
2. Ask something that produces a list with links → Copy → the markdown (list markers, links) is preserved.
3. Click **Regenerate** on the last answer: the bubble shows the loading cue, streams a new answer, the thumbs reset.
4. Regenerate the same answer again; `SELECT regeneration_count FROM messages` grows, no new rows appear.
5. Open an older conversation from the sidebar and regenerate its last answer.
6. Click 👍 → toast "Thanks for your feedback"; `SELECT * FROM message_feedback` shows one row with `rating = 1`.
7. Click 👎 → the same row now has `rating = -1` (still one row).
8. Refresh the browser, reopen the conversation: 👎 is still highlighted.
9. Switch between conversations: each shows its own actions and ratings.
10. Put a wrong `OPENAI_API_KEY` in `.env`, restart, regenerate: the error and **Retry** appear and the previous answer is still there; fix the key, Retry succeeds.
