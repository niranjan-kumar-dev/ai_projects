# Step 10 – Edit a question: branch the conversation and answer again

Every question you asked now has a small pencil underneath. Click it, change the text (several
lines are fine), and press **Save & Submit**: the chatbot retrieves fresh website chunks for the
new wording, streams a new answer with new citations, and the outdated answer disappears from
the view. **Cancel** puts everything back.

## 1. Why an edit creates a branch

ChatGPT keeps *both* versions of an edited question. The simplest way to do that with our schema
(`conversations` ← `messages`) is to treat the edited version as a **new conversation that starts
with a copy of everything before the edited question**:

```
original:  Q1 → A1 → Q2 → A2 → Q3 → A3        (untouched, still in the sidebar)
edit Q2:   Q1 → A1 → Q2' → A2'                 (new conversation, branch icon in the sidebar)
```

- The original is never modified or deleted, so the old answer, its sources, its 👍/👎 and every
  later turn are still there if you open it again.
- The branch only contains the turns *before* the edited question, so the edited question can
  never be paired with the old answer, and the later turns (which answered the old wording) are
  not carried over.
- Refreshing the page or switching conversations needs no extra state: a branch is just another
  conversation, and `_load_conversation` reads it like any other.

## 2. Database: three nullable columns

```sql
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS parent_conversation_id UUID   REFERENCES conversations(id) ON DELETE SET NULL;
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS forked_from_message_id BIGINT REFERENCES messages(id)      ON DELETE SET NULL;
ALTER TABLE messages      ADD COLUMN IF NOT EXISTS edited_from_message_id BIGINT REFERENCES messages(id)      ON DELETE SET NULL;
```

| Column | Meaning |
|---|---|
| `conversations.parent_conversation_id` | the conversation this one was branched from (the sidebar shows the branch icon when set) |
| `conversations.forked_from_message_id` | the question in the parent that was edited |
| `messages.edited_from_message_id` | on the branch's new question: the message it replaced |

`ON DELETE SET NULL` means deleting a parent conversation simply detaches its branches; it
never deletes them and never fails because of them. Message ids are stable: the parent's rows are
not touched, the branch gets its own copies. Run `python scripts/init_db.py` once to add the
columns (locally and on Neon).

## 3. The flow, step by step

1. **Pencil** (`_render_user_message`): a tertiary icon button under the user bubble, faint until
   the message is hovered or focused (scoped CSS on the `useract_` container keys; nothing is
   hidden, so it is keyboard-accessible). Its callback stores the message index in
   `st.session_state["editing_index"]`.
2. **Editor**: that bubble is redrawn as `st.text_area` (the original text, multiline) with
   **Cancel** and **Save & Submit**. An empty message is refused with an inline error.
3. **Save & Submit** (`_submit_edit`, a button callback, so it runs before the script):
   - validates the text;
   - calls `repo.fork_conversation` inside one `get_conn()` transaction: it checks the parent
     belongs to the same chatbot, inserts the new conversation row (same title, parent and
     edited-message links) and copies the turns before the edited question **from what is on
     screen** (so the copy never depends on parent rows that might be missing after an earlier
     failed save). Feedback is not copied; it belongs to the original answers;
   - switches session state to the branch (the copied turns, with their new ids) and queues the
     edited text exactly like a Retry does (`pending_question`), remembering which message it
     replaces (`pending_edited_from`).
4. The next run is the **ordinary new-question path**: `_generate_answer` runs
   `chain.prepare` (question rewrite + fresh vector search over the edited text and only the
   preceding context), streams the answer, and `persist_turn` saves the question (with
   `edited_from_message_id`) and the answer once, after a complete answer.
5. If the model call fails, the existing error + **Retry** appear; nothing is saved; the branch
   still shows the copied turns and the edited question; Retry asks again. Typing a different
   question instead cancels the edit link, so it can never be attached to the wrong question.
   A failed question is remembered in session state (`failed_turn`) and redrawn with its error,
   Retry and pencil on every rerun, so you can also correct a failed question in place.
6. After a complete answer the page reruns once (`st.rerun()`): the new turn is drawn from
   history like every other, so it gets its pencil and answer actions immediately, the welcome
   chips disappear and the sidebar shows the conversation right away.

Why fork *before* generating rather than after? A branch that exists only in memory would have to
be carried through failures and retries, and its copied turns would have no ids (no feedback, no
regeneration) until it was saved. Writing it first keeps all of that simple. The price: if
generation fails and you walk away, a branch holding only the copied turns stays in the sidebar
(same title, branch icon). It is harmless and can be deleted like any conversation.

## 4. Session state

`message_meta` now also records the id of every **user** message (`{"id": …}`), because an edit
has to say which message it replaced. Assistant entries are unchanged (`id`, `rating`, `gen`).
All per-conversation transient keys (`editing_index`, `pending_question`, `pending_edited_from`,
`regenerate_index`) are cleared by `_set_chat`, which every path that changes the open
conversation goes through (new chat, chatbot switch, delete, load, fork).

## Test this step

Automated (no database, no model): `pytest -q tests/test_ui.py tests/test_chain.py`.

Manual, after `python scripts/init_db.py` and starting the UI:

1. Open a conversation, click the pencil under its last question, change the text, Save & Submit:
   the old answer disappears, a new one streams with new citations, the sidebar shows a second
   entry with the branch icon; the original entry still has the old answer.
2. Click the pencil and Cancel: nothing changes.
3. Save an empty message: "Message cannot be empty." and the editor stays open.
4. Paste a two-line question: both lines are kept.
5. In a conversation with several turns, edit an earlier question: only the turns before it are
   kept, the new answer follows, the later turns stay in the original.
6. `SELECT id, parent_conversation_id, forked_from_message_id FROM conversations ORDER BY created_at DESC LIMIT 3;`
   and `SELECT id, conversation_id, role, edited_from_message_id FROM messages ORDER BY id DESC LIMIT 4;`
   show exactly one branch and two new rows per edit.
7. Refresh the page, open the branch from the sidebar: the edited question and its answer are there.
8. With a wrong `OPENAI_API_KEY`: error + Retry, the edited question stays, no rows written;
   fix the key, Retry saves once.
9. Narrow the window: the editor and its buttons stack, nothing overlaps the sidebar.
