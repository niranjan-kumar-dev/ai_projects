# Step 12 – Suggested follow-up questions

Under the latest answer the chatbot now shows up to three short questions a visitor might ask
next. Clicking one asks it as a normal message in the same conversation: fresh retrieval, the
usual loading cues, a streamed answer, one save, and new suggestions under the new answer.

## 1. How suggestions are generated

```
prepare()  -> retrieval (passages)        ─┐
stream_answer() -> streamed answer         ├─ unchanged
finalize() -> validated answer + sources  ─┘
suggest_followups(prepared, result, history)   <- NEW: one small, non-streaming call, after the
                                                   answer is already on screen
```

- **Separate call, not part of the answer.** Embedding a JSON block in the streamed answer would
  delay the first token and need fragile parsing, so suggestions come from one extra request to
  the chatbot's own chat model (`self.llm`, so provider and model follow the chatbot settings).
- **Only what the answer was built from.** The prompt (`FOLLOWUP_PROMPT` in `rag/prompts.py`)
  receives the question, the final answer, the titles plus the first 300 characters of the
  retrieved passages already in memory (no second vector search), and the earlier questions of
  the conversation. It must suggest only questions answerable from those passages, never invent
  services, prices or features, never repeat something already asked, and may return `[]`.
- **Structured output, validated.** The model is asked for a JSON array of strings and nothing
  else. `parse_followups` (`rag/followups.py`) extracts the first array even if it is wrapped in a
  code fence or prose and validates it against a Pydantic model (`FollowUps`); if that fails it
  falls back to lines ending in "?". `clean_followups` then trims, caps at 120 characters, adds a
  missing "?", removes near-duplicates and anything equal to the current or an earlier question,
  and keeps at most `FOLLOWUP_COUNT` (default 3, `.env`; `0` disables the feature and the call).
- **Never on the critical path.** Greetings, out-of-scope replies and failed answers get no
  suggestions and no call. Any error in the extra call is logged and yields none; the answer has
  already been shown and is saved regardless.

Why not `with_structured_output`? On malformed JSON that path costs a second model call, small
local models often fail it, and the fake model used in the unit tests does not implement it. One
plain call plus Pydantic validation is cheaper and testable.

## 2. Cost

One extra gpt-4o-mini request of roughly 600–900 tokens per grounded answer (about a quarter to
a third of a turn's tokens, i.e. a few hundredths of a cent). Nothing is spent on reopening a
conversation (suggestions are stored), on greetings, on out-of-scope answers or on failures. No
extra embedding or retrieval.

## 3. Storage

```sql
ALTER TABLE messages ADD COLUMN IF NOT EXISTS suggestions JSONB;   -- in schema_upgrades.sql
```
The list is saved on the assistant row by `persist_turn`, replaced by `update_message` on
**Regenerate**, and read back by `get_messages`. The column applies itself at app start
(`ensure_schema_upgrades`), locally and on Neon.

Where suggestions appear and disappear:

| Situation | Behaviour |
|---|---|
| Fresh answer | Generated once, stored, shown under that answer |
| Reopen a conversation | Shown from storage under the **last** answer only; no new call |
| Older answers | Their stored suggestions are kept but not shown (only the latest answer has them) |
| Regenerate | New suggestions replace the old ones for that answer |
| Edit a question (branch) | Copied answers in the branch carry none; the branch's new answer gets its own |
| Click a suggestion | Asked like a typed question; the new answer gets new suggestions |

## 4. UI

A caption "Suggested follow-up questions" and a row of tertiary buttons with a sparkle icon,
wrapping on narrow screens, placed under the Sources panel so Copy / Regenerate / thumbs /
Sources are untouched. While an answer is streaming the buttons are disabled, and the click goes
through the same queued-question mechanism as Retry (`pending_question`, popped once), so a
Streamlit rerun cannot submit it twice. A follow-up click also drops any open edit, so it can
never be stored as an "edited" question.

## Test this step

Automated: `pytest -q tests/test_followups.py tests/test_ui.py tests/test_chain.py`.

Manual:
1. Ask a general question about the website → 2–3 suggestions under the answer.
2. Ask about a specific service → suggestions stay on that topic; nothing invented.
3. Click one → it appears as your message, the loading cues run, the answer streams.
4. The earlier turns are still there and were used as context.
5. "Show retrieved chunks" shows new passages for the clicked question.
6. Suggestions never repeat each other or the question just asked.
7. A question with no website coverage → the usual "couldn't find that" reply, no suggestions.
8. Regenerate → the suggestions change with the answer.
9. Edit an earlier question → the branch shows suggestions only under its new answer.
10. Refresh and reopen → the last answer's suggestions are back, no model call made.
11. Set `FOLLOWUP_COUNT=0` (or break the model) → answers unaffected, no suggestions.
12. `SELECT id, suggestions FROM messages ORDER BY id DESC LIMIT 2;` shows one row per answer,
    and the log shows one "Follow-up suggestions" line per grounded answer.
