# Step 13 – Per-chatbot branding and the admin password

Every chatbot can now have its own logo, display name, welcome heading and description, accent
colour, starter questions and chat-input placeholder. Branding is edited on the **Chatbots**
page ("Branding & Appearance") and applied automatically on the **Chat** page for the selected
chatbot. The admin pages (Websites, Chatbots) can be protected with a password.

## 1. Where branding lives

Everything is stored on the existing `chatbots` row; `name` and `description` are reused, the
rest are new nullable columns added by `db/schema_upgrades.sql` (applied automatically at start):

| Setting | Column | Default when empty |
|---|---|---|
| Display name | `display_name` | the chatbot name, title-cased |
| Welcome heading | `welcome_heading` | "Ask anything about {display name}" |
| Welcome description | `description` (existing) | "Answers come only from the website content…" |
| Accent colour | `accent_color` (`#rrggbb`) | the theme's primary colour |
| Starter questions | `starter_questions` (JSON list, max 4) | the three default chips |
| Input placeholder | `input_placeholder` | "Message {display name}…" |
| Logo | `logo` (bytes) + `logo_updated_at` | the robot icon |

Existing chatbots have NULL everywhere and look exactly as before. `branding_of(row)`
(`ui/branding.py`) turns a row into the values to show, filling defaults; the Chat page calls it
on every run for the *selected* chatbot and never keeps branding in session state, so switching
chatbots cannot leak the previous look.

**Logos are stored in PostgreSQL**, not on disk: Streamlit Cloud's filesystem is wiped on every
deploy, and the database (local or Neon) is the one durable store this project has. Uploads are
validated by content with Pillow (a text file renamed to `.png` is rejected), limited to 2 MB,
accepted only as PNG/JPEG/WebP, converted to RGBA, shrunk to at most 256×256 and re-encoded as
PNG (which also strips metadata). The list/edit queries never select the bytes
(`CHATBOT_COLUMNS` in `repository.py`); the Chat page fetches the logo through an
`st.cache_data` function keyed on `logo_updated_at`, so every upload or removal refreshes the
cache for every session and nothing is re-read on normal reruns.

## 2. Validation and safety

- Text fields: whitespace collapsed, length limits (name 60, heading 120, description 300,
  placeholder 80); when rendered on the Chat page they pass through `escape_markdown`, so a
  heading like `*hi*` shows literally. Streamlit's markdown never renders raw HTML anyway.
- Colour: the picker is always active; choosing the theme's own colour (`#60a5fa`, or the
  "Use theme colour" button) means *no custom accent* and nothing is injected. Must be `#rgb` or
  `#rrggbb`; only the validated value reaches the injected `<style>`,
  so user text can never end up in CSS. A contrast warning appears when the colour is too close
  to the dark background (ratio < 3:1); button text colour is picked automatically (black or
  white) for readability.
- Starter questions: one per line in the order they should appear, at most 4, each ≤ 120
  characters, duplicates removed, question mark added.
- Writes happen in one transaction (`update_chatbot_branding` + `set_chatbot_logo`), tied to the
  chatbot id, and every write handler re-checks the admin status.

## 3. Admin password

Set `ADMIN_PASSWORD` in `.env` (locally) or in the Streamlit Cloud secrets. When it is set, the
Websites and Chatbots pages show a password prompt once per browser session (compared with
`hmac.compare_digest`); a "Log out of admin" button sits under "About this app". When it is not
set, the pages stay open, which keeps local development unchanged. The Chat page is never gated.

## 4. Chatbots page

The "Branding & Appearance" panel appears when an existing chatbot is being edited (create the
chatbot first). It lives outside the main form so every change reruns and the **preview** on the
right (logo, heading, description, chips, placeholder, a primary button in the accent colour)
updates live. **Save branding** writes everything at once; **Restore defaults** asks for
confirmation and clears all branding columns and the logo. Uploading a new logo replaces the old
one; "Remove the current logo" goes back to the robot icon.

## 5. Chat page

The sidebar lists chatbots by display name; the welcome screen shows the logo (or the forum
icon), the heading, the description and the starter chips; the chat input uses the placeholder;
assistant bubbles use the logo as avatar; when an accent colour is set, a small stylesheet
recolours the New chat / primary buttons, the starter chips (outline, and filled when selected),
the selected thumbs, the open conversation in the sidebar, the chat input focus ring and send
button, links in answers and keyboard focus outlines. The rules use `data-testid` hooks with
`!important` (Streamlit's own styles have the same specificity and could otherwise win), and are
emitted only for the selected chatbot, so nothing global changes.

`tests/test_branding_db.py` runs the real Chat page against the local database with two temporary
branded chatbots (logo, accent, starters, placeholder), switches between them, checks an unbranded
bot and a reset, then deletes them; it is skipped when PostgreSQL is not reachable.

## Test this step

Automated: `pytest -q tests/test_branding.py tests/test_ui.py`.

Manual:
1. Create a chatbot and open Chat: default robot icon, "Ask anything about …", three chips.
2. Edit it → Branding & Appearance: upload a logo, change heading and description, Save.
3. Pick an accent colour; primary buttons and the selected chip take it.
4. Enter four starter questions (one per line); the Chat page shows all four in that order.
5. Reopen the Chatbots page: the saved values are back in the fields.
6. Brand a second chatbot differently and switch between them in the Chat sidebar.
7. Refresh the browser: the selected chatbot's branding is unchanged.
8. Tick "Remove the current logo", Save: the robot icon is back.
9. Rename a `.txt` to `.png` and upload it: "not a valid image"; a 5 MB image: rejected.
10. Narrow the window: the panel columns and the chips wrap.
11. Chatbots created before this step work unchanged.
12. Set `ADMIN_PASSWORD`, restart: Websites/Chatbots ask for it; Chat does not.
