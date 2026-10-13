# Step 11 – Clickable citations and the Sources panel

Answers cite the passages they used as `[1]`, `[2][3]`. This step makes those markers links to
the pages they come from, shows the retrieved text behind each source, validates the numbers
the model produces, and hides retrieval scores from regular users.

## 1. How a citation number maps to a page

```
format_context(docs)      numbers the retrieved passages 1..n in the prompt the model sees
       │
model answers             "...programming language [2][3]."
       │
finalize()                normalize_citations: "[1, 3]" / "[1-3]" -> "[1][3]" / "[1][2][3]";
                          numbers outside 1..n are REMOVED (the model invented them)
       │
build_sources(docs, text) one Source per passage: n, url, title, score, cited (n appears in
                          the cleaned text), snippet (first 500 chars of the passage), chunk_id
       │
messages.sources (JSONB)  stored with the answer -> the message carries its own mapping
       │
render                    linkify_citations(text, sources): "[2]" -> "[[2]](url)"  (render time only)
                          _render_sources(sources): one card per page, excerpt popover
```

Because the mapping is stored with each assistant message, links are right after a page
refresh, when an old conversation is reopened, after **Regenerate** (the new answer stores a new
list) and in a branch created by **Edit** (the new answer gets fresh sources; the copied turns
keep their own). The stored answer text keeps plain `[n]` markers, so **Copy** gives clean text.

`normalize_citations` and `linkify_citations` never touch fenced code, inline code or the alt
text of markdown images (the logo answers), so a `[1]` inside those is left alone.

## 2. The score is cosine similarity

`repository.similarity_search` orders by pgvector's `<=>` operator (cosine *distance*) and
returns `1 - distance`, i.e. **cosine similarity** in 0..1 for normalised embeddings.
`SCORE_THRESHOLD` (0.25) discards weaker passages. It says how close the passage's embedding is to
the question's, nothing more: it is not a probability that the passage is correct, and the UI no
longer shows it to regular users. Turn on **Advanced settings → Show retrieved chunks** to see
`similarity 0.28` on each card and the full retrieved passages.

## 3. The Sources panel

- Still below the answer actions, still grouped into **Cited** and **Also retrieved**.
- One compact card per page: `**[1][3]** Title` (link to the page), the readable address
  (`example.com/docs/page`, trimmed), and an **Excerpt** popover with the retrieved text. Several
  chunks from the same page merge into one card with all their numbers and excerpts.
- Excerpts are shown verbatim (markdown characters escaped); words from your question are bolded
  so you can spot the relevant part. Old answers saved before this step have no excerpt and show
  "Excerpt not available".
- Missing metadata falls back to the address or "Untitled page"; a source without a URL is
  labelled "source unavailable" and never becomes a link.
- Long URLs wrap (`overflow-wrap: anywhere`) so the panel never grows wider than the chat on a
  phone; links are ordinary anchors, so they are keyboard-focusable.

## 4. Prompt rules (wording only; retrieval unchanged)

The answer prompt now tells the model to cite right after the sentence a passage supports, to
cite only passages whose text directly supports it, to use only the given numbers, to say which
part of a question the content does not cover, and never to extrapolate. The sentinel reply for
"nothing relevant" is unchanged; those answers carry no sources and no markers.

## Test this step

Automated: `pytest -q tests/test_citations.py tests/test_chain.py tests/test_ui.py`.

Manual:
1. A question answered by one page → one cited card, the `[1]` in the answer opens that page.
2. A question answered by several pages → several cards, each number links to its own page.
3. Something not on the website → the "couldn't find that" reply, no Sources panel.
4. An answer with `[1][2][3]` → three links.
5. Two chunks from the same page → one card with `[1][2]` and two excerpts.
6. Open a conversation saved before this step → links still work, "Excerpt not available".
7. Regenerate → the new answer's numbers and cards match each other.
8. Edit a question → the branch's answer has fresh sources; the original keeps its own.
9. Refresh and reopen → unchanged.
10. A page with a very long URL at ~400 px width → the card wraps, nothing overflows.
11. Hallucinated numbers: covered by the unit test (`[9]` with two passages is dropped); the
    warning "Dropped citation number(s)" appears in the log when it happens live.
12. Dead links: the app does not check them (no outbound requests from the UI); the card still
    links to the URL that was indexed.
