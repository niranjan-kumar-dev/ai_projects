# Step 5 – Chatbots and vector search

## What we are building

A **chatbot** is a named configuration: which websites it may read, which chat model it uses,
optional extra instructions. Many chatbots can share one database, and the search for one
chatbot must never return another chatbot's content.

```
chatbots ──< chatbot_websites >── websites ──< pages ──< chunks
```

`create_chatbot.py --name python --websites 1` writes a `chatbots` row and the join rows.

## Vector search in plain words

1. Embed the question with the **same model** used for the chunks → one 384-number vector.
2. Ask PostgreSQL for the chunks whose vectors are closest (smallest cosine distance).
3. Return the text + metadata of the best `top_k` chunks.

The SQL (in `db/repository.py::similarity_search`):

```sql
SELECT c.content, c.metadata, p.url, p.title,
       1 - (c.embedding <=> %(q)s) AS score          -- cosine similarity
FROM chunks c JOIN pages p ON p.id = c.page_id
WHERE c.website_id = ANY(%(ids)s)                      -- <-- the chatbot's websites only
ORDER BY c.embedding <=> %(q)s                         -- nearest first (uses the HNSW index)
LIMIT %(k)s;
```

### Metadata filtering – why it is in the WHERE clause
Filtering *after* the vector search ("get 5 nearest, then drop other websites") could return 0
usable rows. Filtering *inside* the query, on an indexed column, guarantees `k` rows from the
right websites. pgvector 0.8 adds `SET hnsw.iterative_scan = relaxed_order`, which keeps walking
the index until enough rows pass the filter instead of giving up early; we set it per query.

### Scores
`score = 1 - cosine_distance`, in `[0, 1]` for normalised vectors. Rough guide for MiniLM:

| score | meaning |
|---|---|
| > 0.6 | very relevant |
| 0.4 – 0.6 | related |
| < 0.3 | mostly noise |

`SCORE_THRESHOLD` (default 0.25) drops noise; if **no** chunk passes, the chain answers
"not enough information" without even calling the LLM. Raise it for stricter bots, lower it for
small sites where phrasing differs a lot from the questions. Hosted models (OpenAI, Gemini)
produce different score ranges; re-tune after switching.

## Code

* `rag/retriever.py`
  * `search_chunks(chatbot_id, query, k, min_score)` – resolves website ids
    (`get_chatbot_website_ids`), embeds the query, calls `similarity_search`, converts rows to
    LangChain `Document`s with `metadata = {url, title, score, chunk_id, website_id, ...}`.
  * `PgVectorChatbotRetriever(BaseRetriever)` – the same thing as a LangChain retriever so it
    plugs into any chain.
* `db/repository.py::set_chatbot_websites` refuses to mix websites embedded with different models.

## Test this step

```powershell
python scripts/create_chatbot.py --name python --websites 1
python scripts/create_chatbot.py --list
python scripts/ask.py --chatbot 1 --search-only "how do I get started with python" -k 3
```
You should see three chunks with scores and URLs, the most relevant first.

Isolation test: index a second site, attach it to a second chatbot, run the same
`--search-only` query against both. Each chatbot must only ever show its own domain.

Next: [Step 6 – RAG chain and conversation](06-rag-chain-and-conversation.md)
