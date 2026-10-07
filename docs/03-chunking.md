# Step 3 – Chunking

## Why split pages at all?

1. **Embedding models have a window.** all-MiniLM-L6-v2 reads about 256 word-piece tokens
   (≈ 800–1000 characters). Anything beyond that is silently cut off.
2. **Retrieval precision.** One vector for a 3 000-word page is a blurry average of
   everything on it. A vector for one paragraph about "managed hosting" is sharp, so the
   question "what does hosting include?" lands on it.
3. **Prompt budget.** We send the LLM 5 chunks of ~200 words, not 5 whole pages.

## Strategy: recursive character splitting

`RecursiveCharacterTextSplitter` tries separators in order – paragraph break `\n\n`, line `\n`,
sentence end `. `, then words – and only falls to the next one when a piece is still too long.
Result: chunks cut at natural boundaries, not mid-sentence.

Settings (`.env`): `CHUNK_SIZE=800`, `CHUNK_OVERLAP=120`.

* **Size 800 characters** keeps every chunk inside the embedding window with room for the title.
* **Overlap 120** repeats the tail of one chunk at the head of the next, so a sentence that
  straddles the cut is fully present in at least one chunk.

Chunks shorter than 40 characters (a lone heading) are dropped.

### Title prefix
Each chunk is embedded as `"{page title}\n\n{chunk text}"` (`Chunk.embed_text`). A chunk saying
"Prices start at $49/month" says nothing about *what*; prefixing "Managed Hosting | Acme" lets
the vector carry that context. Only the embedding sees the prefix; we store the raw chunk text.

## Metadata – why every chunk carries it

```python
{
  "chunk_id": "42:3",          # page_id:index – stable, unique
  "website_id": 7, "page_id": 42,
  "url": "https://acme.example/services/hosting",
  "title": "Managed Hosting | Acme",
  "chunk_index": 3, "chunk_count": 9,
  "source": "web",
  "crawled_at": "2026-10-07T08:12:00+00:00"
}
```

* `url` / `title` → shown as **sources** under every answer.
* `website_id` → the **filter** that keeps chatbots inside their own websites.
* `chunk_index` / `chunk_count` → lets us later fetch neighbouring chunks for more context.
* `source` → room for PDFs or docs alongside web pages later.

The metadata is stored as JSONB in `chunks.metadata` and the hot fields (`website_id`,
`page_id`, `chunk_index`) are also real columns with indexes.

## Code

`ingestion/chunker.py` – `build_splitter()` and `chunk_page(page, website_id, page_id)` →
`list[Chunk]`. Pure functions: no network, no database, easy to unit-test.

## Test this step

```powershell
pytest tests/test_chunker.py -q
```

Interactive check (prints the number of chunks, the first chunk and its metadata for the README):
```powershell
python -c "from chat_with_website.ingestion.chunker import chunk_page; from chat_with_website.ingestion.models import PageContent; t=open('README.md',encoding='utf8').read(); cs=chunk_page(PageContent('https://x','T',t,'h'),1,1); print(len(cs)); print(cs[0].content[:300]); print(cs[0].metadata)"
```

Tuning hints: small FAQ-style sites → `CHUNK_SIZE=500`; long articles → 1000 (the maximum
sensible for MiniLM; OpenAI/Gemini embeddings accept much longer input, 1500–2000 is fine there).

Next: [Step 4 – Embeddings and storage](04-embeddings-and-storage.md)
