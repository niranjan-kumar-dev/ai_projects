# Step 0 – Architecture and project structure

## What we are building

A chatbot that answers questions using **only the content of websites you choose**.
Nothing is hard-coded about a particular site: you give it a URL, it reads the site,
and from then on a chatbot can answer questions about it, quote the page it used,
and handle follow-up questions.

The technique is called **RAG – Retrieval-Augmented Generation**:

| Word | Plain meaning |
|---|---|
| **Retrieval** | Find the few paragraphs of your data that are relevant to the question. |
| **Augmented** | Put those paragraphs into the prompt. |
| **Generation** | Let a language model (LLM) write the answer from them. |

Without RAG an LLM answers from memory and happily invents facts.
With RAG it works like an open-book exam: it may only use the pages we hand it,
and if the answer is not on those pages it must say so.

## The two pipelines

```
INGESTION (runs once per website, offline)

  URL ──► Crawl ──► Extract & clean ──► Chunk ──► Embed ──► Store
          crawler    extractor           chunker   embedder   PostgreSQL + pgvector

QUERY (runs for every question)

  question + chat history ──► rewrite to standalone question ──► embed
      ──► vector search in PostgreSQL (only this chatbot's websites)
      ──► top chunks + question ──► LLM ──► answer + source URLs
```

### Why these pieces?

* **Crawler** – one URL is rarely enough. Services live on `/services`, contact info on `/contact`.
  We follow links on the same domain, politely (robots.txt, delays, page limit).
* **Extractor** – raw HTML is 90 % noise: menus, scripts, cookie banners, footers.
  Feeding noise to the model wastes its attention and pollutes search results.
* **Chunker** – embedding models read a limited window and search works best on
  focused passages. We split pages into ~800-character overlapping chunks.
* **Embedder** – turns text into a list of numbers (a *vector*) that encodes meaning.
  Similar meanings → nearby vectors. This is what makes "search by meaning" possible.
* **PostgreSQL + pgvector** – a real database that also stores vectors and can find the
  nearest ones. One database holds many websites and many chatbots, with proper
  relations between them.
* **LLM** – writes the final answer. Ollama (free, local) during development;
  OpenAI or Gemini later by changing one line in `.env`.

## Project structure

```
CHAT_WITH_WEBSITE/
├── .env                     your secrets and settings (git-ignored)
├── .env.example             template for .env
├── requirements.txt
├── pyproject.toml           makes `chat_with_website` an installable package
├── README.md
├── docs/                    these step-by-step guides
├── scripts/                 command-line entry points
│   ├── init_db.py           create database + tables
│   ├── ingest.py            index a website
│   ├── create_chatbot.py    (stage 4) create a chatbot and attach websites
│   └── ask.py               (stage 5) chat from the terminal
├── src/chat_with_website/   the application package
│   ├── config.py            all settings, read from .env
│   ├── logging_config.py
│   ├── db/
│   │   ├── schema.sql       tables and indexes
│   │   ├── connection.py    connection pool
│   │   └── repository.py    every SQL statement in one place
│   ├── ingestion/
│   │   ├── models.py        small dataclasses passed between steps
│   │   ├── url_utils.py     URL normalisation + safety checks
│   │   ├── crawler.py
│   │   ├── extractor.py
│   │   ├── chunker.py
│   │   ├── embedder.py      provider factory (HuggingFace / OpenAI / Gemini)
│   │   └── pipeline.py      glues the steps together
│   ├── rag/                 (stages 4-5) retriever, prompts, chain, LLM factory
│   └── ui/                  (stage 6) Streamlit app
└── tests/                   pytest unit tests
```

Rule of thumb used throughout: **scripts call the package, the package never
reads `sys.argv`**. That way the same code serves the CLI, the tests and the UI.

## Configuration (`config.py`)

`Settings` is a `pydantic-settings` model: every attribute can be set by an
environment variable of the same name (case-insensitive) or a line in `.env`.
Two switches matter most:

| Variable | Values | Effect |
|---|---|---|
| `EMBEDDING_PROVIDER` | `huggingface` (default), `openai`, `gemini` | Which model turns text into vectors. Changing it changes the vector size, so the `chunks` table must be reset and websites re-ingested. |
| `LLM_PROVIDER` | `ollama` (default), `openai`, `gemini` | Which model writes answers. Can be changed at any time. |

Defaults per provider live in `EMBEDDING_DEFAULTS` / `LLM_DEFAULTS` in `config.py`.

## Install

```powershell
conda activate chat-with-website
pip install -r requirements.txt
pip install -e .            # installs the package in "editable" mode so imports work everywhere
copy .env.example .env      # then edit .env
```

## Test this step

```powershell
python -c "from chat_with_website.config import settings; print(settings.embedding_provider, settings.resolved_embedding_model, settings.resolved_embedding_dim)"
```
Expected: `huggingface sentence-transformers/all-MiniLM-L6-v2 384`.

Next: [Step 1 – PostgreSQL and pgvector](01-postgres-pgvector-setup.md)
