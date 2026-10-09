# Step 6 – RAG chain and conversation

## The chain

```
question + history
   │
   ├─ 1. condense: history + question ─► LLM ─► standalone question   (skipped if no history)
   ├─ 2. retrieve: standalone question ─► embed ─► pgvector ─► top-k chunks
   ├─ 3. format:   "[1] Title (url)\n text\n\n[2] ..."
   └─ 4. answer:   system rules + context + history + question ─► LLM ─► answer with [n] citations
                                                                   └─► sources (url, title, score, cited?)
```

Built with **LCEL** – LangChain's pipe syntax: `prompt | llm | StrOutputParser()`.
Each stage is a plain object you can run and test alone (`rag/chain.py::RagChain`).

### 1. Why rewrite the question?
"Tell me more about the second service" has no meaning for a vector search – nothing on the site
is literally called "the second service". The condense step reads the history and produces, for
example, "What is included in Acme's managed hosting service?", which *does* match the right
chunk. Only this standalone question is embedded; the full history goes to the answer step.

### 2–3. Grounding
The answer prompt (`rag/prompts.py`) is strict:

* answer **only** from the numbered passages, no outside knowledge;
* cite with `[1]`, `[2]`;
* if the passages do not contain the answer, reply with the fixed sentence
  *"I don't have enough information in the indexed website content to answer that."*

That fixed sentence lets code detect a refusal (`RagAnswer.grounded == False`) and hide sources.
Temperature is 0 for reproducible answers. A chatbot's `system_prompt` is inserted as extra
instructions (tone, persona) without weakening the rules.

### 4. Sources
`build_sources()` lists every passage the model saw, numbered as in the prompt, and marks the
ones whose `[n]` appears in the answer as **cited**. The UI shows cited pages first and the rest
under "also retrieved", so users can always verify an answer.

## Conversation memory

* The UI/CLI keep `history` as LangChain `HumanMessage`/`AIMessage` objects.
* `trim_history()` keeps the last `HISTORY_TURNS` (6) exchanges so the prompt stays small and
  old topics stop steering the rewrite.
* `answer_question(..., conversation_id=...)` persists both messages to the `messages` table, so a
  conversation can be resumed later (`ask.py --conversation <uuid>`, or from the UI sidebar).

## Chat model providers (`rag/llm.py`)

| `LLM_PROVIDER` | class | default model | needs |
|---|---|---|---|
| `ollama` | `ChatOllama` | `llama3.2` | Ollama running locally |
| `openai` (default) | `ChatOpenAI` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| `gemini` | `ChatGoogleGenerativeAI` | `gemini-3.8-flash` | `GOOGLE_API_KEY` |

A chatbot can override provider and model (`--llm-provider`, `--llm-model`, or the Chatbots page),
so one deployment can run a cheap local bot and a premium hosted bot side by side. Unlike
embeddings, switching the chat model never requires re-indexing.

## Test this step

Unit tests use a fake LLM so they run without Ollama:
```powershell
pytest tests/test_chain.py -q
```

Real conversation:
```powershell
python scripts/ask.py --chatbot 1 --debug
```
```
You: What is Python good for?
You: Tell me more about the second point.      ← watch the "[standalone question]" line
You: How can I join the community?
You: Who won the 2022 FIFA World Cup?          ← must refuse, no sources
```
`--debug` prints the rewritten question and the retrieved chunks so you can see *why* an answer
came out the way it did. If answers are wrong, check the chunks first: bad retrieval is the
cause far more often than the LLM.

Next: [Step 7 – Streamlit UI and production](07-streamlit-ui-and-production.md)
