"""Prompt templates for the RAG chain."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

NOT_ENOUGH_INFO = "I don't have enough information in the indexed website content to answer that."

# 1) Turn a follow-up ("tell me more about the second one") into a standalone
#    question, using the conversation so far.  Only the standalone question is
#    embedded and searched - the history itself never goes to the vector search.
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite the user's latest message as a single, self-contained search question "
            "that includes every entity and detail it refers to from the conversation. "
            "Rules:\n"
            "- Output ONLY the rewritten question, nothing else.\n"
            "- If the latest message is already self-contained, output it unchanged.\n"
            "- Never answer the question.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

# 2) Answer strictly from the retrieved context, citing sources.
ANSWER_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a website using ONLY the context below.
{custom_instructions}
Rules:
- Base every statement on the numbered context passages. Do not use outside knowledge.
- Cite the passages you used with their number in square brackets, e.g. [1] or [2][3].
- If the context does not contain the answer, reply exactly: "{not_enough_info}"
  You may add one short sentence about what related information IS available.
- Be concise and direct. Use bullet points for lists of items (services, steps, features).
- Answer in the same language as the question.

Context:
{context}"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)
