"""Prompt templates for the RAG chain."""

from __future__ import annotations

import re

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

NOT_ENOUGH_INFO = "I don't have enough information in the indexed website content to answer that."

# 0) Greetings never reach the retriever: "hi" would just match random passages.
#    The reply is a fixed template so it is instant, free, and identical across providers.
GREETING_WORDS = {
    "hi", "hii", "hiii", "hello", "hey", "heya", "hola", "howdy", "yo", "sup",
    "namaste", "namaskar", "greetings", "good morning", "good afternoon", "good evening",
    "good day", "hi there", "hello there", "hey there", "hiya",
}
_GREETING_SUFFIXES = ("there", "everyone", "all", "team", "bot", "assistant", "friend")

GREETING_REPLY = (
    "Hi! I'm the {chatbot_name} assistant. "
    "Ask me anything about {chatbot_name} and I'll answer from its website content."
)
GREETING_REPLY_NO_NAME = "Hi! Ask me anything about this website and I'll answer from its content."


# When the question cannot be answered from the indexed content, the user sees this
# (NOT_ENOUGH_INFO stays the sentinel the *model* emits so code can detect the refusal).
OUT_OF_SCOPE_REPLY = (
    "Sorry, I couldn't find that in {chatbot_name}'s website content. "
    "I'm the {chatbot_name} assistant - ask me anything about {chatbot_name} "
    "and I'll answer from its website content."
)
OUT_OF_SCOPE_REPLY_NO_NAME = (
    "Sorry, I couldn't find that in the indexed website content. "
    "Ask me anything about this website and I'll answer from its content."
)


def is_greeting(text: str) -> bool:
    """True when the message is only a greeting ("Hi", "hello!", "Hey there :)")."""
    cleaned = re.sub(r"[^a-z\s]", " ", text.lower())  # strip punctuation / emoji
    words = cleaned.split()
    if not words or len(words) > 4:
        return False
    phrase = " ".join(words)
    if phrase in GREETING_WORDS:
        return True
    # "hi there", "hello team" -> first word is a greeting, rest are filler
    return words[0] in GREETING_WORDS and all(w in _GREETING_SUFFIXES for w in words[1:])


def greeting_reply(chatbot_name: str | None) -> str:
    name = (chatbot_name or "").strip()
    return GREETING_REPLY.format(chatbot_name=name) if name else GREETING_REPLY_NO_NAME


def out_of_scope_reply(chatbot_name: str | None) -> str:
    name = (chatbot_name or "").strip()
    return OUT_OF_SCOPE_REPLY.format(chatbot_name=name) if name else OUT_OF_SCOPE_REPLY_NO_NAME

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
