"""Chat model factory: Ollama (free/local), OpenAI or Gemini.

Set `LLM_PROVIDER` / `LLM_MODEL` in .env for the default; a chatbot row can
override both (`llm_provider`, `llm_model`).  Imports are lazy so you only need
the SDK of the provider you actually use.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from chat_with_website.config import LLM_DEFAULTS, settings

log = logging.getLogger(__name__)

PROVIDER_LABELS = {"ollama": "Ollama (local, free)", "openai": "OpenAI", "gemini": "Google Gemini"}


@lru_cache(maxsize=8)
def get_chat_model(provider: str | None = None, model: str | None = None, temperature: float | None = None) -> BaseChatModel:
    provider = provider or settings.llm_provider
    model = model or (settings.llm_model if provider == settings.llm_provider else None) or LLM_DEFAULTS[provider]
    temperature = settings.llm_temperature if temperature is None else temperature
    log.info("Using chat model %s (%s)", model, provider)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, base_url=settings.ollama_base_url, temperature=temperature)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")
        return ChatOpenAI(model=model, api_key=settings.openai_api_key, temperature=temperature, timeout=60, max_retries=2)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set in .env")
        return ChatGoogleGenerativeAI(model=model, google_api_key=settings.google_api_key, temperature=temperature, timeout=60, max_retries=2)

    raise ValueError(f"Unknown LLM provider: {provider}")


def resolve_llm_for_chatbot(chatbot: dict | None) -> BaseChatModel:
    """Apply a chatbot's optional provider/model override on top of the .env default."""
    if not chatbot:
        return get_chat_model()
    return get_chat_model(chatbot.get("llm_provider") or None, chatbot.get("llm_model") or None)


def describe_llm(chatbot: dict | None = None) -> str:
    provider = (chatbot or {}).get("llm_provider") or settings.llm_provider
    model = (chatbot or {}).get("llm_model") or (settings.llm_model if provider == settings.llm_provider else None) or LLM_DEFAULTS[provider]
    return f"{PROVIDER_LABELS.get(provider, provider)} · {model}"
