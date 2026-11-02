"""Concrete LLM providers."""

from __future__ import annotations

from app.config import Settings
from app.errors import LLMError
from app.llm.anthropic_client import AnthropicClient
from app.llm.base import LLMClient
from app.llm.mock_client import MockClient
from app.llm.ollama_client import OllamaClient
from app.llm.openai_client import OpenAIClient

PROVIDERS = ("openai", "anthropic", "ollama", "mock")


def build_client(settings: Settings) -> LLMClient:
    """Instantiate the configured provider."""
    factories = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "ollama": OllamaClient,
        "mock": MockClient,
    }
    factory = factories.get(settings.provider)
    if factory is None:
        raise LLMError(
            f"Неизвестный LLM_PROVIDER «{settings.provider}». Доступны: {', '.join(PROVIDERS)}"
        )
    return factory(settings)


__all__ = [
    "AnthropicClient",
    "LLMClient",
    "MockClient",
    "OllamaClient",
    "OpenAIClient",
    "PROVIDERS",
    "build_client",
]
