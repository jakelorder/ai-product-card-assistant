"""Anthropic Messages API provider."""

from __future__ import annotations

from app.config import Settings
from app.errors import LLMError
from app.llm.base import (
    LLMClient,
    LLMRequest,
    ensure_api_key,
    post_for_json,
)


class AnthropicClient(LLMClient):
    """Uses the assistant prefill trick to force a JSON-only answer."""

    provider = "anthropic"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings.model)
        self.api_key = settings.api_key
        self.base_url = settings.base_url
        self.timeout = settings.timeout
        self.retries = settings.max_retries
        self.temperature = settings.temperature

    async def generate(self, request: LLMRequest) -> str:
        api_key = ensure_api_key(self.api_key, self.provider)
        data = await post_for_json(
            f"{self.base_url}/v1/messages",
            provider=self.provider,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            payload={
                "model": self.model,
                "max_tokens": 2500,
                "temperature": self.temperature,
                "system": request.system,
                "messages": [
                    {"role": "user", "content": request.user},
                    {"role": "assistant", "content": "{"},
                ],
            },
            timeout=self.timeout,
            retries=self.retries,
        )
        return "{" + _text_from_content(data, self.provider)


def _text_from_content(data: dict, provider: str) -> str:
    blocks = data.get("content")
    if not isinstance(blocks, list):
        raise LLMError(f"{provider}: неожиданный формат ответа API", provider=provider)
    text = "".join(
        block.get("text", "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    )
    if not text.strip():
        raise LLMError(f"{provider}: пустой ответ модели", provider=provider)
    return text
