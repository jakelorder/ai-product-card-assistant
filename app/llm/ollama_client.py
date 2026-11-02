"""Local Ollama provider (no API key required)."""

from __future__ import annotations

from app.config import Settings
from app.errors import LLMError
from app.llm.base import LLMClient, LLMRequest, post_for_json


class OllamaClient(LLMClient):
    provider = "ollama"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings.model)
        self.base_url = settings.base_url
        self.timeout = settings.timeout
        self.retries = settings.max_retries
        self.temperature = settings.temperature

    async def generate(self, request: LLMRequest) -> str:
        data = await post_for_json(
            f"{self.base_url}/api/chat",
            provider=self.provider,
            headers={"Content-Type": "application/json"},
            payload={
                "model": self.model,
                "stream": False,
                "format": "json",
                "options": {"temperature": self.temperature},
                "messages": [
                    {"role": "system", "content": request.system},
                    {"role": "user", "content": request.user},
                ],
            },
            timeout=self.timeout,
            retries=self.retries,
        )
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(
                f"{self.provider}: неожиданный формат ответа API",
                provider=self.provider,
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError(f"{self.provider}: пустой ответ модели", provider=self.provider)
        return content
