"""OpenAI and OpenAI-compatible chat-completions providers."""

from __future__ import annotations

from app.config import Settings
from app.llm.base import (
    LLMClient,
    LLMRequest,
    ensure_api_key,
    extract_content,
    post_for_json,
)


class OpenAIClient(LLMClient):
    """Works with OpenAI and any compatible endpoint via ``LLM_BASE_URL``."""

    provider = "openai"

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
            f"{self.base_url}/chat/completions",
            provider=self.provider,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            payload={
                "model": self.model,
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": request.system},
                    {"role": "user", "content": request.user},
                ],
            },
            timeout=self.timeout,
            retries=self.retries,
        )
        return extract_content(data, self.provider)
