"""Provider-agnostic LLM interface and HTTP helper with retries."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

from app.errors import LLMError
from app.models import ProductIn

RETRYABLE_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_ERROR_DETAIL = 300


@dataclass(slots=True)
class LLMRequest:
    """Everything a provider needs for one call.

    ``product`` is passed as structured data so offline providers (mock,
    tests) can build an answer without parsing the prompt back.
    """

    system: str
    user: str
    product: ProductIn
    repair: bool = False


class LLMClient(ABC):
    """Minimal interface every provider implements."""

    provider: str = "base"

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    async def generate(self, request: LLMRequest) -> str:
        """Return the raw model response expected to contain JSON."""


async def post_for_json(
    url: str,
    *,
    provider: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
    retries: int,
    backoff: float = 0.6,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """POST JSON with retries on timeouts and transient provider errors."""
    last_error = "неизвестная ошибка"

    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        for attempt in range(retries + 1):
            try:
                response = await client.post(url, headers=headers, json=payload)
            except httpx.TimeoutException:
                last_error = f"таймаут {timeout:.0f} с"
            except httpx.HTTPError as exc:
                last_error = f"сетевая ошибка ({exc.__class__.__name__})"
            else:
                if response.status_code < 400:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise LLMError(
                            f"{provider}: провайдер вернул не-JSON ответ",
                            provider=provider,
                        ) from exc

                detail = _error_detail(response)
                if response.status_code not in RETRYABLE_STATUSES:
                    raise LLMError(
                        f"{provider}: HTTP {response.status_code}. {detail}",
                        provider=provider,
                        status=response.status_code,
                    )
                last_error = f"HTTP {response.status_code}: {detail}"

            if attempt < retries:
                await asyncio.sleep(backoff * (2**attempt))

    raise LLMError(
        f"{provider}: не удалось получить ответ после {retries + 1} попыток ({last_error})",
        provider=provider,
    )


def _error_detail(response: httpx.Response) -> str:
    """Extract a short human-readable message from a provider error body."""
    detail = response.text or ""
    try:
        body = response.json()
    except ValueError:
        return detail[:MAX_ERROR_DETAIL]

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            detail = str(error.get("message") or error.get("type") or detail)
        elif isinstance(error, str):
            detail = error
        else:
            detail = str(body.get("message") or body.get("detail") or detail)

    return detail[:MAX_ERROR_DETAIL].strip() or "без описания"


def extract_content(data: dict[str, Any], provider: str) -> str:
    """Read ``choices[0].message.content`` that may be a string or a list."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(
            f"{provider}: неожиданный формат ответа API",
            provider=provider,
        ) from exc

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in (None, "text")
        ]
        return "".join(parts)
    raise LLMError(f"{provider}: пустой ответ модели", provider=provider)


def ensure_api_key(api_key: str | None, provider: str) -> str:
    if not api_key:
        raise LLMError(
            f"{provider}: не задан API-ключ. Укажите LLM_API_KEY в .env",
            provider=provider,
        )
    return api_key
