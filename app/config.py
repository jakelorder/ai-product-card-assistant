"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "ollama": "llama3.1",
    "mock": "mock-1",
}

DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "ollama": "http://localhost:11434",
}

API_KEY_ENV_NAMES: dict[str, tuple[str, ...]] = {
    "openai": ("LLM_API_KEY", "OPENAI_API_KEY"),
    "anthropic": ("LLM_API_KEY", "ANTHROPIC_API_KEY"),
    "ollama": ("LLM_API_KEY",),
    "mock": (),
}


@dataclass(frozen=True)
class Settings:
    """Runtime settings.

    ``provider="mock"`` is the default on purpose: the project must be
    runnable right after ``pip install`` without an API key. Real providers
    are enabled through ``LLM_PROVIDER``.
    """

    provider: str = "mock"
    model: str = ""
    api_key: str | None = None
    base_url: str = ""
    timeout: float = 60.0
    max_retries: int = 2
    max_repairs: int = 1
    temperature: float = 0.2

    def __post_init__(self) -> None:
        """Fill provider-dependent defaults for direct construction."""
        if not self.model:
            object.__setattr__(self, "model", DEFAULT_MODELS.get(self.provider, ""))
        if not self.base_url:
            object.__setattr__(
                self,
                "base_url",
                DEFAULT_BASE_URLS.get(self.provider, "").rstrip("/"),
            )
        else:
            object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from the process environment (``.env`` is loaded)."""
        if env is None:
            load_dotenv()
            env = os.environ

        provider = (env.get("LLM_PROVIDER") or "mock").strip().lower()

        api_key: str | None = None
        for name in API_KEY_ENV_NAMES.get(provider, ("LLM_API_KEY",)):
            value = env.get(name)
            if value and value.strip():
                api_key = value.strip()
                break

        model = (env.get("LLM_MODEL") or "").strip() or DEFAULT_MODELS.get(provider, "")
        base_url = (env.get("LLM_BASE_URL") or "").strip() or DEFAULT_BASE_URLS.get(provider) or ""

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=_float(env.get("LLM_TIMEOUT"), 60.0),
            max_retries=_int(env.get("LLM_MAX_RETRIES"), 2),
            max_repairs=_int(env.get("LLM_MAX_REPAIRS"), 1),
            temperature=_float(env.get("LLM_TEMPERATURE"), 0.2),
        )


def _float(value: str | None, default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


def _int(value: str | None, default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default
