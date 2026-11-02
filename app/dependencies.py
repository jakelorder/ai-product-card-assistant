"""FastAPI dependencies and cached singletons."""

from __future__ import annotations

from functools import lru_cache

from app.config import Settings
from app.llm import build_client
from app.pipeline import Pipeline


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def get_pipeline() -> Pipeline:
    settings = get_settings()
    return Pipeline(build_client(settings), settings)
