"""Shared fixtures for the test suite."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import get_pipeline, get_settings
from app.llm.mock_client import MockClient, build_mock_card
from app.main import app
from app.models import ProductCard, ProductIn
from app.pipeline import Pipeline


@pytest.fixture
def settings() -> Settings:
    return Settings(provider="mock", model="mock-1", max_retries=0, max_repairs=1)


@pytest.fixture
def product() -> ProductIn:
    return ProductIn(
        name="Манометр показывающий DM 02",
        manufacturer="Манотомь",
        specs={
            "диапазон измерений": "0–1,6 МПа",
            "класс точности": "1,5",
            "присоединение": "G1/2",
            "диаметр корпуса": "100 мм",
        },
        description="Прибор предназначен для измерения избыточного давления.",
    )


@pytest.fixture
def sparse_product() -> ProductIn:
    return ProductIn(
        name="Термометр биметаллический ТБ-100",
        manufacturer="Демо-Прибор",
        specs={
            "диапазон измерений": "0…+120 °C",
            "материал корпуса": "",
        },
        description=None,
    )


@pytest.fixture
def mock_pipeline(settings: Settings) -> Pipeline:
    return Pipeline(MockClient(settings), settings)


@pytest.fixture
def client(settings: Settings, mock_pipeline: Pipeline):
    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def make_card(product: ProductIn):
    """Build a valid card from the mock provider and apply overrides."""

    def factory(**overrides: Any) -> ProductCard:
        data = build_mock_card(product)
        data.update(overrides)
        return ProductCard.model_validate(data)

    return factory
