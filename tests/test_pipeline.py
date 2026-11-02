"""Pipeline orchestration: happy path, repairs and failure modes."""

from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.errors import InsufficientDataError, LLMError, OutputValidationError
from app.llm.mock_client import MockClient, build_mock_card
from app.models import MISSING_MARK, ProductIn
from app.pipeline import Pipeline
from tests.fakes import SequenceClient


def _valid(product: ProductIn) -> str:
    return json.dumps(build_mock_card(product), ensure_ascii=False)


def _with(**overrides) -> str:
    product = ProductIn(
        name="Манометр DM 02",
        manufacturer="Манотомь",
        specs={"диапазон измерений": "0–1,6 МПа", "класс точности": "1,5"},
    )
    data = build_mock_card(product)
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


async def test_mock_happy_path(settings: Settings, product: ProductIn) -> None:
    result = await Pipeline(MockClient(settings), settings).generate(product)

    assert result.meta.attempts == 1
    assert result.meta.warnings == []
    assert result.meta.provider == "mock"
    assert result.missing_data == []
    assert len(result.card.seo_title) <= 70
    assert 800 <= len(result.card.seo_description) <= 1200
    assert 3 <= len(result.card.search_queries) <= 5
    assert 300 <= len(result.card.commercial_block) <= 700
    assert len(result.card.key_specs) == len(product.specs)
    assert "## Ключевые характеристики" in result.markdown
    assert "| Диапазон измерений | 0–1,6 МПа |" in result.markdown
    assert result.meta.duration_ms >= 0


async def test_sparse_product_reports_missing_data(
    settings: Settings, sparse_product: ProductIn
) -> None:
    result = await Pipeline(MockClient(settings), settings).generate(sparse_product)

    fields = {item.field for item in result.missing_data}
    values = {item.name: item.value for item in result.card.key_specs}

    assert fields == {"specs.материал корпуса", "description"}
    assert values["Материал корпуса"] == MISSING_MARK
    assert "`specs.материал корпуса`" in result.markdown


async def test_insufficient_data_is_rejected(settings: Settings) -> None:
    product = ProductIn(name="Прибор", manufacturer="Завод")

    with pytest.raises(InsufficientDataError) as info:
        await Pipeline(MockClient(settings), settings).generate(product)

    assert {item.field for item in info.value.missing} == {"specs", "description"}


async def test_repair_is_requested_for_short_description(
    settings: Settings, product: ProductIn
) -> None:
    too_short = _with(seo_description="Слишком коротко.")
    client = SequenceClient([too_short, _valid(product)])
    pipeline = Pipeline(client, Settings(max_repairs=1))

    result = await pipeline.generate(product)

    assert result.meta.attempts == 2
    assert len(client.calls) == 2
    assert client.calls[1].repair is True
    assert "seo_description" in client.calls[1].user
    assert 800 <= len(result.card.seo_description) <= 1200


async def test_repair_exhaustion_falls_back_to_warnings(
    settings: Settings, product: ProductIn
) -> None:
    too_short = _with(seo_description="Слишком коротко.")
    client = SequenceClient([too_short, too_short])
    pipeline = Pipeline(client, Settings(max_repairs=1))

    result = await pipeline.generate(product)
    warnings = " ".join(result.meta.warnings)

    assert result.meta.attempts == 2
    assert result.card.seo_description == "Слишком коротко."
    assert "короче 800" in warnings
    assert "Не удалось исправить автоматически" in warnings


async def test_invalid_json_is_repaired(settings: Settings, product: ProductIn) -> None:
    client = SequenceClient(['{"seo_title": }', _valid(product)])

    result = await Pipeline(client, Settings(max_repairs=1)).generate(product)

    assert result.meta.attempts == 2
    assert client.calls[1].repair is True
    assert "невалидный JSON" in client.calls[1].user


async def test_invalid_json_after_all_attempts_raises(
    settings: Settings, product: ProductIn
) -> None:
    client = SequenceClient(["не JSON", "тоже не JSON"])

    with pytest.raises(OutputValidationError, match="после 2 попыток"):
        await Pipeline(client, Settings(max_repairs=1)).generate(product)


async def test_fenced_json_is_accepted(settings: Settings, product: ProductIn) -> None:
    client = SequenceClient([f"```json\n{_valid(product)}\n```"])

    result = await Pipeline(client, Settings(max_repairs=1)).generate(product)

    assert result.meta.attempts == 1


async def test_invented_spec_is_replaced(product: ProductIn) -> None:
    data = build_mock_card(product)
    data["key_specs"].append({"name": "Питание", "value": "220 В", "source_field": "питание"})
    client = SequenceClient([json.dumps(data, ensure_ascii=False)])

    result = await Pipeline(client, Settings(max_repairs=0)).generate(product)
    values = {item.name: item.value for item in result.card.key_specs}
    warnings = " ".join(result.meta.warnings)

    assert result.meta.attempts == 1
    assert values["Питание"] == MISSING_MARK
    assert "Питание" in warnings
    assert "отсутствует во входных данных" in warnings


async def test_invented_number_in_text_is_reported(product: ProductIn) -> None:
    data = build_mock_card(product)
    data["seo_description"] += " Гарантия 5 лет."
    client = SequenceClient([json.dumps(data, ensure_ascii=False)])

    result = await Pipeline(client, Settings(max_repairs=0)).generate(product)
    warnings = " ".join(result.meta.warnings)

    assert "Гарантия 5 лет" in result.card.seo_description
    assert "Не удалось исправить автоматически" in warnings
    assert "5" in warnings


async def test_llm_error_is_propagated(settings: Settings, product: ProductIn) -> None:
    client = SequenceClient([LLMError("провайдер недоступен", provider="fake")])

    with pytest.raises(LLMError, match="провайдер недоступен"):
        await Pipeline(client, settings).generate(product)


async def test_negative_repair_limit_means_single_attempt(product: ProductIn) -> None:
    client = SequenceClient([_valid(product)])
    result = await Pipeline(client, Settings(max_repairs=-5)).generate(product)

    assert result.meta.attempts == 1
