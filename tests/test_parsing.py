"""Tolerant JSON parsing of model responses."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.errors import OutputValidationError
from app.parsing import (
    format_error_item,
    format_validation_error,
    loads_json_object,
    parse_card,
)


def _payload(make_card) -> str:
    return json.dumps(make_card().model_dump(), ensure_ascii=False)


def test_parses_plain_json(make_card) -> None:
    card = parse_card(_payload(make_card))
    assert card.seo_title


def test_parses_fenced_json(make_card) -> None:
    card = parse_card(f"```json\n{_payload(make_card)}\n```")
    assert card.key_specs


def test_parses_json_with_preface(make_card) -> None:
    card = parse_card(f"Вот результат:\n{_payload(make_card)}\nГотово.")
    assert card.search_queries


def test_repairs_trailing_comma() -> None:
    data = loads_json_object('{"a": 1, "b": [1, 2,],}')
    assert data == {"a": 1, "b": [1, 2]}


def test_empty_response_raises() -> None:
    with pytest.raises(OutputValidationError, match="не найден JSON-объект"):
        parse_card("Извините, не могу помочь")


def test_broken_json_raises() -> None:
    with pytest.raises(OutputValidationError, match="невалидный JSON"):
        parse_card('{"seo_title" "x"}')


def test_json_array_is_rejected() -> None:
    with pytest.raises(OutputValidationError, match="не найден JSON-объект"):
        loads_json_object("[1, 2, 3]")


def test_schema_violation_raises_with_details(make_card) -> None:
    data = make_card().model_dump()
    data.pop("seo_title")

    with pytest.raises(OutputValidationError, match="схеме карточки"):
        parse_card(json.dumps(data, ensure_ascii=False))


def test_format_validation_error_for_broken_json() -> None:
    try:
        from app.models import ProductIn

        ProductIn.model_validate_json("{broken")
    except ValidationError as exc:
        message = format_validation_error(exc)

    assert "Некорректный JSON" in message


def test_error_item_translates_common_messages() -> None:
    assert format_error_item({"msg": "Field required", "loc": ("name",)}) == "поле обязательно"
    assert (
        format_error_item({"msg": "Input should be a valid string", "loc": ("name",)})
        == "ожидается строка"
    )
