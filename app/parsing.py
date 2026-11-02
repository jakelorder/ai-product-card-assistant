"""Tolerant parsing of model responses into validated Pydantic models."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from app.errors import OutputValidationError
from app.models import ProductCard

_FENCE_START = re.compile(r"^```[a-zA-Z]*\s*")
_FENCE_END = re.compile(r"\s*```$")
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def parse_card(raw: str) -> ProductCard:
    """Parse a model response, tolerating fences and trailing commas."""
    data = loads_json_object(raw)
    try:
        return ProductCard.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(f"{_location(error)}: {error['msg']}" for error in exc.errors()[:3])
        raise OutputValidationError(
            f"Ответ модели не соответствует схеме карточки ({details})"
        ) from exc


def loads_json_object(raw: str) -> dict:
    """Extract the first JSON object from the response and decode it."""
    candidate = _extract_object(raw)

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as first_error:
        repaired = _TRAILING_COMMA.sub(r"\1", candidate)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            raise OutputValidationError(
                f"Модель вернула невалидный JSON: {first_error.msg} (позиция {first_error.pos})"
            ) from first_error

    if not isinstance(data, dict):
        raise OutputValidationError("Модель вернула JSON, но не объект карточки")
    return data


def format_validation_error(exc: ValidationError) -> str:
    """Turn a Pydantic error into a short Russian message for the web form."""
    errors = exc.errors()
    if not errors:
        return "Некорректные входные данные"
    if errors[0].get("type") == "json_invalid":
        return "Некорректный JSON: проверьте кавычки, запятые и скобки"

    parts = []
    for error in errors[:5]:
        parts.append(f"{_location(error)}: {_russian(error)}")
    return "Проверьте входные данные — " + "; ".join(parts)


def format_error_item(error: dict) -> str:
    """Translate one Pydantic error into a short Russian message."""
    return _russian(error)


def _extract_object(raw: str) -> str:
    text = raw.strip().lstrip("\ufeff")
    if text.startswith("```"):
        text = _FENCE_START.sub("", text)
        text = _FENCE_END.sub("", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise OutputValidationError("В ответе модели не найден JSON-объект карточки")
    return text[start : end + 1]


def _location(error: dict) -> str:
    location = ".".join(str(part) for part in error.get("loc", ())) or "ответ"
    return location


def _russian(error: dict) -> str:
    """Translate the most common Pydantic messages used by this project."""
    message = str(error.get("msg", "некорректное значение"))
    translations = {
        "Field required": "поле обязательно",
        "String should have at least 1 character": "значение не может быть пустым",
        "Input should be a valid string": "ожидается строка",
        "Input should be a valid dictionary": "ожидается объект",
        "Input should be a valid list": "ожидается список",
        "String should have at most 200 characters": "слишком длинное значение",
        "Input should be a valid dictionary or object to extract fields from": (
            "ожидается JSON-объект товара"
        ),
        "JSON decode error": "некорректный JSON",
    }
    if error.get("type") == "json_invalid":
        return "некорректный JSON"
    return translations.get(message, message)
