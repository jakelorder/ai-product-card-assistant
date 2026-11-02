"""Prompt content: guardrails must be present and facts must be passed as is."""

from __future__ import annotations

import json

from app.models import ProductIn
from app.prompts import SYSTEM_PROMPT, build_repair_prompt, build_user_prompt


def test_system_prompt_contains_all_guardrails() -> None:
    for token in (
        "данных нет",
        "до 70",
        "800",
        "1200",
        "3",
        "5",
        "ТОЛЬКО валидный JSON",
        "source_field",
        "не добавляй",
    ):
        assert token in SYSTEM_PROMPT


def test_system_prompt_forbids_inventing_facts() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "выдумыв" in lowered
    assert "критическая ошибка" in lowered


def test_user_prompt_contains_exact_input_values(product: ProductIn) -> None:
    prompt = build_user_prompt(product)

    for key, value in product.specs.items():
        assert key in prompt
        assert value in prompt
    assert product.name in prompt
    assert product.manufacturer in prompt
    assert '"specs"' in prompt


def test_user_prompt_lists_missing_fields(sparse_product: ProductIn) -> None:
    prompt = build_user_prompt(sparse_product)

    assert "specs.материал корпуса" in prompt
    assert "description" in prompt
    assert "данных нет" in prompt


def test_user_prompt_serializes_input_without_inventing_specs() -> None:
    product = ProductIn(name="Прибор", manufacturer="Завод", description="Описание")

    prompt = build_user_prompt(product)
    payload = prompt.split("```json")[1].split("```")[0]
    data = json.loads(payload)

    assert data == {
        "name": "Прибор",
        "manufacturer": "Завод",
        "specs": {},
        "description": "Описание",
    }


def test_repair_prompt_contains_issues_and_previous_answer() -> None:
    product = ProductIn(
        name="Прибор",
        manufacturer="Завод",
        specs={"диапазон": "0–1,6 МПа"},
    )

    prompt = build_repair_prompt(
        product,
        previous_response='{"seo_title": "' + "З" * 90 + '"}',
        issues=["seo_title: 90 символов, требуется не более 70"],
    )

    assert "seo_title: 90 символов" in prompt
    assert "0–1,6 МПа" in prompt
    assert "ПРЕДЫДУЩИЙ ОТВЕТ" in prompt
    assert "не добавляй новых фактов" in prompt
