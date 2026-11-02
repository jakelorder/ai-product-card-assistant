"""Input model: normalization, validation and coverage report."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models import MISSING_MARK, ProductIn


def test_normalizes_scalars_and_description() -> None:
    product = ProductIn(
        name="  Манометр  ",
        manufacturer=" Манотомь ",
        specs={"давление": 1.5, "наличие": True, "пусто": None, "  ": "мусор"},
        description="   ",
    )

    assert product.name == "Манометр"
    assert product.manufacturer == "Манотомь"
    assert product.specs == {"давление": "1,5", "наличие": "да", "пусто": ""}
    assert product.description is None


def test_extra_fields_are_ignored() -> None:
    product = ProductIn.model_validate(
        {
            "name": "Прибор",
            "manufacturer": "Завод",
            "unknown": "ignore me",
            "specs": {"масса": "1 кг"},
        }
    )
    assert not hasattr(product, "unknown")


def test_blank_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProductIn(name="   ", manufacturer="Завод")


def test_specs_must_be_an_object() -> None:
    with pytest.raises(ValidationError):
        ProductIn.model_validate({"name": "Прибор", "manufacturer": "Завод", "specs": ["a", "b"]})


def test_nested_spec_value_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProductIn.model_validate(
            {
                "name": "Прибор",
                "manufacturer": "Завод",
                "specs": {"диапазон": {"min": 0, "max": 1}},
            }
        )


def test_missing_fields_reports_empty_specs_and_description(sparse_product: ProductIn) -> None:
    fields = {item.field for item in sparse_product.missing_fields()}

    assert "specs.материал корпуса" in fields
    assert "description" in fields


def test_missing_fields_reports_absent_specs_block() -> None:
    product = ProductIn(name="Прибор", manufacturer="Завод", specs={})
    fields = {item.field for item in product.missing_fields()}

    assert "specs" in fields
    assert "description" in fields


def test_fully_filled_product_has_no_missing_fields(product: ProductIn) -> None:
    assert product.missing_fields() == []


def test_has_enough_facts() -> None:
    assert not ProductIn(name="Прибор", manufacturer="Завод").has_enough_facts()
    assert ProductIn(
        name="Прибор", manufacturer="Завод", specs={"масса": "1 кг"}
    ).has_enough_facts()
    assert ProductIn(name="Прибор", manufacturer="Завод", description="Описание").has_enough_facts()


def test_missing_marker_constant() -> None:
    assert MISSING_MARK == "данных нет"
