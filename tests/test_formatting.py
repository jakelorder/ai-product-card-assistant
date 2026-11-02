"""Markdown output for the content manager."""

from __future__ import annotations

from app.formatting import render_markdown
from app.models import MISSING_MARK, MissingField, ProductIn, SpecItem


def test_markdown_contains_all_sections(make_card, product: ProductIn) -> None:
    markdown = render_markdown(make_card(), product, [])

    for heading in (
        "## SEO-заголовок",
        "## SEO-описание",
        "## Ключевые характеристики",
        "## Поисковые запросы",
        "## Кому подходит",
        "## Недостающие данные",
    ):
        assert heading in markdown

    assert product.name in markdown
    assert product.manufacturer in markdown
    assert "Длина:" in markdown


def test_markdown_marks_missing_values_and_fields(make_card, product: ProductIn) -> None:
    card = make_card(key_specs=[SpecItem(name="Питание", value=MISSING_MARK, source_field=None)])
    missing = [MissingField(field="specs.питание", reason="значение не заполнено")]

    markdown = render_markdown(card, product, missing)

    assert "| Питание | **данных нет** |" in markdown
    assert "`specs.питание` — значение не заполнено" in markdown


def test_markdown_states_when_nothing_is_missing(make_card, product: ProductIn) -> None:
    markdown = render_markdown(make_card(), product, [])
    assert "Все переданные поля заполнены." in markdown


def test_markdown_escapes_table_pipes(make_card, product: ProductIn) -> None:
    card = make_card(key_specs=[SpecItem(name="A|B", value="C|D", source_field=None)])

    markdown = render_markdown(card, product, [])

    assert "| A\\|B | C\\|D |" in markdown


def test_markdown_renders_notes(make_card, product: ProductIn) -> None:
    card = make_card(notes=["Уточнить комплектацию"])
    markdown = render_markdown(card, product, [])

    assert "## Замечания" in markdown
    assert "Уточнить комплектацию" in markdown


def test_markdown_without_specs_or_queries(make_card, product: ProductIn) -> None:
    card = make_card(key_specs=[], search_queries=[])
    markdown = render_markdown(card, product, [])

    assert f"| Характеристики | {MISSING_MARK} |" in markdown
    assert f"- {MISSING_MARK}" in markdown
