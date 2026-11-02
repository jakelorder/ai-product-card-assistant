"""Grounding and limit checks — the core anti-hallucination guarantees."""

from __future__ import annotations

from app.models import MISSING_MARK, ProductIn, SpecItem
from app.verification import (
    allowed_numbers,
    build_missing_data,
    collect_issues,
    enforce_limits,
    find_invented_facts,
    find_unmatched_specs,
    match_source,
    verify_specs,
)


def test_allowed_numbers_cover_every_input_field(product: ProductIn) -> None:
    allowed = allowed_numbers(product)
    assert {"0", "1.6", "1.5", "1", "2", "100"} <= allowed


def test_match_source_accepts_formatting_differences(product: ProductIn) -> None:
    assert match_source(SpecItem(name="Диапазон", value="0-1,6 МПа"), product) == (
        "диапазон измерений",
        "0–1,6 МПа",
    )


def test_match_source_accepts_number_paraphrase(product: ProductIn) -> None:
    spec = SpecItem(name="Диапазон", value="от 0 до 1,6 МПа")
    assert match_source(spec, product) is not None


def test_match_source_rejects_unknown_value(product: ProductIn) -> None:
    assert match_source(SpecItem(name="Питание", value="220 В"), product) is None


def test_match_source_rejects_wrong_number(product: ProductIn) -> None:
    assert match_source(SpecItem(name="Диапазон", value="0–2,5 МПа"), product) is None


def test_match_source_prefers_declared_source_field(product: ProductIn) -> None:
    spec = SpecItem(name="Класс", value="1,5", source_field="класс точности")
    assert match_source(spec, product) == ("класс точности", "1,5")


def test_verify_specs_replaces_invented_value(make_card, product: ProductIn) -> None:
    card = make_card(
        key_specs=[
            *make_card().key_specs,
            SpecItem(name="Питание", value="220 В", source_field="питание"),
        ]
    )

    verified, warnings = verify_specs(card, product)
    invented = [item for item in verified.key_specs if item.name == "Питание"]

    assert invented[0].value == MISSING_MARK
    assert invented[0].source_field is None
    assert any("отсутствует во входных данных" in warning for warning in warnings)


def test_verify_specs_appends_characteristics_the_model_skipped(
    make_card, product: ProductIn
) -> None:
    card = make_card(key_specs=[SpecItem(name="Класс точности", value="1,5")])

    verified, warnings = verify_specs(card, product)
    values = {item.name: item.value for item in verified.key_specs}

    assert values["диапазон измерений"] == "0–1,6 МПа"
    assert values["диаметр корпуса"] == "100 мм"
    assert any("модель её пропустила" in warning for warning in warnings)


def test_verify_specs_restores_value_marked_as_missing(make_card, product: ProductIn) -> None:
    card = make_card(
        key_specs=[
            SpecItem(name="класс точности", value=MISSING_MARK, source_field="класс точности")
        ]
    )

    verified, warnings = verify_specs(card, product)
    by_name = {item.name: item.value for item in verified.key_specs}

    assert by_name["класс точности"] == "1,5"
    assert any("восстановлено" in warning for warning in warnings)


def test_verify_specs_keeps_missing_marker_for_empty_input(
    make_card, sparse_product: ProductIn
) -> None:
    generated = make_card(
        key_specs=[
            SpecItem(name="материал корпуса", value=MISSING_MARK, source_field="материал корпуса")
        ]
    )

    verified, _ = verify_specs(generated, sparse_product)
    by_name = {item.name: item.value for item in verified.key_specs}

    assert by_name["материал корпуса"] == MISSING_MARK


def test_find_invented_facts_flags_numbers_in_text(make_card, product: ProductIn) -> None:
    card = make_card(seo_description=make_card().seo_description + " Гарантия 5 лет.")

    issues = find_invented_facts(card, product)

    assert any("SEO-описание" in issue and "5" in issue for issue in issues)


def test_find_invented_facts_ignores_numbers_from_input(make_card, product: ProductIn) -> None:
    assert find_invented_facts(make_card(), product) == []


def test_find_unmatched_specs_flags_omitted_input(make_card, product: ProductIn) -> None:
    card = make_card(key_specs=[SpecItem(name="Класс точности", value="1,5")])

    issues = find_unmatched_specs(card, product)

    assert any("диапазон измерений" in issue for issue in issues)


def test_collect_issues_reports_length_and_query_violations(make_card, product: ProductIn) -> None:
    card = make_card(
        seo_title="З" * 90,
        seo_description="Коротко.",
        search_queries=["один"],
        commercial_block="Мало.",
    )

    issues = " ".join(collect_issues(card, product))

    assert "seo_title" in issues
    assert "seo_description" in issues
    assert "commercial_block" in issues
    assert "search_queries" in issues


def test_collect_issues_is_empty_for_valid_card(make_card, product: ProductIn) -> None:
    assert collect_issues(make_card(), product) == []


def test_enforce_limits_truncates_title_and_description(make_card, product: ProductIn) -> None:
    card = make_card(
        seo_title="З" * 90,
        seo_description="Прибор предназначен для контроля параметров. " * 60,
    )

    limited, warnings = enforce_limits(card)

    assert len(limited.seo_title) <= 70
    assert 800 <= len(limited.seo_description) <= 1200
    assert any("заголовок сокращён" in warning.lower() for warning in warnings)
    assert any("описание сокращено" in warning.lower() for warning in warnings)


def test_enforce_limits_warns_about_short_description(make_card) -> None:
    limited, warnings = enforce_limits(make_card(seo_description="Коротко."))

    assert limited.seo_description == "Коротко."
    assert any("короче 800" in warning for warning in warnings)


def test_enforce_limits_cuts_extra_queries(make_card) -> None:
    limited, warnings = enforce_limits(
        make_card(search_queries=[f"запрос {index}" for index in range(8)])
    )

    assert len(limited.search_queries) == 5
    assert any("сокращён до 5" in warning for warning in warnings)


def test_enforce_limits_drops_blank_notes(make_card) -> None:
    limited, _ = enforce_limits(make_card(notes=["  ", "Уточнить питание"]))
    assert limited.notes == ["Уточнить питание"]


def test_build_missing_data_matches_model(sparse_product: ProductIn) -> None:
    assert build_missing_data(sparse_product) == sparse_product.missing_fields()
