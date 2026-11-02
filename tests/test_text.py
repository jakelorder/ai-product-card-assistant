"""Text helpers: normalization, numeric comparison and limit enforcement."""

from __future__ import annotations

from app.text import (
    clamp_description,
    normalize,
    number_tokens,
    truncate_title,
)


def test_normalize_unifies_formatting() -> None:
    assert normalize("0–1,6 МПа") == normalize("0-1,6 мпа")
    assert normalize("Класс  Ёмкости\u00a0") == "класс емкости"
    assert normalize("  G1/2.  ") == "g1/2"


def test_number_tokens_compare_decimals_and_zeros() -> None:
    assert number_tokens("1,50") == number_tokens("1.5")
    assert number_tokens("007") == {"7"}
    assert number_tokens("M20×1,5") == {"20", "1.5"}
    assert number_tokens("без чисел") == set()


def test_truncate_title_keeps_limit_and_word_boundary() -> None:
    title = (
        "Очень длинный SEO-заголовок для карточки товара с расширенными характеристиками прибора"
    )
    result, changed = truncate_title(title)

    assert changed
    assert len(result) <= 70
    assert not result.endswith((" ", ",", ".", "—"))
    assert title.startswith(result)


def test_truncate_title_does_not_touch_short_text() -> None:
    result, changed = truncate_title("Короткий заголовок")
    assert result == "Короткий заголовок"
    assert not changed


def test_clamp_description_cuts_long_text_within_limits() -> None:
    text = "Прибор предназначен для контроля параметров. " * 60
    result, changed = clamp_description(text)

    assert changed
    assert 800 <= len(result) <= 1200
    assert result.endswith(".")


def test_clamp_description_keeps_short_text() -> None:
    result, changed = clamp_description("Слишком коротко.")
    assert result == "Слишком коротко."
    assert not changed


def test_clamp_description_has_hard_limit_even_without_boundaries() -> None:
    text = "x" * 2000
    result, changed = clamp_description(text)

    assert changed
    assert len(result) <= 1200
