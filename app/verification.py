"""Post-generation checks: grounding, limits and coverage.

This module is the safety net that makes the service trustworthy. Prompting
alone cannot guarantee that a model will not invent a value, so every fact
in the answer is compared with the input, and every length limit is enforced
deterministically.
"""

from __future__ import annotations

from app.models import (
    COMMERCIAL_MAX,
    COMMERCIAL_MIN,
    DESCRIPTION_MAX,
    DESCRIPTION_MIN,
    MISSING_MARK,
    QUERIES_MAX,
    QUERIES_MIN,
    TITLE_MAX,
    MissingField,
    ProductCard,
    ProductIn,
    SpecItem,
)
from app.text import (
    clamp_description,
    normalize,
    number_tokens,
    truncate_title,
)

TEXT_FIELDS = (
    ("seo_title", "SEO-заголовок"),
    ("seo_description", "SEO-описание"),
    ("commercial_block", "Commercial-блок"),
)


def allowed_numbers(product: ProductIn) -> set[str]:
    """All numeric tokens a grounded answer is allowed to mention."""
    parts = [
        product.name,
        product.manufacturer,
        product.description or "",
        *product.specs.values(),
    ]
    return number_tokens(" ".join(parts))


def invented_numbers(text: str, allowed: set[str]) -> list[str]:
    """Numbers in the text that do not occur anywhere in the input."""
    return sorted(number_tokens(text) - allowed, key=_number_sort_key)


def find_invented_facts(card: ProductCard, product: ProductIn) -> list[str]:
    """Human-readable list of unsupported numbers in free-text fields."""
    allowed = allowed_numbers(product)
    issues: list[str] = []
    for attribute, label in TEXT_FIELDS:
        found = invented_numbers(getattr(card, attribute), allowed)
        if found:
            issues.append(
                f"{label}: числа {', '.join(found)} отсутствуют во входных "
                "данных — уберите их или замените проверенными значениями"
            )
    return issues


def find_unmatched_specs(card: ProductCard, product: ProductIn) -> list[str]:
    """Specs whose values cannot be traced back to the input."""
    issues: list[str] = []
    for spec in card.key_specs:
        if _is_missing(spec.value):
            restored = _restore_missing(spec, product)
            if restored is not None:
                issues.append(
                    f"key_specs: для характеристики «{restored[0]}» есть "
                    f"значение «{restored[1]}», но указано «{MISSING_MARK}»"
                )
            continue
        if match_source(spec, product) is None:
            issues.append(
                f"key_specs: значение «{spec.value}» для характеристики "
                f"«{spec.name}» не найдено во входных данных"
            )

    present = {
        normalize(match[0])
        for spec in card.key_specs
        if (match := match_source(spec, product)) is not None
    }
    for key, value in product.specs.items():
        if value and normalize(key) not in present:
            issues.append(f"key_specs: не включена характеристика «{key}» из входных данных")
    return issues


def collect_issues(card: ProductCard, product: ProductIn) -> list[str]:
    """Everything that should be sent back to the model for repair."""
    issues: list[str] = []

    title_length = len(card.seo_title)
    if title_length > TITLE_MAX:
        issues.append(f"seo_title: {title_length} символов, требуется не более {TITLE_MAX}")

    description_length = len(card.seo_description)
    if not DESCRIPTION_MIN <= description_length <= DESCRIPTION_MAX:
        issues.append(
            f"seo_description: {description_length} символов, требуется "
            f"{DESCRIPTION_MIN}–{DESCRIPTION_MAX}"
        )

    commercial_length = len(card.commercial_block)
    if not COMMERCIAL_MIN <= commercial_length <= COMMERCIAL_MAX:
        issues.append(
            f"commercial_block: {commercial_length} символов, требуется "
            f"{COMMERCIAL_MIN}–{COMMERCIAL_MAX}"
        )

    query_count = len(card.search_queries)
    if not QUERIES_MIN <= query_count <= QUERIES_MAX:
        issues.append(
            f"search_queries: {query_count} запросов, требуется {QUERIES_MIN}–{QUERIES_MAX}"
        )

    if not card.key_specs:
        issues.append("key_specs: список характеристик пуст")

    issues.extend(find_invented_facts(card, product))
    issues.extend(find_unmatched_specs(card, product))
    return issues


def match_source(spec: SpecItem, product: ProductIn) -> tuple[str, str] | None:
    """Find the input characteristic a generated value came from.

    A value is accepted when it is the same fact up to formatting (case,
    dashes, spaces, decimal separators) or when both contain exactly the
    same numbers (``от -50 до +150 °C`` vs ``-50…+150 °C``).
    """
    value = spec.value.strip()
    if not value or _is_missing(value):
        return None

    candidates = [(key, item) for key, item in product.specs.items() if item.strip()]
    if spec.source_field:
        target = normalize(spec.source_field)
        candidates.sort(key=lambda pair: normalize(pair[0]) != target)

    out_normalized = normalize(value)
    out_numbers = number_tokens(value)

    for key, source_value in candidates:
        source_normalized = normalize(source_value)
        if (
            out_normalized == source_normalized
            or out_normalized in source_normalized
            or source_normalized in out_normalized
        ):
            return key, source_value
        if out_numbers and out_numbers == number_tokens(source_value):
            return key, source_value
    return None


def verify_specs(card: ProductCard, product: ProductIn) -> tuple[ProductCard, list[str]]:
    """Replace unsupported values and add characteristics the model skipped."""
    warnings: list[str] = []
    verified: list[SpecItem] = []
    matched_keys: set[str] = set()

    for spec in card.key_specs:
        if _is_missing(spec.value):
            restored = _restore_missing(spec, product)
            if restored is not None:
                key, value = restored
                matched_keys.add(normalize(key))
                warnings.append(
                    f"Характеристика «{spec.name}»: модель указала "
                    f"«{MISSING_MARK}», но значение «{value}» есть во входных "
                    "данных — восстановлено"
                )
                verified.append(SpecItem(name=spec.name, value=value, source_field=key))
            else:
                verified.append(spec)
                if spec.source_field:
                    matched_keys.add(normalize(spec.source_field))
            continue

        match = match_source(spec, product)
        if match is None:
            warnings.append(
                f"Характеристика «{spec.name}»: значение «{spec.value}» "
                f"отсутствует во входных данных и заменено на «{MISSING_MARK}»"
            )
            verified.append(SpecItem(name=spec.name, value=MISSING_MARK, source_field=None))
            continue

        key, _ = match
        matched_keys.add(normalize(key))
        verified.append(SpecItem(name=spec.name, value=spec.value, source_field=key))

    generated_names = {normalize(item.name) for item in verified}
    for key, value in product.specs.items():
        if normalize(key) in matched_keys or normalize(key) in generated_names:
            continue
        if value:
            warnings.append(
                f"Характеристика «{key}» добавлена из входных данных: модель её пропустила"
            )
        verified.append(
            SpecItem(
                name=key,
                value=value or MISSING_MARK,
                source_field=key,
            )
        )

    return card.model_copy(update={"key_specs": verified}), warnings


def enforce_limits(card: ProductCard) -> tuple[ProductCard, list[str]]:
    """Apply hard limits after the model's last chance to fix them."""
    warnings: list[str] = []

    title, changed = truncate_title(card.seo_title)
    if changed:
        warnings.append(f"SEO-заголовок сокращён до {len(title)} символов (лимит {TITLE_MAX})")

    description, changed = clamp_description(card.seo_description)
    if changed:
        warnings.append(
            f"SEO-описание сокращено до {len(description)} символов (лимит {DESCRIPTION_MAX})"
        )
    elif len(description) < DESCRIPTION_MIN:
        warnings.append(
            f"SEO-описание короче {DESCRIPTION_MIN} символов "
            f"({len(description)}): текст не дополнялся вручную, чтобы "
            "не выдумывать факты"
        )

    commercial, changed = clamp_description(
        card.commercial_block,
        limit=COMMERCIAL_MAX,
        minimum=COMMERCIAL_MIN,
    )
    if changed:
        warnings.append(
            f"Commercial-блок сокращён до {len(commercial)} символов (лимит {COMMERCIAL_MAX})"
        )

    queries = card.search_queries
    if len(queries) > QUERIES_MAX:
        warnings.append(f"Список поисковых запросов сокращён до {QUERIES_MAX}")
        queries = queries[:QUERIES_MAX]
    if len(queries) < QUERIES_MIN:
        warnings.append(
            f"Поисковых запросов меньше {QUERIES_MIN}: модель предложила {len(queries)}"
        )

    notes = [note.strip() for note in card.notes if note and note.strip()]

    updated = card.model_copy(
        update={
            "seo_title": title,
            "seo_description": description,
            "commercial_block": commercial,
            "search_queries": queries,
            "notes": notes,
        }
    )
    return updated, warnings


def build_missing_data(product: ProductIn) -> list[MissingField]:
    """Coverage report is computed in code, never by the model."""
    return product.missing_fields()


def _is_missing(value: str) -> bool:
    return MISSING_MARK in normalize(value)


def _restore_missing(spec: SpecItem, product: ProductIn) -> tuple[str, str] | None:
    """Find an input value for a characteristic the model marked as missing."""
    names = {normalize(spec.name)}
    if spec.source_field:
        names.add(normalize(spec.source_field))
    for key, value in product.specs.items():
        if value and normalize(key) in names:
            return key, value
    return None


def _number_sort_key(token: str) -> tuple[int, float | str]:
    try:
        return (0, float(token))
    except ValueError:
        return (1, token)
