"""Markdown rendering for the copy-paste output."""

from __future__ import annotations

from app.models import MISSING_MARK, MissingField, ProductCard, ProductIn


def render_markdown(
    card: ProductCard,
    product: ProductIn,
    missing: list[MissingField],
) -> str:
    """Build a clean Markdown draft a content manager can paste into a CMS."""
    lines: list[str] = [
        f"# {card.seo_title}",
        "",
        f"**Товар:** {product.name}  ",
        f"**Производитель:** {product.manufacturer}",
        "",
        "## SEO-заголовок",
        "",
        f"{card.seo_title}  ",
        f"*Длина: {len(card.seo_title)} симв. (лимит 70)*",
        "",
        "## SEO-описание",
        "",
        card.seo_description,
        "",
        f"*Длина: {len(card.seo_description)} симв. (норма 800–1200)*",
        "",
        "## Ключевые характеристики",
        "",
        "| Характеристика | Значение |",
        "| --- | --- |",
    ]

    if card.key_specs:
        for spec in card.key_specs:
            name = _table_cell(spec.name)
            value = _table_cell(spec.value)
            if spec.value == MISSING_MARK:
                value = f"**{value}**"
            lines.append(f"| {name} | {value} |")
    else:
        lines.append(f"| Характеристики | {MISSING_MARK} |")

    lines.extend(["", "## Поисковые запросы", ""])
    if card.search_queries:
        lines.extend(f"{index}. {query}" for index, query in enumerate(card.search_queries, 1))
    else:
        lines.append(f"- {MISSING_MARK}")

    lines.extend(
        [
            "",
            "## Кому подходит",
            "",
            card.commercial_block,
            "",
            "## Недостающие данные",
            "",
        ]
    )
    if missing:
        lines.extend(f"- `{item.field}` — {item.reason}" for item in missing)
    else:
        lines.append("- Все переданные поля заполнены.")

    if card.notes:
        lines.extend(["", "## Замечания", ""])
        lines.extend(f"- {note}" for note in card.notes)

    lines.extend(
        [
            "",
            "---",
            "",
            (
                "*Черновик подготовлен автоматически на основе переданных данных. "
                "Перед публикацией проверьте карточку у ответственного менеджера.*"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _table_cell(value: str) -> str:
    return " ".join(value.split()).replace("|", "\\|")
