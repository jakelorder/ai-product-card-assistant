"""Offline deterministic provider.

Used by tests and by the default ``LLM_PROVIDER=mock`` mode, so the project
can be started and demonstrated without an API key. It never invents facts:
every technical value is copied from the input.
"""

from __future__ import annotations

import json

from app.config import Settings
from app.llm.base import LLMClient, LLMRequest
from app.models import (
    COMMERCIAL_MAX,
    COMMERCIAL_MIN,
    MISSING_MARK,
    ProductIn,
)
from app.text import clamp_description, truncate_title


class MockClient(LLMClient):
    provider = "mock"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings.model)

    async def generate(self, request: LLMRequest) -> str:
        return json.dumps(build_mock_card(request.product), ensure_ascii=False)


def build_mock_card(product: ProductIn) -> dict:
    specs = [
        {
            "name": key[:1].upper() + key[1:],
            "value": value or MISSING_MARK,
            "source_field": key,
        }
        for key, value in product.specs.items()
    ]

    return {
        "seo_title": _title(product),
        "seo_description": _description(product),
        "key_specs": specs,
        "search_queries": _queries(product),
        "commercial_block": _commercial_block(product),
        "notes": [],
    }


def _title(product: ProductIn) -> str:
    title, _ = truncate_title(f"{product.name} — характеристики и применение")
    return title


def _description(product: ProductIn) -> str:
    filled = [(key, value) for key, value in product.specs.items() if value]
    if filled:
        specs_sentence = (
            "Подтверждённые характеристики: "
            + "; ".join(f"{key} — {value}" for key, value in filled)
            + "."
        )
    else:
        specs_sentence = (
            "Технические характеристики во входных данных не переданы, поэтому "
            f"в карточке указано «{MISSING_MARK}»."
        )

    sentences = [
        (
            f"{product.name} производителя «{product.manufacturer}» — "
            "контрольно-измерительный прибор для задач промышленного контроля и учёта."
        ),
        specs_sentence,
        (
            "Прибор применяется на производственных площадках, в инженерных системах "
            "и лабораториях, где требуется контроль измеряемых параметров."
        ),
        (
            "При подборе модели учитывайте диапазон измерений, класс точности и "
            "присоединительные размеры, указанные в характеристиках."
        ),
        (
            "В карточке приведены только те сведения, которые переданы во входных "
            f"данных; отсутствующие параметры отмечены как «{MISSING_MARK}»."
        ),
        (
            "Перед заказом рекомендуется уточнить у поставщика комплектацию, условия "
            "эксплуатации и требования к поверке."
        ),
        (
            "Описание ориентировано на специалистов, которые подбирают оборудование "
            "для задач контроля и измерений."
        ),
        "Дополнительные параметры, не указанные во входных данных, в карточке не приводятся.",
        "Для уточнения цены, сроков поставки и исполнения прибора обратитесь к менеджеру.",
        (
            "Прибор стоит рассматривать как часть измерительного контура, требования "
            "к которому определяются условиями объекта."
        ),
    ]

    text = " ".join(sentences[:4])
    for sentence in sentences[4:]:
        if len(text) >= 900:
            break
        text = f"{text} {sentence}"

    text, _ = clamp_description(text)
    return text


def _queries(product: ProductIn) -> list[str]:
    base = product.name
    candidates = [
        base,
        f"{base} купить",
        f"{base} характеристики",
        f"{base} цена",
        f"{base} {product.manufacturer}",
    ]

    queries: list[str] = []
    for query in candidates:
        query = " ".join(query.split())
        if query not in queries:
            queries.append(query)
    return queries[:5]


def _commercial_block(product: ProductIn) -> str:
    text = (
        f"Прибор «{product.name}» от производителя «{product.manufacturer}» "
        "подходит для задач контроля и измерений на производственных и "
        "инженерных объектах. В карточке собраны только подтверждённые "
        "характеристики, поэтому менеджер может опираться на них при подборе "
        "оборудования под конкретный объект. Недостающие параметры стоит "
        "уточнить у поставщика перед оформлением заказа: это снижает риск "
        "ошибки при выборе исполнения и комплектации."
    )
    text, _ = clamp_description(text, limit=COMMERCIAL_MAX, minimum=COMMERCIAL_MIN)
    return text
