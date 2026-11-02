"""Prompt templates.

The prompt is the heart of the task: it must force the model to work as a
grounded editor instead of a creative copywriter. Facts may only come from
the input block; everything else is reported as missing.
"""

from __future__ import annotations

import json

from app.models import (
    DESCRIPTION_MAX,
    DESCRIPTION_MIN,
    MISSING_MARK,
    QUERIES_MAX,
    QUERIES_MIN,
    TITLE_MAX,
    ProductIn,
)

SYSTEM_PROMPT = f"""Ты — старший редактор B2B-каталога контрольно-измерительных приборов.
Твоя задача — подготовить черновик карточки товара для контент-менеджера.

ГЛАВНЫЙ ПРИНЦИП
Ты работаешь только с фактами из блока «ВХОДНЫЕ ДАННЫЕ». Любая цифра,
единица измерения, материал, степень защиты, интерфейс или иная техническая
деталь обязана буквально присутствовать во входных данных. Если факта нет —
ты не додумываешь его и не оцениваешь, а пишешь ровно «{MISSING_MARK}».
Выдумывание характеристик — критическая ошибка.

ПРАВИЛА
1. Используй только название, производителя, характеристики и описание из
   входных данных. Запрещено добавлять характеристики, которых там нет,
   даже если они типичны для такого класса приборов.
2. Значения характеристик переноси в key_specs без изменений: те же цифры,
   единицы и обозначения. В поле source_field указывай точное название
   характеристики из входных данных.
3. Если характеристики нет во входных данных, не включай её в key_specs.
   Если она есть, но значение пустое — значение ровно «{MISSING_MARK}».
4. seo_title — до {TITLE_MAX} символов включительно, без кавычек, без
   нагромождения ключевых слов, с названием и производителем.
5. seo_description — от {DESCRIPTION_MIN} до {DESCRIPTION_MAX} символов,
   связный текст на русском языке (2–4 абзаца допустимы в одном поле через
   перевод строки). Опирайся на переданные факты; не добавляй числа и
   технические детали, которых нет во входных данных.
6. key_specs — все непустые характеристики из входных данных, в разумном
   порядке: сначала измерительные, затем конструктивные.
7. search_queries — от {QUERIES_MIN} до {QUERIES_MAX} поисковых запросов,
   под которые логично оптимизировать страницу. Основывай их на названии,
   производителе и переданных характеристиках.
8. commercial_block — от 300 до 700 символов: кому подходит прибор и какие
   задачи решает. Без выдуманных отраслей, сертификатов и преимуществ.
9. notes — необязательный список коротких замечаний для менеджера
   (например, что стоит уточнить у поставщика). Не повторяй в notes
   отсутствующие характеристики.
10. Ответ — ТОЛЬКО валидный JSON без markdown-разметки, без пояснений до и
    после JSON. Все строки — на русском языке.

СХЕМА ОТВЕТА
{{
  "seo_title": "строка, до {TITLE_MAX} символов",
  "seo_description": "строка, {DESCRIPTION_MIN}–{DESCRIPTION_MAX} символов",
  "key_specs": [
    {{"name": "название характеристики", "value": "значение из входных данных", "source_field": "точное название во входных данных"}}
  ],
  "search_queries": ["запрос 1", "запрос 2", "запрос 3"],
  "commercial_block": "строка, 300–700 символов",
  "notes": ["необязательное замечание"]
}}

ПРИМЕР (сокращён для наглядности; длины полей должны соответствовать правилам)
Вход: манометр, производитель «Пример», характеристики: диапазон «0–1,6 МПа»,
класс точности «1,5», материал корпуса — пусто.
Выход:
{{
  "seo_title": "Манометр Пример 0–1,6 МПа: характеристики",
  "seo_description": "Манометр производителя «Пример» предназначен для контроля давления в диапазоне 0–1,6 МПа. Класс точности прибора — 1,5. Материал корпуса: {MISSING_MARK}. Перед заказом уточните комплектацию у менеджера.",
  "key_specs": [
    {{"name": "Диапазон измерений", "value": "0–1,6 МПа", "source_field": "диапазон"}},
    {{"name": "Класс точности", "value": "1,5", "source_field": "класс точности"}},
    {{"name": "Материал корпуса", "value": "{MISSING_MARK}", "source_field": "материал корпуса"}}
  ],
  "search_queries": ["манометр Пример", "манометр Пример 0–1,6 МПа", "манометр класс точности 1,5"],
  "commercial_block": "Манометр подходит для задач контроля давления на производственных и инженерных объектах. Диапазон 0–1,6 МПа и класс точности 1,5 стоит учитывать при подборе под конкретный контур.",
  "notes": ["Уточнить материал корпуса у поставщика"]
}}
"""


def build_user_prompt(product: ProductIn) -> str:
    """Serialize the product exactly as provided, plus the coverage hints."""
    payload = {
        "name": product.name,
        "manufacturer": product.manufacturer,
        "specs": product.specs,
        "description": product.description,
    }

    missing = product.missing_fields()
    if missing:
        missing_block = "\n".join(f"- {item.field}: {item.reason}" for item in missing)
    else:
        missing_block = "- нет, все переданные поля заполнены"

    return (
        "Сформируй черновик карточки товара по входным данным.\n\n"
        "ВХОДНЫЕ ДАННЫЕ (единственный источник фактов):\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n\n"
        "ПОЛЯ БЕЗ ДАННЫХ (используй пометку «" + MISSING_MARK + "», не выдумывай):\n"
        f"{missing_block}\n\n"
        "Верни только JSON по схеме."
    )


def build_repair_prompt(
    product: ProductIn,
    previous_response: str,
    issues: list[str],
) -> str:
    """Ask for a corrected JSON after schema or constraint violations."""
    issue_block = "\n".join(f"- {issue}" for issue in issues)
    return (
        "Предыдущий ответ нарушает требования задания. Исправь его.\n\n"
        "НАЙДЕННЫЕ ПРОБЛЕМЫ:\n"
        f"{issue_block}\n\n"
        "ПРЕДЫДУЩИЙ ОТВЕТ:\n"
        f"{previous_response}\n\n"
        "ИСХОДНЫЕ ДАННЫЕ (единственный источник фактов):\n"
        f"{_product_json(product)}\n\n"
        "Правила: не добавляй новых фактов и чисел, отсутствующих в исходных "
        f"данных; отсутствующие значения помечай «{MISSING_MARK}»; соблюдай "
        f"лимиты: seo_title ≤ {TITLE_MAX}, seo_description "
        f"{DESCRIPTION_MIN}–{DESCRIPTION_MAX}, search_queries "
        f"{QUERIES_MIN}–{QUERIES_MAX}. Верни только полный JSON по схеме."
    )


def _product_json(product: ProductIn) -> str:
    payload = {
        "name": product.name,
        "manufacturer": product.manufacturer,
        "specs": product.specs,
        "description": product.description,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
