"""Demo input data bundled with the repository.

The original two products were not attached to the assignment, so the files
in ``data/`` are demonstration inputs. They are replaced by real products
without any code changes.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "products.json"

FALLBACK_PAYLOAD = {
    "name": "Манометр показывающий DM 02",
    "manufacturer": "Манотомь",
    "specs": {
        "диапазон измерений": "0–1,6 МПа",
        "класс точности": "1,5",
        "присоединение": "G1/2",
        "диаметр корпуса": "100 мм",
    },
    "description": (
        "Прибор предназначен для измерения избыточного давления "
        "неагрессивных к материалам корпуса сред."
    ),
}


@lru_cache(maxsize=1)
def load_example_products() -> list[dict[str, Any]]:
    try:
        data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [FALLBACK_PAYLOAD]

    if isinstance(data, dict):
        data = data.get("products", [])
    if not isinstance(data, list):
        return [FALLBACK_PAYLOAD]

    products = [item for item in data if isinstance(item, dict)]
    return products or [FALLBACK_PAYLOAD]


def example_options() -> list[dict[str, str]]:
    """Payloads ready for the web form."""
    return [
        {
            "name": str(product.get("name") or f"Пример {index}"),
            "payload": json.dumps(product, ensure_ascii=False, indent=2),
        }
        for index, product in enumerate(load_example_products(), 1)
    ]
