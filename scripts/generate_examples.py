"""Generate example cards for the bundled demo products.

Usage::

    python -m scripts.generate_examples

The provider is taken from ``.env``. By default the deterministic ``mock``
provider is used, so the script works without an API key.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.config import Settings
from app.llm import build_client
from app.models import ProductIn
from app.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "products.json"
OUTPUT_DIR = ROOT / "examples"


async def main() -> None:
    settings = Settings.from_env()
    if settings.provider not in ("mock", "ollama") and not settings.api_key:
        raise SystemExit(
            f"Для провайдера {settings.provider} не задан LLM_API_KEY. "
            "Заполните .env или запустите с LLM_PROVIDER=mock."
        )

    products = [
        ProductIn.model_validate(item) for item in json.loads(DATA_FILE.read_text(encoding="utf-8"))
    ]
    pipeline = Pipeline(build_client(settings), settings)
    OUTPUT_DIR.mkdir(exist_ok=True)

    for index, product in enumerate(products, 1):
        result = await pipeline.generate(product)
        markdown_path = OUTPUT_DIR / f"product-{index}.md"
        json_path = OUTPUT_DIR / f"product-{index}.json"
        markdown_path.write_text(result.markdown, encoding="utf-8")
        json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(
            f"product-{index}: {product.name} | provider={result.meta.provider} | "
            f"description={len(result.card.seo_description)} символов | "
            f"warnings={len(result.meta.warnings)}"
        )

    print(f"Готово: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
