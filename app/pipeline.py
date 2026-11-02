"""Generation pipeline: prompt → LLM → parse → repair → verify → render."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from app.config import Settings
from app.errors import InsufficientDataError, OutputValidationError
from app.formatting import render_markdown
from app.llm.base import LLMClient, LLMRequest
from app.models import CardResult, GenerationMeta, ProductCard, ProductIn
from app.parsing import parse_card
from app.prompts import SYSTEM_PROMPT, build_repair_prompt, build_user_prompt
from app.verification import (
    build_missing_data,
    collect_issues,
    enforce_limits,
    verify_specs,
)

logger = logging.getLogger(__name__)

MAX_PREVIOUS_RESPONSE = 8_000


class Pipeline:
    """Orchestrates one card generation with a bounded repair loop."""

    def __init__(self, client: LLMClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self.max_repairs = max(0, settings.max_repairs)

    async def generate(self, product: ProductIn) -> CardResult:
        started = time.perf_counter()
        missing = build_missing_data(product)

        if not product.has_enough_facts():
            raise InsufficientDataError(missing)

        request = LLMRequest(
            system=SYSTEM_PROMPT,
            user=build_user_prompt(product),
            product=product,
        )

        card: ProductCard | None = None
        attempts = 0

        for attempt in range(self.max_repairs + 1):
            raw = await self.client.generate(request)
            attempts += 1

            try:
                card = parse_card(raw)
            except OutputValidationError as exc:
                if attempt < self.max_repairs:
                    logger.info(
                        "invalid model response, repairing (attempt %s): %s",
                        attempt + 1,
                        exc.message,
                    )
                    request = self._repair_request(product, raw, [exc.message])
                    continue
                raise OutputValidationError(
                    f"Модель не вернула корректный JSON после {attempts} попыток: {exc.message}",
                    provider=self.client.provider,
                ) from exc

            issues = collect_issues(card, product)
            if not issues:
                break

            logger.info("model response has %s issue(s) on attempt %s", len(issues), attempt + 1)
            if attempt < self.max_repairs:
                request = self._repair_request(product, raw, issues)
                continue
            break

        assert card is not None  # the loop either returns a card or raises

        card, limit_warnings = enforce_limits(card)
        card, spec_warnings = verify_specs(card, product)

        warnings = _dedupe([*limit_warnings, *spec_warnings])
        for issue in collect_issues(card, product):
            warnings.append(f"Не удалось исправить автоматически: {issue}")
        warnings = _dedupe(warnings)

        markdown = render_markdown(card, product, missing)
        duration_ms = int((time.perf_counter() - started) * 1000)

        logger.info(
            "card generated: provider=%s attempts=%s duration_ms=%s warnings=%s",
            self.client.provider,
            attempts,
            duration_ms,
            len(warnings),
        )

        return CardResult(
            product=product,
            card=card,
            missing_data=missing,
            markdown=markdown,
            meta=GenerationMeta(
                provider=self.client.provider,
                model=self.client.model,
                attempts=attempts,
                duration_ms=duration_ms,
                warnings=warnings,
                generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )

    def _repair_request(self, product: ProductIn, previous: str, issues: list[str]) -> LLMRequest:
        return LLMRequest(
            system=SYSTEM_PROMPT,
            user=build_repair_prompt(
                product,
                previous[:MAX_PREVIOUS_RESPONSE],
                issues,
            ),
            product=product,
            repair=True,
        )


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
