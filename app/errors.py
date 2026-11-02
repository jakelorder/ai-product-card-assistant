"""Domain errors shared by the API, the pipeline and LLM clients."""

from __future__ import annotations

from app.models import MissingField


class AppError(Exception):
    """Base class for expected, user-facing errors."""

    status_code = 500
    message = "Внутренняя ошибка сервиса"

    def __init__(self, message: str | None = None) -> None:
        if message:
            self.message = message
        super().__init__(self.message)


class InsufficientDataError(AppError):
    """The input does not contain enough facts to build a card."""

    status_code = 422

    def __init__(self, missing: list[MissingField]) -> None:
        self.missing = missing
        super().__init__(
            "Недостаточно данных для формирования карточки: заполните "
            "характеристики или добавьте описание."
        )


class LLMError(AppError):
    """The LLM provider failed or returned an unusable response."""

    status_code = 502

    def __init__(self, message: str, *, provider: str = "", status: int | None = None) -> None:
        self.provider = provider
        self.status = status
        super().__init__(message)


class OutputValidationError(LLMError):
    """The model response does not match the expected JSON schema."""

    def __init__(self, message: str, *, provider: str = "") -> None:
        super().__init__(message, provider=provider)
