"""FastAPI application: web form and JSON API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Form, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from app.config import Settings
from app.dependencies import get_pipeline, get_settings
from app.errors import AppError, InsufficientDataError, LLMError
from app.examples import example_options
from app.models import CardResult, ProductIn
from app.parsing import format_error_item, format_validation_error
from app.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    if settings.provider == "mock":
        logger.warning(
            "LLM_PROVIDER=mock: карточки собираются детерминированно, "
            "без вызова LLM. Задайте реального провайдера в .env."
        )
    elif settings.provider != "ollama" and not settings.api_key:
        logger.warning(
            "Для провайдера %s не задан LLM_API_KEY — запросы завершатся ошибкой 502.",
            settings.provider,
        )
    yield


app = FastAPI(
    title="AI-помощник для карточек товаров",
    description=(
        "Формирует черновик карточки контрольно-измерительного прибора "
        "по данным из JSON и проверяет, что в ответе нет выдуманных фактов."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    return _render_index(request, settings)


@app.post("/generate", response_class=HTMLResponse)
async def generate_form(
    request: Request,
    payload: str = Form(""),
    pipeline: Pipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    """Form endpoint used by the web UI (no JavaScript required)."""
    if not payload.strip():
        return _render_index(
            request,
            settings,
            error="Вставьте JSON товара.",
            status_code=422,
        )

    try:
        product = ProductIn.model_validate_json(payload)
    except ValidationError as exc:
        return _render_index(
            request,
            settings,
            payload=payload,
            error=format_validation_error(exc),
            status_code=422,
        )

    try:
        result = await pipeline.generate(product)
    except AppError as exc:
        return _render_index(
            request,
            settings,
            payload=payload,
            error=exc.message,
            status_code=exc.status_code,
        )
    except Exception:  # pragma: no cover - defensive branch
        logger.exception("unexpected error while generating a card")
        return _render_index(
            request,
            settings,
            payload=payload,
            error="Внутренняя ошибка сервиса. Попробуйте ещё раз.",
            status_code=500,
        )

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "result": result,
            "provider": settings.provider,
            "model": settings.model,
        },
    )


@app.post("/api/generate", response_model=CardResult)
async def generate_api(
    product: ProductIn,
    pipeline: Pipeline = Depends(get_pipeline),
) -> CardResult:
    """JSON API: returns the card, the coverage report and Markdown."""
    return await pipeline.generate(product)


@app.post("/api/generate/markdown", response_class=PlainTextResponse)
async def generate_markdown(
    product: ProductIn,
    pipeline: Pipeline = Depends(get_pipeline),
) -> PlainTextResponse:
    """Same as ``/api/generate`` but returns ready-to-paste Markdown."""
    result = await pipeline.generate(product)
    return PlainTextResponse(result.markdown, media_type="text/markdown")


@app.get("/health")
async def health(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "status": "ok",
        "provider": settings.provider,
        "model": settings.model,
        "mock": settings.is_mock,
    }


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse | HTMLResponse:
    if request.url.path.startswith("/api"):
        body: dict = {"detail": exc.message}
        if isinstance(exc, InsufficientDataError):
            body["missing_data"] = [item.model_dump() for item in exc.missing]
        if isinstance(exc, LLMError) and exc.provider:
            body["provider"] = exc.provider
        return JSONResponse(status_code=exc.status_code, content=body)

    return _render_index(
        request,
        get_settings(),
        error=exc.message,
        status_code=exc.status_code,
    )


@app.exception_handler(RequestValidationError)
async def request_validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [
        {
            "loc": [str(part) for part in error.get("loc", ())],
            "msg": format_error_item(error),
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Некорректный входной JSON",
            "errors": errors,
        },
    )


def _render_index(
    request: Request,
    settings: Settings,
    *,
    payload: str | None = None,
    error: str | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    examples = example_options()
    context = {
        "request": request,
        "examples": examples,
        "payload": payload if payload is not None else examples[0]["payload"],
        "error": error,
        "provider": settings.provider,
        "model": settings.model,
    }
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=context,
        status_code=status_code,
    )
