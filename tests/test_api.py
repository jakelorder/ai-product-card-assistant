"""HTTP contract: web form, JSON API, coverage reporting and error codes."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import get_pipeline
from app.errors import LLMError
from app.main import app
from app.models import ProductIn
from app.pipeline import Pipeline
from tests.fakes import SequenceClient


def test_index_renders_form(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Входные данные (JSON)" in response.text
    assert "Сформировать карточку" in response.text
    assert "mock · mock-1" in response.text


def test_health_reports_provider(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "provider": "mock",
        "model": "mock-1",
        "mock": True,
    }


def test_generate_api_returns_full_payload(client: TestClient, product: ProductIn) -> None:
    response = client.post("/api/generate", json=product.model_dump())

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"product", "card", "missing_data", "markdown", "meta"}
    assert body["missing_data"] == []
    assert body["meta"]["provider"] == "mock"
    assert body["meta"]["attempts"] == 1
    assert len(body["card"]["seo_title"]) <= 70
    assert 800 <= len(body["card"]["seo_description"]) <= 1200
    assert 3 <= len(body["card"]["search_queries"]) <= 5
    assert body["markdown"].startswith("# ")


def test_generate_api_reports_missing_fields(client: TestClient, sparse_product: ProductIn) -> None:
    response = client.post("/api/generate", json=sparse_product.model_dump())

    assert response.status_code == 200
    fields = {item["field"] for item in response.json()["missing_data"]}
    assert fields == {"specs.материал корпуса", "description"}


def test_generate_api_rejects_blank_name(client: TestClient) -> None:
    response = client.post("/api/generate", json={"name": "  ", "manufacturer": "Завод"})

    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "Некорректный входной JSON"
    assert body["errors"][0]["loc"] == ["body", "name"]
    assert body["errors"][0]["msg"] == "значение не может быть пустым"


def test_generate_api_rejects_broken_json(client: TestClient) -> None:
    response = client.post(
        "/api/generate",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Некорректный входной JSON"


def test_generate_api_rejects_insufficient_data(client: TestClient) -> None:
    response = client.post("/api/generate", json={"name": "Прибор", "manufacturer": "Завод"})

    assert response.status_code == 422
    body = response.json()
    assert "Недостаточно данных" in body["detail"]
    assert {item["field"] for item in body["missing_data"]} == {"specs", "description"}


def test_generate_markdown_endpoint(client: TestClient, product: ProductIn) -> None:
    response = client.post("/api/generate/markdown", json=product.model_dump())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "## Ключевые характеристики" in response.text


def test_form_returns_result_page(client: TestClient, product: ProductIn) -> None:
    response = client.post(
        "/generate",
        data={"payload": json.dumps(product.model_dump(), ensure_ascii=False)},
    )

    assert response.status_code == 200
    assert "Markdown для копирования" in response.text
    assert product.name in response.text
    assert "Скопировать" in response.text


def test_form_shows_validation_error(client: TestClient) -> None:
    response = client.post("/generate", data={"payload": "{broken"})

    assert response.status_code == 422
    assert "Некорректный JSON" in response.text


def test_form_rejects_empty_payload(client: TestClient) -> None:
    response = client.post("/generate", data={"payload": "   "})

    assert response.status_code == 422
    assert "Вставьте JSON товара" in response.text


def test_form_shows_insufficient_data(client: TestClient) -> None:
    response = client.post(
        "/generate",
        data={"payload": json.dumps({"name": "Прибор", "manufacturer": "Завод"})},
    )

    assert response.status_code == 422
    assert "Недостаточно данных" in response.text


def test_llm_failure_returns_502(
    client: TestClient, settings: Settings, product: ProductIn
) -> None:
    failing = Pipeline(
        SequenceClient([LLMError("провайдер недоступен", provider="fake")]),
        settings,
    )
    app.dependency_overrides[get_pipeline] = lambda: failing

    response = client.post("/api/generate", json=product.model_dump())

    assert response.status_code == 502
    body = response.json()
    assert "провайдер недоступен" in body["detail"]
    assert body["provider"] == "fake"
