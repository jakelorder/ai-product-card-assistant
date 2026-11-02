"""Provider clients and the shared HTTP helper (retries, error mapping)."""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import Settings
from app.errors import LLMError
from app.llm import build_client
from app.llm.anthropic_client import AnthropicClient
from app.llm.base import (
    LLMRequest,
    ensure_api_key,
    extract_content,
    post_for_json,
)
from app.llm.mock_client import MockClient
from app.llm.ollama_client import OllamaClient
from app.llm.openai_client import OpenAIClient
from app.models import ProductIn


@pytest.fixture
def request_stub(product: ProductIn) -> LLMRequest:
    return LLMRequest(system="system", user="user", product=product)


# --------------------------------------------------------------------------
# HTTP helper
# --------------------------------------------------------------------------


async def test_post_for_json_retries_transient_errors() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(503, json={"error": {"message": "busy"}})
        return httpx.Response(200, json={"ok": True})

    data = await post_for_json(
        "https://example.test/chat",
        provider="test",
        headers={},
        payload={"a": 1},
        timeout=5,
        retries=2,
        backoff=0,
        transport=httpx.MockTransport(handler),
    )

    assert data == {"ok": True}
    assert len(calls) == 3


async def test_post_for_json_does_not_retry_auth_errors() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with pytest.raises(LLMError) as info:
        await post_for_json(
            "https://example.test/chat",
            provider="test",
            headers={},
            payload={},
            timeout=5,
            retries=3,
            backoff=0,
            transport=httpx.MockTransport(handler),
        )

    assert info.value.status == 401
    assert "bad key" in info.value.message
    assert len(calls) == 1


async def test_post_for_json_retries_timeouts() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.TimeoutException("too slow")

    with pytest.raises(LLMError, match="после 2 попыток"):
        await post_for_json(
            "https://example.test/chat",
            provider="test",
            headers={},
            payload={},
            timeout=1,
            retries=1,
            backoff=0,
            transport=httpx.MockTransport(handler),
        )

    assert len(calls) == 2


async def test_post_for_json_rejects_non_json_success() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="<html>oops</html>"))

    with pytest.raises(LLMError, match="не-JSON"):
        await post_for_json(
            "https://example.test/chat",
            provider="test",
            headers={},
            payload={},
            timeout=5,
            retries=0,
            transport=transport,
        )


def test_extract_content_string_and_blocks() -> None:
    assert extract_content({"choices": [{"message": {"content": "text"}}]}, "p") == "text"
    blocks = {
        "choices": [
            {"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}
        ]
    }
    assert extract_content(blocks, "p") == "ab"


def test_extract_content_invalid_shape() -> None:
    with pytest.raises(LLMError, match="неожиданный формат"):
        extract_content({"unexpected": True}, "p")


def test_ensure_api_key() -> None:
    assert ensure_api_key("sk-1", "openai") == "sk-1"
    with pytest.raises(LLMError, match="API-ключ"):
        ensure_api_key(None, "openai")


# --------------------------------------------------------------------------
# Provider clients
# --------------------------------------------------------------------------


def test_build_client_returns_configured_provider() -> None:
    assert isinstance(build_client(Settings(provider="mock")), MockClient)
    assert isinstance(build_client(Settings(provider="openai", api_key="sk-1")), OpenAIClient)
    assert isinstance(build_client(Settings(provider="anthropic", api_key="sk-1")), AnthropicClient)
    assert isinstance(build_client(Settings(provider="ollama")), OllamaClient)


def test_build_client_rejects_unknown_provider() -> None:
    with pytest.raises(LLMError, match="Неизвестный LLM_PROVIDER"):
        build_client(Settings(provider="unknown"))


async def test_openai_missing_key_raises(
    request_stub: LLMRequest,
) -> None:
    client = OpenAIClient(Settings(provider="openai", api_key=None))

    with pytest.raises(LLMError, match="не задан API-ключ"):
        await client.generate(request_stub)


async def test_openai_builds_expected_request(
    request_stub: LLMRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    async def fake_post(url: str, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    monkeypatch.setattr("app.llm.openai_client.post_for_json", fake_post)
    client = OpenAIClient(Settings(provider="openai", api_key="sk-test", model="gpt-test"))

    content = await client.generate(request_stub)

    assert content == '{"ok": true}'
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["payload"]["model"] == "gpt-test"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["messages"][0]["role"] == "system"


async def test_anthropic_prefills_json_and_reads_blocks(
    request_stub: LLMRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    async def fake_post(url: str, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return {
            "content": [
                {"type": "text", "text": '"seo_title": "x"}'},
            ]
        }

    monkeypatch.setattr("app.llm.anthropic_client.post_for_json", fake_post)
    client = AnthropicClient(Settings(provider="anthropic", api_key="sk-ant", model="claude-test"))

    content = await client.generate(request_stub)

    assert content == '{"seo_title": "x"}'
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["payload"]["messages"][-1] == {
        "role": "assistant",
        "content": "{",
    }
    assert captured["headers"]["x-api-key"] == "sk-ant"


async def test_anthropic_empty_answer_raises(
    request_stub: LLMRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_post(url: str, **kwargs):
        return {"content": []}

    monkeypatch.setattr("app.llm.anthropic_client.post_for_json", fake_post)
    client = AnthropicClient(Settings(provider="anthropic", api_key="sk-ant"))

    with pytest.raises(LLMError, match="пустой ответ"):
        await client.generate(request_stub)


async def test_ollama_uses_json_format(
    request_stub: LLMRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    async def fake_post(url: str, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return {"message": {"content": json.dumps({"ok": True})}}

    monkeypatch.setattr("app.llm.ollama_client.post_for_json", fake_post)
    client = OllamaClient(Settings(provider="ollama", model="llama3.1"))

    content = await client.generate(request_stub)

    assert content == '{"ok": true}'
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["format"] == "json"
    assert captured["payload"]["stream"] is False
