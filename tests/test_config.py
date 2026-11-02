"""Configuration: provider defaults, env parsing and API key fallbacks."""

from __future__ import annotations

from app.config import Settings


def test_defaults_to_mock_without_key() -> None:
    settings = Settings.from_env({})

    assert settings.provider == "mock"
    assert settings.is_mock
    assert settings.model == "mock-1"
    assert settings.api_key is None


def test_direct_construction_resolves_provider_defaults() -> None:
    anthropic = Settings(provider="anthropic", api_key="sk-ant")
    ollama = Settings(provider="ollama")
    openai = Settings(provider="openai", api_key="sk")

    assert anthropic.model == "claude-3-5-haiku-latest"
    assert anthropic.base_url == "https://api.anthropic.com"
    assert ollama.model == "llama3.1"
    assert ollama.base_url == "http://localhost:11434"
    assert openai.model == "gpt-4o-mini"
    assert openai.base_url == "https://api.openai.com/v1"


def test_from_env_reads_and_normalizes_values() -> None:
    settings = Settings.from_env(
        {
            "LLM_PROVIDER": " OpenAI ",
            "LLM_MODEL": "gpt-4o",
            "LLM_API_KEY": " sk-test ",
            "LLM_BASE_URL": "https://gateway.example/v1/",
            "LLM_TIMEOUT": "15",
            "LLM_MAX_RETRIES": "4",
            "LLM_MAX_REPAIRS": "2",
            "LLM_TEMPERATURE": "0.7",
        }
    )

    assert settings.provider == "openai"
    assert settings.model == "gpt-4o"
    assert settings.api_key == "sk-test"
    assert settings.base_url == "https://gateway.example/v1"
    assert settings.timeout == 15.0
    assert settings.max_retries == 4
    assert settings.max_repairs == 2
    assert settings.temperature == 0.7


def test_provider_specific_api_key_fallback() -> None:
    openai = Settings.from_env({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-openai"})
    anthropic = Settings.from_env({"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "sk-ant"})

    assert openai.api_key == "sk-openai"
    assert anthropic.api_key == "sk-ant"


def test_invalid_numbers_fall_back_to_defaults() -> None:
    settings = Settings.from_env({"LLM_TIMEOUT": "soon", "LLM_MAX_RETRIES": "many"})

    assert settings.timeout == 60.0
    assert settings.max_retries == 2
