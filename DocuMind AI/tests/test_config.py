"""Unit tests for DocuMind AI centralized configuration and environment management.

This module verifies that Settings correctly loads defaults, reads values from .env,
supports environment variable overrides, validates runtime credentials, and manages
provider configurations safely.
"""

from pathlib import Path
import pytest

from app.core.config import Settings, get_settings, validate_runtime_config


def test_default_configuration() -> None:
    """Verify default Settings instance exposes expected Gemini and RAG defaults."""
    settings = Settings()
    assert settings.APP_NAME == "DocuMind AI"
    assert settings.LLM_PROVIDER == "gemini"
    assert settings.EMBEDDING_PROVIDER == "gemini"
    assert settings.API_HOST == "127.0.0.1"
    assert settings.API_PORT == 8000
    assert settings.CHUNK_SIZE == 1000
    assert settings.CHUNK_OVERLAP == 200
    assert settings.TOP_K == 5
    assert settings.RAW_DATA_DIR == Path("data/raw")
    assert settings.VECTOR_STORE_DIR == Path("data/vector_store")


def test_environment_variable_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify OS environment variables override .env and default settings."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("GEMINI_API_KEY", "test-override-key-12345")
    monkeypatch.setenv("CHUNK_SIZE", "1500")
    monkeypatch.setenv("TOP_K", "8")
    monkeypatch.setenv("API_PORT", "9000")

    # Clear cached settings to instantiate fresh
    get_settings.cache_clear()
    custom_settings = get_settings()

    assert custom_settings.LLM_MODEL == "gemini-2.0-flash"
    assert custom_settings.LLM_TEMPERATURE == 0.7
    assert custom_settings.GEMINI_API_KEY == "test-override-key-12345"
    assert custom_settings.CHUNK_SIZE == 1500
    assert custom_settings.TOP_K == 8
    assert custom_settings.API_PORT == 9000

    # Cleanup cache
    get_settings.cache_clear()


def test_validate_runtime_config_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify validate_runtime_config raises ValueError when GEMINI_API_KEY is empty or placeholder."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        validate_runtime_config()

    monkeypatch.setenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        validate_runtime_config()

    get_settings.cache_clear()


def test_validate_runtime_config_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify validate_runtime_config passes when valid GEMINI_API_KEY is supplied."""
    monkeypatch.setenv("GEMINI_API_KEY", "valid-gemini-test-key")
    get_settings.cache_clear()

    # Should not raise any exception
    validate_runtime_config()

    get_settings.cache_clear()
