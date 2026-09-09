"""Centralized application configuration and environment settings for DocuMind AI.

This module defines the Settings class using Pydantic Settings to manage
all project configurations, directory paths, model hyperparameters,
and API credentials loaded from environment variables or .env files.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and environment configuration."""

    # Application & Project metadata
    APP_NAME: str = "DocuMind AI"
    APP_VERSION: str = "1.0.0"
    PROJECT_NAME: str = "DocuMind AI"
    PROJECT_DESCRIPTION: str = (
        "Intelligent Document Analysis & Agent Platform"
    )
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # API Server configuration
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    # LLM Provider & Model settings
    LLM_PROVIDER: str = "gemini"
    LLM_MODEL: str = "gemini-1.5-flash"
    LLM_TEMPERATURE: float = 0.0

    # Gemini & Provider Credentials (read from environment / .env)
    GEMINI_API_KEY: str = ""
    GOOGLE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None

    # Embedding settings
    EMBEDDING_PROVIDER: str = "gemini"
    EMBEDDING_MODEL: str = "models/text-embedding-004"

    # Text Chunking & RAG settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K: int = 5
    TOP_K_RETRIEVAL: int = 5

    # Storage and Directory paths
    DATA_DIR: Path = Path("data")
    RAW_DATA_DIR: Path = Path("data/raw")
    PROCESSED_DATA_DIR: Path = Path("data/processed")
    VECTOR_STORE_DIR: Path = Path("data/vector_store")
    EVALUATION_DATA_DIR: Path = Path("evaluation")

    # OCR configuration
    TESSERACT_CMD: Optional[str] = None

    # Pydantic Settings configuration: load .env and ignore extra environment variables
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of the application settings."""
    return Settings()


# Default globally accessible instance
settings: Settings = get_settings()


def validate_runtime_config() -> None:
    """Validate runtime credentials for the currently selected providers.

    Raises:
        ValueError: If a required API key or provider setting is missing.
    """
    current_settings = get_settings()
    if current_settings.LLM_PROVIDER.lower() in {"gemini", "google"}:
        active_key = current_settings.GEMINI_API_KEY or current_settings.GOOGLE_API_KEY
        if not active_key or not active_key.strip() or active_key.strip() == "YOUR_GEMINI_API_KEY_HERE":
            raise ValueError(
                "GEMINI_API_KEY is not configured. "
                "Please set GEMINI_API_KEY in your environment or .env file."
            )
