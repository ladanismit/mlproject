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

    # Project metadata
    PROJECT_NAME: str = "DocuMind AI"
    PROJECT_DESCRIPTION: str = (
        "Intelligent Document Analysis & Agent Platform"
    )
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # API Server configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # LLM Provider & Model settings
    LLM_PROVIDER: str = "openai"  # e.g., "openai", "groq", "google"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0

    # Embedding settings
    EMBEDDING_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # API Keys / Sensitive Credentials (read from environment / .env)
    OPENAI_API_KEY: str | None = None
    GROQ_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None

    # Text Chunking & RAG settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 4

    # Storage and Directory paths
    DATA_DIR: Path = Path("data")
    RAW_DATA_DIR: Path = Path("data/raw")
    PROCESSED_DATA_DIR: Path = Path("data/processed")
    VECTOR_STORE_DIR: Path = Path("data/vector_store")
    EVALUATION_DATA_DIR: Path = Path("evaluation")

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

print(settings.PROJECT_NAME)
print(settings.LLM_MODEL)
print(settings.RAW_DATA_DIR)
print(settings.EVALUATION_DATA_DIR)
