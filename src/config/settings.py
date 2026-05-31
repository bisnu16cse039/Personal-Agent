"""
Configuration and settings for the Personal Assistant Agent.

This module loads environment variables, initializes the Ollama client,
and provides singleton access to configuration throughout the application.
"""

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self):
        """Initialize settings from .env file."""
        load_dotenv()

        # Ollama configuration
        self.OLLAMA_BASE_URL: str = os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "qwen3:14b")
        self.MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
        self.MODEL_MAX_TOKENS: int = int(os.getenv("MODEL_MAX_TOKENS", "1024"))

        # LangSmith tracing (optional)
        self.LANGSMITH_ENABLED: bool = os.getenv("LANGSMITH_ENABLED", "false").lower() == "true"

        # Phase 3: Thinking layer (can be disabled for faster Phase 2 mode)
        self.ENABLE_THINKING_LAYER: bool = os.getenv("ENABLE_THINKING_LAYER", "false").lower() == "true"

        # Data paths
        self.SQLITE_CHECKPOINT_PATH: str = os.getenv(
            "SQLITE_CHECKPOINT_PATH", "./data/conversation_history.db"
        )
        self.CHROMADB_PATH: str = os.getenv("CHROMADB_PATH", "./data/chroma_db")

        # Logging
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE_PATH: Optional[str] = os.getenv("LOG_FILE_PATH")

    def __repr__(self) -> str:
        """String representation of settings."""
        return (
            f"Settings("
            f"model={self.PRIMARY_MODEL}, "
            f"temp={self.MODEL_TEMPERATURE}, "
            f"ollama={self.OLLAMA_BASE_URL}, "
            f"thinking_layer={self.ENABLE_THINKING_LAYER}"
            f")"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get or create the global Settings instance.

    Uses functools.lru_cache to ensure only one Settings object is created
    and reused throughout the application lifecycle.

    Returns:
        Settings: Singleton Settings instance

    Example:
        >>> settings = get_settings()
        >>> print(settings.PRIMARY_MODEL)
        qwen3:14b
    """
    return Settings()


def get_ollama_client() -> ChatOpenAI:
    """
    Initialize the Ollama ChatOpenAI client.

    This function creates a ChatOpenAI client configured to connect to a local
    Ollama server using the /api/chat endpoint (not /v1/chat/completions).

    Returns:
        ChatOpenAI: Initialized LLM client

    Raises:
        ConnectionError: If Ollama server is not running or unreachable

    Example:
        >>> client = get_ollama_client()
        >>> from langchain_core.messages import HumanMessage
        >>> response = client.invoke([HumanMessage(content="What is AI?")])
        >>> print(response.content)
    """
    settings = get_settings()

    # Use Ollama's /api/chat endpoint (compatible with OpenAI API format)
    # Note: base_url should end with /v1 for OpenAI-compatible API
    ollama_api_url = settings.OLLAMA_BASE_URL.rstrip("/") + "/v1"

    client = ChatOpenAI(
        base_url=ollama_api_url,
        model=settings.PRIMARY_MODEL,
        temperature=settings.MODEL_TEMPERATURE,
        max_tokens=settings.MODEL_MAX_TOKENS,
        api_key="ollama",  # Ollama doesn't require a real key
    )

    # NOTE: Validation skipped because Ollama doesn't expose /v1/models endpoint
    # The client will fail on first actual invoke() if model/server is unavailable
    # This is acceptable for production since errors bubble up immediately

    return client
