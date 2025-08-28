"""
LLM Provider Abstraction for Job Application Helper.

This module provides a unified interface for different LLM providers.
"""

from src.core.llm_providers.base import (
    ContentType,
    GenerationRequest,
    GenerationResponse,
    LLMProvider,
    ProviderCapabilities,
    ProviderType,
)
from src.core.llm_providers.factory import get_available_providers, get_llm_provider
from src.core.llm_providers.novita_provider import NovitaProvider
from src.core.llm_providers.ollama_provider import OllamaProvider

__all__ = [
    "ContentType",
    "GenerationRequest",
    "GenerationResponse",
    "LLMProvider",
    "NovitaProvider",
    "OllamaProvider",
    "ProviderCapabilities",
    "ProviderType",
    "get_available_providers",
    "get_llm_provider",
]
