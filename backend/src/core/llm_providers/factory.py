"""
Copyright 2024 Job Application Helper Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
LLM Provider Factory.

Factory functions to create and manage LLM providers based on user preference.
Supports both environment variables (for developers) and secure local storage (for users).
"""

from typing import Dict, List, Optional, Tuple
import importlib

from src.core.credentials import get_credentials_manager
from src.core.llm_providers.base import LLMProvider, ProviderType
from src.core.llm_providers.model_config import get_provider_description
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)
def _try_import(module_path: str) -> Optional[object]:
    """Safely import a module, returning None on failure."""
    try:
        return importlib.import_module(module_path)
    except Exception:
        return None


def _get_openai_provider() -> Tuple[Optional[type], bool]:
    mod = _try_import("src.core.llm_providers.openai_provider")
    if not mod:
        return None, False
    provider_cls = getattr(mod, "OpenAIProvider", None)
    available = bool(provider_cls)
    return provider_cls, available


def _get_mistral_provider() -> Tuple[Optional[type], bool]:
    mod = _try_import("src.core.llm_providers.mistral_provider")
    if not mod:
        return None, False
    provider_cls = getattr(mod, "MistralProvider", None)
    available = bool(provider_cls)
    return provider_cls, available


def _get_ollama_provider() -> Tuple[Optional[type], bool]:
    mod = _try_import("src.core.llm_providers.ollama_provider")
    if not mod:
        return None, False
    provider_cls = getattr(mod, "OllamaProvider", None)
    # Also ensure httpx is available via provider flag if present
    httpx_flag = getattr(mod, "HTTPX_AVAILABLE", True)
    available = bool(provider_cls) and bool(httpx_flag)
    return provider_cls, available


def _get_novita_provider() -> Tuple[Optional[type], bool]:
    mod = _try_import("src.core.llm_providers.novita_provider")
    if not mod:
        return None, False
    provider_cls = getattr(mod, "NovitaProvider", None)
    available = bool(provider_cls)
    return provider_cls, available



class APIKeyManager:
    """Manages API keys from multiple sources with priority order."""

    def __init__(self) -> None:
        """Initialize API key manager."""
        self.settings = get_settings()
        self.credentials = get_credentials_manager()

    def get_api_key(
        self, provider: str, override_key: Optional[str] = None
    ) -> Optional[str]:
        """
        Retrieve the API key for a given provider, following this priority:
        1. Direct override (if provided)
        2. Secure local storage (user-entered via UI)
        3. Environment variables (developer setup)

        Args:
            provider: Name of the provider (e.g., "openai", "mistral")
            override_key: Optional direct API key override

        Returns:
            The API key if found, otherwise None.
        """
        # Priority 1: Direct override
        if override_key and override_key.strip() and override_key != f"your_{provider}_api_key_here":
            return override_key.strip()

        # Priority 2: Secure local storage (UI-entered keys)
        stored_key = self.credentials.get_api_key(provider)
        if stored_key:
            return stored_key

        # Priority 3: Environment variables (developer setup)
        env_key = None
        if provider == "openai":
            env_key = getattr(self.settings, "openai_api_key", None)
        elif provider == "mistral":
            env_key = getattr(self.settings, "mistral_api_key", None)
        elif provider == "novita":
            env_key = getattr(self.settings, "novita_api_key", None)
        elif provider == "ollama":
            # Ollama doesn't require API keys, but we check if it's available
            return "local"  # Special value indicating local availability

        if env_key and env_key.strip() and env_key != f"your_{provider}_api_key_here":
            return env_key.strip()

        return None

    def set_api_key(self, provider: str, api_key: str) -> bool:
        """
        Store API key securely for a provider.

        Args:
            provider: Provider name (openai)
            api_key: API key to store

        Returns:
            True if successful, False otherwise
        """
        return self.credentials.set_api_key(provider, api_key)

    def remove_api_key(self, provider: str) -> bool:
        """
        Remove stored API key for a provider.

        Args:
            provider: Provider name (openai)

        Returns:
            True if successful, False otherwise
        """
        return self.credentials.remove_api_key(provider)

    def list_configured_providers(self) -> Dict[str, Dict[str, any]]:
        """
        List all providers and their configuration status.

        Returns:
            Dictionary with provider configuration info
        """
        result = {}

        for provider in ["openai", "mistral", "novita", "ollama"]:
            has_env_key = False
            has_stored_key = False

            # Check environment variable
            env_key = None
            if provider == "openai":
                env_key = self.settings.openai_api_key
            elif provider == "mistral":
                env_key = getattr(self.settings, "mistral_api_key", None)
            elif provider == "novita":
                env_key = getattr(self.settings, "novita_api_key", None)
            elif provider == "ollama":
                # Ollama doesn't use API keys - check if service is available
                env_key = "local"

            if (
                env_key
                and env_key.strip()
                and env_key != f"your_{provider}_api_key_here"
            ):
                has_env_key = True

            # Check stored key
            stored_key = self.credentials.get_api_key(provider)
            if stored_key:
                has_stored_key = True

            result[provider] = {
                "has_env_key": has_env_key,
                "has_stored_key": has_stored_key,
                "configured": has_env_key or has_stored_key,
                "source": (
                    "stored" if has_stored_key else ("env" if has_env_key else "none")
                ),
            }

        return result


# Global instances
_api_key_manager = None


def get_api_key_manager() -> APIKeyManager:
    """Get the global API key manager instance."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


def clear_api_key_manager_cache() -> None:
    """Clear the global API key manager cache to ensure fresh credential loading."""
    global _api_key_manager
    if _api_key_manager is not None:
        # Clear the credentials manager cache
        _api_key_manager.credentials.clear_cache()
        logger.debug("API key manager cache cleared")
    else:
        logger.debug("API key manager not initialized, nothing to clear")


def force_refresh_provider_availability() -> Dict[ProviderType, bool]:
    """
    Force refresh provider availability status by clearing any cached results.

    Returns:
        Fresh availability status for all providers
    """
    logger.debug("Forcing refresh of provider availability status")

    # Clear the API key manager cache
    clear_api_key_manager_cache()

    # Get fresh availability status
    return get_available_providers()


def get_available_providers() -> Dict[ProviderType, bool]:
    """
    Get list of available providers and their availability status.

    Returns:
        Dictionary mapping provider types to availability status.
    """
    providers = {}
    key_manager = get_api_key_manager()

    # Check OpenAI availability
    try:
        OpenAIProvider, openai_mod_available = _get_openai_provider()
        if openai_mod_available and OpenAIProvider:
            openai_key = key_manager.get_api_key("openai")
            providers[ProviderType.OPENAI] = OpenAIProvider(api_key=openai_key).is_available()
        else:
            providers[ProviderType.OPENAI] = False
    except Exception:
        providers[ProviderType.OPENAI] = False

    # Check Mistral availability
    try:
        MistralProvider, mistral_mod_available = _get_mistral_provider()
        if mistral_mod_available and MistralProvider:
            mistral_key = key_manager.get_api_key("mistral")
            providers[ProviderType.MISTRAL] = MistralProvider(api_key=mistral_key).is_available()
        else:
            providers[ProviderType.MISTRAL] = False
    except Exception:
        providers[ProviderType.MISTRAL] = False

    # Check Ollama availability
    try:
        OllamaProvider, ollama_mod_available = _get_ollama_provider()
        if ollama_mod_available and OllamaProvider:
            providers[ProviderType.OLLAMA] = OllamaProvider().is_available()
        else:
            providers[ProviderType.OLLAMA] = False
    except Exception:
        providers[ProviderType.OLLAMA] = False

    # Check Novita availability
    try:
        NovitaProvider, novita_mod_available = _get_novita_provider()
        if novita_mod_available and NovitaProvider:
            novita_key = key_manager.get_api_key("novita")
            providers[ProviderType.NOVITA] = NovitaProvider(api_key=novita_key).is_available()
        else:
            providers[ProviderType.NOVITA] = False
    except Exception:
        providers[ProviderType.NOVITA] = False

    return providers


def get_llm_provider(
    provider_type: ProviderType, api_key: Optional[str] = None
) -> LLMProvider:
    """
    Create an LLM provider instance based on user selection.

    Args:
        provider_type: The type of provider to create.
        api_key: Optional API key override.

    Returns:
        Configured LLM provider instance.

    Raises:
        ValueError: If provider type is not supported or not available.
        ImportError: If required dependencies are not installed.
    """
    key_manager = get_api_key_manager()

    if provider_type == ProviderType.OPENAI:
        OpenAIProvider, openai_mod_available = _get_openai_provider()
        if not (openai_mod_available and OpenAIProvider):
            raise ImportError(
                "OpenAI provider requires the openai package. Install with: pip install openai"
            )
        final_key = key_manager.get_api_key("openai", api_key)
        provider = OpenAIProvider(api_key=final_key)
        if not provider.is_available():
            raise ValueError(
                "OpenAI provider is not available. Please configure your OpenAI API key:\n"
                "- Through the UI (recommended), or\n"
                "- Set OPENAI_API_KEY environment variable"
            )
        return provider

    if provider_type == ProviderType.MISTRAL:
        MistralProvider, mistral_mod_available = _get_mistral_provider()
        if not (mistral_mod_available and MistralProvider):
            raise ImportError(
                "Mistral provider requires the mistralai package. Install with: pip install mistralai"
            )
        final_key = key_manager.get_api_key("mistral", api_key)
        provider = MistralProvider(api_key=final_key)
        if not provider.is_available():
            raise ValueError(
                "Mistral provider is not available. Please configure your Mistral API key:\n"
                "- Through the UI (recommended), or\n"
                "- Set MISTRAL_API_KEY environment variable"
            )
        return provider

    if provider_type == ProviderType.OLLAMA:
        OllamaProvider, ollama_mod_available = _get_ollama_provider()
        if not (ollama_mod_available and OllamaProvider):
            raise ImportError(
                "Ollama provider requires the httpx package. Install with: pip install httpx"
            )
        provider = OllamaProvider(api_key=api_key)  # API key not used for Ollama
        if not provider.is_available():
            raise ValueError(
                "Ollama provider is not available. Please ensure Ollama is running:\n"
                "- Install Ollama from https://ollama.ai\n"
                "- Run 'ollama serve' to start the service\n"
                "- Check that the service is accessible at the configured URL"
            )
        return provider

    if provider_type == ProviderType.NOVITA:
        NovitaProvider, novita_mod_available = _get_novita_provider()
        if not (novita_mod_available and NovitaProvider):
            raise ImportError(
                "Novita provider requires the openai package. Install with: pip install openai"
            )
        final_key = key_manager.get_api_key("novita", api_key)
        provider = NovitaProvider(api_key=final_key)
        if not provider.is_available():
            raise ValueError(
                "Novita provider is not available. Please configure your Novita API key:\n"
                "- Through the UI (recommended), or\n"
                "- Set NOVITA_API_KEY environment variable\n"
                "- Get your API key from https://novita.ai"
            )
        return provider

    raise ValueError(f"Unsupported provider type: {provider_type}")


def clear_provider_cache():
    """Clear the provider cache to force re-initialization."""
    # No longer needed since we removed the cache
    pass


def list_provider_info() -> List[Dict[str, any]]:
    """
    Get detailed information about all providers.

    Returns:
        List of dictionaries with provider information.
    """
    info = []
    available = get_available_providers()
    key_manager = get_api_key_manager()
    config_status = key_manager.list_configured_providers()

    # OpenAI info (prioritized first)
    try:
        OpenAIProvider, openai_mod_available = _get_openai_provider()
        if openai_mod_available and OpenAIProvider:
            openai_key = key_manager.get_api_key("openai")
            openai = OpenAIProvider(api_key=openai_key)
            info.append(
                {
                    "type": ProviderType.OPENAI.value,
                    "name": "OpenAI",
                    "available": available.get(ProviderType.OPENAI, False),
                    "configured": config_status.get("openai", {}).get(
                        "configured", False
                    ),
                    "key_source": config_status.get("openai", {}).get("source", "none"),
                    "capabilities": openai.capabilities,
                    "default_model": openai.get_default_model(),
                    "description": get_provider_description("openai"),
                }
            )
        else:
            info.append(
                {
                    "type": ProviderType.OPENAI.value,
                    "name": "OpenAI",
                    "available": False,
                    "configured": False,
                    "error": "OpenAI package not installed (pip install openai)",
                }
            )
    except Exception as e:
        info.append(
            {
                "type": ProviderType.OPENAI.value,
                "name": "OpenAI",
                "available": False,
                "configured": config_status.get("openai", {}).get("configured", False),
                "error": str(e),
            }
        )

    # Mistral info (second priority)
    try:
        MistralProvider, mistral_mod_available = _get_mistral_provider()
        if mistral_mod_available and MistralProvider:
            mistral_key = key_manager.get_api_key("mistral")
            mistral = MistralProvider(api_key=mistral_key)
            info.append(
                {
                    "type": ProviderType.MISTRAL.value,
                    "name": "Mistral",
                    "available": available.get(ProviderType.MISTRAL, False),
                    "configured": config_status.get("mistral", {}).get(
                        "configured", False
                    ),
                    "key_source": config_status.get("mistral", {}).get(
                        "source", "none"
                    ),
                    "capabilities": mistral.capabilities,
                    "default_model": mistral.get_default_model(),
                    "description": get_provider_description("mistral"),
                }
            )
        else:
            info.append(
                {
                    "type": ProviderType.MISTRAL.value,
                    "name": "Mistral",
                    "available": False,
                    "configured": False,
                    "error": "Mistral package not installed (pip install mistralai)",
                }
            )
    except Exception as e:
        info.append(
            {
                "type": ProviderType.MISTRAL.value,
                "name": "Mistral",
                "available": False,
                "configured": config_status.get("mistral", {}).get("configured", False),
                "error": str(e),
            }
        )

    # Ollama info (third priority)
    try:
        OllamaProvider, ollama_mod_available = _get_ollama_provider()
        if ollama_mod_available and OllamaProvider:
            ollama = OllamaProvider()
            info.append(
                {
                    "type": ProviderType.OLLAMA.value,
                    "name": "Ollama",
                    "available": available.get(ProviderType.OLLAMA, False),
                    "configured": config_status.get("ollama", {}).get("configured", False),
                    "key_source": "local",  # No API key needed
                    "capabilities": ollama.capabilities,
                    "default_model": ollama.get_default_model(),
                    "description": get_provider_description("ollama"),
                }
            )
        else:
            info.append(
                {
                    "type": ProviderType.OLLAMA.value,
                    "name": "Ollama",
                    "available": False,
                    "configured": False,
                    "error": "httpx package not installed (pip install httpx)",
                }
            )
    except Exception as e:
        info.append(
            {
                "type": ProviderType.OLLAMA.value,
                "name": "Ollama",
                "available": False,
                "configured": config_status.get("ollama", {}).get("configured", False),
                "error": str(e),
            }
        )

    # Novita info (fourth priority)
    try:
        NovitaProvider, novita_mod_available = _get_novita_provider()
        if novita_mod_available and NovitaProvider:
            novita_key = key_manager.get_api_key("novita")
            novita = NovitaProvider(api_key=novita_key)
            info.append(
                {
                    "type": ProviderType.NOVITA.value,
                    "name": "Novita",
                    "available": available.get(ProviderType.NOVITA, False),
                    "configured": config_status.get("novita", {}).get(
                        "configured", False
                    ),
                    "key_source": config_status.get("novita", {}).get(
                        "source", "none"
                    ),
                    "capabilities": novita.capabilities,
                    "default_model": novita.get_default_model(),
                    "description": get_provider_description("novita"),
                }
            )
        else:
            info.append(
                {
                    "type": ProviderType.NOVITA.value,
                    "name": "Novita",
                    "available": False,
                    "configured": False,
                    "error": "openai package not installed (pip install openai)",
                }
            )
    except Exception as e:
        info.append(
            {
                "type": ProviderType.NOVITA.value,
                "name": "Novita",
                "available": False,
                "configured": config_status.get("novita", {}).get(
                    "configured", False
                ),
                "error": str(e),
            }
        )

    return info
