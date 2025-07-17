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

from typing import Dict, List, Optional

from src.core.credentials import get_credentials_manager
from src.core.llm_providers.base import LLMProvider, ProviderType
from src.core.llm_providers.openai_provider import OPENAI_AVAILABLE, OpenAIProvider
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class APIKeyManager:
    """Manages API keys from multiple sources with priority order."""

    def __init__(self):
        """Initialize API key manager."""
        self.settings = get_settings()
        self.credentials = get_credentials_manager()

    def get_api_key(
        self, provider: str, override_key: Optional[str] = None
    ) -> Optional[str]:
        """
        Get API key for a provider with priority order:
        1. Override key (passed directly)
        2. Secure local storage (user-entered via UI)
        3. Environment variables (developer setup)

        Args:
            provider: Provider name (openai)
            override_key: Direct API key override

        Returns:
            API key if found, None otherwise
        """
        # Priority 1: Direct override
        if (
            override_key
            and override_key.strip()
            and override_key != f"your_{provider}_api_key_here"
        ):
            return override_key.strip()

        # Priority 2: Secure local storage (UI-entered keys)
        stored_key = self.credentials.get_api_key(provider)
        if stored_key:
            return stored_key

        # Priority 3: Environment variables (developer setup)
        env_key = None
        if provider == "openai":
            env_key = self.settings.openai_api_key

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

        for provider in ["openai"]:
            has_env_key = False
            has_stored_key = False

            # Check environment variable
            if provider == "openai":
                env_key = self.settings.openai_api_key

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


# Global API key manager instance
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
        if OPENAI_AVAILABLE:
            openai_key = key_manager.get_api_key("openai")
            openai_provider = OpenAIProvider(api_key=openai_key)
            providers[ProviderType.OPENAI] = openai_provider.is_available()
        else:
            providers[ProviderType.OPENAI] = False
    except Exception:
        providers[ProviderType.OPENAI] = False

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
        if not OPENAI_AVAILABLE:
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

    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")


def get_default_provider() -> LLMProvider:
    """
    Get the default LLM provider based on configuration or availability.

    This function will:
    1. Check if a default provider is set in configuration
    2. Fall back to the first available provider
    3. Raise an error if no providers are available

    Returns:
        Configured LLM provider instance.

    Raises:
        ValueError: If no providers are available.
    """
    logger.info("🔍 Starting LLM provider initialization...")
    settings = get_settings()
    key_manager = get_api_key_manager()

    # Log current configuration status
    config_status = key_manager.list_configured_providers()
    logger.debug(f"📊 Provider configuration status: {config_status}")

    # Try to get provider from configuration
    default_provider_name = getattr(settings, "default_llm_provider", None)
    if default_provider_name:
        logger.info(f"🎯 Trying configured default provider: {default_provider_name}")
        try:
            provider_type = ProviderType(default_provider_name.lower())
            provider = get_llm_provider(provider_type)
            logger.info(
                f"✅ Successfully initialized configured provider: {provider_type.value}"
            )
            return provider
        except (ValueError, ImportError) as e:
            logger.warning(
                f"❌ Configured default provider '{default_provider_name}' not available: {e}"
            )

    # Fall back to first available provider
    logger.info("🔄 Falling back to first available provider...")
    available = get_available_providers()
    logger.debug(f"📋 Available providers: {available}")

    # Prefer OpenAI (GPT-4.1)
    for provider_type in [
        ProviderType.OPENAI
    ]:
        logger.debug(f"🔍 Checking {provider_type.value} provider...")

        if available.get(provider_type, False):
            logger.info(
                f"✅ {provider_type.value} is available, attempting to initialize..."
            )
            try:
                provider = get_llm_provider(provider_type)
                logger.info(
                    f"🎉 Successfully initialized {provider_type.value} as default provider"
                )
                return provider
            except Exception as e:
                logger.error(f"❌ Failed to initialize {provider_type.value}: {e}")
        else:
            # Get more details about why it's not available
            provider_config = config_status.get(provider_type.value.lower(), {})
            logger.debug(
                f"❌ {provider_type.value} not available - config: {provider_config}"
            )

    # No providers available
    logger.error("❌ No LLM providers are available")

    error_msg = (
        "No LLM providers are available. Please configure at least one API key:\n\n"
    )

    for provider, status in config_status.items():
        if provider == "openai":
            error_msg += f"• OpenAI: {'✓ Configured' if status['configured'] else '✗ Not configured'}\n"

    error_msg += "\nYou can configure API keys:\n"
    error_msg += "- Through the UI (recommended for regular users)\n"
    error_msg += "- Set environment variables (for developers):\n"
    error_msg += "  - OPENAI_API_KEY for OpenAI"

    logger.error(f"🚨 Provider initialization failed: {error_msg}")
    raise ValueError(error_msg)


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
        if OPENAI_AVAILABLE:
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
                    "description": "GPT-4.1 with 1M token context and superior coding capabilities",
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

    return info
