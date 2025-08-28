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
API Key Management Endpoints.

Provides secure API key management through the backend with encryption
and proper validation. All keys are stored locally and encrypted.
"""

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.llm_providers.factory import (
    clear_api_key_manager_cache,
    clear_provider_cache,
    force_refresh_provider_availability,
    get_api_key_manager,
    list_provider_info,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/keys", tags=["api-keys"])


class APIKeyRequest(BaseModel):
    """Request model for setting API keys."""

    provider: str = Field(..., description="Provider name (openai, mistral, huggingface, ollama)")
    api_key: str = Field(..., description="API key for the provider")


class APIKeyResponse(BaseModel):
    """Response model for API key operations."""

    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Status message")
    provider: Optional[str] = Field(None, description="Provider name")
    configured: Optional[bool] = Field(
        None, description="Whether provider is configured"
    )


class ProviderStatusResponse(BaseModel):
    """Response model for provider status."""

    providers: Dict[str, Dict] = Field(..., description="Provider configuration status")
    has_any_configured: bool = Field(
        ..., description="Whether any provider is configured"
    )
    has_env_configured: bool = Field(
        ..., description="Whether any provider is configured via environment variables"
    )


@router.get("/status", response_model=ProviderStatusResponse)
async def get_api_key_status():
    """
    Get the status of all API key providers.

    Returns:
        Status of all configured providers without exposing actual keys.
    """
    try:
        key_manager = get_api_key_manager()
        config_status = key_manager.list_configured_providers()

        # Check if any provider is configured
        has_any_configured = any(
            status.get("configured", False) for status in config_status.values()
        )

        # Check if any provider is configured via environment (not just stored)
        has_env_configured = any(
            status.get("has_env_key", False) for status in config_status.values()
        )

        logger.info(
            f"API key status requested - configured providers: {[k for k, v in config_status.items() if v.get('configured')]}"
        )
        logger.info(
            f"Environment-configured providers: {[k for k, v in config_status.items() if v.get('has_env_key')]}"
        )

        return ProviderStatusResponse(
            providers=config_status,
            has_any_configured=has_any_configured,
            has_env_configured=has_env_configured,
        )
    except Exception as e:
        logger.error(f"Error getting API key status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get API key status")


@router.get("/providers", response_model=List[Dict])
async def get_provider_info():
    """
    Get detailed information about all available providers.

    Returns:
        List of provider information including capabilities and configuration status.
    """
    try:
        provider_info = list_provider_info()
        logger.info(
            f"Provider info requested - {len(provider_info)} providers available"
        )
        return provider_info
    except Exception as e:
        logger.error(f"Error getting provider info: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get provider information"
        )


@router.post("/set", response_model=APIKeyResponse)
async def set_api_key(request: APIKeyRequest):
    """
    Set an API key for a provider.

    The key will be encrypted and stored locally. It will never be logged
    or transmitted to any external service except the provider's API.
    """
    try:
        key_manager = get_api_key_manager()

        # Validate provider
        if request.provider not in ["openai", "mistral", "novita", "ollama"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid provider. Supported: openai, mistral, novita, ollama",
            )

        # Validate API key format
        if request.provider == "openai" and not request.api_key.startswith("sk-"):
            raise HTTPException(
                status_code=400,
                detail="Invalid OpenAI API key format. Should start with 'sk-'",
            )
        
        if request.provider == "novita" and len(request.api_key.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Invalid Novita API key format. Please provide a valid API key.",
            )
        
        if request.provider == "ollama":
            # Ollama doesn't use API keys, but we can accept any value for consistency
            # The actual availability is checked by connection test
            pass

        # Store the API key
        success = key_manager.set_api_key(request.provider, request.api_key)

        if success:
            # Clear any cached provider availability to ensure fresh loading
            clear_api_key_manager_cache()
            force_refresh_provider_availability()
            clear_provider_cache()

            logger.info(f"API key set successfully for {request.provider}")
            return APIKeyResponse(
                success=True,
                message=f"API key for {request.provider} has been securely stored",
                provider=request.provider,
                configured=True,
            )
        logger.error(f"Failed to store API key for {request.provider}")
        raise HTTPException(status_code=500, detail="Failed to store API key")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting API key for {request.provider}: {e}")
        raise HTTPException(status_code=500, detail="Failed to set API key")


@router.delete("/{provider}", response_model=APIKeyResponse)
async def remove_api_key(provider: str):
    """
    Remove an API key for a provider.

    The key will be permanently deleted from local storage.
    """
    try:
        key_manager = get_api_key_manager()

        # Validate provider
        if provider not in ["openai", "mistral", "novita", "ollama"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid provider. Supported: openai, mistral, novita, ollama",
            )

        # Remove the API key
        success = key_manager.remove_api_key(provider)

        if success:
            # Clear any cached provider availability to ensure fresh loading
            clear_api_key_manager_cache()
            force_refresh_provider_availability()
            clear_provider_cache()

            logger.info(f"API key removed successfully for {provider}")
            return APIKeyResponse(
                success=True,
                message=f"API key for {provider} has been removed",
                provider=provider,
                configured=False,
            )
        logger.error(f"Failed to remove API key for {provider}")
        raise HTTPException(status_code=500, detail="Failed to remove API key")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing API key for {provider}: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove API key")


@router.post("/test/{provider}", response_model=APIKeyResponse)
async def test_api_key(provider: str):
    """
    Test if an API key for a provider is valid.

    This will make a minimal API call to verify the key works.
    """
    try:
        key_manager = get_api_key_manager()

        # Validate provider
        if provider not in ["openai", "mistral", "novita", "ollama"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid provider. Supported: openai, mistral, novita, ollama",
            )

        # Get the API key
        api_key = key_manager.get_api_key(provider)
        if not api_key:
            raise HTTPException(
                status_code=404, detail=f"No API key configured for {provider}"
            )

        # Test the API key by creating a provider instance
        from src.core.llm_providers.base import ProviderType
        from src.core.llm_providers.factory import get_llm_provider

        if provider == "openai":
            provider_type = ProviderType.OPENAI
        elif provider == "mistral":
            provider_type = ProviderType.MISTRAL
        elif provider == "novita":
            provider_type = ProviderType.NOVITA
        elif provider == "ollama":
            provider_type = ProviderType.OLLAMA
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported provider: {provider}"
            )

        llm_provider = get_llm_provider(provider_type, api_key)

        # Test availability
        if llm_provider.is_available():
            logger.info(f"API key test successful for {provider}")
            return APIKeyResponse(
                success=True,
                message=f"API key for {provider} is valid and working",
                provider=provider,
                configured=True,
            )
        logger.warning(f"API key test failed for {provider} - provider not available")
        return APIKeyResponse(
            success=False,
            message=f"API key for {provider} is not working",
            provider=provider,
            configured=False,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing API key for {provider}: {e}")
        raise HTTPException(status_code=500, detail="Failed to test API key")
