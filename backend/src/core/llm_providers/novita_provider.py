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
Novita LLM Provider Implementation.

Provides integration with Novita AI API using OpenAI SDK compatibility.
Optimized for gpt-oss-20b and other models available through Novita.
"""

from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional

try:
    from openai import AsyncOpenAI, OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from src.core.llm_providers.base import (
    GenerationRequest,
    LLMProvider,
    ProviderCapabilities,
    ProviderType,
)
from src.core.llm_providers.model_config import (
    get_model_config,
    get_models_for_provider,
    get_safe_token_limits,
)
from src.utils.config import get_settings


class NovitaProvider(LLMProvider):
    """Novita provider implementation using OpenAI SDK compatibility."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Novita provider.

        Args:
            api_key: Novita API key. If None, uses environment variable.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required for Novita provider. Install with: pip install openai"
            )

        super().__init__(api_key)
        self.settings = get_settings()
        self._client = None
        self._async_client = None

        # Set API key from settings if not provided
        if not self.api_key:
            self.api_key = getattr(self.settings, "novita_api_key", None)

        # Configure base URL
        self.base_url = (
            getattr(self.settings, "novita_base_url", None)
            or "https://api.novita.ai/openai"
        )

    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.NOVITA

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return Novita provider capabilities based on model configurations."""
        models = get_models_for_provider("novita")
        model_names = [model.name for model in models]
        
        # Use default model capabilities as baseline
        default_config = get_model_config(self.get_default_model())
        if default_config:
            return ProviderCapabilities(
                max_tokens=default_config.max_output_tokens,
                supports_streaming=default_config.supports_streaming,
                supports_function_calling=default_config.supports_function_calling,
                rate_limit_per_minute=default_config.rate_limit_rpm,
                cost_per_1k_tokens=default_config.cost_per_1k_input_tokens,
                models=model_names,
            )
        
        # Fallback if model config not found
        return ProviderCapabilities(
            max_tokens=8192,
            supports_streaming=True,
            supports_function_calling=False,
            rate_limit_per_minute=100,
            cost_per_1k_tokens=0.0001,
            models=model_names,
        )

    def is_available(self) -> bool:
        """Check if Novita provider is available."""
        return (
            OPENAI_AVAILABLE
            and self.api_key is not None
            and self.api_key != "your_novita_api_key_here"
            and len(self.api_key.strip()) > 0
        )

    def get_default_model(self) -> str:
        """Get the default Novita model - gpt-oss-20b."""
        return "openai/gpt-oss-20b"

    @property
    def client(self) -> OpenAI:
        """Lazy load OpenAI client configured for Novita."""
        if self._client is None:
            if not self.is_available():
                raise ValueError("Novita API key is required")
            
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
            self.logger.info("Novita client initialized")
        return self._client

    @property
    def async_client(self) -> AsyncOpenAI:
        """Lazy load async OpenAI client configured for Novita."""
        if self._async_client is None:
            if not self.is_available():
                raise ValueError("Novita API key is required")
            
            self._async_client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
            self.logger.info("Novita async client initialized")
        return self._async_client



    def _make_api_call(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
    ) -> Any:
        """Make API call to Novita using OpenAI SDK with proper parameter handling."""
        try:
            # Get safe token limits for this specific model
            token_limits = get_safe_token_limits(model)
            safe_max_tokens = min(max_tokens, token_limits["recommended_output"])
            
            # Build API parameters according to Novita documentation
            api_params = {
                "model": model,
                "messages": messages,
                "max_tokens": safe_max_tokens,
                "temperature": temperature,
                "stream": False,
            }
            
            # Add optional parameters based on Novita API documentation
            # Note: reasoning_effort and verbosity are not supported by Novita
            if reasoning_effort:
                self.logger.debug(f"Ignoring reasoning_effort '{reasoning_effort}' - not supported by Novita")
            if verbosity:
                self.logger.debug(f"Ignoring verbosity '{verbosity}' - not supported by Novita")
            
            # Optional parameters that Novita supports (from documentation)
            # These can be added based on model capabilities
            model_config = get_model_config(model)
            if model_config and model_config.supports_function_calling:
                # Add function calling support if available
                pass
            
            # Additional Novita parameters from documentation:
            # - top_p: Nucleus sampling (0.0 to 1.0)
            # - top_k: Limits candidate token count
            # - presence_penalty: Controls repeated tokens (-2.0 to 2.0)
            # - frequency_penalty: Control token frequency (-2.0 to 2.0)
            # - repetition_penalty: Penalizes or encourages repetition
            # - stop: Strings that will terminate generation when encountered
            
            # For now, we'll use default values, but these could be made configurable
            # through the GenerationRequest in the future
            
            self.logger.debug(f"Novita API call: {model} with {safe_max_tokens} max tokens")
            
            response = self.client.chat.completions.create(**api_params)
            return response
        except Exception as e:
            self.logger.error(f"Novita API call failed: {e}")
            raise

    def _parse_response(self, response: Any) -> tuple[str, int, Optional[str]]:
        """Parse Novita response into (content, tokens_used, request_id)."""
        try:
            choice = response.choices[0]
            content = choice.message.content or ""
            
            # Get usage information
            tokens_used = 0
            if hasattr(response, "usage") and response.usage:
                tokens_used = response.usage.total_tokens
            
            # Get request ID if available
            request_id = getattr(response, "id", None)
            
            return content, tokens_used, request_id
        except Exception as e:
            self.logger.error(f"Failed to parse Novita response: {e}")
            return "", 0, None

    async def generate_content_stream(
        self, request: GenerationRequest, timeout: float = 30.0
    ) -> AsyncGenerator[str, None]:
        """
        Generate content using streaming.

        Args:
            request: Generation request with prompt and parameters.
            timeout: Request timeout in seconds.

        Yields:
            Content chunks as they are generated.
        """
        try:
            # Check availability
            if not self.is_available():
                raise ValueError("Novita provider is not available")

            # Build messages and parameters
            messages = self._build_messages(request)
            model = request.model or self.get_default_model()
            
            # Get safe token limits for this specific model
            token_limits = get_safe_token_limits(model, request.max_tokens)
            max_tokens = request.max_tokens or token_limits["recommended_output"]

            self.logger.info(f"Starting Novita streaming with model: {model} (max_tokens: {max_tokens})")

            # Note: reasoning_effort and verbosity are not supported by Novita
            if request.reasoning_effort:
                self.logger.debug(f"Ignoring reasoning_effort '{request.reasoning_effort}' - not supported by Novita")
            if request.verbosity:
                self.logger.debug(f"Ignoring verbosity '{request.verbosity}' - not supported by Novita")

            # Make streaming API call
            stream = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=request.temperature,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content

        except Exception as e:
            self.logger.error(f"Novita streaming generation failed: {e}")
            raise


def get_novita_provider(api_key: Optional[str] = None) -> NovitaProvider:
    """
    Factory function to create Novita provider.

    Args:
        api_key: Optional API key override.

    Returns:
        Configured Novita provider.
    """
    return NovitaProvider(api_key=api_key)
