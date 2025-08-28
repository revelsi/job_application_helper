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
Mistral LLM Provider Implementation.

Provides integration with Mistral AI API using the unified LLM provider interface.
Supports Mistral Small and Medium models with function calling capabilities.
"""

from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional

try:
    from mistralai import Mistral

    MISTRAL_AVAILABLE = True
except ImportError:
    # Create a dummy class for type hints when Mistral is not available
    class Mistral:
        pass

    MISTRAL_AVAILABLE = False

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


class MistralProvider(LLMProvider):
    """Mistral provider implementation with Mistral Small and Medium support."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Mistral provider.

        Args:
            api_key: Mistral API key. If None, uses environment variable.
        """
        if not MISTRAL_AVAILABLE:
            raise ImportError(
                "Mistral package is not installed. Install with: pip install mistralai"
            )

        super().__init__(api_key)
        self.settings = get_settings()
        self._client = None

        # Set API key from settings if not provided
        if not self.api_key:
            self.api_key = getattr(self.settings, "mistral_api_key", None)

    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.MISTRAL

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return Mistral provider capabilities based on model configurations."""
        models = get_models_for_provider("mistral")
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
            supports_function_calling=True,
            rate_limit_per_minute=60,
            cost_per_1k_tokens=0.0002,
            models=model_names,
        )

    def is_available(self) -> bool:
        """Check if Mistral provider is available."""
        return (
            MISTRAL_AVAILABLE
            and self.api_key is not None
            and self.api_key != "your_mistral_api_key_here"
            and len(self.api_key.strip()) > 0
        )

    def get_default_model(self) -> str:
        """Get the default Mistral model - mistral-small-latest."""
        return "mistral-small-latest"

    @property
    def client(self) -> Mistral:
        """Lazy load Mistral client."""
        if self._client is None:
            if not self.is_available():
                raise ValueError("Mistral API key is required")
            self._client = Mistral(api_key=self.api_key)
            self.logger.info("Mistral client initialized")
        return self._client

    def _build_messages(self, request: GenerationRequest) -> List[Dict[str, Any]]:
        """Build messages in Mistral format."""
        messages = []

        # For Mistral models, we build context in the user message

        # Build user message with context if provided
        user_content = ""

        # Add context if provided
        if request.context:
            context_parts = []
            for key, value in request.context.items():
                if value and isinstance(value, str) and len(value.strip()) > 0:
                    context_parts.append(f"{key.replace('_', ' ').title()}: {value}")

            if context_parts:
                user_content += "Context:\n" + "\n".join(context_parts) + "\n\n"

        # Add main prompt
        user_content += request.prompt

        messages.append({"role": "user", "content": user_content})

        return messages



    def _make_api_call(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make API call to Mistral."""
        try:
            # Build API call parameters based on official documentation
            api_params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }

            # Note: reasoning_effort and verbosity parameters are ignored as Mistral doesn't support these features
            if reasoning_effort:
                self.logger.debug(f"Ignoring reasoning_effort '{reasoning_effort}' - not supported by Mistral")
            if verbosity:
                self.logger.debug(f"Ignoring verbosity '{verbosity}' - not supported by Mistral")

            response = self.client.chat.complete(**api_params)
            return response
        except Exception as e:
            self.logger.error(f"Mistral API call failed: {e}")
            raise

    def _make_streaming_api_call(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float,
        reasoning_effort: Optional[str] = None,
    ):
        """Make streaming API call to Mistral."""
        try:
            # Build API call parameters based on official documentation
            api_params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            }



            stream = self.client.chat.stream(**api_params)
            return stream
        except Exception as e:
            self.logger.error(f"Mistral streaming API call failed: {e}")
            raise





    def _parse_response(
        self, response: Dict[str, Any]
    ) -> tuple[str, int, Optional[str]]:
        """Parse Mistral response into (content, tokens_used, request_id)."""
        raw_content = response.choices[0].message.content

        # Return the content as-is
        content = raw_content

        tokens_used = (
            getattr(response.usage, "total_tokens", 0)
            if hasattr(response, "usage") and response.usage
            else 0
        )
        request_id = getattr(response, "id", None)

        return content, tokens_used, request_id

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
                raise ValueError("Mistral provider is not available")

            # Check rate limits
            self._wait_for_rate_limit()

            # Build messages and parameters
            messages = self._build_messages(request)
            model = request.model or self.get_default_model()
            
            # Get safe token limits for this specific model
            token_limits = get_safe_token_limits(model, request.max_tokens)
            max_tokens = request.max_tokens or token_limits["recommended_output"]

            self.logger.info(f"Starting Mistral streaming with model: {model} (max_tokens: {max_tokens})")

            # Make streaming API call
            stream = self._make_streaming_api_call(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=request.temperature,
            )

            # Process stream chunks
            for chunk in stream:
                if (
                    hasattr(chunk, "data")
                    and chunk.data
                    and hasattr(chunk.data, "choices")
                ):
                    if chunk.data.choices and len(chunk.data.choices) > 0:
                        delta = chunk.data.choices[0].delta
                        if hasattr(delta, "content") and delta.content:
                            yield delta.content

        except Exception as e:
            self.logger.error(f"Mistral streaming generation failed: {e}")
            raise


def get_mistral_provider(api_key: Optional[str] = None) -> MistralProvider:
    """
    Factory function to create Mistral provider.

    Args:
        api_key: Optional API key override.

    Returns:
        Configured Mistral provider.
    """
    return MistralProvider(api_key=api_key)
