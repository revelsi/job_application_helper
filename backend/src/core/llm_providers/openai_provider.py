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
OpenAI LLM Provider Implementation.

Provides integration with OpenAI API using the unified LLM provider interface.
"""

from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    # Create a dummy class for type hints when OpenAI is not available
    class OpenAI:
        pass

    OPENAI_AVAILABLE = False

from src.core.llm_providers.base import (
    GenerationRequest,
    LLMProvider,
    ProviderCapabilities,
    ProviderType,
)
from src.utils.config import get_settings


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation with GPT-4.1-mini support."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. If None, uses environment variable.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package is not installed. Install with: pip install openai"
            )

        super().__init__(api_key)
        self.settings = get_settings()
        self._client = None

        # Set API key from settings if not provided
        if not self.api_key:
            self.api_key = self.settings.openai_api_key

    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.OPENAI

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return OpenAI provider capabilities optimized for GPT-4.1-mini."""
        return ProviderCapabilities(
            max_tokens=32768,  # GPT-4.1-mini max output tokens
            supports_streaming=True,
            supports_function_calling=True,
            rate_limit_per_minute=60,  # Conservative estimate
            cost_per_1k_tokens=0.0004,  # $0.40 per million input tokens
            models=[
                "gpt-4.1-mini",
                "gpt-4.1",
                "gpt-4.1-nano",
            ],
        )

    def is_available(self) -> bool:
        """Check if OpenAI provider is available."""
        return (
            OPENAI_AVAILABLE
            and self.api_key is not None
            and self.api_key != "your_openai_api_key_here"
            and len(self.api_key.strip()) > 0
        )

    def get_default_model(self) -> str:
        """Get the default OpenAI model - GPT-4.1-mini for cost efficiency."""
        return "gpt-4.1-mini"

    @property
    def client(self) -> OpenAI:
        """Lazy load OpenAI client."""
        if self._client is None:
            if not self.is_available():
                raise ValueError("OpenAI API key is required")
            self._client = OpenAI(api_key=self.api_key)
            self.logger.info("OpenAI client initialized")
        return self._client

    def _build_messages(self, request: GenerationRequest) -> List[Dict[str, Any]]:
        """Build messages in OpenAI format."""
        messages = []

        # Add system message with context-aware prompts
        system_prompt = self._get_system_prompt(request.content_type, request.context)
        messages.append({"role": "system", "content": system_prompt})

        # Add context if provided (excluding system prompt context)
        if request.context:
            context_parts = []
            for key, value in request.context.items():
                if value and key not in [
                    "industry",
                    "experience_level",
                    "company_size",
                ]:  # Skip system prompt context
                    context_parts.append(f"{key.replace('_', ' ').title()}: {value}")

            if context_parts:
                context_message = "Context:\n" + "\n".join(context_parts)
                messages.append({"role": "user", "content": context_message})

        # Add main prompt
        messages.append({"role": "user", "content": request.prompt})

        return messages

    def _make_api_call(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        """Make API call to OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            raise

    def _parse_response(
        self, response: Dict[str, Any]
    ) -> tuple[str, int, Optional[str]]:
        """Parse OpenAI response into (content, tokens_used, request_id)."""
        content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if response.usage else 0
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
                raise ValueError("OpenAI provider is not available")

            # Check rate limits
            self._wait_for_rate_limit()

            # Build messages
            messages = self._build_messages(request)
            model = request.model or self.get_default_model()
            max_tokens = request.max_tokens or self.capabilities.max_tokens

            self.logger.info(f"Starting streaming generation with model: {model}")

            # Make streaming API call
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=request.temperature,
                stream=True,
            )

            # Yield content chunks
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            self.logger.error(f"Streaming generation failed: {e}")
            raise


def get_openai_provider(api_key: Optional[str] = None) -> OpenAIProvider:
    """
    Factory function to create OpenAI provider.

    Args:
        api_key: Optional API key override.

    Returns:
        Configured OpenAI provider.
    """
    return OpenAIProvider(api_key=api_key)
