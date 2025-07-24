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
Supports reasoning models like Magistral Small with <think></think> output parsing.
"""

import re
from typing import Any, AsyncGenerator, Dict, List, Optional

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
from src.utils.config import get_settings


class MistralProvider(LLMProvider):
    """Mistral provider implementation with Magistral Small reasoning support."""

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
            self.api_key = getattr(self.settings, 'mistral_api_key', None)

    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.MISTRAL

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return Mistral provider capabilities optimized for reasoning models."""
        return ProviderCapabilities(
            max_tokens=32768,  # Magistral Small max output tokens
            supports_streaming=False,  # Streaming not supported yet for reasoning
            supports_function_calling=False,  # Function calling not supported yet
            rate_limit_per_minute=60,  # Conservative estimate
            cost_per_1k_tokens=0.0002,  # Estimated cost per 1k tokens
            models=[
                "magistral-small-2506",
                "magistral-medium-2506",
            ],
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
        """Get the default Mistral model - Magistral Small for reasoning."""
        return "magistral-small-2506"

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

        # For reasoning models, we let Mistral handle the system prompt
        # by using prompt_mode="reasoning", but we can add context in user message
        
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
    ) -> Dict[str, Any]:
        """Make API call to Mistral."""
        try:
            response = self.client.chat.complete(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                prompt_mode="reasoning",  # Enable reasoning mode for <think></think> output
            )
            return response
        except Exception as e:
            self.logger.error(f"Mistral API call failed: {e}")
            raise

    def _parse_reasoning_response(self, content: str) -> tuple[str, str]:
        """
        Parse reasoning response to extract thinking and final answer.
        
        Args:
            content: Raw response content with <think></think> tags
            
        Returns:
            Tuple of (reasoning_trace, final_answer)
        """
        # Extract thinking section
        think_pattern = r'<think>(.*?)</think>'
        think_match = re.search(think_pattern, content, re.DOTALL)
        
        if think_match:
            reasoning_trace = think_match.group(1).strip()
            # Extract final answer (everything after </think>)
            final_answer = content[think_match.end():].strip()
        else:
            # Fallback: if no thinking tags, treat entire content as final answer
            reasoning_trace = ""
            final_answer = content.strip()
            
        return reasoning_trace, final_answer

    def _parse_response(
        self, response: Dict[str, Any]
    ) -> tuple[str, int, Optional[str]]:
        """Parse Mistral response into (content, tokens_used, request_id)."""
        raw_content = response.choices[0].message.content
        
        # Parse reasoning vs final answer
        reasoning_trace, final_answer = self._parse_reasoning_response(raw_content)
        
        # For now, return the final answer as the main content
        # In the future, we could modify the response format to include reasoning
        content = final_answer
        
        # Add reasoning trace as a comment for debugging (optional)
        if reasoning_trace and len(reasoning_trace) > 0:
            self.logger.debug(f"Reasoning trace: {reasoning_trace[:200]}...")
        
        tokens_used = getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') and response.usage else 0
        request_id = getattr(response, "id", None)
        
        return content, tokens_used, request_id

    async def generate_content_stream(
        self, request: GenerationRequest, timeout: float = 30.0
    ) -> AsyncGenerator[str, None]:
        """
        Generate content using fallback streaming for reasoning models.

        Since Mistral reasoning models don't support native streaming,
        this method calls generate_content() and yields the complete result.

        Args:
            request: Generation request with prompt and parameters.
            timeout: Request timeout in seconds.

        Yields:
            Content chunks as they are generated.
        """
        # For reasoning models, we can't do real streaming, so we get the full response
        # and yield it as one chunk to maintain compatibility with streaming interface
        response = self.generate_content(request, timeout)
        
        # Yield the complete content as a single chunk
        if response.content:
            yield response.content


def get_mistral_provider(api_key: Optional[str] = None) -> MistralProvider:
    """
    Factory function to create Mistral provider.

    Args:
        api_key: Optional API key override.

    Returns:
        Configured Mistral provider.
    """
    return MistralProvider(api_key=api_key) 