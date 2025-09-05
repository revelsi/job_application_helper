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
from src.core.llm_providers.model_config import (
    get_model_config,
    get_models_for_provider,
    get_safe_token_limits,
)
from src.utils.config import get_settings


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation with GPT-5-mini support."""

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
        """Return OpenAI provider capabilities based on model configurations."""
        models = get_models_for_provider("openai")
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
            max_tokens=16384,
            supports_streaming=True,
            supports_function_calling=True,
            rate_limit_per_minute=60,
            cost_per_1k_tokens=0.00015,
            models=model_names,
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
        """Get the default OpenAI model - GPT-5-mini for latest capabilities."""
        return "gpt-5-mini"

    def _supports_reasoning_effort(self, model: str) -> bool:
        """Check if a model supports reasoning_effort parameter."""
        # GPT-5 family supports reasoning_effort
        return isinstance(model, str) and model.startswith("gpt-5")
    
    def _supports_temperature(self, model: str) -> bool:
        """Check if a model supports temperature parameter."""
        # GPT-5 family does not support temperature parameter
        return not (isinstance(model, str) and model.startswith("gpt-5"))
    
    def _supports_verbosity(self, model: str) -> bool:
        """Check if a model supports verbosity parameter."""
        # GPT-5 family supports verbosity
        return isinstance(model, str) and model.startswith("gpt-5")

    @property
    def client(self) -> OpenAI:
        """Lazy load OpenAI client."""
        if self._client is None:
            if not self.is_available():
                raise ValueError("OpenAI API key is required")
            self._client = OpenAI(api_key=self.api_key)
            self.logger.info("OpenAI client initialized")
        return self._client



    def _make_api_call(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make API call to OpenAI."""
        try:
            # Get safe token limits for this specific model
            token_limits = get_safe_token_limits(model)
            safe_max_tokens = min(max_tokens, token_limits["recommended_output"])
            
            self.logger.debug(f"OpenAI API call: {model} with {safe_max_tokens} max tokens, reasoning_effort: {reasoning_effort}")
            
            # For GPT-5 family, use the Responses API; otherwise use Chat Completions
            if isinstance(model, str) and model.startswith("gpt-5"):
                # Flatten messages to a single input string to ensure compatibility
                input_segments: List[str] = []
                for m in messages:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    input_segments.append(f"{role.title()}: {content}")
                input_text = "\n".join(input_segments)

                api_params = {
                    "model": model,
                    "input": input_text,
                    "max_output_tokens": safe_max_tokens,
                }

                # Reasoning effort mapping for Responses API
                if reasoning_effort and self._supports_reasoning_effort(model):
                    api_params["reasoning"] = {"effort": reasoning_effort}
                    self.logger.debug(f"Using reasoning.effort: {reasoning_effort} for model: {model}")

                # Verbosity is not officially part of Responses API; avoid sending unknown params
                response = self.client.responses.create(**api_params)
                return response
            else:
                # Build API call parameters for Chat Completions
                api_params = {
                    "model": model,
                    "messages": messages,
                }

                # Only add temperature for models that support it (not GPT-5 models)
                if self._supports_temperature(model):
                    api_params["temperature"] = temperature
                    self.logger.debug(f"Using temperature: {temperature} for model: {model}")
                else:
                    self.logger.debug(f"Skipping temperature parameter for model: {model} (not supported)")

                # Use max_tokens for non-GPT-5 models
                api_params["max_tokens"] = safe_max_tokens

                response = self.client.chat.completions.create(**api_params)
                return response
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            raise

    def _parse_response(
        self, response: Dict[str, Any]
    ) -> tuple[str, int, Optional[str]]:
        """Parse OpenAI response into (content, tokens_used, request_id).

        Handles GPT-5 family responses where message.content can be a list of parts.
        """
        extracted_text = ""

        # First, try Responses API convenience field
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text:
            extracted_text = output_text
        else:
            # Try Chat Completions format
            choice = response.choices[0] if getattr(response, "choices", None) else None
            message = getattr(choice, "message", None) if choice else None

            if message is not None:
                # Standard string content
                if isinstance(getattr(message, "content", None), str):
                    extracted_text = message.content or ""
                else:
                    # Newer models may return an array of content parts
                    parts = getattr(message, "content", None)
                    if isinstance(parts, list):
                        collected: list[str] = []
                        for part in parts:
                            # part can be a pydantic-like object or dict
                            part_type = getattr(part, "type", None) or (
                                part.get("type") if isinstance(part, dict) else None
                            )
                            # Prefer explicit text fields
                            if part_type in {"output_text", "text"}:
                                text_val = getattr(part, "text", None) or (
                                    part.get("text") if isinstance(part, dict) else None
                                )
                                if isinstance(text_val, str):
                                    collected.append(text_val)
                            # Fallback: if the part itself is a string
                            elif isinstance(part, str):
                                collected.append(part)
                        extracted_text = "".join(collected)

                # As a final fallback, some models may include refusal/reasoning fields
                if not extracted_text:
                    refusal = getattr(message, "refusal", None)
                    if isinstance(refusal, str) and refusal:
                        extracted_text = refusal

                    reasoning = getattr(message, "reasoning", None)
                    if not extracted_text and isinstance(reasoning, str) and reasoning:
                        extracted_text = reasoning

        tokens_used = 0
        usage = getattr(response, "usage", None)
        if usage is not None:
            # Try total_tokens first, otherwise sum input/output tokens if available
            total_tokens = getattr(usage, "total_tokens", None)
            if isinstance(total_tokens, int):
                tokens_used = total_tokens
            else:
                input_tokens = getattr(usage, "prompt_tokens", None) or getattr(
                    usage, "input_tokens", None
                )
                output_tokens = getattr(usage, "completion_tokens", None) or getattr(
                    usage, "output_tokens", None
                )
                if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                    tokens_used = input_tokens + output_tokens

        request_id = getattr(response, "id", None)

        return extracted_text or "", tokens_used, request_id

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
            
            # Get safe token limits for this specific model
            token_limits = get_safe_token_limits(model, request.max_tokens)
            max_tokens = request.max_tokens or token_limits["recommended_output"]

            self.logger.info(f"Starting streaming generation with model: {model} (max_tokens: {max_tokens})")

            # Streaming implementation varies by model family
            if isinstance(model, str) and model.startswith("gpt-5"):
                # Responses API streaming
                input_segments: List[str] = []
                for m in messages:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    input_segments.append(f"{role.title()}: {content}")
                input_text = "\n".join(input_segments)

                reasoning_param = None
                if request.reasoning_effort and self._supports_reasoning_effort(model):
                    reasoning_param = {"effort": request.reasoning_effort}

                with self.client.responses.stream(
                    model=model,
                    input=input_text,
                    max_output_tokens=max_tokens,
                    **({"reasoning": reasoning_param} if reasoning_param else {}),
                ) as stream:
                    for event in stream:
                        # Emit only output text deltas
                        if getattr(event, "type", "") == "response.output_text.delta":
                            delta = getattr(event, "delta", None)
                            if isinstance(delta, str) and delta:
                                yield delta
                    stream.close()
            else:
                # Chat Completions streaming
                stream_params = {
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "max_tokens": max_tokens,
                }
                if self._supports_temperature(model):
                    stream_params["temperature"] = request.temperature

                stream = self.client.chat.completions.create(**stream_params)

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
