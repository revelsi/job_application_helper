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
Ollama LLM Provider Implementation.

Provides integration with local Ollama models using the unified LLM provider interface.
Supports local model inference without requiring API keys.
"""

from collections.abc import AsyncGenerator
import json
from typing import Any, Dict, List, Optional

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

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


class OllamaProvider(LLMProvider):
    """Ollama provider implementation for local model inference."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Ollama provider.

        Args:
            api_key: Not used for Ollama (local inference), kept for interface compatibility.
        """
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx package is required for Ollama provider. Install with: pip install httpx"
            )

        super().__init__(api_key)
        self.settings = get_settings()
        self.base_url = self.settings.ollama_base_url.rstrip("/")
        self.timeout = self.settings.ollama_timeout
        self._client = None
        self._available_models = None

    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.OLLAMA

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return Ollama provider capabilities based on model configurations."""
        models = get_models_for_provider("ollama")
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
            max_tokens=4096,
            supports_streaming=True,
            supports_function_calling=False,
            rate_limit_per_minute=1000,
            cost_per_1k_tokens=0.0,
            models=model_names,
        )

    def is_available(self) -> bool:
        """Check if Ollama provider is available and has required models."""
        try:
            # First check if Ollama service is running
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    return False
                
                # Get available models
                data = response.json()
                available_models = data.get("models", [])
                available_model_names = [m["name"] for m in available_models]
                
                # Check if at least one of the required models is available
                required_models = ["gemma3:1b", "llama3.2:1b", "hf.co/LiquidAI/LFM2-1.2B-GGUF:Q4_K_M"]
                has_required_model = any(model in available_model_names for model in required_models)
                
                if not has_required_model:
                    self.logger.info("Ollama service is running but no required models are downloaded")
                    return False
                
                return True
        except Exception as e:
            self.logger.debug(f"Ollama not available: {e}")
            return False

    def get_default_model(self) -> str:
        """Get the default Ollama model."""
        # Try to get the first available model, fallback to gemma3:1b
        try:
            models = self._get_available_models()
            if models:
                return models[0]["name"]
        except Exception:
            pass
        return "gemma3:1b"

    @property
    def client(self) -> httpx.Client:
        """Get HTTP client for Ollama API."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url, timeout=httpx.Timeout(self.timeout)
            )
        return self._client

    def _get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from Ollama."""
        if self._available_models is None:
            try:
                response = self.client.get("/api/tags")
                response.raise_for_status()
                data = response.json()
                self._available_models = data.get("models", [])
            except Exception as e:
                self.logger.debug(f"Failed to get Ollama models: {e}")
                self._available_models = []
        return self._available_models

    def _check_model_available(self, model: str) -> bool:
        """Check if a specific model is available locally."""
        try:
            available_models = self._get_available_models()
            return any(m["name"] == model for m in available_models)
        except Exception:
            return False

    async def _download_model(self, model: str) -> bool:
        """
        Download a model from Ollama library.
        
        Args:
            model: Model name to download
            
        Returns:
            True if download was successful, False otherwise
        """
        try:
            self.logger.info(f"Starting download of model: {model}")
            
            payload = {
                "model": model,
                "stream": False  # Get single response instead of stream
            }
            
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:  # 5 minute timeout for downloads
                response = await client.post(f"{self.base_url}/api/pull", json=payload)
                response.raise_for_status()
                
                result = response.json()
                if result.get("status") == "success":
                    self.logger.info(f"Successfully downloaded model: {model}")
                    # Clear cached models to refresh the list
                    self._available_models = None
                    return True
                self.logger.error(f"Failed to download model {model}: {result}")
                return False
                    
        except Exception as e:
            self.logger.error(f"Error downloading model {model}: {e}")
            return False

    def _ensure_model_available(self, model: str) -> bool:
        """
        Ensure a model is available, downloading it if necessary.
        
        Args:
            model: Model name to check/ensure
            
        Returns:
            True if model is available, False otherwise
        """
        if self._check_model_available(model):
            return True
            
        # Model not available, try to download it
        import asyncio
        try:
            # Run the async download in a sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(self._download_model(model))
            loop.close()
            return success
        except Exception as e:
            self.logger.error(f"Failed to ensure model {model} is available: {e}")
            return False





    def _make_api_call(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make API call to Ollama."""
        try:
            # Note: reasoning_effort and verbosity parameters are ignored as Ollama doesn't support these features
            if reasoning_effort:
                self.logger.debug(f"Ignoring reasoning_effort '{reasoning_effort}' - not supported by Ollama")
            if verbosity:
                self.logger.debug(f"Ignoring verbosity '{verbosity}' - not supported by Ollama")
            # Get safe token limits for this specific model
            token_limits = get_safe_token_limits(model)
            safe_max_tokens = min(max_tokens, token_limits["recommended_output"])
            
            self.logger.debug(f"Ollama API call: {model} with {safe_max_tokens} max tokens")
            
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": safe_max_tokens,
                },
            }

            response = self.client.post("/api/chat", json=payload)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            self.logger.error(f"Ollama API call failed: {e}")
            raise

    def _parse_response(
        self, response: Dict[str, Any]
    ) -> tuple[str, int, Optional[str]]:
        """Parse Ollama response into (content, tokens_used, request_id)."""
        content = response.get("message", {}).get("content", "")
        
        # Ollama doesn't provide token counts in the same way
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        tokens_used = len(content) // 4 if content else 0
        
        # Ollama doesn't provide request IDs
        request_id = None
        
        return content, tokens_used, request_id

    async def generate_content_stream(
        self, request: GenerationRequest, timeout: float = 300.0
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
                raise ValueError("Ollama provider is not available")

            # Build messages and parameters
            messages = self._build_messages(request)
            model = request.model or self.get_default_model()
            
            # Check if model is available
            if not self._check_model_available(model):
                raise ValueError(
                    f"Model '{model}' is not available locally. "
                    f"Please download it first using: ollama pull {model}"
                )
            
            # Get safe token limits for this specific model
            token_limits = get_safe_token_limits(model, request.max_tokens)
            max_tokens = request.max_tokens or token_limits["recommended_output"]

            self.logger.info(f"Starting Ollama streaming with model: {model} (max_tokens: {max_tokens})")
            self.logger.debug(f"Messages: {messages}")
            self.logger.debug(f"Temperature: {request.temperature}")

            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": max_tokens,
                },
            }
            self.logger.debug(f"Ollama payload: {payload}")

            # Make streaming API call
            self.logger.debug(f"Making streaming request to: {self.base_url}/api/chat")
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                self.logger.debug("Created async client, starting stream...")
                async with client.stream(
                    "POST", f"{self.base_url}/api/chat", json=payload
                ) as response:
                    self.logger.debug(f"Got response status: {response.status_code}")
                    response.raise_for_status()
                    self.logger.debug("Response status OK, starting to read lines...")
                    
                    line_count = 0
                    async for line in response.aiter_lines():
                        line_count += 1
                        self.logger.debug(f"Received line {line_count}: {line[:100]}...")
                        if line.strip():
                            try:
                                data = json.loads(line)
                                # Check if this is a content chunk
                                if "message" in data and "content" in data["message"]:
                                    content = data["message"]["content"]
                                    if content:
                                        self.logger.debug(f"Yielding content: {content[:50]}...")
                                        yield content
                                # Check if this is the final message
                                elif data.get("done", False):
                                    self.logger.debug("Received done signal, breaking...")
                                    break
                            except json.JSONDecodeError as e:
                                self.logger.debug(f"Failed to parse JSON line: {line.strip()}, error: {e}")
                                continue
                    
                    self.logger.info(f"Ollama streaming completed. Processed {line_count} lines.")

        except Exception as e:
            self.logger.error(f"Ollama streaming generation failed: {e}")
            raise


def get_ollama_provider(api_key: Optional[str] = None) -> OllamaProvider:
    """
    Factory function to create Ollama provider.

    Args:
        api_key: Not used for Ollama, kept for interface compatibility.

    Returns:
        Configured Ollama provider.
    """
    return OllamaProvider(api_key=api_key)
