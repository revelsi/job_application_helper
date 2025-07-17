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
Base LLM Provider Abstract Class.

Defines the common interface that all LLM providers must implement.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.core.prompts import PromptType, get_prompt_manager
from src.utils.logging import get_logger


class ProviderType(Enum):
    """Available LLM provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    AUTO = "auto"  # Automatically select based on available API keys


class ContentType(Enum):
    """Types of content that can be generated."""

    COVER_LETTER = "cover_letter"
    INTERVIEW_ANSWER = "interview_answer"
    GENERAL_RESPONSE = "general_response"
    CONTENT_REFINEMENT = "content_refinement"


@dataclass
class GenerationRequest:
    """Request for content generation."""

    prompt: str
    content_type: ContentType
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    model: Optional[str] = None  # Provider-specific model name
    context: Optional[Dict[str, Any]] = None


@dataclass
class GenerationResponse:
    """Response from content generation."""

    content: str
    model_used: str
    provider_used: str
    tokens_used: int
    generation_time: float
    success: bool
    error: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class ProviderCapabilities:
    """Capabilities and limits of an LLM provider."""

    max_tokens: int
    supports_streaming: bool
    supports_function_calling: bool
    rate_limit_per_minute: int
    cost_per_1k_tokens: Optional[float] = None
    models: List[str] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM provider.

        Args:
            api_key: API key for the provider. If None, uses environment variable.
        """
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.api_key = api_key
        self.prompt_manager = get_prompt_manager()
        self._rate_limit_info = {
            "requests_made": 0,
            "last_request_time": datetime.now() - timedelta(minutes=1),
            "reset_time": datetime.now(),
        }

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities and limits."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available (API key set, etc.)."""
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass

    @abstractmethod
    def _build_messages(self, request: GenerationRequest) -> List[Dict[str, Any]]:
        """Build messages in provider-specific format."""
        pass

    @abstractmethod
    def _make_api_call(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        """Make the actual API call to the provider."""
        pass

    @abstractmethod
    def _parse_response(
        self, response: Dict[str, Any]
    ) -> tuple[str, int, Optional[str]]:
        """Parse the provider response into (content, tokens_used, request_id)."""
        pass

    @abstractmethod
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
        pass

    def _get_system_prompt(
        self, content_type: ContentType, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Get system prompt based on content type using PromptManager."""
        # Map ContentType to PromptType
        content_to_prompt_map = {
            ContentType.COVER_LETTER: PromptType.COVER_LETTER,
            ContentType.INTERVIEW_ANSWER: PromptType.INTERVIEW_ANSWER,
            ContentType.CONTENT_REFINEMENT: PromptType.CONTENT_REFINEMENT,
            ContentType.GENERAL_RESPONSE: PromptType.GENERAL_RESPONSE,
        }

        prompt_type = content_to_prompt_map.get(
            content_type, PromptType.GENERAL_RESPONSE
        )
        return self.prompt_manager.build_system_prompt(prompt_type, context)

    def _check_rate_limit(self) -> bool:
        """Check if we can make a request within rate limits."""
        now = datetime.now()

        # Reset counter if a minute has passed
        if now >= self._rate_limit_info["reset_time"]:
            self._rate_limit_info["requests_made"] = 0
            self._rate_limit_info["reset_time"] = now + timedelta(minutes=1)

        # Check if we're within limits
        max_requests = self.capabilities.rate_limit_per_minute
        if self._rate_limit_info["requests_made"] >= max_requests:
            self.logger.warning("Rate limit exceeded, request blocked")
            return False

        return True

    def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limits."""
        if not self._check_rate_limit():
            wait_time = (
                self._rate_limit_info["reset_time"] - datetime.now()
            ).total_seconds()
            if wait_time > 0:
                self.logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)

    def generate_content(
        self, request: GenerationRequest, timeout: float = 30.0
    ) -> GenerationResponse:
        """
        Generate content using this provider.

        Args:
            request: Generation request with prompt and parameters.
            timeout: Request timeout in seconds.

        Returns:
            Generation response with content and metadata.
        """
        start_time = time.time()

        try:
            # Check availability
            if not self.is_available():
                raise ValueError(
                    f"{self.provider_type.value} provider is not available"
                )

            # Rate limiting
            self._wait_for_rate_limit()

            # Build messages
            messages = self._build_messages(request)

            # Determine model and parameters
            model = request.model or self.get_default_model()
            max_tokens = request.max_tokens or self.capabilities.max_tokens

            self.logger.debug(
                f"Generating {request.content_type.value} content with {model}"
            )

            # Make API call
            response = self._make_api_call(
                messages, model, max_tokens, request.temperature
            )

            # Update rate limiting
            self._rate_limit_info["requests_made"] += 1
            self._rate_limit_info["last_request_time"] = datetime.now()

            # Parse response
            content, tokens_used, request_id = self._parse_response(response)

            generation_time = time.time() - start_time

            self.logger.info(
                f"Content generated successfully in {generation_time:.2f}s, "
                f"tokens: {tokens_used}"
            )

            return GenerationResponse(
                content=content,
                model_used=model,
                provider_used=self.provider_type.value,
                tokens_used=tokens_used,
                generation_time=generation_time,
                success=True,
                request_id=request_id,
            )

        except Exception as e:
            generation_time = time.time() - start_time
            error_msg = f"Content generation failed: {str(e)}"
            self.logger.error(error_msg)

            return GenerationResponse(
                content="",
                model_used=request.model or "unknown",
                provider_used=self.provider_type.value,
                tokens_used=0,
                generation_time=generation_time,
                success=False,
                error=error_msg,
            )

    def generate_cover_letter(
        self,
        job_description: str,
        company_name: str,
        position_title: str,
        candidate_context: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> GenerationResponse:
        """Generate a tailored cover letter."""
        # Use prompt template for cover letter
        variables = {
            "company_name": company_name,
            "position_title": position_title,
            "job_description": job_description,
        }
        prompt = self.prompt_manager.build_user_prompt(
            PromptType.COVER_LETTER, variables
        )

        context = {
            "company_name": company_name,
            "position_title": position_title,
            **(candidate_context or {}),
        }

        request = GenerationRequest(
            prompt=prompt,
            content_type=ContentType.COVER_LETTER,
            model=model,
            context=context,
        )

        return self.generate_content(request)

    def answer_interview_question(
        self,
        question: str,
        candidate_context: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> GenerationResponse:
        """Generate an answer to an interview question."""
        # Use prompt template for interview answer
        variables = {"question": question}
        prompt = self.prompt_manager.build_user_prompt(
            PromptType.INTERVIEW_ANSWER, variables
        )

        request = GenerationRequest(
            prompt=prompt,
            content_type=ContentType.INTERVIEW_ANSWER,
            model=model,
            context=candidate_context,
        )

        return self.generate_content(request)

    def refine_content(
        self, original_content: str, refinement_goals: str, model: Optional[str] = None
    ) -> GenerationResponse:
        """Refine existing content based on specific goals."""
        # Use prompt template for content refinement
        variables = {
            "refinement_goals": refinement_goals,
            "original_content": original_content,
        }
        prompt = self.prompt_manager.build_user_prompt(
            PromptType.CONTENT_REFINEMENT, variables
        )

        request = GenerationRequest(
            prompt=prompt, content_type=ContentType.CONTENT_REFINEMENT, model=model
        )

        return self.generate_content(request)

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limiting status."""
        return {
            "provider": self.provider_type.value,
            "requests_made": self._rate_limit_info["requests_made"],
            "max_requests_per_minute": self.capabilities.rate_limit_per_minute,
            "reset_time": self._rate_limit_info["reset_time"].isoformat(),
            "can_make_request": self._check_rate_limit(),
        }
