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
Test LLM Provider Architecture

Tests the provider-agnostic LLM integration system.
This tests the OpenAI provider architecture.
"""

from unittest.mock import Mock, patch

from src.core.llm_providers.base import (
    ContentType,
    GenerationRequest,
    ProviderCapabilities,
    ProviderType,
)
from src.core.llm_providers.factory import (
    get_api_key_manager,
    get_available_providers,
)
from src.core.llm_providers.openai_provider import OpenAIProvider


class TestProviderArchitecture:
    """Test cases for the provider-agnostic architecture."""

    def test_get_available_providers(self):
        """Test provider availability detection."""
        available = get_available_providers()

        assert isinstance(available, dict)
        assert ProviderType.OPENAI in available

        # Should return boolean values
        for provider_type, is_available in available.items():
            assert isinstance(is_available, bool)

    def test_api_key_manager(self):
        """Test API key management functionality."""
        key_manager = get_api_key_manager()

        # Test getting API keys (should handle missing keys gracefully)
        openai_key = key_manager.get_api_key("openai")

        # Should return string or None
        assert openai_key is None or isinstance(openai_key, str)

    @patch("src.core.llm_providers.openai_provider.OpenAI")
    def test_openai_provider_initialization(self, mock_openai_class):
        """Test OpenAI provider initialization."""
        # Mock the OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(api_key="test_key")

        assert provider.provider_type == ProviderType.OPENAI
        assert provider.is_available()
        assert isinstance(provider.capabilities, ProviderCapabilities)
        assert provider.get_default_model() == "gpt-5-mini"

    @patch("src.core.llm_providers.openai_provider.OpenAI")
    def test_openai_provider_content_generation(self, mock_openai_class):
        """Test content generation with OpenAI provider."""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated content"
        mock_response.usage.total_tokens = 100
        mock_response.id = "test_request_id"

        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(api_key="test_key")

        request = GenerationRequest(
            prompt="Test prompt",
            content_type=ContentType.GENERAL_RESPONSE,
            max_tokens=150,
            temperature=0.7,
        )

        response = provider.generate_content(request)

        assert response.success
        assert response.content == "Generated content"
        assert response.provider_used == "openai"
        assert response.tokens_used == 100
        assert response.request_id == "test_request_id"

    @patch("src.core.llm_providers.openai_provider.OpenAI")
    def test_cover_letter_generation(self, mock_openai_class):
        """Test cover letter generation functionality."""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "Dear Hiring Manager, I am writing to express my interest..."
        )
        mock_response.usage.total_tokens = 250
        mock_response.id = "cover_letter_id"

        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(api_key="test_key")

        response = provider.generate_cover_letter(
            job_description="Software Engineer position",
            company_name="Tech Corp",
            position_title="Software Engineer",
            candidate_context={
                "experience_years": 5,
                "key_skills": ["Python", "JavaScript"],
            },
        )

        assert response.success
        assert "Dear Hiring Manager" in response.content
        assert response.tokens_used == 250

    @patch("src.core.llm_providers.openai_provider.OpenAI")
    def test_interview_question_answering(self, mock_openai_class):
        """Test interview question answering functionality."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "In my previous role, I faced a challenging situation..."
        )
        mock_response.usage.total_tokens = 180
        mock_response.id = "interview_id"

        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(api_key="test_key")

        response = provider.answer_interview_question(
            question="Tell me about a challenging project you worked on.",
            candidate_context={
                "experience_years": 5,
                "key_skills": ["Problem Solving", "Leadership"],
            },
        )

        assert response.success
        assert "challenging situation" in response.content
        assert response.tokens_used == 180

    @patch("src.core.llm_providers.openai_provider.OpenAI")
    def test_content_refinement(self, mock_openai_class):
        """Test content refinement functionality."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "Refined and improved content with better clarity..."
        )
        mock_response.usage.total_tokens = 120
        mock_response.id = "refinement_id"

        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(api_key="test_key")

        response = provider.refine_content(
            original_content="Original content that needs improvement.",
            refinement_goals="Make it more professional and specific.",
        )

        assert response.success
        assert "Refined and improved" in response.content
        assert response.tokens_used == 120

    @patch("src.core.llm_providers.openai_provider.OpenAI")
    def test_error_handling(self, mock_openai_class):
        """Test error handling in provider."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock API error
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        provider = OpenAIProvider(api_key="test_key")

        request = GenerationRequest(
            prompt="Test prompt", content_type=ContentType.GENERAL_RESPONSE
        )

        response = provider.generate_content(request)

        assert not response.success
        assert "API Error" in response.error

    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        provider = OpenAIProvider(api_key="test_key")

        rate_status = provider.get_rate_limit_status()

        assert isinstance(rate_status, dict)
        assert "requests_made" in rate_status
        assert "max_requests_per_minute" in rate_status
        assert "reset_time" in rate_status

    def test_provider_factory_with_no_keys(self):
        """Test provider factory when no API keys are available."""
        # Clear all environment variables and mock the key manager to return None
        env_vars_to_clear = ["OPENAI_API_KEY"]

        with patch.dict("os.environ", dict.fromkeys(env_vars_to_clear, ""), clear=False):
            with patch(
                "src.core.llm_providers.factory.get_api_key_manager"
            ) as mock_key_manager:
                # Mock key manager to return None for all providers
                mock_manager = Mock()
                mock_manager.get_api_key.return_value = None
                mock_manager.list_configured_providers.return_value = {
                    "openai": {"configured": False, "source": "none"},
                }
                mock_key_manager.return_value = mock_manager

                # Test that no providers are available (except Ollama which doesn't need API keys)
                available = get_available_providers()
                # Only Ollama should be available since it doesn't require API keys
                assert available.get(ProviderType.OLLAMA, False), "Ollama should be available"
                # Other providers should not be available without API keys
                other_providers = [ProviderType.OPENAI, ProviderType.MISTRAL, ProviderType.NOVITA]
                for provider in other_providers:
                    assert not available.get(provider, False), f"{provider.value} should not be available without API key"

    @patch("src.core.llm_providers.factory.get_api_key_manager")
    def test_provider_selection_priority(self, mock_key_manager):
        """Test that provider selection works correctly with OpenAI only."""
        # Mock API key manager to return OpenAI key (current default priority)
        mock_manager = Mock()
        mock_manager.get_api_key.side_effect = lambda provider, override=None: (
            "test_key" if provider == "openai" else None
        )
        mock_key_manager.return_value = mock_manager

        with patch(
            "src.core.llm_providers.openai_provider.OpenAIProvider.is_available",
            return_value=True,
        ):
            # Test that OpenAI is available when configured
            available = get_available_providers()
            assert available.get(ProviderType.OPENAI, False), "OpenAI should be available"

    def test_generation_request_validation(self):
        """Test GenerationRequest validation."""
        # Valid request
        request = GenerationRequest(
            prompt="Test prompt",
            content_type=ContentType.GENERAL_RESPONSE,
            max_tokens=100,
            temperature=0.7,
        )

        assert request.prompt == "Test prompt"
        assert request.content_type == ContentType.GENERAL_RESPONSE
        assert request.max_tokens == 100
        assert request.temperature == 0.7

    def test_provider_capabilities(self):
        """Test provider capabilities structure."""
        provider = OpenAIProvider(api_key="test_key")
        capabilities = provider.capabilities

        assert isinstance(capabilities.max_tokens, int)
        assert isinstance(capabilities.supports_streaming, bool)
        assert isinstance(capabilities.supports_function_calling, bool)
        assert isinstance(capabilities.rate_limit_per_minute, int)
        assert capabilities.models is not None
        assert len(capabilities.models) > 0
