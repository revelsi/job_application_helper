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
Tests for the PromptManager.

Tests prompt template management and dynamic prompt generation.
"""

from pathlib import Path
import tempfile

import pytest

from src.core.prompts import (
    PromptManager,
    PromptTemplate,
    PromptType,
    get_prompt_manager,
)


class TestPromptManager:
    """Test cases for PromptManager."""

    def test_init_loads_default_prompts(self):
        """Test that initialization loads default prompts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PromptManager(custom_prompts_path=Path(temp_dir) / "custom.json")

            # Should have all default prompt types
            for prompt_type in PromptType:
                template = manager.get_prompt_template(prompt_type)
                assert isinstance(template, PromptTemplate)
                assert template.system_prompt
                assert template.user_prompt_template
                assert template.description

    def test_get_prompt_template(self):
        """Test retrieving prompt templates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PromptManager(custom_prompts_path=Path(temp_dir) / "custom.json")

            # Test cover letter template
            template = manager.get_prompt_template(PromptType.COVER_LETTER)
            assert "career strategist" in template.system_prompt.lower()
            assert "cover letter" in template.system_prompt.lower()
            assert "{company_name}" in template.user_prompt_template
            assert "{position_title}" in template.user_prompt_template

    def test_build_system_prompt_basic(self):
        """Test building basic system prompt without context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PromptManager(custom_prompts_path=Path(temp_dir) / "custom.json")

            prompt = manager.build_system_prompt(PromptType.COVER_LETTER)
            assert "career strategist" in prompt.lower()
            assert "cover letter" in prompt.lower()

    def test_build_system_prompt_with_context(self):
        """Test building system prompt with dynamic context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PromptManager(custom_prompts_path=Path(temp_dir) / "custom.json")

            context = {
                "industry": "tech",
                "experience_level": "mid",
                "company_size": "startup",
            }

            prompt = manager.build_system_prompt(PromptType.COVER_LETTER, context)

            # Should include original prompt
            assert "career strategist" in prompt.lower()

            # Should include industry guidance
            assert "technical skills" in prompt.lower()
            assert "innovation" in prompt.lower()

            # Should include experience level guidance
            assert "leadership potential" in prompt.lower()

            # Should include company size guidance
            assert "adaptability" in prompt.lower()
            assert "entrepreneurial" in prompt.lower()

    def test_build_user_prompt(self):
        """Test building user prompt with variable substitution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PromptManager(custom_prompts_path=Path(temp_dir) / "custom.json")

            variables = {
                "company_name": "TechCorp",
                "position_title": "Senior Developer",
                "job_description": "Python development role",
                "candidate_context": "Experienced Python developer with 5 years of experience",
            }

            prompt = manager.build_user_prompt(PromptType.COVER_LETTER, variables)

            assert "TechCorp" in prompt
            assert "Senior Developer" in prompt
            assert "Python development role" in prompt

    def test_build_user_prompt_missing_variables(self):
        """Test building user prompt with missing variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PromptManager(custom_prompts_path=Path(temp_dir) / "custom.json")

            # Missing some variables
            variables = {"company_name": "TechCorp"}

            # Should not raise error, just return template with placeholders
            prompt = manager.build_user_prompt(PromptType.COVER_LETTER, variables)

            # Should show warning but still return prompt
            # Missing variables should remain as placeholders
            assert "{position_title}" in prompt
            assert "{job_description}" in prompt

    def test_save_and_load_custom_prompt(self):
        """Test saving and loading custom prompts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_path = Path(temp_dir) / "custom.json"
            manager = PromptManager(custom_prompts_path=custom_path)

            # Create custom template
            custom_template = PromptTemplate(
                system_prompt="Custom system prompt for testing",
                user_prompt_template="Custom user prompt: {test_var}",
                description="Test custom prompt",
                version="2.0",
                tags=["test", "custom"],
                context_variables=["test_var"],
            )

            # Save custom prompt
            success = manager.save_custom_prompt(
                PromptType.COVER_LETTER, custom_template
            )
            assert success
            assert custom_path.exists()

            # Create new manager instance to test loading
            manager2 = PromptManager(custom_prompts_path=custom_path)

            # Should load custom prompt instead of default
            loaded_template = manager2.get_prompt_template(PromptType.COVER_LETTER)
            assert loaded_template.system_prompt == "Custom system prompt for testing"
            assert loaded_template.version == "2.0"
            assert "test" in loaded_template.tags

    def test_list_available_prompts(self):
        """Test listing available prompts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PromptManager(custom_prompts_path=Path(temp_dir) / "custom.json")

            prompts_info = manager.list_available_prompts()

            # Should have all prompt types
            assert len(prompts_info) == len(PromptType)

            # Check structure
            for prompt_type in PromptType:
                info = prompts_info[prompt_type.value]
                assert "description" in info
                assert "version" in info
                assert "tags" in info
                assert "context_variables" in info
                assert "is_custom" in info
                assert "source" in info
                assert info["source"] == "default"  # All default initially

    def test_industry_guidance(self):
        """Test industry-specific guidance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PromptManager(custom_prompts_path=Path(temp_dir) / "custom.json")

            # Test tech industry
            tech_guidance = manager._get_industry_guidance("tech")
            assert "technical skills" in tech_guidance.lower()
            assert "innovation" in tech_guidance.lower()

            # Test finance industry
            finance_guidance = manager._get_industry_guidance("finance")
            assert "analytical" in finance_guidance.lower()
            assert "regulatory" in finance_guidance.lower()

            # Test unknown industry
            unknown_guidance = manager._get_industry_guidance("unknown_industry")
            assert unknown_guidance is None

    def test_experience_level_guidance(self):
        """Test experience level-specific guidance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PromptManager(custom_prompts_path=Path(temp_dir) / "custom.json")

            # Test entry level
            entry_guidance = manager._get_experience_level_guidance("entry")
            assert "education" in entry_guidance.lower()
            assert "enthusiasm" in entry_guidance.lower()

            # Test senior level
            senior_guidance = manager._get_experience_level_guidance("senior")
            assert "leadership" in senior_guidance.lower()
            assert "strategic" in senior_guidance.lower()

    def test_company_size_guidance(self):
        """Test company size-specific guidance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PromptManager(custom_prompts_path=Path(temp_dir) / "custom.json")

            # Test startup
            startup_guidance = manager._get_company_size_guidance("startup")
            assert "adaptability" in startup_guidance.lower()
            assert "entrepreneurial" in startup_guidance.lower()

            # Test large company
            large_guidance = manager._get_company_size_guidance("large")
            assert "process" in large_guidance.lower()
            assert "complex" in large_guidance.lower()


class TestPromptManagerSingleton:
    """Test the global prompt manager singleton."""

    def test_get_prompt_manager_singleton(self):
        """Test that get_prompt_manager returns the same instance."""
        manager1 = get_prompt_manager()
        manager2 = get_prompt_manager()

        assert manager1 is manager2
        assert isinstance(manager1, PromptManager)

    def test_prompt_manager_properties(self):
        """Test that the global manager has expected properties."""
        manager = get_prompt_manager()

        assert hasattr(manager, "get_prompt_template")
        assert hasattr(manager, "build_system_prompt")
        assert hasattr(manager, "build_user_prompt")
        assert hasattr(manager, "save_custom_prompt")
        assert hasattr(manager, "list_available_prompts")


if __name__ == "__main__":
    pytest.main([__file__])
