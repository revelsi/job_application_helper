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
Tests for the prompt system.

Tests for core prompt management functionality.
"""

import pytest

from src.core.prompts import (
    PromptManager,
    PromptType,
    get_prompt_manager,
    SYSTEM_PROMPT,
)


class TestPromptManager:
    """Test the prompt manager."""

    def test_init(self):
        """Test prompt manager initialization."""
        manager = PromptManager()
        assert manager is not None
        assert hasattr(manager, 'logger')

    def test_build_messages_basic(self):
        """Test basic message building."""
        manager = PromptManager()
        
        messages = manager.build_messages(
            PromptType.GENERAL_RESPONSE,
            "Hello, test message"
        )
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Hello, test message" in messages[1]["content"]
        assert "career advisor" in messages[0]["content"].lower()

    def test_build_messages_with_context(self):
        """Test message building with document context."""
        manager = PromptManager()
        
        messages = manager.build_messages(
            PromptType.COVER_LETTER,
            "Write a cover letter",
            context="I am a software engineer with 5 years experience"
        )
        
        assert len(messages) == 2
        assert "DOCUMENT CONTEXT:" in messages[1]["content"]
        assert "software engineer" in messages[1]["content"]
        assert "INSTRUCTIONS:" in messages[1]["content"]

    def test_build_messages_with_conversation_history(self):
        """Test message building with conversation history."""
        manager = PromptManager()
        
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
        
        messages = manager.build_messages(
            PromptType.GENERAL_RESPONSE,
            "Follow up question",
            conversation_history=history
        )
        
        assert len(messages) == 4  # system + 2 history + current user
        assert messages[1]["content"] == "Previous question"
        assert messages[2]["content"] == "Previous answer"

    def test_build_messages_cover_letter_instructions(self):
        """Test cover letter specific instructions."""
        manager = PromptManager()
        
        messages = manager.build_messages(
            PromptType.COVER_LETTER,
            "Write a cover letter",
            context="Some context"
        )
        
        user_content = messages[1]["content"]
        assert "professional cover letter" in user_content
        assert "quantified achievements" in user_content

    def test_build_messages_interview_instructions(self):
        """Test interview answer specific instructions."""
        manager = PromptManager()
        
        messages = manager.build_messages(
            PromptType.INTERVIEW_ANSWER,
            "Answer this question",
            context="Some context"
        )
        
        user_content = messages[1]["content"]
        assert "STAR method" in user_content
        assert "1-2 minutes" in user_content

    def test_build_prompt_legacy_format(self):
        """Test legacy prompt format building."""
        manager = PromptManager()
        
        prompt = manager.build_prompt(
            PromptType.GENERAL_RESPONSE,
            "Test message",
            context="Test context"
        )
        
        assert isinstance(prompt, str)
        assert "career advisor" in prompt
        assert "Test message" in prompt
        assert "Test context" in prompt

    def test_build_system_prompt(self):
        """Test system prompt building."""
        manager = PromptManager()
        
        system_prompt = manager.build_system_prompt(PromptType.COVER_LETTER)
        
        assert system_prompt == SYSTEM_PROMPT
        assert "career advisor" in system_prompt.lower()
        assert "STRICT CONTEXT ADHERENCE" in system_prompt

    def test_prompt_types_enum(self):
        """Test prompt types enum."""
        assert PromptType.COVER_LETTER.value == "cover_letter"
        assert PromptType.INTERVIEW_ANSWER.value == "interview_answer"
        assert PromptType.GENERAL_RESPONSE.value == "general_response"

    def test_conversation_history_truncation(self):
        """Test that conversation history is truncated to last 8 messages."""
        manager = PromptManager()
        
        # Create 10 messages
        history = []
        for i in range(10):
            history.extend([
                {"role": "user", "content": f"User message {i}"},
                {"role": "assistant", "content": f"Assistant message {i}"}
            ])
        
        messages = manager.build_messages(
            PromptType.GENERAL_RESPONSE,
            "Current question",
            conversation_history=history
        )
        
        # Should have: system + last 8 history + current user = 10 messages
        assert len(messages) == 10
        # First history message should be from message 6 (last 8 of 20 total)
        assert "User message 6" in messages[1]["content"]

    def test_no_context_instructions(self):
        """Test instructions when no context is provided."""
        manager = PromptManager()
        
        messages = manager.build_messages(
            PromptType.GENERAL_RESPONSE,
            "Test question"
        )
        
        user_content = messages[1]["content"]
        assert "No specific documents available" in user_content
        assert "Provide general guidance" in user_content


class TestPromptManagerSingleton:
    """Test the global prompt manager singleton."""

    def test_get_prompt_manager_singleton(self):
        """Test that get_prompt_manager returns the same instance."""
        manager1 = get_prompt_manager()
        manager2 = get_prompt_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, PromptManager)

    def test_system_prompt_constant(self):
        """Test that SYSTEM_PROMPT is properly defined."""
        assert SYSTEM_PROMPT is not None
        assert len(SYSTEM_PROMPT) > 100
        assert "career advisor" in SYSTEM_PROMPT.lower()
        assert "STRICT CONTEXT ADHERENCE" in SYSTEM_PROMPT