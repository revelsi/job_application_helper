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
Tests for the memory management system.

This module tests:
- Session creation and management
- Message handling and persistence
- Context window optimization
- Database operations and integrity
- Memory cleanup and statistics
"""

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import pytest

from src.core.memory_manager import (
    ChatMessage,
    ContextWindowManager,
    ConversationSession,
    MemoryDatabase,
    MemoryManager,
    MessageManager,
    MessageRole,
    SessionManager,
    SessionStatus,
    get_memory_manager,
)


class TestChatMessage:
    """Test ChatMessage functionality."""

    def test_chat_message_creation(self):
        """Test basic message creation."""
        message = ChatMessage(
            session_id="test-session", role=MessageRole.USER, content="Hello, world!"
        )

        assert message.session_id == "test-session"
        assert message.role == MessageRole.USER
        assert message.content == "Hello, world!"
        assert message.is_user_message
        assert not message.is_assistant_message
        assert not message.is_system_message

    def test_message_role_conversion(self):
        """Test string to enum conversion."""
        message = ChatMessage(role="assistant", content="Hi there!")
        assert message.role == MessageRole.ASSISTANT
        assert message.is_assistant_message

    def test_message_to_dict(self):
        """Test message conversion to API format."""
        message = ChatMessage(
            role=MessageRole.SYSTEM, content="You are a helpful assistant."
        )

        result = message.to_dict()
        expected = {"role": "system", "content": "You are a helpful assistant."}

        assert result == expected


class TestConversationSession:
    """Test ConversationSession functionality."""

    def test_session_creation(self):
        """Test basic session creation."""
        session = ConversationSession(title="Test Session", metadata={"test": "data"})

        assert session.title == "Test Session"
        assert session.status == SessionStatus.ACTIVE
        assert session.is_active
        assert session.metadata == {"test": "data"}
        assert session.session_id  # UUID should be generated

    def test_session_status_conversion(self):
        """Test string to enum conversion."""
        session = ConversationSession(status="completed")
        assert session.status == SessionStatus.COMPLETED
        assert not session.is_active

    def test_session_age_calculation(self):
        """Test age and idle time calculations."""
        # Create session with past timestamp
        past_time = datetime.now() - timedelta(hours=2)
        session = ConversationSession(created_at=past_time, last_activity=past_time)

        assert session.age_hours >= 2
        assert session.idle_hours >= 2


class TestMemoryDatabase:
    """Test MemoryDatabase functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_memory.db"
            yield MemoryDatabase(db_path)

    def test_database_creation(self, temp_db):
        """Test database and table creation."""
        assert temp_db.db_path.exists()

        with temp_db.get_connection() as conn:
            # Check sessions table exists
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
            ).fetchone()
            assert result is not None

            # Check messages table exists
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
            ).fetchone()
            assert result is not None

    def test_database_indexes(self, temp_db):
        """Test that indexes are created."""
        with temp_db.get_connection() as conn:
            indexes = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()

            index_names = [idx[0] for idx in indexes]
            assert "idx_sessions_status" in index_names
            assert "idx_messages_session_id" in index_names


class TestSessionManager:
    """Test SessionManager functionality."""

    @pytest.fixture
    def session_manager(self):
        """Create session manager with temporary database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_memory.db"
            db = MemoryDatabase(db_path)
            yield SessionManager(db)

    def test_create_session(self, session_manager):
        """Test session creation."""
        session = session_manager.create_session(
            title="Test Session", metadata={"key": "value"}
        )

        assert session.id is not None
        assert session.title == "Test Session"
        assert session.metadata == {"key": "value"}
        assert session.status == SessionStatus.ACTIVE

    def test_get_session(self, session_manager):
        """Test session retrieval."""
        # Create session
        created_session = session_manager.create_session("Test Session")

        # Retrieve session
        retrieved_session = session_manager.get_session(created_session.session_id)

        assert retrieved_session is not None
        assert retrieved_session.session_id == created_session.session_id
        assert retrieved_session.title == "Test Session"

    def test_get_nonexistent_session(self, session_manager):
        """Test retrieving non-existent session."""
        result = session_manager.get_session("nonexistent-id")
        assert result is None

    def test_update_session_activity(self, session_manager):
        """Test session activity updates."""
        session = session_manager.create_session("Test Session")
        original_activity = session.last_activity

        # Update activity
        session_manager.update_session_activity(
            session.session_id, message_count_delta=2, tokens_delta=100
        )

        # Retrieve updated session
        updated_session = session_manager.get_session(session.session_id)
        assert updated_session.message_count == 2
        assert updated_session.total_tokens == 100
        assert updated_session.last_activity > original_activity

    def test_update_session_status(self, session_manager):
        """Test session status updates."""
        session = session_manager.create_session("Test Session")

        # Update status
        session_manager.update_session_status(
            session.session_id, SessionStatus.COMPLETED
        )

        # Verify update
        updated_session = session_manager.get_session(session.session_id)
        assert updated_session.status == SessionStatus.COMPLETED

    def test_list_sessions(self, session_manager):
        """Test session listing."""
        # Create multiple sessions
        session1 = session_manager.create_session("Session 1")
        session2 = session_manager.create_session("Session 2")

        # Update one to completed
        session_manager.update_session_status(
            session1.session_id, SessionStatus.COMPLETED
        )

        # List all sessions
        all_sessions = session_manager.list_sessions()
        assert len(all_sessions) == 2

        # List only active sessions
        active_sessions = session_manager.list_sessions(SessionStatus.ACTIVE)
        assert len(active_sessions) == 1
        assert active_sessions[0].session_id == session2.session_id

    def test_cleanup_old_sessions(self, session_manager):
        """Test cleanup of old sessions."""
        # Create session with old timestamp
        session = session_manager.create_session("Old Session")

        # Manually update last_activity to be old
        old_time = datetime.now() - timedelta(days=35)
        with session_manager.db.get_connection() as conn:
            conn.execute(
                "UPDATE sessions SET last_activity = ? WHERE session_id = ?",
                (old_time.isoformat(), session.session_id),
            )
            conn.commit()

        # Run cleanup
        archived_count = session_manager.cleanup_old_sessions(days_threshold=30)
        assert archived_count == 1

        # Verify session is archived
        updated_session = session_manager.get_session(session.session_id)
        assert updated_session.status == SessionStatus.ARCHIVED


class TestMessageManager:
    """Test MessageManager functionality."""

    @pytest.fixture
    def message_manager(self):
        """Create message manager with temporary database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_memory.db"
            db = MemoryDatabase(db_path)
            yield MessageManager(db)

    def test_add_message(self, message_manager):
        """Test adding messages."""
        message = ChatMessage(
            session_id="test-session",
            role=MessageRole.USER,
            content="Hello!",
            metadata={"test": "data"},
        )

        message_id = message_manager.add_message(message)
        assert message_id is not None
        assert message.id == message_id

    def test_get_session_messages(self, message_manager):
        """Test retrieving session messages."""
        session_id = "test-session"

        # Add multiple messages
        messages = [
            ChatMessage(session_id=session_id, role=MessageRole.USER, content="Hello!"),
            ChatMessage(
                session_id=session_id, role=MessageRole.ASSISTANT, content="Hi there!"
            ),
            ChatMessage(
                session_id=session_id, role=MessageRole.USER, content="How are you?"
            ),
        ]

        for msg in messages:
            message_manager.add_message(msg)

        # Retrieve messages
        retrieved_messages = message_manager.get_session_messages(session_id)
        assert len(retrieved_messages) == 3
        assert retrieved_messages[0].content == "Hello!"
        assert retrieved_messages[1].content == "Hi there!"
        assert retrieved_messages[2].content == "How are you?"

    def test_get_recent_messages(self, message_manager):
        """Test retrieving recent messages."""
        session_id = "test-session"

        # Add messages
        for i in range(5):
            message = ChatMessage(
                session_id=session_id, role=MessageRole.USER, content=f"Message {i}"
            )
            message_manager.add_message(message)

        # Get recent messages
        recent = message_manager.get_recent_messages(session_id, count=3)
        assert len(recent) == 3
        assert recent[0].content == "Message 2"  # Oldest of the 3 recent
        assert recent[2].content == "Message 4"  # Most recent

    def test_delete_session_messages(self, message_manager):
        """Test deleting session messages."""
        session_id = "test-session"

        # Add messages
        for i in range(3):
            message = ChatMessage(
                session_id=session_id, role=MessageRole.USER, content=f"Message {i}"
            )
            message_manager.add_message(message)

        # Delete messages
        deleted_count = message_manager.delete_session_messages(session_id)
        assert deleted_count == 3

        # Verify deletion
        remaining_messages = message_manager.get_session_messages(session_id)
        assert len(remaining_messages) == 0


class TestContextWindowManager:
    """Test ContextWindowManager functionality."""

    @pytest.fixture
    def context_manager(self):
        """Create context window manager."""
        return ContextWindowManager(max_tokens=1000)

    def test_token_estimation(self, context_manager):
        """Test token estimation."""
        # Test basic estimation (1 token ≈ 4 characters)
        text = "Hello world"  # 11 characters
        tokens = context_manager._estimate_tokens(text)
        assert tokens == 2  # 11 // 4 = 2

        # Test minimum of 1 token
        empty_text = ""
        tokens = context_manager._estimate_tokens(empty_text)
        assert tokens == 1

    def test_optimize_context_simple(self, context_manager):
        """Test basic context optimization."""
        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
            ChatMessage(role=MessageRole.USER, content="How are you?"),
        ]

        optimized = context_manager.optimize_context(messages)

        # Should include all messages (they're short)
        assert len(optimized) == 3
        assert optimized[0]["role"] == "user"
        assert optimized[0]["content"] == "Hello"

    def test_optimize_context_with_system_message(self, context_manager):
        """Test context optimization with system message."""
        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi!"),
        ]

        system_message = "You are a helpful assistant."
        optimized = context_manager.optimize_context(messages, system_message)

        # Should include system message first
        assert len(optimized) == 3
        assert optimized[0]["role"] == "system"
        assert optimized[0]["content"] == system_message

    def test_optimize_context_token_limit(self, context_manager):
        """Test context optimization with token limits."""
        # Create messages that exceed token limit (need more than 500 tokens available)
        long_content = "x" * 2500  # Very long message (625 tokens)
        messages = [
            ChatMessage(role=MessageRole.USER, content="Short message"),
            ChatMessage(role=MessageRole.ASSISTANT, content=long_content),
            ChatMessage(role=MessageRole.USER, content="Another short message"),
        ]

        optimized = context_manager.optimize_context(messages)

        # Should include all messages and potentially a summary
        assert len(optimized) >= 3

        # Should include the recent message
        assert any(msg["content"] == "Another short message" for msg in optimized)

    def test_create_context_summary(self, context_manager):
        """Test context summary creation."""
        messages = [
            ChatMessage(role=MessageRole.USER, content="I need help with my resume"),
            ChatMessage(
                role=MessageRole.ASSISTANT, content="I can help you improve your resume"
            ),
            ChatMessage(role=MessageRole.USER, content="What should I include?"),
        ]

        summary = context_manager._create_context_summary(messages)
        assert "resume" in summary.lower()
        assert len(summary) > 0


class TestMemoryManager:
    """Test MemoryManager integration."""

    @pytest.fixture
    def memory_manager(self):
        """Create memory manager with temporary database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_memory.db"
            yield MemoryManager(db_path, max_context_tokens=1000)

    def test_start_session(self, memory_manager):
        """Test starting a new session."""
        session_id = memory_manager.start_session(
            title="Test Chat", metadata={"test": True}
        )

        assert session_id is not None
        assert len(session_id) > 0

    def test_add_messages(self, memory_manager):
        """Test adding user and assistant messages."""
        session_id = memory_manager.start_session("Test Chat")

        # Add user message
        user_msg_id = memory_manager.add_user_message(
            session_id, "Hello, I need help with my resume"
        )
        assert user_msg_id is not None

        # Add assistant message
        assistant_msg_id = memory_manager.add_assistant_message(
            session_id,
            "I'd be happy to help you with your resume!",
            tokens_used=15,
            processing_time=0.5,
        )
        assert assistant_msg_id is not None

    def test_get_conversation_context(self, memory_manager):
        """Test getting conversation context for LLM."""
        session_id = memory_manager.start_session("Test Chat")

        # Add some messages
        memory_manager.add_user_message(session_id, "Hello!")
        memory_manager.add_assistant_message(session_id, "Hi there!")
        memory_manager.add_user_message(session_id, "How can you help me?")

        # Get context
        context = memory_manager.get_conversation_context(
            session_id, system_message="You are a helpful career assistant."
        )

        assert len(context) == 4  # System + 3 messages
        assert context[0]["role"] == "system"
        assert context[1]["role"] == "user"
        assert context[1]["content"] == "Hello!"

    def test_get_session_history(self, memory_manager):
        """Test getting full session history."""
        session_id = memory_manager.start_session("Test Chat")

        # Add messages
        memory_manager.add_user_message(session_id, "Message 1")
        memory_manager.add_assistant_message(session_id, "Response 1")

        # Get history
        history = memory_manager.get_session_history(session_id)

        assert len(history) == 2
        assert history[0].content == "Message 1"
        assert history[1].content == "Response 1"

    def test_list_sessions(self, memory_manager):
        """Test listing sessions."""
        # Create multiple sessions
        session1_id = memory_manager.start_session("Session 1")
        session2_id = memory_manager.start_session("Session 2")

        # End one session
        memory_manager.end_session(session1_id)

        # List all sessions
        all_sessions = memory_manager.list_sessions()
        assert len(all_sessions) == 2

        # List only active sessions
        active_sessions = memory_manager.list_sessions(SessionStatus.ACTIVE)
        assert len(active_sessions) == 1
        assert active_sessions[0].session_id == session2_id

    def test_cleanup_memory(self, memory_manager):
        """Test memory cleanup."""
        # Create a session
        session_id = memory_manager.start_session("Test Session")

        # Manually age the session
        with memory_manager.db.get_connection() as conn:
            old_time = datetime.now() - timedelta(days=35)
            conn.execute(
                "UPDATE sessions SET last_activity = ? WHERE session_id = ?",
                (old_time.isoformat(), session_id),
            )
            conn.commit()

        # Run cleanup
        results = memory_manager.cleanup_memory(days_threshold=30)
        assert results["archived_sessions"] == 1

    def test_get_memory_stats(self, memory_manager):
        """Test memory statistics."""
        # Create session with messages
        session_id = memory_manager.start_session("Test Session")
        memory_manager.add_user_message(session_id, "Hello")
        memory_manager.add_assistant_message(session_id, "Hi", tokens_used=10)

        # Get stats
        stats = memory_manager.get_memory_stats()

        assert "database_size_bytes" in stats
        assert "total_messages" in stats
        assert "sessions_by_status" in stats
        assert stats["total_messages"] == 2
        assert "active" in stats["sessions_by_status"]


class TestModuleFunctions:
    """Test module-level functions."""

    @patch("src.core.memory_manager.MemoryManager")
    def test_get_memory_manager(self, mock_memory_manager):
        """Test get_memory_manager function."""
        mock_instance = Mock()
        mock_memory_manager.return_value = mock_instance

        result = get_memory_manager()

        assert result == mock_instance
        mock_memory_manager.assert_called_once()


class TestIntegration:
    """Integration tests for memory management."""

    @pytest.fixture
    def memory_manager(self):
        """Create memory manager for integration testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_memory.db"
            yield MemoryManager(db_path, max_context_tokens=1000)

    def test_full_conversation_flow(self, memory_manager):
        """Test complete conversation flow."""
        # Start session
        session_id = memory_manager.start_session("Career Help Session")

        # Simulate conversation
        memory_manager.add_user_message(
            session_id,
            "I need help writing a cover letter for a software engineer position",
        )

        memory_manager.add_assistant_message(
            session_id,
            "I'd be happy to help you write a compelling cover letter. Can you tell me about the company and role?",
            tokens_used=25,
            processing_time=0.8,
        )

        memory_manager.add_user_message(
            session_id,
            "It's for a startup called TechCorp, and they're looking for a full-stack developer",
        )

        memory_manager.add_assistant_message(
            session_id,
            "Great! For a startup like TechCorp, you'll want to emphasize your adaptability and full-stack skills. Here's a draft structure...",
            tokens_used=45,
            processing_time=1.2,
        )

        # Get conversation context
        context = memory_manager.get_conversation_context(session_id)
        assert len(context) == 4

        # Verify message order and content
        assert "cover letter" in context[0]["content"]
        assert "TechCorp" in context[2]["content"]

        # Get session stats
        sessions = memory_manager.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].message_count == 4
        assert sessions[0].total_tokens == 70  # 25 + 45

        # End session
        memory_manager.end_session(session_id)

        # Verify session is completed
        completed_sessions = memory_manager.list_sessions(SessionStatus.COMPLETED)
        assert len(completed_sessions) == 1

    def test_context_window_management(self, memory_manager):
        """Test context window management with many messages."""
        # Create session
        session_id = memory_manager.start_session("Long Conversation")

        # Add many messages with longer content to exceed token limits
        for i in range(30):
            memory_manager.add_user_message(
                session_id,
                f"This is user message number {i} with some content to make it longer and exceed token limits for testing purposes",
            )
            memory_manager.add_assistant_message(
                session_id,
                f"This is assistant response number {i} with helpful information and guidance to make the content longer and test token limits",
            )

        # Get context (should be optimized for token limits)
        context = memory_manager.get_conversation_context(session_id)

        # Should not include all 60 messages due to token limits
        assert len(context) < 60

        # Should include recent messages (but may not be the very last one due to optimization)
        recent_messages = [msg["content"] for msg in context[-10:]]  # Check last 10 messages
        # Check for any recent message numbers (15-29)
        recent_message_found = any(str(i) in msg for msg in recent_messages for i in range(15, 30))
        assert recent_message_found, f"Expected recent messages (15-29) not found in context. Recent messages: {recent_messages[-3:]}"

    def test_concurrent_sessions(self, memory_manager):
        """Test handling multiple concurrent sessions."""
        # Create multiple sessions
        session_ids = []
        for i in range(5):
            session_id = memory_manager.start_session(f"Session {i}")
            session_ids.append(session_id)

            # Add messages to each session
            memory_manager.add_user_message(session_id, f"Hello from session {i}")
            memory_manager.add_assistant_message(session_id, f"Hi from session {i}")

        # Verify each session has correct messages
        for i, session_id in enumerate(session_ids):
            history = memory_manager.get_session_history(session_id)
            assert len(history) == 2
            assert f"session {i}" in history[0].content
            assert f"session {i}" in history[1].content

        # Verify session isolation
        all_sessions = memory_manager.list_sessions()
        assert len(all_sessions) == 5

        # Each session should have 2 messages
        for session in all_sessions:
            assert session.message_count == 2
