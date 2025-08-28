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
Memory management system for job application helper chat functionality.

This module provides:
- Conversation history tracking and persistence
- Session management with proper isolation
- Context window management for LLM interactions
- Message threading and conversation branching
- Memory optimization and cleanup strategies
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import uuid

from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MessageRole(Enum):
    """Roles for chat messages."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SessionStatus(Enum):
    """Status of conversation sessions."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


@dataclass
class ChatMessage:
    """Represents a single message in a conversation."""

    id: Optional[int] = None
    session_id: str = ""
    role: MessageRole = MessageRole.USER
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_message_id: Optional[int] = None
    tokens_used: Optional[int] = None
    processing_time: Optional[float] = None

    def __post_init__(self):
        """Validate and process fields after initialization."""
        if isinstance(self.role, str):
            self.role = MessageRole(self.role)

    @property
    def is_user_message(self) -> bool:
        """Check if this is a user message."""
        return self.role == MessageRole.USER

    @property
    def is_assistant_message(self) -> bool:
        """Check if this is an assistant message."""
        return self.role == MessageRole.ASSISTANT

    @property
    def is_system_message(self) -> bool:
        """Check if this is a system message."""
        return self.role == MessageRole.SYSTEM

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for API calls."""
        return {"role": self.role.value, "content": self.content}


@dataclass
class ConversationSession:
    """Represents a conversation session with metadata."""

    id: Optional[int] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    message_count: int = 0
    total_tokens: int = 0
    context_summary: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and process fields after initialization."""
        if isinstance(self.status, str):
            self.status = SessionStatus(self.status)

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.status == SessionStatus.ACTIVE

    @property
    def age_hours(self) -> float:
        """Get session age in hours."""
        return (datetime.now() - self.created_at).total_seconds() / 3600

    @property
    def idle_hours(self) -> float:
        """Get hours since last activity."""
        return (datetime.now() - self.last_activity).total_seconds() / 3600


class MemoryDatabase:
    """Database operations for memory management."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize memory database."""
        self.settings = get_settings()
        self.db_path = db_path or self.settings.data_dir / "memory.db"
        self._setup_database()

    def _setup_database(self):
        """Set up the memory database with required tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            self._create_tables(conn)
            self._create_indexes(conn)

    def _create_tables(self, conn: sqlite3.Connection):
        """Create database tables."""
        # Sessions table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                last_activity TEXT NOT NULL,
                message_count INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                context_summary TEXT DEFAULT '',
                tags TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}'
            )
        """
        )

        # Messages table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                parent_message_id INTEGER,
                tokens_used INTEGER,
                processing_time REAL,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id),
                FOREIGN KEY (parent_message_id) REFERENCES messages (id)
            )
        """
        )

        conn.commit()

    def _create_indexes(self, conn: sqlite3.Connection):
        """Create database indexes for performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions (status)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions (last_activity)",
            "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages (session_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_messages_role ON messages (role)",
        ]

        for index_sql in indexes:
            conn.execute(index_sql)

        conn.commit()

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


class SessionManager:
    """Manages conversation sessions."""

    def __init__(self, db: MemoryDatabase):
        """Initialize session manager."""
        self.db = db

    def create_session(
        self, title: str = "", metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationSession:
        """Create a new conversation session."""
        session = ConversationSession(
            title=title or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            metadata=metadata or {},
        )

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO sessions (session_id, title, status, created_at, last_activity, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    session.session_id,
                    session.title,
                    session.status.value,
                    session.created_at.isoformat(),
                    session.last_activity.isoformat(),
                    json.dumps(session.metadata),
                ),
            )
            session.id = cursor.lastrowid
            conn.commit()

        logger.info(f"Created new session: {session.session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get session by ID."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()

            if not row:
                return None

            return self._row_to_session(row)

    def update_session_activity(
        self, session_id: str, message_count_delta: int = 0, tokens_delta: int = 0
    ):
        """Update session activity metrics."""
        with self.db.get_connection() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET last_activity = ?,
                    message_count = message_count + ?,
                    total_tokens = total_tokens + ?
                WHERE session_id = ?
            """,
                (
                    datetime.now().isoformat(),
                    message_count_delta,
                    tokens_delta,
                    session_id,
                ),
            )
            conn.commit()

    def update_session_status(self, session_id: str, status: SessionStatus):
        """Update session status."""
        with self.db.get_connection() as conn:
            conn.execute(
                "UPDATE sessions SET status = ? WHERE session_id = ?",
                (status.value, session_id),
            )
            conn.commit()

    def list_sessions(
        self, status: Optional[SessionStatus] = None, limit: int = 50
    ) -> List[ConversationSession]:
        """List sessions with optional filtering."""
        query = "SELECT * FROM sessions"
        params = []

        if status:
            query += " WHERE status = ?"
            params.append(status.value)

        query += " ORDER BY last_activity DESC LIMIT ?"
        params.append(limit)

        with self.db.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_session(row) for row in rows]

    def cleanup_old_sessions(self, days_threshold: int = 30) -> int:
        """Archive old inactive sessions."""
        cutoff_date = datetime.now() - timedelta(days=days_threshold)

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE sessions
                SET status = 'archived'
                WHERE status = 'active'
                AND last_activity < ?
            """,
                (cutoff_date.isoformat(),),
            )

            archived_count = cursor.rowcount
            conn.commit()

        logger.info(f"Archived {archived_count} old sessions")
        return archived_count

    def _row_to_session(self, row: sqlite3.Row) -> ConversationSession:
        """Convert database row to ConversationSession."""
        return ConversationSession(
            id=row["id"],
            session_id=row["session_id"],
            title=row["title"],
            status=SessionStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_activity=datetime.fromisoformat(row["last_activity"]),
            message_count=row["message_count"],
            total_tokens=row["total_tokens"],
            context_summary=row["context_summary"],
            tags=json.loads(row["tags"]),
            metadata=json.loads(row["metadata"]),
        )


class MessageManager:
    """Manages chat messages within sessions."""

    def __init__(self, db: MemoryDatabase):
        """Initialize message manager."""
        self.db = db

    def add_message(self, message: ChatMessage) -> int:
        """Add a message to the database."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (session_id, role, content, timestamp, metadata,
                                    parent_message_id, tokens_used, processing_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    message.session_id,
                    message.role.value,
                    message.content,
                    message.timestamp.isoformat(),
                    json.dumps(message.metadata),
                    message.parent_message_id,
                    message.tokens_used,
                    message.processing_time,
                ),
            )
            message.id = cursor.lastrowid
            conn.commit()

        return message.id

    def get_session_messages(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[ChatMessage]:
        """Get all messages for a session."""
        query = "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC"
        params = [session_id]

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self.db.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_message(row) for row in rows]

    def get_recent_messages(
        self, session_id: str, count: int = 10
    ) -> List[ChatMessage]:
        """Get recent messages for context."""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (session_id, count),
            ).fetchall()

            # Reverse to get chronological order
            return [self._row_to_message(row) for row in reversed(rows)]

    def delete_session_messages(self, session_id: str) -> int:
        """Delete all messages for a session."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            deleted_count = cursor.rowcount
            conn.commit()

        return deleted_count

    def _row_to_message(self, row: sqlite3.Row) -> ChatMessage:
        """Convert database row to ChatMessage."""
        return ChatMessage(
            id=row["id"],
            session_id=row["session_id"],
            role=MessageRole(row["role"]),
            content=row["content"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            metadata=json.loads(row["metadata"]),
            parent_message_id=row["parent_message_id"],
            tokens_used=row["tokens_used"],
            processing_time=row["processing_time"],
        )


class ContextWindowManager:
    """Manages context window optimization for LLM interactions."""

    def __init__(
        self, max_tokens: int = 120000
    ):  # Updated for GPT-5-mini 128K context
        """Initialize context window manager."""
        self.max_tokens = max_tokens
        self.system_message_tokens = 500  # Increased reserved space for system messages
        self.response_buffer_tokens = (
            16000  # Increased buffer to match new chat_max_tokens for reasoning models
        )
        # Ensure we have at least some tokens available for conversation history
        self.available_tokens = max(
            1000, max_tokens - self.system_message_tokens - self.response_buffer_tokens
        )

    def optimize_context(
        self, messages: List[ChatMessage], system_message: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Optimize message context for LLM API calls."""
        optimized_messages = []

        # Add system message if provided
        if system_message:
            optimized_messages.append({"role": "system", "content": system_message})

        # If no messages, return just the system message
        if not messages:
            return optimized_messages

        # Estimate tokens and fit within limits
        current_tokens = 0
        included_messages = []

        # Process messages in chronological order (oldest first)
        for message in messages:
            estimated_tokens = self._estimate_tokens(message.content)

            if current_tokens + estimated_tokens <= self.available_tokens:
                included_messages.append(message.to_dict())
                current_tokens += estimated_tokens
            else:
                # If we can't fit this message, try to include a summary of what we're missing
                if len(included_messages) > 0:
                    # Get the messages that weren't included (remaining messages)
                    excluded_messages = messages[messages.index(message) :]
                    if excluded_messages:
                        summary = self._create_context_summary(excluded_messages)
                        if summary:
                            summary_tokens = self._estimate_tokens(summary)
                            if current_tokens + summary_tokens <= self.available_tokens:
                                included_messages.append(
                                    {
                                        "role": "system",
                                        "content": f"Previous conversation summary: {summary}",
                                    }
                                )
                break

        optimized_messages.extend(included_messages)
        return optimized_messages

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)."""
        return max(1, len(text) // 4)

    def _create_context_summary(self, messages: List[ChatMessage]) -> str:
        """Create a summary of conversation context."""
        if not messages:
            return ""

        # Simple summarization - extract key topics and decisions
        user_messages = [msg.content for msg in messages if msg.is_user_message]
        assistant_messages = [
            msg.content for msg in messages if msg.is_assistant_message
        ]

        if not user_messages:
            return ""

        # Create a more comprehensive summary
        summary_parts = []

        # Include recent user requests (last 2-3)
        if user_messages:
            recent_user_messages = user_messages[-3:]  # Last 3 user messages
            if len(recent_user_messages) == 1:
                summary_parts.append(
                    f"User requested: {recent_user_messages[0][:150]}..."
                )
            else:
                user_summary = "; ".join(
                    [msg[:100] + "..." for msg in recent_user_messages]
                )
                summary_parts.append(f"User discussed: {user_summary}")

        # Include recent assistant responses (last 2-3)
        if assistant_messages:
            recent_assistant_messages = assistant_messages[
                -3:
            ]  # Last 3 assistant messages
            if len(recent_assistant_messages) == 1:
                summary_parts.append(
                    f"Assistant provided: {recent_assistant_messages[0][:150]}..."
                )
            else:
                assistant_summary = "; ".join(
                    [msg[:100] + "..." for msg in recent_assistant_messages]
                )
                summary_parts.append(f"Assistant helped with: {assistant_summary}")

        return " ".join(summary_parts)


class MemoryManager:
    """Main memory management system for chat functionality."""

    def __init__(
        self, db_path: Optional[Path] = None, max_context_tokens: int = 120000
    ):  # Updated for GPT-5-mini
        """Initialize memory manager."""
        self.db = MemoryDatabase(db_path)
        self.session_manager = SessionManager(self.db)
        self.message_manager = MessageManager(self.db)
        self.context_manager = ContextWindowManager(max_context_tokens)

        logger.info("Memory manager initialized")

    def start_session(
        self, title: str = "", metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new conversation session."""
        session = self.session_manager.create_session(title, metadata)
        return session.session_id

    def add_user_message(
        self, session_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Add a user message to the session."""
        message = ChatMessage(
            session_id=session_id,
            role=MessageRole.USER,
            content=content,
            metadata=metadata or {},
        )

        message_id = self.message_manager.add_message(message)
        self.session_manager.update_session_activity(session_id, message_count_delta=1)

        return message_id

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        tokens_used: Optional[int] = None,
        processing_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add an assistant message to the session."""
        message = ChatMessage(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=content,
            tokens_used=tokens_used,
            processing_time=processing_time,
            metadata=metadata or {},
        )

        message_id = self.message_manager.add_message(message)
        self.session_manager.update_session_activity(
            session_id, message_count_delta=1, tokens_delta=tokens_used or 0
        )

        return message_id

    def get_conversation_context(
        self, session_id: str, system_message: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Get optimized conversation context for LLM API calls."""
        messages = self.message_manager.get_session_messages(session_id)
        return self.context_manager.optimize_context(messages, system_message)

    def get_session_history(self, session_id: str) -> List[ChatMessage]:
        """Get full conversation history for a session."""
        return self.message_manager.get_session_messages(session_id)

    def list_sessions(
        self, status: Optional[SessionStatus] = None
    ) -> List[ConversationSession]:
        """List conversation sessions."""
        return self.session_manager.list_sessions(status)

    def end_session(self, session_id: str):
        """Mark a session as completed."""
        self.session_manager.update_session_status(session_id, SessionStatus.COMPLETED)

    def get_active_sessions(self) -> List[ConversationSession]:
        """Get all active sessions."""
        return self.session_manager.list_sessions(SessionStatus.ACTIVE)

    def create_session(
        self, title: str = "", metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new session and return its ID."""
        session = self.session_manager.create_session(title, metadata)
        return session.session_id

    def cleanup_memory(self, days_threshold: int = 30) -> Dict[str, int]:
        """Clean up old conversation data."""
        archived_sessions = self.session_manager.cleanup_old_sessions(days_threshold)

        return {"archived_sessions": archived_sessions}

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        with self.db.get_connection() as conn:
            session_stats = conn.execute(
                """
                SELECT status, COUNT(*) as count, SUM(message_count) as total_messages, SUM(total_tokens) as total_tokens
                FROM sessions
                GROUP BY status
            """
            ).fetchall()

            total_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

            db_size = self.db.db_path.stat().st_size if self.db.db_path.exists() else 0

        stats = {
            "database_size_bytes": db_size,
            "total_messages": total_messages,
            "sessions_by_status": {
                row["status"]: {
                    "count": row["count"],
                    "messages": row["total_messages"],
                    "tokens": row["total_tokens"],
                }
                for row in session_stats
            },
        }

        return stats


def get_memory_manager() -> MemoryManager:
    """Get the default memory manager instance."""
    return MemoryManager()
