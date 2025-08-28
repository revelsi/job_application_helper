"""
Job Application Helper - Core Module

This module contains the core functionality for the job application helper:
- Document processing and storage
- LLM integration
- Memory management and conversation tracking
- Simplified chat and document handling
- Content generation
"""

__version__ = "1.0.0"
__author__ = "Job Application Helper Team"

# Export key classes and functions
from src.core.memory_manager import (
    ChatMessage,
    ConversationSession,
    MemoryManager,
    MessageRole,
    SessionStatus,
    get_memory_manager,
)
from src.core.simple_chat_controller import SimpleChatController
from src.core.simple_document_handler import (
    SimpleDocumentHandler,
    SimpleDocumentUploadResult,
    create_simple_document_handler,
)
from src.core.simple_document_service import (
    DocumentContent,
    DocumentSearchResult,
    SimpleDocumentService,
    get_simple_document_service,
)
from src.core.storage import (
    DocumentMetadata,
    DocumentStatus,
    DocumentType,
    StorageSystem,
    get_storage_system,
)
