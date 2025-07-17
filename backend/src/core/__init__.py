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
from src.core.memory_manager import (  # noqa: F401
    ChatMessage,
    ConversationSession,
    MemoryManager,
    MessageRole,
    SessionStatus,
    get_memory_manager,
)
from src.core.simple_chat_controller import (  # noqa: F401
    SimpleChatController,
    SimpleChatResponse,
    create_simple_chat_controller,
)
from src.core.simple_document_handler import (  # noqa: F401
    SimpleDocumentHandler,
    SimpleDocumentUploadResult,
    create_simple_document_handler,
)
from src.core.simple_document_service import (  # noqa: F401
    SimpleDocumentService,
    DocumentContent,
    DocumentSearchResult,
    get_simple_document_service,
)
from src.core.storage import (  # noqa: F401
    DocumentMetadata,
    DocumentType,
    DocumentStatus,
    StorageSystem,
    get_storage_system,
)
