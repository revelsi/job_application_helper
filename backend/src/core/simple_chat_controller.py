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
Simplified Chat Controller for Job Application Helper.

This controller provides chat functionality without the complexity of RAG:
- Direct LLM integration with document context
- Simple document retrieval and context building
- Memory management for conversation history
- Support for different content types (cover letters, interview answers, etc.)
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional

from src.core.llm_providers.base import (
    ContentType,
    GenerationRequest,
    LLMProvider,
)
from src.core.memory_manager import MemoryManager, SessionStatus
from src.core.prompts import PromptManager
from src.core.simple_document_service import get_simple_document_service
from src.utils.config import get_settings
from src.utils.logging import get_logger
from src.utils.security import get_prompt_sanitizer


@dataclass
class SimpleChatResponse:
    """Response from simplified chat processing."""

    content: str
    success: bool
    session_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class SimpleChatController:
    """
    Simplified chat controller that provides document-aware responses
    without complex RAG functionality.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        memory_manager: Optional[MemoryManager] = None,
        document_service: Optional[Any] = None,
    ):
        """
        Initialize the simplified chat controller.

        Args:
            llm_provider: LLM provider for response generation (optional - can be set per request)
            memory_manager: Memory manager for conversation history
            document_service: Document service for document operations
        """
        self.settings = get_settings()
        self.logger = get_logger(f"{__name__}.SimpleChatController")

        # Core components
        self.llm_provider = llm_provider  # Can be None - will be set per request
        self.memory_manager = memory_manager or MemoryManager()
        self.document_service = document_service or get_simple_document_service()
        self.prompt_manager = PromptManager()
        self.prompt_sanitizer = get_prompt_sanitizer()
        
        # Query analyzer will be initialized when needed with the specific provider
        self.query_analyzer = None

        # Component availability flags
        self.llm_available = self.llm_provider is not None and self.llm_provider.is_available()
        self.memory_available = self.memory_manager is not None
        self.documents_available = self.document_service is not None
        self.query_analyzer_available = False  # Will be set when query analyzer is initialized

        # Current session tracking
        self.current_session_id: Optional[str] = None

        self.logger.info(
            f"Simplified chat controller initialized "
            f"(LLM: {'✅' if self.llm_available else '❌'}, "
            f"Memory: {'✅' if self.memory_available else '❌'}, "
            f"Documents: {'✅' if self.documents_available else '❌'})"
        )

    def process_message(
        self,
        message: str,
        llm_provider: LLMProvider,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> SimpleChatResponse:
        """
        Process a chat message and generate a response.

        Args:
            message: User message
            conversation_history: Recent conversation history
            session_id: Session ID for memory management
            model: Optional model override
            reasoning_effort: Reasoning effort level for reasoning models (minimal, low, medium, high)

        Returns:
            SimpleChatResponse with generated content
        """
        start_time = time.time()

        try:
            # Sanitize user input
            sanitized_message = self.prompt_sanitizer.sanitize_prompt(message)
            if sanitized_message != message:
                self.logger.warning("User message was sanitized for security")

            # Setup session if memory is available
            if self.memory_available and session_id:
                self.current_session_id = session_id

                # Ensure session exists - create it if it doesn't
                existing_session = self.memory_manager.session_manager.get_session(
                    session_id
                )
                if not existing_session:
                    self.logger.info(f"Creating new session: {session_id}")
                    # Create session with specific ID
                    new_session = self.memory_manager.session_manager.create_session(
                        title=f"Chat Session {session_id}",
                        metadata={"created_by": "chat_controller"},
                    )
                    # Update the session to have our desired ID
                    with self.memory_manager.db.get_connection() as conn:
                        conn.execute(
                            "UPDATE sessions SET session_id = ? WHERE id = ?",
                            (session_id, new_session.id),
                        )
                        conn.commit()

                self.memory_manager.add_user_message(
                    session_id, sanitized_message.strip()
                )

            # Initialize query analysis with the specific provider
            self._ensure_query_analyzer(llm_provider)
            
            # Analyze query for intent and document weighting
            # Note: This is valuable for document weighting and intent detection
            query_analysis = self.query_analyzer.analyze_query(
                sanitized_message, conversation_history
            )
            if query_analysis:
                self.logger.debug(
                    f"Query analysis: intent={query_analysis.intent_type}, confidence={query_analysis.confidence}"
                )

            # Detect content type (enhanced with query analysis)
            content_type = self._detect_content_type(sanitized_message, query_analysis)

            # Get document context (enhanced with document weighting)
            self.logger.info("📄 Retrieving document context...")
            context = self._get_document_context(sanitized_message, query_analysis)
            self.logger.info(f"✅ Document context retrieved: {len(context.get('context_text', ''))} characters")

            # Get conversation history from memory manager if available
            conversation_context = None
            if self.memory_available and self.current_session_id:
                # Get conversation context from memory manager (includes history)
                conversation_context = self.memory_manager.get_conversation_context(
                    self.current_session_id
                )
                self.logger.debug(
                    f"Retrieved {len(conversation_context)} conversation messages from memory"
                )

            # Build prompt with context and conversation history
            self.logger.info("🔨 Building messages with context...")
            prompt_data = self._build_prompt(
                sanitized_message, context, content_type, conversation_context
            )
            messages = prompt_data["messages"]
            legacy_prompt = prompt_data["legacy_prompt"]
            self.logger.info(f"✅ Messages built: {len(messages)} messages, {len(legacy_prompt)} characters")

            # Generate response with proper token management
            safe_max_tokens = self._get_safe_max_tokens(model, legacy_prompt)
            generation_request = GenerationRequest(
                messages=messages,  # Use modern message structure
                content_type=content_type,
                context=context,
                model=model,
                reasoning_effort=reasoning_effort,
                max_tokens=safe_max_tokens,
            )

            llm_response = llm_provider.generate_content(generation_request)

            # Process response
            if llm_response.success:
                # Add response to memory
                if self.memory_available and self.current_session_id:
                    self.memory_manager.add_assistant_message(
                        self.current_session_id,
                        llm_response.content,
                        tokens_used=llm_response.tokens_used,
                        processing_time=time.time() - start_time,
                    )

                # Add source information if documents were used
                enhanced_content = self._enhance_response_with_sources(
                    llm_response.content, context
                )

                return SimpleChatResponse(
                    content=enhanced_content,
                    success=True,
                    session_id=self.current_session_id,
                    metadata={
                        "content_type": content_type.value,
                        "tokens_used": llm_response.tokens_used,
                        "processing_time": time.time() - start_time,
                        "has_document_context": context.get("has_context", False),
                        "document_counts": context.get("document_counts", {}),
                        "conversation_messages": len(conversation_context)
                        if conversation_context
                        else 0,
                    },
                )
            return SimpleChatResponse(
                content="I apologize, but I encountered an error generating a response.",
                success=False,
                session_id=self.current_session_id,
                error=llm_response.error,
                metadata={
                    "processing_time": time.time() - start_time,
                },
            )

        except Exception as e:
            error_msg = f"Chat processing failed: {e!s}"
            self.logger.error(error_msg)

            return SimpleChatResponse(
                content="I apologize, but I encountered an error processing your request.",
                success=False,
                session_id=self.current_session_id,
                error=error_msg,
                metadata={
                    "processing_time": time.time() - start_time,
                },
            )

    async def process_message_stream(
        self,
        message: str,
        llm_provider: LLMProvider,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Process a chat message and generate a streaming response.

        Args:
            message: User message
            conversation_history: Recent conversation history
            session_id: Session ID for memory management
            model: Optional specific model to use for generation

        Yields:
            Chunks of generated content
        """
        self.logger.info(f"🚀 Starting process_message_stream with provider: {llm_provider.provider_type}")
        try:
            # Sanitize user input
            sanitized_message = self.prompt_sanitizer.sanitize_prompt(message)
            if sanitized_message != message:
                self.logger.warning("User message was sanitized for security")

            # Setup session if memory is available
            if self.memory_available and session_id:
                self.current_session_id = session_id

                # Ensure session exists - create it if it doesn't
                existing_session = self.memory_manager.session_manager.get_session(
                    session_id
                )
                if not existing_session:
                    self.logger.info(
                        f"Creating new session for streaming: {session_id}"
                    )
                    # Create session with specific ID
                    new_session = self.memory_manager.session_manager.create_session(
                        title=f"Chat Session {session_id}",
                        metadata={"created_by": "chat_controller"},
                    )
                    # Update the session to have our desired ID
                    with self.memory_manager.db.get_connection() as conn:
                        conn.execute(
                            "UPDATE sessions SET session_id = ? WHERE id = ?",
                            (session_id, new_session.id),
                        )
                        conn.commit()

                self.memory_manager.add_user_message(
                    session_id, sanitized_message.strip()
                )

            # Initialize query analysis with the specific provider
            self._ensure_query_analyzer(llm_provider)
            
            # Analyze query for intent and document weighting
            # Note: This is valuable for document weighting and intent detection
            query_analysis = self.query_analyzer.analyze_query(
                sanitized_message, conversation_history
            )
            if query_analysis:
                self.logger.debug(
                    f"Query analysis: intent={query_analysis.intent_type}, confidence={query_analysis.confidence}"
                )

            # Detect content type (enhanced with query analysis)
            content_type = self._detect_content_type(sanitized_message, query_analysis)

            # Get document context (enhanced with document weighting)
            self.logger.info("📄 Retrieving document context...")
            context = self._get_document_context(sanitized_message, query_analysis)
            self.logger.info(f"✅ Document context retrieved: {len(context.get('context_text', ''))} characters")

            # Get conversation history from memory manager if available
            conversation_context = None
            if self.memory_available and self.current_session_id:
                # Get conversation context from memory manager (includes history)
                conversation_context = self.memory_manager.get_conversation_context(
                    self.current_session_id
                )
                self.logger.debug(
                    f"Retrieved {len(conversation_context)} conversation messages from memory for streaming"
                )

            # Build prompt with context and conversation history
            self.logger.info("🔨 Building messages with context...")
            prompt_data = self._build_prompt(
                sanitized_message, context, content_type, conversation_context
            )
            messages = prompt_data["messages"]
            legacy_prompt = prompt_data["legacy_prompt"]
            self.logger.info(f"✅ Messages built: {len(messages)} messages, {len(legacy_prompt)} characters")

            # Generate streaming response with proper token management
            safe_max_tokens = self._get_safe_max_tokens(model, legacy_prompt)
            generation_request = GenerationRequest(
                messages=messages,  # Use modern message structure
                content_type=content_type,
                context=context,
                model=model,
                reasoning_effort=reasoning_effort,
                max_tokens=safe_max_tokens,
            )

            # Accumulate content for memory
            accumulated_content = ""

            self.logger.info(f"🔥 About to call generate_content_stream with model: {model}")
            async for chunk in llm_provider.generate_content_stream(
                generation_request
            ):
                self.logger.debug(f"📦 Received chunk: {chunk[:50]}...")
                accumulated_content += chunk
                yield chunk
            
            self.logger.info(f"✅ Streaming completed. Total content length: {len(accumulated_content)}")

            # Add complete response to memory
            if self.memory_available and self.current_session_id:
                self.memory_manager.add_assistant_message(
                    self.current_session_id,
                    accumulated_content,
                    tokens_used=0,  # We don't have token count for streaming
                    processing_time=0.0,
                )

        except Exception as e:
            error_msg = f"Chat streaming failed: {e!s}"
            self.logger.error(error_msg)
            yield f"Error: {error_msg}\n"

    def _detect_content_type(self, message: str, query_analysis=None) -> ContentType:
        """
        Detect the content type from the user message, enhanced with query analysis.

        Args:
            message: User message
            query_analysis: Optional QueryAnalysis from QueryAnalyzer

        Returns:
            ContentType enum value
        """
        # Use QueryAnalyzer result if available and confident
        if query_analysis and query_analysis.confidence > 0.7:
            intent_to_content_map = {
                "cover_letter": ContentType.COVER_LETTER,
                "behavioral_interview": ContentType.INTERVIEW_ANSWER,
                "interview_answer": ContentType.INTERVIEW_ANSWER,
                "content_refinement": ContentType.CONTENT_REFINEMENT,
                "ats_optimizer": ContentType.CONTENT_REFINEMENT,
                "achievement_quantifier": ContentType.CONTENT_REFINEMENT,
                "general": ContentType.GENERAL_RESPONSE,
            }

            content_type = intent_to_content_map.get(query_analysis.intent_type)
            if content_type:
                self.logger.debug(
                    f"Using QueryAnalyzer content type: {content_type} (confidence: {query_analysis.confidence})"
                )
                return content_type
        message_lower = message.lower()

        # Cover letter indicators
        cover_letter_keywords = [
            "cover letter",
            "covering letter",
            "write a cover letter",
            "generate cover letter",
            "create cover letter",
        ]

        # Interview answer indicators
        interview_keywords = [
            "interview",
            "behavioral",
            "tell me about",
            "describe a time",
            "give me an example",
            "star method",
            "why should we hire you",
        ]

        # Content refinement indicators
        refinement_keywords = [
            "improve",
            "refine",
            "better",
            "enhance",
            "optimize",
            "revise",
            "rewrite",
        ]

        # Check for specific content types
        if any(keyword in message_lower for keyword in cover_letter_keywords):
            return ContentType.COVER_LETTER
        if any(keyword in message_lower for keyword in interview_keywords):
            return ContentType.INTERVIEW_ANSWER
        if any(keyword in message_lower for keyword in refinement_keywords):
            return ContentType.CONTENT_REFINEMENT
        return ContentType.GENERAL_RESPONSE

    def _get_document_context(
        self, message: str, query_analysis=None
    ) -> Dict[str, Any]:
        """
        Get document context - simple document dumping approach.
        
        Just gets the most recent documents from each category.
        The message parameter is kept for interface compatibility but currently ignored.
        
        Args:
            message: User message (currently unused, kept for interface compatibility)
            query_analysis: Optional QueryAnalysis with document weights (currently unused)

        Returns:
            Dictionary with document context
        """
        if not self.documents_available:
            return {
                "has_context": False,
                "context_text": "",
                "source_documents": [],
                "document_counts": {},
            }

        # Apply dynamic document weighting if query analysis is available
        if query_analysis and query_analysis.document_weights:
            weights = query_analysis.document_weights
            base_length = getattr(self.settings, "max_context_length", 100000)

            # Calculate dynamic limits based on document weights
            max_candidate_doc_length = int(base_length * weights.get("candidate", 0.4))
            max_job_doc_length = int(base_length * weights.get("job", 0.3))
            max_company_doc_length = int(base_length * weights.get("company", 0.3))

            self.logger.debug(
                f"Using dynamic document limits: candidate={max_candidate_doc_length}, job={max_job_doc_length}, company={max_company_doc_length}"
            )
        else:
            # Use default limits
            max_candidate_doc_length = getattr(
                self.settings, "max_candidate_doc_length", 50000
            )
            max_job_doc_length = getattr(self.settings, "max_job_doc_length", 30000)
            max_company_doc_length = getattr(
                self.settings, "max_company_doc_length", 20000
            )

        # Get context from document service - simple document dumping
        context = self.document_service.get_document_context(
            max_context_length=getattr(self.settings, "max_context_length", 100000),
            max_candidate_doc_length=max_candidate_doc_length,
            max_job_doc_length=max_job_doc_length,
            max_company_doc_length=max_company_doc_length,
        )

        return context

    def _build_prompt(
        self,
        message: str,
        context: Dict[str, Any],
        content_type: ContentType,
        conversation_context: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build the prompt for LLM generation using PromptManager.

        Args:
            message: User message
            context: Document context
            content_type: Type of content to generate
            conversation_context: Previous conversation messages for context

        Returns:
            Formatted prompt string
        """
        # Map ContentType to PromptType
        prompt_type = self._map_content_type_to_prompt_type(content_type)
        
        # Extract context information for prompt enhancement
        context_info = self._extract_context_info(context)
        
        # Build conversation history for PromptManager
        conversation_history = self._format_conversation_history(conversation_context)
        
        # Use PromptManager to build modern message structure
        messages = self.prompt_manager.build_messages(
            prompt_type=prompt_type,
            user_query=message,
            context=context.get("context_text", ""),
            conversation_history=conversation_history,
            **context_info
        )
        
        # For backward compatibility, also provide the legacy prompt format
        # by combining all messages into a single string
        legacy_prompt = self._messages_to_legacy_prompt(messages)
        
        return {
            "messages": messages,
            "legacy_prompt": legacy_prompt
        }
    
    def _map_content_type_to_prompt_type(self, content_type: ContentType):
        """Map ContentType to PromptType for PromptManager."""
        from src.core.prompts import PromptType
        
        mapping = {
            ContentType.COVER_LETTER: PromptType.COVER_LETTER,
            ContentType.INTERVIEW_ANSWER: PromptType.INTERVIEW_ANSWER,
            ContentType.CONTENT_REFINEMENT: PromptType.GENERAL_RESPONSE,
            ContentType.GENERAL_RESPONSE: PromptType.GENERAL_RESPONSE,
        }
        
        return mapping.get(content_type, PromptType.GENERAL_RESPONSE)
    
    def _extract_context_info(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract contextual information for prompt enhancement."""
        context_info = {}
        
        # Try to detect industry from document context
        context_text = context.get("context_text", "").lower()
        if any(keyword in context_text for keyword in ["software", "developer", "engineer", "tech", "programming"]):
            context_info["industry"] = "tech"
        elif any(keyword in context_text for keyword in ["finance", "banking", "investment", "financial"]):
            context_info["industry"] = "finance"
        elif any(keyword in context_text for keyword in ["healthcare", "medical", "hospital", "patient"]):
            context_info["industry"] = "healthcare"
        
        # Try to detect experience level
        if any(keyword in context_text for keyword in ["senior", "lead", "principal", "architect", "manager"]):
            context_info["experience_level"] = "senior"
        elif any(keyword in context_text for keyword in ["junior", "entry", "graduate", "intern"]):
            context_info["experience_level"] = "entry"
        else:
            context_info["experience_level"] = "mid"
        
        return context_info
    
    def _format_conversation_history(self, conversation_context: Optional[List[Dict[str, Any]]]) -> Optional[List]:
        """Format conversation history for PromptManager."""
        if not conversation_context or len(conversation_context) <= 1:
            return None
        
        # Filter out system messages and format for PromptManager
        formatted_history = []
        for msg in conversation_context:
            if msg.get("role") != "system":
                formatted_history.append({
                    "role": msg.get("role"),
                    "content": msg.get("content", "")
                })
        
        # Return recent history (more intelligent limit based on content length)
        return self._get_conversation_history_within_token_budget(formatted_history)
    
    def _get_conversation_history_within_token_budget(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 8000
    ) -> List[Dict[str, str]]:
        """
        Get conversation history that fits within token budget.
        Uses rough estimation: 1 token ≈ 4 characters.
        """
        if not messages:
            return []
        
        selected_messages = []
        current_tokens = 0
        
        # Start from most recent messages
        for msg in reversed(messages):
            content = msg.get("content", "")
            estimated_tokens = len(content) // 4  # Rough estimation
            
            if current_tokens + estimated_tokens > max_tokens:
                break
            
            selected_messages.insert(0, msg)  # Insert at beginning to maintain order
            current_tokens += estimated_tokens
        
        return selected_messages

    def _messages_to_legacy_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert message structure back to legacy prompt format for backward compatibility.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Single prompt string combining all messages
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(content)
            elif role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)

    def _get_safe_max_tokens(self, model: Optional[str], prompt: str) -> int:
        """
        Get safe max_tokens for the given model and prompt.
        Uses model-specific limits instead of generic settings.
        
        Args:
            model: Model name (if None, uses provider default)
            prompt: The prompt to estimate token usage for
            
        Returns:
            Safe max_tokens value that respects model limits
        """
        from src.core.llm_providers.model_config import get_safe_token_limits
        
        # Estimate input tokens (rough estimation: 1 token ≈ 4 characters)
        estimated_input_tokens = len(prompt) // 4
        
        # Get model-specific safe limits
        if model:
            token_limits = get_safe_token_limits(model, estimated_input_tokens)
            model_max_output = token_limits["recommended_output"]
        else:
            # Fallback for unknown models
            model_max_output = 2048
        
        # Use the smaller of: model limit, settings limit, or calculated safe limit
        settings_max = getattr(self.settings, "chat_max_tokens", 16000)
        safe_max_tokens = min(model_max_output, settings_max)
        
        self.logger.debug(
            f"Token calculation - Model: {model}, Input tokens: ~{estimated_input_tokens}, "
            f"Model limit: {model_max_output}, Settings: {settings_max}, "
            f"Using: {safe_max_tokens}"
        )
        
        return safe_max_tokens

    def _enhance_response_with_sources(
        self, response: str, context: Dict[str, Any]
    ) -> str:
        """
        Enhance response with source document information.

        Args:
            response: Generated response
            context: Document context used

        Returns:
            Enhanced response with source information
        """
        if not context.get("has_context", False):
            return response

        source_documents = context.get("source_documents", [])
        if not source_documents:
            return response

        # Add source information
        if len(source_documents) == 1:
            source_info = (
                f"\n\n*Based on information from: {source_documents[0]['filename']}*"
            )
        else:
            filenames = [doc["filename"] for doc in source_documents]
            unique_filenames = list(set(filenames))
            if len(unique_filenames) <= 3:
                file_list = ", ".join(unique_filenames)
            else:
                file_list = f"{', '.join(unique_filenames[:3])}, and {len(unique_filenames) - 3} more"
            source_info = f"\n\n*Based on information from: {file_list}*"

        return response + source_info

    def _ensure_session(self) -> str:
        """Ensure we have a valid session ID."""
        if not self.memory_available:
            return "no_session"

        return self.memory_manager.create_session()

    def get_stats(self) -> Dict[str, Any]:
        """Get chat controller statistics."""
        stats = {
            "llm_available": self.llm_available,
            "memory_available": self.memory_available,
            "documents_available": self.documents_available,
            "current_session": self.current_session_id,
        }

        # Add document stats
        if self.documents_available:
            stats["documents"] = self.document_service.get_stats()

        # Add memory stats
        if self.memory_available:
            stats["memory"] = {
                "total_sessions": len(self.memory_manager.get_active_sessions()),
                "active_sessions": len(self.memory_manager.get_active_sessions()),
            }

        # Add LLM stats
        if self.llm_available:
            stats["llm"] = {
                "provider_type": self.llm_provider.provider_type.value,
                "available": self.llm_provider.is_available(),
                "default_model": self.llm_provider.get_default_model(),
            }

        return stats

    def clear_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Clear the chat session and conversation history.

        Args:
            session_id: Session ID to clear. If None, clears current session.

        Returns:
            Dictionary with operation result
        """
        try:
            target_session_id = session_id or self.current_session_id

            # If no session specified and no current session, still return success
            # (nothing to clear is a valid state)
            if not target_session_id:
                self.logger.info("No active session to clear - returning success")
                return {
                    "success": True,
                    "message": "No active session to clear",
                    "session_id": None,
                    "messages_deleted": 0,
                }

            # Clear session from memory manager if available
            if self.memory_available:
                # First check if session exists
                existing_session = self.memory_manager.session_manager.get_session(
                    target_session_id
                )

                if existing_session:
                    # Delete messages for existing session
                    deleted_count = (
                        self.memory_manager.message_manager.delete_session_messages(
                            target_session_id
                        )
                    )
                    self.memory_manager.session_manager.update_session_status(
                        target_session_id, SessionStatus.COMPLETED
                    )

                    self.logger.info(
                        f"Cleared existing session {target_session_id} - deleted {deleted_count} messages"
                    )

                    # Reset current session if it was the one being cleared
                    if target_session_id == self.current_session_id:
                        self.current_session_id = None

                    return {
                        "success": True,
                        "message": f"Session cleared successfully. Removed {deleted_count} messages.",
                        "session_id": target_session_id,
                        "messages_deleted": deleted_count,
                    }
                # Session doesn't exist, but that's fine - nothing to clear
                self.logger.info(
                    f"Session {target_session_id} doesn't exist - nothing to clear"
                )

                # Reset current session if it was the one being cleared
                if target_session_id == self.current_session_id:
                    self.current_session_id = None

                return {
                    "success": True,
                    "message": "Session cleared (no messages found)",
                    "session_id": target_session_id,
                    "messages_deleted": 0,
                }
            # If no memory manager, just reset current session
            if target_session_id == self.current_session_id:
                self.current_session_id = None

            return {
                "success": True,
                "message": "Session cleared (no persistent memory)",
                "session_id": target_session_id,
                "messages_deleted": 0,
            }

        except Exception as e:
            error_msg = f"Failed to clear session: {e!s}"
            self.logger.error(error_msg)

            return {
                "success": False,
                "message": error_msg,
                "session_id": target_session_id
                if "target_session_id" in locals()
                else None,
                "error": str(e),
            }

    def _ensure_query_analyzer(self, llm_provider: LLMProvider):
        """Ensure query analyzer is initialized with the correct provider."""
        if self.query_analyzer is None or self.query_analyzer.llm_provider != llm_provider:
            from src.core.query_analyzer import QueryAnalyzer
            self.query_analyzer = QueryAnalyzer(llm_provider=llm_provider)
            self.query_analyzer_available = self.query_analyzer.llm_available
