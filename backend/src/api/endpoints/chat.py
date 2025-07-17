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

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.api.models import ChatRequest, ChatResponse
from src.core.simple_chat_controller import SimpleChatController, create_simple_chat_controller
from src.core.llm_providers.factory import get_default_provider
from src.core.memory_manager import MemoryManager
from src.utils.logging import get_logger

router = APIRouter(prefix="/chat", tags=["chat"])

logger = get_logger(__name__)


def get_chat_controller() -> SimpleChatController:
    """Dependency to get simplified chat controller instance."""
    logger.info("🚀 Starting simplified chat controller initialization...")

    try:
        # Initialize LLM provider
        logger.info("🔧 Initializing LLM provider...")
        
        # Get fresh API key status to ensure we have the latest configuration
        from src.core.llm_providers.factory import get_api_key_manager
        key_manager = get_api_key_manager()
        config_status = key_manager.list_configured_providers()
        logger.debug(f"📊 Current provider configuration status: {config_status}")
        
        # Try to get a fresh provider instance
        llm_provider = None
        try:
            llm_provider = get_default_provider()
            logger.info(f"✅ LLM provider initialized: {type(llm_provider).__name__}")
            logger.debug(f"📊 LLM provider details: {llm_provider.get_default_model()}")
        except ValueError as e:
            logger.warning(f"❌ No LLM provider available: {e}")
            # Continue without LLM provider (demo mode)
            llm_provider = None

        # Initialize memory manager
        logger.info("🧠 Initializing memory manager...")
        memory_manager = MemoryManager()
        logger.info("✅ Memory manager initialized")

        # Create simplified chat controller
        controller = create_simple_chat_controller(
            llm_provider=llm_provider,
            memory_manager=memory_manager,
        )

        logger.info(
            f"🎉 Simplified chat controller created successfully - "
            f"LLM: {controller.llm_available}, "
            f"Memory: {controller.memory_available}, "
            f"Documents: {controller.documents_available}"
        )
        return controller

    except Exception as e:
        logger.error(f"💥 Failed to initialize chat controller: {e}")
        # Return a basic controller even on failure
        return create_simple_chat_controller()


@router.post("/complete", response_model=ChatResponse)
async def chat_complete(
    request: ChatRequest, chat_controller: SimpleChatController = Depends(get_chat_controller)
):
    """Process chat message using the simplified chat controller."""
    logger.info("💬 Received chat completion request")
    logger.debug(
        f"📝 Message: {request.message[:100]}{'...' if len(request.message) > 100 else ''}"
    )
    logger.debug(f"🔗 Session ID: {request.session_id}")

    try:
        logger.info("🔄 Processing chat message...")

        # Convert Pydantic model to dict for history
        history = (
            [msg.dict() for msg in request.conversation_history]
            if request.conversation_history
            else None
        )

        logger.debug(
            f"📚 Conversation history: {len(history) if history else 0} messages"
        )
        logger.debug(
            f"🤖 Controller status - LLM: {chat_controller.llm_available}, "
            f"Memory: {chat_controller.memory_available}, "
            f"Documents: {chat_controller.documents_available}"
        )

        # Use simplified chat controller
        chat_response = chat_controller.process_message(
            message=request.message,
            conversation_history=history,
            session_id=request.session_id,
        )

        logger.info(f"✅ Chat response generated - success: {chat_response.success}")
        logger.debug(f"📊 Response length: {len(chat_response.content)} characters")
        logger.debug(f"📈 Metadata: {chat_response.metadata}")

        if not chat_response.success:
            logger.error(f"❌ Chat processing failed: {chat_response.error}")

        # Convert SimpleChatResponse to API response
        api_response = ChatResponse(
            response=chat_response.content,
            session_id=chat_controller.current_session_id or "no_session",
            success=chat_response.success,
            error=chat_response.error,
            metadata=chat_response.metadata or {},
        )

        logger.info("📤 Sending chat response")
        return api_response

    except Exception as e:
        logger.error(f"💥 Error in chat_complete: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(
    request: ChatRequest, chat_controller: SimpleChatController = Depends(get_chat_controller)
):
    """Process chat message with streaming response."""
    logger.info("💬 Received chat streaming request")
    logger.debug(
        f"📝 Message: {request.message[:100]}{'...' if len(request.message) > 100 else ''}"
    )
    logger.debug(f"🔗 Session ID: {request.session_id}")

    async def generate_stream():
        try:
            logger.info("🔄 Processing chat message with streaming...")

            # Convert Pydantic model to dict for history
            history = (
                [msg.dict() for msg in request.conversation_history]
                if request.conversation_history
                else None
            )

            logger.debug(
                f"📚 Conversation history: {len(history) if history else 0} messages"
            )
            logger.debug(
                f"🤖 Controller status - LLM: {chat_controller.llm_available}, "
                f"Memory: {chat_controller.memory_available}, "
                f"Documents: {chat_controller.documents_available}"
            )

            # Use simplified chat controller's streaming method
            async for chunk in chat_controller.process_message_stream(
                message=request.message,
                conversation_history=history,
                session_id=request.session_id,
            ):
                # Send chunk as JSON
                yield f"data: {json.dumps({'chunk': chunk, 'session_id': chat_controller.current_session_id or 'no_session'})}\n\n"

            # Send end marker
            yield f"data: {json.dumps({'done': True, 'session_id': chat_controller.current_session_id or 'no_session'})}\n\n"

        except Exception as e:
            logger.error(f"💥 Error in chat_stream: {e}")
            error_data = {
                "error": str(e),
                "session_id": chat_controller.current_session_id or "no_session",
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/plain")


class ClearSessionRequest(BaseModel):
    """Request model for clearing chat session."""
    session_id: Optional[str] = None

@router.post("/clear")
async def clear_chat_session(
    request: ClearSessionRequest,
    chat_controller: SimpleChatController = Depends(get_chat_controller)
):
    """Clear the chat session and conversation history."""
    logger.info(f"💬 Received clear chat session request for session: {request.session_id or 'current'}")
    
    try:
        result = chat_controller.clear_session(request.session_id)
        
        if result["success"]:
            logger.info(f"✅ Chat session cleared successfully: {result['message']}")
        else:
            logger.warning(f"⚠️ Failed to clear chat session: {result['message']}")
        
        return result
    
    except Exception as e:
        logger.error(f"💥 Error in clear_chat_session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
