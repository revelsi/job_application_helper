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
from src.core.llm_providers.base import ProviderType
from src.core.llm_providers.factory import get_llm_provider
from src.core.simple_chat_controller import SimpleChatController
from src.utils.logging import get_logger

router = APIRouter(prefix="/chat", tags=["chat"])

logger = get_logger(__name__)


@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify chat router is working."""
    logger.info("🧪 Test endpoint called")
    return {"status": "ok", "message": "Chat router is working"}


def get_provider_for_request(request: ChatRequest):
    """Get the appropriate LLM provider based on request parameters."""
    if not request.provider:
        raise ValueError("Provider must be specified in the request")
    
    # Map provider string to ProviderType
    provider_mapping = {
        "openai": ProviderType.OPENAI,
        "mistral": ProviderType.MISTRAL,
        "novita": ProviderType.NOVITA,
        "ollama": ProviderType.OLLAMA,
    }
    
    if request.provider not in provider_mapping:
        raise ValueError(f"Unsupported provider: {request.provider}")
    
    try:
        provider_type = provider_mapping[request.provider]
        provider = get_llm_provider(provider_type)
        logger.info(f"🎯 Using requested provider: {request.provider}")
        return provider
    except Exception as e:
        logger.error(f"❌ Failed to get requested provider {request.provider}: {e}")
        raise ValueError(f"Provider {request.provider} is not available: {e}")


def get_chat_controller() -> SimpleChatController:
    """Dependency to get simplified chat controller instance."""
    logger.info("🚀 Starting simplified chat controller initialization...")
    logger.info("🔧 This is the get_chat_controller dependency function")

    try:
        # Initialize document service
        logger.info("📄 Initializing document service...")
        from src.core.simple_document_service import get_simple_document_service
        document_service = get_simple_document_service()

        # Initialize chat controller without a default LLM provider
        logger.info("💬 Initializing chat controller...")
        from src.core.simple_chat_controller import SimpleChatController
        chat_controller = SimpleChatController(
            llm_provider=None,  # No default provider - will be set per request
            document_service=document_service,
        )

        logger.info("✅ Chat controller initialization completed successfully")
        return chat_controller

    except Exception as e:
        logger.error(f"❌ Chat controller initialization failed: {e}")
        raise


@router.post("/complete", response_model=ChatResponse)
async def chat_complete(
    request: ChatRequest,
    chat_controller: SimpleChatController = Depends(get_chat_controller),
):
    """Complete chat endpoint with document context."""
    try:
        # Get the provider for this request
        llm_provider = get_provider_for_request(request)
        
        # Process the message with the specific provider
        response = chat_controller.process_message(
            message=request.message,
            llm_provider=llm_provider,
            conversation_history=request.conversation_history,
            session_id=request.session_id,
            model=request.model,
            reasoning_effort=request.reasoning_effort,
        )

        return ChatResponse(
            response=response.content,
            session_id=response.session_id or request.session_id or "default",
            success=response.success,
            error=response.error,
            metadata=response.metadata,
        )

    except Exception as e:
        logger.error(f"❌ Chat completion failed: {e}")
        return ChatResponse(
            response="",
            session_id=request.session_id or "error",
            success=False,
            error=str(e),
            metadata={},
        )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    chat_controller: SimpleChatController = Depends(get_chat_controller),
):
    """Streaming chat endpoint with document context."""
    logger.info(f"🎯 Chat stream endpoint called with provider: {request.provider}, model: {request.model}")
    async def generate_stream():
        try:
            # Send initial processing indicator
            yield f"data: {json.dumps({'type': 'processing', 'content': 'Processing your request...'})}\n\n"
            
            logger.info(f"🔍 Getting provider for request: {request.provider}")
            # Get the provider for this request
            llm_provider = get_provider_for_request(request)
            logger.info(f"✅ Got provider: {llm_provider.provider_type}")
            
            # Accumulate the complete response
            complete_response = ""
            
            # Process the message with the specific provider
            logger.info("🚀 Starting chat_controller.process_message_stream...")
            async for chunk in chat_controller.process_message_stream(
                message=request.message,
                llm_provider=llm_provider,
                conversation_history=request.conversation_history,
                session_id=request.session_id,
                model=request.model,
                reasoning_effort=request.reasoning_effort,
            ):
                logger.debug(f"📦 Chat endpoint received chunk: {chunk[:50]}...")
                complete_response += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            
            # Send final complete answer
            yield f"data: {json.dumps({'type': 'answer', 'content': complete_response})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"❌ Chat streaming failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream")


class ClearSessionRequest(BaseModel):
    """Request model for clearing chat session."""

    session_id: Optional[str] = None


@router.post("/clear")
async def clear_chat_session(
    request: ClearSessionRequest,
    chat_controller: SimpleChatController = Depends(get_chat_controller),
):
    """Clear the chat session and conversation history."""
    logger.info(
        f"💬 Received clear chat session request for session: {request.session_id or 'current'}"
    )

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
