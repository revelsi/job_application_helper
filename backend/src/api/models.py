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

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(
        None, description="Session ID for conversation continuity"
    )
    conversation_history: Optional[List[Message]] = Field(
        [], description="Recent conversation context"
    )


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI response content")
    session_id: str = Field(..., description="Session ID")
    success: bool = Field(True, description="Whether the request was successful")
    error: Optional[str] = Field(None, description="Error message if any")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Response metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Based on your CV, I can help you write a cover letter...",
                "session_id": "session_123",
                "success": True,
                "metadata": {
                    "provider": "openai",
                    "tokens_used": 250,
                    "document_context_used": True,
                    "documents_referenced": 2,
                },
            }
        }


class DocumentUploadResponse(BaseModel):
    document_id: str = Field(..., description="Unique document identifier")
    success: bool = Field(True, description="Whether upload was successful")
    message: str = Field(..., description="Status message")
    file_name: Optional[str] = Field(None, description="Uploaded file name")
    error: Optional[str] = Field(None, description="Error message if any")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Upload metadata"
    )


class HealthResponse(BaseModel):
    status: str = Field("ok", description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Dict[str, bool] = Field(
        default_factory=dict, description="Service availability"
    )
