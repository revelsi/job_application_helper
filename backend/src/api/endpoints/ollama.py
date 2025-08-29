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
Ollama Model Management API Endpoints.

Provides endpoints for managing Ollama models including listing, downloading,
and checking availability.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
import httpx
from pydantic import BaseModel, Field

from src.core.llm_providers.ollama_provider import get_ollama_provider
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ollama", tags=["ollama"])


class ModelInfo(BaseModel):
    """Model information response."""
    name: str = Field(..., description="Model name")
    size: Optional[int] = Field(None, description="Model size in bytes")
    modified_at: Optional[str] = Field(None, description="Last modified timestamp")
    digest: Optional[str] = Field(None, description="Model digest")


class ModelDownloadRequest(BaseModel):
    """Request model for downloading a model."""
    model: str = Field(..., description="Model name to download")


class ModelDownloadResponse(BaseModel):
    """Response model for model download operations."""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Status message")
    model: str = Field(..., description="Model name")
    error: Optional[str] = Field(None, description="Error message if any")


@router.get("/models", response_model=List[ModelInfo])
async def list_available_models():
    """Get list of locally available Ollama models."""
    try:
        provider = get_ollama_provider()
        if not provider.is_available():
            raise HTTPException(
                status_code=503,
                detail="Ollama service is not available. Please ensure Ollama is running."
            )
        
        models = provider._get_available_models()
        return [
            ModelInfo(
                name=model["name"],
                size=model.get("size"),
                modified_at=model.get("modified_at"),
                digest=model.get("digest")
            )
            for model in models
        ]
    except Exception as e:
        logger.error(f"Failed to list Ollama models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {e!s}"
        )


@router.post("/models/download", response_model=ModelDownloadResponse)
async def download_model(request: ModelDownloadRequest):
    """Download a model from Ollama library."""
    try:
        provider = get_ollama_provider()
        if not provider.is_available():
            raise HTTPException(
                status_code=503,
                detail="Ollama service is not available. Please ensure Ollama is running."
            )
        
        # Check if model is already available
        if provider._check_model_available(request.model):
            return ModelDownloadResponse(
                success=True,
                message=f"Model '{request.model}' is already available",
                model=request.model
            )
        
        # Download the model
        success = await provider._download_model(request.model)
        
        if success:
            return ModelDownloadResponse(
                success=True,
                message=f"Successfully downloaded model '{request.model}'",
                model=request.model
            )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download model '{request.model}'"
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download model {request.model}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download model: {e!s}"
        )


@router.get("/models/{model_name}/status")
async def check_model_status(model_name: str):
    """Check if a specific model is available."""
    try:
        provider = get_ollama_provider()
        if not provider.is_available():
            raise HTTPException(
                status_code=503,
                detail="Ollama service is not available. Please ensure Ollama is running."
            )
        
        is_available = provider._check_model_available(model_name)
        
        return {
            "model": model_name,
            "available": is_available,
            "message": f"Model '{model_name}' is {'available' if is_available else 'not available'}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check model status for {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check model status: {e!s}"
        )


@router.get("/status")
async def get_ollama_status():
    """Get Ollama service status."""
    try:
        provider = get_ollama_provider()
        
        # Check if Ollama service is running
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{provider.base_url}/api/tags")
                service_running = response.status_code == 200
        except Exception:
            service_running = False
        
        if not service_running:
            return {
                "status": "unavailable",
                "message": "Ollama service is not running. Please start Ollama with 'ollama serve'",
                "available_models": 0,
                "models": [],
                "required_models": ["gemma3:1b", "llama3.2:1b"]
            }
        
        # Check if required models are available
        models = provider._get_available_models()
        available_model_names = [model["name"] for model in models]
        required_models = ["gemma3:1b", "llama3.2:1b"]
        missing_models = [model for model in required_models if model not in available_model_names]
        
        if missing_models:
            return {
                "status": "unavailable",
                "message": f"Ollama is running but missing required models: {', '.join(missing_models)}. Run the setup script to download models.",
                "available_models": len(models),
                "models": available_model_names,
                "required_models": required_models,
                "missing_models": missing_models
            }
        
        # All good
        return {
            "status": "available",
            "message": f"Ollama service is running with {len(models)} models available",
            "available_models": len(models),
            "models": available_model_names,
            "required_models": required_models
        }
    except Exception as e:
        logger.error(f"Failed to get Ollama status: {e}")
        return {
            "status": "error",
            "message": f"Failed to check Ollama status: {e!s}",
            "available_models": 0,
            "models": [],
            "required_models": ["gemma3:1b", "llama3.2:1b"]
        }
