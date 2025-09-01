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
Model Information API Endpoints.

Provides endpoints for retrieving model configurations and capabilities.
"""

from typing import Any, Dict, List
import re

from fastapi import APIRouter, HTTPException, Path
from pydantic import validator

from src.core.llm_providers.model_config import (
    get_model_display_info,
    get_models_for_provider,
)

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/providers")
async def get_provider_models() -> Dict[str, List[Dict[str, Any]]]:
    """Get all available models grouped by provider with display information."""
    provider_models = {}
    
    for provider in ["openai", "mistral", "novita", "ollama"]:
        models = get_models_for_provider(provider)
        provider_models[provider] = []
        
        for model in models:
            display_info = get_model_display_info(model.name)
            provider_models[provider].append(display_info)
    
    return provider_models


@router.get("/provider/{provider}")
async def get_models_for_provider_endpoint(
    provider: str = Path(..., regex="^[a-zA-Z0-9_-]+$", min_length=1, max_length=50)
) -> List[Dict[str, Any]]:
    """Get models for a specific provider with display information."""
    # Validate provider name against allowed providers
    allowed_providers = ["openai", "mistral", "novita", "ollama"]
    if provider not in allowed_providers:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid provider '{provider}'. Allowed providers: {', '.join(allowed_providers)}"
        )
    
    models = get_models_for_provider(provider)
    model_info = []
    
    for model in models:
        display_info = get_model_display_info(model.name)
        model_info.append(display_info)
    
    return model_info


@router.get("/info/{model_name}")
async def get_model_info(
    model_name: str = Path(..., regex="^[a-zA-Z0-9._-]+$", min_length=1, max_length=100)
) -> Dict[str, Any]:
    """Get detailed information about a specific model."""
    # Sanitize model name to prevent injection
    if not re.match(r"^[a-zA-Z0-9._-]+$", model_name):
        raise HTTPException(
            status_code=400, 
            detail="Invalid model name format. Only alphanumeric characters, dots, underscores, and hyphens allowed."
        )
    
    try:
        model_info = get_model_display_info(model_name)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}") from e
