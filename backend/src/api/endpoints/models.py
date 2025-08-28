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

from fastapi import APIRouter

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
async def get_models_for_provider_endpoint(provider: str) -> List[Dict[str, Any]]:
    """Get models for a specific provider with display information."""
    models = get_models_for_provider(provider)
    model_info = []
    
    for model in models:
        display_info = get_model_display_info(model.name)
        model_info.append(display_info)
    
    return model_info


@router.get("/info/{model_name}")
async def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific model."""
    return get_model_display_info(model_name)
