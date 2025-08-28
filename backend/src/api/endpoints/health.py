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

from fastapi import APIRouter

from src.api.models import HealthResponse
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Simple health check endpoint."""
    logger.info("🔍 Health check requested")
    
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(),
        services={
            "llm": True,
            "documents": True,
            "storage": True,
            "backend_ready": True
        }
    )
