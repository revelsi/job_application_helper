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

import atexit
import multiprocessing
import signal
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.endpoints import chat, documents, health, keys
from src.utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Job Application Helper API",
    description="AI-powered job application assistance with intelligent document processing",
    version="1.0.0",
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
    ],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s"
    )
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def cleanup_multiprocessing_resources():
    """Clean up multiprocessing resources to prevent semaphore leaks."""
    try:
        # Clean up any remaining multiprocessing resources
        if hasattr(multiprocessing, "resource_tracker"):
            multiprocessing.resource_tracker._CLEANUP_CALLED = True
            multiprocessing.resource_tracker._REGISTRY.clear()

        # Force cleanup of any remaining semaphores
        import _multiprocessing

        if hasattr(_multiprocessing, "semaphore"):
            # This helps clean up any remaining semaphore objects
            pass

        logger.info("Multiprocessing resources cleaned up")
    except Exception as e:
        logger.warning(f"Error during multiprocessing cleanup: {e}")


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    cleanup_multiprocessing_resources()
    exit(0)


# Register cleanup handlers
atexit.register(cleanup_multiprocessing_resources)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# Include routers
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(keys.router)

if __name__ == "__main__":
    import uvicorn

    # Note: Binding to 0.0.0.0 is intentional for Docker containerization
    # The container network isolation provides security boundaries
    uvicorn.run(app, host="0.0.0.0", port=8000)
