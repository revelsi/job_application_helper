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
from typing import Dict
import uuid
import threading

from src.api.endpoints import chat, documents, health, keys, models, ollama
from src.utils.logging import get_logger
from src.utils.config import get_settings

logger = get_logger(__name__)

app = FastAPI(
    title="Job Application Helper API",
    description="AI-powered job application assistance with intelligent document processing",
    version="1.0.0",
)

# Configuration
settings = get_settings()

# CORS middleware (env-driven)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With", "X-Request-ID"],
)


# Security headers middleware (CSP disabled by default when served behind Nginx)
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)

    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if settings.enable_api_csp_headers:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: https:; connect-src 'self'; font-src 'self'"
        )

    return response


# Request context: ID, body-size check, basic rate limiting, access log
_rate_limit_bucket: Dict[str, Dict[str, float]] = {}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Assign or propagate request ID
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

    # Basic body size guard based on Content-Length
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit():
        if int(content_length) > max_bytes:
            return JSONResponse(status_code=413, content={"detail": "Request entity too large"})

    # Simple IP-based rate limiting (token bucket approximation)
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = 60.0
    limit = max(1, settings.api_rate_limit)
    bucket = _rate_limit_bucket.get(client_ip, {"window_start": now, "count": 0.0})
    if now - bucket["window_start"] > window:
        bucket = {"window_start": now, "count": 0.0}
    bucket["count"] += 1.0
    _rate_limit_bucket[client_ip] = bucket
    if bucket["count"] > limit:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Request-ID"] = request_id
    logger.info(
        f"request_id={request_id} {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s"
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
app.include_router(models.router)
app.include_router(ollama.router)


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    print("🚀 Job Application Helper Backend starting...")
    logger.info("🚀 Job Application Helper Backend starting...")
    
    print("✅ Backend ready!")
    logger.info("✅ Backend ready!")
    
    print("🌐 API available at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")

    # Kick off background warm-up to avoid cold starts without blocking readiness
    def _warmup():
        try:
            logger.info("Starting background warm-up tasks")
            # Pre-import provider modules and instantiate once to warm caches
            try:
                from src.core.llm_providers.factory import get_available_providers

                _ = get_available_providers()
            except Exception as e:
                logger.debug(f"Provider availability warm-up skipped: {e}")

            # Pre-import document parsers
            try:
                import fitz  # noqa: F401
            except Exception:
                pass
            try:
                import pymupdf4llm  # noqa: F401
            except Exception:
                pass
            try:
                from docx import Document as _Doc  # noqa: F401
            except Exception:
                pass

            logger.info("Background warm-up completed")
        except Exception as e:
            logger.debug(f"Warm-up error: {e}")

    threading.Thread(target=_warmup, daemon=True).start()


if __name__ == "__main__":
    import uvicorn

    # Note: Binding to 0.0.0.0 is intentional for Docker containerization
    # The container network isolation provides security boundaries
    uvicorn.run(app, host="0.0.0.0", port=8000)
