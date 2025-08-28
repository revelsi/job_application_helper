#!/usr/bin/env python3
"""
FastAPI Backend Startup Script
Properly configures Python path and starts the API server
"""

import multiprocessing
import os
from pathlib import Path
import sys

# Configure multiprocessing to prevent semaphore leaks on macOS
if sys.platform == "darwin":  # macOS
    multiprocessing.set_start_method("spawn", force=True)

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Set environment variables for proper module resolution
os.environ["PYTHONPATH"] = str(backend_dir)

if __name__ == "__main__":
    import uvicorn

    # Start the server using import string for proper reload support
    uvicorn.run(
        "src.api.main:app",  # Use import string instead of app object
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(backend_dir)],
        log_level="info",
    )
