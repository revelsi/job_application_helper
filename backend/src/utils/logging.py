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
Logging configuration for Job Application Helper.

This module provides centralized logging configuration with:
- JSON structured logging for production
- Colorful console logging for development
- Security-aware logging (no sensitive data)
"""

import logging
from pathlib import Path
import sys
from typing import Optional

from pythonjsonlogger import jsonlogger

from src.utils.config import Settings


class SecurityFilter(logging.Filter):
    """Filter out sensitive information from logs."""

    SENSITIVE_KEYS = {
        "api_key",
        "password",
        "token",
        "secret",
        "encryption_key",
        "openai_api_key",
        "linkedin_api_key",
    }

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter sensitive information from log records."""
        if hasattr(record, "msg") and isinstance(record.msg, str):
            msg_lower = record.msg.lower()
            for sensitive_key in self.SENSITIVE_KEYS:
                if sensitive_key in msg_lower:
                    # Mask the sensitive information
                    record.msg = record.msg.replace(
                        record.msg[record.msg.lower().find(sensitive_key) :],
                        f"{sensitive_key}=***MASKED***",
                    )
        return True


def setup_logging(settings: Settings, log_file: Optional[Path] = None) -> None:
    """
    Setup logging configuration.

    Args:
        settings: Application settings
        log_file: Optional log file path
    """
    # Create logs directory
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level))

    if settings.environment == "development":
        # Development: colorful and readable
        console_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        # Production: JSON structured logging
        console_format = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"
        )

    console_handler.setFormatter(console_format)
    console_handler.addFilter(SecurityFilter())
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        file_format = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(funcName)s %(lineno)d %(message)s"
        )
        file_handler.setFormatter(file_format)
        file_handler.addFilter(SecurityFilter())
        root_logger.addHandler(file_handler)

    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    # Enable verbose logging for our application modules in development
    if settings.environment == "development":
        logging.getLogger("src.core").setLevel(logging.DEBUG)
        logging.getLogger("src.api").setLevel(logging.DEBUG)
        logging.getLogger("src.utils").setLevel(logging.DEBUG)

        # Enable more detailed logging for specific modules
        logging.getLogger("src.core.llm_providers").setLevel(logging.DEBUG)
        logging.getLogger("src.core.chat_controller").setLevel(logging.DEBUG)
        logging.getLogger("src.core.simple_chat_controller").setLevel(logging.DEBUG)
        logging.getLogger("src.core.credentials").setLevel(logging.DEBUG)

        # Also enable DEBUG level for the console handler in development
        console_handler.setLevel(logging.DEBUG)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)
