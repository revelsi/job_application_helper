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
Security utilities for Job Application Helper.

Provides input sanitization and validation functions to protect against
security vulnerabilities in user inputs and AI interactions.
"""

import re
from pathlib import Path
from typing import Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)


class PromptSanitizer:
    """Sanitizes user prompts to prevent injection attacks."""

    # Patterns that could indicate prompt injection attempts
    DANGEROUS_PATTERNS = [
        r"ignore\s+previous\s+instructions",
        r"forget\s+everything\s+above",
        r"new\s+instructions:",
        r"system\s*:",
        r"assistant\s*:",
        r"human\s*:",
        r"<\s*system\s*>",
        r"<\s*\/?\s*assistant\s*>",
        r"<\s*\/?\s*human\s*>",
        r"override\s+safety\s+protocols",
        r"disregard\s+previous\s+context",
        r"act\s+as\s+if\s+you\s+are",
        r"pretend\s+to\s+be",
        r"roleplay\s+as",
    ]

    # Maximum safe prompt length
    MAX_PROMPT_LENGTH = 8000

    def __init__(self):
        """Initialize the prompt sanitizer."""
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.DANGEROUS_PATTERNS
        ]

    def sanitize_prompt(self, prompt: str) -> str:
        """
        Sanitize a user prompt to prevent injection attacks.

        Args:
            prompt: Raw user input prompt

        Returns:
            Sanitized prompt safe for LLM processing
        """
        if not isinstance(prompt, str):
            logger.warning("Non-string prompt provided to sanitizer")
            return ""

        original_length = len(prompt)

        # Truncate to safe length
        if len(prompt) > self.MAX_PROMPT_LENGTH:
            prompt = prompt[: self.MAX_PROMPT_LENGTH]
            logger.info(
                f"Truncated prompt from {original_length} to {len(prompt)} characters"
            )

        # Remove dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(prompt):
                prompt = pattern.sub("[FILTERED]", prompt)
                logger.warning(
                    "Detected and filtered potential prompt injection attempt"
                )

        # Clean up multiple spaces and newlines
        prompt = re.sub(r"\s+", " ", prompt).strip()

        return prompt

    def validate_prompt(self, prompt: str) -> tuple[bool, Optional[str]]:
        """
        Validate a prompt for safety without modifying it.

        Args:
            prompt: Prompt to validate

        Returns:
            Tuple of (is_safe, warning_message)
        """
        if not isinstance(prompt, str):
            return False, "Invalid prompt type"

        if len(prompt) > self.MAX_PROMPT_LENGTH:
            return (
                False,
                f"Prompt too long ({len(prompt)} > {self.MAX_PROMPT_LENGTH} chars)",
            )

        for pattern in self.compiled_patterns:
            if pattern.search(prompt):
                return False, "Potential prompt injection detected"

        return True, None


class InputValidator:
    """Validates various types of user input for security."""

    @staticmethod
    def validate_filename(filename: str) -> tuple[bool, Optional[str]]:
        """
        Validate filename for security.

        Args:
            filename: Filename to validate

        Returns:
            Tuple of (is_safe, error_message)
        """
        if not isinstance(filename, str):
            return False, "Filename must be a string"

        # Check for path traversal attempts
        # First check for directory separators
        if "/" in filename or "\\" in filename:
            return False, "Filename contains invalid path characters"

        # Check for dangerous parent directory references
        # Only flag ".." if it's the entire component or part of a path
        if (
            filename == ".."
            or filename.startswith("../")
            or filename.startswith("..\\")
        ):
            return False, "Filename contains invalid path characters"

        # Check for dangerous extensions
        dangerous_extensions = {".exe", ".bat", ".cmd", ".sh", ".ps1", ".scr", ".vbs"}
        file_path = Path(filename)
        if file_path.suffix.lower() in dangerous_extensions:
            return False, "Dangerous file extension detected"

        # Check filename length
        if len(filename) > 255:
            return False, "Filename too long"

        return True, None

    @staticmethod
    def sanitize_error_message(error_msg: str, user_facing: bool = True) -> str:
        """
        Sanitize error messages to prevent information disclosure.

        Args:
            error_msg: Original error message
            user_facing: Whether this message will be shown to users

        Returns:
            Sanitized error message
        """
        if not user_facing:
            return error_msg  # Keep full details for logs

        # Remove potentially sensitive information from user-facing errors
        sensitive_patterns = [
            r"/[^/]+/[^/]+/[^/]+/",  # File paths
            r"[A-Za-z]:\\[^\\]+\\[^\\]+\\",  # Windows paths
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Email addresses
            r"sk-[a-zA-Z0-9]{40,}",  # OpenAI API keys (more flexible pattern)
            r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}",  # UUIDs
        ]

        sanitized = error_msg
        for pattern in sensitive_patterns:
            sanitized = re.sub(pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE)

        # Generic user-friendly messages for common errors
        if "permission" in sanitized.lower():
            return "Access denied. Please check file permissions."
        elif "not found" in sanitized.lower():
            return "File or resource not found."
        elif "connection" in sanitized.lower():
            return "Connection error. Please check your internet connection."

        return sanitized


# Global instances
_prompt_sanitizer = None
_input_validator = None


def get_prompt_sanitizer() -> PromptSanitizer:
    """Get the global prompt sanitizer instance."""
    global _prompt_sanitizer
    if _prompt_sanitizer is None:
        _prompt_sanitizer = PromptSanitizer()
    return _prompt_sanitizer


def get_input_validator() -> InputValidator:
    """Get the global input validator instance."""
    global _input_validator
    if _input_validator is None:
        _input_validator = InputValidator()
    return _input_validator
