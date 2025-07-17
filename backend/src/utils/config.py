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
Configuration management for Job Application Helper.

This module handles all configuration settings including:
- Environment variables
- API keys
- File paths
- Security settings
"""

from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation."""

    # Environment
    environment: str = Field(default="production", env="ENVIRONMENT")
    host: str = Field(default="127.0.0.1", env="HOST")
    port: int = Field(default=8000, env="PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    default_llm_provider: str = Field(default="openai", env="DEFAULT_LLM_PROVIDER")

    # External APIs
    linkedin_api_key: Optional[str] = Field(default=None, env="LINKEDIN_API_KEY")

    # Data Storage
    data_dir: Path = Field(default=Path("./data"), env="DATA_DIR")
    documents_path: Path = Field(default=Path("./data/documents"), env="DOCUMENTS_PATH")
    cache_path: Path = Field(default=Path("./data/cache"), env="CACHE_PATH")

    # Security
    enable_encryption: bool = Field(default=True, env="ENABLE_ENCRYPTION")
    encryption_key: Optional[str] = Field(default=None, env="ENCRYPTION_KEY")
    max_upload_size_mb: int = Field(default=10, env="MAX_UPLOAD_SIZE_MB")

    # Document Processing Configuration
    max_context_length: int = Field(default=24000, env="MAX_CONTEXT_LENGTH")

    # Chat Generation Configuration
    chat_max_tokens: int = Field(default=2000, env="CHAT_MAX_TOKENS")
    conversation_context_limit: int = Field(default=3, env="CONVERSATION_CONTEXT_LIMIT")

    # Multi-Query Configuration
    enable_multi_query_detection: bool = Field(
        default=True, env="ENABLE_MULTI_QUERY_DETECTION"
    )
    max_sub_queries: int = Field(default=3, env="MAX_SUB_QUERIES")

    # API Rate Limiting
    api_rate_limit: int = Field(default=60, env="API_RATE_LIMIT")

    @validator(
        "data_dir", "documents_path", "cache_path", pre=True
    )
    @classmethod
    def convert_to_path(cls, v):
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v

    @validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


def ensure_directories(settings: Settings) -> None:
    """Ensure all required directories exist."""
    directories = [
        settings.data_dir,
        settings.documents_path,
        settings.cache_path,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def ensure_encryption_setup(settings: Settings) -> Optional[str]:
    """
    Ensure encryption is properly set up if enabled.

    Returns:
        Encryption key if encryption is enabled and set up, None otherwise
    """
    if not settings.enable_encryption:
        return None

    # Priority 1: Check if we already have a valid key in settings
    if (
        settings.encryption_key
        and settings.encryption_key != "your_encryption_key_here"
    ):
        return settings.encryption_key

    # Priority 2: Check for existing key file (for Docker compatibility)
    key_file = settings.data_dir / ".encryption_key"
    if key_file.exists():
        try:
            key = key_file.read_text().strip()
            if key and len(key) > 0:
                print("🔐 Job Application Helper - Security Setup")
                print("   ✅ Using existing encryption key from data directory")
                return key
        except Exception as e:
            print(f"⚠️  Warning: Failed to read existing encryption key: {e}")

    # Priority 3: Generate new key only if no existing key found
    print("🔐 Job Application Helper - Security Setup")
    print("   ⚠️  No existing encryption key found - generating new key")
    print("   ⚠️  WARNING: This will make existing encrypted data unrecoverable!")
    
    key = Fernet.generate_key().decode()
    success = update_env_file("ENCRYPTION_KEY", key)

    if success:
        print("   ✅ Encryption enabled for data protection")
        print("   🔑 Generated new encryption key")
        print("   📁 Key saved to .env file")
        print("   💡 Tip: Back up your .env file to preserve access to your data")
        print("   ⚙️  To disable encryption, set ENABLE_ENCRYPTION=false in .env")
        return key
    else:
        print("   ⚠️  Warning: Could not save encryption key to .env file")
        print("   Encryption will work for this session only")
        return key


def update_env_file(key: str, value: str) -> bool:
    """
    Update or add a key-value pair in the .env file.

    Args:
        key: Environment variable name
        value: Environment variable value

    Returns:
        True if successful, False otherwise
    """
    try:
        env_path = Path(".env")

        # Read existing content
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = []

        # Find and update existing key or add new one
        key_pattern = f"{key}="
        updated = False

        for i, line in enumerate(lines):
            if line.strip().startswith(key_pattern):
                lines[i] = f"{key}={value}\n"
                updated = True
                break

        if not updated:
            # Add new key-value pair
            if lines and not lines[-1].endswith("\n"):
                lines.append("\n")
            lines.append(f"{key}={value}\n")

        # Write back to file
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return True

    except Exception as e:
        print(f"Error updating .env file: {e}")
        return False


def get_encryption_key(settings: Settings) -> bytes:
    """Get or generate encryption key (legacy function)."""
    if settings.encryption_key:
        return settings.encryption_key.encode()

    # Generate a new key if none provided
    key = Fernet.generate_key()
    print(f"Warning: No encryption key provided. Generated new key: {key.decode()}")
    print(
        "Please add this key to your .env file as ENCRYPTION_KEY for data persistence."
    )
    return key


def validate_required_settings(settings: Settings) -> None:
    """Validate that required settings are provided."""
    errors = []

    # Check for placeholder values
    placeholder_values = {
        "your_openai_api_key_here",
        "your_encryption_key_here",
        "your_linkedin_api_key_here",
    }

    openai_valid = (
        settings.openai_api_key and settings.openai_api_key not in placeholder_values
    )

    if not openai_valid:
        errors.append(
            "OPENAI_API_KEY must be provided"
        )

    if errors:
        raise ValueError(
            "Configuration errors:\n" + "\n".join(f"- {error}" for error in errors)
        )


# Global settings instance
settings = get_settings()
