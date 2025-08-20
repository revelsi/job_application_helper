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
Credentials Manager for Job Application Helper.

Handles secure storage and retrieval of API keys and other sensitive credentials.
Provides encryption at rest and clean separation from document storage.
"""

import json
from pathlib import Path
from typing import Dict, Optional

try:
    from cryptography.fernet import Fernet

    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

from src.utils.config import get_settings
from src.utils.logging import get_logger


class CredentialsManager:
    """Manages secure storage of API keys and credentials."""

    def __init__(self, credentials_path: Optional[Path] = None):
        """
        Initialize credentials manager.

        Args:
            credentials_path: Path to credentials file. If None, uses default.
        """
        self.logger = get_logger(f"{__name__}.CredentialsManager")
        self.settings = get_settings()

        # Set credentials file path
        if credentials_path:
            self.credentials_path = credentials_path
        else:
            self.credentials_path = self.settings.data_dir / "credentials.enc"

        # Ensure parent directory exists
        self.credentials_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize encryption
        self._fernet = None
        self._initialize_encryption()

    def _initialize_encryption(self):
        """Initialize encryption key."""
        if not ENCRYPTION_AVAILABLE:
            self.logger.warning(
                "Cryptography package not available - credentials will not be encrypted"
            )
            return

        if not self.settings.enable_encryption:
            self.logger.info("Encryption disabled in configuration")
            return

        try:
            # Get or generate encryption key
            encryption_key = self._get_or_create_encryption_key()
            if encryption_key:
                self._fernet = Fernet(
                    encryption_key.encode()
                    if isinstance(encryption_key, str)
                    else encryption_key
                )
                self.logger.info("Credentials encryption initialized")
            else:
                self.logger.warning(
                    "Could not initialize encryption - credentials will be stored in plain text"
                )
        except Exception as e:
            self.logger.error(f"Failed to initialize encryption: {e}")

    def _get_or_create_encryption_key(self) -> Optional[str]:
        """Get existing encryption key or create a new one."""
        # Priority 1: Check for key in settings (environment variable)
        if (
            self.settings.encryption_key
            and self.settings.encryption_key != "your_encryption_key_here"
        ):
            self.logger.info("Using encryption key from environment variable")
            return self.settings.encryption_key

        # Priority 2: Check for existing key file
        key_file = self.settings.data_dir / ".encryption_key"
        if key_file.exists():
            try:
                key = key_file.read_text().strip()
                if key and len(key) > 0:
                    self.logger.info(f"Using existing encryption key from {key_file}")
                    return key
            except Exception as e:
                self.logger.error(f"Failed to read existing encryption key: {e}")

        # Priority 3: Generate new key only if no existing key found
        self.logger.warning("No existing encryption key found - generating new key")
        self.logger.warning(
            "⚠️  WARNING: This will make existing encrypted credentials unrecoverable!"
        )

        key = Fernet.generate_key().decode()

        # Save key to a secure location
        try:
            key_file.write_text(key)
            key_file.chmod(0o600)  # Restrict permissions
            self.logger.info(f"New encryption key saved to {key_file}")
            return key
        except Exception as e:
            self.logger.error(f"Failed to save encryption key: {e}")
            return key  # Return key even if we couldn't save it

    def _encrypt_data(self, data: str) -> str:
        """Encrypt data if encryption is available."""
        if self._fernet:
            return self._fernet.encrypt(data.encode()).decode()
        return data

    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data if encryption is available."""
        if self._fernet:
            try:
                return self._fernet.decrypt(encrypted_data.encode()).decode()
            except Exception as e:
                self.logger.error(f"Failed to decrypt data: {e}")
                self.logger.error(
                    "This usually means the encryption key has changed or is incorrect"
                )
                self.logger.error("Existing encrypted credentials may be unrecoverable")
                # Return as-is if decryption fails (might be plain text)
                return encrypted_data
        return encrypted_data

    def _load_credentials(self) -> Dict[str, str]:
        """Load credentials from file."""
        if not self.credentials_path.exists():
            return {}

        try:
            encrypted_content = self.credentials_path.read_text()
            if not encrypted_content.strip():
                return {}

            # Decrypt if encryption is enabled
            content = self._decrypt_data(encrypted_content)

            # Parse JSON
            credentials = json.loads(content)
            self.logger.debug("Credentials loaded successfully")
            return credentials

        except Exception as e:
            self.logger.error(f"Failed to load credentials: {e}")
            return {}

    def _save_credentials(self, credentials: Dict[str, str]) -> bool:
        """Save credentials to file."""
        try:
            # Convert to JSON
            content = json.dumps(credentials, indent=2)

            # Encrypt if encryption is enabled
            encrypted_content = self._encrypt_data(content)

            # Write to file
            self.credentials_path.write_text(encrypted_content)
            self.credentials_path.chmod(0o600)  # Restrict permissions

            self.logger.debug("Credentials saved successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save credentials: {e}")
            return False

    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a provider.

        Args:
            provider: Provider name (e.g., 'openai')

        Returns:
            API key if found, None otherwise
        """
        credentials = self._load_credentials()
        api_key = credentials.get(f"{provider}_api_key")

        if api_key and api_key.strip() and api_key != f"your_{provider}_api_key_here":
            return api_key.strip()

        return None

    def set_api_key(self, provider: str, api_key: str) -> bool:
        """
        Set API key for a provider.

        Args:
            provider: Provider name (e.g., 'openai')
            api_key: API key to store

        Returns:
            True if successful, False otherwise
        """
        if not api_key or not api_key.strip():
            self.logger.warning(f"Empty API key provided for {provider}")
            return False

        credentials = self._load_credentials()
        credentials[f"{provider}_api_key"] = api_key.strip()

        success = self._save_credentials(credentials)
        if success:
            self.logger.info(f"API key set for {provider}")

        return success

    def remove_api_key(self, provider: str) -> bool:
        """
        Remove API key for a provider.

        Args:
            provider: Provider name (e.g., 'openai')

        Returns:
            True if successful, False otherwise
        """
        credentials = self._load_credentials()
        key_name = f"{provider}_api_key"

        if key_name in credentials:
            del credentials[key_name]
            success = self._save_credentials(credentials)
            if success:
                self.logger.info(f"API key removed for {provider}")
            return success

        # Key doesn't exist, but that's not an error
        return True

    def list_configured_providers(self) -> Dict[str, bool]:
        """
        List providers that have API keys configured.

        Returns:
            Dictionary mapping provider names to configuration status
        """
        credentials = self._load_credentials()
        result = {}

        # Check for known providers
        for provider in ["openai"]:
            key_name = f"{provider}_api_key"
            has_key = (
                key_name in credentials
                and credentials[key_name]
                and credentials[key_name].strip()
                and credentials[key_name] != f"your_{provider}_api_key_here"
            )
            result[provider] = has_key

        return result

    def clear_all_credentials(self) -> bool:
        """
        Clear all stored credentials.

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.credentials_path.exists():
                self.credentials_path.unlink()
            self.logger.info("All credentials cleared")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear credentials: {e}")
            return False

    def get_credentials_info(self) -> Dict[str, any]:
        """
        Get information about the credentials storage.

        Returns:
            Dictionary with credentials system info
        """
        return {
            "credentials_file": str(self.credentials_path),
            "file_exists": self.credentials_path.exists(),
            "encryption_enabled": self._fernet is not None,
            "encryption_available": ENCRYPTION_AVAILABLE,
            "configured_providers": self.list_configured_providers(),
        }

    def clear_cache(self) -> None:
        """
        Clear any internal caching to ensure fresh credential loading.
        This is useful when credentials are updated externally.
        """
        # Force reload of credentials on next access
        # The _load_credentials method already reads from disk each time,
        # so there's no actual caching to clear, but this method provides
        # a clear interface for cache invalidation if needed in the future
        self.logger.debug(
            "Credentials cache cleared (credentials are always loaded fresh from disk)"
        )

    def force_reload(self) -> Dict[str, str]:
        """
        Force reload credentials from disk, bypassing any potential caching.

        Returns:
            Freshly loaded credentials dictionary
        """
        self.logger.debug("Forcing credential reload from disk")
        return self._load_credentials()


# Global credentials manager instance
_credentials_manager = None


def get_credentials_manager() -> CredentialsManager:
    """Get the global credentials manager instance."""
    global _credentials_manager
    if _credentials_manager is None:
        _credentials_manager = CredentialsManager()
    return _credentials_manager
