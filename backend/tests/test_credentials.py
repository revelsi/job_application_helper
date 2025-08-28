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
Tests for the CredentialsManager.

Tests secure storage and retrieval of API keys and credentials.
"""

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from src.core.credentials import CredentialsManager, get_credentials_manager


class TestCredentialsManager:
    """Test cases for CredentialsManager."""

    def test_init_creates_credentials_file_path(self):
        """Test that initialization creates proper file path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            creds_path = Path(temp_dir) / "test_creds.enc"
            manager = CredentialsManager(credentials_path=creds_path)

            assert manager.credentials_path == creds_path
            assert manager.credentials_path.parent.exists()

    @patch("src.core.credentials.ENCRYPTION_AVAILABLE", False)
    def test_init_without_encryption(self):
        """Test initialization when encryption is not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            creds_path = Path(temp_dir) / "test_creds.enc"
            manager = CredentialsManager(credentials_path=creds_path)

            assert manager._fernet is None

    def test_set_and_get_api_key(self):
        """Test storing and retrieving API keys."""
        with tempfile.TemporaryDirectory() as temp_dir:
            creds_path = Path(temp_dir) / "test_creds.enc"
            manager = CredentialsManager(credentials_path=creds_path)

            # Set API key
            success = manager.set_api_key("openai", "test-api-key-123")
            assert success

            # Get API key
            retrieved_key = manager.get_api_key("openai")
            assert retrieved_key == "test-api-key-123"

    def test_get_nonexistent_api_key(self):
        """Test retrieving non-existent API key returns None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            creds_path = Path(temp_dir) / "test_creds.enc"
            manager = CredentialsManager(credentials_path=creds_path)

            key = manager.get_api_key("nonexistent")
            assert key is None

    def test_set_empty_api_key(self):
        """Test that empty API keys are rejected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            creds_path = Path(temp_dir) / "test_creds.enc"
            manager = CredentialsManager(credentials_path=creds_path)

            # Empty string
            success = manager.set_api_key("openai", "")
            assert not success

            # Whitespace only
            success = manager.set_api_key("openai", "   ")
            assert not success

    def test_remove_api_key(self):
        """Test removing API keys."""
        with tempfile.TemporaryDirectory() as temp_dir:
            creds_path = Path(temp_dir) / "test_creds.enc"
            manager = CredentialsManager(credentials_path=creds_path)

            # Set then remove
            manager.set_api_key("openai", "test-key")
            assert manager.get_api_key("openai") == "test-key"

            success = manager.remove_api_key("openai")
            assert success
            assert manager.get_api_key("openai") is None

    def test_remove_nonexistent_api_key(self):
        """Test removing non-existent API key succeeds."""
        with tempfile.TemporaryDirectory() as temp_dir:
            creds_path = Path(temp_dir) / "test_creds.enc"
            manager = CredentialsManager(credentials_path=creds_path)

            success = manager.remove_api_key("nonexistent")
            assert success

    def test_list_configured_providers(self):
        """Test listing configured providers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            creds_path = Path(temp_dir) / "test_creds.enc"
            manager = CredentialsManager(credentials_path=creds_path)

            # Initially no providers configured
            providers = manager.list_configured_providers()
            # Only OpenAI is supported now
            assert providers["openai"] is False

            # Set one provider
            manager.set_api_key("openai", "test-key")
            providers = manager.list_configured_providers()
            # Only OpenAI is supported now
            assert providers["openai"] is True

    def test_clear_all_credentials(self):
        """Test clearing all credentials."""
        with tempfile.TemporaryDirectory() as temp_dir:
            creds_path = Path(temp_dir) / "test_creds.enc"
            manager = CredentialsManager(credentials_path=creds_path)

            # Set some credentials
            manager.set_api_key("openai", "test-key-1")
            manager.set_api_key("openai", "test-key-2")

            # Clear all
            success = manager.clear_all_credentials()
            assert success

            # Verify they're gone
            # Only OpenAI keys should remain after clearing
            assert manager.get_api_key("openai") is None

    def test_get_credentials_info(self):
        """Test getting credentials system information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            creds_path = Path(temp_dir) / "test_creds.enc"
            manager = CredentialsManager(credentials_path=creds_path)

            info = manager.get_credentials_info()

            assert "credentials_file" in info
            assert "file_exists" in info
            assert "encryption_enabled" in info
            assert "encryption_available" in info
            assert "configured_providers" in info

            assert info["credentials_file"] == str(creds_path)
            assert isinstance(info["configured_providers"], dict)

    def test_api_key_trimming(self):
        """Test that API keys are properly trimmed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            creds_path = Path(temp_dir) / "test_creds.enc"
            manager = CredentialsManager(credentials_path=creds_path)

            # Set key with whitespace
            manager.set_api_key("openai", "  test-key-with-spaces  ")

            # Should be trimmed when retrieved
            key = manager.get_api_key("openai")
            assert key == "test-key-with-spaces"

    def test_placeholder_key_rejection(self):
        """Test that placeholder keys are not considered valid."""
        with tempfile.TemporaryDirectory() as temp_dir:
            creds_path = Path(temp_dir) / "test_creds.enc"
            manager = CredentialsManager(credentials_path=creds_path)

            # Manually save a placeholder key (simulating old config)
            credentials = {"openai_api_key": "your_openai_api_key_here"}
            content = json.dumps(credentials)
            manager.credentials_path.write_text(content)

            # Should return None for placeholder
            key = manager.get_api_key("openai")
            assert key is None

    @patch("src.core.credentials.ENCRYPTION_AVAILABLE", True)
    def test_encryption_key_generation(self):
        """Test encryption key generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            creds_path = Path(temp_dir) / "test_creds.enc"

            # Mock settings to not have an encryption key
            with patch("src.core.credentials.get_settings") as mock_settings:
                mock_settings.return_value.enable_encryption = True
                mock_settings.return_value.encryption_key = None
                mock_settings.return_value.data_dir = Path(temp_dir)

                manager = CredentialsManager(credentials_path=creds_path)

                # Should have generated encryption
                assert manager._fernet is not None


class TestCredentialsManagerSingleton:
    """Test the global credentials manager singleton."""

    def test_get_credentials_manager_singleton(self):
        """Test that get_credentials_manager returns the same instance."""
        manager1 = get_credentials_manager()
        manager2 = get_credentials_manager()

        assert manager1 is manager2
        assert isinstance(manager1, CredentialsManager)

    def test_credentials_manager_properties(self):
        """Test that the global manager has expected properties."""
        manager = get_credentials_manager()

        assert hasattr(manager, "get_api_key")
        assert hasattr(manager, "set_api_key")
        assert hasattr(manager, "remove_api_key")
        assert hasattr(manager, "list_configured_providers")
        assert hasattr(manager, "clear_all_credentials")
        assert hasattr(manager, "get_credentials_info")


if __name__ == "__main__":
    pytest.main([__file__])
