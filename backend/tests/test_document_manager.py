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
Tests for the simplified Document Manager.
"""

from datetime import datetime
import unittest
from unittest.mock import Mock, patch

from src.core.document_manager import (
    DocumentManager,
    get_document_manager,
)
from src.core.storage import DocumentType


class TestDocumentManager(unittest.TestCase):
    """Test the simplified DocumentManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock storage system
        self.mock_storage = Mock()
        
        # Mock document service
        self.mock_document_service = Mock()
        
        # Create document manager with mocked dependencies
        with patch("src.core.document_manager.get_storage_system", return_value=self.mock_storage), \
             patch("src.core.document_manager.get_simple_document_service", return_value=self.mock_document_service):
            self.document_manager = DocumentManager()

    def test_initialization(self):
        """Test DocumentManager initialization."""
        self.assertIsNotNone(self.document_manager.storage)
        self.assertIsNotNone(self.document_manager.document_service)
        self.assertIsNotNone(self.document_manager.logger)

    def test_clear_job_application_documents_success(self):
        """Test successful clearing of job application documents."""
        # Mock document data
        mock_doc1 = Mock()
        mock_doc1.id = "doc1"
        mock_doc1.original_filename = "job_desc.txt"
        
        mock_doc2 = Mock()
        mock_doc2.id = "doc2"
        mock_doc2.original_filename = "company_info.txt"
        
        # Mock storage system responses
        self.mock_storage.list_documents.side_effect = [
            [mock_doc1],  # JOB_DESCRIPTION
            [mock_doc2],  # ROLE_REQUIREMENTS
            [],  # COMPANY_INFO
            [],  # COMPANY_VALUES
            [],  # COMPANY_CAREERS
        ]
        
        # Mock successful operations
        self.mock_document_service.remove_document.return_value = True
        self.mock_storage.delete_document.return_value = True
        
        # Execute
        result = self.document_manager.clear_job_application_documents()
        
        # Verify
        self.assertTrue(result.success)
        self.assertEqual(result.documents_affected, 2)
        self.assertIn("Successfully cleared 2 job-specific documents", result.message)
        
        # Verify service calls
        self.mock_document_service.remove_document.assert_any_call("doc1")
        self.mock_document_service.remove_document.assert_any_call("doc2")
        self.mock_storage.delete_document.assert_any_call("doc1", delete_file=True)
        self.mock_storage.delete_document.assert_any_call("doc2", delete_file=True)

    def test_clear_candidate_documents_success(self):
        """Test successful clearing of candidate documents."""
        # Mock document data
        mock_doc = Mock()
        mock_doc.id = "cv1"
        mock_doc.original_filename = "resume.pdf"
        
        # Mock storage system responses
        self.mock_storage.list_documents.side_effect = [
            [mock_doc],  # CV
            [],  # COVER_LETTER
            [],  # CERTIFICATE
            [],  # PORTFOLIO
        ]
        
        # Mock successful operations
        self.mock_document_service.remove_document.return_value = True
        self.mock_storage.delete_document.return_value = True
        
        # Execute
        result = self.document_manager.clear_candidate_documents()
        
        # Verify
        self.assertTrue(result.success)
        self.assertEqual(result.documents_affected, 1)
        self.assertIn("Successfully cleared 1 candidate documents", result.message)

    def test_clear_all_documents_success(self):
        """Test successful clearing of all documents."""
        # Mock document data
        mock_doc1 = Mock()
        mock_doc1.id = "doc1"
        mock_doc1.original_filename = "file1.txt"
        
        mock_doc2 = Mock()
        mock_doc2.id = "doc2"
        mock_doc2.original_filename = "file2.txt"
        
        # Mock storage system response
        self.mock_storage.list_documents.return_value = [mock_doc1, mock_doc2]
        
        # Mock successful operations
        self.mock_document_service.remove_document.return_value = True
        self.mock_storage.delete_document.return_value = True
        
        # Execute
        result = self.document_manager.clear_all_documents()
        
        # Verify
        self.assertTrue(result.success)
        self.assertEqual(result.documents_affected, 2)
        self.assertIn("Successfully cleared 2 documents", result.message)

    def test_get_document_statistics_success(self):
        """Test successful retrieval of document statistics."""
        # Mock document data
        mock_doc1 = Mock()
        mock_doc1.document_type = DocumentType.CV
        
        mock_doc2 = Mock()
        mock_doc2.document_type = DocumentType.JOB_DESCRIPTION
        
        # Mock storage system response
        self.mock_storage.list_documents.return_value = [mock_doc1, mock_doc2]
        
        # Execute
        result = self.document_manager.get_document_statistics()
        
        # Verify
        self.assertEqual(result["total_documents"], 2)
        self.assertEqual(result["by_type"]["cv"], 1)
        self.assertEqual(result["by_type"]["job_description"], 1)

    def test_list_documents_by_type_success(self):
        """Test successful listing of documents by type."""
        # Mock document data
        mock_doc = Mock()
        mock_doc.id = "doc1"
        mock_doc.original_filename = "resume.pdf"
        mock_doc.document_type = DocumentType.CV
        mock_doc.upload_date = datetime.now()
        mock_doc.file_size = 1024
        mock_doc.tags = ["candidate"]
        
        # Mock storage system response
        self.mock_storage.list_documents.return_value = [mock_doc]
        
        # Execute
        result = self.document_manager.list_documents_by_type(DocumentType.CV)
        
        # Verify
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "doc1")
        self.assertEqual(result[0]["filename"], "resume.pdf")
        self.assertEqual(result[0]["type"], "cv")

    def test_delete_document_success(self):
        """Test successful deletion of a specific document."""
        # Mock successful operations
        self.mock_document_service.remove_document.return_value = True
        self.mock_storage.delete_document.return_value = True
        
        # Execute
        result = self.document_manager.delete_document("doc1")
        
        # Verify
        self.assertTrue(result.success)
        self.assertEqual(result.documents_affected, 1)
        self.assertIn("Successfully deleted document doc1", result.message)
        
        # Verify service calls
        self.mock_document_service.remove_document.assert_called_once_with("doc1")
        self.mock_storage.delete_document.assert_called_once_with("doc1", delete_file=True)

    def test_clear_documents_with_storage_error(self):
        """Test handling of storage errors during document clearing."""
        # Mock storage error
        self.mock_storage.list_documents.side_effect = Exception("Storage error")
        
        # Execute
        result = self.document_manager.clear_job_application_documents()
        
        # Verify
        self.assertFalse(result.success)
        self.assertEqual(result.documents_affected, 0)
        self.assertIn("Failed to clear job documents", result.message)

    def test_delete_document_with_service_error(self):
        """Test handling of service errors during document deletion."""
        # Mock service error
        self.mock_document_service.remove_document.side_effect = Exception("Service error")
        
        # Execute
        result = self.document_manager.delete_document("doc1")
        
        # Verify
        self.assertFalse(result.success)
        self.assertEqual(result.documents_affected, 0)
        self.assertIn("Failed to delete document doc1", result.message)

    def test_get_document_manager_factory(self):
        """Test the factory function for creating document manager."""
        with patch("src.core.document_manager.get_storage_system"), \
             patch("src.core.document_manager.get_simple_document_service"):
            manager = get_document_manager()
            self.assertIsInstance(manager, DocumentManager)


if __name__ == "__main__":
    unittest.main()
