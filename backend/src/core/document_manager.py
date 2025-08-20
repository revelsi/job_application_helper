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
Document Management Service for Job Application Helper.

This module provides high-level document management operations including:
- Bulk document operations
- Job-specific document clearing
- Document organization by workflow
- Integration with storage systems
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.core.simple_document_service import get_simple_document_service
from src.core.storage import (
    DocumentType,
    StorageSystem,
    get_storage_system,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentOperationResult:
    """Result of a document management operation."""

    success: bool
    message: str
    documents_affected: int
    details: List[str]


class DocumentManagerError(Exception):
    """Base exception for document manager operations."""

    pass


class DocumentManager:
    """High-level document management service."""

    def __init__(self, storage_system: Optional[StorageSystem] = None):
        """
        Initialize document manager.

        Args:
            storage_system: Optional storage system instance
        """
        self.storage = storage_system or get_storage_system()
        self.document_service = get_simple_document_service()
        self.logger = logger

    def clear_job_application_documents(self) -> DocumentOperationResult:
        """
        Clear all job-specific documents to start a fresh application.

        This removes job descriptions, company information, and other
        session-specific documents while preserving candidate profile documents.

        Returns:
            DocumentOperationResult with operation status and details
        """
        try:
            # Define job-specific document types to clear
            job_doc_types = [
                DocumentType.JOB_DESCRIPTION,
                DocumentType.ROLE_REQUIREMENTS,
                DocumentType.COMPANY_INFO,
                DocumentType.COMPANY_VALUES,
                DocumentType.COMPANY_CAREERS,
            ]

            cleared_count = 0
            details = []

            for doc_type in job_doc_types:
                try:
                    # Get documents of this type
                    docs = self.storage.list_documents(
                        document_type=doc_type, limit=100
                    )

                    type_count = 0
                    for doc in docs:
                        try:
                            # Remove from document service
                            self.document_service.remove_document(doc.id)

                            # Remove from storage
                            self.storage.delete_document(doc.id, delete_file=True)

                            type_count += 1
                            cleared_count += 1

                            self.logger.debug(
                                f"Cleared {doc_type.value}: {doc.original_filename}"
                            )

                        except Exception as e:
                            self.logger.warning(
                                f"Failed to clear document {doc.id}: {e}"
                            )

                    if type_count > 0:
                        details.append(
                            f"Cleared {type_count} {doc_type.value} documents"
                        )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to clear {doc_type.value} documents: {e}"
                    )
                    details.append(
                        f"Warning: Could not clear {doc_type.value} documents"
                    )
                    raise

            success_message = (
                f"Successfully cleared {cleared_count} job-specific documents"
            )
            self.logger.info(success_message)

            return DocumentOperationResult(
                success=True,
                message=success_message,
                documents_affected=cleared_count,
                details=details,
            )

        except Exception as e:
            error_message = f"Failed to clear job documents: {e!s}"
            self.logger.error(error_message)
            return DocumentOperationResult(
                success=False,
                message=error_message,
                documents_affected=0,
                details=[str(e)],
            )

    def clear_candidate_documents(self) -> DocumentOperationResult:
        """
        Clear all candidate-specific documents (CVs, cover letters, etc.).

        This removes personal documents while preserving job-related documents.

        Returns:
            DocumentOperationResult with operation status and details
        """
        try:
            # Define candidate-specific document types to clear
            candidate_doc_types = [
                DocumentType.CV,
                DocumentType.COVER_LETTER,
                DocumentType.CERTIFICATE,
                DocumentType.PORTFOLIO,
            ]

            cleared_count = 0
            details = []

            for doc_type in candidate_doc_types:
                try:
                    # Get documents of this type
                    docs = self.storage.list_documents(
                        document_type=doc_type, limit=100
                    )

                    type_count = 0
                    for doc in docs:
                        try:
                            # Remove from document service
                            self.document_service.remove_document(doc.id)

                            # Remove from storage
                            self.storage.delete_document(doc.id, delete_file=True)

                            type_count += 1
                            cleared_count += 1

                            self.logger.debug(
                                f"Cleared {doc_type.value}: {doc.original_filename}"
                            )

                        except Exception as e:
                            self.logger.warning(
                                f"Failed to clear document {doc.id}: {e}"
                            )

                    if type_count > 0:
                        details.append(
                            f"Cleared {type_count} {doc_type.value} documents"
                        )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to clear {doc_type.value} documents: {e}"
                    )
                    details.append(
                        f"Warning: Could not clear {doc_type.value} documents"
                    )

            success_message = (
                f"Successfully cleared {cleared_count} candidate documents"
            )
            self.logger.info(success_message)

            return DocumentOperationResult(
                success=True,
                message=success_message,
                documents_affected=cleared_count,
                details=details,
            )

        except Exception as e:
            error_message = f"Failed to clear candidate documents: {e!s}"
            self.logger.error(error_message)
            return DocumentOperationResult(
                success=False,
                message=error_message,
                documents_affected=0,
                details=[str(e)],
            )

    def clear_all_documents(self) -> DocumentOperationResult:
        """
        Clear all documents from the system.

        Returns:
            DocumentOperationResult with operation status and details
        """
        try:
            # Get all documents
            all_docs = self.storage.list_documents(limit=1000)
            cleared_count = 0
            details = []

            for doc in all_docs:
                try:
                    # Remove from document service
                    self.document_service.remove_document(doc.id)

                    # Remove from storage
                    self.storage.delete_document(doc.id, delete_file=True)

                    cleared_count += 1
                    self.logger.debug(f"Cleared document: {doc.original_filename}")

                except Exception as e:
                    self.logger.warning(f"Failed to clear document {doc.id}: {e}")

            success_message = f"Successfully cleared {cleared_count} documents"
            self.logger.info(success_message)

            return DocumentOperationResult(
                success=True,
                message=success_message,
                documents_affected=cleared_count,
                details=[f"Cleared {cleared_count} total documents"],
            )

        except Exception as e:
            error_message = f"Failed to clear all documents: {e!s}"
            self.logger.error(error_message)
            return DocumentOperationResult(
                success=False,
                message=error_message,
                documents_affected=0,
                details=[str(e)],
            )

    def get_document_statistics(self) -> Dict:
        """
        Get comprehensive document statistics.

        Returns:
            Dictionary with document counts and statistics
        """
        try:
            stats = {"total_documents": 0, "by_type": {}}

            # Get all documents
            all_docs = self.storage.list_documents(limit=1000)
            stats["total_documents"] = len(all_docs)

            # Count by type
            for doc in all_docs:
                doc_type = doc.document_type.value
                stats["by_type"][doc_type] = stats["by_type"].get(doc_type, 0) + 1

            return stats

        except Exception as e:
            self.logger.error(f"Error getting document statistics: {e}")
            return {"total_documents": 0, "by_type": {}, "error": str(e)}

    def list_documents_by_type(self, document_type: DocumentType) -> List[Dict]:
        """
        List documents of a specific type.

        Args:
            document_type: Type of documents to list

        Returns:
            List of document information dictionaries
        """
        try:
            docs = self.storage.list_documents(document_type=document_type, limit=100)

            return [
                {
                    "id": doc.id,
                    "filename": doc.original_filename,
                    "type": doc.document_type.value,
                    "upload_date": doc.upload_date.isoformat(),
                    "size": doc.file_size,
                    "tags": doc.tags,
                }
                for doc in docs
            ]

        except Exception as e:
            self.logger.error(f"Error listing documents by type {document_type}: {e}")
            return []

    def delete_document(self, document_id: str) -> DocumentOperationResult:
        """
        Delete a specific document.

        Args:
            document_id: ID of the document to delete

        Returns:
            DocumentOperationResult with operation status
        """
        try:
            # Remove from document service
            self.document_service.remove_document(document_id)

            # Remove from storage
            self.storage.delete_document(document_id, delete_file=True)

            success_message = f"Successfully deleted document {document_id}"
            self.logger.info(success_message)

            return DocumentOperationResult(
                success=True,
                message=success_message,
                documents_affected=1,
                details=[f"Deleted document {document_id}"],
            )

        except Exception as e:
            error_message = f"Failed to delete document {document_id}: {e!s}"
            self.logger.error(error_message)
            return DocumentOperationResult(
                success=False,
                message=error_message,
                documents_affected=0,
                details=[str(e)],
            )


def get_document_manager() -> DocumentManager:
    """Factory function to create document manager."""
    return DocumentManager()
