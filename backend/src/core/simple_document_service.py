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
Simplified Document Service for Job Application Helper.

This service provides basic document storage and retrieval functionality
without the complexity of RAG, vector stores, or embeddings. It focuses on:
- Simple document storage with metadata
- Basic text-based document retrieval
- Document categorization
- Content extraction for LLM context
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.document_processor import extract_text_from_file
from src.core.storage import DocumentMetadata, DocumentType, get_storage_system
from src.utils.logging import get_logger


@dataclass
class DocumentContent:
    """Content of a document with metadata."""

    metadata: DocumentMetadata
    content: str
    word_count: int


@dataclass
class DocumentSearchResult:
    """Result from document search."""

    document: DocumentContent
    relevance_score: float
    match_reason: str


class SimpleDocumentService:
    """
    Simplified document service that provides basic document storage and retrieval
    without complex RAG functionality.
    """

    def __init__(self):
        """Initialize the simple document service."""
        self.storage = get_storage_system()
        self.logger = get_logger(f"{__name__}.SimpleDocumentService")

        # Cache for document content to avoid repeated file reads
        self._content_cache: Dict[int, str] = {}

        self.logger.info("Simple document service initialized")

    def store_document(
        self,
        source_path: Path,
        document_type: DocumentType,
        tags: Optional[List[str]] = None,
        notes: str = "",
        original_filename: Optional[str] = None,
    ) -> DocumentMetadata:
        """
        Store a document using the existing storage system.

        Args:
            source_path: Path to the source file
            document_type: Type of document
            tags: Optional tags for the document
            notes: Optional notes
            original_filename: Original filename from upload

        Returns:
            Document metadata with assigned ID
        """
        return self.storage.store_document(
            source_path=source_path,
            document_type=document_type,
            tags=tags,
            notes=notes,
            original_filename=original_filename,
        )

    def get_document_content(self, document_id: int) -> Optional[DocumentContent]:
        """
        Get document content by ID.

        Args:
            document_id: Document ID

        Returns:
            Document content or None if not found
        """
        metadata = self.storage.get_document(document_id)
        if not metadata or not metadata.file_path:
            return None

        # Check cache first
        if document_id in self._content_cache:
            content = self._content_cache[document_id]
        else:
            # Extract content from file
            try:
                content = extract_text_from_file(metadata.file_path)
                # Cache the content (limit cache size)
                if len(self._content_cache) > 100:
                    # Remove oldest entry
                    oldest_id = next(iter(self._content_cache))
                    del self._content_cache[oldest_id]
                self._content_cache[document_id] = content
            except Exception as e:
                self.logger.error(
                    f"Failed to extract content from document {document_id}: {e}"
                )
                return None

        return DocumentContent(
            metadata=metadata,
            content=content,
            word_count=len(content.split()),
        )

    def list_documents(
        self,
        document_type: Optional[DocumentType] = None,
        limit: int = 100,
    ) -> List[DocumentMetadata]:
        """
        List documents with optional filtering.

        Args:
            document_type: Optional filter by document type
            limit: Maximum number of results

        Returns:
            List of document metadata
        """
        return self.storage.list_documents(document_type=document_type, limit=limit)

    def search_documents(
        self,
        query: str,
        document_type: Optional[DocumentType] = None,
        limit: int = 10,
    ) -> List[DocumentSearchResult]:
        """
        Search documents using simple text matching.

        Args:
            query: Search query
            document_type: Optional filter by document type
            limit: Maximum number of results

        Returns:
            List of search results ordered by relevance
        """
        # Get documents to search
        documents = self.list_documents(document_type=document_type, limit=100)

        # Search through document content
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for doc_metadata in documents:
            try:
                doc_content = self.get_document_content(doc_metadata.id)
                if not doc_content:
                    continue

                content_lower = doc_content.content.lower()
                content_words = set(content_lower.split())

                # Calculate simple relevance score
                relevance_score = 0.0
                match_reasons = []

                # Exact phrase matching (highest score)
                if query_lower in content_lower:
                    relevance_score += 1.0
                    match_reasons.append("exact phrase match")

                # Word matching
                matched_words = query_words.intersection(content_words)
                if matched_words:
                    word_match_score = len(matched_words) / len(query_words)
                    relevance_score += word_match_score * 0.7
                    match_reasons.append(f"word matches: {', '.join(matched_words)}")

                # Filename matching
                if query_lower in doc_metadata.original_filename.lower():
                    relevance_score += 0.3
                    match_reasons.append("filename match")

                # Tags matching
                if doc_metadata.tags:
                    for tag in doc_metadata.tags:
                        if query_lower in tag.lower():
                            relevance_score += 0.2
                            match_reasons.append(f"tag match: {tag}")

                # Only include if there's some relevance
                if relevance_score > 0:
                    results.append(
                        DocumentSearchResult(
                            document=doc_content,
                            relevance_score=relevance_score,
                            match_reason="; ".join(match_reasons),
                        )
                    )

            except Exception as e:
                self.logger.error(f"Error searching document {doc_metadata.id}: {e}")
                continue

        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        return results[:limit]

    def get_documents_by_type(
        self,
        document_type: DocumentType,
        limit: int = 20,
    ) -> List[DocumentContent]:
        """
        Get all documents of a specific type with their content.

        Args:
            document_type: Type of documents to retrieve
            limit: Maximum number of documents

        Returns:
            List of documents with content
        """
        metadata_list = self.list_documents(document_type=document_type, limit=limit)

        documents = []
        for metadata in metadata_list:
            content = self.get_document_content(metadata.id)
            if content:
                documents.append(content)

        return documents

    def get_candidate_documents(self, limit: int = 10) -> List[DocumentContent]:
        """Get candidate documents (CV, cover letters, etc.)."""
        candidate_types = [
            DocumentType.CV,
            DocumentType.COVER_LETTER,
            DocumentType.CERTIFICATE,
            DocumentType.PORTFOLIO,
        ]

        documents = []
        for doc_type in candidate_types:
            docs = self.get_documents_by_type(doc_type, limit=limit)
            documents.extend(docs)

        # Sort by upload date (newest first)
        documents.sort(key=lambda x: x.metadata.upload_date, reverse=True)

        return documents[:limit]

    def get_job_documents(self, limit: int = 10) -> List[DocumentContent]:
        """Get job-related documents."""
        job_types = [
            DocumentType.JOB_DESCRIPTION,
            DocumentType.ROLE_REQUIREMENTS,
        ]

        documents = []
        for doc_type in job_types:
            docs = self.get_documents_by_type(doc_type, limit=limit)
            documents.extend(docs)

        # Sort by upload date (newest first)
        documents.sort(key=lambda x: x.metadata.upload_date, reverse=True)

        return documents[:limit]

    def get_company_documents(self, limit: int = 10) -> List[DocumentContent]:
        """Get company-related documents."""
        company_types = [
            DocumentType.COMPANY_INFO,
            DocumentType.COMPANY_VALUES,
            DocumentType.COMPANY_CAREERS,
        ]

        documents = []
        for doc_type in company_types:
            docs = self.get_documents_by_type(doc_type, limit=limit)
            documents.extend(docs)

        # Sort by upload date (newest first)
        documents.sort(key=lambda x: x.metadata.upload_date, reverse=True)

        return documents[:limit]



    def get_document_context(
        self,
        max_context_length: int = 100000,  # Updated for GPT-5-mini 128K context
        max_candidate_doc_length: int = 50000,  # Allow full CV content
        max_job_doc_length: int = 30000,  # Allow detailed job descriptions
        max_company_doc_length: int = 20000,  # Allow comprehensive company info
    ) -> Dict[str, Any]:
        """
        Get document context for LLM prompting - simple and honest.
        
        Just gets the most recent documents from each category and includes them.
        The LLM is smart enough to figure out what's relevant.

        Args:
            max_context_length: Maximum total context length in characters
            max_candidate_doc_length: Maximum length per candidate document
            max_job_doc_length: Maximum length per job document
            max_company_doc_length: Maximum length per company document

        Returns:
            Dictionary with context information
        """
        # Get recent documents from each category - simple and predictable
        candidate_docs = self.get_candidate_documents(limit=3)
        job_docs = self.get_job_documents(limit=3)
        company_docs = self.get_company_documents(limit=3)

        # Build context sections
        context_sections = []
        current_length = 0

        # Add candidate information (most recent first)
        if candidate_docs:
            context_sections.append("### Your Background & Experience:")
            for doc in candidate_docs:
                if current_length + len(doc.content) > max_context_length:
                    break
                context_sections.append(f"**{doc.metadata.original_filename}:**")
                # Truncate to max length
                doc_content = doc.content[:max_candidate_doc_length]
                context_sections.append(doc_content)
                current_length += len(doc_content)
            context_sections.append("")

        # Add job information (most recent first)
        if job_docs:
            context_sections.append("### Job Information:")
            for doc in job_docs:
                if current_length + len(doc.content) > max_context_length:
                    break
                context_sections.append(f"**{doc.metadata.original_filename}:**")
                # Truncate to max length
                doc_content = doc.content[:max_job_doc_length]
                context_sections.append(doc_content)
                current_length += len(doc_content)
            context_sections.append("")

        # Add company information (most recent first)
        if company_docs:
            context_sections.append("### Company Information:")
            for doc in company_docs:
                if current_length + len(doc.content) > max_context_length:
                    break
                context_sections.append(f"**{doc.metadata.original_filename}:**")
                # Truncate to max length
                doc_content = doc.content[:max_company_doc_length]
                context_sections.append(doc_content)
                current_length += len(doc_content)
            context_sections.append("")

        context_text = "\n".join(context_sections)

        # Compile source information
        source_documents = []
        for doc in candidate_docs + job_docs + company_docs:
            source_documents.append(
                {
                    "filename": doc.metadata.original_filename,
                    "type": doc.metadata.document_type.value,
                    "upload_date": doc.metadata.upload_date.isoformat(),
                    "word_count": doc.word_count,
                }
            )

        return {
            "query": "",  # Query is no longer used for retrieval
            "context_text": context_text,
            "context_length": len(context_text),
            "source_documents": source_documents,
            "has_context": len(context_text.strip()) > 0,
            "document_counts": {
                "candidate": len(candidate_docs),
                "job": len(job_docs),
                "company": len(company_docs),
            },
        }

    def delete_document(self, document_id: int) -> bool:
        """
        Delete a document.

        Args:
            document_id: Document ID to delete

        Returns:
            True if deleted successfully
        """
        try:
            # Remove from cache if present
            if document_id in self._content_cache:
                del self._content_cache[document_id]

            # Delete from storage
            return self.storage.delete_document(document_id)

        except Exception as e:
            self.logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get document service statistics."""
        try:
            # Count documents by type
            document_counts = {}
            total_documents = 0

            for doc_type in DocumentType:
                docs = self.list_documents(document_type=doc_type, limit=1000)
                count = len(docs)
                document_counts[doc_type.value] = count
                total_documents += count

            return {
                "total_documents": total_documents,
                "documents_by_type": document_counts,
                "cache_size": len(self._content_cache),
                "storage_path": str(self.storage.file_manager.base_path),
            }

        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


# Global instance
_simple_document_service = None


def get_simple_document_service() -> SimpleDocumentService:
    """Get the global simple document service instance."""
    global _simple_document_service
    if _simple_document_service is None:
        _simple_document_service = SimpleDocumentService()
    return _simple_document_service
