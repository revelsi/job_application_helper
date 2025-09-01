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

import os
from pathlib import Path
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from src.api.models import DocumentUploadResponse
from src.core.document_manager import DocumentManager
from src.core.simple_document_handler import (
    SimpleDocumentHandler,
    SimpleDocumentUploadResult,
    create_simple_document_handler,
)
from src.core.simple_document_service import get_simple_document_service

router = APIRouter(prefix="/documents", tags=["documents"])


def get_document_handler() -> SimpleDocumentHandler:
    """Dependency to get simplified document handler instance."""
    return create_simple_document_handler()


def get_document_manager() -> DocumentManager:
    """Dependency to get document manager instance."""
    # Use simplified document service instead of RAG
    return DocumentManager()


@router.delete("/clear/job")
async def clear_job_documents(
    document_manager: DocumentManager = Depends(get_document_manager),
):
    """Clear job-specific documents."""
    try:
        result = document_manager.clear_job_application_documents()
        return {
            "success": result.success,
            "message": result.message,
            "details": result.details,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to clear job documents: {e}"
        ) from e


@router.delete("/clear/company")
async def clear_company_documents(
    document_manager: DocumentManager = Depends(get_document_manager),
):
    """Clear company-specific documents."""
    try:
        result = document_manager.clear_company_documents()
        return {
            "success": result.success,
            "message": result.message,
            "details": result.details,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to clear company documents: {e}"
        ) from e


@router.get("/list")
async def list_documents(
    document_manager: DocumentManager = Depends(get_document_manager),
):
    """List all documents with their metadata."""
    try:
        from src.core.storage import DocumentType

        documents = []

        # Get all document types
        for doc_type in DocumentType:
            docs = document_manager.storage.list_documents(
                document_type=doc_type, limit=100
            )
            for doc in docs:
                # Normalize fields to match API tests expectations
                upload_ts = None
                if hasattr(doc, "upload_timestamp") and doc.upload_timestamp is not None:
                    try:
                        upload_ts = doc.upload_timestamp.isoformat()
                    except Exception:
                        upload_ts = str(doc.upload_timestamp)
                elif hasattr(doc, "upload_date") and doc.upload_date is not None:
                    try:
                        upload_ts = doc.upload_date.isoformat()
                    except Exception:
                        upload_ts = str(doc.upload_date)

                documents.append(
                    {
                        "id": doc.id,
                        "filename": getattr(doc, "filename", doc.original_filename),
                        "original_filename": doc.original_filename,
                        "type": doc.document_type.value,
                        "category": (
                            "personal"
                            if doc.document_type
                            in [
                                DocumentType.CV,
                                DocumentType.CERTIFICATE,
                                DocumentType.PORTFOLIO,
                                DocumentType.COVER_LETTER,
                            ]
                            else "job-specific"
                        ),
                        "upload_timestamp": upload_ts,
                        "file_size": doc.file_size,
                        "tags": doc.tags or [],
                    }
                )

        return {"success": True, "documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}") from e


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form(...),
    document_handler: SimpleDocumentHandler = Depends(get_document_handler),
):
    """Upload document with category routing."""
    try:
        # Route to appropriate specific endpoint based on category
        if category == "personal":
            return await upload_candidate_document(file, document_handler)
        if category == "job-specific":
            return await upload_job_document(file, document_handler)
        raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/upload/candidate", response_model=DocumentUploadResponse)
async def upload_candidate_document(
    file: UploadFile = File(...),
    document_handler: SimpleDocumentHandler = Depends(get_document_handler),
):
    """Upload candidate document using simplified document handler."""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename).suffix
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = Path(tmp_file.name)

        try:
            # Use simplified document handler - consume the generator properly
            final_result = None
            progress_generator = document_handler.upload_candidate_document(
                tmp_file_path, file.filename
            )

            # Consume all progress updates until we get the final result
            try:
                while True:
                    next(progress_generator)  # Consume progress updates
            except StopIteration as e:
                # The return value is in e.value
                final_result = e.value

            # Ensure we have a valid result
            if not isinstance(final_result, SimpleDocumentUploadResult):
                raise ValueError("Invalid result from document handler")

            # Convert to API response
            return DocumentUploadResponse(
                document_id=(
                    str(final_result.document_id)
                    if final_result.document_id
                    else "unknown"
                ),
                success=final_result.success,
                message=final_result.message,
                file_name=file.filename,
                error=final_result.error,
                metadata=final_result.metadata,
            )
        finally:
            # Clean up temp file
            if tmp_file_path.exists():
                os.unlink(tmp_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/upload/job", response_model=DocumentUploadResponse)
async def upload_job_document(
    file: UploadFile = File(...),
    document_handler: SimpleDocumentHandler = Depends(get_document_handler),
):
    """Upload job document using simplified document handler."""
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename).suffix
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = Path(tmp_file.name)

        try:
            # Use simplified document handler - consume the generator properly
            final_result = None
            progress_generator = document_handler.upload_job_document(
                tmp_file_path, file.filename
            )

            # Consume all progress updates until we get the final result
            try:
                while True:
                    next(progress_generator)  # Consume progress updates
            except StopIteration as e:
                # The return value is in e.value
                final_result = e.value

            # Ensure we have a valid result
            if not isinstance(final_result, SimpleDocumentUploadResult):
                raise ValueError("Invalid result from document handler")

            return DocumentUploadResponse(
                document_id=(
                    str(final_result.document_id)
                    if final_result.document_id
                    else "unknown"
                ),
                success=final_result.success,
                message=final_result.message,
                file_name=file.filename,
                error=final_result.error,
                metadata=final_result.metadata,
            )
        finally:
            if tmp_file_path.exists():
                os.unlink(tmp_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/delete/{document_id}")
async def delete_document(
    document_id: int,
    document_manager: DocumentManager = Depends(get_document_manager),
):
    """Delete a document by ID."""
    try:
        success = document_manager.storage.delete_document(document_id)
        if success:
            return {"success": True, "message": "Document deleted successfully"}
        return {"success": False, "message": "Document not found or deletion failed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}") from e


@router.get("/stats")
async def get_document_stats():
    """Get document statistics."""
    try:
        service = get_simple_document_service()
        stats = service.get_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}") from e
