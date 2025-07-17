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
Simplified Document Handler for Job Application Helper.

This handler provides document upload and processing functionality
without the complexity of RAG:
- Document upload with validation
- Text extraction from various file formats
- Simple document type classification
- Progress tracking for uploads
- Error handling and recovery
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, Optional

from src.core.document_processor import extract_text_from_file
from src.core.simple_document_service import get_simple_document_service
from src.core.storage import DocumentType
from src.utils.logging import get_logger
from src.utils.security import get_input_validator


@dataclass
class SimpleDocumentUploadResult:
    """Result from document upload operation."""
    
    success: bool
    message: str
    document_id: Optional[int] = None
    error: Optional[str] = None
    metadata: Optional[Dict] = None


class SimpleDocumentHandler:
    """
    Simplified document handler that provides upload and processing
    without complex RAG functionality.
    """
    
    def __init__(self):
        """Initialize the simple document handler."""
        self.document_service = get_simple_document_service()
        self.validator = get_input_validator()
        self.logger = get_logger(f"{__name__}.SimpleDocumentHandler")
        
        # Document validation settings
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.allowed_extensions = {".txt", ".pdf", ".docx", ".md"}
        
        self.logger.info("Simple document handler initialized")
    
    def upload_candidate_document(
        self,
        file_path: Path,
        original_filename: str,
        progress_callback: Optional[callable] = None,
    ) -> Generator[str, None, SimpleDocumentUploadResult]:
        """
        Upload and process a candidate document.
        Always treat as candidate information (default: CV).
        """
        try:
            # Validate file
            validation_result = self._validate_file(file_path)
            if not validation_result["success"]:
                yield validation_result["message"]
                return SimpleDocumentUploadResult(
                    success=False,
                    message=validation_result["message"],
                    error=validation_result["error"]
                )
            
            yield "🔄 **Processing Document...**\n\n📤 Uploading file...\n⏳ Please wait while we process your document"
            
            # Extract text content
            yield "🔄 **Processing Document...**\n\n✅ File uploaded\n📝 Extracting text content...\n⏳ Please wait"
            
            document_text = extract_text_from_file(file_path)
            text_length = len(document_text)
            
            # Validate content length
            if text_length < 20:
                error_msg = f"⚠️ Document appears to be empty or contains minimal text ({text_length} characters). Please check the file format and content."
                yield error_msg
                return SimpleDocumentUploadResult(
                    success=False,
                    message=error_msg,
                    error="Document content too short"
                )
            
            # Set document type based on user intent (always candidate)
            detected_type = DocumentType.CV  # Default; can extend to allow user selection
            
            # Store document
            yield "🔄 **Processing Document...**\n\n✅ File uploaded\n✅ Text extracted\n✅ Document type set (candidate)\n📁 Storing document...\n⏳ Please wait"
            
            doc_metadata = self.document_service.store_document(
                source_path=file_path,
                document_type=detected_type,
                tags=["candidate_profile", "uploaded_via_handler"],
                notes=f"Document ({detected_type.value}) uploaded at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                original_filename=original_filename,
            )
            
            # Create user-friendly document type display
            type_display_map = {
                DocumentType.CV: "CV/Resume",
                DocumentType.COVER_LETTER: "Cover Letter",
                DocumentType.CERTIFICATE: "Certificate",
                DocumentType.PORTFOLIO: "Portfolio",
                DocumentType.OTHER: "Document",
            }
            doc_type_display = type_display_map.get(detected_type, "Document")
            
            final_message = f"✅ **{doc_type_display} Upload Complete!**\n\n📄 **File:** {original_filename}\n🤖 **Type:** {doc_type_display}\n📊 **Content:** {text_length:,} characters\n\n💡 **This document is now part of your candidate profile!**"
            yield final_message
            
            return SimpleDocumentUploadResult(
                success=True,
                message=final_message,
                document_id=doc_metadata.id,
                metadata={
                    "file_name": original_filename,
                    "text_length": text_length,
                    "document_type": detected_type.value,
                },
            )
        
        except Exception as e:
            error_msg = f"❌ Upload failed: {str(e)}. Please try again or check file format."
            self.logger.error(f"Candidate document upload failed: {e}")
            yield error_msg
            return SimpleDocumentUploadResult(
                success=False,
                message=error_msg,
                error=str(e)
            )

    def upload_job_document(
        self,
        file_path: Path,
        original_filename: str,
        progress_callback: Optional[callable] = None,
    ) -> Generator[str, None, SimpleDocumentUploadResult]:
        """
        Upload and process a job description document.
        Always treat as job information (default: JOB_DESCRIPTION).
        """
        try:
            # Validate file
            validation_result = self._validate_file(file_path)
            if not validation_result["success"]:
                yield validation_result["message"]
                return SimpleDocumentUploadResult(
                    success=False,
                    message=validation_result["message"],
                    error=validation_result["error"]
                )
            
            yield "📁 **Processing Job Description...**\n\n✅ File received\n📝 Extracting text content..."
            
            # Extract document text
            document_text = extract_text_from_file(file_path)
            text_length = len(document_text)
            
            if text_length < 100:
                error_msg = f"⚠️ Job description appears too short ({text_length} characters). Please provide a more detailed job description."
                yield error_msg
                return SimpleDocumentUploadResult(
                    success=False,
                    message=error_msg,
                    error="Job description too short"
                )
            
            # Set document type based on user intent (always job)
            detected_type = DocumentType.JOB_DESCRIPTION
            
            yield "📁 **Processing Job Description...**\n\n✅ Text extracted\n📁 Storing document..."
            
            # Store document
            doc_metadata = self.document_service.store_document(
                source_path=file_path,
                document_type=detected_type,
                tags=["job_description", "current_application", "uploaded_via_handler"],
                notes=f"Job document uploaded at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                original_filename=original_filename,
            )
            
            final_message = f"✅ **Job Description Upload Complete!**\n\n📄 **File:** {original_filename}\n📊 **Content:** {text_length:,} characters\n\n🎯 **Job is now available for analysis and matching!**"
            yield final_message
            
            return SimpleDocumentUploadResult(
                success=True,
                message=final_message,
                document_id=doc_metadata.id,
                metadata={
                    "file_name": original_filename,
                    "text_length": text_length,
                    "document_type": detected_type.value,
                },
            )
        
        except Exception as e:
            error_msg = f"❌ Upload failed: {str(e)}. Please try again or check file format."
            self.logger.error(f"Job document upload failed: {e}")
            yield error_msg
            return SimpleDocumentUploadResult(
                success=False,
                message=error_msg,
                error=str(e)
            )
    
    def _validate_file(self, file_path: Path) -> Dict[str, any]:
        """
        Validate uploaded file.
        
        Args:
            file_path: Path to file to validate
            
        Returns:
            Dictionary with validation result
        """
        try:
            # Check file exists
            if not file_path.exists():
                return {
                    "success": False,
                    "message": "❌ File not found",
                    "error": "File does not exist"
                }
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                return {
                    "success": False,
                    "message": f"❌ File too large ({file_size / (1024*1024):.1f}MB). Maximum size is {self.max_file_size / (1024*1024):.0f}MB",
                    "error": "File too large"
                }
            
            # Check file extension
            file_extension = file_path.suffix.lower()
            if file_extension not in self.allowed_extensions:
                return {
                    "success": False,
                    "message": f"❌ File type not supported ({file_extension}). Allowed types: {', '.join(self.allowed_extensions)}",
                    "error": "File type not supported"
                }
            
            return {"success": True, "message": "File validation passed"}
        
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ File validation failed: {str(e)}",
                "error": str(e)
            }


def create_simple_document_handler() -> SimpleDocumentHandler:
    """Create a simple document handler instance."""
    return SimpleDocumentHandler() 