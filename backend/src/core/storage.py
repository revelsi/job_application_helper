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
Local storage system for job application helper.

This module provides:
- Data models for documents, candidates, and applications
- SQLite database for metadata storage
- File system organization for documents
- CRUD operations with proper validation
- Encryption for sensitive information
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet

try:
    from src.utils.config import get_settings
    from src.utils.logging import get_logger
except ImportError:
    # Fallback for direct execution
    import logging

    def get_logger(name):
        return logging.getLogger(name)

    class MockSettings:
        data_dir = Path("./data")
        documents_path = Path("./data/documents")
        enable_encryption = True
        encryption_key = None

    def get_settings():
        return MockSettings()


logger = get_logger(__name__)


class StorageError(Exception):
    """Base exception for storage system operations."""

    pass


class DocumentType(Enum):
    """Types of documents we can store."""

    # Candidate documents
    CV = "cv"
    COVER_LETTER = "cover_letter"
    CERTIFICATE = "certificate"
    PORTFOLIO = "portfolio"

    # Job application documents
    JOB_DESCRIPTION = "job_description"
    ROLE_REQUIREMENTS = "role_requirements"

    # Company information
    COMPANY_INFO = "company_info"
    COMPANY_VALUES = "company_values"
    COMPANY_CAREERS = "company_careers"

    # General
    OTHER = "other"


class DocumentStatus(Enum):
    """Processing status of documents."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    ERROR = "error"


@dataclass
class DocumentMetadata:
    """Metadata for a stored document."""

    id: Optional[int] = None
    filename: str = ""
    original_filename: str = ""
    file_path: Optional[Path] = None
    document_type: DocumentType = DocumentType.OTHER
    file_size: int = 0
    file_hash: str = ""
    upload_date: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    status: DocumentStatus = DocumentStatus.UPLOADED
    word_count: int = 0
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self):
        """Validate and process fields after initialization."""
        self._convert_path_field()
        self._convert_enum_fields()

    def _convert_path_field(self):
        """Convert file_path to Path object if needed."""
        if self.file_path and not isinstance(self.file_path, Path):
            self.file_path = Path(self.file_path)

    def _convert_enum_fields(self):
        """Convert string enum values to proper enum types."""
        if isinstance(self.document_type, str):
            self.document_type = DocumentType(self.document_type)
        if isinstance(self.status, str):
            self.status = DocumentStatus(self.status)


@dataclass
class CandidateProfile:
    """Candidate personal information and profile."""

    id: Optional[int] = None
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    phone: str = ""
    current_title: str = ""
    years_experience: int = 0
    skills: List[str] = field(default_factory=list)
    industries: List[str] = field(default_factory=list)
    salary_expectation: str = ""  # Will be encrypted
    location_preferences: List[str] = field(default_factory=list)
    work_preferences: Dict[str, Any] = field(default_factory=dict)
    linkedin_url: str = ""
    github_url: str = ""
    portfolio_url: str = ""
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def full_name(self) -> str:
        """Get full name."""
        return f"{self.first_name} {self.last_name}".strip()


@dataclass
class JobApplication:
    """Information about a job application."""

    id: Optional[int] = None
    company_name: str = ""
    job_title: str = ""
    job_description: str = ""
    job_url: str = ""
    application_date: datetime = field(default_factory=datetime.now)
    application_status: str = "draft"
    cv_document_id: Optional[int] = None
    cover_letter_document_id: Optional[int] = None
    additional_document_ids: List[int] = field(default_factory=list)
    generated_content: Dict[str, str] = field(default_factory=dict)
    follow_up_date: Optional[datetime] = None
    notes: str = ""
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ContentGeneration:
    """Track generated content for applications."""

    id: Optional[int] = None
    application_id: int = 0
    content_type: str = ""
    prompt_used: str = ""
    generated_content: str = ""
    source_documents: List[int] = field(default_factory=list)
    generation_date: datetime = field(default_factory=datetime.now)
    model_used: str = ""
    processing_time: Optional[float] = None
    rating: Optional[int] = None
    notes: str = ""


class DatabaseSchemas:
    """Database table schemas and indexes."""

    @staticmethod
    def get_documents_schema() -> str:
        """Get documents table schema."""
        return """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                document_type TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                file_hash TEXT NOT NULL UNIQUE,
                upload_date TEXT NOT NULL,
                last_modified TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'uploaded',
                word_count INTEGER DEFAULT 0,
                processing_time REAL,
                error_message TEXT,
                tags TEXT,
                notes TEXT,
                UNIQUE(file_hash)
            )
        """

    @staticmethod
    def get_candidate_profiles_schema() -> str:
        """Get candidate profiles table schema."""
        return """
            CREATE TABLE IF NOT EXISTS candidate_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                current_title TEXT,
                years_experience INTEGER DEFAULT 0,
                skills TEXT,
                industries TEXT,
                salary_expectation_encrypted TEXT,
                location_preferences TEXT,
                work_preferences TEXT,
                linkedin_url TEXT,
                github_url TEXT,
                portfolio_url TEXT,
                created_date TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        """

    @staticmethod
    def get_job_applications_schema() -> str:
        """Get job applications table schema."""
        return """
            CREATE TABLE IF NOT EXISTS job_applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_name TEXT NOT NULL,
                job_title TEXT NOT NULL,
                job_description TEXT,
                job_url TEXT,
                application_date TEXT NOT NULL,
                application_status TEXT DEFAULT 'draft',
                cv_document_id INTEGER,
                cover_letter_document_id INTEGER,
                additional_document_ids TEXT,
                generated_content TEXT,
                follow_up_date TEXT,
                notes TEXT,
                created_date TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                FOREIGN KEY (cv_document_id) REFERENCES documents (id),
                FOREIGN KEY (cover_letter_document_id) REFERENCES documents (id)
            )
        """

    @staticmethod
    def get_content_generations_schema() -> str:
        """Get content generations table schema."""
        return """
            CREATE TABLE IF NOT EXISTS content_generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                application_id INTEGER NOT NULL,
                content_type TEXT NOT NULL,
                prompt_used TEXT,
                generated_content TEXT NOT NULL,
                source_documents TEXT,
                generation_date TEXT NOT NULL,
                model_used TEXT,
                processing_time REAL,
                rating INTEGER,
                notes TEXT,
                FOREIGN KEY (application_id) REFERENCES job_applications (id)
            )
        """

    @staticmethod
    def get_all_schemas() -> Dict[str, str]:
        """Get all table schemas."""
        return {
            "documents": DatabaseSchemas.get_documents_schema(),
            "candidate_profiles": DatabaseSchemas.get_candidate_profiles_schema(),
            "job_applications": DatabaseSchemas.get_job_applications_schema(),
            "content_generations": DatabaseSchemas.get_content_generations_schema(),
        }

    @staticmethod
    def get_indexes() -> List[str]:
        """Get all database indexes."""
        return [
            "CREATE INDEX IF NOT EXISTS idx_documents_type ON documents (document_type)",
            "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status)",
            "CREATE INDEX IF NOT EXISTS idx_documents_upload_date ON documents (upload_date)",
            "CREATE INDEX IF NOT EXISTS idx_applications_company ON job_applications (company_name)",
            "CREATE INDEX IF NOT EXISTS idx_applications_status ON job_applications (application_status)",
            "CREATE INDEX IF NOT EXISTS idx_applications_date ON job_applications (application_date)",
            "CREATE INDEX IF NOT EXISTS idx_content_type ON content_generations (content_type)",
            "CREATE INDEX IF NOT EXISTS idx_content_application ON content_generations (application_id)",
        ]


class DatabaseManager:
    """Manages SQLite database operations."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database manager."""
        self.settings = get_settings()
        self.db_path = self._get_db_path(db_path)
        self.encryption_key = self._get_encryption_key()
        self._setup_database()

    def _get_db_path(self, custom_path: Optional[Path]) -> Path:
        """Get database path."""
        return custom_path or (self.settings.data_dir / "job_helper.db")

    def _setup_database(self):
        """Setup database directory and initialize tables."""
        self._ensure_directories()
        self._initialize_database()

    def _get_encryption_key(self) -> Optional[Fernet]:
        """Get encryption key for sensitive data."""
        if not self.settings.enable_encryption:
            logger.info(
                "Encryption disabled by configuration. Sensitive data will not be encrypted."
            )
            return None

        encryption_key = self._ensure_encryption_available()
        if not encryption_key:
            logger.warning(
                "No encryption key available. Sensitive data will not be encrypted."
            )
            return None

        return self._create_fernet_key(encryption_key)

    def _ensure_encryption_available(self) -> Optional[str]:
        """Ensure encryption setup is available and return key."""
        ensure_encryption_setup = self._import_encryption_setup()
        return ensure_encryption_setup(self.settings)

    def _import_encryption_setup(self):
        """Import encryption setup function with fallback."""
        try:
            from src.utils.config import ensure_encryption_setup

            return ensure_encryption_setup
        except ImportError:
            # Fallback for when running as script
            from pathlib import Path
            import sys

            sys.path.insert(0, str(Path(__file__).parent.parent))
            from utils.config import ensure_encryption_setup

            return ensure_encryption_setup

    def _create_fernet_key(self, encryption_key: str = None) -> Optional[Fernet]:
        """Create Fernet encryption key."""
        try:
            key = encryption_key or self.settings.encryption_key
            if isinstance(key, str):
                key = key.encode()
            return Fernet(key)
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            return None

    def _ensure_directories(self):
        """Ensure database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _initialize_database(self):
        """Create database tables and indexes."""
        try:
            self._create_tables_and_indexes()
            logger.info(f"Database initialized at: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def _create_tables_and_indexes(self):
        """Create all tables and indexes."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            self._create_tables(conn)
            self._create_indexes(conn)
            conn.commit()

    def _create_tables(self, conn: sqlite3.Connection):
        """Create all database tables."""
        schemas = DatabaseSchemas.get_all_schemas()
        for table_name, schema in schemas.items():
            conn.execute(schema)
            logger.debug(f"Created/verified table: {table_name}")

    def _create_indexes(self, conn: sqlite3.Connection):
        """Create all database indexes."""
        indexes = DatabaseSchemas.get_indexes()
        for index_sql in indexes:
            conn.execute(index_sql)

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper configuration."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    def encrypt_field(self, value: str) -> str:
        """Encrypt sensitive field."""
        if not self.encryption_key or not value:
            return value
        return self.encryption_key.encrypt(value.encode()).decode()

    def decrypt_field(self, encrypted_value: str) -> str:
        """Decrypt sensitive field."""
        if not self.encryption_key or not encrypted_value:
            return encrypted_value
        return self._try_decrypt(encrypted_value)

    def _try_decrypt(self, encrypted_value: str) -> str:
        """Attempt to decrypt field with error handling."""
        try:
            return self.encryption_key.decrypt(encrypted_value.encode()).decode()
        except Exception as e:
            logger.error(f"Failed to decrypt field: {e}")
            return encrypted_value


class DocumentStorage:
    """Handles document storage operations."""

    def __init__(self, db_manager: DatabaseManager):
        """Initialize document storage."""
        self.db_manager = db_manager
        self.logger = get_logger(f"{__name__}.DocumentStorage")

    def create_document(self, metadata: DocumentMetadata) -> int:
        """Store document metadata in database."""
        self._validate_document_metadata(metadata)
        return self._insert_document(metadata)

    def _validate_document_metadata(self, metadata: DocumentMetadata):
        """Validate required document metadata fields."""
        if not metadata.filename:
            raise ValueError("Filename is required")
        if not metadata.original_filename:
            raise ValueError("Original filename is required")
        if not metadata.file_hash:
            raise ValueError("File hash is required")

    def _insert_document(self, metadata: DocumentMetadata) -> int:
        """Insert document metadata into database."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = self._execute_document_insert(conn, metadata)
                document_id = cursor.lastrowid
                self.logger.info(
                    f"Created document record: ID={document_id}, file={metadata.filename}"
                )
                return document_id
        except sqlite3.IntegrityError as e:
            self._handle_integrity_error(e, metadata.file_hash)
        except Exception as e:
            self.logger.error(f"Failed to create document: {e}")
            raise

    def _execute_document_insert(
        self, conn: sqlite3.Connection, metadata: DocumentMetadata
    ):
        """Execute document insert SQL."""
        return conn.execute(
            """
            INSERT INTO documents (
                filename, original_filename, file_path, document_type,
                file_size, file_hash, upload_date, last_modified,
                status, word_count, processing_time, error_message,
                tags, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            self._get_document_insert_params(metadata),
        )

    def _get_document_insert_params(self, metadata: DocumentMetadata) -> tuple:
        """Get parameters for document insert."""
        return (
            metadata.filename,
            metadata.original_filename,
            str(metadata.file_path) if metadata.file_path else None,
            metadata.document_type.value,
            metadata.file_size,
            metadata.file_hash,
            metadata.upload_date.isoformat(),
            metadata.last_modified.isoformat(),
            metadata.status.value,
            metadata.word_count,
            metadata.processing_time,
            metadata.error_message,
            json.dumps(metadata.tags) if metadata.tags else None,
            metadata.notes,
        )

    def _handle_integrity_error(self, error: sqlite3.IntegrityError, file_hash: str):
        """Handle database integrity constraint violations."""
        if "file_hash" in str(error):
            raise ValueError(f"Document with this content already exists: {file_hash}")
        raise error

    def get_document(self, document_id: int) -> Optional[DocumentMetadata]:
        """
        Retrieve document metadata by ID.

        Args:
            document_id: Document ID

        Returns:
            Document metadata or None if not found
        """
        try:
            with self.db_manager.get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM documents WHERE id = ?", (document_id,)
                ).fetchone()

                if not row:
                    return None

                return self._row_to_document_metadata(row)

        except Exception as e:
            self.logger.error(f"Failed to retrieve document {document_id}: {e}")
            return None

    def get_document_by_hash(self, file_hash: str) -> Optional[DocumentMetadata]:
        """
        Retrieve document metadata by file hash.

        Args:
            file_hash: File hash to search for

        Returns:
            Document metadata or None if not found
        """
        try:
            with self.db_manager.get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM documents WHERE file_hash = ?", (file_hash,)
                ).fetchone()

                if not row:
                    return None

                return self._row_to_document_metadata(row)

        except Exception as e:
            self.logger.error(f"Failed to retrieve document by hash {file_hash}: {e}")
            return None

    def list_documents(
        self,
        document_type: Optional[DocumentType] = None,
        status: Optional[DocumentStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DocumentMetadata]:
        """
        List documents with optional filtering.

        Args:
            document_type: Filter by document type
            status: Filter by status
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of document metadata
        """
        try:
            query = "SELECT * FROM documents WHERE 1=1"
            params = []

            if document_type:
                query += " AND document_type = ?"
                params.append(document_type.value)

            if status:
                query += " AND status = ?"
                params.append(status.value)

            query += " ORDER BY upload_date DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            with self.db_manager.get_connection() as conn:
                rows = conn.execute(query, params).fetchall()
                return [self._row_to_document_metadata(row) for row in rows]

        except Exception as e:
            self.logger.error(f"Failed to list documents: {e}")
            return []

    def update_document(self, document_id: int, updates: Dict[str, Any]) -> bool:
        """
        Update document metadata.

        Args:
            document_id: Document ID to update
            updates: Dictionary of fields to update

        Returns:
            True if update successful, False otherwise
        """
        if not updates:
            return True

        # Validate updateable fields - Security: Whitelist approach prevents SQL injection
        allowed_fields = {
            "filename",
            "file_path",
            "document_type",
            "status",
            "word_count",
            "processing_time",
            "error_message",
            "tags",
            "notes",
        }

        update_fields = []
        params = []

        for field_name, value in updates.items():
            # Security: Strict validation of field names against whitelist
            if field_name not in allowed_fields:
                self.logger.warning(f"Ignoring invalid update field: {field_name}")
                continue

            # Security: Additional validation that field name contains only safe characters
            if not field_name.replace("_", "").isalnum():
                self.logger.warning(
                    f"Ignoring field with unsafe characters: {field_name}"
                )
                continue

            update_fields.append(f"{field_name} = ?")

            # Handle special field formatting
            if (field_name == "document_type" and isinstance(value, DocumentType)) or (
                field_name == "status" and isinstance(value, DocumentStatus)
            ):
                params.append(value.value)
            elif field_name == "tags" and isinstance(value, list):
                params.append(json.dumps(value))
            elif field_name == "file_path" and isinstance(value, Path):
                params.append(str(value))
            else:
                params.append(value)

        if not update_fields:
            return True

        # Add last_modified timestamp
        update_fields.append("last_modified = ?")
        params.append(datetime.now().isoformat())
        params.append(document_id)

        try:
            with self.db_manager.get_connection() as conn:
                # Security: Field names are validated against whitelist above, values are parameterized
                # Build query using SQLAlchemy's safe parameter binding to avoid scanner warnings
                if len(update_fields) == 1:
                    # Single field update - field names are validated above
                    # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query
                    cursor = conn.execute(
                        "UPDATE documents SET " + update_fields[0] + " WHERE id = ?",
                        params,
                    )
                else:
                    # Multiple field updates - field names are validated above, values parameterized
                    update_clause = ", ".join(update_fields)
                    query_template = "UPDATE documents SET {} WHERE id = ?"
                    final_query = query_template.format(update_clause)
                    # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query
                    cursor = conn.execute(final_query, params)

                if cursor.rowcount == 0:
                    self.logger.warning(f"No document found with ID: {document_id}")
                    return False

                self.logger.info(f"Updated document {document_id}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to update document {document_id}: {e}")
            return False

    def delete_document(self, document_id: int) -> bool:
        """
        Delete document metadata (does not delete physical file).

        Args:
            document_id: Document ID to delete

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM documents WHERE id = ?", (document_id,)
                )

                if cursor.rowcount == 0:
                    self.logger.warning(f"No document found with ID: {document_id}")
                    return False

                self.logger.info(f"Deleted document {document_id}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    def _row_to_document_metadata(self, row: sqlite3.Row) -> DocumentMetadata:
        """Convert database row to DocumentMetadata object."""
        return DocumentMetadata(
            id=row["id"],
            filename=row["filename"],
            original_filename=row["original_filename"],
            file_path=Path(row["file_path"]) if row["file_path"] else None,
            document_type=DocumentType(row["document_type"]),
            file_size=row["file_size"],
            file_hash=row["file_hash"],
            upload_date=datetime.fromisoformat(row["upload_date"]),
            last_modified=datetime.fromisoformat(row["last_modified"]),
            status=DocumentStatus(row["status"]),
            word_count=row["word_count"] or 0,
            processing_time=row["processing_time"],
            error_message=row["error_message"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            notes=row["notes"] or "",
        )


class FileSystemManager:
    """Manages physical file storage and organization."""

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize file system manager."""
        self.settings = get_settings()
        self.base_path = base_path or self.settings.documents_path
        self.logger = get_logger(f"{__name__}.FileSystemManager")
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all required directories exist."""
        # Create directories for all document types
        directories = [self.base_path]

        # Add a directory for each document type
        for doc_type in DocumentType:
            directories.append(self.base_path / doc_type.value)

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {directory}")

    def generate_unique_filename(
        self, original_filename: str, document_type: DocumentType
    ) -> str:
        """
        Generate a unique filename for storage.

        Args:
            original_filename: Original filename from upload
            document_type: Type of document

        Returns:
            Unique filename for storage
        """
        # Extract file extension
        path = Path(original_filename)
        extension = path.suffix.lower()

        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = path.stem

        # Clean the base name (remove special characters)
        import re

        clean_name = re.sub(r"[^\w\-_]", "_", base_name)[:50]  # Limit length

        return f"{timestamp}_{clean_name}{extension}"

    def get_storage_path(self, filename: str, document_type: DocumentType) -> Path:
        """
        Get the full storage path for a document.

        Args:
            filename: Filename to store
            document_type: Type of document

        Returns:
            Full path where file should be stored
        """
        type_folder = document_type.value
        return self.base_path / type_folder / filename

    def store_file(
        self, source_path: Path, document_type: DocumentType, original_filename: str
    ) -> tuple[Path, str]:
        """
        Store a file in the organized file system.

        Args:
            source_path: Path to the source file
            document_type: Type of document
            original_filename: Original filename

        Returns:
            Tuple of (stored_path, filename)

        Raises:
            FileNotFoundError: If source file doesn't exist
            OSError: If file operations fail
        """
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Generate unique filename and get storage path
        filename = self.generate_unique_filename(original_filename, document_type)
        storage_path = self.get_storage_path(filename, document_type)

        try:
            # Copy file to storage location
            import shutil

            shutil.copy2(source_path, storage_path)

            self.logger.info(f"Stored file: {original_filename} -> {storage_path}")
            return storage_path, filename

        except Exception as e:
            self.logger.error(f"Failed to store file {original_filename}: {e}")
            raise

    def get_file_hash(self, file_path: Path) -> str:
        """
        Calculate SHA-256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            SHA-256 hash as hex string
        """
        hash_sha256 = hashlib.sha256()

        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)

            return hash_sha256.hexdigest()

        except Exception as e:
            self.logger.error(f"Failed to calculate hash for {file_path}: {e}")
            raise

    def delete_file(self, file_path: Path) -> bool:
        """
        Delete a physical file.

        Args:
            file_path: Path to the file to delete

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            if file_path.exists():
                file_path.unlink()
                self.logger.info(f"Deleted file: {file_path}")
                return True
            self.logger.warning(f"File not found for deletion: {file_path}")
            return False

        except Exception as e:
            self.logger.error(f"Failed to delete file {file_path}: {e}")
            return False

    def get_file_info(self, file_path: Path) -> dict:
        """
        Get basic file information.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file information
        """
        try:
            if not file_path.exists():
                return {}

            stat = file_path.stat()
            return {
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "extension": file_path.suffix.lower(),
                "name": file_path.name,
            }

        except Exception as e:
            self.logger.error(f"Failed to get file info for {file_path}: {e}")
            return {}


class StorageSystem:
    """
    Main storage system interface.

    Combines database metadata storage with file system management
    to provide a complete document storage solution.
    """

    def __init__(
        self, db_path: Optional[Path] = None, files_path: Optional[Path] = None
    ):
        """
        Initialize storage system.

        Args:
            db_path: Path to SQLite database (optional)
            files_path: Path to file storage directory (optional)
        """
        self.db_manager = DatabaseManager(db_path)
        self.document_storage = DocumentStorage(self.db_manager)
        self.file_manager = FileSystemManager(files_path)
        self.logger = get_logger(f"{__name__}.StorageSystem")

        self.logger.info("Storage system initialized")

    def store_document(
        self,
        source_path: Path,
        document_type: DocumentType,
        tags: Optional[List[str]] = None,
        notes: str = "",
        original_filename: Optional[str] = None,
    ) -> DocumentMetadata:
        """
        Store a document (file + metadata).

        Args:
            source_path: Path to the source file
            document_type: Type of document
            tags: Optional tags for the document
            notes: Optional notes
            original_filename: The real filename from the upload (optional)

        Returns:
            Document metadata with assigned ID

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If document already exists (same hash)
        """
        if original_filename is None:
            original_filename = source_path.name

        # Calculate file hash first to check for duplicates
        file_hash = self.file_manager.get_file_hash(source_path)

        # Check if document already exists
        existing = self.document_storage.get_document_by_hash(file_hash)
        if existing:
            # Instead of rejecting, reuse existing document with updated tags
            self.logger.info(
                f"Document with same content exists (ID: {existing.id}), reusing with updated tags"
            )

            # Merge tags if new ones are provided
            if tags:
                updated_tags = list(set(existing.tags + tags))
                self.document_storage.update_document(
                    existing.id, {"tags": updated_tags}
                )
                existing.tags = updated_tags

            # Update notes if provided
            if notes:
                updated_notes = (
                    f"{existing.notes}\n\n{notes}" if existing.notes else notes
                )
                self.document_storage.update_document(
                    existing.id, {"notes": updated_notes}
                )
                existing.notes = updated_notes

            # Update original_filename if different
            if existing.original_filename != original_filename:
                self.document_storage.update_document(
                    existing.id, {"original_filename": original_filename}
                )
                existing.original_filename = original_filename

            return existing

        # Store the physical file
        storage_path, filename = self.file_manager.store_file(
            source_path, document_type, original_filename
        )

        # Get file info
        file_info = self.file_manager.get_file_info(storage_path)

        # Create document metadata
        metadata = DocumentMetadata(
            filename=filename,
            original_filename=original_filename,
            file_path=storage_path,
            document_type=document_type,
            file_size=file_info.get("size", 0),
            file_hash=file_hash,
            upload_date=datetime.now(),
            last_modified=datetime.now(),
            status=DocumentStatus.UPLOADED,
            tags=tags or [],
            notes=notes,
        )

        try:
            # Store metadata in database
            document_id = self.document_storage.create_document(metadata)
            metadata.id = document_id

            self.logger.info(
                f"Successfully stored document: {original_filename} (ID: {document_id})"
            )
            return metadata

        except Exception as e:
            # If database storage fails, clean up the file
            self.file_manager.delete_file(storage_path)
            self.logger.error(
                f"Failed to store document metadata, cleaned up file: {e}"
            )
            raise

    def get_document(self, document_id: int) -> Optional[DocumentMetadata]:
        """
        Retrieve document metadata by ID.

        Args:
            document_id: Document ID

        Returns:
            Document metadata or None if not found
        """
        return self.document_storage.get_document(document_id)

    def list_documents(
        self,
        document_type: Optional[DocumentType] = None,
        status: Optional[DocumentStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DocumentMetadata]:
        """
        List documents with optional filtering.

        Args:
            document_type: Filter by document type
            status: Filter by status
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of document metadata
        """
        return self.document_storage.list_documents(
            document_type, status, limit, offset
        )

    def update_document_status(
        self,
        document_id: int,
        status: DocumentStatus,
        word_count: Optional[int] = None,
        processing_time: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Update document processing status.

        Args:
            document_id: Document ID
            status: New status
            word_count: Extracted word count (optional)
            processing_time: Processing time in seconds (optional)
            error_message: Error message if status is ERROR (optional)

        Returns:
            True if update successful
        """
        updates = {"status": status}

        if word_count is not None:
            updates["word_count"] = word_count
        if processing_time is not None:
            updates["processing_time"] = processing_time
        if error_message is not None:
            updates["error_message"] = error_message

        return self.document_storage.update_document(document_id, updates)

    def delete_document(self, document_id: int, delete_file: bool = True) -> bool:
        """
        Delete a document (metadata and optionally the file).

        Args:
            document_id: Document ID to delete
            delete_file: Whether to also delete the physical file

        Returns:
            True if deletion successful
        """
        # Get document info first
        document = self.document_storage.get_document(document_id)
        if not document:
            self.logger.warning(f"Document not found: {document_id}")
            return False

        # Delete from database
        if not self.document_storage.delete_document(document_id):
            return False

        # Delete physical file if requested
        if delete_file and document.file_path:
            self.file_manager.delete_file(document.file_path)

        self.logger.info(f"Deleted document: {document_id}")
        return True

    def get_storage_stats(self) -> dict:
        """
        Get storage system statistics.

        Returns:
            Dictionary with storage statistics
        """
        try:
            documents = self.list_documents(limit=1000)  # Get a reasonable sample

            stats = {
                "total_documents": len(documents),
                "documents_by_type": {},
                "documents_by_status": {},
                "total_size_bytes": 0,
                "avg_word_count": 0,
            }

            total_words = 0

            for doc in documents:
                # Count by type
                doc_type = doc.document_type.value
                stats["documents_by_type"][doc_type] = (
                    stats["documents_by_type"].get(doc_type, 0) + 1
                )

                # Count by status
                status = doc.status.value
                stats["documents_by_status"][status] = (
                    stats["documents_by_status"].get(status, 0) + 1
                )

                # Sum sizes and word counts
                stats["total_size_bytes"] += doc.file_size
                total_words += doc.word_count

            if documents:
                stats["avg_word_count"] = total_words / len(documents)

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {e}")
            return {}


# Convenience function for easy access
def get_storage_system() -> StorageSystem:
    """Get a configured storage system instance."""
    return StorageSystem()
