#!/usr/bin/env python3
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
Document Listing Script

Lists all documents stored in the Job Application Helper system.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.storage import DocumentType, get_storage_system
from src.core.simple_document_service import get_simple_document_service
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"📄 {title}")
    print(f"{'='*60}")


def list_storage_documents():
    """List documents in the storage system."""
    print_header("DOCUMENT STORAGE SYSTEM")
    
    try:
        storage_system = get_storage_system()
        
        # Get all documents
        all_docs = storage_system.list_documents(limit=1000)
        
        if not all_docs:
            print("📂 No documents found in storage system")
            return
        
        print(f"📊 Total documents: {len(all_docs)}")
        
        # Group by document type
        by_type = {}
        for doc in all_docs:
            doc_type = doc.document_type.value
            if doc_type not in by_type:
                by_type[doc_type] = []
            by_type[doc_type].append(doc)
        
        # Display by type
        for doc_type, docs in by_type.items():
            print(f"\n📋 {doc_type.upper()} ({len(docs)} documents)")
            print("-" * 40)
            
            for doc in docs:
                size_mb = doc.file_size / (1024 * 1024) if doc.file_size else 0
                size_str = f"{size_mb:.1f}MB" if size_mb >= 1 else f"{doc.file_size/1024:.1f}KB" if doc.file_size else "Unknown"
                
                print(f"   📄 {doc.original_filename}")
                print(f"      ID: {doc.id}")
                print(f"      Size: {size_str}")
                print(f"      Uploaded: {doc.upload_date}")
                print(f"      Tags: {', '.join(doc.tags) if doc.tags else 'None'}")
                if doc.notes:
                    print(f"      Notes: {doc.notes}")
                print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error accessing storage system: {e}")
        return False


def list_document_service_contents():
    """List documents in the simple document service."""
    print_header("DOCUMENT SERVICE")
    
    try:
        document_service = get_simple_document_service()
        
        # Get all documents from service
        all_docs = document_service.get_all_documents()
        
        if not all_docs:
            print("📂 No documents found in document service")
            return
        
        print(f"📊 Total documents in service: {len(all_docs)}")
        
        # Group by document type
        by_type = {}
        for doc in all_docs:
            doc_type = doc.get("document_type", "unknown")
            if doc_type not in by_type:
                by_type[doc_type] = []
            by_type[doc_type].append(doc)
        
        # Display by type
        for doc_type, docs in by_type.items():
            print(f"\n📋 {doc_type.upper()} ({len(docs)} documents)")
            print("-" * 40)
            
            for doc in docs:
                content_length = len(doc.get("content", ""))
                
                print(f"   📄 {doc.get('filename', 'Unknown')}")
                print(f"      ID: {doc.get('document_id', 'Unknown')}")
                print(f"      Content Length: {content_length:,} characters")
                print(f"      Metadata: {doc.get('metadata', {})}")
                print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error accessing document service: {e}")
        return False


def show_system_summary():
    """Show a summary of the document system."""
    print_header("SYSTEM SUMMARY")
    
    try:
        # Storage system stats
        storage_system = get_storage_system()
        storage_docs = storage_system.list_documents(limit=1000)
        
        # Document service stats
        document_service = get_simple_document_service()
        service_docs = document_service.get_all_documents()
        
        print(f"📊 Storage System: {len(storage_docs)} documents")
        print(f"📊 Document Service: {len(service_docs)} documents")
        
        # Check for consistency
        if len(storage_docs) != len(service_docs):
            print(f"⚠️  Document count mismatch between storage and service")
        else:
            print("✅ Document counts are consistent")
        
        # Show document types
        if storage_docs:
            type_counts = {}
            for doc in storage_docs:
                doc_type = doc.document_type.value
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            print("\n📋 Document Types:")
            for doc_type, count in type_counts.items():
                print(f"   {doc_type}: {count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating system summary: {e}")
        return False


def main():
    """Main function."""
    print("📄 JOB APPLICATION HELPER - DOCUMENT LISTING")
    print("=" * 60)
    
    # Run all checks
    checks = [
        ("System Summary", show_system_summary),
        ("Storage Documents", list_storage_documents),
        ("Document Service", list_document_service_contents),
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            print(f"\n⏳ Running {check_name}...")
            result = check_func()
            results[check_name] = result
            status = "✅ COMPLETED" if result else "❌ FAILED"
            print(f"📊 {check_name}: {status}")
        except Exception as e:
            print(f"❌ Error in {check_name}: {e}")
            results[check_name] = False
    
    # Summary
    print_header("LISTING SUMMARY")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"📊 Checks completed: {passed}/{total}")
    
    for check_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"   {status} {check_name}")
    
    if passed < total:
        print(f"\n⚠️  {total - passed} checks failed. Please review the output above.")
        return 1
    else:
        print("\n🎉 All checks completed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main()) 