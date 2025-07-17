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
System Evaluation Script

Evaluates the Job Application Helper system with simplified architecture.
Tests LLM integration, document processing, and chat functionality.
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.llm_providers.factory import get_default_provider
from src.core.simple_chat_controller import create_simple_chat_controller
from src.core.simple_document_service import get_simple_document_service
from src.core.simple_document_handler import create_simple_document_handler
from src.core.storage import DocumentType, get_storage_system
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SystemEvaluator:
    """Evaluates the simplified Job Application Helper system."""

    def __init__(self):
        self.settings = get_settings()
        self.results = {}
        self.start_time = time.time()

    def run_evaluation(self) -> Dict:
        """Run complete system evaluation."""
        print("🔍 JOB APPLICATION HELPER - SYSTEM EVALUATION")
        print("=" * 60)
        print("Testing simplified architecture without RAG")
        print()

        # Run evaluation tests
        tests = [
            ("LLM Provider", self.test_llm_provider),
            ("Document Storage", self.test_document_storage),
            ("Document Service", self.test_document_service),
            ("Document Handler", self.test_document_handler),
            ("Chat Controller", self.test_chat_controller),
            ("Integration", self.test_integration),
        ]

        results = {}
        
        for test_name, test_func in tests:
            try:
                print(f"\n⏳ Running {test_name} test...")
                result = test_func()
                results[test_name] = result
                status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
                print(f"📊 {test_name}: {status}")
                
                if not result.get("success", False) and result.get("error"):
                    print(f"   Error: {result['error']}")
                    
            except Exception as e:
                print(f"❌ Error in {test_name}: {e}")
                results[test_name] = {"success": False, "error": str(e)}

        # Generate summary
        self.generate_summary(results)
        
        return results

    def test_llm_provider(self) -> Dict:
        """Test LLM provider functionality."""
        try:
            provider = get_default_provider()
            
            if not provider:
                return {"success": False, "error": "No LLM provider available"}
            
            # Test basic generation
            from src.core.llm_providers.base import GenerationRequest, ContentType
            
            request = GenerationRequest(
                prompt="Hello, this is a test. Please respond with 'Test successful'.",
                max_tokens=50,
                temperature=0.1,
                content_type=ContentType.GENERAL_RESPONSE
            )
            
            response = provider.generate_content(request)
            
            if response.success:
                return {
                    "success": True,
                    "provider": response.provider_used,
                    "model": response.model_used,
                    "tokens": response.tokens_used,
                    "response": response.content[:100]
                }
            else:
                return {"success": False, "error": response.error}
                
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_document_storage(self) -> Dict:
        """Test document storage system."""
        try:
            storage_system = get_storage_system()
            
            # Test listing documents
            docs = storage_system.list_documents(limit=10)
            
            # Test storage stats
            stats = storage_system.get_storage_stats()
            
            return {
                "success": True,
                "document_count": len(docs),
                "storage_stats": stats
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_document_service(self) -> Dict:
        """Test simple document service."""
        try:
            document_service = get_simple_document_service()
            
            # Test getting all documents
            all_docs = document_service.get_all_documents()
            
            # Test search functionality
            search_results = document_service.search_documents("test query")
            
            return {
                "success": True,
                "total_documents": len(all_docs),
                "search_results": len(search_results)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_document_handler(self) -> Dict:
        """Test document handler functionality."""
        try:
            handler = create_simple_document_handler()
            
            # Create a test document
            test_content = "This is a test CV document with experience and skills."
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_content)
                test_file = Path(f.name)
            
            try:
                # Test document upload
                upload_results = list(handler.upload_candidate_document(
                    test_file, 
                    "test_cv.txt"
                ))
                
                # Check if upload was successful
                final_result = upload_results[-1] if upload_results else None
                
                if final_result and final_result.success:
                    return {
                        "success": True,
                        "document_id": final_result.document_id,
                        "metadata": final_result.metadata
                    }
                else:
                    return {"success": False, "error": "Upload failed"}
                    
            finally:
                # Clean up test file
                if test_file.exists():
                    test_file.unlink()
                    
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_chat_controller(self) -> Dict:
        """Test chat controller functionality."""
        try:
            chat_controller = create_simple_chat_controller()
            
            # Test basic chat
            response = chat_controller.process_message(
                "Hello, can you help me with my job application?",
                session_id="test_session"
            )
            
            if response.success:
                return {
                    "success": True,
                    "response_length": len(response.content),
                    "metadata": response.metadata
                }
            else:
                return {"success": False, "error": response.error}
                
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_integration(self) -> Dict:
        """Test integration between components."""
        try:
            # Test document upload and chat integration
            handler = create_simple_document_handler()
            chat_controller = create_simple_chat_controller()
            
            # Create test document
            test_content = "John Doe\nSoftware Engineer\nExperience: Python, React, Node.js\nEducation: Computer Science"
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_content)
                test_file = Path(f.name)
            
            try:
                # Upload document
                upload_results = list(handler.upload_candidate_document(
                    test_file, 
                    "john_doe_cv.txt"
                ))
                
                final_result = upload_results[-1] if upload_results else None
                
                if not final_result or not final_result.success:
                    return {"success": False, "error": "Document upload failed"}
                
                # Test chat with document context
                response = chat_controller.process_message(
                    "What programming languages do I know?",
                    session_id="integration_test"
                )
                
                if response.success:
                    return {
                        "success": True,
                        "document_uploaded": True,
                        "chat_response": response.content[:200],
                        "documents_used": response.metadata.get("documents_used", 0)
                    }
                else:
                    return {"success": False, "error": f"Chat failed: {response.error}"}
                    
            finally:
                # Clean up test file
                if test_file.exists():
                    test_file.unlink()
                    
        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate_summary(self, results: Dict):
        """Generate evaluation summary."""
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in results.values() if result.get("success", False))
        total = len(results)
        
        print(f"Tests passed: {passed}/{total}")
        print(f"Success rate: {passed/total*100:.1f}%")
        
        print("\n📋 Test Results:")
        for test_name, result in results.items():
            status = "✅" if result.get("success", False) else "❌"
            print(f"   {status} {test_name}")
            
            if not result.get("success", False) and result.get("error"):
                print(f"      Error: {result['error']}")
        
        # System recommendations
        print("\n💡 System Status:")
        
        if passed == total:
            print("   🎉 All tests passed! System is working correctly.")
        else:
            print(f"   ⚠️  {total - passed} tests failed. Please review the issues above.")
            
        # Configuration recommendations
        if not results.get("LLM Provider", {}).get("success", False):
            print("   🔧 Configure LLM provider API keys in environment variables")
            
        if not results.get("Document Storage", {}).get("success", False):
            print("   🔧 Check document storage system configuration")
            
        print(f"\n⏱️  Evaluation completed in {time.time() - self.start_time:.2f} seconds")


def main():
    """Main evaluation function."""
    try:
        evaluator = SystemEvaluator()
        results = evaluator.run_evaluation()
        
        # Return appropriate exit code
        passed = sum(1 for result in results.values() if result.get("success", False))
        total = len(results)
        
        return 0 if passed == total else 1
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        logger.error(f"Evaluation error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
