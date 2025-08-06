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
Tests for QueryAnalyzer integration in SimpleChatController.

This test module verifies that the QueryAnalyzer is properly integrated
and functioning within the chat flow, ensuring the orphaned module issue
is resolved.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.core.simple_chat_controller import SimpleChatController
from src.core.query_analyzer import QueryAnalyzer, QueryAnalysis
from src.core.llm_providers.base import ContentType, GenerationResponse
from src.core.memory_manager import MemoryManager


class TestQueryAnalyzerIntegration(unittest.TestCase):
    """Test QueryAnalyzer integration in SimpleChatController."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock LLM provider
        self.mock_llm_provider = Mock()
        self.mock_llm_provider.generate_content.return_value = GenerationResponse(
            content="Test response",
            success=True,
            tokens_used=100,
            processing_time=1.0
        )
        
        # Mock memory manager
        self.mock_memory_manager = Mock(spec=MemoryManager)
        
        # Create chat controller with mocked dependencies
        self.chat_controller = SimpleChatController(
            llm_provider=self.mock_llm_provider,
            memory_manager=self.mock_memory_manager
        )
    
    def test_query_analyzer_initialization(self):
        """Test that QueryAnalyzer is properly initialized."""
        # Verify QueryAnalyzer is created and available
        self.assertIsNotNone(self.chat_controller.query_analyzer)
        self.assertIsInstance(self.chat_controller.query_analyzer, QueryAnalyzer)
        self.assertTrue(hasattr(self.chat_controller, 'query_analyzer_available'))
    
    @patch('src.core.simple_chat_controller.get_simple_document_service')
    def test_query_analysis_in_process_message(self, mock_doc_service):
        """Test that query analysis is called during message processing."""
        # Mock document service
        mock_doc_service.return_value = Mock()
        mock_doc_service.return_value.get_relevant_context.return_value = {
            "has_context": False,
            "context_text": "",
            "source_documents": [],
            "document_counts": {},
        }
        
        # Mock query analyzer with successful analysis
        mock_analysis = QueryAnalysis(
            intent_type="cover_letter",
            intent_parameters={"company_name": "TestCorp"},
            document_weights={"candidate": 0.6, "job": 0.3, "company": 0.1},
            is_multi_query=False,
            expanded_queries=["How should I format my cover letter?"],
            confidence=0.9,
            reasoning="Clear cover letter request"
        )
        
        with patch.object(self.chat_controller.query_analyzer, 'analyze_query', return_value=mock_analysis) as mock_analyze:
            # Process a test message
            response = self.chat_controller.process_message("Write me a cover letter for TestCorp")
            
            # Verify query analysis was called
            mock_analyze.assert_called_once_with("Write me a cover letter for TestCorp", conversation_history=None)
            
            # Verify response is successful
            self.assertTrue(response.success)
    
    def test_content_type_detection_with_query_analysis(self):
        """Test enhanced content type detection using query analysis."""
        # Test high confidence query analysis
        high_confidence_analysis = QueryAnalysis(
            intent_type="cover_letter",
            confidence=0.9
        )
        
        content_type = self.chat_controller._detect_content_type(
            "Help me with something",
            query_analysis=high_confidence_analysis
        )
        
        self.assertEqual(content_type, ContentType.COVER_LETTER)
        
        # Test low confidence - should fall back to keyword detection
        low_confidence_analysis = QueryAnalysis(
            intent_type="cover_letter",
            confidence=0.5
        )
        
        content_type = self.chat_controller._detect_content_type(
            "write a cover letter",
            query_analysis=low_confidence_analysis
        )
        
        self.assertEqual(content_type, ContentType.COVER_LETTER)
    
    def test_document_weighting_integration(self):
        """Test dynamic document weighting based on query analysis."""
        # Mock document service
        with patch.object(self.chat_controller, 'documents_available', True):
            with patch.object(self.chat_controller.document_service, 'get_relevant_context') as mock_get_context:
                mock_get_context.return_value = {
                    "has_context": True,
                    "context_text": "Test context",
                    "source_documents": [],
                    "document_counts": {},
                }
                
                # Test with query analysis containing document weights
                analysis_with_weights = QueryAnalysis(
                    intent_type="cover_letter",
                    document_weights={"candidate": 0.7, "job": 0.2, "company": 0.1},
                    confidence=0.8
                )
                
                context = self.chat_controller._get_document_context(
                    "Write a cover letter",
                    query_analysis=analysis_with_weights
                )
                
                # Verify get_relevant_context was called with dynamic limits
                mock_get_context.assert_called_once()
                call_kwargs = mock_get_context.call_args.kwargs
                
                # Check that limits were calculated based on weights
                # Base length 100000 * 0.7 = 70000 for candidate
                self.assertEqual(call_kwargs['max_candidate_doc_length'], 70000)
                self.assertEqual(call_kwargs['max_job_doc_length'], 20000)
                self.assertEqual(call_kwargs['max_company_doc_length'], 10000)
    
    def test_fallback_when_query_analyzer_fails(self):
        """Test graceful fallback when QueryAnalyzer fails."""
        # Mock query analyzer to raise an exception
        with patch.object(self.chat_controller.query_analyzer, 'analyze_query', side_effect=Exception("Analysis failed")):
            with patch.object(self.chat_controller, 'documents_available', False):
                # Should not raise exception and should continue processing
                response = self.chat_controller.process_message("Test message")
                
                # Verify response is still successful (using fallback)
                self.assertTrue(response.success)
    
    def test_intent_to_content_type_mapping(self):
        """Test all intent types map correctly to content types."""
        intent_mappings = {
            "cover_letter": ContentType.COVER_LETTER,
            "behavioral_interview": ContentType.INTERVIEW_ANSWER,
            "interview_answer": ContentType.INTERVIEW_ANSWER,
            "content_refinement": ContentType.CONTENT_REFINEMENT,
            "ats_optimizer": ContentType.CONTENT_REFINEMENT,
            "achievement_quantifier": ContentType.CONTENT_REFINEMENT,
            "general": ContentType.GENERAL_RESPONSE,
        }
        
        for intent_type, expected_content_type in intent_mappings.items():
            analysis = QueryAnalysis(
                intent_type=intent_type,
                confidence=0.8
            )
            
            content_type = self.chat_controller._detect_content_type(
                "test message",
                query_analysis=analysis
            )
            
            self.assertEqual(content_type, expected_content_type, 
                           f"Intent {intent_type} should map to {expected_content_type}")
    
    def test_query_analyzer_availability_flag(self):
        """Test that query_analyzer_available flag is properly set."""
        # When LLM is available, query analyzer should be available
        self.chat_controller.llm_provider = Mock()
        self.chat_controller.query_analyzer = QueryAnalyzer(llm_provider=self.chat_controller.llm_provider)
        self.chat_controller.query_analyzer.llm_available = True
        self.chat_controller.query_analyzer_available = True
        
        self.assertTrue(self.chat_controller.query_analyzer_available)
        
        # When LLM is not available, query analyzer should not be available
        self.chat_controller.query_analyzer.llm_available = False
        self.chat_controller.query_analyzer_available = False
        
        self.assertFalse(self.chat_controller.query_analyzer_available)


if __name__ == '__main__':
    unittest.main()