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
LLM-based Query Analyzer for Job Application Helper.

This module provides sophisticated query analysis using LLM providers for:
- Intent detection and classification
- Document relevance weighting
- Query expansion and refinement
- Multi-query detection and handling
- Context-aware analysis

The analyzer uses the same LLM infrastructure as other components for consistent,
intelligent query understanding and routing.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.llm_providers.base import ContentType, GenerationRequest, LLMProvider
from src.core.llm_providers.factory import get_default_provider
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QueryAnalysis:
    """
    Results from query analysis containing intent, weighting, and metadata.
    """
    intent_type: str = "general"
    intent_parameters: Dict[str, Any] = None
    document_weights: Dict[str, float] = None
    is_multi_query: bool = False
    expanded_queries: List[str] = None
    confidence: float = 0.0
    reasoning: str = ""
    
    def __post_init__(self):
        """Initialize default values."""
        if self.intent_parameters is None:
            self.intent_parameters = {}
        if self.document_weights is None:
            self.document_weights = {"candidate": 0.4, "job": 0.3, "company": 0.3}
        if self.expanded_queries is None:
            self.expanded_queries = []


class QueryAnalyzer:
    """
    LLM-based query analyzer for intelligent intent detection and document weighting.
    
    This analyzer uses sophisticated LLM reasoning to understand user queries,
    detect intent, and determine optimal document retrieval strategies.
    """
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """
        Initialize the query analyzer.
        
        Args:
            llm_provider: LLM provider for analysis (will get default if None)
        """
        self.llm_provider = llm_provider or get_default_provider()
        self.llm_available = self.llm_provider is not None
        
        if not self.llm_available:
            logger.warning("Query analyzer initialized without LLM provider - using fallback rules")
        else:
            logger.info("Query analyzer initialized with LLM provider")
    
    def analyze_query(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> QueryAnalysis:
        """
        Analyze a user query to determine intent and document relevance.
        
        Args:
            query: User query to analyze
            conversation_history: Recent conversation context
            
        Returns:
            QueryAnalysis with intent, weights, and metadata
        """
        if not self.llm_available:
            logger.debug("No LLM available - using rule-based analysis")
            return self._fallback_analysis(query)
        
        try:
            # Use LLM for sophisticated query analysis
            return self._llm_analysis(query, conversation_history)
        except Exception as e:
            logger.warning(f"LLM query analysis failed: {e}")
            return self._fallback_analysis(query)
    
    def _llm_analysis(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> QueryAnalysis:
        """
        Perform LLM-based query analysis for sophisticated intent detection.
        
        Args:
            query: User query to analyze
            conversation_history: Recent conversation context
            
        Returns:
            QueryAnalysis with LLM-determined intent and weights
        """
        # Build context from conversation history
        context_summary = ""
        if conversation_history:
            recent_messages = conversation_history[-3:]  # Last 3 exchanges
            context_summary = "\n".join([
                f"{'User' if msg.get('role') == 'user' else 'Assistant'}: {msg.get('content', '')[:200]}"
                for msg in recent_messages
            ])
        
        # Create sophisticated analysis prompt
        analysis_prompt = f"""You are an expert query analyzer for a job application assistant system.

TASK: Analyze the user query to determine intent, document relevance, and provide strategic guidance.

**USER QUERY:** {query}

**CONVERSATION CONTEXT:**
{context_summary if context_summary else "No previous context"}

**ANALYSIS FRAMEWORK:**

1. **INTENT CLASSIFICATION** - Classify into ONE primary intent:
   - `cover_letter`: Writing or improving cover letters
   - `behavioral_interview`: Behavioral interview questions and STAR method responses
   - `achievement_quantifier`: Quantifying achievements and impact
   - `ats_optimizer`: ATS optimization and keyword matching
   - `interview_answer`: General interview question responses
   - `content_refinement`: Improving existing content
   - `general`: General advice, questions, or information

2. **DOCUMENT RELEVANCE WEIGHTS** - Assign weights (0.0-1.0, must sum to 1.0):
   - `candidate`: CV, experience, skills, achievements, personal background
   - `job`: Job descriptions, requirements, role details
   - `company`: Company information, culture, research, background

3. **QUERY CHARACTERISTICS**:
   - `multi_query`: Does this contain multiple distinct questions? (YES/NO)
   - `confidence`: How confident are you in this analysis? (0.0-1.0)

4. **STRATEGIC EXPANSION** - Suggest 1-3 expanded/refined queries that would help answer the original question better.

**RESPONSE FORMAT:**
Respond with ONLY a valid JSON object:

```json
{
    "intent_type": "cover_letter",
    "intent_parameters": {
        "company_name": "if mentioned",
        "role_title": "if mentioned",
        "specific_focus": "any specific aspects mentioned"
    },
    "document_weights": {
        "candidate": 0.6,
        "job": 0.3,
        "company": 0.1
    },
    "is_multi_query": false,
    "expanded_queries": [
        "What specific achievements should I highlight for this role?",
        "How can I connect my experience to this company's needs?"
    ],
    "confidence": 0.9,
    "reasoning": "Brief explanation of the analysis and weighting strategy"
}
```

**WEIGHTING GUIDELINES:**
- Cover letters: High candidate (0.5-0.7), moderate job (0.2-0.4), low company (0.1-0.2)
- Interview prep: Balanced candidate/job (0.4-0.5 each), moderate company (0.1-0.2)
- General advice: Balanced across all three (0.3-0.4 each)
- Company research: High company (0.5-0.7), moderate job (0.2-0.3), low candidate (0.1-0.2)

Analysis:"""

        # Use LLM for analysis
        request = GenerationRequest(
            prompt=analysis_prompt,
            content_type=ContentType.GENERAL_RESPONSE,
            context={"task": "query_analysis"},
            max_tokens=500,
            temperature=0.2,  # Low temperature for consistent analysis
        )
        
        response = self.llm_provider.generate_content(request)
        
        if not response.success:
            logger.error(f"LLM query analysis failed: {response.error}")
            return self._fallback_analysis(query)
        
        try:
            # Parse LLM response
            content = response.content.strip()
            
            # Clean JSON from markdown formatting
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            analysis_data = json.loads(content)
            
            # Validate and normalize weights
            weights = analysis_data.get("document_weights", {})
            weights = self._normalize_weights(weights)
            
            return QueryAnalysis(
                intent_type=analysis_data.get("intent_type", "general"),
                intent_parameters=analysis_data.get("intent_parameters", {}),
                document_weights=weights,
                is_multi_query=analysis_data.get("is_multi_query", False),
                expanded_queries=analysis_data.get("expanded_queries", []),
                confidence=float(analysis_data.get("confidence", 0.0)),
                reasoning=analysis_data.get("reasoning", "LLM analysis completed")
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse LLM query analysis: {e}")
            logger.debug(f"Raw LLM response: {response.content}")
            return self._fallback_analysis(query)
    
    def _fallback_analysis(self, query: str) -> QueryAnalysis:
        """
        Fallback rule-based analysis when LLM is unavailable.
        
        Args:
            query: User query to analyze
            
        Returns:
            QueryAnalysis with rule-based intent and weights
        """
        query_lower = query.lower()
        
        # Rule-based intent detection
        intent_type = "general"
        intent_parameters = {}
        
        # Cover letter patterns
        if any(term in query_lower for term in ["cover letter", "application letter", "motivation letter"]):
            intent_type = "cover_letter"
        
        # Interview patterns
        elif any(term in query_lower for term in ["interview", "behavioral", "star method", "tell me about"]):
            intent_type = "behavioral_interview"
        
        # Achievement patterns
        elif any(term in query_lower for term in ["quantify", "achievement", "impact", "results", "metrics"]):
            intent_type = "achievement_quantifier"
        
        # ATS patterns
        elif any(term in query_lower for term in ["ats", "keyword", "optimize", "applicant tracking"]):
            intent_type = "ats_optimizer"
        
        # Content refinement patterns
        elif any(term in query_lower for term in ["improve", "refine", "better", "enhance", "rewrite"]):
            intent_type = "content_refinement"
        
        # Rule-based document weighting
        weights = {"candidate": 0.4, "job": 0.3, "company": 0.3}
        
        # Candidate-focused queries
        if any(term in query_lower for term in ["my", "i", "me", "skills", "experience", "background"]):
            weights = {"candidate": 0.6, "job": 0.25, "company": 0.15}
        
        # Job-focused queries
        elif any(term in query_lower for term in ["job", "role", "position", "requirements", "qualifications"]):
            weights = {"candidate": 0.25, "job": 0.6, "company": 0.15}
        
        # Company-focused queries
        elif any(term in query_lower for term in ["company", "organization", "culture", "research"]):
            weights = {"candidate": 0.15, "job": 0.25, "company": 0.6}
        
        # Multi-query detection
        is_multi_query = any(separator in query for separator in ["?", "and", "also", "additionally", ";"])
        
        return QueryAnalysis(
            intent_type=intent_type,
            intent_parameters=intent_parameters,
            document_weights=weights,
            is_multi_query=is_multi_query,
            expanded_queries=[],
            confidence=0.7,  # Moderate confidence for rule-based
            reasoning=f"Rule-based analysis: detected {intent_type} intent with {max(weights, key=weights.get)} focus"
        )
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize document weights to ensure they sum to 1.0.
        
        Args:
            weights: Raw weights dictionary
            
        Returns:
            Normalized weights dictionary
        """
        # Ensure all required keys exist
        required_keys = ["candidate", "job", "company"]
        normalized = {}
        
        for key in required_keys:
            normalized[key] = max(0.0, min(1.0, weights.get(key, 0.33)))
        
        # Normalize to sum to 1.0
        total = sum(normalized.values())
        if total > 0:
            for key in normalized:
                normalized[key] /= total
        else:
            # Equal weights as fallback
            for key in normalized:
                normalized[key] = 1.0 / len(required_keys)
        
        return normalized
    
    def get_expanded_queries(self, original_query: str) -> List[str]:
        """
        Generate expanded queries for better context retrieval.
        
        Args:
            original_query: Original user query
            
        Returns:
            List of expanded/refined queries
        """
        analysis = self.analyze_query(original_query)
        return analysis.expanded_queries
    
    def get_document_priority(self, query: str) -> str:
        """
        Get the highest priority document type for a query.
        
        Args:
            query: User query
            
        Returns:
            Document type with highest weight ('candidate', 'job', or 'company')
        """
        analysis = self.analyze_query(query)
        return max(analysis.document_weights, key=analysis.document_weights.get) 