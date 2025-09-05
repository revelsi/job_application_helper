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
Embeddings-based Semantic Query Analyzer for Job Application Helper.

This module provides lightweight, fast query analysis using sentence-transformer
embeddings instead of invoking an LLM for routing. It performs:
- Intent routing via semantic similarity against canonical intent prompts
- Document relevance weighting based on similarity to category exemplars
- Optional query expansion via nearest-neighbor variants (cheap heuristic)

Design goals:
- Remove per-message LLM call overhead from the analyzer path
- Keep the same external interface (QueryAnalysis) for compatibility
- Deterministic and fast with minimal memory footprint
"""

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional

from src.core.llm_providers.base import LLMProvider
from src.utils.logging import get_logger
from src.utils.config import get_settings

try:
    # Import lazily to avoid heavy import cost during cold start if unused
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:  # pragma: no cover - handled at runtime with graceful fallback
    SentenceTransformer = None  # type: ignore
    np = None  # type: ignore

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

    def __init__(self, llm_provider: LLMProvider):
        """Initialize the embeddings-based analyzer and load the model."""
        if llm_provider is None:
            raise ValueError("LLM provider is required for QueryAnalyzer")

        self.llm_provider = llm_provider
        self.settings = get_settings()

        # Initialize embeddings model
        self.embedding_model_name = self.settings.embedding_model_name
        self.embedding_cache_dir = (
            str(self.settings.embedding_cache_dir)
            if getattr(self.settings, "embedding_cache_dir", None)
            else None
        )

        self.embedder = None
        self.embeddings_available = False
        # Backward-compatibility flag expected by SimpleChatController/tests
        self.llm_available = False

        if SentenceTransformer is not None:
            try:
                self.embedder = SentenceTransformer(
                    self.embedding_model_name,
                    cache_folder=self.embedding_cache_dir,
                )
                self.embeddings_available = True
                self.llm_available = True
                logger.info(
                    f"Query analyzer using embeddings model: {self.embedding_model_name}"
                )
            except Exception as e:
                logger.warning(f"Failed to load embeddings model: {e}")
        else:
            logger.warning("sentence-transformers not installed; falling back to rules")

        # Precompute canonical intent exemplars and document category exemplars
        self.intent_labels = [
            "cover_letter",
            "behavioral_interview",
            "interview_answer",
            "content_refinement",
            "ats_optimizer",
            "achievement_quantifier",
            "general",
        ]

        self.intent_texts = {
            "cover_letter": "Requests to write or craft a professional cover letter.",
            "behavioral_interview": "Behavioral interview questions using STAR method, experiences.",
            "interview_answer": "General interview Q&A, responding to interview questions.",
            "content_refinement": "Improve, refine, or optimize existing text content.",
            "ats_optimizer": "Optimize resume or content for ATS and keywords.",
            "achievement_quantifier": "Quantify achievements, metrics, impact in resume or letters.",
            "general": "General career advice or miscellaneous requests.",
        }

        self.document_categories = ["candidate", "job", "company"]
        self.document_texts = {
            "candidate": "Information about the candidate, resume, CV, personal background.",
            "job": "Job description, role requirements, responsibilities, qualifications.",
            "company": "Company background, culture, mission, values, news.",
        }

        # Cache exemplar embeddings
        self.intent_matrix = None
        self.doccat_matrix = None
        if self.embeddings_available:
            self.intent_matrix = self._embed([self.intent_texts[i] for i in self.intent_labels])
            self.doccat_matrix = self._embed([self.document_texts[c] for c in self.document_categories])

    def analyze_query(
        self, query: str, conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> QueryAnalysis:
        """
        Analyze a user query to determine intent and document relevance.

        Args:
            query: User query to analyze
            conversation_history: Recent conversation context

        Returns:
            QueryAnalysis with intent, weights, and metadata
        """
        if not self.embeddings_available:
            logger.debug("Embeddings unavailable - using rule-based analysis")
            return self._fallback_analysis(query)

        try:
            return self._embedding_analysis(query)
        except Exception as e:
            logger.warning(f"Embeddings analysis failed: {e}")
            return self._fallback_analysis(query)

    def _embedding_analysis(self, query: str) -> QueryAnalysis:
        """Perform semantic routing using embeddings and cosine similarity."""
        assert self.embedder is not None and np is not None

        query_vec = self._embed([query])  # shape (1, d)

        # Intent classification
        intent_scores = self._cosine_sim(query_vec, self.intent_matrix)[0]
        best_intent_idx = int(np.argmax(intent_scores))
        intent_type = self.intent_labels[best_intent_idx]
        intent_confidence = float(intent_scores[best_intent_idx])

        # Document weighting
        doc_scores = self._cosine_sim(query_vec, self.doccat_matrix)[0]
        # Convert to non-negative and normalize
        doc_scores = np.clip(doc_scores, 0.0, None)
        total = float(np.sum(doc_scores)) or 1.0
        weights = {
            cat: float(score) / total for cat, score in zip(self.document_categories, doc_scores)
        }

        # Simple multi-query heuristic
        is_multi_query = any(sep in query for sep in [";", " and ", " also ", "? " ])

        # Lightweight expansion: take top-2 intent synonyms as hints (no LLM call)
        expanded_queries = []
        if intent_type == "cover_letter":
            expanded_queries = ["tailored cover letter", "quantified achievements"]
        elif intent_type in {"behavioral_interview", "interview_answer"}:
            expanded_queries = ["STAR method example", "concise interview answer"]
        elif intent_type == "content_refinement":
            expanded_queries = ["refine this text", "improve clarity and impact"]
        elif intent_type == "ats_optimizer":
            expanded_queries = ["ATS keywords", "optimize resume for ATS"]
        elif intent_type == "achievement_quantifier":
            expanded_queries = ["quantify results", "metrics and impact"]

        return QueryAnalysis(
            intent_type=intent_type,
            intent_parameters={},
            document_weights=self._normalize_weights(weights),
            is_multi_query=is_multi_query,
            expanded_queries=expanded_queries,
            confidence=float(min(max(intent_confidence, 0.0), 1.0)),
            reasoning="Embeddings-based semantic routing",
        )

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
        if any(
            term in query_lower
            for term in ["cover letter", "application letter", "motivation letter"]
        ):
            intent_type = "cover_letter"

        # Interview patterns
        elif any(
            term in query_lower
            for term in ["interview", "behavioral", "star method", "tell me about"]
        ):
            intent_type = "behavioral_interview"

        # Achievement patterns
        elif any(
            term in query_lower
            for term in ["quantify", "achievement", "impact", "results", "metrics"]
        ):
            intent_type = "achievement_quantifier"

        # ATS patterns
        elif any(
            term in query_lower
            for term in ["ats", "keyword", "optimize", "applicant tracking"]
        ):
            intent_type = "ats_optimizer"

        # Content refinement patterns
        elif any(
            term in query_lower
            for term in ["improve", "refine", "better", "enhance", "rewrite"]
        ):
            intent_type = "content_refinement"

        # Rule-based document weighting
        weights = {"candidate": 0.4, "job": 0.3, "company": 0.3}

        # Candidate-focused queries
        if any(
            term in query_lower
            for term in ["my", "i", "me", "skills", "experience", "background"]
        ):
            weights = {"candidate": 0.6, "job": 0.25, "company": 0.15}

        # Job-focused queries
        elif any(
            term in query_lower
            for term in ["job", "role", "position", "requirements", "qualifications"]
        ):
            weights = {"candidate": 0.25, "job": 0.6, "company": 0.15}

        # Company-focused queries
        elif any(
            term in query_lower
            for term in ["company", "organization", "culture", "research"]
        ):
            weights = {"candidate": 0.15, "job": 0.25, "company": 0.6}

        # Multi-query detection
        is_multi_query = any(
            separator in query
            for separator in ["?", "and", "also", "additionally", ";"]
        )

        return QueryAnalysis(
            intent_type=intent_type,
            intent_parameters=intent_parameters,
            document_weights=weights,
            is_multi_query=is_multi_query,
            expanded_queries=[],
            confidence=0.7,  # Moderate confidence for rule-based
            reasoning=f"Rule-based analysis: detected {intent_type} intent with {max(weights, key=weights.get)} focus",
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

    # --- Embedding helpers ---
    def _embed(self, texts: List[str]):
        assert self.embedder is not None
        # sentence-transformers returns List[List[float]]; convert to numpy array for ops
        vectors = self.embedder.encode(texts, normalize_embeddings=True)
        if np is not None:
            return np.array(vectors, dtype=float)
        return vectors

    def _cosine_sim(self, a, b):
        assert np is not None
        # a: (n, d), b: (m, d) with unit-normalized rows
        return np.matmul(a, b.T)

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
