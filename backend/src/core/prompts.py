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
Prompt System for Job Application Helper.

Streamlined prompt management:
- Single system prompt for all tasks
- 3 core content types: cover letter, interview answer, general response
- Clean message building
- High performance implementation
"""

from enum import Enum
from typing import Dict, List, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Single system prompt for all job application tasks
SYSTEM_PROMPT = """You are an expert career advisor specializing in professional job application materials.

CORE CAPABILITIES:
- Cover letter writing with modern best practices
- Interview preparation and response crafting
- Professional content refinement and optimization
- Career strategy and positioning advice

FUNDAMENTAL PRINCIPLES:
🔒 **STRICT CONTEXT ADHERENCE**: Use ONLY information explicitly provided in documents. Never invent or assume details about the candidate's background, experience, or qualifications.

📊 **EVIDENCE-BASED APPROACH**: Support all claims with specific examples and quantifiable results from the provided context.

🎯 **PROFESSIONAL EXCELLENCE**: Maintain high standards for clarity, structure, and persuasive impact in all communications.

⚡ **ATS AWARENESS**: Optimize content for modern hiring systems while preserving human readability.

When information is missing from the provided context, clearly state what additional details would be needed rather than making assumptions."""


class PromptType(Enum):
    """Core prompt types for job application assistance."""
    COVER_LETTER = "cover_letter"
    INTERVIEW_ANSWER = "interview_answer"  
    GENERAL_RESPONSE = "general_response"


class PromptManager:
    """Efficient prompt manager for job application assistance."""

    def __init__(self):
        """Initialize the prompt manager."""
        self.logger = get_logger(f"{__name__}.PromptManager")
        self.logger.info("Prompt manager initialized")

    def build_messages(
        self,
        prompt_type: PromptType,
        user_query: str,
        context: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> List[Dict[str, str]]:
        """
        Build messages for LLM generation.
        
        Args:
            prompt_type: Type of prompt (cover_letter, interview_answer, general_response)
            user_query: User's request
            context: Document context
            conversation_history: Previous messages (last 8 kept)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            List of message dicts with role and content
        """
        messages = []
        
        # System prompt
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
        
        # Conversation history (last 8 messages only for performance)
        if conversation_history:
            for msg in conversation_history[-8:]:
                if msg.get("role") in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Build user message
        user_parts = []
        
        if context and context.strip():
            user_parts.extend(["DOCUMENT CONTEXT:", context, ""])
        
        user_parts.extend(["REQUEST:", user_query])
        
        # Simple instructions based on type
        instructions = self._get_instructions(prompt_type, bool(context))
        if instructions:
            user_parts.extend(["", "INSTRUCTIONS:", instructions])
        
        messages.append({"role": "user", "content": "\n".join(user_parts)})
        return messages

    def _get_instructions(self, prompt_type: PromptType, has_context: bool) -> str:
        """Get simple instructions based on type and context availability."""
        base = []
        
        if has_context:
            base.extend([
                "- Use ONLY information from the document context",
                "- Reference specific details from the documents",
                "- State what's missing if context is incomplete"
            ])
        else:
            base.extend([
                "- No specific documents available",
                "- Provide general guidance",
                "- Suggest uploading relevant documents for personalization"
            ])
        
        if prompt_type == PromptType.COVER_LETTER:
            base.extend([
                "- Structure as professional cover letter",
                "- Include quantified achievements from context",
                "- Compelling opening and confident call to action"
            ])
        elif prompt_type == PromptType.INTERVIEW_ANSWER:
            base.extend([
                "- Structured, professional response",
                "- Use STAR method for behavioral questions",
                "- Keep concise (1-2 minutes when spoken)"
            ])
        
        return "\n".join(base)

    def build_prompt(self, prompt_type: PromptType, user_query: str, 
                    context: str = "", **kwargs) -> str:
        """Build single prompt string."""
        messages = self.build_messages(prompt_type, user_query, context)
        return "\n\n".join(msg["content"] for msg in messages)

    def build_user_prompt(self, prompt_type: PromptType, variables: Dict) -> str:
        """Build user prompt from variables."""
        user_query = variables.get("user_query", "")
        context = variables.get("context", "")
        
        messages = self.build_messages(prompt_type, user_query, context)
        # Return just the user message content
        for msg in messages:
            if msg["role"] == "user":
                return msg["content"]
        return user_query

    def build_system_prompt(self, prompt_type: PromptType, context=None) -> str:
        """Get system prompt."""
        return SYSTEM_PROMPT


# Global instance
_prompt_manager = None

def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager