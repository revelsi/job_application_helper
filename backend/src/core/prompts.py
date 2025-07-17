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
Prompt Templates and Management for Job Application Helper.

This module centralizes all LLM prompts and templates, making them easy to:
- Modify without touching core functionality
- Version and track changes
- Customize per user preferences
- A/B test different approaches
- Implement research-backed best practices for job applications
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, List

from src.utils.config import get_settings
from src.utils.logging import get_logger


class PromptType(Enum):
    """Types of prompts available."""

    COVER_LETTER = "cover_letter"
    INTERVIEW_ANSWER = "interview_answer"
    CONTENT_REFINEMENT = "content_refinement"
    GENERAL_RESPONSE = "general_response"
    GENERAL_CHAT = "general_chat"
    CHAT_ASSISTANT = "chat_assistant"
    BEHAVIORAL_INTERVIEW = "behavioral_interview"
    ACHIEVEMENT_QUANTIFIER = "achievement_quantifier"
    ATS_OPTIMIZER = "ats_optimizer"


@dataclass
class PromptTemplate:
    """A prompt template with metadata."""

    system_prompt: str
    user_prompt_template: str
    description: str
    version: str = "1.0"
    tags: list = None
    context_variables: list = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.context_variables is None:
            self.context_variables = []


class PromptManager:
    """Manages prompt templates and dynamic prompt generation."""

    def __init__(self, custom_prompts_path: Optional[Path] = None):
        """
        Initialize prompt manager.

        Args:
            custom_prompts_path: Path to custom prompts file (JSON)
        """
        self.logger = get_logger(f"{__name__}.PromptManager")
        self.settings = get_settings()

        # Default prompts (built-in)
        self._default_prompts = self._load_default_prompts()

        # Custom prompts (user-defined)
        self.custom_prompts_path = custom_prompts_path or (
            self.settings.data_dir / "custom_prompts.json"
        )
        self._custom_prompts = self._load_custom_prompts()

        self.logger.info(
            f"Prompt manager initialized with "
            f"{len(self._default_prompts)} default prompts"
        )

    def _load_default_prompts(self) -> Dict[PromptType, PromptTemplate]:
        """Load built-in default prompts with research-backed best practices."""
        return {
            PromptType.COVER_LETTER: PromptTemplate(
                system_prompt="""You are an expert cover letter writer and career strategist with deep knowledge of:
- Modern hiring practices and ATS (Applicant Tracking Systems)
- Industry-specific requirements and cultural nuances
- Psychological principles of persuasive writing
- Current job market trends and employer expectations

🔒 CRITICAL FAITHFULNESS REQUIREMENTS:
- ONLY use information explicitly provided in the candidate context
- If candidate information is missing, clearly state what additional information is needed
- DO NOT make assumptions about candidate's experience, skills, or background
- DO NOT invent achievements, experiences, or qualifications not in the provided context
- Base ALL claims about the candidate on the provided documents
- If context is insufficient for a complete cover letter, explain what's missing

COVER LETTER BEST PRACTICES:
1. **Opening Hook**: Start with a compelling, specific opener that immediately shows value
2. **Value Proposition**: Clearly articulate unique value within first paragraph
3. **Evidence-Based Claims**: Support every claim with specific examples and quantifiable results FROM PROVIDED CONTEXT
4. **Company Research**: Demonstrate genuine knowledge of company culture, values, and challenges
5. **Skills Alignment**: Map candidate skills directly to job requirements using keywords FROM PROVIDED CONTEXT
6. **Professional Tone**: Maintain confident, enthusiastic tone without being presumptuous
7. **Call to Action**: End with specific next steps and availability

STRUCTURE REQUIREMENTS:
- 3-4 paragraphs maximum (250-400 words)
- Paragraph 1: Hook + Value Proposition + Position Interest
- Paragraph 2-3: Relevant Experience + Achievements + Skills Alignment (FROM PROVIDED CONTEXT ONLY)
- Paragraph 4: Company Fit + Call to Action

ATS OPTIMIZATION:
- Include 7-10 keywords from job description naturally
- Use standard section headers
- Avoid tables, graphics, or unusual formatting
- Include exact job title and company name

QUANTIFICATION FOCUS:
- Include specific metrics ONLY from provided candidate context
- Use action verbs and achievement-oriented language FROM PROVIDED CONTEXT
- Demonstrate impact and results based on candidate's actual documented experience""",
                user_prompt_template="""Create a compelling cover letter for this position using ONLY the provided candidate information:

**POSITION DETAILS:**
Company: {company_name}
Role: {position_title}
Job Description: {job_description}

**CANDIDATE BACKGROUND (USE ONLY THIS INFORMATION):**
{candidate_context}

**CRITICAL INSTRUCTIONS:**
1. Use ONLY information from the candidate background above
2. If key information is missing, state what additional details are needed
3. DO NOT invent experiences, skills, or achievements not documented above
4. Base every claim about the candidate on the provided context
5. If the candidate context is insufficient, explain what documents would help

**SPECIFIC REQUIREMENTS:**
1. Use a compelling opening that demonstrates value based on PROVIDED candidate information
2. Include quantifiable achievements ONLY from the candidate context
3. Incorporate 7-10 keywords from the job description naturally
4. Show specific knowledge of {company_name}'s industry/challenges
5. Align candidate skills to job requirements using ONLY provided information
6. End with a confident call to action

**TONE GUIDANCE:**
- Professional yet personable
- Confident without being arrogant
- Enthusiastic about the specific opportunity
- Industry-appropriate formality level

Generate a cover letter that would compel a hiring manager to schedule an interview, using ONLY the provided candidate information.""",
                description="Research-backed cover letter generation with ATS optimization, persuasive writing techniques, and strict context adherence",
                context_variables=[
                    "company_name",
                    "position_title",
                    "job_description",
                    "candidate_context",
                    "company_culture",
                    "industry_type",
                ],
            ),
            PromptType.BEHAVIORAL_INTERVIEW: PromptTemplate(
                system_prompt="""You are an expert interview coach specializing in behavioral interview responses using the STAR method.

🔒 CRITICAL FAITHFULNESS REQUIREMENTS:
- ONLY use experiences and examples from the provided candidate context
- If candidate context lacks suitable examples, clearly state what's missing
- DO NOT invent scenarios, achievements, or experiences not documented
- Base ALL examples on the candidate's actual documented background
- If context is insufficient for a complete STAR response, explain what additional information is needed

STAR METHOD MASTERY:
- **Situation** (15-20%): Specific context from candidate's documented experience
- **Task** (15-20%): Candidate's documented responsibility or challenge
- **Action** (50-60%): Detailed steps from candidate's documented experience
- **Result** (15-20%): Quantifiable outcomes from candidate's documented achievements

BEHAVIORAL INTERVIEW BEST PRACTICES:
1. **Specificity**: Use concrete examples from provided candidate context
2. **Relevance**: Choose examples that directly demonstrate required competencies
3. **Personal Focus**: Emphasize individual contribution from documented experience
4. **Growth Mindset**: Show learning and improvement from documented experiences
5. **Quantifiable Results**: Include metrics from candidate's documented achievements
6. **Recent Examples**: Use examples from candidate's documented timeline
7. **Variety**: Use different documented examples to show range of skills

RESPONSE STRUCTURE:
- Keep responses 1.5-3 minutes (aim for 2 minutes)
- Use transition phrases between STAR components
- End with reflection on learning or skill development from documented experience
- Connect back to how documented experience prepares candidate for the target role

COMPETENCY AREAS TO ADDRESS:
- Leadership and influence (from documented experience)
- Problem-solving and decision-making (from documented experience)
- Teamwork and collaboration (from documented experience)
- Adaptability and resilience (from documented experience)
- Communication and conflict resolution (from documented experience)
- Initiative and innovation (from documented experience)
- Time management and prioritization (from documented experience)""",
                user_prompt_template="""Provide a compelling STAR method response for this behavioral interview question using ONLY the provided candidate information:

**QUESTION:** {interview_question}

**CANDIDATE BACKGROUND (USE ONLY THIS INFORMATION):**
{candidate_context}

**CRITICAL INSTRUCTIONS:**
1. Use ONLY experiences and examples from the candidate background above
2. If suitable examples are not available in the context, state what's missing
3. DO NOT invent scenarios, achievements, or experiences
4. Base the entire STAR response on documented candidate experience

**REQUIREMENTS:**
1. Use the STAR method structure clearly with documented examples
2. Include specific quantifiable results from candidate's documented achievements
3. Focus on personal actions and contributions from provided context
4. Demonstrate the competency being assessed using documented experience
5. Keep response to 1.5-2 minutes when spoken
6. End with learning/growth reflection from documented experience
7. Connect to target role requirements using actual candidate background

**COMPETENCY FOCUS:** {target_competency}

Generate a response that would impress hiring managers using ONLY the candidate's documented experience.""",
                description="STAR method behavioral interview responses with competency focus and strict context adherence",
                context_variables=[
                    "interview_question",
                    "candidate_context",
                    "target_competency",
                    "position_requirements",
                    "company_values",
                ],
            ),
            PromptType.ACHIEVEMENT_QUANTIFIER: PromptTemplate(
                system_prompt="""You are an expert at transforming vague accomplishments into compelling, quantified achievements.

QUANTIFICATION STRATEGIES:
1. **Financial Impact**: Revenue generated, costs saved, budget managed
2. **Efficiency Gains**: Time reduced, processes improved, productivity increased
3. **Scale Metrics**: Team size, project scope, customer base, geographic reach
4. **Quality Improvements**: Error reduction, satisfaction scores, compliance rates
5. **Growth Metrics**: Percentage increases, market share, user adoption
6. **Comparative Data**: Before/after scenarios, benchmarks exceeded

ACHIEVEMENT TRANSFORMATION FRAMEWORK:
- Convert responsibilities into results
- Add specific timeframes and deadlines met
- Include comparative context (industry standards, previous performance)
- Emphasize personal contribution vs. team effort
- Use strong action verbs (achieved, generated, reduced, improved, led)

QUANTIFICATION TECHNIQUES:
- Estimate when exact numbers aren't available (use ranges)
- Include frequency (daily, weekly, monthly)
- Show progression over time
- Compare to goals or expectations
- Include stakeholder impact""",
                user_prompt_template="""Transform these experiences into quantified, compelling achievements:

**ORIGINAL EXPERIENCES:**
{original_content}

**TARGET ROLE:** {target_position}

**REQUIREMENTS:**
1. Convert each responsibility into a quantified achievement
2. Add specific metrics, percentages, or dollar amounts
3. Include timeframes and deadlines
4. Use strong action verbs
5. Show progression and growth
6. Align with target role requirements

Provide 3 versions for each achievement:
- Version 1: Conservative quantification
- Version 2: Moderate quantification
- Version 3: Ambitious but truthful quantification""",
                description="Transform vague experiences into quantified, compelling achievements",
                context_variables=[
                    "original_content",
                    "target_position",
                    "industry_context",
                    "experience_level",
                ],
            ),
            PromptType.ATS_OPTIMIZER: PromptTemplate(
                system_prompt="""You are an ATS (Applicant Tracking System) optimization expert with deep knowledge of:
- How ATS systems parse and rank resumes/cover letters
- Keyword matching algorithms and scoring
- Industry-specific terminology and phrases
- Modern ATS best practices and common pitfalls

ATS OPTIMIZATION PRINCIPLES:
1. **Keyword Integration**: Natural incorporation of exact job description terms
2. **Semantic Matching**: Use synonyms and related terms for broader coverage
3. **Context Relevance**: Keywords must appear in meaningful context
4. **Density Balance**: 2-3% keyword density (avoid stuffing)
5. **Standard Formatting**: Clean, simple structure ATS can parse
6. **Section Headers**: Use conventional headings ATS recognizes
7. **File Format**: Ensure compatibility with common ATS systems

KEYWORD STRATEGIES:
- Include exact job title and variations
- Use both abbreviations and full terms (e.g., "AI" and "Artificial Intelligence")
- Include industry-specific software, tools, and methodologies
- Incorporate soft skills mentioned in job posting
- Use action verbs from job description
- Include relevant certifications and qualifications""",
                user_prompt_template="""Optimize this content for ATS systems:

**JOB DESCRIPTION:**
{job_description}

**ORIGINAL CONTENT:**
{original_content}

**OPTIMIZATION REQUIREMENTS:**
1. Identify top 15 keywords from job description
2. Naturally integrate 10-12 keywords into content
3. Maintain readability and professional tone
4. Include both exact terms and semantic variations
5. Ensure 2-3% keyword density
6. Preserve original meaning and achievements

**OUTPUT FORMAT:**
1. List of identified keywords with importance ranking
2. Optimized content with keywords naturally integrated
3. Keyword density analysis
4. ATS compatibility score and recommendations""",
                description="ATS optimization for maximum keyword matching and parsing success",
                context_variables=[
                    "job_description",
                    "original_content",
                    "target_role",
                    "industry_type",
                ],
            ),
            PromptType.INTERVIEW_ANSWER: PromptTemplate(
                system_prompt="""You are an experienced interview coach helping candidates prepare compelling responses.

INTERVIEW RESPONSE BEST PRACTICES:
1. **Direct Answering**: Address the question immediately and clearly
2. **Specific Examples**: Use concrete situations, not hypotheticals
3. **Value Demonstration**: Show how you add value to organizations
4. **Cultural Fit**: Align responses with company values and culture
5. **Enthusiasm**: Show genuine interest in role and company
6. **Conciseness**: Keep responses 1-3 minutes, well-structured
7. **Future Focus**: Connect past experience to future contribution

RESPONSE STRUCTURE OPTIONS:
- **STAR Method**: For behavioral questions (Situation, Task, Action, Result)
- **Problem-Solution**: For technical/analytical questions
- **Past-Present-Future**: For experience and career questions
- **Feature-Benefit**: For skills and strengths questions

COMMON QUESTION TYPES:
- "Tell me about yourself" (2-minute professional summary)
- "Why do you want this role?" (motivation and fit)
- "Greatest strength/weakness" (self-awareness and growth)
- "Where do you see yourself in 5 years?" (career vision)
- "Why should we hire you?" (value proposition)
- "Questions for us?" (engagement and research)""",
                user_prompt_template="""Create a compelling interview response for this question:

**QUESTION:** {question}

**CANDIDATE BACKGROUND:**
{candidate_context}

**COMPANY/ROLE CONTEXT:**
{company_context}

**REQUIREMENTS:**
1. Answer the question directly and completely
2. Use specific examples from candidate's background
3. Demonstrate value and relevant skills
4. Show cultural fit and enthusiasm
5. Keep response 1-2 minutes when spoken
6. End with forward-looking statement
7. Avoid generic or cliché responses

**TONE:** Professional, confident, authentic, and engaging""",
                description="Comprehensive interview question responses with strategic positioning",
                context_variables=[
                    "question",
                    "candidate_context",
                    "company_context",
                    "position_requirements",
                    "interview_type",
                ],
            ),
            PromptType.CONTENT_REFINEMENT: PromptTemplate(
                system_prompt="""You are a professional writing editor specializing in job application materials.

CONTENT REFINEMENT PRINCIPLES:
1. **Clarity Enhancement**: Remove ambiguity and improve readability
2. **Impact Amplification**: Strengthen key messages and achievements
3. **Conciseness**: Eliminate redundancy while preserving meaning
4. **Professional Tone**: Maintain appropriate formality and confidence
5. **Action-Oriented Language**: Use strong verbs and active voice
6. **Quantification**: Add metrics and specific details where possible
7. **Flow Optimization**: Improve logical progression and transitions

EDITING TECHNIQUES:
- Replace weak verbs with powerful action words
- Convert passive voice to active voice
- Add specific details and context
- Remove filler words and redundant phrases
- Strengthen opening and closing statements
- Ensure parallel structure in lists
- Improve sentence variety and rhythm

PRESERVATION PRIORITIES:
- Maintain original voice and personality
- Keep factual accuracy intact
- Preserve core messages and intent
- Respect word count limitations
- Honor specific formatting requirements""",
                user_prompt_template="""Refine this content to maximize impact and professionalism:

**ORIGINAL CONTENT:**
{original_content}

**REFINEMENT GOALS:**
{refinement_goals}

**CONSTRAINTS:**
- Target length: {target_length}
- Tone: {desired_tone}
- Audience: {target_audience}

**REQUIREMENTS:**
1. Enhance clarity and impact
2. Strengthen action-oriented language
3. Add quantification where appropriate
4. Improve flow and readability
5. Maintain original voice and intent
6. Ensure professional tone throughout

**OUTPUT:**
1. Refined content
2. Summary of key improvements made
3. Suggestions for further enhancement""",
                description="Professional content refinement with impact optimization",
                context_variables=[
                    "original_content",
                    "refinement_goals",
                    "target_length",
                    "desired_tone",
                    "target_audience",
                ],
            ),
            PromptType.GENERAL_RESPONSE: PromptTemplate(
                system_prompt="""You are a knowledgeable career advisor with expertise in modern job search strategies.

🔒 CRITICAL FAITHFULNESS REQUIREMENTS (when documents are available):
- ONLY use information explicitly provided in the user's uploaded documents
- If information is not in the documents, clearly state "I don't have that information in your documents"
- DO NOT make assumptions about the user's experience, skills, or background
- DO NOT provide generic advice when specific document context is available
- When referencing information, be specific about which document it comes from
- If context is insufficient to answer fully, ask for additional documents

CAREER GUIDANCE EXPERTISE:
- Current job market trends and industry insights
- Resume and cover letter best practices
- Interview preparation and salary negotiation
- Professional networking and personal branding
- Career transition and skill development
- Remote work and digital presence optimization

RESPONSE APPROACH:
1. **Actionable Advice**: Provide specific, implementable recommendations based on available context
2. **Current Best Practices**: Reference up-to-date industry standards
3. **Personalized Guidance**: Tailor advice to user's specific situation from provided documents
4. **Resource Recommendations**: Suggest tools, platforms, and next steps
5. **Realistic Expectations**: Set appropriate timelines and expectations
6. **Follow-up Questions**: Ask clarifying questions when context is insufficient

COMMUNICATION STYLE:
- Professional yet approachable
- Encouraging and supportive
- Direct and honest when information is missing
- Evidence-based recommendations from provided context
- Structured and organized responses""",
                user_prompt_template="{user_query}",
                description="Comprehensive career advice with current best practices and strict context adherence when documents are available",
                context_variables=[
                    "user_query",
                    "career_level",
                    "industry",
                    "location",
                    "goals",
                ],
            ),
            PromptType.GENERAL_CHAT: PromptTemplate(
                system_prompt="""You are a knowledgeable career advisor with expertise in modern job search strategies.

🔒 CRITICAL FAITHFULNESS REQUIREMENTS (when documents are available):
- ONLY use information explicitly provided in the user's uploaded documents
- If information is not in the documents, clearly state "I don't have that information in your documents"
- DO NOT make assumptions about the user's experience, skills, or background
- DO NOT provide generic advice when specific document context is available
- When referencing information, be specific about which document it comes from
- If context is insufficient to answer fully, ask for additional documents

CAREER GUIDANCE EXPERTISE:
- Current job market trends and industry insights
- Resume and cover letter best practices
- Interview preparation and salary negotiation
- Professional networking and personal branding
- Career transition and skill development
- Remote work and digital presence optimization

RESPONSE APPROACH:
1. **Actionable Advice**: Provide specific, implementable recommendations based on available context
2. **Current Best Practices**: Reference up-to-date industry standards
3. **Personalized Guidance**: Tailor advice to user's specific situation from provided documents
4. **Resource Recommendations**: Suggest tools, platforms, and next steps
5. **Realistic Expectations**: Set appropriate timelines and expectations
6. **Follow-up Questions**: Ask clarifying questions when context is insufficient

COMMUNICATION STYLE:
- Professional yet approachable
- Encouraging and supportive
- Direct and honest when information is missing
- Evidence-based recommendations from provided context
- Structured and organized responses""",
                user_prompt_template="{user_query}",
                description="General chat responses with career guidance and strict context adherence when documents are available",
                context_variables=[
                    "user_query",
                    "career_level",
                    "industry",
                    "location",
                    "goals",
                ],
            ),
            PromptType.CHAT_ASSISTANT: PromptTemplate(
                system_prompt="""You are an AI career coach assistant providing ongoing support for job seekers.

CONVERSATION APPROACH:
1. **Active Listening**: Reference previous conversation context
2. **Personalized Support**: Remember user's goals, challenges, and progress
3. **Encouraging Tone**: Maintain positivity while being realistic
4. **Skill Building**: Help users develop interview, writing, and networking skills
5. **Progress Tracking**: Acknowledge improvements and celebrate wins
6. **Resource Sharing**: Recommend relevant tools, articles, and strategies

SUPPORT AREAS:
- Job search strategy and planning
- Application materials review and improvement
- Interview preparation and practice
- Networking guidance and scripts
- Career development and skill building
- Confidence building and motivation

INTERACTION STYLE:
- Conversational and supportive
- Ask follow-up questions to understand needs
- Provide specific, actionable advice
- Offer encouragement during challenging times
- Celebrate progress and achievements""",
                user_prompt_template="{message}",
                description="Interactive career coaching with personalized support",
                context_variables=[
                    "message",
                    "conversation_history",
                    "user_profile",
                    "current_goals",
                ],
            ),
        }

    def _load_custom_prompts(self) -> Dict[PromptType, PromptTemplate]:
        """Load custom user-defined prompts from file."""
        if not self.custom_prompts_path.exists():
            return {}

        try:
            with open(self.custom_prompts_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            custom_prompts = {}
            for prompt_type_str, prompt_data in data.items():
                try:
                    prompt_type = PromptType(prompt_type_str)
                    custom_prompts[prompt_type] = PromptTemplate(**prompt_data)
                except (ValueError, TypeError) as e:
                    self.logger.warning(
                        f"Skipping invalid custom prompt {prompt_type_str}: {e}"
                    )

            self.logger.info(f"Loaded {len(custom_prompts)} custom prompts")
            return custom_prompts

        except Exception as e:
            self.logger.error(f"Failed to load custom prompts: {e}")
            return {}

    def get_prompt_template(self, prompt_type: PromptType) -> PromptTemplate:
        """
        Get prompt template for a given type.

        Args:
            prompt_type: Type of prompt to retrieve

        Returns:
            PromptTemplate with system and user prompts
        """
        # Check custom prompts first (user overrides)
        if prompt_type in self._custom_prompts:
            return self._custom_prompts[prompt_type]

        # Fall back to default prompts
        if prompt_type in self._default_prompts:
            return self._default_prompts[prompt_type]

        raise ValueError(f"No prompt template found for type: {prompt_type}")

    def build_system_prompt(
        self, prompt_type: PromptType, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build dynamic system prompt based on context.

        Args:
            prompt_type: Type of prompt
            context: Context variables for dynamic content

        Returns:
            Formatted system prompt
        """
        template = self.get_prompt_template(prompt_type)
        system_prompt = template.system_prompt

        # Add dynamic context if provided
        if context:
            # Add industry-specific guidance
            if context.get("industry"):
                industry_guidance = self._get_industry_guidance(context["industry"])
                if industry_guidance:
                    system_prompt += (
                        f"\n\nIndustry-specific guidance: {industry_guidance}"
                    )

            # Add experience level guidance
            if context.get("experience_level"):
                level_guidance = self._get_experience_level_guidance(
                    context["experience_level"]
                )
                if level_guidance:
                    system_prompt += (
                        f"\n\nExperience level considerations: {level_guidance}"
                    )

            # Add company size guidance
            if context.get("company_size"):
                size_guidance = self._get_company_size_guidance(context["company_size"])
                if size_guidance:
                    system_prompt += f"\n\nCompany size considerations: {size_guidance}"

        return system_prompt

    def build_user_prompt(
        self, prompt_type: PromptType, variables: Dict[str, Any]
    ) -> str:
        """
        Build user prompt with variable substitution.

        Args:
            prompt_type: Type of prompt
            variables: Variables to substitute in template

        Returns:
            Formatted user prompt
        """
        template = self.get_prompt_template(prompt_type)

        try:
            return template.user_prompt_template.format(**variables)
        except KeyError as e:
            self.logger.warning(f"Missing variable {e} for prompt type {prompt_type}")
            # Return template with missing variables as placeholders
            return template.user_prompt_template

    def build_prompt(
        self,
        prompt_type: PromptType,
        user_query: str,
        context: str = "",
        conversation_history: Optional[List] = None,
        **kwargs
    ) -> str:
        """
        Build a complete prompt combining system and user prompts.
        
        This method provides backward compatibility with the old chat controller.
        
        Args:
            prompt_type: Type of prompt to build
            user_query: User's query/message
            context: Document context
            conversation_history: Previous conversation messages
            **kwargs: Additional variables for prompt formatting
            
        Returns:
            Complete formatted prompt
        """
        # Get system prompt
        system_prompt = self.build_system_prompt(prompt_type)
        
        # Prepare variables for user prompt
        variables = {
            "user_query": user_query,
            "context": context,
            **kwargs
        }
        
        # Build user prompt
        user_prompt = self.build_user_prompt(prompt_type, variables)
        
        # Combine system and user prompts
        if context:
            full_prompt = f"""{system_prompt}

DOCUMENT CONTEXT:
{context}

USER REQUEST: {user_query}

Please provide a helpful response based on the document context above. If the context doesn't contain relevant information, please say so and provide general guidance.

Response:"""
        else:
            full_prompt = f"""{system_prompt}

USER REQUEST: {user_query}

IMPORTANT: You don't have access to any specific documents about this user's background, experience, or job details. Please provide general guidance and suggest that the user upload relevant documents (CV, job descriptions, etc.) for more personalized assistance.

Response:"""
        
        return full_prompt

    def _get_industry_guidance(self, industry: str) -> Optional[str]:
        """Get industry-specific guidance."""
        industry_guides = {
            "tech": (
                "Focus on technical skills, innovation, and problem-solving. "
                "Mention specific technologies and methodologies."
            ),
            "finance": (
                "Emphasize analytical skills, attention to detail, and "
                "regulatory knowledge. Highlight quantitative achievements."
            ),
            "healthcare": (
                "Stress patient care, compliance, and teamwork. "
                "Mention relevant certifications and experience."
            ),
            "education": (
                "Highlight teaching philosophy, student outcomes, and "
                "continuous learning. Show passion for education."
            ),
            "retail": (
                "Focus on customer service, sales achievements, and team "
                "collaboration. Mention specific metrics."
            ),
            "consulting": (
                "Emphasize problem-solving, client management, and strategic "
                "thinking. Show diverse project experience."
            ),
            "nonprofit": (
                "Highlight mission alignment, community impact, and resource "
                "efficiency. Show passion for the cause."
            ),
        }
        return industry_guides.get(industry.lower())

    def _get_experience_level_guidance(self, level: str) -> Optional[str]:
        """Get experience level-specific guidance."""
        level_guides = {
            "entry": (
                "Emphasize education, academic achievements, and relevant coursework, as work experience may be limited. "
                "Highlight transferable skills, internships, and enthusiasm for learning. "
                "If the candidate has little work experience, make the diploma or degree a central point in the narrative."
            ),
            "mid": (
                "Balance technical skills with leadership potential. "
                "Highlight career progression and key achievements. "
                "Education can be mentioned, but focus on professional experience and results."
            ),
            "senior": (
                "Emphasize leadership, strategic thinking, and mentoring. "
                "Show impact on business outcomes. "
                "Education is less critical unless highly relevant; prioritize work experience and accomplishments."
            ),
            "executive": (
                "Focus on vision, transformation, and organizational impact. "
                "Highlight board experience and industry recognition. "
                "Education is secondary unless it is a major differentiator; the main emphasis should be on leadership and results."
            ),
        }
        return level_guides.get(level.lower())

    def _get_company_size_guidance(self, size: str) -> Optional[str]:
        """Get company size-specific guidance."""
        size_guides = {
            "startup": (
                "Emphasize adaptability, wearing multiple hats, and comfort "
                "with ambiguity. Show entrepreneurial spirit."
            ),
            "small": (
                "Highlight direct impact, close collaboration, and versatility. "
                "Show ability to work in tight-knit teams."
            ),
            "medium": (
                "Balance specialization with cross-functional collaboration. "
                "Show growth potential and scalability mindset."
            ),
            "large": (
                "Emphasize process orientation, scale experience, and ability "
                "to work in complex organizations."
            ),
        }
        return size_guides.get(size.lower())

    def save_custom_prompt(
        self, prompt_type: PromptType, template: PromptTemplate
    ) -> bool:
        """
        Save a custom prompt template.

        Args:
            prompt_type: Type of prompt
            template: Prompt template to save

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load existing custom prompts
            if self.custom_prompts_path.exists():
                with open(self.custom_prompts_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {}

            # Add/update the prompt
            data[prompt_type.value] = {
                "system_prompt": template.system_prompt,
                "user_prompt_template": template.user_prompt_template,
                "description": template.description,
                "version": template.version,
                "tags": template.tags,
                "context_variables": template.context_variables,
            }

            # Ensure directory exists
            self.custom_prompts_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to file
            with open(self.custom_prompts_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Update in-memory cache
            self._custom_prompts[prompt_type] = template

            self.logger.info(f"Saved custom prompt for {prompt_type.value}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save custom prompt: {e}")
            return False

    def list_available_prompts(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available prompts with metadata.

        Returns:
            Dictionary with prompt information
        """
        result = {}

        for prompt_type in PromptType:
            template = self.get_prompt_template(prompt_type)
            is_custom = prompt_type in self._custom_prompts

            result[prompt_type.value] = {
                "description": template.description,
                "version": template.version,
                "tags": template.tags,
                "context_variables": template.context_variables,
                "is_custom": is_custom,
                "source": "custom" if is_custom else "default",
            }

        return result

    def get_behavioral_interview_prompt(
        self,
        question: str,
        candidate_context: str,
        target_competency: str = "general",
        position_requirements: str = "",
        company_values: str = "",
    ) -> tuple[str, str]:
        """
        Get specialized behavioral interview prompt using STAR method.

        Args:
            question: The behavioral interview question
            candidate_context: Background information about the candidate
            target_competency: The competency being assessed
            position_requirements: Requirements for the target position
            company_values: Company values and culture information

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        variables = {
            "interview_question": question,
            "candidate_context": candidate_context,
            "target_competency": target_competency,
            "position_requirements": position_requirements,
            "company_values": company_values,
        }

        system_prompt = self.build_system_prompt(PromptType.BEHAVIORAL_INTERVIEW)
        user_prompt = self.build_user_prompt(PromptType.BEHAVIORAL_INTERVIEW, variables)

        return system_prompt, user_prompt

    def get_achievement_quantifier_prompt(
        self,
        original_content: str,
        target_position: str,
        industry_context: str = "",
        experience_level: str = "mid-level",
    ) -> tuple[str, str]:
        """
        Get prompt for transforming experiences into quantified achievements.

        Args:
            original_content: Original experience descriptions
            target_position: Target job title/position
            industry_context: Industry-specific context
            experience_level: Candidate's experience level

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        variables = {
            "original_content": original_content,
            "target_position": target_position,
            "industry_context": industry_context,
            "experience_level": experience_level,
        }

        system_prompt = self.build_system_prompt(PromptType.ACHIEVEMENT_QUANTIFIER)
        user_prompt = self.build_user_prompt(
            PromptType.ACHIEVEMENT_QUANTIFIER, variables
        )

        return system_prompt, user_prompt

    def get_ats_optimizer_prompt(
        self,
        job_description: str,
        original_content: str,
        target_role: str = "",
        industry_type: str = "",
    ) -> tuple[str, str]:
        """
        Get prompt for ATS optimization of content.

        Args:
            job_description: Full job description text
            original_content: Content to optimize
            target_role: Target job title
            industry_type: Industry context

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        variables = {
            "job_description": job_description,
            "original_content": original_content,
            "target_role": target_role,
            "industry_type": industry_type,
        }

        system_prompt = self.build_system_prompt(PromptType.ATS_OPTIMIZER)
        user_prompt = self.build_user_prompt(PromptType.ATS_OPTIMIZER, variables)

        return system_prompt, user_prompt

    def get_enhanced_cover_letter_prompt(
        self,
        company_name: str,
        position_title: str,
        job_description: str,
        candidate_context: str,
        company_culture: str = "",
        industry_type: str = "",
    ) -> tuple[str, str]:
        """
        Get enhanced cover letter prompt with research-backed best practices.

        Args:
            company_name: Name of the company
            position_title: Job title/position
            job_description: Full job description
            candidate_context: Candidate background and experience
            company_culture: Company culture information
            industry_type: Industry context

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        variables = {
            "company_name": company_name,
            "position_title": position_title,
            "job_description": job_description,
            "candidate_context": candidate_context,
            "company_culture": company_culture,
            "industry_type": industry_type,
        }

        system_prompt = self.build_system_prompt(
            PromptType.COVER_LETTER, {"industry": industry_type}
        )
        user_prompt = self.build_user_prompt(PromptType.COVER_LETTER, variables)

        return system_prompt, user_prompt

    def get_enhanced_content_refinement_prompt(
        self,
        original_content: str,
        refinement_goals: str,
        target_length: str = "maintain current length",
        desired_tone: str = "professional",
        target_audience: str = "hiring managers",
    ) -> tuple[str, str]:
        """
        Get enhanced content refinement prompt.

        Args:
            original_content: Content to refine
            refinement_goals: Specific goals for refinement
            target_length: Desired length constraints
            desired_tone: Target tone
            target_audience: Intended audience

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        variables = {
            "original_content": original_content,
            "refinement_goals": refinement_goals,
            "target_length": target_length,
            "desired_tone": desired_tone,
            "target_audience": target_audience,
        }

        system_prompt = self.build_system_prompt(PromptType.CONTENT_REFINEMENT)
        user_prompt = self.build_user_prompt(PromptType.CONTENT_REFINEMENT, variables)

        return system_prompt, user_prompt

    def get_enhanced_interview_answer_prompt(
        self,
        question: str,
        candidate_context: str,
        company_context: str = "",
        position_requirements: str = "",
        interview_type: str = "general",
    ) -> tuple[str, str]:
        """
        Get enhanced interview answer prompt with strategic positioning.

        Args:
            question: Interview question
            candidate_context: Candidate background
            company_context: Company and role context
            position_requirements: Job requirements
            interview_type: Type of interview (behavioral, technical, etc.)

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        variables = {
            "question": question,
            "candidate_context": candidate_context,
            "company_context": company_context,
            "position_requirements": position_requirements,
            "interview_type": interview_type,
        }

        system_prompt = self.build_system_prompt(PromptType.INTERVIEW_ANSWER)
        user_prompt = self.build_user_prompt(PromptType.INTERVIEW_ANSWER, variables)

        return system_prompt, user_prompt


# Global prompt manager instance
_prompt_manager = None


def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager
