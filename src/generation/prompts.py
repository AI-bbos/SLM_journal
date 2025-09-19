"""Prompt templates for journal queries."""

from typing import Optional, Dict, Any
from datetime import datetime


class PromptTemplate:
    """Templates for generating prompts."""

    @staticmethod
    def question_answering(query: str, context: str) -> str:
        """Template for question answering."""
        return f"""You are a helpful assistant analyzing personal journal entries.
Based on the journal entries provided below, answer the following question.
Be specific and reference relevant entries when applicable.

Question: {query}

Journal Entries:
{context}

Answer:"""

    @staticmethod
    def summarization(context: str, focus: Optional[str] = None) -> str:
        """Template for summarizing journal entries."""
        prompt = """Please summarize the following journal entries, highlighting key themes,
emotions, and important events.

Journal Entries:
{context}"""

        if focus:
            prompt += f"\n\nFocus particularly on: {focus}"

        prompt += "\n\nSummary:"
        return prompt.format(context=context)

    @staticmethod
    def reflection(query: str, context: str) -> str:
        """Template for reflective analysis."""
        return f"""Analyze the following journal entries to provide insights about: {query}

Look for patterns, changes over time, recurring themes, and emotional evolution.

Journal Entries:
{context}

Reflective Analysis:"""

    @staticmethod
    def temporal_analysis(time_period: str, context: str) -> str:
        """Template for analyzing entries over time."""
        return f"""Analyze how thoughts and experiences have evolved during {time_period}
based on these journal entries. Identify key changes, patterns, and developments.

Journal Entries:
{context}

Temporal Analysis:"""

    @staticmethod
    def emotion_analysis(context: str) -> str:
        """Template for emotional analysis."""
        return f"""Analyze the emotional content of these journal entries.
Identify dominant emotions, emotional patterns, and triggers.

Journal Entries:
{context}

Emotional Analysis:"""

    @staticmethod
    def topic_extraction(context: str, num_topics: int = 5) -> str:
        """Template for extracting main topics."""
        return f"""Identify the {num_topics} main topics or themes discussed in these journal entries.
For each topic, provide a brief description.

Journal Entries:
{context}

Main Topics:"""

    @staticmethod
    def comparison(period1: str, period2: str, context1: str, context2: str) -> str:
        """Template for comparing two time periods."""
        return f"""Compare journal entries from {period1} with those from {period2}.
Identify what has changed, what has remained consistent, and any notable developments.

Entries from {period1}:
{context1}

Entries from {period2}:
{context2}

Comparison:"""

    @staticmethod
    def goal_tracking(goal: str, context: str) -> str:
        """Template for tracking progress on goals."""
        return f"""Track progress on the following goal based on journal entries: {goal}

Look for mentions of progress, setbacks, learnings, and current status.

Journal Entries:
{context}

Goal Progress Analysis:"""

    @staticmethod
    def custom(template: str, **kwargs) -> str:
        """Use a custom template with variable substitution."""
        return template.format(**kwargs)


class PromptEnhancer:
    """Enhance prompts with additional context and instructions."""

    @staticmethod
    def add_format_instructions(prompt: str, format_type: str = "structured") -> str:
        """Add formatting instructions to prompt."""
        format_instructions = {
            "structured": "\n\nProvide a structured response with clear sections.",
            "bullet": "\n\nOrganize your response using bullet points.",
            "narrative": "\n\nProvide a flowing narrative response.",
            "analytical": "\n\nProvide an analytical response with evidence and reasoning."
        }

        instruction = format_instructions.get(format_type, "")
        return prompt + instruction

    @staticmethod
    def add_length_constraint(prompt: str, length: str = "medium") -> str:
        """Add length constraints to prompt."""
        length_constraints = {
            "brief": "\n\nKeep your response brief (2-3 sentences).",
            "short": "\n\nProvide a short response (1 paragraph).",
            "medium": "\n\nProvide a moderate length response (2-3 paragraphs).",
            "detailed": "\n\nProvide a detailed response with thorough analysis."
        }

        constraint = length_constraints.get(length, "")
        return prompt + constraint

    @staticmethod
    def add_tone(prompt: str, tone: str = "helpful") -> str:
        """Add tone instructions to prompt."""
        tone_instructions = {
            "helpful": "\n\nBe helpful and supportive in your response.",
            "analytical": "\n\nUse an analytical and objective tone.",
            "empathetic": "\n\nRespond with empathy and understanding.",
            "professional": "\n\nUse a professional and formal tone.",
            "casual": "\n\nUse a casual and conversational tone."
        }

        instruction = tone_instructions.get(tone, "")
        return prompt + instruction

    @staticmethod
    def add_date_context(prompt: str, query_date: Optional[datetime] = None) -> str:
        """Add current date context to prompt."""
        date = query_date or datetime.now()
        date_str = date.strftime("%B %d, %Y")
        return f"Today's date is {date_str}.\n\n{prompt}"