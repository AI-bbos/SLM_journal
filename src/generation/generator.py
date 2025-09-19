"""Response generator that combines retrieval and generation."""

import time
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.domain.models import QueryResult, SearchResult
from src.generation.llm import LLMInterface
from src.generation.prompts import PromptTemplate, PromptEnhancer
from src.retrieval.context import ContextBuilder

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generates responses using retrieved context and language models."""

    def __init__(
        self,
        llm: LLMInterface,
        context_builder: Optional[ContextBuilder] = None,
        max_context_tokens: int = 2048
    ):
        self.llm = llm
        self.context_builder = context_builder or ContextBuilder(
            max_tokens=max_context_tokens
        )

    def generate_response(
        self,
        query: str,
        search_results: List[SearchResult],
        response_type: str = "question_answering",
        **kwargs
    ) -> QueryResult:
        """
        Generate a response to a query using search results.

        Args:
            query: The user's query
            search_results: Retrieved search results
            response_type: Type of response to generate
            **kwargs: Additional parameters for generation

        Returns:
            Complete query result with response and metadata
        """
        start_time = time.time()

        try:
            context = self.context_builder.build_context(
                search_results,
                query,
                max_entries=kwargs.get('max_entries', 10)
            )

            prompt = self._build_prompt(
                query,
                context,
                response_type,
                **kwargs
            )

            response = self.llm.generate(
                prompt,
                max_tokens=kwargs.get('max_tokens', 512),
                temperature=kwargs.get('temperature', 0.7)
            )

            processing_time = time.time() - start_time

            result = QueryResult(
                query=query,
                response=response,
                sources=search_results[:5],
                context_used=context,
                processing_time=processing_time,
                model_used=self.llm.get_model_info()['model_name'],
                metadata={
                    'response_type': response_type,
                    'context_entries': len(search_results),
                    'prompt_length': len(prompt),
                    **kwargs
                }
            )

            logger.info(f"Generated response in {processing_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            processing_time = time.time() - start_time

            return QueryResult(
                query=query,
                response=f"Error generating response: {e}",
                sources=[],
                context_used="",
                processing_time=processing_time,
                model_used=self.llm.get_model_info()['model_name'],
                metadata={'error': str(e)}
            )

    def _build_prompt(
        self,
        query: str,
        context: str,
        response_type: str,
        **kwargs
    ) -> str:
        """Build the prompt for the language model."""
        prompt_methods = {
            'question_answering': PromptTemplate.question_answering,
            'summarization': PromptTemplate.summarization,
            'reflection': PromptTemplate.reflection,
            'temporal_analysis': PromptTemplate.temporal_analysis,
            'emotion_analysis': PromptTemplate.emotion_analysis,
            'topic_extraction': PromptTemplate.topic_extraction,
            'goal_tracking': PromptTemplate.goal_tracking
        }

        method = prompt_methods.get(response_type, PromptTemplate.question_answering)

        if response_type in ['summarization', 'emotion_analysis', 'topic_extraction']:
            prompt = method(context, **kwargs)
        elif response_type == 'temporal_analysis':
            time_period = kwargs.get('time_period', 'the selected period')
            prompt = method(time_period, context)
        elif response_type == 'goal_tracking':
            goal = kwargs.get('goal', query)
            prompt = method(goal, context)
        else:
            prompt = method(query, context)

        prompt = self._enhance_prompt(prompt, **kwargs)

        return prompt

    def _enhance_prompt(self, prompt: str, **kwargs) -> str:
        """Enhance the prompt with additional instructions."""
        format_type = kwargs.get('format', 'structured')
        length = kwargs.get('length', 'medium')
        tone = kwargs.get('tone', 'helpful')

        prompt = PromptEnhancer.add_format_instructions(prompt, format_type)
        prompt = PromptEnhancer.add_length_constraint(prompt, length)
        prompt = PromptEnhancer.add_tone(prompt, tone)

        if kwargs.get('include_date_context', True):
            prompt = PromptEnhancer.add_date_context(prompt)

        return prompt

    def generate_summary(
        self,
        search_results: List[SearchResult],
        focus: Optional[str] = None,
        time_period: Optional[str] = None,
        **kwargs
    ) -> QueryResult:
        """Generate a summary of journal entries."""
        query = f"Summary of journal entries"
        if focus:
            query += f" focusing on {focus}"
        if time_period:
            query += f" from {time_period}"

        return self.generate_response(
            query,
            search_results,
            response_type='summarization',
            focus=focus,
            time_period=time_period,
            **kwargs
        )

    def analyze_emotions(
        self,
        search_results: List[SearchResult],
        **kwargs
    ) -> QueryResult:
        """Analyze emotions in journal entries."""
        return self.generate_response(
            "Analyze emotions in these journal entries",
            search_results,
            response_type='emotion_analysis',
            **kwargs
        )

    def track_goal_progress(
        self,
        goal: str,
        search_results: List[SearchResult],
        **kwargs
    ) -> QueryResult:
        """Track progress on a specific goal."""
        return self.generate_response(
            f"Track progress on goal: {goal}",
            search_results,
            response_type='goal_tracking',
            goal=goal,
            **kwargs
        )

    def extract_topics(
        self,
        search_results: List[SearchResult],
        num_topics: int = 5,
        **kwargs
    ) -> QueryResult:
        """Extract main topics from journal entries."""
        return self.generate_response(
            f"Extract {num_topics} main topics",
            search_results,
            response_type='topic_extraction',
            num_topics=num_topics,
            **kwargs
        )

    def generate_reflection(
        self,
        topic: str,
        search_results: List[SearchResult],
        **kwargs
    ) -> QueryResult:
        """Generate reflective analysis on a topic."""
        return self.generate_response(
            topic,
            search_results,
            response_type='reflection',
            **kwargs
        )

    def compare_time_periods(
        self,
        results1: List[SearchResult],
        results2: List[SearchResult],
        period1: str,
        period2: str,
        **kwargs
    ) -> QueryResult:
        """Compare two time periods."""
        context1 = self.context_builder.build_context(results1, f"entries from {period1}")
        context2 = self.context_builder.build_context(results2, f"entries from {period2}")

        prompt = PromptTemplate.comparison(period1, period2, context1, context2)
        prompt = self._enhance_prompt(prompt, **kwargs)

        start_time = time.time()

        try:
            response = self.llm.generate(
                prompt,
                max_tokens=kwargs.get('max_tokens', 512),
                temperature=kwargs.get('temperature', 0.7)
            )

            processing_time = time.time() - start_time

            return QueryResult(
                query=f"Compare {period1} with {period2}",
                response=response,
                sources=results1[:3] + results2[:3],
                context_used=f"Period 1: {context1}\n\nPeriod 2: {context2}",
                processing_time=processing_time,
                model_used=self.llm.get_model_info()['model_name'],
                metadata={
                    'comparison_periods': [period1, period2],
                    'entries_period1': len(results1),
                    'entries_period2': len(results2)
                }
            )

        except Exception as e:
            logger.error(f"Error in comparison: {e}")
            processing_time = time.time() - start_time

            return QueryResult(
                query=f"Compare {period1} with {period2}",
                response=f"Error generating comparison: {e}",
                sources=[],
                context_used="",
                processing_time=processing_time,
                model_used=self.llm.get_model_info()['model_name'],
                metadata={'error': str(e)}
            )