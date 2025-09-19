"""Context building for language model generation."""

from typing import List, Optional
import logging
from datetime import datetime

from src.domain.models import SearchResult

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds context for language model from search results."""

    def __init__(
        self,
        max_tokens: int = 2048,
        include_metadata: bool = True,
        chronological: bool = False
    ):
        self.max_tokens = max_tokens
        self.include_metadata = include_metadata
        self.chronological = chronological

    def build_context(
        self,
        results: List[SearchResult],
        query: str,
        max_entries: Optional[int] = None
    ) -> str:
        """
        Build context string from search results.

        Args:
            results: Search results to include
            query: Original query for context
            max_entries: Maximum number of entries to include

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        if self.chronological:
            results = sorted(results, key=lambda x: x.entry.date)

        context_parts = []
        token_count = 0
        entries_included = 0

        for result in results:
            if max_entries and entries_included >= max_entries:
                break

            entry_text = self._format_entry(result)
            entry_tokens = len(entry_text.split())

            if token_count + entry_tokens > self.max_tokens:
                if entries_included == 0:
                    truncated = self._truncate_entry(entry_text, self.max_tokens)
                    context_parts.append(truncated)
                    entries_included += 1
                break

            context_parts.append(entry_text)
            token_count += entry_tokens
            entries_included += 1

        if not context_parts:
            return ""

        context = "\n\n---\n\n".join(context_parts)

        header = f"Context from {entries_included} journal entries:\n\n"
        return header + context

    def build_structured_context(
        self,
        results: List[SearchResult],
        query: str
    ) -> dict:
        """
        Build structured context with metadata.

        Returns:
            Dictionary with context and metadata
        """
        if not results:
            return {
                "context": "",
                "metadata": {
                    "entries_count": 0,
                    "date_range": None,
                    "total_words": 0
                }
            }

        context = self.build_context(results, query)

        dates = [r.entry.date for r in results]
        total_words = sum(r.entry.word_count for r in results[:10])

        metadata = {
            "entries_count": len(results),
            "date_range": {
                "earliest": min(dates).isoformat(),
                "latest": max(dates).isoformat()
            },
            "total_words": total_words,
            "entry_types": list(set(r.entry.entry_type.value for r in results)),
            "average_score": sum(r.score for r in results) / len(results)
        }

        return {
            "context": context,
            "metadata": metadata
        }

    def _format_entry(self, result: SearchResult) -> str:
        """Format a single entry for context."""
        parts = []

        if self.include_metadata:
            date_str = result.entry.date.strftime("%B %d, %Y")
            parts.append(f"Date: {date_str}")

            if result.entry.title:
                parts.append(f"Title: {result.entry.title}")

            if result.entry.tags:
                parts.append(f"Tags: {', '.join(result.entry.tags)}")

            parts.append("")

        parts.append(result.entry.content)

        if result.entry.is_chunked:
            parts.append(
                f"\n[Part {result.entry.chunk_index + 1} of {result.entry.total_chunks}]"
            )

        return "\n".join(parts)

    def _truncate_entry(self, entry_text: str, max_tokens: int) -> str:
        """Truncate entry text to fit within token limit."""
        words = entry_text.split()
        truncated_words = words[:max_tokens - 10]
        return " ".join(truncated_words) + " [truncated]"

    def build_temporal_context(
        self,
        results: List[SearchResult],
        query: str,
        days_before: int = 3,
        days_after: int = 3
    ) -> str:
        """Build context including temporal neighbors."""
        if not results:
            return ""

        primary_dates = [r.entry.date for r in results[:3]]
        avg_date = sum(
            (d.timestamp() for d in primary_dates),
            0
        ) / len(primary_dates)
        center_date = datetime.fromtimestamp(avg_date)

        temporal_results = []
        for result in results:
            days_diff = abs((result.entry.date - center_date).days)
            if days_diff <= days_before or days_diff <= days_after:
                temporal_results.append(result)

        return self.build_context(temporal_results, query)

    def build_summary_context(
        self,
        results: List[SearchResult],
        query: str,
        include_excerpts: bool = True
    ) -> str:
        """Build a summary-focused context."""
        if not results:
            return ""

        summary_parts = [
            f"Query: {query}",
            f"Found {len(results)} relevant entries",
            ""
        ]

        date_range = self._get_date_range(results)
        summary_parts.append(f"Date range: {date_range}")

        topics = self._extract_topics(results)
        if topics:
            summary_parts.append(f"Common topics: {', '.join(topics[:5])}")

        summary_parts.append("")

        if include_excerpts:
            summary_parts.append("Key excerpts:")
            for i, result in enumerate(results[:5], 1):
                excerpt = self._get_excerpt(result.entry.content, query)
                date_str = result.entry.date.strftime("%Y-%m-%d")
                summary_parts.append(f"{i}. [{date_str}] {excerpt}")

        return "\n".join(summary_parts)

    def _get_date_range(self, results: List[SearchResult]) -> str:
        """Get formatted date range from results."""
        dates = [r.entry.date for r in results]
        min_date = min(dates).strftime("%B %d, %Y")
        max_date = max(dates).strftime("%B %d, %Y")

        if min_date == max_date:
            return min_date
        return f"{min_date} to {max_date}"

    def _extract_topics(self, results: List[SearchResult]) -> List[str]:
        """Extract common topics from results."""
        word_freq = {}

        for result in results:
            words = result.entry.content.lower().split()
            for word in words:
                if len(word) > 5:
                    word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:10]]

    def _get_excerpt(self, content: str, query: str, max_length: int = 100) -> str:
        """Extract relevant excerpt from content."""
        query_words = query.lower().split()
        sentences = content.split('.')

        for sentence in sentences:
            if any(word in sentence.lower() for word in query_words):
                excerpt = sentence.strip()
                if len(excerpt) > max_length:
                    excerpt = excerpt[:max_length] + "..."
                return excerpt

        first_sentence = sentences[0].strip() if sentences else ""
        if len(first_sentence) > max_length:
            first_sentence = first_sentence[:max_length] + "..."
        return first_sentence