"""Re-ranking service for improving search results."""

from typing import List, Optional
import logging
from datetime import datetime

from src.domain.models import SearchResult

logger = logging.getLogger(__name__)


class RankingService:
    """Service for re-ranking search results."""

    def __init__(self, diversity_weight: float = 0.2, recency_weight: float = 0.1):
        self.diversity_weight = diversity_weight
        self.recency_weight = recency_weight

    def rerank(
        self,
        results: List[SearchResult],
        query: str,
        promote_recent: bool = False,
        diversify: bool = True
    ) -> List[SearchResult]:
        """
        Re-rank search results based on multiple factors.

        Args:
            results: Initial search results
            query: Original query
            promote_recent: Whether to boost recent entries
            diversify: Whether to diversify results

        Returns:
            Re-ranked search results
        """
        if not results:
            return results

        for result in results:
            result.relevance_score = self._compute_final_score(
                result,
                query,
                promote_recent
            )

        if diversify and len(results) > 3:
            results = self._diversify_results(results)

        results.sort(key=lambda x: x.relevance_score, reverse=True)

        return results

    def _compute_final_score(
        self,
        result: SearchResult,
        query: str,
        promote_recent: bool
    ) -> float:
        """Compute final ranking score."""
        base_score = result.combined_score

        recency_score = 0.0
        if promote_recent:
            days_old = (datetime.now() - result.entry.date).days
            if days_old < 7:
                recency_score = 0.3
            elif days_old < 30:
                recency_score = 0.2
            elif days_old < 90:
                recency_score = 0.1

        query_terms = set(query.lower().split())
        title_boost = 0.0
        if result.entry.title:
            title_terms = set(result.entry.title.lower().split())
            overlap = len(query_terms.intersection(title_terms))
            title_boost = min(overlap * 0.15, 0.3)

        length_penalty = 0.0
        word_count = result.entry.word_count
        if word_count < 50:
            length_penalty = -0.1
        elif word_count > 1000:
            length_penalty = -0.05

        final_score = (
            base_score * 0.7 +
            recency_score * self.recency_weight +
            title_boost +
            length_penalty
        )

        return min(max(final_score, 0.0), 1.0)

    def _diversify_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Diversify results to avoid redundancy."""
        diversified = []
        seen_dates = set()
        seen_files = set()

        for result in results:
            date_key = result.entry.date.date()
            file_key = result.entry.file_path

            date_penalty = 0.1 if date_key in seen_dates else 0
            file_penalty = 0.15 if file_key in seen_files else 0

            result.relevance_score -= (date_penalty + file_penalty) * self.diversity_weight

            seen_dates.add(date_key)
            seen_files.add(file_key)
            diversified.append(result)

        return diversified

    def group_by_time_period(
        self,
        results: List[SearchResult],
        period: str = "month"
    ) -> dict:
        """Group results by time period."""
        grouped = {}

        for result in results:
            if period == "day":
                key = result.entry.date.date()
            elif period == "week":
                key = result.entry.date.isocalendar()[1]
            elif period == "month":
                key = result.entry.date.strftime("%Y-%m")
            elif period == "year":
                key = result.entry.date.year
            else:
                key = "all"

            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)

        return grouped

    def filter_by_quality(
        self,
        results: List[SearchResult],
        min_words: int = 20,
        min_score: float = 0.3
    ) -> List[SearchResult]:
        """Filter results by quality metrics."""
        filtered = []

        for result in results:
            if result.entry.word_count < min_words:
                continue

            if result.combined_score < min_score:
                continue

            filtered.append(result)

        return filtered