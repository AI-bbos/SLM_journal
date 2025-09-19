"""Semantic search implementation for journal entries."""

from typing import List, Optional, Tuple
import logging
from datetime import datetime, timedelta

from src.domain.models import JournalEntry, SearchResult
from src.embedding.embedder import EmbeddingService
from src.storage.vector_store import VectorStore
from src.storage.metadata_store import MetadataStore

logger = logging.getLogger(__name__)


class SemanticSearcher:
    """Performs semantic search on journal entries."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        metadata_store: MetadataStore
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.metadata_store = metadata_store

    def search(
        self,
        query: str,
        k: int = 10,
        date_filter: Optional[Tuple[datetime, datetime]] = None,
        similarity_threshold: float = 0.3
    ) -> List[SearchResult]:
        """
        Perform semantic search on journal entries.

        Args:
            query: The search query
            k: Number of results to return
            date_filter: Optional tuple of (start_date, end_date)
            similarity_threshold: Minimum similarity score

        Returns:
            List of search results
        """
        query_embedding = self.embedding_service.embed_query(query)

        similar_entries = self.vector_store.search(
            query_embedding,
            k=k * 3 if date_filter else k,
            threshold=similarity_threshold
        )

        if not similar_entries:
            logger.info(f"No similar entries found for query: {query}")
            return []

        entry_ids = [entry_id for entry_id, _ in similar_entries]
        entries = self.metadata_store.get_entries(entry_ids)

        entry_map = {entry.id: entry for entry in entries}

        results = []
        for entry_id, score in similar_entries:
            entry = entry_map.get(entry_id)
            if not entry:
                continue

            if date_filter:
                start_date, end_date = date_filter
                if not (start_date <= entry.date <= end_date):
                    continue

            search_result = SearchResult(
                entry=entry,
                score=score,
                relevance_score=self._calculate_relevance(query, entry)
            )
            results.append(search_result)

        results.sort(key=lambda x: x.combined_score, reverse=True)

        return results[:k]

    def search_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        query: Optional[str] = None,
        k: int = 20
    ) -> List[SearchResult]:
        """Search entries within a date range, optionally with semantic filtering."""
        if query:
            return self.search(query, k=k, date_filter=(start_date, end_date))

        entries = self.metadata_store.search_entries(
            start_date=start_date,
            end_date=end_date,
            limit=k
        )

        results = []
        for entry in entries:
            search_result = SearchResult(
                entry=entry,
                score=1.0,
                relevance_score=0.0
            )
            results.append(search_result)

        return results

    def search_similar_to_entry(
        self,
        entry_id: str,
        k: int = 10,
        exclude_same_file: bool = True
    ) -> List[SearchResult]:
        """Find entries similar to a given entry."""
        entry = self.metadata_store.get_entry(entry_id)
        if not entry:
            logger.warning(f"Entry {entry_id} not found")
            return []

        query = f"{entry.title or ''} {entry.content[:200]}"
        results = self.search(query, k=k * 2)

        filtered_results = []
        for result in results:
            if result.entry.id == entry_id:
                continue

            if exclude_same_file and result.entry.file_path == entry.file_path:
                continue

            filtered_results.append(result)

        return filtered_results[:k]

    def search_by_keywords(
        self,
        keywords: List[str],
        k: int = 10,
        match_all: bool = False
    ) -> List[SearchResult]:
        """Search using keyword matching combined with semantic search."""
        if match_all:
            query = " AND ".join(keywords)
        else:
            query = " OR ".join(keywords)

        results = self.search(query, k=k * 2)

        keyword_filtered = []
        for result in results:
            content_lower = result.entry.content.lower()
            keywords_lower = [kw.lower() for kw in keywords]

            if match_all:
                if all(kw in content_lower for kw in keywords_lower):
                    keyword_filtered.append(result)
            else:
                if any(kw in content_lower for kw in keywords_lower):
                    keyword_filtered.append(result)

        for result in keyword_filtered:
            keyword_count = sum(
                1 for kw in keywords
                if kw.lower() in result.entry.content.lower()
            )
            result.relevance_score += keyword_count * 0.1

        keyword_filtered.sort(key=lambda x: x.combined_score, reverse=True)

        return keyword_filtered[:k]

    def _calculate_relevance(self, query: str, entry: JournalEntry) -> float:
        """Calculate additional relevance factors."""
        relevance = 0.0

        query_words = set(query.lower().split())
        content_words = set(entry.content.lower().split())
        overlap = len(query_words.intersection(content_words))
        relevance += overlap * 0.05

        days_old = (datetime.now() - entry.date).days
        if days_old < 30:
            relevance += 0.2
        elif days_old < 90:
            relevance += 0.1
        elif days_old < 365:
            relevance += 0.05

        if entry.title and any(word in entry.title.lower() for word in query_words):
            relevance += 0.3

        if entry.tags:
            tag_overlap = sum(
                1 for tag in entry.tags
                if any(word in tag.lower() for word in query_words)
            )
            relevance += tag_overlap * 0.15

        return min(relevance, 1.0)