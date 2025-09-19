"""Domain models for journal entries and query results."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class EntryType(Enum):
    """Types of journal entries."""
    DAILY = "daily"
    NOTE = "note"
    REFLECTION = "reflection"
    IDEA = "idea"
    DREAM = "dream"
    OTHER = "other"


@dataclass
class JournalEntry:
    """Represents a single journal entry."""

    id: str
    content: str
    date: datetime
    entry_type: EntryType = EntryType.DAILY
    title: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 1

    def __post_init__(self):
        """Validate and process the entry after initialization."""
        if not self.content:
            raise ValueError("Journal entry content cannot be empty")

        if not self.title:
            self.title = self._generate_title()

    def _generate_title(self) -> str:
        """Generate a title from the first line or first few words."""
        first_line = self.content.split('\n')[0]
        if len(first_line) > 50:
            return first_line[:50] + "..."
        return first_line or f"Entry from {self.date.strftime('%Y-%m-%d')}"

    @property
    def word_count(self) -> int:
        """Calculate word count of the entry."""
        return len(self.content.split())

    @property
    def is_chunked(self) -> bool:
        """Check if this entry is part of a larger chunked document."""
        return self.total_chunks > 1


@dataclass
class SearchResult:
    """Represents a single search result from vector search."""

    entry: JournalEntry
    score: float
    relevance_score: float = 0.0
    highlighted_text: Optional[str] = None

    @property
    def combined_score(self) -> float:
        """Calculate combined score for ranking."""
        return (self.score * 0.7) + (self.relevance_score * 0.3)


@dataclass
class QueryResult:
    """Represents the final result of a query."""

    query: str
    response: str
    sources: List[SearchResult]
    context_used: str
    processing_time: float
    model_used: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def format_response(self, include_sources: bool = True) -> str:
        """Format the response for display."""
        output = [self.response]

        if include_sources and self.sources:
            output.append("\n\n---\nSources:")
            for i, source in enumerate(self.sources[:5], 1):
                date_str = source.entry.date.strftime("%Y-%m-%d")
                output.append(f"{i}. {date_str}: {source.entry.title}")

        return "\n".join(output)