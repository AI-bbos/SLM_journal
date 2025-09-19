"""Basic tests for the journal query system."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from src.domain.models import JournalEntry, EntryType
from src.ingestion.parser import JournalParser
from src.ingestion.preprocessor import TextPreprocessor
from src.application.config import Config
from src.application.engine import QueryEngine


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_journal_content():
    """Sample journal content for testing."""
    return """# 2023-01-01

Today was a great day. I went for a walk in the park and felt really peaceful.
The weather was perfect and I met some interesting people.

# 2023-01-02

Had some challenges at work today. Feeling a bit stressed about the project deadline.
Need to focus on time management and prioritization.

# 2023-01-03

Worked on my meditation practice this morning. Feeling much more centered and calm.
The breathing exercises really help with anxiety."""


def test_journal_entry_creation():
    """Test creating a journal entry."""
    entry = JournalEntry(
        id="test_1",
        content="This is a test entry.",
        date=datetime.now(),
        title="Test Entry"
    )

    assert entry.id == "test_1"
    assert entry.content == "This is a test entry."
    assert entry.title == "Test Entry"
    assert entry.word_count == 5


def test_text_preprocessor():
    """Test text preprocessing."""
    preprocessor = TextPreprocessor()

    text = "This is a test.    Multiple   spaces. \n\n\nMultiple newlines."
    cleaned = preprocessor.clean_text(text)

    assert "  " not in cleaned
    assert "\n\n\n" not in cleaned


def test_markdown_parser(temp_dir, sample_journal_content):
    """Test parsing markdown journal files."""
    # Create a test markdown file
    md_file = temp_dir / "test.md"
    md_file.write_text(sample_journal_content)

    parser = JournalParser()
    entries = parser.parse_file(md_file)

    assert len(entries) == 3
    assert "great day" in entries[0]['content']
    assert "challenges at work" in entries[1]['content']
    assert "meditation practice" in entries[2]['content']


def test_config_creation():
    """Test configuration creation."""
    config = Config()

    assert config.embedding.model_type == "all-MiniLM-L6-v2"
    assert config.llm.model_type == "local"
    assert config.search.k == 10


def test_query_engine_initialization():
    """Test query engine initialization."""
    config = Config()
    config.llm.model_type = "mock"  # Use mock LLM for testing

    engine = QueryEngine(config)

    # Should not raise any errors
    engine.initialize()

    assert engine._initialized is True
    assert engine.embedding_service is not None
    assert engine.vector_store is not None
    assert engine.metadata_store is not None


def test_query_engine_with_mock_data(temp_dir, sample_journal_content):
    """Test query engine with mock data."""
    # Create test data
    md_file = temp_dir / "test.md"
    md_file.write_text(sample_journal_content)

    # Configure engine
    config = Config()
    config.data_path = temp_dir
    config.storage_path = temp_dir / "storage"
    config.llm.model_type = "mock"

    engine = QueryEngine(config)
    engine.initialize()

    # Index the data
    stats = engine.ingest_data()
    assert stats['total_entries'] > 0

    # Test a query
    result = engine.query("What did I do for relaxation?")
    assert result.response is not None
    assert len(result.sources) > 0


if __name__ == "__main__":
    pytest.main([__file__])