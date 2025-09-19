"""Main data ingestion module for loading journal entries."""

from pathlib import Path
from typing import List, Optional, Generator
import logging
from tqdm import tqdm
from datetime import datetime

from src.domain.models import JournalEntry, EntryType
from src.ingestion.parser import JournalParser
from src.ingestion.preprocessor import TextPreprocessor, ChunkConfig


logger = logging.getLogger(__name__)


class DataIngester:
    """Ingests journal entries from various file formats."""

    def __init__(
        self,
        data_path: Path,
        chunk_config: Optional[ChunkConfig] = None,
        recursive: bool = True
    ):
        self.data_path = Path(data_path)
        self.parser = JournalParser()
        self.preprocessor = TextPreprocessor(chunk_config)
        self.recursive = recursive
        self.supported_extensions = {'.txt', '.md', '.markdown', '.json', '.text', '.html', '.htm'}

    def ingest_all(self) -> List[JournalEntry]:
        """Ingest all journal entries from the data path."""
        entries = []

        for entry in self.ingest_entries():
            entries.append(entry)

        logger.info(f"Ingested {len(entries)} journal entries")
        return entries

    def ingest_entries(self) -> Generator[JournalEntry, None, None]:
        """Generator that yields journal entries one by one."""
        files = self._discover_files()

        for file_path in tqdm(files, desc="Ingesting journal entries"):
            try:
                yield from self._process_file(file_path)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

    def _discover_files(self) -> List[Path]:
        """Discover all journal files in the data path."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

        files = []

        if self.data_path.is_file():
            if self._is_supported_file(self.data_path):
                files.append(self.data_path)
        else:
            pattern = '**/*' if self.recursive else '*'
            for file_path in self.data_path.glob(pattern):
                if file_path.is_file() and self._is_supported_file(file_path):
                    files.append(file_path)

        logger.info(f"Discovered {len(files)} journal files")
        return sorted(files)

    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if file has a supported extension."""
        return file_path.suffix.lower() in self.supported_extensions

    def _process_file(self, file_path: Path) -> Generator[JournalEntry, None, None]:
        """Process a single file and yield journal entries."""
        raw_entries = self.parser.parse_file(file_path)

        for raw_entry in raw_entries:
            content = raw_entry.get('content', '')

            if not content or len(content.strip()) < 10:
                continue

            chunks = self.preprocessor.chunk_text(content)

            if len(chunks) == 1:
                entry_id = self.preprocessor.generate_chunk_id(
                    content, str(file_path), 0
                )
                yield JournalEntry(
                    id=entry_id,
                    content=self.preprocessor.clean_text(content),
                    date=raw_entry['date'],
                    title=raw_entry.get('title'),
                    tags=raw_entry.get('tags', []),
                    file_path=str(file_path),
                    entry_type=self._determine_entry_type(content),
                    metadata={
                        **raw_entry.get('metadata', {}),
                        **self.preprocessor.extract_metadata(content)
                    },
                    chunk_index=0,
                    total_chunks=1
                )
            else:
                for idx, (chunk_text, start_idx, end_idx) in enumerate(chunks):
                    entry_id = self.preprocessor.generate_chunk_id(
                        chunk_text, str(file_path), idx
                    )

                    title = raw_entry.get('title')
                    if title and len(chunks) > 1:
                        title = f"{title} (Part {idx + 1}/{len(chunks)})"

                    yield JournalEntry(
                        id=entry_id,
                        content=chunk_text,
                        date=raw_entry['date'],
                        title=title,
                        tags=raw_entry.get('tags', []),
                        file_path=str(file_path),
                        entry_type=self._determine_entry_type(chunk_text),
                        metadata={
                            **raw_entry.get('metadata', {}),
                            **self.preprocessor.extract_metadata(chunk_text),
                            'chunk_start': start_idx,
                            'chunk_end': end_idx
                        },
                        chunk_index=idx,
                        total_chunks=len(chunks)
                    )

    def _determine_entry_type(self, content: str) -> EntryType:
        """Determine the type of journal entry based on content."""
        content_lower = content.lower()

        if any(word in content_lower for word in ['dream', 'dreamt', 'dreamed']):
            return EntryType.DREAM
        elif any(word in content_lower for word in ['idea', 'concept', 'invention', 'project']):
            return EntryType.IDEA
        elif any(word in content_lower for word in ['reflect', 'thinking about', 'contemplat']):
            return EntryType.REFLECTION
        elif any(word in content_lower for word in ['note:', 'notes:', 'reminder:']):
            return EntryType.NOTE
        elif any(word in content_lower for word in ['today', 'yesterday', 'this morning', 'tonight']):
            return EntryType.DAILY

        return EntryType.OTHER

    def get_statistics(self) -> dict:
        """Get statistics about the ingested data."""
        entries = list(self.ingest_entries())

        if not entries:
            return {"total_entries": 0}

        dates = [e.date for e in entries]
        types_count = {}
        for entry in entries:
            types_count[entry.entry_type.value] = types_count.get(entry.entry_type.value, 0) + 1

        return {
            "total_entries": len(entries),
            "date_range": {
                "earliest": min(dates).isoformat(),
                "latest": max(dates).isoformat()
            },
            "entry_types": types_count,
            "total_words": sum(e.word_count for e in entries),
            "average_words_per_entry": sum(e.word_count for e in entries) // len(entries),
            "chunked_entries": sum(1 for e in entries if e.is_chunked),
            "unique_files": len(set(e.file_path for e in entries))
        }