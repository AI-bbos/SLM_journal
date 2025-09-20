"""SQLite-based metadata storage for journal entries."""

import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import logging
from datetime import datetime
from contextlib import contextmanager

from src.domain.models import JournalEntry, EntryType

logger = logging.getLogger(__name__)


class MetadataStore:
    """SQLite storage for journal entry metadata."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    date TIMESTAMP NOT NULL,
                    title TEXT,
                    entry_type TEXT,
                    file_path TEXT,
                    chunk_index INTEGER DEFAULT 0,
                    total_chunks INTEGER DEFAULT 1,
                    tags TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_date ON entries(date)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entry_type ON entries(entry_type)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_path ON entries(file_path)
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    response TEXT,
                    sources TEXT,
                    processing_time REAL,
                    model_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def add_entries(self, entries: List[JournalEntry]) -> None:
        """Add journal entries to the store."""
        with self._get_connection() as conn:
            for entry in entries:
                conn.execute("""
                    INSERT OR REPLACE INTO entries (
                        id, content, date, title, entry_type, file_path,
                        chunk_index, total_chunks, tags, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.id,
                    entry.content,
                    entry.date,
                    entry.title,
                    entry.entry_type.value,
                    entry.file_path,
                    entry.chunk_index,
                    entry.total_chunks,
                    json.dumps(entry.tags),
                    json.dumps(entry.metadata)
                ))

            conn.commit()
            logger.info(f"Added {len(entries)} entries to metadata store")

    def get_entry(self, entry_id: str) -> Optional[JournalEntry]:
        """Get a single entry by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM entries WHERE id = ?", (entry_id,)
            ).fetchone()

            if row:
                return self._row_to_entry(row)
            return None

    def get_entries(self, entry_ids: List[str]) -> List[JournalEntry]:
        """Get multiple entries by IDs."""
        entries = []
        with self._get_connection() as conn:
            placeholders = ','.join(['?'] * len(entry_ids))
            cursor = conn.execute(
                f"SELECT * FROM entries WHERE id IN ({placeholders})",
                entry_ids
            )

            for row in cursor:
                entries.append(self._row_to_entry(row))

        return entries

    def search_entries(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        entry_type: Optional[EntryType] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[JournalEntry]:
        """Search entries by various criteria."""
        query = "SELECT * FROM entries WHERE 1=1"
        params = []

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        if entry_type:
            query += " AND entry_type = ?"
            params.append(entry_type.value)

        query += " ORDER BY date DESC LIMIT ?"
        params.append(limit)

        entries = []
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)

            for row in cursor:
                entry = self._row_to_entry(row)

                if tags:
                    if any(tag in entry.tags for tag in tags):
                        entries.append(entry)
                else:
                    entries.append(entry)

        return entries

    def save_query_history(
        self,
        query: str,
        response: str,
        sources: List[str],
        processing_time: float,
        model_used: str
    ) -> None:
        """Save query history for analysis."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO query_history (
                    query, response, sources, processing_time, model_used
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                query,
                response,
                json.dumps(sources),
                processing_time,
                model_used
            ))
            conn.commit()

    def get_statistics(self) -> dict:
        """Get statistics about the stored entries."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

            date_range = conn.execute("""
                SELECT MIN(date) as min_date, MAX(date) as max_date
                FROM entries
            """).fetchone()

            type_counts = {}
            cursor = conn.execute("""
                SELECT entry_type, COUNT(*) as count
                FROM entries
                GROUP BY entry_type
            """)
            for row in cursor:
                type_counts[row['entry_type']] = row['count']

            unique_files = conn.execute("""
                SELECT COUNT(DISTINCT file_path) FROM entries
            """).fetchone()[0]

            query_count = conn.execute("""
                SELECT COUNT(*) FROM query_history
            """).fetchone()[0]

            return {
                'total_entries': total,
                'date_range': {
                    'earliest': date_range['min_date'].isoformat() if date_range['min_date'] and hasattr(date_range['min_date'], 'isoformat') else str(date_range['min_date']) if date_range['min_date'] else None,
                    'latest': date_range['max_date'].isoformat() if date_range['max_date'] and hasattr(date_range['max_date'], 'isoformat') else str(date_range['max_date']) if date_range['max_date'] else None
                },
                'entry_types': type_counts,
                'unique_files': unique_files,
                'query_count': query_count
            }

    def _row_to_entry(self, row: sqlite3.Row) -> JournalEntry:
        """Convert a database row to a JournalEntry."""
        return JournalEntry(
            id=row['id'],
            content=row['content'],
            date=row['date'],
            title=row['title'],
            entry_type=EntryType(row['entry_type']),
            file_path=row['file_path'],
            chunk_index=row['chunk_index'],
            total_chunks=row['total_chunks'],
            tags=json.loads(row['tags']) if row['tags'] else [],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )

    def clear_all(self) -> None:
        """Clear all entries from the store."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM entries")
            conn.execute("DELETE FROM query_history")
            conn.commit()
            logger.info("Cleared all entries from metadata store")