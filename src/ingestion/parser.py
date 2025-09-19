"""Parser for different journal file formats."""

import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import json

from src.domain.models import JournalEntry, EntryType
from src.ingestion.html_converter import HTMLToMarkdownConverter


class BaseParser(ABC):
    """Abstract base class for journal parsers."""

    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file."""
        pass

    @abstractmethod
    def parse(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse the file and return raw entry data."""
        pass

    def extract_date(self, text: str, file_path: Path) -> datetime:
        """Extract date from text or filename."""
        date_patterns = [
            r'(\d{4})-(\d{2})-(\d{2})',
            r'(\d{2})/(\d{2})/(\d{4})',
            r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{4})',
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text[:200], re.IGNORECASE)
            if match:
                try:
                    if '/' in pattern:
                        return datetime.strptime(match.group(), '%m/%d/%Y')
                    elif '-' in pattern:
                        return datetime.strptime(match.group(), '%Y-%m-%d')
                    else:
                        month_map = {
                            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                            'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                            'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                        }
                        day, month_str, year = match.groups()
                        month = month_map[month_str[:3].lower()]
                        return datetime(int(year), month, int(day))
                except:
                    continue

        match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', str(file_path))
        if match:
            try:
                year, month, day = match.groups()
                return datetime(int(year), int(month), int(day))
            except:
                pass

        return datetime.fromtimestamp(file_path.stat().st_mtime)


class MarkdownParser(BaseParser):
    """Parser for Markdown journal files."""

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in ['.md', '.markdown']

    def parse(self, file_path: Path) -> List[Dict[str, Any]]:
        content = file_path.read_text(encoding='utf-8')
        entries = []

        sections = re.split(r'^#{1,3}\s+', content, flags=re.MULTILINE)

        if len(sections) > 1:
            for section in sections[1:]:
                lines = section.split('\n')
                title = lines[0].strip()
                entry_content = '\n'.join(lines[1:]).strip()

                if entry_content:
                    entries.append({
                        'content': entry_content,
                        'title': title,
                        'date': self.extract_date(entry_content, file_path),
                        'file_path': str(file_path),
                        'tags': self._extract_tags(entry_content)
                    })
        else:
            entries.append({
                'content': content,
                'title': None,
                'date': self.extract_date(content, file_path),
                'file_path': str(file_path),
                'tags': self._extract_tags(content)
            })

        return entries

    def _extract_tags(self, content: str) -> List[str]:
        """Extract hashtags from content."""
        tags = re.findall(r'#(\w+)', content)
        return list(set(tags))


class TextParser(BaseParser):
    """Parser for plain text journal files."""

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in ['.txt', '.text']

    def parse(self, file_path: Path) -> List[Dict[str, Any]]:
        content = file_path.read_text(encoding='utf-8')
        entries = []

        date_pattern = r'^(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})'
        sections = re.split(date_pattern, content, flags=re.MULTILINE)

        if len(sections) > 2:
            for i in range(1, len(sections), 2):
                if i + 1 < len(sections):
                    date_str = sections[i]
                    entry_content = sections[i + 1].strip()

                    try:
                        if '-' in date_str:
                            date = datetime.strptime(date_str, '%Y-%m-%d')
                        else:
                            date = datetime.strptime(date_str, '%m/%d/%Y')
                    except:
                        date = self.extract_date(entry_content, file_path)

                    if entry_content:
                        entries.append({
                            'content': entry_content,
                            'title': None,
                            'date': date,
                            'file_path': str(file_path),
                            'tags': []
                        })
        else:
            entries.append({
                'content': content,
                'title': None,
                'date': self.extract_date(content, file_path),
                'file_path': str(file_path),
                'tags': []
            })

        return entries


class JSONParser(BaseParser):
    """Parser for JSON formatted journal files."""

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.json'

    def parse(self, file_path: Path) -> List[Dict[str, Any]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        entries = []

        if isinstance(data, list):
            for item in data:
                entries.append(self._parse_json_entry(item, file_path))
        elif isinstance(data, dict):
            if 'entries' in data:
                for item in data['entries']:
                    entries.append(self._parse_json_entry(item, file_path))
            else:
                entries.append(self._parse_json_entry(data, file_path))

        return entries

    def _parse_json_entry(self, item: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """Parse a single JSON entry."""
        content = item.get('content', '') or item.get('text', '') or str(item)

        date_str = item.get('date', '') or item.get('timestamp', '')
        if date_str:
            try:
                date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except:
                date = self.extract_date(content, file_path)
        else:
            date = self.extract_date(content, file_path)

        return {
            'content': content,
            'title': item.get('title'),
            'date': date,
            'file_path': str(file_path),
            'tags': item.get('tags', []),
            'metadata': {k: v for k, v in item.items()
                        if k not in ['content', 'text', 'date', 'timestamp', 'title', 'tags']}
        }


class HTMLParser(BaseParser):
    """Parser for HTML journal files."""

    def __init__(self):
        self.converter = HTMLToMarkdownConverter()

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in ['.html', '.htm']

    def parse(self, file_path: Path) -> List[Dict[str, Any]]:
        # Convert HTML to Markdown
        markdown_content = self.converter.convert_html_file(file_path)

        # Now parse as markdown
        entries = []

        # Split by date headers
        sections = re.split(r'^#{1,3}\s+', markdown_content, flags=re.MULTILINE)

        if len(sections) > 1:
            for section in sections[1:]:
                lines = section.split('\n')
                title = lines[0].strip()
                entry_content = '\n'.join(lines[1:]).strip()

                if entry_content:
                    # Check if title is a date
                    if self._is_date_like(title):
                        date = self.extract_date(title, file_path)
                        actual_title = None
                    else:
                        date = self.extract_date(entry_content, file_path)
                        actual_title = title

                    entries.append({
                        'content': entry_content,
                        'title': actual_title,
                        'date': date,
                        'file_path': str(file_path),
                        'tags': self._extract_tags(entry_content)
                    })
        else:
            # Single entry
            entries.append({
                'content': markdown_content,
                'title': None,
                'date': self.extract_date(markdown_content, file_path),
                'file_path': str(file_path),
                'tags': self._extract_tags(markdown_content)
            })

        return entries

    def _is_date_like(self, text: str) -> bool:
        """Check if text looks like a date."""
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}',
            r'^\d{1,2}/\d{1,2}/\d{4}',
            r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
        ]
        for pattern in date_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        return False

    def _extract_tags(self, content: str) -> List[str]:
        """Extract hashtags from content."""
        tags = re.findall(r'#(\w+)', content)
        # Also look for "Tags:" line
        tag_line = re.search(r'Tags?:\s*(.+)', content, re.IGNORECASE)
        if tag_line:
            additional_tags = [t.strip() for t in tag_line.group(1).split(',')]
            tags.extend(additional_tags)
        return list(set(tags))


class JournalParser:
    """Main parser that delegates to appropriate format parser."""

    def __init__(self):
        self.parsers = [
            HTMLParser(),
            MarkdownParser(),
            TextParser(),
            JSONParser()
        ]

    def parse_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse a journal file using the appropriate parser."""
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser.parse(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return [{
            'content': content,
            'title': None,
            'date': datetime.fromtimestamp(file_path.stat().st_mtime),
            'file_path': str(file_path),
            'tags': []
        }]