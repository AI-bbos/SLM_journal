"""HTML to Markdown converter for journal entries."""

import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from bs4 import BeautifulSoup, NavigableString, Tag
import markdownify
import logging

logger = logging.getLogger(__name__)


class HTMLToMarkdownConverter:
    """Convert HTML content to clean Markdown format."""

    def __init__(self):
        # Configure markdownify settings
        self.md_config = {
            'heading_style': 'ATX',  # Use # for headings
            'strip': ['script', 'style', 'meta', 'link'],  # Remove these tags
            'bullets': '-',  # Use - for unordered lists
            'code_language': '',
            'wrap': False,
            'wrap_width': 0
        }

    def convert_html_file(self, file_path: Path) -> str:
        """
        Convert an HTML file to Markdown.

        Args:
            file_path: Path to HTML file

        Returns:
            Markdown content
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()

        return self.convert_html_to_markdown(html_content)

    def convert_html_to_markdown(self, html_content: str) -> str:
        """
        Convert HTML content to clean Markdown.

        Args:
            html_content: HTML string

        Returns:
            Markdown formatted string
        """
        # Parse HTML
        soup = BeautifulSoup(html_content, 'lxml')

        # Pre-process: Extract and preserve important metadata
        metadata = self._extract_metadata(soup)

        # Remove unwanted elements
        self._clean_html(soup)

        # Handle special journal elements
        self._process_journal_elements(soup)

        # Check if we can extract individual entries
        entries = self.extract_journal_entries(soup)

        if entries and len(entries) > 1:
            # Multiple entries found, format them nicely
            markdown_parts = []
            for date, content in entries:
                if date:
                    # Try to format the date consistently
                    formatted_date = self._format_date(date)
                    markdown_parts.append(f"# {formatted_date}")
                    markdown_parts.append("")

                markdown_parts.append(content)
                markdown_parts.append("")

            markdown = "\n".join(markdown_parts)
        else:
            # Single entry or failed to extract, convert as whole
            # Convert to markdown using markdownify
            markdown = markdownify.markdownify(
                str(soup),
                **self.md_config
            )

        # Post-process markdown
        markdown = self._post_process_markdown(markdown)

        # Add metadata if found and no entries were extracted
        if metadata and not (entries and len(entries) > 1):
            markdown = self._add_metadata_to_markdown(metadata, markdown)

        return markdown

    def _format_date(self, date_str: str) -> str:
        """Format date string consistently."""
        if not date_str:
            return ""

        # Try to parse and reformat the date
        try:
            # Try common formats
            for fmt in [
                '%B %d, %Y',        # March 15, 2024
                '%Y-%m-%d',         # 2024-03-15
                '%m/%d/%Y',         # 03/15/2024
                '%d %B %Y',         # 15 March 2024
            ]:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    return date_obj.strftime('%Y-%m-%d')
                except:
                    continue

            # If direct parsing fails, try extracting date parts
            import re
            # Look for month names
            month_match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', date_str, re.IGNORECASE)
            if month_match:
                # Extract day and year
                day_match = re.search(r'\b(\d{1,2})\b', date_str)
                year_match = re.search(r'\b(\d{4})\b', date_str)

                if day_match and year_match:
                    month_name = month_match.group(1)
                    day = day_match.group(1)
                    year = year_match.group(1)

                    try:
                        date_obj = datetime.strptime(f"{month_name} {day}, {year}", '%B %d, %Y')
                        return date_obj.strftime('%Y-%m-%d')
                    except:
                        try:
                            date_obj = datetime.strptime(f"{month_name} {day}, {year}", '%b %d, %Y')
                            return date_obj.strftime('%Y-%m-%d')
                        except:
                            pass

        except:
            pass

        # If all parsing fails, return cleaned string
        return re.sub(r'[^\w\s-]', '', date_str).strip()

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from HTML."""
        metadata = {}

        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text(strip=True)

        # Extract meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                if name in ['date', 'publish-date', 'article:published_time']:
                    metadata['date'] = content
                elif name in ['author', 'article:author']:
                    metadata['author'] = content
                elif name in ['keywords', 'tags']:
                    metadata['tags'] = content
                elif name in ['description', 'og:description']:
                    metadata['description'] = content

        # Look for common date patterns in content
        if 'date' not in metadata:
            date_elem = soup.find(class_=re.compile(r'date|time|published', re.I))
            if date_elem:
                date_text = date_elem.get_text(strip=True)
                metadata['date'] = date_text

        return metadata

    def _clean_html(self, soup: BeautifulSoup) -> None:
        """Remove unwanted HTML elements."""
        # Remove script and style elements
        for element in soup(['script', 'style', 'meta', 'link', 'noscript']):
            element.decompose()

        # Remove comments
        for comment in soup.find_all(text=lambda text: isinstance(text, NavigableString) and isinstance(text, type(text))):
            if '<!--' in str(comment):
                comment.extract()

        # Remove common navigation/footer elements
        for selector in ['.navigation', '.nav', '.footer', '.header', '.sidebar',
                        '.menu', '.advertisement', '.ads', '.social-share', '.comments',
                        '#navigation', '#footer', '#header', '#sidebar', '#comments']:
            for elem in soup.select(selector):
                elem.decompose()

        # Remove elements by common attribute patterns
        for elem in soup.find_all(class_=re.compile(r'nav|footer|header|sidebar|menu|ads?|social|comment', re.I)):
            elem.decompose()

    def _process_journal_elements(self, soup: BeautifulSoup) -> None:
        """Process special journal-specific HTML elements."""

        # Convert common blog/journal structures
        # Article tags often contain the main content
        articles = soup.find_all('article')
        if articles:
            # Keep only article content, remove surrounding elements
            new_soup = BeautifulSoup('<div></div>', 'lxml')
            for article in articles:
                new_soup.div.append(article)
            soup.body.clear()
            soup.body.append(new_soup.div)

        # Process date headers
        for elem in soup.find_all(['h1', 'h2', 'h3']):
            text = elem.get_text(strip=True)
            # Check if it looks like a date
            if self._looks_like_date(text):
                # Convert to consistent date header format
                elem.name = 'h1'

        # Convert divs with specific classes to sections
        for div in soup.find_all('div', class_=re.compile(r'entry|post|content|body|text', re.I)):
            # Keep the content but mark as important
            div.name = 'section'

        # Handle blockquotes (often used for reflections/quotes)
        for blockquote in soup.find_all('blockquote'):
            # Ensure blockquotes are preserved properly
            blockquote['class'] = 'journal-quote'

    def _looks_like_date(self, text: str) -> bool:
        """Check if text looks like a date."""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # 2024-01-15
            r'\d{1,2}/\d{1,2}/\d{4}',  # 1/15/2024
            r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}',  # 15 January 2024
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}',  # January 15, 2024
        ]

        for pattern in date_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _post_process_markdown(self, markdown: str) -> str:
        """Clean up and format the converted markdown."""

        # Remove excessive blank lines
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)

        # Clean up whitespace around headers
        markdown = re.sub(r'\n+#', '\n\n#', markdown)
        markdown = re.sub(r'#([^\n]+)\n+', r'# \1\n\n', markdown)

        # Fix list formatting
        markdown = re.sub(r'\n\s*-\s+', '\n- ', markdown)
        markdown = re.sub(r'\n\s*\d+\.\s+', '\n1. ', markdown)

        # Remove HTML artifacts
        markdown = re.sub(r'<[^>]+>', '', markdown)
        markdown = re.sub(r'&nbsp;', ' ', markdown)
        markdown = re.sub(r'&amp;', '&', markdown)
        markdown = re.sub(r'&lt;', '<', markdown)
        markdown = re.sub(r'&gt;', '>', markdown)
        markdown = re.sub(r'&quot;', '"', markdown)

        # Clean up spacing
        markdown = re.sub(r'[ \t]+', ' ', markdown)
        markdown = re.sub(r'[ \t]+$', '', markdown, flags=re.MULTILINE)

        # Ensure proper paragraph spacing
        lines = markdown.split('\n')
        cleaned_lines = []
        prev_empty = False

        for line in lines:
            is_empty = not line.strip()

            # Don't add multiple empty lines
            if is_empty and prev_empty:
                continue

            cleaned_lines.append(line)
            prev_empty = is_empty

        return '\n'.join(cleaned_lines).strip()

    def _add_metadata_to_markdown(self, metadata: Dict[str, Any], markdown: str) -> str:
        """Add metadata to the beginning of markdown if relevant."""
        header_lines = []

        # Add date as header if found
        if 'date' in metadata:
            date_str = metadata['date']
            # Try to parse and format the date
            try:
                # Try common date formats
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%B %d, %Y', '%d %B %Y']:
                    try:
                        date_obj = datetime.strptime(date_str, fmt)
                        date_str = date_obj.strftime('%Y-%m-%d')
                        break
                    except:
                        continue
            except:
                pass  # Use original date string if parsing fails

            header_lines.append(f"# {date_str}")
            header_lines.append("")

        # Add title if different from date
        if 'title' in metadata and not self._looks_like_date(metadata['title']):
            if not header_lines:  # No date header
                header_lines.append(f"# {metadata['title']}")
            else:  # Has date header, make title h2
                header_lines.append(f"## {metadata['title']}")
            header_lines.append("")

        # Add tags if present
        if 'tags' in metadata:
            tags = metadata['tags']
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(',')]
            if tags:
                header_lines.append(f"Tags: {', '.join(tags)}")
                header_lines.append("")

        if header_lines:
            return '\n'.join(header_lines) + '\n' + markdown

        return markdown

    def extract_journal_entries(self, soup: BeautifulSoup) -> List[Tuple[str, str]]:
        """
        Extract individual journal entries from HTML.

        Returns:
            List of (date, content) tuples
        """
        entries = []

        # Look for common journal/blog entry patterns
        # Pattern 1: Individual article elements
        for article in soup.find_all('article'):
            date = self._extract_entry_date(article)
            content = self._extract_entry_content(article)
            if content and content.strip():
                entries.append((date, content))

        # Pattern 2: Divs with entry/post classes
        if not entries:
            for div in soup.find_all('div', class_=re.compile(r'entry|post|blog-post', re.I)):
                date = self._extract_entry_date(div)
                content = self._extract_entry_content(div)
                if content and content.strip():
                    entries.append((date, content))

        # Pattern 3: Any div with class="entry"
        if not entries:
            for div in soup.find_all('div', class_='entry'):
                date = self._extract_entry_date(div)
                content = self._extract_entry_content(div)
                if content and content.strip():
                    entries.append((date, content))

        # Pattern 4: Sections separated by date headers
        if not entries:
            entries = self._extract_by_date_headers(soup)

        return entries

    def _extract_entry_date(self, element: Tag) -> str:
        """Extract date from an entry element."""
        # Look for date in various places
        date_elem = element.find(class_=re.compile(r'date|time|published', re.I))
        if date_elem:
            return date_elem.get_text(strip=True)

        # Check for time element
        time_elem = element.find('time')
        if time_elem:
            return time_elem.get('datetime', time_elem.get_text(strip=True))

        # Look in headers
        for header in element.find_all(['h1', 'h2', 'h3']):
            text = header.get_text(strip=True)
            if self._looks_like_date(text):
                return text

        return ""

    def _extract_entry_content(self, element: Tag) -> str:
        """Extract content from an entry element."""
        # Clone the element to avoid modifying the original
        content_elem = element

        # Convert to markdown
        markdown = markdownify.markdownify(str(content_elem), **self.md_config)
        return self._post_process_markdown(markdown)

    def _extract_by_date_headers(self, soup: BeautifulSoup) -> List[Tuple[str, str]]:
        """Extract entries separated by date headers."""
        entries = []
        current_date = ""
        current_content = []

        for elem in soup.find_all(['h1', 'h2', 'h3', 'p', 'div', 'section']):
            if elem.name in ['h1', 'h2', 'h3']:
                text = elem.get_text(strip=True)
                if self._looks_like_date(text):
                    # Save previous entry if exists
                    if current_content:
                        content = '\n\n'.join(current_content)
                        entries.append((current_date, self._post_process_markdown(content)))

                    # Start new entry
                    current_date = text
                    current_content = []
                else:
                    # Regular header, add to content
                    current_content.append(markdownify.markdownify(str(elem), **self.md_config))
            else:
                # Add to current entry
                content = markdownify.markdownify(str(elem), **self.md_config)
                if content.strip():
                    current_content.append(content)

        # Save last entry
        if current_content:
            content = '\n\n'.join(current_content)
            entries.append((current_date, self._post_process_markdown(content)))

        return entries