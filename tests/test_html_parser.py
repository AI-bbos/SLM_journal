"""Tests for HTML parsing functionality."""

import pytest
import tempfile
from pathlib import Path

from src.ingestion.html_converter import HTMLToMarkdownConverter
from src.ingestion.parser import HTMLParser


@pytest.fixture
def sample_html():
    """Sample HTML content for testing."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>My Journal</title>
    <meta name="keywords" content="journal, personal">
    <style>body { font-family: Arial; }</style>
</head>
<body>
    <nav class="navigation">
        <a href="/">Home</a>
    </nav>

    <article class="entry">
        <h2>January 15, 2024</h2>
        <p>Today was a <strong>productive day</strong>. I worked on my goals and made significant progress.</p>
        <p>Key accomplishments:</p>
        <ul>
            <li>Finished the project proposal</li>
            <li>Went for a 30-minute walk</li>
            <li>Read 20 pages of my book</li>
        </ul>
    </article>

    <article class="entry">
        <h2>January 16, 2024</h2>
        <p>Reflection time. Sometimes the best days are the quiet ones where you can just <em>think</em>.</p>
        <blockquote>
            "The unexamined life is not worth living." - Socrates
        </blockquote>
    </article>

    <footer class="footer">
        <p>Copyright 2024</p>
    </footer>
</body>
</html>"""


def test_html_to_markdown_converter():
    """Test HTML to Markdown conversion."""
    converter = HTMLToMarkdownConverter()

    html = """<article>
        <h2>Test Entry</h2>
        <p>This is a <strong>test</strong> paragraph.</p>
        <ul><li>Item 1</li><li>Item 2</li></ul>
    </article>"""

    markdown = converter.convert_html_to_markdown(html)

    assert "# Test Entry" in markdown
    assert "**test**" in markdown
    assert "- Item 1" in markdown
    assert "- Item 2" in markdown


def test_html_parser_with_file(sample_html):
    """Test HTML parser with a file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write(sample_html)
        f.flush()

        parser = HTMLParser()
        entries = parser.parse(Path(f.name))

        # Should extract 2 entries from the articles
        assert len(entries) >= 1

        # Check that content was converted to markdown
        for entry in entries:
            assert 'content' in entry
            assert 'date' in entry
            assert 'file_path' in entry


def test_html_parser_recognition():
    """Test that HTML parser recognizes HTML files."""
    parser = HTMLParser()

    assert parser.can_parse(Path("test.html"))
    assert parser.can_parse(Path("test.htm"))
    assert not parser.can_parse(Path("test.md"))
    assert not parser.can_parse(Path("test.txt"))


def test_html_metadata_extraction():
    """Test extraction of metadata from HTML."""
    converter = HTMLToMarkdownConverter()

    html = """<!DOCTYPE html>
<html>
<head>
    <title>My Blog Post</title>
    <meta name="author" content="John Doe">
    <meta name="keywords" content="personal, growth, mindfulness">
    <meta name="date" content="2024-01-15">
</head>
<body>
    <h1>Test Content</h1>
    <p>This is test content.</p>
</body>
</html>"""

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'lxml')
    metadata = converter._extract_metadata(soup)

    assert metadata['title'] == "My Blog Post"
    assert metadata['tags'] == "personal, growth, mindfulness"
    assert metadata['date'] == "2024-01-15"


def test_html_cleaning():
    """Test that unwanted HTML elements are removed."""
    converter = HTMLToMarkdownConverter()

    html = """<html>
<head>
    <script>console.log('remove me');</script>
    <style>.test { color: red; }</style>
</head>
<body>
    <nav class="navigation">Navigation</nav>
    <main>
        <h1>Keep This</h1>
        <p>And this content.</p>
    </main>
    <footer class="footer">Footer</footer>
</body>
</html>"""

    markdown = converter.convert_html_to_markdown(html)

    # Should keep main content
    assert "Keep This" in markdown
    assert "And this content" in markdown

    # Should remove unwanted elements
    assert "Navigation" not in markdown
    assert "Footer" not in markdown
    assert "console.log" not in markdown


def test_date_formatting():
    """Test date formatting functionality."""
    converter = HTMLToMarkdownConverter()

    test_cases = [
        ("March 15, 2024", "2024-03-15"),
        ("2024-03-15", "2024-03-15"),
        ("15 March 2024", "2024-03-15"),
        ("Mar 15, 2024", "2024-03-15"),
    ]

    for input_date, expected in test_cases:
        formatted = converter._format_date(input_date)
        assert formatted == expected, f"Failed for {input_date}: got {formatted}, expected {expected}"


if __name__ == "__main__":
    pytest.main([__file__])