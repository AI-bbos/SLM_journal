# Example Journal Files

This directory contains sample journal files in different formats to demonstrate the system's parsing capabilities.

## Files

### sample-journal.md
A Markdown journal with multiple date-based entries showing personal reflections, meditation practice, and goal tracking.

### blog-export.html
An HTML file simulating a blog export with:
- Multiple article entries
- Rich formatting (bold, italic, lists, blockquotes)
- Metadata extraction (dates, titles, tags)
- Automatic removal of navigation/footer elements

## Usage

To test with these examples:

```bash
# Index the example files
python main.py --data-path data/examples index

# Query the examples
python main.py --data-path data/examples query "What are my thoughts on meditation?"

# Analyze emotions in the examples
python main.py --data-path data/examples emotions
```

## Adding Your Own Files

1. **Markdown**: Use `#` headers with dates like `# 2024-01-15`
2. **HTML**: Export from your blog or save web pages with journal content
3. **Text**: Start entries with dates like `2024-01-15: Today was...`
4. **JSON**: Structure entries with `date`, `content`, and optional `title` fields

The system automatically detects file formats and converts them to a unified internal representation for searching and querying.