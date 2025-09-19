# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Personal Journal Query System - a RAG-based application for querying personal journal entries using local AI models. The system is built with Python and optimized for Apple Silicon Macs.

## Development Setup

1. **Python Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Common Commands**:
   - **Index journals**: `python main.py index`
   - **Run queries**: `python main.py query "question"`
   - **Run tests**: `python -m pytest tests/ -v`
   - **Lint code**: `ruff check src/ tests/`
   - **Format code**: `black src/ tests/`
   - **Type checking**: `mypy src/`

## Architecture

The project follows a clean, modular architecture:

```
src/
├── domain/          # Core models (JournalEntry, QueryResult, SearchResult)
├── ingestion/       # File parsing (MD, TXT, JSON, HTML) and preprocessing
├── embedding/       # Text embedding generation (sentence-transformers)
├── storage/         # Vector storage (FAISS) and metadata (SQLite)
├── retrieval/       # Semantic search and result ranking
├── generation/      # LLM integration (llama-cpp-python, Ollama)
├── application/     # Main query engine and configuration
└── cli/             # Command-line interface
```

## Key Design Patterns

- **Repository Pattern**: Data access abstraction in storage layer
- **Factory Pattern**: Model creation in embedding and LLM modules
- **Strategy Pattern**: Swappable backends for embeddings/LLMs
- **Dependency Injection**: Loose coupling throughout the system

## Configuration

- Main config class: `src.application.config.Config`
- Uses Pydantic for validation and settings management
- Environment variables: prefixed with `JOURNAL_`
- Config files: JSON/YAML supported

## Testing

- Tests use pytest with fixtures for temporary directories
- Mock LLM available for testing without downloading models
- Basic tests cover model creation, parsing, and query engine

## Local Models

The system supports:
- **Local models**: Via llama-cpp-python (default: Phi-2)
- **Ollama**: For larger models like Llama-2, Mistral
- **Mock**: For testing and development

## Performance Considerations

- Apple Silicon optimization via Metal Performance Shaders
- FAISS for efficient vector similarity search
- Configurable chunking and embedding batch sizes
- Caching for embeddings and model outputs

## Adding Features

Common extension points:
- New file parsers: `src/ingestion/parser.py`
- HTML conversion rules: `src/ingestion/html_converter.py`
- New LLM backends: `src/generation/llm.py`
- New prompt templates: `src/generation/prompts.py`
- New CLI commands: `src/cli/main.py`

## HTML Processing

The system includes sophisticated HTML to Markdown conversion:
- Uses BeautifulSoup4 and markdownify for parsing and conversion
- Automatically removes navigation, ads, footers, scripts
- Extracts individual journal entries from multi-entry HTML files
- Preserves important formatting (bold, italic, lists, blockquotes)
- Handles date extraction from various HTML structures