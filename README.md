# Personal Journal Query System

A sophisticated RAG (Retrieval-Augmented Generation) system for querying personal journal entries using local AI models. Build your searchable knowledge base locally with privacy-first design.

## Features

- **Local AI Processing**: Uses local language models (no data sent to external services)
- **Multiple Format Support**: Markdown, plain text, JSON, and HTML journal files
- **Semantic Search**: Vector-based similarity search with FAISS
- **Intelligent Chunking**: Automatically splits long entries while preserving context
- **Rich CLI Interface**: Beautiful command-line interface with Rich
- **Flexible Queries**: Question answering, summarization, emotion analysis, goal tracking
- **Local Processing**: Query your journals privately on your own machine
- **Cloud Indexing Support**: Process large datasets using Google Colab or cloud services

## Quick Start

### 1. Setup

```bash
# Clone or download the project
cd SLM_journal

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Your Journal Files

Place your journal files in the `data/` directory:

```
data/
├── 2023-01-journal.md
├── 2023-02-journal.md
├── notes/
│   ├── daily-notes.txt
│   └── reflections.md
```

Supported formats:
- **Markdown** (.md, .markdown): With date headers like `# 2023-01-15`
- **Plain Text** (.txt): With date prefixes like `2023-01-15:`
- **JSON** (.json): With structured entries
- **HTML** (.html, .htm): Blog exports, web pages, or HTML-formatted journals

#### HTML Journal Support

The system can automatically convert HTML files to clean Markdown, making it perfect for:

- **Blog exports** from WordPress, Blogger, Medium
- **Web page archives** of your online journal entries
- **HTML-formatted** personal notes and reflections

Features of HTML processing:
- Automatically removes navigation, ads, footers, and other non-content elements
- Preserves formatting (bold, italic, lists, quotes, headings)
- Extracts metadata (title, author, date, tags) from HTML meta tags
- Identifies individual journal entries in multi-entry HTML files
- Converts dates to consistent YYYY-MM-DD format

Example HTML structure that works well:
```html
<article class="entry">
    <h2>March 15, 2024</h2>
    <p>Today was a <strong>productive day</strong>...</p>
</article>
```

### 3. Index Your Journals

#### For Small to Medium Collections (< 500 entries)
```bash
python main.py index
```

#### For Large Collections (1000+ entries)
For large journal collections, indexing requires significant memory. You have three options:

**Option A: Emergency Local Mode** (slow but works on 32GB+ Macs)
```bash
./index_emergency_low_memory.sh
```

**Option B: Google Colab** (recommended - free and fast)
```bash
# See CLOUD_PROCESSING.md for complete instructions
# Upload your journals to Colab, index there, download results
```

**Option C: Cloud Server** (AWS, etc.)
```bash
# See CLOUD_PROCESSING.md for AWS setup instructions
```

The indexing process:
- Parses all journal files
- Generates embeddings using sentence-transformers
- Builds a searchable vector index
- Stores metadata in SQLite

**Note**: The system starts with a mock AI model for immediate functionality. See [Model Setup](#model-setup) below for upgrading to local AI models.

### 4. Start Querying

```bash
# Ask questions about your journals
python main.py query "What are my main goals this year?"

# Analyze emotions over time
python main.py emotions --days 30

# Summarize a specific period
python main.py summary --since 2023-01-01 --until 2023-01-31

# Track progress on goals
python main.py goal "meditation practice"
```

## Command Reference

### Basic Commands

```bash
# Index/re-index journal files
python main.py index [--force]

# Query journal entries
python main.py query "your question" [--limit 10] [--since DATE] [--until DATE]

# Show system statistics
python main.py stats

# Show configuration
python main.py config-show
```

### Analysis Commands

```bash
# Analyze emotions
python main.py emotions [--days 30] [--since DATE] [--until DATE]

# Summarize time period
python main.py summary [--days 7] [--since DATE] [--until DATE] [--focus TOPIC]

# Track goal progress
python main.py goal "goal description" [--days 90]
```

### Query Options

- `--limit`: Number of results to return (default: 10)
- `--since`: Start date in YYYY-MM-DD format
- `--until`: End date in YYYY-MM-DD format
- `--type`: Response type (question_answering, summarization, reflection, emotion_analysis)
- `--sources`: Show source entries
- `--format`: Output format (rich, plain, json)

## Configuration

### Environment Variables

```bash
export JOURNAL_DATA_PATH="/path/to/journals"
export JOURNAL_LLM_MODEL_TYPE="local"  # or "ollama", "mock"
export JOURNAL_EMBEDDING_MODEL_TYPE="all-MiniLM-L6-v2"
```

### Configuration File

Create a `config.yaml`:

```yaml
data_path: "/Users/yourname/Documents/journals"
embedding:
  model_type: "all-MiniLM-L6-v2"
  batch_size: 32
llm:
  model_type: "local"
  model_name: "phi-2"
  use_metal: true
search:
  k: 15
  similarity_threshold: 0.3
```

Use with: `python main.py --config config.yaml`

## Model Setup

The system supports three types of AI models:

### 1. Mock Model (Default)
- **Purpose**: Immediate functionality without downloads
- **Responses**: Generic but functional for testing
- **Usage**: No setup required, works out of the box

### 2. Local Models (Recommended)
Download and run models locally on your Mac:

```bash
# Download Phi-2 (2.7B parameters, efficient)
python main.py download-model --model phi-2

# Switch to local model
export JOURNAL_LLM_MODEL_TYPE="local"

# Or update config permanently
python main.py config-show  # see current config
```

**Available Models:**
- **Phi-2**: Fast, efficient, good for Q&A (2.7B parameters)
- **Llama-2-7B**: Larger, more capable (requires more RAM)
- **Mistral-7B**: Balanced performance and size

### 3. Ollama Integration
Use [Ollama](https://ollama.ai) for easy model management:

```bash
# Install Ollama
brew install ollama  # or download from ollama.ai

# Pull a model
ollama pull llama2

# Configure system to use Ollama
export JOURNAL_LLM_MODEL_TYPE="ollama"
export JOURNAL_LLM_MODEL_NAME="llama2"
```

### Model Comparison

| Model Type | Setup Time | Response Quality | Privacy | Resource Usage |
|------------|------------|------------------|---------|----------------|
| Mock       | Instant    | Basic           | Complete| Minimal        |
| Local      | 5-10 min   | High            | Complete| Moderate       |
| Ollama     | 2-5 min    | High            | Complete| Moderate       |

## Example Queries

### Personal Insights
- "What patterns do you see in my mood over the past month?"
- "How has my thinking about relationships evolved?"
- "What are the main themes in my journal entries?"

### Goal Tracking
- "How am I progressing with my fitness goals?"
- "What challenges have I faced with my meditation practice?"
- "Show me entries related to my career development"

### Reflection
- "What was I worried about six months ago?"
- "How did I handle stress during difficult periods?"
- "What insights have I gained about myself this year?"

### Time-Based Analysis
- "Summarize my thoughts from last week"
- "Compare my mindset from January vs June"
- "What was I grateful for during the holidays?"

## Architecture

The system uses a modular, object-oriented design:

```
src/
├── domain/          # Core models (JournalEntry, QueryResult)
├── ingestion/       # File parsing and preprocessing
├── embedding/       # Text embedding generation
├── storage/         # Vector and metadata storage
├── retrieval/       # Semantic search and ranking
├── generation/      # LLM integration and response generation
├── application/     # Main engine and configuration
└── cli/             # Command-line interface
```

### Key Components

- **DataIngester**: Parses multiple file formats
- **EmbeddingService**: Generates vector embeddings
- **FAISSVectorStore**: Efficient similarity search
- **SemanticSearcher**: Retrieval with ranking
- **ResponseGenerator**: LLM-powered responses
- **QueryEngine**: Orchestrates the full pipeline

## Hardware Requirements

### For Querying (after indexing)
- **Minimum**: MacBook Air M1 (8GB RAM)
- **Recommended**: Any Mac with 16GB+ RAM
- 2GB free disk space for indexed data
- macOS 11.0+ or Linux/Windows

### For Local Indexing
- **Small collections** (< 500 entries): 16GB+ RAM
- **Medium collections** (500-1000 entries): 32GB+ RAM
- **Large collections** (1000+ entries): **Cloud processing recommended**
- 5GB free disk space for models and indices
- SSD storage recommended

### Performance Notes
- **Indexing**: Memory-intensive, see [Memory Optimization Guide](MEMORY_OPTIMIZATION.md)
- **Querying**: Fast and lightweight (~2-5 seconds, 2-4GB RAM)
- **Apple Silicon**: Optimized for querying with Metal Performance Shaders
- Models cached locally after first download

### Cloud Processing
For large journal collections, cloud processing is more reliable:
- **Google Colab**: Free, 12GB+ RAM, GPU acceleration
- **AWS t3.xlarge**: ~$0.17/hour, 16GB RAM
- **Local emergency mode**: Very slow but works on 32GB+ systems

## Troubleshooting

### Common Issues

**"Failed to load model" Error**
```bash
# The system now defaults to mock model, but if you see this error:
export JOURNAL_LLM_MODEL_TYPE="mock"
# Or download a model:
python main.py download-model
```

**Import Errors**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate
# Reinstall dependencies
pip install -r requirements.txt
```

**Model Download Fails**
```bash
# Use mock model for immediate functionality
export JOURNAL_LLM_MODEL_TYPE="mock"
# Or try Ollama instead:
brew install ollama
ollama pull llama2
export JOURNAL_LLM_MODEL_TYPE="ollama"
```

**Memory Issues During Indexing**
```bash
# For large datasets, use emergency mode
./index_emergency_low_memory.sh

# Or use optimized configuration
export JOURNAL_EMBEDDING__BATCH_SIZE=1
export JOURNAL_USE_CPU_ONLY=true
export JOURNAL_EMBEDDING__USE_CACHE=false

# Best option: Use cloud processing
# See CLOUD_PROCESSING.md for complete guide
```

**No Results Found**
```bash
# Re-index with force flag
python main.py index --force
# Check data path
python main.py config-show
```

### Performance Tuning

For large journal collections (>10,000 entries):

1. Use IVFFlat index: `export JOURNAL_STORAGE_INDEX_TYPE="IVFFlat"`
2. Increase chunk overlap: `export JOURNAL_INGESTION_OVERLAP_TOKENS=100`
3. Adjust similarity threshold: `export JOURNAL_SEARCH_SIMILARITY_THRESHOLD=0.4`

## Privacy & Security

- **All processing is local** - no data sent to external services
- **No internet required** after initial model download
- **Data stored locally** in SQLite and FAISS indexes
- **No logging of personal content** (only system metrics)

## Development

### Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

### Code Style

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

### Adding New Features

The modular architecture makes it easy to extend:

1. **New file formats**: Add parser in `src/ingestion/parser.py`
2. **New models**: Implement `LLMInterface` in `src/generation/llm.py`
3. **New query types**: Add templates in `src/generation/prompts.py`
4. **New CLI commands**: Extend `src/cli/main.py`

## License

This project is open source. Feel free to modify and adapt for your personal use.

## Contributing

This is a personal project, but suggestions and improvements are welcome! The code is designed to be readable and extensible.

---

**Built with ❤️ for personal knowledge management**