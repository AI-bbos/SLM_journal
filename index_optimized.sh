#!/bin/bash
# Memory-optimized indexing script for 32GB MacBook Pro

echo "Starting memory-optimized indexing..."
echo "This will process your journal entries in small batches to avoid memory issues."
echo ""

# Activate virtual environment
source .venv/bin/activate

# Set memory-optimized environment variables
export JOURNAL_USE_CPU_ONLY=true
export JOURNAL_EMBEDDING_THREADS=4
export JOURNAL_EMBEDDING_MAX_SEQ_LENGTH=256
export JOURNAL_LLM_MODEL_TYPE=mock
export JOURNAL_EMBEDDING__BATCH_SIZE=4
export JOURNAL_EMBEDDING__CACHE_SIZE=500
export JOURNAL_EMBEDDING__USE_CACHE=false
export JOURNAL_INGESTION__MAX_TOKENS=200
export JOURNAL_INGESTION__BATCH_SIZE=25

# Run indexing with force rebuild
echo "Running indexing with memory optimizations..."
python main.py index --force

echo ""
echo "Indexing complete! You can now run queries with:"
echo "python main.py query 'your question here'"