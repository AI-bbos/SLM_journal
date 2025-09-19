#!/bin/bash
# EMERGENCY low-memory indexing script for systems with OOM issues

echo "🚨 EMERGENCY LOW-MEMORY MODE 🚨"
echo "This script uses the most aggressive memory optimization settings."
echo "Processing will be very slow but should avoid OOM kills."
echo ""

# Activate virtual environment
source .venv/bin/activate

# Ultra-aggressive memory settings
export JOURNAL_USE_CPU_ONLY=true
export JOURNAL_EMBEDDING_THREADS=2  # Minimal threads
export JOURNAL_EMBEDDING_MAX_SEQ_LENGTH=128  # Very short sequences
export JOURNAL_LLM_MODEL_TYPE=mock
export JOURNAL_EMBEDDING__BATCH_SIZE=1  # Process one at a time
export JOURNAL_EMBEDDING__CACHE_SIZE=0  # No cache
export JOURNAL_EMBEDDING__USE_CACHE=false
export JOURNAL_INGESTION__MAX_TOKENS=128  # Very small chunks
export JOURNAL_INGESTION__OVERLAP_TOKENS=10
export JOURNAL_INGESTION__BATCH_SIZE=5  # Tiny batches

# Set Python memory optimization
export PYTHONHASHSEED=0
export MALLOC_ARENA_MAX=1

echo "Memory settings:"
echo "- CPU only mode: ON"
echo "- Threads: 2"
echo "- Embedding batch size: 1"
echo "- Cache: DISABLED"
echo "- Processing batch size: 5"
echo "- Max tokens per chunk: 128"
echo ""

# Monitor memory usage in background
(
    echo "Memory monitoring started (logging to memory_usage.log)..."
    while true; do
        ps aux | grep "python main.py" | grep -v grep | awk '{print strftime("%Y-%m-%d %H:%M:%S"), "Memory:", $6/1024 "MB"}' >> memory_usage.log
        sleep 10
    done
) &
MONITOR_PID=$!

echo "Starting indexing..."
python main.py index --force

# Stop memory monitoring
kill $MONITOR_PID 2>/dev/null

echo ""
echo "✅ Indexing complete!"
echo "📊 Memory usage log saved to memory_usage.log"
echo ""
echo "You can now run queries with:"
echo "python main.py query 'your question here'"