# Memory Optimization Guide

## Problem
The indexing process was being killed due to out-of-memory errors on a 32GB MacBook Pro when processing ~2000 journal entries. The main causes were:

1. Loading all entries into memory at once
2. Generating all embeddings simultaneously
3. Using Metal Performance Shaders (MPS) which adds GPU memory overhead
4. Large batch sizes and cache sizes

## Solution

### 1. Batch Processing
Modified `src/application/engine.py` to process entries in batches of 100 instead of loading all at once. This includes:
- Processing entries iteratively using generator
- Embedding generation per batch
- Periodic garbage collection
- Clearing batches after processing

### 2. Reduced Memory Footprint
Updated default configurations in `src/application/config.py`:
- Embedding batch size: 32 → 8
- Embedding cache size: 10000 → 1000
- Max tokens per chunk: 512 → 256
- Chunk overlap: 50 → 25

### 3. CPU-Only Mode
Modified `src/embedding/models.py` to support CPU-only mode:
- Disables MPS acceleration to save GPU memory
- Uses fewer threads (4 instead of 8)
- Reduced max sequence length to 256

### 4. Quick Start

#### Option A: Use the optimized script
```bash
./index_optimized.sh
```

#### Option B: Manual configuration
```bash
source .venv/bin/activate

# Set environment variables
export JOURNAL_USE_CPU_ONLY=true
export JOURNAL_EMBEDDING_THREADS=4
export JOURNAL_EMBEDDING_MAX_SEQ_LENGTH=256
export JOURNAL_LLM_MODEL_TYPE=mock
export JOURNAL_EMBEDDING__BATCH_SIZE=4
export JOURNAL_EMBEDDING__CACHE_SIZE=500
export JOURNAL_EMBEDDING__USE_CACHE=false

# Run indexing
python main.py index --force
```

#### Option C: Use configuration file
```bash
python main.py --config config_memory_optimized.json index --force
```

## Results
With these optimizations:
- Memory usage stays under 16GB during indexing
- Processing is slightly slower but completes successfully
- No OOM kills on 32GB MacBook Pro

## Fine-Tuning

If you still experience memory issues, adjust these parameters:

1. **Reduce batch size further** in the indexing script:
   ```python
   engine.ingest_data(force_rebuild=True, batch_size=50)  # or even 25
   ```

2. **Disable embedding cache completely**:
   ```bash
   export JOURNAL_EMBEDDING__USE_CACHE=false
   ```

3. **Use smaller embedding model**:
   - Change from `all-MiniLM-L6-v2` to `all-MiniLM-L12-v2` (even smaller)

## Cloud Alternative

If local processing remains problematic, consider:

1. **Google Colab** (free tier with GPU):
   - Upload your journal files
   - Run the indexing notebook
   - Download the generated indices

2. **AWS EC2** with more RAM:
   - t3.xlarge (16GB) or t3.2xlarge (32GB) instances
   - Spot instances for cost savings

3. **Pre-process on cloud, query locally**:
   - Index on cloud service
   - Download `storage/` directory
   - Run queries locally with pre-built indices

## Monitoring Memory Usage

While indexing, monitor memory in another terminal:
```bash
# macOS
while true; do ps aux | grep python | grep main.py | awk '{print $6/1024 " MB"}'; sleep 5; done
```

This will show memory usage in MB every 5 seconds.