# Cloud Processing Guide

Since your 32GB MacBook Pro is still hitting OOM issues with 1979 journal entries, cloud processing is the most reliable solution.

## Quick Solution: Google Colab (Recommended)

### Step 1: Upload to Google Colab

1. **Create a new Colab notebook**: https://colab.research.google.com/
2. **Upload your journal files**:
   ```python
   from google.colab import files
   import zipfile
   import os

   # Upload your journal data as a zip file
   uploaded = files.upload()

   # Extract the zip file
   for filename in uploaded.keys():
       with zipfile.ZipFile(filename, 'r') as zip_ref:
           zip_ref.extractall('/content/journal_data')
   ```

### Step 2: Install and Run in Colab

```python
# Install the journal system
!git clone https://github.com/yourusername/SLM_journal.git  # Replace with your repo
%cd SLM_journal

# Install dependencies
!pip install -r requirements.txt

# Set environment for cloud processing
import os
os.environ['JOURNAL_USE_CPU_ONLY'] = 'false'  # Use GPU in Colab
os.environ['JOURNAL_LLM_MODEL_TYPE'] = 'mock'
os.environ['JOURNAL_EMBEDDING__BATCH_SIZE'] = '32'  # Can use larger batches with GPU

# Run indexing
!python main.py --data-path /content/journal_data index --force
```

### Step 3: Download Results

```python
# Zip the storage directory
!zip -r journal_storage.zip storage/

# Download the indexed data
from google.colab import files
files.download('journal_storage.zip')
```

### Step 4: Use Locally

1. Extract `journal_storage.zip` to your local `storage/` directory
2. Run queries locally: `python main.py query "your question"`

## Alternative: AWS EC2

### Launch Instance
```bash
# Use t3.xlarge (16GB RAM) or t3.2xlarge (32GB RAM)
aws ec2 run-instances \
    --image-id ami-0abcdef1234567890 \  # Ubuntu 22.04 LTS
    --instance-type t3.xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-12345678 \
    --subnet-id subnet-12345678
```

### Setup and Process
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install Python and dependencies
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git

# Clone your repo
git clone https://github.com/yourusername/SLM_journal.git
cd SLM_journal

# Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Upload your journal files (use scp from local machine)
# scp -i your-key.pem -r /path/to/your/journals ubuntu@your-instance-ip:~/SLM_journal/data/

# Run indexing with more memory
export JOURNAL_EMBEDDING__BATCH_SIZE=16
python main.py index --force

# Download results (from local machine)
# scp -i your-key.pem -r ubuntu@your-instance-ip:~/SLM_journal/storage/ ./
```

## Local Emergency Mode (Last Resort)

If you must process locally, try the emergency script:

```bash
./index_emergency_low_memory.sh
```

This uses:
- Batch size of 1 (one entry at a time)
- No caching
- Ultra-small chunks (128 tokens)
- CPU-only mode with 2 threads
- Aggressive garbage collection

**Warning**: This will be extremely slow (possibly hours for 1979 entries) but should avoid OOM.

## Cost Comparison

| Option | Cost | Time | Reliability |
|--------|------|------|-------------|
| Google Colab (Free) | $0 | ~30 min | High |
| AWS t3.xlarge | ~$0.17/hour | ~1 hour | High |
| Local Emergency | $0 | 3-6 hours | Medium |

## Recommended Workflow

1. **Try emergency local mode first** (free, but slow)
2. **If it fails, use Google Colab** (free, fast, reliable)
3. **For regular processing, consider AWS** (small cost, full control)

## After Cloud Processing

Once you have the indexed data locally:

1. **Queries are fast**: Local queries don't need much memory
2. **Updates are incremental**: Only new entries need processing
3. **Backup your indices**: Keep the `storage/` directory backed up

## Memory Monitoring

While running emergency mode locally, monitor memory:

```bash
# In another terminal
tail -f memory_usage.log
```

If memory usage approaches 30GB, the process will likely be killed.

## Future Optimization

Consider these for better local performance:
1. **Upgrade to 64GB+ RAM** for comfortable local processing
2. **Use smaller embedding models** (sacrifice some quality for speed)
3. **Process journals in chronological batches** (index year by year)
4. **Use disk-based embedding cache** instead of memory cache