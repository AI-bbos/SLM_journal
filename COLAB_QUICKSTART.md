# Google Colab Quick Start Guide

## Step 1: Upload Your Journal Data

```python
# Upload your journal files
from google.colab import files
import zipfile
import os

# Upload your journal data zip file
uploaded = files.upload()

# Extract the uploaded zip
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('/content/journal_data')
        print(f"✅ Extracted {filename} to /content/journal_data")

# Check what was uploaded
!ls -la /content/journal_data/
```

## Step 2: Clone and Setup

```python
# Clone the repository
!git clone https://github.com/AI-bbos/SLM_journal.git

# IMPORTANT: Navigate to the correct directory
%cd SLM_journal
!pwd  # Should show /content/SLM_journal

# Install dependencies
!pip install -q -r requirements.txt

# Verify we're in the right place
!ls  # Should show main.py, src/, requirements.txt, etc.
```

## Step 3: Index Your Journals

```python
# Method 1: Use the robust launcher (recommended)
!python colab_runner.py --data-path /content/journal_data index --force

# Method 2: Use the standard launcher (if you're sure you're in the right directory)
!python main.py --data-path /content/journal_data index --force
```

### If you get import errors:
```python
# Check your location and fix if needed
!pwd
%cd /content/SLM_journal  # Navigate to correct directory
!ls  # Verify main.py and src/ are present

# Try the robust launcher
!python colab_runner.py --data-path /content/journal_data index --force
```

## Step 4: Test Queries

```python
# Test some queries
!python main.py --data-path /content/journal_data query "What are my main themes?"
!python main.py --data-path /content/journal_data query "How has my thinking evolved?"
```

## Step 5: Download Results

```python
# Create a zip of the indexed data
!zip -r journal_storage.zip storage/

# Download the storage directory
from google.colab import files
files.download('journal_storage.zip')
```

## Step 6: Use Locally

1. Extract `journal_storage.zip` in your local `SLM_journal/` directory
2. Run queries locally: `python main.py query "your question"`

## Troubleshooting

### If you get import errors:
```python
# Run the setup script again
!python colab_setup.py

# Or manually add paths
import sys
sys.path.append('/content/SLM_journal')
sys.path.append('/content/SLM_journal/src')
```

### If indexing fails:
```python
# Check your data structure
!ls -la /content/journal_data/

# Try with smaller batch size
import os
os.environ['JOURNAL_EMBEDDING__BATCH_SIZE'] = '16'
!python main.py --data-path /content/journal_data index --force
```

### If you run out of memory:
```python
# Use CPU only mode
import os
os.environ['JOURNAL_USE_CPU_ONLY'] = 'true'
os.environ['JOURNAL_EMBEDDING__BATCH_SIZE'] = '8'
!python main.py --data-path /content/journal_data index --force
```

## One-Cell Complete Solution

For convenience, here's a complete solution in one cell:

```python
# Complete Colab setup and indexing
from google.colab import files
import zipfile
import os

# 1. Upload data
print("📁 Upload your journal zip file...")
uploaded = files.upload()

# 2. Extract data
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('/content/journal_data')

# 3. Clone repo and install
!git clone https://github.com/AI-bbos/SLM_journal.git
%cd SLM_journal
!pip install -q -r requirements.txt

# 4. Setup environment
!python colab_setup.py

# 5. Index journals
!python main.py --data-path /content/journal_data index --force

# 6. Test query
!python main.py --data-path /content/journal_data query "What are my main themes?"

# 7. Prepare download
!zip -r journal_storage.zip storage/
print("🎉 Complete! Download journal_storage.zip below.")
files.download('journal_storage.zip')
```