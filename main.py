#!/usr/bin/env python3
"""
Personal Journal Query System

A RAG-based system for querying personal journal entries using local AI models.
Optimized for Apple Silicon Macs.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.cli.main import main

if __name__ == "__main__":
    main()