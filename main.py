#!/usr/bin/env python3
"""
Personal Journal Query System

A RAG-based system for querying personal journal entries using local AI models.
Optimized for Apple Silicon Macs.
"""

import sys
from pathlib import Path

# Add both current directory and src to Python path for compatibility
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(src_path))

try:
    # Try importing from src package first
    from src.cli.main import main
except ImportError:
    # Fallback for when running from src directory
    from cli.main import main

if __name__ == "__main__":
    main()