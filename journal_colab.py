#!/usr/bin/env python3
"""
Simple Colab-compatible launcher for the Personal Journal Query System.
This bypasses complex import issues by using a simpler approach.
"""

import sys
import os
from pathlib import Path

# Setup paths - this is the key fix
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir / "src"

# Insert src directory first so absolute imports work
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(current_dir))

# Set environment for Colab
os.environ.setdefault('JOURNAL_LLM_MODEL_TYPE', 'mock')
os.environ.setdefault('JOURNAL_USE_CPU_ONLY', 'false')
os.environ.setdefault('JOURNAL_EMBEDDING__BATCH_SIZE', '32')
os.environ.setdefault('JOURNAL_EMBEDDING__CACHE_SIZE', '5000')

def main():
    """Run the CLI with arguments passed from command line."""
    try:
        from cli.main import main as cli_main
        cli_main()
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("\n📋 Debug info:")
        print(f"Current directory: {Path.cwd()}")
        print(f"Script directory: {current_dir}")
        print(f"Python path: {sys.path[:3]}...")

        # Check if files exist
        key_files = ["src/cli/main.py", "src/application/engine.py"]
        for f in key_files:
            exists = "✅" if Path(f).exists() else "❌"
            print(f"{exists} {f}")

        sys.exit(1)

if __name__ == "__main__":
    main()