#!/usr/bin/env python3
"""
Setup script for Google Colab environment.
Run this before using the journal system in Colab.
"""

import sys
import os
from pathlib import Path

def setup_colab_environment():
    """Configure the environment for Colab execution."""

    print("🚀 Setting up Personal Journal Query System for Colab...")

    # Get the current directory
    current_dir = Path.cwd()

    # Check if we're in the right directory
    if not (current_dir / "main.py").exists():
        print("❌ Error: main.py not found. Make sure you're in the SLM_journal directory.")
        return False

    # Add paths to Python path
    src_path = current_dir / "src"

    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
        print(f"✅ Added {current_dir} to Python path")

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        print(f"✅ Added {src_path} to Python path")

    # Set Colab-optimized environment variables
    os.environ['JOURNAL_LLM_MODEL_TYPE'] = 'mock'
    os.environ['JOURNAL_USE_CPU_ONLY'] = 'false'  # Use GPU in Colab
    os.environ['JOURNAL_EMBEDDING__BATCH_SIZE'] = '32'  # Larger batches in Colab
    os.environ['JOURNAL_EMBEDDING__CACHE_SIZE'] = '5000'
    os.environ['JOURNAL_EMBEDDING__USE_CACHE'] = 'true'

    print("✅ Set Colab-optimized environment variables")

    # Check for data directory
    data_dirs = ['/content/journal_data', 'data', '/content/data']
    data_path = None

    for path in data_dirs:
        if Path(path).exists():
            data_path = path
            break

    if data_path:
        print(f"✅ Found data directory: {data_path}")
        os.environ['JOURNAL_DATA_PATH'] = data_path
    else:
        print("⚠️  No data directory found. You'll need to specify --data-path when running commands.")

    # Test import
    try:
        from src.cli.main import main
        print("✅ Successfully imported journal system")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Trying alternative import...")
        try:
            from cli.main import main
            print("✅ Successfully imported journal system (alternative path)")
            return True
        except ImportError as e2:
            print(f"❌ Alternative import also failed: {e2}")
            return False

def main():
    """Main setup function."""
    success = setup_colab_environment()

    if success:
        print("\n🎉 Setup complete! You can now run:")
        print("   python main.py index --force")
        print("   python main.py query 'your question here'")
    else:
        print("\n❌ Setup failed. Please check the errors above.")

    return success

if __name__ == "__main__":
    main()