#!/usr/bin/env python3
"""
Robust Colab runner that handles directory issues automatically.
"""

import sys
import os
from pathlib import Path

def find_project_root():
    """Find the actual project root directory."""
    current = Path.cwd()

    # Look for main.py and src/ directory
    candidates = [
        current,
        current / "SLM_journal",  # In case of nested clone
        current.parent,
        current.parent / "SLM_journal"
    ]

    for candidate in candidates:
        if (candidate / "main.py").exists() and (candidate / "src").is_dir():
            return candidate

    return None

def setup_environment(project_root):
    """Setup environment for the project."""
    os.chdir(project_root)

    # Add paths
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))

    # Set Colab defaults
    defaults = {
        'JOURNAL_LLM_MODEL_TYPE': 'mock',
        'JOURNAL_USE_CPU_ONLY': 'false',
        'JOURNAL_EMBEDDING__BATCH_SIZE': '32',
        'JOURNAL_EMBEDDING__CACHE_SIZE': '5000',
        'JOURNAL_EMBEDDING__USE_CACHE': 'true',
    }

    for key, value in defaults.items():
        os.environ.setdefault(key, value)

def main():
    """Main entry point."""
    print("🔍 Finding project root...")

    project_root = find_project_root()
    if not project_root:
        print("❌ Could not find project root!")
        print("📁 Current directory:", Path.cwd())
        print("📋 Expected files: main.py, src/")
        return False

    print(f"✅ Found project at: {project_root}")

    # Setup environment
    setup_environment(project_root)
    print(f"📍 Working directory: {Path.cwd()}")

    # Import and run
    try:
        from src.cli.main import main as cli_main
        print("🚀 Starting journal system...")
        cli_main()
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")

        # Debug info
        print("\n🔧 Debug info:")
        print(f"Python path: {sys.path[:3]}...")

        key_files = ["src/cli/main.py", "src/storage/__init__.py"]
        for f in key_files:
            exists = "✅" if Path(f).exists() else "❌"
            print(f"{exists} {f}")

        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)