#!/usr/bin/env python3
"""
Demo script for the Personal Journal Query System
Uses mock LLM to avoid model downloads
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.application.engine import QueryEngine
from src.application.config import Config


def main():
    """Run a simple demo."""
    print("🔍 Personal Journal Query System Demo")
    print("=" * 50)
    print("\n💡 Note: Using mock AI model for instant demo.")
    print("   For better responses, run: python main.py download-model")

    # Configure for demo with mock LLM
    config = Config()
    config.data_path = Path("data/examples")
    config.llm.model_type = "mock"
    config.storage_path = Path("demo_storage")

    # Create engine
    engine = QueryEngine(config)

    try:
        print("\n📚 Indexing journal entries...")
        stats = engine.ingest_data(force_rebuild=True)

        print(f"✅ Indexed {stats['total_entries']} entries")
        if 'date_range' in stats and stats['date_range']['earliest']:
            print(f"📅 Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")

        print("\n🤔 Running sample queries...")

        # Sample queries
        queries = [
            "What are my thoughts on meditation?",
            "How am I doing with work stress?",
            "What books am I reading?",
        ]

        for query in queries:
            print(f"\n❓ Query: {query}")
            result = engine.query(query, k=3)
            print(f"💭 Response: {result.response}")

            if result.sources:
                print(f"📖 Sources ({len(result.sources)}):")
                for i, source in enumerate(result.sources[:2], 1):
                    date_str = source.entry.date.strftime("%Y-%m-%d")
                    print(f"   {i}. [{date_str}] {source.entry.title or 'Untitled'}")

        print(f"\n📊 System Stats:")
        system_stats = engine.get_statistics()
        print(f"   • Total entries: {system_stats['metadata']['total_entries']}")
        print(f"   • Vector dimension: {system_stats['embedding_model']['dimension']}")
        print(f"   • Model: {system_stats['llm_model']['model_name']}")

        print("\n✨ Demo complete! The system successfully:")
        print("   • Parsed markdown journal files")
        print("   • Generated semantic embeddings")
        print("   • Built searchable vector index")
        print("   • Performed natural language queries")
        print("   • Generated contextual responses")

        print(f"\n🎯 To try with your own journals:")
        print(f"   1. Place journal files in data/ directory")
        print(f"   2. Run: python main.py index")
        print(f"   3. Query: python main.py query \"your question\"")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())