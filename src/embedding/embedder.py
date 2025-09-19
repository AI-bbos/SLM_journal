"""Main embedding service for generating and managing embeddings."""

from typing import List, Dict, Optional, Union
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm

from src.domain.models import JournalEntry
from src.embedding.models import EmbeddingModel, EmbeddingModelFactory, ModelType

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings from journal entries."""

    def __init__(
        self,
        model: Optional[EmbeddingModel] = None,
        batch_size: int = 32
    ):
        self.model = model or EmbeddingModelFactory.create(ModelType.MINILM)
        self.batch_size = batch_size
        self.dimension = self.model.get_dimension()

    def embed_entries(
        self,
        entries: List[JournalEntry],
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for journal entries with minimal memory usage.

        Returns:
            Dictionary mapping entry IDs to embeddings
        """
        embeddings = {}
        import gc

        # Use very small batch size to minimize memory usage
        ultra_small_batch = min(self.batch_size, 2)  # Never more than 2 at once

        if show_progress:
            pbar = tqdm(total=len(entries), desc="Generating embeddings")

        # Process entries in ultra-small batches
        for i in range(0, len(entries), ultra_small_batch):
            batch_entries = entries[i:i + ultra_small_batch]

            batch_texts = []
            batch_ids = []

            for entry in batch_entries:
                text = self._prepare_entry_text(entry)
                batch_texts.append(text)
                batch_ids.append(entry.id)

            try:
                # Generate embeddings with smallest possible batch
                batch_embeddings = self.model.encode(batch_texts, len(batch_texts))

                for entry_id, embedding in zip(batch_ids, batch_embeddings):
                    embeddings[entry_id] = embedding

                # Aggressively clean up
                del batch_texts
                del batch_ids
                del batch_embeddings
                del batch_entries

                if show_progress:
                    pbar.update(len(batch_entries))

                # Force garbage collection every few iterations
                if i % 10 == 0:
                    gc.collect()

            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i}: {e}")
                # Continue with next batch
                continue

        if show_progress:
            pbar.close()

        # Final cleanup
        gc.collect()

        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query."""
        query_text = self._prepare_query_text(query)
        return self.model.encode(query_text, batch_size=1)[0]

    def _prepare_entry_text(self, entry: JournalEntry) -> str:
        """Prepare journal entry text for embedding."""
        components = []

        if entry.title:
            components.append(f"Title: {entry.title}")

        components.append(entry.content)

        if entry.tags:
            components.append(f"Tags: {', '.join(entry.tags)}")

        date_str = entry.date.strftime("%B %d, %Y")
        components.append(f"Date: {date_str}")

        return "\n".join(components)

    def _prepare_query_text(self, query: str) -> str:
        """Prepare query text for embedding."""
        query = query.strip()

        if not any(query.endswith(p) for p in ['.', '?', '!']):
            query += '.'

        return query

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        entry_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute cosine similarity between query and entries.

        Returns:
            Dictionary mapping entry IDs to similarity scores
        """
        similarities = {}

        query_norm = query_embedding / np.linalg.norm(query_embedding)

        for entry_id, entry_embedding in entry_embeddings.items():
            entry_norm = entry_embedding / np.linalg.norm(entry_embedding)
            similarity = np.dot(query_norm, entry_norm)
            similarities[entry_id] = float(similarity)

        return similarities

    def batch_compute_similarity(
        self,
        query_embedding: np.ndarray,
        entry_embeddings: np.ndarray,
        entry_ids: List[str]
    ) -> Dict[str, float]:
        """
        Efficiently compute similarity for many embeddings at once.

        Args:
            query_embedding: Query embedding vector
            entry_embeddings: Matrix of entry embeddings
            entry_ids: List of entry IDs corresponding to embeddings

        Returns:
            Dictionary mapping entry IDs to similarity scores
        """
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        entry_norms = entry_embeddings / np.linalg.norm(
            entry_embeddings, axis=1, keepdims=True
        )

        similarities = np.dot(entry_norms, query_norm)

        return {
            entry_id: float(score)
            for entry_id, score in zip(entry_ids, similarities)
        }

    def get_model_info(self) -> dict:
        """Get information about the embedding model."""
        info = {
            "model_name": self.model.model_name,
            "dimension": self.dimension,
            "batch_size": self.batch_size
        }

        if hasattr(self.model, 'get_cache_stats'):
            info["cache_stats"] = self.model.get_cache_stats()

        return info