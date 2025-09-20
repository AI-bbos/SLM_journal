"""Vector storage using FAISS for efficient similarity search."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
import numpy as np
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector storage."""

    @abstractmethod
    def add(self, embeddings: Dict[str, np.ndarray]) -> None:
        """Add embeddings to the store."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the vector store to disk."""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load the vector store from disk."""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store optimized for macOS."""

    def __init__(self, dimension: int, index_type: str = "Flat"):
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS not installed. Install with: pip install faiss-cpu"
            )

        self.dimension = dimension
        self.index_type = index_type

        if index_type == "Flat":
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "IVFFlat":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            self.index.nprobe = 10
        else:
            self.index = faiss.IndexFlatIP(dimension)

        self.id_map = {}
        self.reverse_map = {}
        self.next_idx = 0

    def add(self, embeddings: Dict[str, np.ndarray]) -> None:
        """Add embeddings to the store."""
        if not embeddings:
            return

        vectors = []
        entry_ids = []

        for entry_id, embedding in embeddings.items():
            embedding = embedding.astype(np.float32)

            embedding = embedding / np.linalg.norm(embedding)

            vectors.append(embedding)
            entry_ids.append(entry_id)

        vectors_array = np.array(vectors).astype(np.float32)

        if self.index_type == "IVFFlat" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(vectors_array)

        start_idx = self.next_idx
        self.index.add(vectors_array)

        for i, entry_id in enumerate(entry_ids):
            idx = start_idx + i
            self.id_map[idx] = entry_id
            self.reverse_map[entry_id] = idx

        self.next_idx += len(entry_ids)
        logger.info(f"Added {len(embeddings)} vectors to store (total: {self.next_idx})")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        threshold: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors."""
        if self.index.ntotal == 0:
            return []

        query_embedding = query_embedding.astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1)

        k = min(k, self.index.ntotal)

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            if threshold and dist < threshold:
                continue

            entry_id = self.id_map.get(idx)
            if entry_id:
                results.append((entry_id, float(dist)))

        return results

    def remove(self, entry_ids: List[str]) -> None:
        """Remove entries from the store (requires rebuilding index)."""
        if not entry_ids:
            return

        indices_to_remove = [
            self.reverse_map[entry_id]
            for entry_id in entry_ids
            if entry_id in self.reverse_map
        ]

        if not indices_to_remove:
            return

        logger.warning("Removing vectors requires rebuilding the index")

        all_vectors = []
        all_ids = []

        for i in range(self.index.ntotal):
            if i not in indices_to_remove:
                vector = self.index.reconstruct(i)
                entry_id = self.id_map[i]
                all_vectors.append(vector)
                all_ids.append(entry_id)

        if self.index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            self.index.nprobe = 10

        self.id_map = {}
        self.reverse_map = {}
        self.next_idx = 0

        if all_vectors:
            embeddings = {
                entry_id: vector
                for entry_id, vector in zip(all_ids, all_vectors)
            }
            self.add(embeddings)

    def save(self, path: Path) -> None:
        """Save the vector store to disk."""
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        index_path = path / "index.faiss"
        faiss.write_index(self.index, str(index_path))

        metadata_path = path / "metadata.pkl"
        metadata = {
            'id_map': self.id_map,
            'reverse_map': self.reverse_map,
            'next_idx': self.next_idx,
            'dimension': self.dimension,
            'index_type': self.index_type
        }

        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved vector store to {path}")

    def load(self, path: Path) -> None:
        """Load the vector store from disk."""
        import faiss

        path = Path(path)

        index_path = path / "index.faiss"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))

        metadata_path = path / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            self.id_map = metadata['id_map']
            self.reverse_map = metadata['reverse_map']
            self.next_idx = metadata['next_idx']
            self.dimension = metadata['dimension']
            self.index_type = metadata['index_type']

            logger.info(f"Loaded vector store from {path} ({self.next_idx} vectors)")

    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'memory_usage_mb': self.index.ntotal * self.dimension * 4 / (1024 * 1024)
        }