"""Embedding model implementations optimized for Apple Silicon."""

from abc import ABC, abstractmethod
from typing import List, Union, Optional
import numpy as np
from pathlib import Path
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available embedding model types."""
    MINILM = "all-MiniLM-L6-v2"
    MPNET = "all-mpnet-base-v2"
    E5_SMALL = "e5-small-v2"


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get model name."""
        pass


class SentenceTransformerModel(EmbeddingModel):
    """Sentence transformer embedding model optimized for macOS."""

    def __init__(self, model_type: ModelType = ModelType.MINILM, use_cpu_only: bool = False):
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            import os

            # Set thread count based on environment or default
            num_threads = int(os.environ.get('JOURNAL_EMBEDDING_THREADS', '4'))
            torch.set_num_threads(num_threads)

            # Allow forcing CPU mode for memory-constrained environments
            if use_cpu_only or os.environ.get('JOURNAL_USE_CPU_ONLY', 'false').lower() == 'true':
                self.device = 'cpu'
                logger.info(f"Using CPU for inference with {num_threads} threads")
            elif torch.backends.mps.is_available():
                self.device = 'mps'
                logger.info("Using Apple Metal Performance Shaders (MPS) for acceleration")
            else:
                self.device = 'cpu'
                logger.info(f"Using CPU for inference with {num_threads} threads")

            self.model = SentenceTransformer(
                model_type.value,
                device=self.device
            )

            # Reduce max sequence length for memory efficiency
            max_seq_length = int(os.environ.get('JOURNAL_EMBEDDING_MAX_SEQ_LENGTH', '256'))
            self.model.max_seq_length = max_seq_length

            self._model_type = model_type
            self._dimension = self.model.get_sentence_embedding_dimension()

        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_type.value


class CachedEmbeddingModel(EmbeddingModel):
    """Wrapper that adds caching to any embedding model."""

    def __init__(self, base_model: EmbeddingModel, cache_size: int = 10000):
        self.base_model = base_model
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encode texts with caching."""
        if isinstance(texts, str):
            texts = [texts]

        results = []
        texts_to_encode = []
        text_indices = []

        for i, text in enumerate(texts):
            text_hash = hash(text)
            if text_hash in self.cache:
                results.append((i, self.cache[text_hash]))
                self.cache_hits += 1
            else:
                texts_to_encode.append(text)
                text_indices.append(i)
                self.cache_misses += 1

        if texts_to_encode:
            new_embeddings = self.base_model.encode(texts_to_encode, batch_size)

            for text, embedding, idx in zip(texts_to_encode, new_embeddings, text_indices):
                text_hash = hash(text)

                if len(self.cache) >= self.cache_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]

                self.cache[text_hash] = embedding
                results.append((idx, embedding))

        results.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in results])

    def get_dimension(self) -> int:
        return self.base_model.get_dimension()

    @property
    def model_name(self) -> str:
        return f"cached_{self.base_model.model_name}"

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0

        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate
        }


class EmbeddingModelFactory:
    """Factory for creating embedding models."""

    @staticmethod
    def create(
        model_type: Union[str, ModelType] = ModelType.MINILM,
        use_cache: bool = True,
        cache_size: int = 10000,
        use_cpu_only: bool = False
    ) -> EmbeddingModel:
        """Create an embedding model."""
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type)
            except ValueError:
                model_type = ModelType.MINILM
                logger.warning(f"Unknown model type, using {model_type.value}")

        base_model = SentenceTransformerModel(model_type, use_cpu_only=use_cpu_only)

        if use_cache:
            return CachedEmbeddingModel(base_model, cache_size)

        return base_model