"""Main query engine that orchestrates all components."""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

from src.domain.models import JournalEntry, QueryResult, SearchResult
from src.application.config import Config
from src.ingestion import DataIngester, TextPreprocessor, ChunkConfig
from src.embedding import EmbeddingService, EmbeddingModelFactory
from src.storage import FAISSVectorStore, MetadataStore
from src.retrieval import SemanticSearcher, RankingService, ContextBuilder
from src.generation import LocalLLM, OllamaLLM, MockLLM, ResponseGenerator

logger = logging.getLogger(__name__)


class QueryEngine:
    """Main engine for querying journal entries."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._setup_logging()

        # Initialize components
        self.embedding_service: Optional[EmbeddingService] = None
        self.vector_store: Optional[FAISSVectorStore] = None
        self.metadata_store: Optional[MetadataStore] = None
        self.searcher: Optional[SemanticSearcher] = None
        self.ranker: Optional[RankingService] = None
        self.generator: Optional[ResponseGenerator] = None

        self._initialized = False

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=self.config.log_file
        )

    def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        logger.info("Initializing query engine...")

        # Initialize embedding service
        embedding_model = EmbeddingModelFactory.create(
            model_type=self.config.embedding.model_type,
            use_cache=self.config.embedding.use_cache,
            cache_size=self.config.embedding.cache_size
        )
        self.embedding_service = EmbeddingService(
            model=embedding_model,
            batch_size=self.config.embedding.batch_size
        )

        # Initialize storage
        self.vector_store = FAISSVectorStore(
            dimension=self.embedding_service.dimension,
            index_type=self.config.storage.index_type
        )
        self.metadata_store = MetadataStore(self.config.storage.db_path)

        # Try to load existing vector store
        if self.config.storage.vector_store_path.exists():
            try:
                self.vector_store.load(self.config.storage.vector_store_path)
                logger.info("Loaded existing vector store")
            except Exception as e:
                logger.warning(f"Could not load vector store: {e}")

        # Initialize search and ranking
        self.searcher = SemanticSearcher(
            self.embedding_service,
            self.vector_store,
            self.metadata_store
        )
        self.ranker = RankingService(
            diversity_weight=self.config.search.diversity_weight,
            recency_weight=self.config.search.recency_weight
        )

        # Initialize language model
        llm = self._create_llm()
        context_builder = ContextBuilder(
            max_tokens=self.config.search.max_context_tokens
        )
        self.generator = ResponseGenerator(llm, context_builder)

        self._initialized = True
        logger.info("Query engine initialized successfully")

    def _create_llm(self):
        """Create language model based on configuration."""
        if self.config.llm.model_type == "mock":
            return MockLLM()
        elif self.config.llm.model_type == "local":
            return LocalLLM(
                model_path=self.config.llm.model_path,
                n_ctx=self.config.llm.n_ctx,
                n_threads=self.config.llm.n_threads,
                use_metal=self.config.llm.use_metal
            )
        elif self.config.llm.model_type == "ollama":
            return OllamaLLM(model_name=self.config.llm.model_name)
        else:
            raise ValueError(f"Unknown LLM type: {self.config.llm.model_type}")

    def ingest_data(self, force_rebuild: bool = False, batch_size: int = 10) -> Dict[str, Any]:
        """Ingest journal data and build indexes in batches to reduce memory usage.

        Args:
            force_rebuild: Whether to rebuild existing indexes
            batch_size: Number of entries to process at once (default 100)
        """
        if not self._initialized:
            self.initialize()

        logger.info("Starting data ingestion...")

        # Check if we need to rebuild
        stats = self.metadata_store.get_statistics()
        if stats['total_entries'] > 0 and not force_rebuild:
            logger.info(f"Found {stats['total_entries']} existing entries. Use force_rebuild=True to rebuild.")
            return stats

        # Create ingester
        chunk_config = ChunkConfig(
            max_tokens=self.config.ingestion.max_tokens,
            overlap_tokens=self.config.ingestion.overlap_tokens,
            min_chunk_size=self.config.ingestion.min_chunk_size,
            preserve_sentences=self.config.ingestion.preserve_sentences
        )

        ingester = DataIngester(
            data_path=self.config.data_path,
            chunk_config=chunk_config,
            recursive=self.config.ingestion.recursive
        )

        # Clear existing data if rebuilding
        if force_rebuild:
            self.metadata_store.clear_all()
            self.vector_store = FAISSVectorStore(
                dimension=self.embedding_service.dimension,
                index_type=self.config.storage.index_type
            )

        # Process entries one by one to minimize memory usage
        total_entries = 0
        entry_batch = []

        # Use streaming with ultra-small batches
        import gc

        for entry in ingester.ingest_entries():
            entry_batch.append(entry)

            # Process very small batches
            if len(entry_batch) >= batch_size:
                try:
                    # Generate embeddings for this tiny batch with minimal batch size
                    batch_embeddings = self.embedding_service.embed_entries(
                        entry_batch,
                        show_progress=False
                    )

                    # Store batch data
                    self.metadata_store.add_entries(entry_batch)
                    self.vector_store.add(batch_embeddings)

                    total_entries += len(entry_batch)

                    # More frequent progress updates
                    if total_entries % 50 == 0:
                        logger.info(f"Processed {total_entries} entries...")

                    # Aggressively clear memory
                    del entry_batch
                    del batch_embeddings
                    entry_batch = []

                    # Force multiple garbage collections
                    gc.collect()
                    gc.collect()  # Sometimes need multiple calls

                except Exception as e:
                    logger.error(f"Error processing batch at entry {total_entries}: {e}")
                    # Continue with next batch on error
                    entry_batch = []
                    gc.collect()
                    continue

        # Process remaining entries
        if entry_batch:
            try:
                batch_embeddings = self.embedding_service.embed_entries(entry_batch, show_progress=False)
                self.metadata_store.add_entries(entry_batch)
                self.vector_store.add(batch_embeddings)
                total_entries += len(entry_batch)
                del entry_batch
                del batch_embeddings
                gc.collect()
            except Exception as e:
                logger.error(f"Error processing final batch: {e}")

        if total_entries == 0:
            logger.warning("No entries found in data path")
            return {"total_entries": 0}

        # Save vector store
        self.vector_store.save(self.config.storage.vector_store_path)

        # Final cleanup before creating searcher
        gc.collect()

        # Update searcher
        try:
            self.searcher = SemanticSearcher(
                self.embedding_service,
                self.vector_store,
                self.metadata_store
            )
        except Exception as e:
            logger.error(f"Error creating searcher: {e}")
            # Continue without searcher for now
            self.searcher = None

        stats = self.metadata_store.get_statistics()
        logger.info(f"Ingestion complete: {stats['total_entries']} entries")

        return stats

    def query(
        self,
        query: str,
        k: Optional[int] = None,
        date_filter: Optional[Tuple[datetime, datetime]] = None,
        response_type: str = "question_answering",
        **kwargs
    ) -> QueryResult:
        """
        Query the journal entries.

        Args:
            query: The search query
            k: Number of results to return
            date_filter: Optional date range filter
            response_type: Type of response to generate
            **kwargs: Additional parameters

        Returns:
            Query result with response and sources
        """
        if not self._initialized:
            self.initialize()

        k = k or self.config.search.k

        # Search for relevant entries
        search_results = self.searcher.search(
            query,
            k=k,
            date_filter=date_filter,
            similarity_threshold=self.config.search.similarity_threshold
        )

        if not search_results:
            return QueryResult(
                query=query,
                response="No relevant entries found for your query.",
                sources=[],
                context_used="",
                processing_time=0.0,
                model_used=self.generator.llm.get_model_info()['model_name']
            )

        # Re-rank results
        search_results = self.ranker.rerank(
            search_results,
            query,
            promote_recent=self.config.search.promote_recent,
            diversify=self.config.search.diversify
        )

        # Generate response
        result = self.generator.generate_response(
            query,
            search_results,
            response_type=response_type,
            **kwargs
        )

        # Save query history
        source_ids = [r.entry.id for r in result.sources]
        self.metadata_store.save_query_history(
            query=query,
            response=result.response,
            sources=source_ids,
            processing_time=result.processing_time,
            model_used=result.model_used
        )

        return result

    def search_by_date(
        self,
        start_date: datetime,
        end_date: datetime,
        query: Optional[str] = None,
        k: int = 20
    ) -> List[SearchResult]:
        """Search entries by date range."""
        if not self._initialized:
            self.initialize()

        return self.searcher.search_by_date_range(start_date, end_date, query, k)

    def find_similar(self, entry_id: str, k: int = 10) -> List[SearchResult]:
        """Find entries similar to a given entry."""
        if not self._initialized:
            self.initialize()

        return self.searcher.search_similar_to_entry(entry_id, k)

    def summarize_period(
        self,
        start_date: datetime,
        end_date: datetime,
        focus: Optional[str] = None
    ) -> QueryResult:
        """Summarize entries from a specific time period."""
        search_results = self.search_by_date(start_date, end_date)

        period_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

        return self.generator.generate_summary(
            search_results,
            focus=focus,
            time_period=period_str
        )

    def analyze_emotions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> QueryResult:
        """Analyze emotions in journal entries."""
        if start_date and end_date:
            search_results = self.search_by_date(start_date, end_date)
        else:
            # Get recent entries
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            search_results = self.search_by_date(start_date, end_date)

        return self.generator.analyze_emotions(search_results)

    def track_goal(self, goal: str, days_back: int = 90) -> QueryResult:
        """Track progress on a specific goal."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        search_results = self.searcher.search(
            goal,
            k=20,
            date_filter=(start_date, end_date)
        )

        return self.generator.track_goal_progress(goal, search_results)

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        if not self._initialized:
            self.initialize()

        stats = {
            "metadata": self.metadata_store.get_statistics(),
            "vector_store": self.vector_store.get_stats(),
            "embedding_model": self.embedding_service.get_model_info(),
            "llm_model": self.generator.llm.get_model_info()
        }

        return stats

    def reload_data(self) -> Dict[str, Any]:
        """Reload data from disk."""
        return self.ingest_data(force_rebuild=True)