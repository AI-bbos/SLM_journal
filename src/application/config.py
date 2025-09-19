"""Configuration management using Pydantic."""

from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""
    model_config = ConfigDict(protected_namespaces=())

    model_type: str = "all-MiniLM-L6-v2"
    batch_size: int = 8  # Reduced from 32 to lower memory usage
    cache_size: int = 1000  # Reduced from 10000 to save memory
    use_cache: bool = True


class LLMConfig(BaseModel):
    """Configuration for language models."""
    model_config = ConfigDict(protected_namespaces=())

    model_path: Optional[str] = None
    model_type: str = "mock"  # "local", "ollama", "mock" - start with mock for easy setup
    model_name: str = "phi-2"
    max_tokens: int = 512
    temperature: float = 0.7
    n_ctx: int = 4096
    n_threads: int = 8
    use_metal: bool = True


class StorageConfig(BaseModel):
    """Configuration for storage."""
    data_path: Path = Path("data")
    db_path: Path = Path("storage/metadata.db")
    vector_store_path: Path = Path("storage/vectors")
    index_type: str = "Flat"  # "Flat" or "IVFFlat"


class SearchConfig(BaseModel):
    """Configuration for search and retrieval."""
    k: int = 10
    similarity_threshold: float = 0.3
    max_context_tokens: int = 2048
    diversity_weight: float = 0.2
    recency_weight: float = 0.1
    promote_recent: bool = False
    diversify: bool = True


class IngestionConfig(BaseModel):
    """Configuration for data ingestion."""
    recursive: bool = True
    max_tokens: int = 256  # Reduced from 512 to lower memory usage
    overlap_tokens: int = 25  # Reduced from 50
    min_chunk_size: int = 100
    preserve_sentences: bool = True
    batch_size: int = 50  # Process entries in smaller batches


class Config(BaseSettings):
    """Main configuration class."""

    # Basic settings
    data_path: Path = Field(default=Path("data"), description="Path to journal files")
    storage_path: Path = Field(default=Path("storage"), description="Path to storage directory")

    # Component configurations
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)

    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    # Development
    debug: bool = False
    profile: bool = False

    model_config = ConfigDict(
        env_prefix="JOURNAL_",
        env_file=".env",
        case_sensitive=False,
        protected_namespaces=()
    )

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization processing."""
        # Ensure storage paths are absolute and created
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Update storage config with absolute paths
        self.storage.db_path = self.storage_path / "metadata.db"
        self.storage.vector_store_path = self.storage_path / "vectors"

        # Ensure data path exists
        if not self.data_path.exists():
            self.data_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load_from_file(cls, config_path: Path) -> "Config":
        """Load configuration from a file."""
        if config_path.suffix.lower() == '.json':
            import json
            with open(config_path) as f:
                data = json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            import yaml
            with open(config_path) as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        return cls(**data)

    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to a file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)

        if config_path.suffix.lower() == '.json':
            import json
            with open(config_path, 'w') as f:
                json.dump(self.model_dump(), f, indent=2, default=str)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(self.model_dump(), f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration as dictionary."""
        return self.embedding.model_dump()

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration as dictionary."""
        return self.llm.model_dump()

    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration as dictionary."""
        return self.storage.model_dump()

    def get_search_config(self) -> Dict[str, Any]:
        """Get search configuration as dictionary."""
        return self.search.model_dump()

    def get_ingestion_config(self) -> Dict[str, Any]:
        """Get ingestion configuration as dictionary."""
        return self.ingestion.model_dump()