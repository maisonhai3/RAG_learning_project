from pydantic import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application configuration settings."""
    
    # Environment
    environment: str = "development"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 1500
    openai_temperature: float = 0.7
    
    # Embedding Configuration
    embedding_model: str = "all-mpnet-base-v2"
    embedding_dimension: int = 768
    max_chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Vector Database Configuration
    vector_db_type: str = "faiss"
    vector_db_path: str = "./data/vectordb"
    index_type: str = "IVFFlat"
    
    # Retrieval Configuration
    max_retrieval_chunks: int = 5
    similarity_threshold: float = 0.7
    search_strategy: str = "hybrid"
    
    # Rate Limiting
    requests_per_minute: int = 60
    burst_limit: int = 10
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Crawling Configuration
    fastapi_docs_url: str = "https://fastapi.tiangolo.com"
    max_concurrent_requests: int = 5
    request_delay: float = 1.0
    respect_robots_txt: bool = True
    
    # Cache Configuration
    redis_url: Optional[str] = None
    cache_ttl: int = 3600
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
