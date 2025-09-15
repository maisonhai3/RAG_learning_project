import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Gemini Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Model Configuration
    MODEL: str = os.getenv("MODEL", "gpt-3.5-turbo")
    # MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1500"))
    # TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))

    # Vector Database Configuration
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./data/vectordb/index")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")

    # RAG Configuration
    MAX_CHUNKS: int = int(os.getenv("MAX_CHUNKS", "5"))
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    
    # Re-ranking Configuration
    RERANKING_STRATEGY: str = os.getenv("RERANKING_STRATEGY", "disabled")  # disabled, cross_encoder, llm_based, diversity_mmr, hybrid
    RERANKING_MODEL: str = os.getenv("RERANKING_MODEL", "ms-marco-MiniLM-L-6-v2")  # Cross-encoder model name
    MMR_LAMBDA: float = float(os.getenv("MMR_LAMBDA", "0.7"))  # MMR diversity parameter

    # Rate Limiting
    REQUESTS_PER_MINUTE: int = int(os.getenv("REQUESTS_PER_MINUTE", "60"))

    # Data Paths
    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"
    VECTOR_DATA_DIR: str = "data/vectordb"


settings = Settings()
