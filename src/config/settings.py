import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Vector Database Configuration
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./data/vectordb/index")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
    
    # RAG Configuration
    MAX_CHUNKS: int = int(os.getenv("MAX_CHUNKS", "5"))
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    
    # Rate Limiting
    REQUESTS_PER_MINUTE: int = int(os.getenv("REQUESTS_PER_MINUTE", "60"))
    
    # Data Paths
    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"
    VECTOR_DATA_DIR: str = "data/vectordb"


settings = Settings()
