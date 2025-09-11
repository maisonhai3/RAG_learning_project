# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Build knowledge base (required before first run)
PYTHONPATH=. python scripts/build_knowledge_base.py

# Start development server
PYTHONPATH=. uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Testing
```bash
# Run test suite
PYTHONPATH=. pytest tests/ -v

# Run system integration test
PYTHONPATH=. python test_system.py

# Health check
curl http://localhost:8000/health
```

### Code Quality (when available)
```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/
```

## Architecture Overview

This is a **FastAPI Documentation RAG (Retrieval-Augmented Generation) Chatbot** with the following key components:

### Core Pipeline Architecture
```
FastAPI Documentation → Web Scraping → Text Processing → Vector Embeddings → 
Semantic Search → LLM Generation → API Response
```

### Module Structure
- **`src/crawler/`** - Web scraping and document collection
  - `scraper.py` - Async FastAPI documentation scraper
  - `text_processor.py` - HTML parsing and text chunking
- **`src/embeddings/`** - Vector embeddings and storage
  - `embedding_service.py` - Sentence transformers integration
  - `vector_store.py` - FAISS vector database management
- **`src/retrieval/`** - Search and retrieval logic
  - `retriever.py` - Semantic, keyword, and hybrid search
- **`src/generation/`** - LLM integration
  - `llm_service.py` - OpenAI GPT integration
  - `prompt_templates.py` - Prompt engineering templates
- **`src/api/`** - FastAPI REST API
  - `main.py` - Main application with RAGService orchestration
  - `models.py` - Pydantic request/response models
- **`src/config/`** - Configuration management
  - `settings.py` - Environment-based configuration

### Data Flow
1. **Knowledge Base Building** (`scripts/build_knowledge_base.py`):
   - Crawls FastAPI documentation
   - Processes HTML and chunks text
   - Generates embeddings using sentence-transformers
   - Stores in FAISS vector database
2. **Query Processing** (RAGService in `src/api/main.py`):
   - Embeds user question
   - Performs semantic search in vector store
   - Builds context-aware prompts
   - Generates responses via OpenAI API

### Key Dependencies
- **FastAPI** - Modern async web framework
- **sentence-transformers** - Text embeddings (default: all-mpnet-base-v2)
- **FAISS** - Efficient vector similarity search
- **OpenAI** - LLM for response generation
- **BeautifulSoup4** - HTML parsing for web scraping
- **asyncio/aiohttp** - Async web requests

## Configuration

### Environment Variables (.env)
```env
# Required for full LLM functionality
OPENAI_API_KEY=sk-your-openai-api-key

# Optional configuration
API_HOST=0.0.0.0
API_PORT=8000
EMBEDDING_MODEL=all-mpnet-base-v2
VECTOR_DB_PATH=./data/vectordb/index
MAX_CHUNKS=5
DEFAULT_TEMPERATURE=0.7
```

### Important Notes
- **PYTHONPATH Required**: Always use `PYTHONPATH=. python script.py` for module imports
- **Knowledge Base First**: Must run `build_knowledge_base.py` before starting the API
- **Demo Mode**: System works without OpenAI API key (search-only mode)
- **Vector Database**: Stored as FAISS index files in `data/vectordb/`

## Development Patterns

### RAG Service Integration
The main application uses a centralized `RAGService` class that orchestrates all components. When adding features:
1. Extend the service in `src/api/main.py`
2. Add new endpoints that use the service
3. Update request/response models in `src/api/models.py`

### Search Strategies
The system supports multiple search approaches:
- **Semantic**: Pure vector similarity search
- **Keyword**: Traditional text matching
- **Hybrid**: Combined approach (default)

### Error Handling
- Graceful degradation when OpenAI API unavailable
- Health check endpoint monitors all components
- Comprehensive logging throughout pipeline

## Docker Usage

```bash
# Build and run with Docker Compose
docker-compose up --build

# Build knowledge base in container
docker-compose exec fastapi-chatbot python scripts/build_knowledge_base.py
```

## Project Context

This is a production-ready RAG system demonstrating modern NLP and API development practices. The codebase follows FastAPI best practices with async/await patterns, proper dependency injection, and comprehensive error handling. The system is designed for horizontal scaling and can serve as a foundation for similar document Q&A applications.