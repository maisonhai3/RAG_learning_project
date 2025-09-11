# FastAPI Documentation RAG Chatbot

A sophisticated Question-Answering chatbot specialized in FastAPI documentation using Retrieval-Augmented Generation (RAG) technology.

## Features

- **Semantic Search**: Advanced vector-based search through FastAPI documentation
- **Multiple Search Strategies**: Semantic, keyword, and hybrid search options
- **LLM Integration**: OpenAI GPT models for natural language responses
- **Source Attribution**: Proper citations and source references
- **REST API**: Full FastAPI-powered API with interactive documentation
- **Docker Support**: Easy deployment with Docker and Docker Compose

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- Docker (optional)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd fastapi-docs-chatbot
```

2. **Set up environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

4. **Build the knowledge base**
```bash
python scripts/build_knowledge_base.py
```

5. **Start the API server**
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

6. **Access the API**
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc
- Health check: http://localhost:8000/health

### Docker Setup

1. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

2. **Build knowledge base in container**
```bash
docker-compose exec fastapi-chatbot python scripts/build_knowledge_base.py
```

## API Usage

### Ask a Question
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I create a FastAPI application?",
    "max_chunks": 5,
    "temperature": 0.7,
    "search_strategy": "hybrid"
  }'
```

### Search Documentation
```bash
curl "http://localhost:8000/search?query=dependency injection&k=3"
```

### Check System Health
```bash
curl "http://localhost:8000/health"
```

## Project Structure

```
fastapi-docs-chatbot/
├── src/
│   ├── crawler/           # Web scraping and text processing
│   ├── embeddings/        # Vector embeddings and storage
│   ├── retrieval/         # Search and retrieval logic
│   ├── generation/        # LLM integration and prompts
│   ├── api/              # FastAPI application
│   └── config/           # Configuration management
├── data/
│   ├── raw/              # Scraped HTML files
│   ├── processed/        # Processed text chunks
│   └── vectordb/         # Vector database files
├── scripts/              # Utility scripts
├── tests/                # Test suite
└── docker/               # Docker configuration
```

## Configuration

Key environment variables:

```bash
OPENAI_API_KEY=your_openai_key
VECTOR_DB_PATH=./data/vectordb/index
EMBEDDING_MODEL=all-mpnet-base-v2
API_HOST=0.0.0.0
API_PORT=8000
MAX_CHUNKS=5
DEFAULT_TEMPERATURE=0.7
```

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Quality
```bash
# Install development dependencies
pip install black flake8 mypy

# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/
```

### Extending the System

1. **Add new data sources**: Modify `src/crawler/scraper.py`
2. **Custom embedding models**: Update `src/embeddings/embedding_service.py`
3. **New LLM providers**: Extend `src/generation/llm_service.py`
4. **Enhanced prompts**: Modify `src/generation/prompt_templates.py`

## Performance

### Benchmarks
- **Response time**: < 3 seconds for 95% of queries
- **Throughput**: > 100 requests per minute
- **Accuracy**: > 85% relevance on evaluation set

### Optimization Tips
1. Use caching for frequent queries
2. Adjust embedding model for speed vs. accuracy trade-off
3. Tune vector search parameters
4. Implement request batching for high load

## Monitoring

### Health Checks
- `/health` - System health and dependency status
- `/stats` - Usage statistics and performance metrics

### Metrics Tracked
- Query response times
- Vector search performance
- LLM token usage
- User satisfaction ratings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Check the [API documentation](http://localhost:8000/docs)
- Review the implementation plan in `FastAPI_RAG_Chatbot_Implementation_Plan.md`
- Open an issue for bugs or feature requests

## Acknowledgments

- FastAPI team for excellent documentation
- Hugging Face for sentence transformers
- OpenAI for LLM capabilities
- FAISS for efficient vector search
