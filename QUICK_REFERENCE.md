# FastAPI RAG Chatbot - Quick Reference Guide

## 🚀 Quick Commands

### Setup & Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env

# Build knowledge base
PYTHONPATH=. python scripts/build_knowledge_base.py

# Start server
PYTHONPATH=. uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Run all tests
PYTHONPATH=. pytest tests/ -v

# Test system integration
PYTHONPATH=. python test_system.py

# Health check
curl http://localhost:8000/health
```

## 📡 API Quick Reference

### Ask a Question
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I create a FastAPI application?",
    "max_chunks": 5,
    "temperature": 0.7
  }'
```

### Search Documentation
```bash
curl "http://localhost:8000/search?query=authentication&k=3"
```

### Check System Health
```bash
curl http://localhost:8000/health
```

## 🔧 Configuration Quick Reference

### Environment Variables (.env)
```env
# Required for full functionality
OPENAI_API_KEY=sk-your-openai-api-key

# Optional settings
API_HOST=0.0.0.0
API_PORT=8000
EMBEDDING_MODEL=all-mpnet-base-v2
MAX_CHUNKS=5
DEFAULT_TEMPERATURE=0.7
VECTOR_DB_PATH=./data/vectordb/index
```

### Embedding Models
```env
# High quality (default)
EMBEDDING_MODEL=all-mpnet-base-v2

# Faster/smaller
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Q&A optimized
EMBEDDING_MODEL=multi-qa-mpnet-base-dot-v1
```

## 🐳 Docker Quick Reference

### Build & Run
```bash
# Build image
docker build -t fastapi-rag-chatbot .

# Run container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  fastapi-rag-chatbot

# Run with volume for data persistence
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e OPENAI_API_KEY=your_key \
  fastapi-rag-chatbot
```

## 🔍 Troubleshooting Quick Fixes

### Module Import Errors
```bash
# Always use PYTHONPATH
PYTHONPATH=. python your_script.py
```

### Empty Search Results
```bash
# Rebuild knowledge base
PYTHONPATH=. python scripts/build_knowledge_base.py

# Check vector count
PYTHONPATH=. python -c "
from src.embeddings.vector_store import VectorStore
vs = VectorStore(768)
vs.load_index('./data/vectordb/index')
print(f'Vectors: {vs.index.ntotal}')
"
```

### LLM Not Working
```bash
# Check API key
echo $OPENAI_API_KEY

# Or run in demo mode (search only)
# System automatically falls back if no API key
```

### Memory Issues
```python
# Reduce batch size in build script
embeddings = embedding_service.embed_texts(texts, batch_size=16)

# Or use smaller model
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## 📊 System Status Commands

### Check Components
```bash
# Test embedding service
PYTHONPATH=. python -c "
from src.embeddings.embedding_service import EmbeddingService
service = EmbeddingService()
print('✅ Embedding service OK')
"

# Test vector store
PYTHONPATH=. python -c "
from src.embeddings.vector_store import VectorStore
vs = VectorStore(768)
vs.load_index('./data/vectordb/index')
print(f'✅ Vector store OK: {vs.index.ntotal} vectors')
"

# Test retrieval
PYTHONPATH=. python -c "
from src.retrieval.retriever import AdvancedRetriever
from src.embeddings.embedding_service import EmbeddingService
from src.embeddings.vector_store import VectorStore
service = EmbeddingService()
vs = VectorStore(768)
vs.load_index('./data/vectordb/index')
retriever = AdvancedRetriever(vs, service)
results = retriever.semantic_search('test query', k=1)
print(f'✅ Retrieval OK: {len(results)} results')
"
```

## 🎯 Common Use Cases

### 1. Simple Question
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FastAPI?"}'
```

### 2. Detailed Search
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I handle database connections in FastAPI?",
    "max_chunks": 8,
    "temperature": 0.3
  }'
```

### 3. Browse Documentation
```bash
curl "http://localhost:8000/search?query=dependency%20injection&k=5"
```

### 4. Check System Status
```bash
curl http://localhost:8000/health | python -m json.tool
```

## 📱 Interactive Usage

### Access Web Interface
1. Start server: `PYTHONPATH=. uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
2. Open browser: http://localhost:8000/docs
3. Try the `/ask` endpoint with your questions
4. Browse available endpoints and schemas

### Python Client Example
```python
import requests

# Ask a question
response = requests.post("http://localhost:8000/ask", 
    json={"question": "How do I create FastAPI endpoints?"})
print(response.json()["answer"])

# Search documentation  
response = requests.get("http://localhost:8000/search", 
    params={"query": "authentication", "k": 3})
for result in response.json():
    print(f"- {result['title']}: {result['relevance_score']:.3f}")
```

## 🔐 Production Checklist

### Before Deployment
- [ ] Set secure `OPENAI_API_KEY`
- [ ] Configure proper `API_HOST` and `API_PORT`
- [ ] Build comprehensive knowledge base
- [ ] Run full test suite
- [ ] Set up monitoring and logging
- [ ] Configure rate limiting
- [ ] Set up backup for vector database

### Security
- [ ] Use HTTPS in production
- [ ] Add API key authentication
- [ ] Configure CORS properly
- [ ] Set up input validation
- [ ] Enable request logging

### Performance
- [ ] Use multiple workers: `--workers 4`
- [ ] Configure caching (Redis)
- [ ] Monitor memory usage
- [ ] Set up load balancing
- [ ] Optimize embedding model choice

## 📞 Getting Help

### Resources
- **Interactive API Docs**: http://localhost:8000/docs
- **System Test**: `PYTHONPATH=. python test_system.py`
- **Health Check**: http://localhost:8000/health
- **Full Documentation**: `USER_DOCUMENTATION.md`

### Debug Information
```bash
# Get system info
PYTHONPATH=. python -c "
import sys
print(f'Python: {sys.version}')
from src.config.settings import Settings
settings = Settings()
print(f'Embedding model: {settings.EMBEDDING_MODEL}')
print(f'Vector DB path: {settings.VECTOR_DB_PATH}')
"
```

### Common Commands Summary
```bash
# Complete setup
pip install -r requirements.txt && \
cp .env.example .env && \
PYTHONPATH=. python scripts/build_knowledge_base.py && \
PYTHONPATH=. uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Test everything
PYTHONPATH=. pytest tests/ && \
PYTHONPATH=. python test_system.py && \
curl http://localhost:8000/health

# Reset and rebuild
rm -rf data/vectordb/ && \
PYTHONPATH=. python scripts/build_knowledge_base.py
```
