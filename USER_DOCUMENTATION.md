# FastAPI RAG Chatbot - User Documentation

## 📖 Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [System Overview](#system-overview)
3. [Installation & Setup](#installation--setup)
4. [Building the Knowledge Base](#building-the-knowledge-base)
5. [Using the API](#using-the-api)
6. [API Endpoints Reference](#api-endpoints-reference)
7. [Configuration Options](#configuration-options)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Usage](#advanced-usage)
10. [Examples & Use Cases](#examples--use-cases)

---

## Quick Start Guide

### Prerequisites
- Python 3.11 or higher
- 8GB+ RAM recommended
- Internet connection for initial setup
- OpenAI API key (optional, for full chat functionality)

### 5-Minute Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment (optional)
cp .env.example .env
# Edit .env to add your OpenAI API key if desired

# 3. Build the knowledge base
PYTHONPATH=. python scripts/build_knowledge_base.py

# 4. Start the API server
PYTHONPATH=. uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 5. Open your browser to http://localhost:8000/docs
```

You're now ready to use the FastAPI documentation chatbot!

---

## System Overview

### What is this system?

The FastAPI RAG Chatbot is a Retrieval-Augmented Generation system that:

- **Crawls** FastAPI's official documentation
- **Processes** the content into searchable chunks
- **Embeds** text using AI models for semantic understanding
- **Retrieves** relevant information based on your questions
- **Generates** contextual answers using language models

### Key Components

```
User Question → Semantic Search → Context Retrieval → LLM Generation → Answer
```

1. **Knowledge Base**: FastAPI documentation stored as embeddings
2. **Retrieval Engine**: Finds relevant documentation sections
3. **Language Model**: Generates human-like answers
4. **API Server**: FastAPI web service for easy integration

---

## Installation & Setup

### Method 1: Local Installation

```bash
# Clone the repository
git clone <your-repository-url>
cd fastapi-docs-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### Method 2: Docker Installation

```bash
# Build Docker image
docker build -t fastapi-rag-chatbot .

# Run container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_api_key_here \
  fastapi-rag-chatbot
```

### Environment Configuration

Edit the `.env` file with your preferences:

```env
# OpenAI API (required for full chat functionality)
OPENAI_API_KEY=sk-your-openai-api-key-here

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# RAG Settings
EMBEDDING_MODEL=all-mpnet-base-v2
MAX_CHUNKS=5
DEFAULT_TEMPERATURE=0.7

# Database Settings
VECTOR_DB_PATH=./data/vectordb/index
```

---

## Building the Knowledge Base

### Automatic Build (Recommended)

```bash
# Run the complete build pipeline
PYTHONPATH=. python scripts/build_knowledge_base.py
```

This will:
1. Crawl FastAPI documentation (50+ pages)
2. Process and chunk the text content
3. Generate embeddings using AI models
4. Build a searchable vector database

### Build Output

```
🚀 Starting FastAPI Documentation Knowledge Base Build...

📥 Step 1: Crawling FastAPI documentation...
✅ Crawled 50 pages.

📝 Step 2: Processing and chunking text...
✅ Generated 50 text chunks.

🧠 Step 3: Generating embeddings and building vector store...
✅ Vector store built with 50 embeddings.
💾 Saved to: ./data/vectordb/index

🎉 Knowledge base build complete!
```

### Manual Build Steps

If you want to customize the process:

```bash
# Step 1: Crawl documentation only
PYTHONPATH=. python -c "
from src.crawler.scraper import FastAPIScraper
import asyncio
scraper = FastAPIScraper()
asyncio.run(scraper.crawl(max_pages=100))
"

# Step 2: Process text manually
PYTHONPATH=. python -c "
from src.crawler.text_processor import TextProcessor
processor = TextProcessor()
# ... custom processing logic
"

# Step 3: Build embeddings
PYTHONPATH=. python -c "
from src.embeddings.embedding_service import EmbeddingService
service = EmbeddingService()
# ... custom embedding logic
"
```

---

## Using the API

### Starting the Server

```bash
# Development mode (with auto-reload)
PYTHONPATH=. uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
PYTHONPATH=. uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Accessing the API

- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### Basic Usage Examples

#### 1. Ask a Question (Full Chat Mode)

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I create a FastAPI application?",
    "max_chunks": 5,
    "temperature": 0.7
  }'
```

**Response:**
```json
{
  "answer": "To create a FastAPI application, you need to import FastAPI and create an instance. Here's a basic example:\n\n```python\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get(\"/\")\ndef read_root():\n    return {\"Hello\": \"World\"}\n```\n\nThis creates a simple FastAPI application with a root endpoint that returns a JSON response.",
  "sources": [
    {
      "title": "FastAPI Documentation - First Steps",
      "url": "https://fastapi.tiangolo.com/tutorial/first-steps/",
      "relevance_score": 0.89,
      "excerpt": "Create a file main.py with: from fastapi import FastAPI..."
    }
  ],
  "confidence_score": 0.85,
  "processing_time": 1.2,
  "tokens_used": 150,
  "search_results_count": 3
}
```

#### 2. Search Documentation (Demo Mode)

```bash
curl -X GET "http://localhost:8000/search?query=authentication&k=3"
```

**Response:**
```json
[
  {
    "title": "Security and Authentication",
    "url": "https://fastapi.tiangolo.com/tutorial/security/",
    "relevance_score": 0.92,
    "excerpt": "FastAPI provides several tools to help you deal with security easily, rapidly, in a standard way..."
  },
  {
    "title": "OAuth2 with Password",
    "url": "https://fastapi.tiangolo.com/tutorial/security/simple-oauth2/",
    "relevance_score": 0.87,
    "excerpt": "Now let's build from the previous chapter and add the missing parts to have a complete security flow..."
  }
]
```

#### 3. Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-11T10:30:00Z",
  "version": "1.0.0",
  "dependencies": {
    "embedding_service": "operational",
    "vector_store": "loaded",
    "llm_service": "available"
  }
}
```

---

## API Endpoints Reference

### POST `/ask` - Main Chat Endpoint

Ask questions and get AI-generated answers with sources.

**Request Body:**
```json
{
  "question": "string (required)",
  "max_chunks": "integer (optional, default: 5)",
  "temperature": "float (optional, default: 0.7)",
  "include_sources": "boolean (optional, default: true)"
}
```

**Response:**
```json
{
  "answer": "string",
  "sources": ["array of source objects"],
  "confidence_score": "float",
  "processing_time": "float",
  "tokens_used": "integer",
  "search_results_count": "integer"
}
```

### GET `/search` - Document Search

Search documentation without AI generation.

**Query Parameters:**
- `query` (required): Search query string
- `k` (optional, default: 5): Number of results to return

**Response:**
```json
[
  {
    "title": "string",
    "url": "string", 
    "relevance_score": "float",
    "excerpt": "string"
  }
]
```

### GET `/health` - Health Check

Check system status and component health.

**Response:**
```json
{
  "status": "healthy|degraded|unhealthy",
  "timestamp": "ISO datetime",
  "version": "string",
  "dependencies": {
    "component_name": "status"
  }
}
```

### POST `/feedback` - User Feedback

Submit feedback to improve the system.

**Request Body:**
```json
{
  "question": "string",
  "answer": "string", 
  "rating": "integer (1-5)",
  "feedback": "string (optional)"
}
```

### GET `/stats` - Usage Statistics

Get system usage and performance metrics.

**Response:**
```json
{
  "total_queries": "integer",
  "avg_response_time": "float",
  "knowledge_base_size": "integer",
  "uptime": "string"
}
```

---

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | None | OpenAI API key for chat functionality |
| `API_HOST` | 0.0.0.0 | API server host |
| `API_PORT` | 8000 | API server port |
| `EMBEDDING_MODEL` | all-mpnet-base-v2 | Sentence transformer model |
| `MAX_CHUNKS` | 5 | Maximum context chunks for LLM |
| `DEFAULT_TEMPERATURE` | 0.7 | LLM temperature setting |
| `VECTOR_DB_PATH` | ./data/vectordb/index | Vector database path |
| `REQUESTS_PER_MINUTE` | 60 | Rate limiting setting |

### Embedding Models

You can use different embedding models by changing the `EMBEDDING_MODEL` setting:

```env
# High quality, larger size
EMBEDDING_MODEL=all-mpnet-base-v2

# Faster, smaller size  
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Optimized for Q&A
EMBEDDING_MODEL=multi-qa-mpnet-base-dot-v1
```

### LLM Providers

Currently supports OpenAI GPT models. Configure in `.env`:

```env
# GPT-3.5 (faster, cheaper)
LLM_MODEL=gpt-3.5-turbo

# GPT-4 (higher quality)
LLM_MODEL=gpt-4

# Custom endpoint
LLM_PROVIDER=custom
LLM_ENDPOINT=https://your-custom-endpoint.com
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Module Import Errors

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Always use PYTHONPATH when running scripts
PYTHONPATH=. python scripts/build_knowledge_base.py
PYTHONPATH=. uvicorn src.api.main:app
```

#### 2. Empty Search Results

**Problem:** API returns no results for queries

**Solutions:**
```bash
# Rebuild knowledge base with more documents
PYTHONPATH=. python scripts/build_knowledge_base.py

# Check vector database
PYTHONPATH=. python -c "
from src.embeddings.vector_store import VectorStore
vs = VectorStore(768)
vs.load_index('./data/vectordb/index')
print(f'Vectors loaded: {vs.index.ntotal}')
"
```

#### 3. LLM Service Failures

**Problem:** `ValueError: OPENAI_API_KEY environment variable required`

**Solutions:**
```bash
# Option 1: Add API key to .env
echo "OPENAI_API_KEY=sk-your-key-here" >> .env

# Option 2: Use demo mode (search only)
# The system will automatically fall back to search-only mode
```

#### 4. Memory Issues

**Problem:** Out of memory during embedding generation

**Solutions:**
```python
# Reduce batch size in embedding service
service = EmbeddingService()
embeddings = service.embed_texts(texts, batch_size=16)  # Reduce from 32

# Or process in smaller chunks
for i in range(0, len(texts), 100):
    batch = texts[i:i+100]
    # Process batch
```

#### 5. Slow Response Times

**Problem:** API responses are slow

**Solutions:**
```bash
# Check system resources
top
free -h

# Reduce context size
curl -X POST "http://localhost:8000/ask" \
  -d '{"question": "...", "max_chunks": 3}'  # Reduce from 5

# Use smaller embedding model
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Debug Mode

Enable detailed logging:

```bash
# Set debug environment
export LOG_LEVEL=DEBUG
export PYTHONPATH=.

# Run with debug logging
uvicorn src.api.main:app --log-level debug

# Check specific components
python -c "
from src.embeddings.embedding_service import EmbeddingService
service = EmbeddingService()
print('Embedding service loaded successfully')
"
```

### Health Check Diagnostics

```bash
# Check API health
curl http://localhost:8000/health | python -m json.tool

# Check individual components
PYTHONPATH=. python test_system.py

# Verify vector database
ls -la data/vectordb/
```

---

## Advanced Usage

### Custom Data Sources

To add your own documentation:

1. **Extend the scraper:**
```python
# src/crawler/scraper.py
class CustomScraper(FastAPIScraper):
    def __init__(self, base_url="https://your-docs.com"):
        super().__init__(base_url)
    
    def extract_links(self, html):
        # Custom link extraction logic
        pass
```

2. **Update the build script:**
```python
# scripts/build_knowledge_base.py
scraper = CustomScraper("https://your-docs.com")
pages = await scraper.crawl(max_pages=100)
```

### Custom Embeddings

Use your own embedding model:

```python
# src/embeddings/embedding_service.py
class CustomEmbeddingService(EmbeddingService):
    def __init__(self, model_path="./models/custom-model"):
        self.model = SentenceTransformer(model_path)
```

### API Integration

#### Python Client

```python
import requests

class FastAPIRAGClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def ask(self, question, max_chunks=5):
        response = requests.post(
            f"{self.base_url}/ask",
            json={"question": question, "max_chunks": max_chunks}
        )
        return response.json()
    
    def search(self, query, k=5):
        response = requests.get(
            f"{self.base_url}/search",
            params={"query": query, "k": k}
        )
        return response.json()

# Usage
client = FastAPIRAGClient()
result = client.ask("How do I handle database connections in FastAPI?")
print(result["answer"])
```

#### JavaScript Client

```javascript
class FastAPIRAGClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async ask(question, maxChunks = 5) {
        const response = await fetch(`${this.baseUrl}/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                max_chunks: maxChunks
            })
        });
        return await response.json();
    }
    
    async search(query, k = 5) {
        const response = await fetch(
            `${this.baseUrl}/search?query=${encodeURIComponent(query)}&k=${k}`
        );
        return await response.json();
    }
}

// Usage
const client = new FastAPIRAGClient();
client.ask('How do I create API endpoints?').then(result => {
    console.log(result.answer);
});
```

### Batch Processing

Process multiple questions efficiently:

```python
import asyncio
import aiohttp

async def batch_ask(questions):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for question in questions:
            task = session.post(
                'http://localhost:8000/ask',
                json={'question': question}
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        results = []
        for response in responses:
            results.append(await response.json())
        return results

# Usage
questions = [
    "How do I create a FastAPI app?",
    "What is dependency injection?", 
    "How to handle errors?"
]
results = asyncio.run(batch_ask(questions))
```

---

## Examples & Use Cases

### 1. Developer Documentation Assistant

Create a chat interface for your development team:

```python
def documentation_chat():
    """Interactive documentation chat."""
    client = FastAPIRAGClient()
    
    print("FastAPI Documentation Assistant")
    print("Type 'quit' to exit\n")
    
    while True:
        question = input("Ask a question: ")
        if question.lower() == 'quit':
            break
        
        result = client.ask(question)
        print(f"\nAnswer: {result['answer']}\n")
        
        if result['sources']:
            print("Sources:")
            for source in result['sources']:
                print(f"- {source['title']}: {source['url']}")
        print("-" * 50)
```

### 2. Knowledge Base Search Tool

Build a search interface:

```python
def search_knowledge_base(query, num_results=5):
    """Search the knowledge base and display results."""
    client = FastAPIRAGClient()
    results = client.search(query, k=num_results)
    
    print(f"Search results for: '{query}'\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")
        print(f"   Score: {result['relevance_score']:.3f}")
        print(f"   URL: {result['url']}")
        print(f"   Excerpt: {result['excerpt'][:200]}...")
        print()
```

### 3. API Documentation Validator

Validate your API documentation coverage:

```python
def validate_documentation_coverage():
    """Check coverage of common FastAPI topics."""
    client = FastAPIRAGClient()
    
    topics = [
        "creating endpoints",
        "request validation", 
        "response models",
        "dependency injection",
        "database integration",
        "authentication",
        "testing",
        "deployment"
    ]
    
    coverage_report = {}
    for topic in topics:
        results = client.search(f"FastAPI {topic}", k=3)
        coverage_report[topic] = {
            'results_count': len(results),
            'avg_relevance': sum(r['relevance_score'] for r in results) / len(results) if results else 0
        }
    
    print("Documentation Coverage Report:")
    for topic, stats in coverage_report.items():
        status = "✅" if stats['avg_relevance'] > 0.5 else "⚠️"
        print(f"{status} {topic}: {stats['results_count']} docs (avg relevance: {stats['avg_relevance']:.3f})")
```

### 4. Content Generation Assistant

Generate documentation based on existing content:

```python
def generate_tutorial_outline(topic):
    """Generate a tutorial outline for a given topic."""
    client = FastAPIRAGClient()
    
    question = f"Create a step-by-step tutorial outline for {topic} in FastAPI"
    result = client.ask(question, max_chunks=8)
    
    print(f"Tutorial Outline: {topic}")
    print("=" * 50)
    print(result['answer'])
    
    if result['sources']:
        print("\nReference Materials:")
        for source in result['sources']:
            print(f"- {source['title']}: {source['url']}")
```

### 5. Error Diagnosis Helper

Help debug common FastAPI errors:

```python
def diagnose_error(error_message):
    """Help diagnose FastAPI errors."""
    client = FastAPIRAGClient()
    
    question = f"How do I fix this FastAPI error: {error_message}"
    result = client.ask(question)
    
    print("Error Diagnosis:")
    print("-" * 30)
    print(f"Error: {error_message}")
    print(f"\nSolution: {result['answer']}")
    
    return result
```

---

## Performance Optimization

### Server Optimization

```bash
# Use multiple workers in production
uvicorn src.api.main:app --workers 4 --host 0.0.0.0 --port 8000

# Enable gzip compression
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --access-log

# Use production ASGI server
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Memory Optimization

```python
# Reduce embedding batch size
EMBEDDING_BATCH_SIZE=16

# Limit vector database size
MAX_VECTORS=10000

# Use quantized models
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Smaller model
```

### Caching Strategies

```python
# Add Redis caching for frequent queries
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cached_ask(question):
    cache_key = f"ask:{hash(question)}"
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return json.loads(cached_result)
    
    # Make API call
    result = client.ask(question)
    
    # Cache for 1 hour
    redis_client.setex(cache_key, 3600, json.dumps(result))
    return result
```

---

## Security Considerations

### API Security

```python
# Add API key authentication
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/ask")
async def ask_question(request: QuestionRequest, token: str = Security(security)):
    if token.credentials != "your-api-key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    # ... rest of endpoint
```

### Rate Limiting

```python
# Add rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/ask")
@limiter.limit("10/minute")
async def ask_question(request: Request, question_request: QuestionRequest):
    # ... endpoint logic
```

### Input Validation

```python
# Strict input validation
from pydantic import validator

class QuestionRequest(BaseModel):
    question: str
    
    @validator('question')
    def validate_question(cls, v):
        if len(v) > 500:
            raise ValueError('Question too long')
        if len(v.strip()) < 3:
            raise ValueError('Question too short')
        return v.strip()
```

---

## Monitoring & Analytics

### Health Monitoring

```python
# Enhanced health check
@app.get("/health")
async def health_check():
    try:
        # Test embedding service
        embedding_test = embedding_service.embed_query("test")
        
        # Test vector store
        vector_count = vector_store.index.ntotal
        
        # Test LLM service (if available)
        llm_status = "available" if llm_service else "unavailable"
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "components": {
                "embedding_service": "operational",
                "vector_store": f"loaded ({vector_count} vectors)",
                "llm_service": llm_status
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow()
        }
```

### Usage Analytics

```python
# Track usage metrics
import time
from collections import defaultdict

metrics = defaultdict(int)
response_times = []

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response_times.append(process_time)
    metrics[f"{request.method}_{request.url.path}"] += 1
    
    return response

@app.get("/metrics")
async def get_metrics():
    return {
        "total_requests": sum(metrics.values()),
        "endpoint_usage": dict(metrics),
        "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
        "total_vectors": vector_store.index.ntotal
    }
```

---

This comprehensive documentation covers all aspects of using the FastAPI RAG Chatbot system. For additional help or specific use cases not covered here, please refer to the interactive API documentation at `/docs` when the server is running.
