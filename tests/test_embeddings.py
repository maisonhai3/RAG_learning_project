import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pytest
from src.embeddings.embedding_service import EmbeddingService
from src.embeddings.vector_store import VectorStore
import numpy as np


def test_embedding_service_initialization():
    """Test embedding service initialization."""
    service = EmbeddingService()
    assert service.model_name == "all-mpnet-base-v2"
    assert service.get_embedding_dimension() > 0


def test_embed_single_text():
    """Test embedding a single text."""
    service = EmbeddingService()
    text = "FastAPI is a modern web framework for Python"
    embedding = service.embed_query(text)
    
    assert isinstance(embedding, np.ndarray)
    assert len(embedding.shape) == 1
    assert embedding.shape[0] == service.get_embedding_dimension()


def test_embed_multiple_texts():
    """Test embedding multiple texts."""
    service = EmbeddingService()
    texts = [
        "FastAPI is fast",
        "Python web framework",
        "REST API development"
    ]
    embeddings = service.embed_texts(texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] == service.get_embedding_dimension()


def test_vector_store_operations():
    """Test vector store basic operations."""
    dimension = 384  # Common dimension for sentence transformers
    store = VectorStore(dimension=dimension)
    
    # Create sample data
    vectors = np.random.random((3, dimension)).astype('float32')
    texts = ["text1", "text2", "text3"]
    metadata = [{"id": i} for i in range(3)]
    
    # Add vectors
    store.add_vectors(vectors, texts, metadata)
    
    # Test search
    query_vector = np.random.random((dimension,)).astype('float32')
    results = store.search(query_vector, k=2)
    
    assert len(results) <= 2
    assert len(results) > 0
    assert all(hasattr(r, 'text') for r in results)
    assert all(hasattr(r, 'score') for r in results)
