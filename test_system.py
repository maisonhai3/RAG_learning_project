#!/usr/bin/env python3
"""
Test script for the FastAPI RAG Chatbot system.
Tests all components without requiring OpenAI API key.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.embeddings.embedding_service import EmbeddingService
from src.embeddings.vector_store import VectorStore
from src.retrieval.retriever import AdvancedRetriever
from src.config.settings import Settings

def test_complete_system():
    """Test the complete RAG system pipeline."""
    print("🧪 Testing FastAPI RAG Chatbot System")
    print("=" * 50)
    
    # Test 1: Embedding Service
    print("\n1️⃣ Testing Embedding Service...")
    embedding_service = EmbeddingService()
    test_query = "How do I create a FastAPI application?"
    query_embedding = embedding_service.embed_query(test_query)
    print(f"✅ Query embedding generated: {query_embedding.shape}")
    
    # Test 2: Vector Store
    print("\n2️⃣ Testing Vector Store...")
    try:
        settings = Settings()
        vector_store = VectorStore(dimension=embedding_service.get_embedding_dimension())
        vector_store.load_index(settings.VECTOR_DB_PATH)
        print(f"✅ Vector store loaded with {vector_store.index.ntotal} vectors")
    except Exception as e:
        print(f"❌ Vector store loading failed: {e}")
        return False
    
    # Test 3: Retrieval System
    print("\n3️⃣ Testing Retrieval System...")
    retriever = AdvancedRetriever(vector_store, embedding_service)
    results = retriever.semantic_search(test_query, k=3)
    print(f"✅ Retrieved {len(results)} relevant documents")
    
    if results:
        print(f"Top result relevance score: {results[0].relevance_score:.3f}")
        print(f"Top result preview: {results[0].text[:100]}...")
    
    # Test 4: Search Functionality
    print("\n4️⃣ Testing Search Variations...")
    test_queries = [
        "What is FastAPI?",
        "How to handle database connections?",
        "FastAPI authentication and security",
        "Creating REST API endpoints"
    ]
    
    for query in test_queries:
        results = retriever.semantic_search(query, k=2)
        print(f"   '{query}' → {len(results)} results (avg score: {sum(r.relevance_score for r in results)/len(results) if results else 0:.3f})")
    
    print("\n✅ System test completed successfully!")
    print("\n📊 System Summary:")
    print(f"   - Embedding model: {embedding_service.model_name}")
    print(f"   - Vector dimension: {embedding_service.get_embedding_dimension()}")
    print(f"   - Knowledge base size: {vector_store.index.ntotal} documents")
    print(f"   - Retrieval system: Operational")
    
    return True

if __name__ == "__main__":
    success = test_complete_system()
    sys.exit(0 if success else 1)
