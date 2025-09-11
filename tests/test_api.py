import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# Set testing environment
os.environ["TESTING"] = "true"

from src.api.main import app
from fastapi.testclient import TestClient


client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "version" in data


def test_stats_endpoint():
    """Test stats endpoint."""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "vector_store_stats" in data
    assert "embedding_model" in data


def test_search_endpoint():
    """Test search endpoint."""
    response = client.get("/search?query=FastAPI+tutorial&k=3")
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert isinstance(data["results"], list)
