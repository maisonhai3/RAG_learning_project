from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    QuestionRequest, ChatResponse, HealthResponse,
    FeedbackRequest
)
from src.services import RAGService


# Initialize RAG service
rag_service = RAGService()

app = FastAPI(
    title="FastAPI Documentation Chatbot",
    description="RAG-powered Q&A system for FastAPI documentation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        dependencies=rag_service.is_healthy()
    )


@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: QuestionRequest):
    """Main chatbot endpoint."""
    return await rag_service.process_question(request)


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback."""
    # In a real implementation, save to database
    return {"message": "Feedback received", "timestamp": datetime.now()}


@app.get("/stats")
async def get_usage_stats():
    """Get usage statistics."""
    return {
        "vector_store_stats": rag_service.get_vector_store_stats(),
        "total_documents": rag_service.get_total_documents(),
        "embedding_model": rag_service.get_embedding_model_name(),
        "llm_model": rag_service.get_llm_model_name()
    }


@app.get("/search")
async def search_docs(query: str, k: int = 5):
    """Direct document search without LLM generation."""
    sources = rag_service.search_documents(query, k)
    return {"query": query, "results": sources}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
