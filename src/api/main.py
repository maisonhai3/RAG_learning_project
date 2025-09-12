import os
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    QuestionRequest, ChatResponse, HealthResponse,
    FeedbackRequest, Source
)
from src.config.settings import Settings
from src.embeddings.embedding_service import EmbeddingService
from src.embeddings.vector_store import VectorStore
from src.generation.llm_service import LLMService
from src.generation.prompt_templates import PromptTemplates
from src.retrieval.retriever import AdvancedRetriever

settings = Settings()


class RAGService:
    def __init__(self):
        """Initialize RAG service components."""
        # Initialize embedding service
        self.embedding_service = EmbeddingService()

        # Initialize vector store
        embedding_dim = self.embedding_service.get_embedding_dimension()
        self.vector_store = VectorStore(dimension=embedding_dim)

        # Load existing vector store if available
        vector_path = os.getenv("VECTOR_DB_PATH", "./data/vectordb/index")
        if os.path.exists(f"{vector_path}.faiss"):
            self.vector_store.load_index(vector_path)

        # Initialize retriever
        self.retriever = AdvancedRetriever(
            self.vector_store,
            self.embedding_service
        )

        # Initialize LLM service (optional for demo)
        try:
            # Determine which API key to use based on provider
            api_key = ""
            if settings.LLM_PROVIDER == "openai":
                api_key = settings.OPENAI_API_KEY
            elif settings.LLM_PROVIDER == "gemini":
                api_key = settings.GEMINI_API_KEY

            self.llm_service = LLMService(
                provider=settings.LLM_PROVIDER,
                api_key=api_key,
                model=settings.MODEL
            )
        except Exception as e:
            print(f"Warning: Could not initialize LLM service: {e}")
            print("Running in demo mode without LLM")
            self.llm_service = None

    async def process_question(self, request: QuestionRequest) -> ChatResponse:
        """Process a question through the RAG pipeline."""
        start_time = time.time()

        try:
            # Step 1: Retrieve relevant context
            if request.search_strategy == "semantic":
                results = self.retriever.semantic_search(
                    request.question,
                    k=request.max_chunks
                )
            elif request.search_strategy == "keyword":
                results = self.retriever.keyword_search(
                    request.question,
                    self.vector_store.texts,
                    self.vector_store.metadata,
                    k=request.max_chunks
                )
            else:  # hybrid
                results = self.retriever.hybrid_search(
                    request.question,
                    k=request.max_chunks
                )

            # Step 2: Build prompt with context
            prompt = PromptTemplates.build_context_aware_prompt(
                request.question,
                results
            )

            # Step 3: Generate response
            llm_response = await self.llm_service.generate_response(
                prompt,
                temperature=request.temperature
            )

            # Step 4: Prepare sources
            sources = []
            if request.include_sources:
                for result in results:
                    sources.append(Source(
                        title=result.metadata.get('title', 'FastAPI Documentation'),
                        url=result.source_url,
                        relevance_score=result.relevance_score,
                        excerpt=result.text[:200] + "..." if len(result.text) > 200 else result.text
                    ))

            # Calculate confidence based on retrieval scores
            confidence = sum(r.relevance_score for r in results) / len(results) if results else 0.0

            processing_time = time.time() - start_time

            return ChatResponse(
                answer=llm_response.content,
                sources=sources,
                confidence_score=min(confidence, 1.0),
                processing_time=processing_time,
                tokens_used=llm_response.tokens_used,
                search_results_count=len(results)
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


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
        dependencies={
            "vector_store": "loaded" if rag_service.vector_store.index.ntotal > 0 else "empty",
            "embedding_service": "ready",
            "llm_service": "ready"
        }
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
        "vector_store_stats": rag_service.vector_store.get_stats(),
        "total_documents": rag_service.vector_store.index.ntotal,
        "embedding_model": rag_service.embedding_service.model_name,
        "llm_model": rag_service.llm_service.model
    }


@app.get("/search")
async def search_docs(query: str, k: int = 5):
    """Direct document search without LLM generation."""
    results = rag_service.retriever.semantic_search(query, k)

    sources = []
    for result in results:
        sources.append({
            "title": result.metadata.get('title', 'FastAPI Documentation'),
            "url": result.source_url,
            "relevance_score": result.relevance_score,
            "excerpt": result.text[:300] + "..." if len(result.text) > 300 else result.text
        })

    return {"query": query, "results": sources}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
