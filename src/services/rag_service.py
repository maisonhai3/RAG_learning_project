import os
import time
from typing import Optional

try:
    from fastapi import HTTPException
except ImportError:
    # For testing without FastAPI installed
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

from src.api.models import QuestionRequest, ChatResponse, Source
from src.config.settings import Settings
from src.embeddings.embedding_service import EmbeddingService
from src.embeddings.vector_store import VectorStore
from src.generation.llm_service import LLMService
from src.generation.prompt_templates import PromptTemplates
from src.retrieval.retriever import AdvancedRetriever
from src.services.reranking_service import RerankingService, RerankingStrategy


class RAGService:
    """
    Main RAG (Retrieval-Augmented Generation) service that orchestrates
    the entire pipeline from question processing to answer generation.
    """

    def __init__(self):
        """Initialize RAG service components."""
        self.settings = Settings()
        
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
        self.llm_service = self._initialize_llm_service()
        
        # Initialize re-ranking service
        reranking_strategy = getattr(self.settings, "RERANKING_STRATEGY", "disabled").lower()
        try:
            strategy = RerankingStrategy(reranking_strategy)
        except ValueError:
            strategy = RerankingStrategy.DISABLED
            
        self.reranking_service = RerankingService(
            strategy=strategy,
            embedding_service=self.embedding_service,
            llm_service=self.llm_service,
            model_name=getattr(self.settings, 'RERANKING_MODEL', 'ms-marco-MiniLM-L-6-v2'),
            mmr_lambda=getattr(self.settings, 'MMR_LAMBDA', 0.7)
        )

    def _initialize_llm_service(self) -> Optional[LLMService]:
        """Initialize LLM service with proper error handling."""
        try:
            # Determine which API key to use based on provider
            api_key = ""
            if self.settings.LLM_PROVIDER == "openai":
                api_key = self.settings.OPENAI_API_KEY
            elif self.settings.LLM_PROVIDER == "gemini":
                api_key = self.settings.GEMINI_API_KEY

            return LLMService(
                provider=self.settings.LLM_PROVIDER,
                api_key=api_key,
                model=self.settings.MODEL
            )
        except Exception as e:
            print(f"Warning: Could not initialize LLM service: {e}")
            print("Running in demo mode without LLM")
            return None

    async def process_question(self, request: QuestionRequest) -> ChatResponse:
        """
        Process a question through the RAG pipeline.
        
        Args:
            request: The question request containing query and parameters
            
        Returns:
            ChatResponse: The generated response with sources and metadata
            
        Raises:
            HTTPException: If processing fails
        """
        start_time = time.time()

        try:
            # Step 1: Retrieve relevant context
            results = self._retrieve_context(request)
            
            # Step 1.5: Re-rank retrieved documents (if enabled)
            if self.reranking_service.is_enabled():
                results = self.reranking_service.rerank(request.question, results)

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
            sources = self._prepare_sources(results, request.include_sources)

            # Calculate confidence based on retrieval scores
            confidence = self._calculate_confidence(results)

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

    def _retrieve_context(self, request: QuestionRequest):
        """Retrieve relevant context based on search strategy."""
        if request.search_strategy == "semantic":
            return self.retriever.semantic_search(
                request.question,
                k=request.max_chunks
            )
        elif request.search_strategy == "keyword":
            return self.retriever.keyword_search(
                request.question,
                self.vector_store.texts,
                self.vector_store.metadata,
                k=request.max_chunks
            )
        else:  # hybrid
            return self.retriever.hybrid_search(
                request.question,
                k=request.max_chunks
            )

    def _prepare_sources(self, results, include_sources: bool) -> list:
        """Prepare source information from search results."""
        sources = []
        if include_sources:
            for result in results:
                sources.append(Source(
                    title=result.metadata.get('title', 'FastAPI Documentation'),
                    url=result.source_url,
                    relevance_score=result.relevance_score,
                    excerpt=result.text[:200] + "..." if len(result.text) > 200 else result.text
                ))
        return sources

    def _calculate_confidence(self, results) -> float:
        """Calculate confidence score based on retrieval results."""
        if not results:
            return 0.0
        return sum(r.relevance_score for r in results) / len(results)

    def get_vector_store_stats(self) -> dict:
        """Get vector store statistics."""
        return self.vector_store.get_stats()

    def get_total_documents(self) -> int:
        """Get total number of documents in vector store."""
        return self.vector_store.index.ntotal

    def get_embedding_model_name(self) -> str:
        """Get embedding model name."""
        return self.embedding_service.model_name

    def get_llm_model_name(self) -> str:
        """Get LLM model name."""
        return self.llm_service.model if self.llm_service else "N/A"

    def search_documents(self, query: str, k: int = 5) -> list:
        """Direct document search without LLM generation."""
        results = self.retriever.semantic_search(query, k)
        
        sources = []
        for result in results:
            sources.append({
                "title": result.metadata.get('title', 'FastAPI Documentation'),
                "url": result.source_url,
                "relevance_score": result.relevance_score,
                "excerpt": result.text[:300] + "..." if len(result.text) > 300 else result.text
            })
        
        return sources

    def is_healthy(self) -> dict:
        """Check service health status."""
        return {
            "vector_store": "loaded" if self.vector_store.index.ntotal > 0 else "empty",
            "embedding_service": "ready",
            "llm_service": "ready" if self.llm_service else "unavailable",
            "reranking_service": f"enabled ({self.reranking_service.strategy.value})" if self.reranking_service.is_enabled() else "disabled"
        }