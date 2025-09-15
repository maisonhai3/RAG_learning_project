import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from enum import Enum

import numpy as np

try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from src.retrieval.retriever import RetrievalResult


class RerankingStrategy(Enum):
    """Available re-ranking strategies."""
    CROSS_ENCODER = "cross_encoder"
    LLM_BASED = "llm_based" 
    DIVERSITY_MMR = "diversity_mmr"
    HYBRID = "hybrid"
    DISABLED = "disabled"


class BaseReranker(ABC):
    """Abstract base class for re-ranking implementations."""
    
    @abstractmethod
    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Re-rank search results based on query relevance.
        
        Args:
            query: The search query
            results: List of search results to re-rank
            
        Returns:
            List of re-ranked search results with updated scores
        """
        pass


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder based re-ranking using sentence-transformers."""
    
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder re-ranker.
        
        Args:
            model_name: Name of the cross-encoder model to use
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for CrossEncoderReranker")
            
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        
    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Re-rank using cross-encoder model."""
        if not results:
            return results
            
        # Prepare query-document pairs
        pairs = [(query, result.text) for result in results]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Update results with new scores and sort
        reranked_results = []
        for result, score in zip(results, scores):
            new_result = RetrievalResult(
                text=result.text,
                source_url=result.source_url,
                metadata=result.metadata,
                relevance_score=float(score)
            )
            reranked_results.append(new_result)
            
        # Sort by new scores (descending)
        reranked_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return reranked_results


class LLMBasedReranker(BaseReranker):
    """LLM-based re-ranking using existing LLM service."""
    
    def __init__(self, llm_service=None):
        """
        Initialize LLM-based re-ranker.
        
        Args:
            llm_service: LLM service instance for scoring
        """
        self.llm_service = llm_service
        
    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        raise NotImplementedError


class DiversityMMRReranker(BaseReranker):
    """Maximal Marginal Relevance (MMR) re-ranking for diversity."""
    
    def __init__(self, lambda_param: float = 0.7, embedding_service=None):
        """
        Initialize MMR re-ranker.
        
        Args:
            lambda_param: Balance between relevance and diversity (0-1)
            embedding_service: Service for computing embeddings
        """
        self.lambda_param = lambda_param
        self.embedding_service = embedding_service
        
    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Re-rank using MMR algorithm."""
        if not results or not self.embedding_service:
            return results
            
        if len(results) <= 1:
            return results
            
        # Get query embedding
        query_embedding = self.embedding_service.embed_query(query)
        if query_embedding.size == 0:
            return results
            
        # Get document embeddings
        doc_texts = [result.text for result in results]
        doc_embeddings = self.embedding_service.embed_texts(doc_texts)
        
        if len(doc_embeddings) == 0:
            return results
            
        # MMR algorithm
        selected_indices = []
        remaining_indices = list(range(len(results)))
        
        while remaining_indices and len(selected_indices) < len(results):
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance score (cosine similarity with query)
                relevance = self._cosine_similarity(query_embedding, doc_embeddings[idx])
                
                # Diversity score (max similarity with already selected docs)
                if selected_indices:
                    max_sim = max(
                        self._cosine_similarity(doc_embeddings[idx], doc_embeddings[sel_idx])
                        for sel_idx in selected_indices
                    )
                else:
                    max_sim = 0
                    
                # MMR score
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                mmr_scores.append((idx, mmr_score))
                
            # Select document with highest MMR score
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            
        # Return results in MMR order with updated scores
        reranked_results = []
        for i, idx in enumerate(selected_indices):
            result = results[idx]
            new_result = RetrievalResult(
                text=result.text,
                source_url=result.source_url,
                metadata=result.metadata,
                relevance_score=1.0 - (i / len(selected_indices))  # Decreasing score by rank
            )
            reranked_results.append(new_result)
            
        return reranked_results
        
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if a.size == 0 or b.size == 0:
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class HybridReranker(BaseReranker):
    """Hybrid re-ranking combining multiple strategies."""
    
    def __init__(self, rerankers: List[Tuple[BaseReranker, float]]):
        """
        Initialize hybrid re-ranker.
        
        Args:
            rerankers: List of (reranker, weight) tuples
        """
        self.rerankers = rerankers
        
    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Re-rank using weighted combination of strategies."""
        if not results or not self.rerankers:
            return results
            
        # Get scores from each reranker
        all_scores = {}
        
        for reranker, weight in self.rerankers:
            reranked = reranker.rerank(query, results)
            for i, result in enumerate(reranked):
                if i not in all_scores:
                    all_scores[i] = 0.0
                all_scores[i] += weight * result.relevance_score
                
        # Sort by combined scores
        sorted_indices = sorted(all_scores.keys(), key=lambda x: all_scores[x], reverse=True)
        
        # Return results in new order
        reranked_results = []
        for rank, idx in enumerate(sorted_indices):
            result = results[idx]
            new_result = RetrievalResult(
                text=result.text,
                source_url=result.source_url,
                metadata=result.metadata,
                relevance_score=all_scores[idx]
            )
            reranked_results.append(new_result)
            
        return reranked_results


class RerankingService:
    """Main re-ranking service that orchestrates different strategies."""
    
    def __init__(self, strategy: RerankingStrategy = RerankingStrategy.DISABLED,
                 embedding_service=None, llm_service=None, 
                 model_name: str = "ms-marco-MiniLM-L-6-v2", mmr_lambda: float = 0.7):
        """
        Initialize re-ranking service.
        
        Args:
            strategy: Re-ranking strategy to use
            embedding_service: Embedding service for certain strategies
            llm_service: LLM service for LLM-based re-ranking
            model_name: Model name for cross-encoder re-ranking
            mmr_lambda: Lambda parameter for MMR diversity
        """
        self.strategy = strategy
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.model_name = model_name
        self.mmr_lambda = mmr_lambda
        self.reranker = self._create_reranker()
        
    def _create_reranker(self) -> Optional[BaseReranker]:
        """Create appropriate re-ranker based on strategy."""
        if self.strategy == RerankingStrategy.DISABLED:
            return None
            
        elif self.strategy == RerankingStrategy.CROSS_ENCODER:
            try:
                return CrossEncoderReranker(self.model_name)
            except ImportError:
                print("Warning: sentence-transformers not available, disabling cross-encoder re-ranking")
                return None
                
        elif self.strategy == RerankingStrategy.LLM_BASED:
            return LLMBasedReranker(self.llm_service)
            
        elif self.strategy == RerankingStrategy.DIVERSITY_MMR:
            return DiversityMMRReranker(lambda_param=self.mmr_lambda, embedding_service=self.embedding_service)
            
        elif self.strategy == RerankingStrategy.HYBRID:
            rerankers = []
            
            # Try to add cross-encoder
            try:
                cross_encoder = CrossEncoderReranker(self.model_name)
                rerankers.append((cross_encoder, 0.7))
            except ImportError:
                pass
                
            # Add MMR for diversity
            if self.embedding_service:
                mmr = DiversityMMRReranker(lambda_param=self.mmr_lambda, embedding_service=self.embedding_service)
                rerankers.append((mmr, 0.3))
                
            return HybridReranker(rerankers) if rerankers else None
            
        return None
        
    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Re-rank search results.
        
        Args:
            query: The search query
            results: List of search results to re-rank
            
        Returns:
            List of re-ranked search results
        """
        if not self.reranker or not results:
            return results
            
        try:
            return self.reranker.rerank(query, results)
        except Exception as e:
            print(f"Warning: Re-ranking failed with error: {e}")
            return results
            
    def is_enabled(self) -> bool:
        """Check if re-ranking is enabled."""
        return self.strategy != RerankingStrategy.DISABLED and self.reranker is not None