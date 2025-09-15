import re
from typing import List, Dict

from src.embeddings.embedding_service import EmbeddingService
from src.embeddings.vector_store import VectorStore


class RetrievalResult:
    def __init__(
        self, text: str, metadata: Dict, relevance_score: float, source_url: str = ""
    ):
        self.text = text
        self.metadata = metadata
        self.relevance_score = relevance_score
        self.source_url = source_url


class AdvancedRetriever:
    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService):
        """Initialize retriever with vector store and embedding service."""
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def semantic_search(
        self, query: str, k: int = 5, min_score: float = 0.1
    ) -> List[RetrievalResult]:
        """Perform semantic search using vector similarity."""
        if not query.strip():
            return []

        # Get query embedding
        query_embedding = self.embedding_service.embed_query(query)
        if query_embedding.size == 0:
            return []

        # Search in vector store
        search_results = self.vector_store.search(query_embedding, k)

        # Convert to RetrievalResult and filter by minimum score
        results = []
        for result in search_results:
            if result.score >= min_score:
                retrieval_result = RetrievalResult(
                    text=result.text,
                    metadata=result.metadata,
                    relevance_score=result.score,
                    source_url=result.metadata.get("url", ""),
                )
                results.append(retrieval_result)

        return results

    def keyword_search(
        self, query: str, texts: List[str], metadata: List[Dict], k: int = 5
    ) -> List[RetrievalResult]:
        """Simple keyword-based search for comparison."""
        query_words = set(query.lower().split())

        scores = []
        for i, text in enumerate(texts):
            text_words = set(re.findall(r"\w+", text.lower()))
            # Simple overlap score
            overlap = len(query_words.intersection(text_words))
            score = overlap / len(query_words) if query_words else 0
            scores.append((score, i))

        # Sort by score and take top k
        scores.sort(reverse=True)

        results = []
        for score, idx in scores[:k]:
            if score > 0:
                results.append(
                    RetrievalResult(
                        text=texts[idx],
                        metadata=metadata[idx],
                        relevance_score=score,
                        source_url=metadata[idx].get("url", ""),
                    )
                )

        return results

    def hybrid_search(
        self, query: str, k: int = 5, alpha: float = 0.7
    ) -> List[RetrievalResult]:
        """Combine semantic and keyword search with weighted scoring."""
        # Get semantic results
        semantic_results = self.semantic_search(query, k * 2)

        # Get keyword results from stored texts
        keyword_results = self.keyword_search(
            query, self.vector_store.texts, self.vector_store.metadata, k * 2
        )

        # Combine and re-rank
        combined_results = {}

        # Add semantic results
        for result in semantic_results:
            text_key = result.text[:100]  # Use first 100 chars as key
            combined_results[text_key] = {
                "result": result,
                "semantic_score": result.relevance_score,
                "keyword_score": 0.0,
            }

        # Add keyword scores
        for result in keyword_results:
            text_key = result.text[:100]
            if text_key in combined_results:
                combined_results[text_key]["keyword_score"] = result.relevance_score
            else:
                combined_results[text_key] = {
                    "result": result,
                    "semantic_score": 0.0,
                    "keyword_score": result.relevance_score,
                }

        # Calculate hybrid scores
        final_results = []
        for data in combined_results.values():
            hybrid_score = (
                alpha * data["semantic_score"] + (1 - alpha) * data["keyword_score"]
            )

            result = data["result"]
            result.relevance_score = hybrid_score
            final_results.append(result)

        # Sort by hybrid score and return top k
        final_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return final_results[:k]

    def filtered_search(
        self, query: str, filters: Dict, k: int = 5
    ) -> List[RetrievalResult]:
        """Search with metadata filtering."""
        # First get all semantic results
        all_results = self.semantic_search(query, k * 5)

        # Apply filters
        filtered_results = []
        for result in all_results:
            include = True
            for filter_key, filter_value in filters.items():
                if filter_key in result.metadata:
                    if result.metadata[filter_key] != filter_value:
                        include = False
                        break

            if include:
                filtered_results.append(result)

        return filtered_results[:k]

    def get_context_window(
        self, results: List[RetrievalResult], max_length: int = 2000
    ) -> str:
        """Combine multiple retrieval results into a context window."""
        context_parts = []
        current_length = 0

        for result in results:
            # Add source information
            source_info = f"Source: {result.source_url}\n" if result.source_url else ""
            text_with_source = f"{source_info}{result.text}"

            if current_length + len(text_with_source) <= max_length:
                context_parts.append(text_with_source)
                current_length += len(text_with_source)
            else:
                # Add partial text if possible
                remaining = max_length - current_length
                if remaining > 100:  # Only add if meaningful chunk remains
                    context_parts.append(text_with_source[:remaining] + "...")
                break

        return "\n\n---\n\n".join(context_parts)
