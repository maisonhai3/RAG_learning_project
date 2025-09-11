from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import torch
import os


class EmbeddingService:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """Initialize the embedding service with a specified model."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Convert a list of texts to embeddings."""
        if not texts:
            return np.array([])
        
        # Remove empty texts and track indices
        valid_texts = [(i, text) for i, text in enumerate(texts) if text.strip()]
        if not valid_texts:
            return np.array([])
            
        indices, clean_texts = zip(*valid_texts)
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            clean_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Convert a single query to embedding."""
        if not query.strip():
            return np.array([])
        
        embedding = self.model.encode(
            query,
            convert_to_numpy=True
        )
        return embedding
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from this model."""
        return self.model.get_sentence_embedding_dimension()
