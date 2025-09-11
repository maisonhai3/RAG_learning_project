import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Any
import json


class SearchResult:
    def __init__(self, text: str, metadata: Dict, score: float):
        self.text = text
        self.metadata = metadata
        self.score = score


class VectorStore:
    def __init__(self, dimension: int, index_type: str = "Flat"):
        """Initialize vector store with specified dimension and index type."""
        self.dimension = dimension
        self.index_type = index_type
        
        # Create FAISS index
        if index_type == "Flat":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product
        elif index_type == "IVF":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
            
        self.texts = []  # Store original texts
        self.metadata = []  # Store metadata for each vector
        
    def add_vectors(self, vectors: np.ndarray, texts: List[str], 
                   metadata: List[Dict]):
        """Add vectors with their corresponding texts and metadata."""
        if len(vectors) != len(texts) or len(texts) != len(metadata):
            raise ValueError("Vectors, texts, and metadata must have same length")
            
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)
        
        # Add to index
        self.index.add(vectors.astype('float32'))
        
        # Store texts and metadata
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        
        # Train index if needed
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            if self.index.ntotal >= 100:  # Minimum training samples
                self.index.train(vectors.astype('float32'))
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[SearchResult]:
        """Search for similar vectors."""
        if self.index.ntotal == 0:
            return []
            
        # Normalize query vector
        query_vector = query_vector.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_vector)
        
        # Search
        scores, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.texts):
                results.append(SearchResult(
                    text=self.texts[idx],
                    metadata=self.metadata[idx],
                    score=float(score)
                ))
        
        return results
    
    def save_index(self, path: str):
        """Save the complete vector store to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.faiss")
        
        # Save texts and metadata
        store_data = {
            'texts': self.texts,
            'metadata': self.metadata,
            'dimension': self.dimension,
            'index_type': self.index_type
        }
        
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump(store_data, f)
    
    def load_index(self, path: str):
        """Load vector store from disk."""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.faiss")
        
        # Load texts and metadata
        with open(f"{path}.pkl", 'rb') as f:
            store_data = pickle.load(f)
            
        self.texts = store_data['texts']
        self.metadata = store_data['metadata']
        self.dimension = store_data['dimension']
        self.index_type = store_data['index_type']
        
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'is_trained': getattr(self.index, 'is_trained', True)
        }
