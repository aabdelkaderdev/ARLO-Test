"""Embedding service - provides fallback embeddings using sentence-transformers."""
from typing import List, Optional
import numpy as np


class EmbeddingService:
    """Service for generating embeddings using sentence-transformers.
    
    Uses the all-MiniLM-L6-v2 model which is small (~80MB) and fast.
    Model is loaded lazily on first use.
    """
    
    MODEL_NAME = "all-MiniLM-L6-v2"
    
    def __init__(self):
        self._model = None
    
    def _load_model(self):
        """Lazy load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"Loading embedding model: {self.MODEL_NAME}")
                self._model = SentenceTransformer(self.MODEL_NAME)
                print("Embedding model loaded successfully")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for fallback embeddings. "
                    "Install it with: pip install sentence-transformers"
                )
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        self._load_model()
        
        # Generate embeddings
        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        
        # Convert numpy arrays to lists
        return embeddings.tolist()
    
    async def get_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """Async wrapper for get_embeddings (runs in thread pool)."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_embeddings, texts)
