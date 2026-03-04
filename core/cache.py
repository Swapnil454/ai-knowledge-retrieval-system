from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
import hashlib

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Semantic cache for storing and retrieving answers based on query similarity.
    Uses cosine similarity to find cached answers for similar queries.
    Automatically invalidates when document set changes.
    """

    def __init__(self, threshold=0.85, max_size=1000):
        """
        Initialize the semantic cache.
        
        Args:
            threshold: Minimum similarity score to consider a cache hit
            max_size: Maximum number of items to store in cache
        """
        self.cache = []
        self.threshold = threshold
        self.max_size = max_size
        self.document_hash = None  # Track document state

    def set_document_hash(self, documents_info: str):
        """
        Set the document hash to track when documents change.
        Clears cache if documents have changed.
        
        Args:
            documents_info: String identifying current document set
        """
        new_hash = hashlib.md5(documents_info.encode()).hexdigest()
        
        if self.document_hash is not None and self.document_hash != new_hash:
            logger.info("Documents changed - clearing cache")
            self.clear()
            
        self.document_hash = new_hash

    def search(self, query_embedding):
        """
        Search for a cached answer matching the query embedding.
        
        Args:
            query_embedding: The embedding vector of the query
            
        Returns:
            Cached answer if similarity exceeds threshold, None otherwise
        """
        if not self.cache:
            return None
            
        try:
            query_embedding = np.array(query_embedding).reshape(1, -1)
            
            for item in self.cache:
                cached_embedding = np.array(item["embedding"]).reshape(1, -1)
                
                sim = cosine_similarity(
                    query_embedding,
                    cached_embedding
                )[0][0]

                if sim > self.threshold:
                    logger.debug(f"Cache hit with similarity {sim:.4f}")
                    return item["answer"]

            return None
        except Exception as e:
            logger.error(f"Cache search error: {e}")
            return None

    def add(self, query_embedding, answer):
        """
        Add a new query-answer pair to the cache.
        
        Args:
            query_embedding: The embedding vector of the query
            answer: The answer string to cache
        """
        try:
            # Evict oldest if at max size
            if len(self.cache) >= self.max_size:
                self.cache.pop(0)
                
            self.cache.append({
                "embedding": np.array(query_embedding).tolist(),
                "answer": answer
            })
            logger.debug(f"Added to cache. Size: {len(self.cache)}")
        except Exception as e:
            logger.error(f"Cache add error: {e}")

    def clear(self):
        """Clear all cached items."""
        self.cache = []
        logger.info("Cache cleared")

    def size(self):
        """Return the number of cached items."""
        return len(self.cache)