from rank_bm25 import BM25Okapi
import logging
import re

logger = logging.getLogger(__name__)


class BM25Search:
    """
    BM25-based keyword search for document retrieval.
    Complements vector search with lexical matching.
    """

    def __init__(self, chunks: list):
        """
        Initialize BM25 index with document chunks.
        
        Args:
            chunks: List of document chunks (dicts with 'text' key)
        """
        self.chunks = chunks
        self.texts = [chunk["text"] for chunk in chunks]

        # Tokenize with basic preprocessing
        tokenized = [self._tokenize(text) for text in self.texts]

        self.bm25 = BM25Okapi(tokenized)
        logger.debug(f"BM25 index built with {len(chunks)} documents")

    def _tokenize(self, text: str) -> list:
        """
        Tokenize text for BM25 indexing.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of lowercase tokens
        """
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        # Filter very short tokens
        return [t for t in tokens if len(t) > 1]

    def search(self, query: str, top_k: int = 5) -> list:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of indices of matching documents
        """
        try:
            tokenized_query = self._tokenize(query)
            
            if not tokenized_query:
                logger.warning("Empty tokenized query")
                return list(range(min(top_k, len(self.chunks))))

            scores = self.bm25.get_scores(tokenized_query)

            ranked = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )

            return ranked[:top_k]
            
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return list(range(min(top_k, len(self.chunks))))