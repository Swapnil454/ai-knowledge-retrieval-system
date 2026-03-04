from sentence_transformers import CrossEncoder
import logging

from core.config import RERANK_MODEL

logger = logging.getLogger(__name__)

try:
    reranker = CrossEncoder(RERANK_MODEL)
    MODEL_AVAILABLE = True
except Exception as e:
    logger.error(f"Could not load reranker model: {e}")
    reranker = None
    MODEL_AVAILABLE = False


def rerank(question: str, chunks: list) -> list:
    """
    Rerank chunks based on relevance to the question.
    
    Args:
        question: The query string
        chunks: List of text chunks to rerank
        
    Returns:
        List of chunks sorted by relevance (most relevant first)
    """
    if not chunks:
        return []
    
    if not MODEL_AVAILABLE or reranker is None:
        # Fallback: return chunks in original order
        logger.warning("Reranker not available, returning original order")
        return chunks
    
    try:
        pairs = [[question, chunk] for chunk in chunks]

        scores = reranker.predict(pairs)

        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True
        )

        ranked_chunks = [chunk for score, chunk in ranked]
        
        logger.debug(f"Reranked {len(chunks)} chunks")
        return ranked_chunks
        
    except Exception as e:
        logger.error(f"Reranking error: {e}")
        return chunks  # Return original order on error