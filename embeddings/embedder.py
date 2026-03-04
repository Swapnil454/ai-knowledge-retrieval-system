from sentence_transformers import SentenceTransformer
import logging
import numpy as np

from core.config import MODEL_NAME

logger = logging.getLogger(__name__)

try:
    model = SentenceTransformer(MODEL_NAME)
    MODEL_AVAILABLE = True
    logger.info(f"Loaded embedding model: {MODEL_NAME}")
except Exception as e:
    logger.error(f"Could not load embedding model: {e}")
    model = None
    MODEL_AVAILABLE = False


def embed_text(texts: list) -> np.ndarray:
    """
    Generate embeddings for text chunks.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        Numpy array of embedding vectors
    """
    if not texts:
        raise ValueError("No texts provided for embedding")
    
    if not MODEL_AVAILABLE or model is None:
        raise RuntimeError("Embedding model not available")
    
    try:
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        logger.debug(f"Generated {len(embeddings)} embeddings")
        return embeddings
        
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise