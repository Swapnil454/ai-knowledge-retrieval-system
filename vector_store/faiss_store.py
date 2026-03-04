import faiss
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


def build_index(embeddings):
    """
    Build a FAISS index from embeddings.
    
    Args:
        embeddings: numpy array or list of embeddings
        
    Returns:
        FAISS index object
    """
    embeddings = np.array(embeddings).astype("float32")
    
    if len(embeddings) == 0:
        raise ValueError("Cannot build index with empty embeddings")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    logger.info(f"Built index with {len(embeddings)} vectors of dimension {dimension}")
    return index


def merge_indices(embeddings_list):
    """
    Create a single index from multiple embedding sets.
    Used when combining multiple document uploads.
    
    Args:
        embeddings_list: List of all embeddings to index
        
    Returns:
        Combined FAISS index
    """
    return build_index(embeddings_list)


def search_index(index, query_embedding, k=5):
    """
    Search the FAISS index for similar vectors.
    
    Args:
        index: FAISS index
        query_embedding: Query vector(s)
        k: Number of results to return
        
    Returns:
        Tuple of (distances, indices)
    """
    query_embedding = np.array(query_embedding).astype("float32")
    
    # Ensure 2D array
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Ensure k doesn't exceed index size
    actual_k = min(k, index.ntotal)
    
    if actual_k == 0:
        return np.array([[]]), np.array([[]])

    distances, indices = index.search(query_embedding, actual_k)
    return distances, indices


def save_index(index, path):
    """
    Save FAISS index to disk.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    faiss.write_index(index, path)
    logger.info(f"Index saved to {path}")


def load_index(path):
    """
    Load FAISS index from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index file not found: {path}")
    return faiss.read_index(path)