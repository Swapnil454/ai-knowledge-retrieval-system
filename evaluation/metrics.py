from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

logger = logging.getLogger(__name__)


def relevance_score(query_embedding, doc_embedding) -> float:
    """
    Calculate cosine similarity between query and document embeddings.
    
    Args:
        query_embedding: Query embedding vector
        doc_embedding: Document embedding vector
        
    Returns:
        Similarity score between 0 and 1
    """
    try:
        score = cosine_similarity(
            [query_embedding],
            [doc_embedding]
        )
        return float(score[0][0])
    except Exception as e:
        logger.error(f"Error calculating relevance: {e}")
        return 0.0


def mrr_score(relevant_indices: list, retrieved_indices: list) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        relevant_indices: List of relevant document indices
        retrieved_indices: List of retrieved document indices (ranked)
        
    Returns:
        MRR score
    """
    for i, idx in enumerate(retrieved_indices):
        if idx in relevant_indices:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(relevant_indices: list, retrieved_indices: list, k: int = 5) -> float:
    """
    Calculate Precision@K.
    
    Args:
        relevant_indices: List of relevant document indices
        retrieved_indices: List of retrieved document indices
        k: Number of top results to consider
        
    Returns:
        Precision score
    """
    top_k = retrieved_indices[:k]
    relevant_in_top_k = sum(1 for idx in top_k if idx in relevant_indices)
    return relevant_in_top_k / k


def recall_at_k(relevant_indices: list, retrieved_indices: list, k: int = 5) -> float:
    """
    Calculate Recall@K.
    
    Args:
        relevant_indices: List of relevant document indices
        retrieved_indices: List of retrieved document indices
        k: Number of top results to consider
        
    Returns:
        Recall score
    """
    if not relevant_indices:
        return 0.0
    top_k = retrieved_indices[:k]
    relevant_in_top_k = sum(1 for idx in top_k if idx in relevant_indices)
    return relevant_in_top_k / len(relevant_indices)


def f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall.
    
    Args:
        precision: Precision score
        recall: Recall score
        
    Returns:
        F1 score
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)