"""
Configuration settings for the AI Knowledge Retrieval System.
Modify these values to customize system behavior.
"""
import os

# =============================================================================
# Model Configuration
# =============================================================================

# Embedding model for semantic search
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Cross-encoder model for reranking results
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Question-answering model for answer generation (extractive)
QA_MODEL = os.getenv("QA_MODEL", "deepset/roberta-base-squad2")

# Generative model for natural responses (ChatGPT-like)
GEN_MODEL = os.getenv("GEN_MODEL", "google/flan-t5-base")

# =============================================================================
# Chunking Configuration
# =============================================================================

# Maximum words per chunk
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))

# Overlap between consecutive chunks (for context continuity)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# =============================================================================
# Retrieval Configuration
# =============================================================================

# Number of chunks to retrieve before reranking
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "10"))

# Number of chunks to use for answer generation
TOP_K_CONTEXT = int(os.getenv("TOP_K_CONTEXT", "7"))

# Number of query expansions to generate
MULTI_QUERY_COUNT = int(os.getenv("MULTI_QUERY_COUNT", "3"))

# =============================================================================
# Cache Configuration
# =============================================================================

# Minimum similarity score for cache hits (0.0 to 1.0)
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.85"))

# Maximum number of cached query-answer pairs
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "1000"))

# =============================================================================
# Storage Configuration
# =============================================================================

# Path to save/load FAISS index
VECTOR_INDEX_PATH = os.getenv("VECTOR_INDEX_PATH", "data/vector.index")

# Path for query logs
QUERY_LOG_PATH = os.getenv("QUERY_LOG_PATH", "data/query_logs.json")

# =============================================================================
# Logging Configuration
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")