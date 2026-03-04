MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

QA_MODEL = "deepset/roberta-base-squad2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K_RETRIEVAL = 5
TOP_K_CONTEXT = 5

MULTI_QUERY_COUNT = 3

CACHE_SIMILARITY_THRESHOLD = 0.85

VECTOR_INDEX_PATH = "data/vector.index"