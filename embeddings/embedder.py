from sentence_transformers import SentenceTransformer
from core.config import MODEL_NAME

model = SentenceTransformer(MODEL_NAME)


def embed_text(texts):

    embeddings = model.encode(
        texts,
        show_progress_bar=False
    )

    return embeddings