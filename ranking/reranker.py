from sentence_transformers import CrossEncoder
from core.config import RERANK_MODEL

reranker = CrossEncoder(RERANK_MODEL)


def rerank(question, chunks):

    pairs = [[question, chunk] for chunk in chunks]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, chunks),
        reverse=True
    )

    ranked_chunks = [chunk for score, chunk in ranked]

    return ranked_chunks