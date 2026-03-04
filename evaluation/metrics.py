from sklearn.metrics.pairwise import cosine_similarity


def relevance_score(query_embedding, doc_embedding):

    score = cosine_similarity(
        [query_embedding],
        [doc_embedding]
    )

    return score[0][0]