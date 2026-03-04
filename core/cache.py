from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache:

    def __init__(self, threshold=0.85):

        self.cache = []
        self.threshold = threshold

    def search(self, query_embedding):

        for item in self.cache:

            sim = cosine_similarity(
                [query_embedding],
                [item["embedding"]]
            )[0][0]

            if sim > self.threshold:

                return item["answer"]

        return None

    def add(self, query_embedding, answer):

        self.cache.append({
            "embedding": query_embedding,
            "answer": answer
        })