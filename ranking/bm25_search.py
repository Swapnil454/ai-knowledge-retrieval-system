from rank_bm25 import BM25Okapi


class BM25Search:

    def __init__(self, chunks):

        self.texts = [chunk["text"] for chunk in chunks]

        tokenized = [text.split() for text in self.texts]

        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, top_k=5):

        tokenized_query = query.split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        return ranked[:top_k]