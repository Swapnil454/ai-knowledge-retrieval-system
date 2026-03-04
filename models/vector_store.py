import faiss
import numpy as np

def build_index(embeddings):

    embeddings = np.array(embeddings)

    if len(embeddings) == 0:
        raise ValueError("No embeddings generated from document")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def search_index(index, query_embedding, k=5):

    distances, indices = index.search(query_embedding, k)

    return distances, indices