import faiss
import numpy as np


def build_index(embeddings):

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def search_index(index, query_embedding, k=5):

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    return distances, indices

def save_index(index, path):

    faiss.write_index(index, path)


def load_index(path):

    return faiss.read_index(path)