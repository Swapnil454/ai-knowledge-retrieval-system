from loaders.pdf_loader import load_pdf
from loaders.text_chunker import chunk_text
from models.embedding_model import create_embeddings, model
from models.vector_store import build_index, search_index


def process_document(file):

    text = load_pdf(file)

    if not text.strip():
        raise ValueError("No text could be extracted from this PDF.")

    chunks = chunk_text(text)

    embeddings = create_embeddings(chunks)

    index = build_index(embeddings)

    return chunks, index


def ask_question(question, chunks, index):

    query_embedding = model.encode([question])

    indices = search_index(index, query_embedding)

    results = []

    for i in indices[0]:
        results.append(chunks[i])

    return results